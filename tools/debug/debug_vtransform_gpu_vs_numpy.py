"""
Debug script: compare GPU vtransform step-by-step against numpy reference.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval

from tools.trt_infer import HybridBEVFusion, TRTRunner
from tools.trt_infer_zero_torch import (
    ZeroTorchTRTRunner,
    ZeroTorchVoxelization,
    NumpyVTransformGeometry,
    NumpyTransFusionBBoxCoder,
    ZeroTorchBEVFusion,
    CudaBuffer,
    make_cuda_buffer_from_array,
)
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import (
    compute_depth_map_cuda,
    vtransform_gpu_workspace_size,
    vtransform_post_depthnet_cuda,
)


def compare(name, a, b, rtol=1e-5, atol=1e-5):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    close = np.allclose(a, b, rtol=rtol, atol=atol)
    status = "PASS" if close else "FAIL"
    print(f"  {name}: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} [{status}]")
    return close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--swin-engine", default="swin_int8_sm86.engine")
    parser.add_argument("--neck-engine", default="camera_neck_int8_sm86.engine")
    parser.add_argument("--depthnet-engine", default="vtransform_depthnet_int8_sm86.engine")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
    data = next(iter(data_loader))

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().cuda()

    img = data["img"].data[0].cuda()
    points = [p.cuda() for p in data["points"].data[0]]
    metas = data["metas"].data[0]
    camera2ego = data["camera2ego"].data[0].cuda()
    lidar2ego = data["lidar2ego"].data[0].cuda()
    lidar2camera = data["lidar2camera"].data[0].cuda()
    lidar2image = data["lidar2image"].data[0].cuda()
    camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
    camera2lidar = data["camera2lidar"].data[0].cuda()
    img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
    lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

    B, N = img.shape[0], img.shape[1]
    iH, iW = model.encoders["camera"]["vtransform"].image_size
    fH, fW = model.encoders["camera"]["vtransform"].feature_size
    D = model.encoders["camera"]["vtransform"].D
    C_bev = 80  # will be confirmed from depthnet output

    img_np = img.cpu().numpy()
    points_np = [p.cpu().numpy() for p in points]
    camera2ego_np = camera2ego.cpu().numpy()
    lidar2ego_np = lidar2ego.cpu().numpy()
    lidar2camera_np = lidar2camera.cpu().numpy()
    lidar2image_np = lidar2image.cpu().numpy()
    camera_intrinsics_np = camera_intrinsics.cpu().numpy()
    camera2lidar_np = camera2lidar.cpu().numpy()
    img_aug_matrix_np = img_aug_matrix.cpu().numpy()
    lidar_aug_matrix_np = lidar_aug_matrix.cpu().numpy()

    vtransform = model.encoders["camera"]["vtransform"]
    vtransform_geom = NumpyVTransformGeometry(
        image_size=tuple(vtransform.image_size),
        feature_size=tuple(vtransform.feature_size),
        xbound=vtransform.xbound,
        ybound=vtransform.ybound,
        zbound=vtransform.zbound,
        dbound=vtransform.dbound,
    )

    # ================================================================
    # Run original model up to camera neck to get depthnet inputs
    # ================================================================
    swin_trt = TRTRunner(args.swin_engine, logger)
    neck_trt = TRTRunner(args.neck_engine, logger)
    depthnet_trt = TRTRunner(args.depthnet_engine, logger)

    with torch.no_grad():
        img_flat = img.view(B * N, 3, 256, 704).float()
        swin_outputs = []
        for i in range(B * N):
            outs = swin_trt(img_flat[i : i + 1])
            swin_outputs.append([o.float() for o in outs])
        num_scales = len(swin_outputs[0])
        multi_scale_feats = []
        for s in range(num_scales):
            feat = torch.cat([swin_outputs[i][s] for i in range(B * N)], dim=0)
            multi_scale_feats.append(feat)

        neck_out = neck_trt(multi_scale_feats[0].float(),
                            multi_scale_feats[1].float(),
                            multi_scale_feats[2].float())
        x_cam = neck_out[0].float()

    x_cam_5d = x_cam.view(B, N, x_cam.shape[1], x_cam.shape[2], x_cam.shape[3]).cpu().numpy()

    # ================================================================
    # Numpy reference path for vtransform
    # ================================================================
    print("=" * 60)
    print("Running numpy reference vtransform")
    print("=" * 60)

    depth_map_np = vtransform_geom.compute_depth_map(points_np, img_aug_matrix_np, lidar_aug_matrix_np, lidar2image_np, B, N)

    # Use ZeroTorchTRTRunner for depthnet to handle CudaBuffer inputs
    depthnet_zt = ZeroTorchTRTRunner(args.depthnet_engine, logger)
    depth_inputs = {
        depthnet_zt.input_names[0]: make_cuda_buffer_from_array(x_cam_5d),
        depthnet_zt.input_names[1]: make_cuda_buffer_from_array(depth_map_np),
    }
    depthnet_out = depthnet_zt(depth_inputs)
    cam_feats_flat_np = depthnet_out[0].astype(np.float32)
    if cam_feats_flat_np.shape[0] == B * N * D * fH * fW:
        C_bev = cam_feats_flat_np.shape[1]
    else:
        C_bev = cam_feats_flat_np.shape[-1]
    cam_feats_6d_np = cam_feats_flat_np.reshape(B, N, D, fH, fW, C_bev)

    camera2lidar_rots = camera2lidar_np[..., :3, :3]
    camera2lidar_trans = camera2lidar_np[..., :3, 3]
    intrins = camera_intrinsics_np[..., :3, :3]
    post_rots = img_aug_matrix_np[..., :3, :3]
    post_trans = img_aug_matrix_np[..., :3, 3]
    extra_rots = lidar_aug_matrix_np[..., :3, :3]
    extra_trans = lidar_aug_matrix_np[..., :3, 3]

    geom_np = vtransform_geom.get_geometry(
        camera2lidar_rots, camera2lidar_trans,
        intrins, post_rots, post_trans,
        extra_rots=extra_rots, extra_trans=extra_trans)

    indices = vtransform_geom.precompute_bev_indices(geom_np, B)

    Nprime = B * N * D * fH * fW
    x_flat = cam_feats_6d_np.reshape(Nprime, C_bev)
    x_flat = x_flat[indices["kept"]]
    x_flat = x_flat[indices["sort_indices"]]

    from tools.trt_infer_zero_torch import np_bev_pool_v2
    out_np = np_bev_pool_v2(
        x_flat, indices["geom_feats"],
        indices["interval_starts"], indices["interval_lengths"],
        indices["B"], indices["D"], indices["H"], indices["W"])

    camera_bev_np = np.concatenate(np.split(out_np, out_np.shape[2], axis=2), axis=1).squeeze(2)

    print(f"C_bev={C_bev}, Nprime={Nprime}")
    print(f"numpy camera_bev shape: {camera_bev_np.shape}, sum: {camera_bev_np.sum():.4f}")

    # ================================================================
    # GPU path for vtransform
    # ================================================================
    print("=" * 60)
    print("Running GPU vtransform")
    print("=" * 60)

    # 1. compute_depth_map on GPU
    points_list = [p[:, :3] for p in points_np]
    num_points = sum(p.shape[0] for p in points_list)
    points_concat = np.concatenate(points_list, axis=0).astype(np.float32)
    prefix_sum = np.zeros(B + 1, dtype=np.int32)
    for b in range(B):
        prefix_sum[b + 1] = prefix_sum[b] + points_list[b].shape[0]

    points_gpu = make_cuda_buffer_from_array(points_concat)
    prefix_sum_gpu = make_cuda_buffer_from_array(prefix_sum)

    inv_lidar_aug_rot = np.linalg.inv(lidar_aug_matrix_np[..., :3, :3]).astype(np.float32)
    inv_lidar_aug_trans = lidar_aug_matrix_np[..., :3, 3].astype(np.float32)
    lidar2image_rot = lidar2image_np[..., :3, :3].astype(np.float32)
    lidar2image_trans = lidar2image_np[..., :3, 3].astype(np.float32)
    img_aug_rot = img_aug_matrix_np[..., :3, :3].astype(np.float32)
    img_aug_trans = img_aug_matrix_np[..., :3, 3].astype(np.float32)

    depth_map_gpu = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)
    _ilr = make_cuda_buffer_from_array(inv_lidar_aug_rot)
    _ilt = make_cuda_buffer_from_array(inv_lidar_aug_trans)
    _l2r = make_cuda_buffer_from_array(lidar2image_rot)
    _l2t = make_cuda_buffer_from_array(lidar2image_trans)
    _iar = make_cuda_buffer_from_array(img_aug_rot)
    _iat = make_cuda_buffer_from_array(img_aug_trans)
    compute_depth_map_cuda(
        points_gpu.ptr, prefix_sum_gpu.ptr, num_points,
        _ilr.ptr, _ilt.ptr, _l2r.ptr, _l2t.ptr, _iar.ptr, _iat.ptr,
        B, N, iH, iW, depth_map_gpu.ptr,
    )
    depth_map_gpu_arr = depth_map_gpu.download((B, N, 1, iH, iW), np.float32)
    compare("depth_map", depth_map_np, depth_map_gpu_arr, rtol=1e-4, atol=1e-4)

    # Depth map difference analysis
    nz_np = depth_map_np != 0
    nz_gpu = depth_map_gpu_arr != 0
    both_nz = nz_np & nz_gpu
    diff_at_both = np.abs(depth_map_np.astype(np.float64) - depth_map_gpu_arr.astype(np.float64)) > 1e-4
    num_diff_at_both = np.count_nonzero(both_nz & diff_at_both)
    num_both_nz = np.count_nonzero(both_nz)
    print(f"  both nonzero: {num_both_nz}, differ at both: {num_diff_at_both}")
    print(f"  np-only nonzero: {np.count_nonzero(nz_np & ~nz_gpu)}")
    print(f"  gpu-only nonzero: {np.count_nonzero(~nz_np & nz_gpu)}")
    # Union of nonzero pixels
    union_nz = nz_np | nz_gpu
    print(f"  union nonzero pixels: {np.count_nonzero(union_nz)}")

    # Determinism test: run compute_depth_map_cuda again with same inputs
    depth_map_gpu2 = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)
    compute_depth_map_cuda(
        points_gpu.ptr, prefix_sum_gpu.ptr, num_points,
        _ilr.ptr, _ilt.ptr, _l2r.ptr, _l2t.ptr, _iar.ptr, _iat.ptr,
        B, N, iH, iW, depth_map_gpu2.ptr,
    )
    depth_map_gpu2_arr = depth_map_gpu2.download((B, N, 1, iH, iW), np.float32)
    diff_runs = np.abs(depth_map_gpu_arr.astype(np.float64) - depth_map_gpu2_arr.astype(np.float64))
    print(f"  CUDA determinism: max_diff={diff_runs.max():.6f}, nz_diff={np.count_nonzero((depth_map_gpu_arr != 0) ^ (depth_map_gpu2_arr != 0))}")

    # 2. depthnet TRT
    depth_inputs_gpu = {
        depthnet_zt.input_names[0]: make_cuda_buffer_from_array(x_cam_5d),
        depthnet_zt.input_names[1]: depth_map_gpu,
    }
    cam_feats_gpu = depthnet_zt(depth_inputs_gpu, return_gpu_buffers=True)[0]
    print(f"cam_feats_gpu nbytes={cam_feats_gpu.nbytes}")

    # 3. geometry + bev_pool on GPU
    inv_post_rots = np.linalg.inv(post_rots).astype(np.float32)
    inv_intrins = np.linalg.inv(intrins).astype(np.float32)
    combine_rots = np.matmul(camera2lidar_rots, inv_intrins).astype(np.float32)

    frustum = vtransform_geom.frustum.astype(np.float32)
    dx = vtransform_geom.dx
    bx = vtransform_geom.bx
    nx = vtransform_geom.nx

    workspace_size = vtransform_gpu_workspace_size(B, N, D, fH, fW, int(nx[0]), int(nx[1]), int(nx[2]), C_bev)
    workspace_gpu = CudaBuffer(workspace_size)
    camera_bev_gpu = CudaBuffer(B * C_bev * int(nx[2]) * int(nx[0]) * int(nx[1]) * 4, fill_value=0, dtype=np.float32)
    geom_feats_out_gpu = CudaBuffer(Nprime * 4 * 4, fill_value=0, dtype=np.int32)
    interval_starts_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)
    interval_lengths_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)

    out_K = np.zeros(1, dtype=np.int32)
    out_M = np.zeros(1, dtype=np.int32)

    _frustum_gpu = make_cuda_buffer_from_array(frustum)
    _inv_post_rots_gpu = make_cuda_buffer_from_array(inv_post_rots)
    _post_trans_gpu = make_cuda_buffer_from_array(post_trans.astype(np.float32))
    _combine_rots_gpu = make_cuda_buffer_from_array(combine_rots)
    _camera2lidar_trans_gpu = make_cuda_buffer_from_array(camera2lidar_trans.astype(np.float32))
    _extra_rots_gpu = make_cuda_buffer_from_array(extra_rots.astype(np.float32))
    _extra_trans_gpu = make_cuda_buffer_from_array(extra_trans.astype(np.float32))

    vtransform_post_depthnet_cuda(
        _frustum_gpu.ptr,
        _inv_post_rots_gpu.ptr,
        _post_trans_gpu.ptr,
        _combine_rots_gpu.ptr,
        _camera2lidar_trans_gpu.ptr,
        _extra_rots_gpu.ptr,
        _extra_trans_gpu.ptr,
        B, N, D, fH, fW,
        float(dx[0]), float(dx[1]), float(dx[2]),
        float(bx[0]), float(bx[1]), float(bx[2]),
        int(nx[0]), int(nx[1]), int(nx[2]),
        cam_feats_gpu.ptr, 0, C_bev,
        camera_bev_gpu.ptr,
        geom_feats_out_gpu.ptr, interval_starts_gpu.ptr, interval_lengths_gpu.ptr,
        out_K.ctypes.data, out_M.ctypes.data,
        workspace_gpu.ptr, workspace_size,
    )

    camera_bev_gpu_arr = camera_bev_gpu.download((B, C_bev * int(nx[2]), int(nx[0]), int(nx[1])), np.float32)
    print(f"GPU camera_bev shape: {camera_bev_gpu_arr.shape}, sum: {camera_bev_gpu_arr.sum():.4f}")
    compare("camera_bev", camera_bev_np, camera_bev_gpu_arr, rtol=1e-3, atol=1e-3)

    # ================================================================
    # Isolate geometry/index correctness: run GPU geom+bev_pool with numpy cam_feats
    # ================================================================
    print("=" * 60)
    print("GPU geometry + numpy cam_feats")
    print("=" * 60)
    camera_bev_gpu2 = CudaBuffer(B * C_bev * int(nx[2]) * int(nx[0]) * int(nx[1]) * 4, fill_value=0, dtype=np.float32)
    cam_feats_np_gpu = make_cuda_buffer_from_array(cam_feats_flat_np.reshape(Nprime, C_bev))
    vtransform_post_depthnet_cuda(
        _frustum_gpu.ptr,
        _inv_post_rots_gpu.ptr,
        _post_trans_gpu.ptr,
        _combine_rots_gpu.ptr,
        _camera2lidar_trans_gpu.ptr,
        _extra_rots_gpu.ptr,
        _extra_trans_gpu.ptr,
        B, N, D, fH, fW,
        float(dx[0]), float(dx[1]), float(dx[2]),
        float(bx[0]), float(bx[1]), float(bx[2]),
        int(nx[0]), int(nx[1]), int(nx[2]),
        cam_feats_np_gpu.ptr, 0, C_bev,
        camera_bev_gpu2.ptr,
        geom_feats_out_gpu.ptr, interval_starts_gpu.ptr, interval_lengths_gpu.ptr,
        out_K.ctypes.data, out_M.ctypes.data,
        workspace_gpu.ptr, workspace_size,
    )
    camera_bev_gpu2_arr = camera_bev_gpu2.download((B, C_bev * int(nx[2]), int(nx[0]), int(nx[1])), np.float32)
    print(f"GPU geom + numpy cam_feats shape: {camera_bev_gpu2_arr.shape}, sum: {camera_bev_gpu2_arr.sum():.4f}")
    compare("camera_bev (gpu_geom + np_cam)", camera_bev_np, camera_bev_gpu2_arr, rtol=1e-3, atol=1e-3)

    # ================================================================
    # Compare intermediate geometry/index data
    # ================================================================
    print("=" * 60)
    print("Comparing intermediate geometry/index data")
    print("=" * 60)

    # Download geom from workspace for comparison
    # Need to call a custom function or we can just trust the camera_bev comparison
    # For now, let's compare geom_feats, interval counts, and sorted cam_feats indirectly
    print(f"numpy K={x_flat.shape[0]}, GPU K={out_K[0]}")
    print(f"numpy M={indices['interval_starts'].shape[0]}, GPU M={out_M[0]}")

    # Compare geom_feats from GPU vs numpy
    geom_feats_gpu_arr = geom_feats_out_gpu.download((out_K[0], 4), np.int32)
    compare("geom_feats", indices["geom_feats"], geom_feats_gpu_arr, rtol=0, atol=0)

    interval_starts_gpu_arr = interval_starts_gpu.download((out_M[0],), np.int32)
    compare("interval_starts", indices["interval_starts"], interval_starts_gpu_arr, rtol=0, atol=0)

    interval_lengths_gpu_arr = interval_lengths_gpu.download((out_M[0],), np.int32)
    compare("interval_lengths", indices["interval_lengths"], interval_lengths_gpu_arr, rtol=0, atol=0)

    # Compare sorted cam_feats (indirectly by running numpy bev_pool with GPU indices)
    from tools.trt_infer_zero_torch import np_bev_pool_v2
    cam_feats_flat_gpu_arr = cam_feats_gpu.download((Nprime, C_bev), np.float32)
    x_flat_gpu = cam_feats_flat_gpu_arr.reshape(Nprime, C_bev)
    kept_gpu = np.ones(Nprime, dtype=bool)
    # We don't have the kept mask from GPU, but we can compare the final bev_pool output

    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
