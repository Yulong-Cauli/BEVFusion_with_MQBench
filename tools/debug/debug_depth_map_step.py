"""
Debug script: compare compute_depth_map step-by-step for points that land in image.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval

from tools.trt_infer_zero_torch import (
    NumpyVTransformGeometry,
    CudaBuffer,
    make_cuda_buffer_from_array,
)
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import (
    compute_depth_map_cuda,
)


def matvec3(M, x, y, z):
    ox = M[0, 0] * x + M[0, 1] * y + M[0, 2] * z
    oy = M[1, 0] * x + M[1, 1] * y + M[1, 2] * z
    oz = M[2, 0] * x + M[2, 1] * y + M[2, 2] * z
    return ox, oy, oz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
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

    points = [p.cuda() for p in data["points"].data[0]]
    lidar2image = data["lidar2image"].data[0].cuda()
    img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
    lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

    points_np = [p.cpu().numpy() for p in points]
    lidar2image_np = lidar2image.cpu().numpy()
    img_aug_matrix_np = img_aug_matrix.cpu().numpy()
    lidar_aug_matrix_np = lidar_aug_matrix.cpu().numpy()

    B, N = 1, 6
    iH, iW = model.encoders["camera"]["vtransform"].image_size
    vtransform = model.encoders["camera"]["vtransform"]
    vtransform_geom = NumpyVTransformGeometry(
        image_size=tuple(vtransform.image_size),
        feature_size=tuple(vtransform.feature_size),
        xbound=vtransform.xbound,
        ybound=vtransform.ybound,
        zbound=vtransform.zbound,
        dbound=vtransform.dbound,
    )

    depth_map_np = vtransform_geom.compute_depth_map(points_np, img_aug_matrix_np, lidar_aug_matrix_np, lidar2image_np, B, N)

    # GPU depth map
    points_list = points_np
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
        _ilr.ptr, _ilt.ptr,
        _l2r.ptr, _l2t.ptr,
        _iar.ptr, _iat.ptr,
        B, N, iH, iW, depth_map_gpu.ptr,
    )
    depth_map_gpu_arr = depth_map_gpu.download((B, N, 1, iH, iW), np.float32)

    diff = np.abs(depth_map_np.astype(np.float64) - depth_map_gpu_arr.astype(np.float64))
    mismatches = np.nonzero(diff > 1e-4)
    print(f"nonzero pixels (numpy): {np.count_nonzero(depth_map_np)}")
    print(f"nonzero pixels (gpu):   {np.count_nonzero(depth_map_gpu_arr)}")
    print(f"mismatched pixels (>1e-4): {len(mismatches[0])}")

    # Find some pixels where numpy is nonzero and gpu is 0
    nz_np_only = np.nonzero((depth_map_np != 0) & (depth_map_gpu_arr == 0))
    print(f"np-only pixels: {len(nz_np_only[0])}")

    # Find some pixels where gpu is nonzero and numpy is 0
    nz_gpu_only = np.nonzero((depth_map_np == 0) & (depth_map_gpu_arr != 0))
    print(f"gpu-only pixels: {len(nz_gpu_only[0])}")

    # Now trace a few np-only pixels back to which point should have written there
    b, c = 0, 0
    cur_coords_all = []
    cur_img_aug = img_aug_matrix_np[b]
    cur_lidar_aug = lidar_aug_matrix_np[b]
    cur_l2i = lidar2image_np[b]

    for idx, p in enumerate(points_np[b]):
        pt = p[:3] - cur_lidar_aug[:3, 3]
        pt = np.linalg.inv(cur_lidar_aug[:3, :3]).dot(pt)
        pt = cur_l2i[c, :3, :3].dot(pt) + cur_l2i[c, :3, 3]
        dist = pt[2]
        pt[2] = np.clip(pt[2], 1e-5, 1e5)
        pt[:2] /= pt[2]
        pt = cur_img_aug[c, :3, :3].dot(pt) + cur_img_aug[c, :3, 3]
        px, py = pt[1], pt[0]
        if 0 <= py < iH and 0 <= px < iW:
            iy, ix = int(py), int(px)
            cur_coords_all.append((idx, p[:3], dist, iy, ix))

    print(f"\nTotal points projecting into image for batch={b} camera={c}: {len(cur_coords_all)}")

    # Check if any of these landed in np-only pixels
    np_only_set = set()
    for i in range(len(nz_np_only[0])):
        np_only_set.add((nz_np_only[3][i], nz_np_only[4][i]))

    for idx, pt, dist, iy, ix in cur_coords_all[:20]:
        in_np_only = (iy, ix) in np_only_set
        np_val = depth_map_np[b, c, 0, iy, ix]
        gpu_val = depth_map_gpu_arr[b, c, 0, iy, ix]
        print(f"  point {idx}: proj=({iy},{ix}), dist={dist:.4f}, np={np_val:.4f}, gpu={gpu_val:.4f}, np_only={in_np_only}")

    # Let's manually compute for the first np-only pixel
    if len(nz_np_only[0]) > 0:
        iy0, ix0 = nz_np_only[3][0], nz_np_only[4][0]
        print(f"\n--- Tracing np-only pixel ({iy0}, {ix0}) for camera {c} ---")
        for idx, pt, dist, iy, ix in cur_coords_all:
            if iy == iy0 and ix == ix0:
                print(f"  Point {idx} at {pt} projects to ({iy},{ix}) with dist={dist:.4f}")
                # Also compute CUDA version
                p = pt
                lr = inv_lidar_aug_rot[b]
                lt = inv_lidar_aug_trans[b]
                lx = p[0] - lt[0]
                ly = p[1] - lt[1]
                lz = p[2] - lt[2]
                rx = lr[0]*lx + lr[1]*ly + lr[2]*lz
                ry = lr[3]*lx + lr[4]*ly + lr[5]*lz
                rz = lr[6]*lx + lr[7]*ly + lr[8]*lz
                l2r = lidar2image_rot[b, c]
                l2t = lidar2image_trans[b, c]
                cx = l2r[0]*rx + l2r[1]*ry + l2r[2]*rz + l2t[0]
                cy = l2r[3]*rx + l2r[4]*ry + l2r[5]*rz + l2t[1]
                cz = l2r[6]*rx + l2r[7]*ry + l2r[8]*rz + l2t[2]
                dist_cuda = cz
                cz = max(1e-5, min(cz, 1e5))
                cx /= cz; cy /= cz
                iar = img_aug_rot[b, c]
                iat = img_aug_trans[b, c]
                ix2 = iar[0]*cx + iar[1]*cy + iar[2]*cz + iat[0]
                iy2 = iar[3]*cx + iar[4]*cy + iar[5]*cz + iat[1]
                iz2 = iar[6]*cx + iar[7]*cy + iar[8]*cz + iat[2]
                py_cuda = iy2
                px_cuda = ix2
                iy_cuda = int(py_cuda)
                ix_cuda = int(px_cuda)
                print(f"    CUDA projection: ({iy_cuda},{ix_cuda}) with dist_cuda={dist_cuda:.4f}")
                if iy_cuda != iy or ix_cuda != ix:
                    print(f"    MISMATCH: numpy says ({iy},{ix}) but CUDA says ({iy_cuda},{ix_cuda})")


if __name__ == "__main__":
    main()
