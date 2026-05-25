"""
Debug script: test compute_depth_map_cuda with serial execution (1 thread).
If serial matches numpy, the issue is race conditions from parallel writes.
If serial still differs, the issue is data layout or kernel logic.
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
    _lib,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
    parser.add_argument("--ckpt", default="pretrained/bevfusion-det.pth")
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

    # GPU depth map - SERIAL (1 block, 1 thread)
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

    depth_map_serial = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)
    _ilr = make_cuda_buffer_from_array(inv_lidar_aug_rot)
    _ilt = make_cuda_buffer_from_array(inv_lidar_aug_trans)
    _l2r = make_cuda_buffer_from_array(lidar2image_rot)
    _l2t = make_cuda_buffer_from_array(lidar2image_trans)
    _iar = make_cuda_buffer_from_array(img_aug_rot)
    _iat = make_cuda_buffer_from_array(img_aug_trans)

    ret = _lib.compute_depth_map_cuda(
        ctypes.c_void_p(points_gpu.ptr),
        ctypes.c_void_p(prefix_sum_gpu.ptr),
        num_points,
        ctypes.c_void_p(_ilr.ptr),
        ctypes.c_void_p(_ilt.ptr),
        ctypes.c_void_p(_l2r.ptr),
        ctypes.c_void_p(_l2t.ptr),
        ctypes.c_void_p(_iar.ptr),
        ctypes.c_void_p(_iat.ptr),
        B, N, iH, iW,
        ctypes.c_void_p(depth_map_serial.ptr),
    )
    print(f"Serial CUDA return: {ret}")
    depth_map_serial_arr = depth_map_serial.download((B, N, 1, iH, iW), np.float32)

    # GPU depth map - PARALLEL (normal)
    depth_map_parallel = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)
    ret2 = _lib.compute_depth_map_cuda(
        ctypes.c_void_p(points_gpu.ptr),
        ctypes.c_void_p(prefix_sum_gpu.ptr),
        num_points,
        ctypes.c_void_p(_ilr.ptr),
        ctypes.c_void_p(_ilt.ptr),
        ctypes.c_void_p(_l2r.ptr),
        ctypes.c_void_p(_l2t.ptr),
        ctypes.c_void_p(_iar.ptr),
        ctypes.c_void_p(_iat.ptr),
        B, N, iH, iW,
        ctypes.c_void_p(depth_map_parallel.ptr),
    )
    print(f"Parallel CUDA return: {ret2}")
    depth_map_parallel_arr = depth_map_parallel.download((B, N, 1, iH, iW), np.float32)

    print("Per-camera nonzero counts:")
    for c in range(N):
        nz_np = np.count_nonzero(depth_map_np[0, c, 0])
        nz_serial = np.count_nonzero(depth_map_serial_arr[0, c, 0])
        nz_parallel = np.count_nonzero(depth_map_parallel_arr[0, c, 0])
        diff_serial = np.count_nonzero((depth_map_np[0, c, 0] != 0) ^ (depth_map_serial_arr[0, c, 0] != 0))
        diff_parallel = np.count_nonzero((depth_map_np[0, c, 0] != 0) ^ (depth_map_parallel_arr[0, c, 0] != 0))
        print(f"  camera {c}: np={nz_np}, serial={nz_serial}, parallel={nz_parallel}, diff_serial={diff_serial}, diff_parallel={diff_parallel}")


if __name__ == "__main__":
    import ctypes
    main()
