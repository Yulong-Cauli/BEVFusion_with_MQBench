"""
Debug script: compare compute_depth_map_cuda vs numpy using only first N points.
If they match for small N, the issue is likely race conditions from concurrent writes.
"""
import os
import sys
import numpy as np
import torch
import ctypes

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
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import _lib


def main():
    configs.load("configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml", recursive=True)
    cfg = Config(recursive_eval(configs), filename="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
    data = next(iter(data_loader))

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load("pretrained/bevfusion-det.pth", map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().cuda()

    points = [p.cuda() for p in data["points"].data[0]]
    lidar2image = data["lidar2image"].data[0].cuda()
    img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
    lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

    points_np_full = [p.cpu().numpy() for p in points]
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

    inv_lidar_aug_rot = np.linalg.inv(lidar_aug_matrix_np[..., :3, :3]).astype(np.float32)
    inv_lidar_aug_trans = lidar_aug_matrix_np[..., :3, 3].astype(np.float32)
    lidar2image_rot = lidar2image_np[..., :3, :3].astype(np.float32)
    lidar2image_trans = lidar2image_np[..., :3, 3].astype(np.float32)
    img_aug_rot = img_aug_matrix_np[..., :3, :3].astype(np.float32)
    img_aug_trans = img_aug_matrix_np[..., :3, 3].astype(np.float32)

    _ilr = make_cuda_buffer_from_array(inv_lidar_aug_rot)
    _ilt = make_cuda_buffer_from_array(inv_lidar_aug_trans)
    _l2r = make_cuda_buffer_from_array(lidar2image_rot)
    _l2t = make_cuda_buffer_from_array(lidar2image_trans)
    _iar = make_cuda_buffer_from_array(img_aug_rot)
    _iat = make_cuda_buffer_from_array(img_aug_trans)

    for limit in [1, 10, 100, 1000, 5000, points_np_full[0].shape[0]]:
        points_np = [points_np_full[0][:limit]]
        depth_map_np = vtransform_geom.compute_depth_map(points_np, img_aug_matrix_np, lidar_aug_matrix_np, lidar2image_np, B, N)

        points_concat = points_np[0][:, :3].astype(np.float32)
        prefix_sum = np.array([0, points_concat.shape[0]], dtype=np.int32)
        points_gpu = make_cuda_buffer_from_array(points_concat)
        prefix_sum_gpu = make_cuda_buffer_from_array(prefix_sum)
        depth_map_gpu = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)

        _lib.compute_depth_map_cuda(
            ctypes.c_void_p(points_gpu.ptr),
            ctypes.c_void_p(prefix_sum_gpu.ptr),
            points_concat.shape[0],
            ctypes.c_void_p(_ilr.ptr),
            ctypes.c_void_p(_ilt.ptr),
            ctypes.c_void_p(_l2r.ptr),
            ctypes.c_void_p(_l2t.ptr),
            ctypes.c_void_p(_iar.ptr),
            ctypes.c_void_p(_iat.ptr),
            B, N, iH, iW,
            ctypes.c_void_p(depth_map_gpu.ptr),
        )
        depth_map_gpu_arr = depth_map_gpu.download((B, N, 1, iH, iW), np.float32)

        diff_pixels = np.count_nonzero((depth_map_np != 0) ^ (depth_map_gpu_arr != 0))
        both_nz = np.count_nonzero((depth_map_np != 0) & (depth_map_gpu_arr != 0))
        print(f"limit={limit:5d}: np_nz={np.count_nonzero(depth_map_np):5d}, gpu_nz={np.count_nonzero(depth_map_gpu_arr):5d}, "
              f"both_nz={both_nz:5d}, diff_pixels={diff_pixels:5d}")


if __name__ == "__main__":
    main()
