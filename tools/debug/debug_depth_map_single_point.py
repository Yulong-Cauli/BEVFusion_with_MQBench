"""
Debug script: compare compute_depth_map for a SINGLE point.
This isolates whether the projection math in CUDA matches numpy.
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
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import (
    _lib,
)


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

    points_np = [p.cpu().numpy() for p in points]
    lidar2image_np = lidar2image.cpu().numpy()
    img_aug_matrix_np = img_aug_matrix.cpu().numpy()
    lidar_aug_matrix_np = lidar_aug_matrix.cpu().numpy()

    B, N = 1, 6
    iH, iW = model.encoders["camera"]["vtransform"].image_size

    # Numpy reference for the whole point cloud
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

    # Find first 5 points that project into the image for camera 0
    b, c = 0, 0
    test_points = []
    for idx, p in enumerate(points_np[b]):
        pt = p[:3]
        # numpy math
        cur_lidar_aug = lidar_aug_matrix_np[b]
        cur_l2i = lidar2image_np[b, c]
        cur_img_aug = img_aug_matrix_np[b, c]
        tmp = pt - cur_lidar_aug[:3, 3]
        tmp = np.linalg.inv(cur_lidar_aug[:3, :3]).dot(tmp)
        tmp = cur_l2i[:3, :3].dot(tmp) + cur_l2i[:3, 3]
        dist = tmp[2]
        tmp[2] = np.clip(tmp[2], 1e-5, 1e5)
        tmp[:2] /= tmp[2]
        tmp = cur_img_aug[:3, :3].dot(tmp) + cur_img_aug[:3, 3]
        px, py = tmp[1], tmp[0]
        if 0 <= py < iH and 0 <= px < iW:
            iy, ix = int(py), int(px)
            test_points.append((idx, pt.copy(), float(dist), iy, ix))
        if len(test_points) >= 5:
            break

    print(f"Testing {len(test_points)} points that project into camera {c}")

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

    for idx, pt, expected_dist, expected_iy, expected_ix in test_points:
        # Run CUDA on ONLY this point
        points_concat = pt.astype(np.float32).reshape(1, -1)
        prefix_sum = np.array([0, 1], dtype=np.int32)

        points_gpu = make_cuda_buffer_from_array(points_concat)
        prefix_sum_gpu = make_cuda_buffer_from_array(prefix_sum)
        depth_map_gpu = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)

        ret = _lib.compute_depth_map_cuda(
            ctypes.c_void_p(points_gpu.ptr),
            ctypes.c_void_p(prefix_sum_gpu.ptr),
            1,
            ctypes.c_void_p(_ilr.ptr),
            ctypes.c_void_p(_ilt.ptr),
            ctypes.c_void_p(_l2r.ptr),
            ctypes.c_void_p(_l2t.ptr),
            ctypes.c_void_p(_iar.ptr),
            ctypes.c_void_p(_iat.ptr),
            B, N, iH, iW,
            ctypes.c_void_p(depth_map_gpu.ptr),
        )
        if ret != 0:
            print(f"  point {idx}: CUDA error {ret}")
            continue

        arr = depth_map_gpu.download((B, N, 1, iH, iW), np.float32)
        nz = np.nonzero(arr[0, :, 0])
        nz_cameras = nz[0].tolist()
        if len(nz_cameras) == 0:
            print(f"  point {idx}: expected=({expected_iy},{expected_ix}) dist={expected_dist:.4f}, CUDA wrote NOTHING")
        else:
            written = []
            for cc in nz_cameras:
                cy, cx = np.where(arr[0, cc, 0] != 0)
                written.append((cc, int(cy[0]), int(cx[0]), float(arr[0, cc, 0, cy[0], cx[0]])))
            print(f"  point {idx}: expected cam{c}=({expected_iy},{expected_ix}) dist={expected_dist:.4f}, CUDA wrote {written}")


if __name__ == "__main__":
    main()
