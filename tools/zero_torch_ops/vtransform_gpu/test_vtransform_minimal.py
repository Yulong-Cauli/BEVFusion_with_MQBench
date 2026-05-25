"""
Minimal smoke test for vtransform GPU kernels.
Uses synthetic inputs to catch CUDA crashes and verify basic functionality.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.trt_infer_zero_torch import CudaBuffer, make_cuda_buffer_from_array
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import (
    compute_depth_map_cuda,
    vtransform_gpu_workspace_size,
    vtransform_post_depthnet_cuda,
)


def test_compute_depth_map():
    print("test_compute_depth_map...")
    B, N = 1, 6
    iH, iW = 256, 704
    num_points = 1000

    points_np = np.random.randn(num_points, 3).astype(np.float32) * 10.0
    points_np[:, 0] = np.clip(points_np[:, 0], -50, 50)
    points_np[:, 1] = np.clip(points_np[:, 1], -50, 50)
    points_np[:, 2] = np.clip(points_np[:, 2], -5, 5)

    points_prefix_sum = np.array([0, num_points], dtype=np.int32)
    points_gpu = make_cuda_buffer_from_array(points_np)
    prefix_gpu = make_cuda_buffer_from_array(points_prefix_sum)

    inv_lidar_aug_rot = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(B, axis=0)
    inv_lidar_aug_trans = np.zeros((B, 3), dtype=np.float32)
    lidar2image_rot = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3).repeat(B, axis=0).repeat(N, axis=1)
    lidar2image_trans = np.zeros((B, N, 3), dtype=np.float32)
    # Simple projection: point -> image center
    for b in range(B):
        for c in range(N):
            lidar2image_rot[b, c] = np.array([
                [800.0, 0.0, 352.0],
                [0.0, 800.0, 128.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
    img_aug_rot = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3).repeat(B, axis=0).repeat(N, axis=1)
    img_aug_trans = np.zeros((B, N, 3), dtype=np.float32)

    depth_map_gpu = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)

    compute_depth_map_cuda(
        points_gpu.ptr, prefix_gpu.ptr, num_points,
        make_cuda_buffer_from_array(inv_lidar_aug_rot).ptr,
        make_cuda_buffer_from_array(inv_lidar_aug_trans).ptr,
        make_cuda_buffer_from_array(lidar2image_rot).ptr,
        make_cuda_buffer_from_array(lidar2image_trans).ptr,
        make_cuda_buffer_from_array(img_aug_rot).ptr,
        make_cuda_buffer_from_array(img_aug_trans).ptr,
        B, N, iH, iW, depth_map_gpu.ptr,
    )
    print("  compute_depth_map_cuda OK")

    depth_map = depth_map_gpu.download((B, N, iH, iW), np.float32)
    nonzero = np.count_nonzero(depth_map)
    print(f"  nonzero depth pixels: {nonzero}")
    # Synthetic projection may not land in image; we only care that kernel doesn't crash
    print(f"  (nonzero={nonzero} is OK for synthetic data)")


def test_vtransform_post_depthnet():
    print("test_vtransform_post_depthnet...")
    B, N = 1, 6
    D, fH, fW = 8, 4, 4  # small for fast test
    C_bev = 16
    grid_x, grid_y, grid_z = 8, 8, 4

    frustum = np.random.randn(D, fH, fW, 3).astype(np.float32)
    inv_post_rots = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3).repeat(B, axis=0).repeat(N, axis=1)
    post_trans = np.zeros((B, N, 3), dtype=np.float32)
    combine_rots = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3).repeat(B, axis=0).repeat(N, axis=1)
    camera2lidar_trans = np.zeros((B, N, 3), dtype=np.float32)
    extra_rots = np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(B, axis=0)
    extra_trans = np.zeros((B, 3), dtype=np.float32)

    Nprime = B * N * D * fH * fW
    cam_feats_half = np.random.randn(Nprime, C_bev).astype(np.float16)
    cam_feats_gpu = make_cuda_buffer_from_array(cam_feats_half)

    workspace_size = vtransform_gpu_workspace_size(B, N, D, fH, fW, grid_x, grid_y, grid_z, C_bev)
    workspace_gpu = CudaBuffer(workspace_size)
    camera_bev_gpu = CudaBuffer(B * C_bev * grid_z * grid_x * grid_y * 4, fill_value=0, dtype=np.float32)
    geom_feats_out_gpu = CudaBuffer(Nprime * 4 * 4, fill_value=0, dtype=np.int32)
    interval_starts_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)
    interval_lengths_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)

    out_K = np.zeros(1, dtype=np.int32)
    out_M = np.zeros(1, dtype=np.int32)

    vtransform_post_depthnet_cuda(
        make_cuda_buffer_from_array(frustum).ptr,
        make_cuda_buffer_from_array(inv_post_rots).ptr,
        make_cuda_buffer_from_array(post_trans).ptr,
        make_cuda_buffer_from_array(combine_rots).ptr,
        make_cuda_buffer_from_array(camera2lidar_trans).ptr,
        make_cuda_buffer_from_array(extra_rots).ptr,
        make_cuda_buffer_from_array(extra_trans).ptr,
        B, N, D, fH, fW,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        grid_x, grid_y, grid_z,
        cam_feats_gpu.ptr, 1, C_bev,
        camera_bev_gpu.ptr,
        geom_feats_out_gpu.ptr, interval_starts_gpu.ptr, interval_lengths_gpu.ptr,
        out_K.ctypes.data, out_M.ctypes.data,
        workspace_gpu.ptr, workspace_size,
    )
    print("  vtransform_post_depthnet_cuda OK")
    print(f"  K={out_K[0]}, M={out_M[0]}")

    camera_bev = camera_bev_gpu.download((B, C_bev * grid_z, grid_x, grid_y), np.float32)
    print(f"  camera_bev shape: {camera_bev.shape}, sum: {camera_bev.sum():.4f}")
    assert out_K[0] > 0, "Expected some valid points"
    assert out_M[0] > 0, "Expected some intervals"


if __name__ == "__main__":
    test_compute_depth_map()
    test_vtransform_post_depthnet()
    print("All tests passed!")
