"""
ctypes wrapper for libbevfusion_vtransform_gpu.so
No torch import required at runtime.
"""
import ctypes
import os
import numpy as np

_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_vtransform_gpu.so")
_lib = ctypes.CDLL(_SO_PATH)

_lib.compute_depth_map_cuda.argtypes = [
    ctypes.c_void_p,  # points
    ctypes.c_void_p,  # points_prefix_sum
    ctypes.c_int,     # total_points
    ctypes.c_void_p,  # inv_lidar_aug_rot
    ctypes.c_void_p,  # inv_lidar_aug_trans
    ctypes.c_void_p,  # lidar2image_rot
    ctypes.c_void_p,  # lidar2image_trans
    ctypes.c_void_p,  # img_aug_rot
    ctypes.c_void_p,  # img_aug_trans
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,  # depth_map
]
_lib.compute_depth_map_cuda.restype = ctypes.c_int

_lib.vtransform_gpu_workspace_size.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.vtransform_gpu_workspace_size.restype = ctypes.c_size_t

_lib.vtransform_post_depthnet_cuda.argtypes = [
    ctypes.c_void_p,  # frustum
    ctypes.c_void_p,  # inv_post_rots
    ctypes.c_void_p,  # post_trans
    ctypes.c_void_p,  # combine_rots
    ctypes.c_void_p,  # camera2lidar_trans
    ctypes.c_void_p,  # extra_rots (nullable)
    ctypes.c_void_p,  # extra_trans (nullable)
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,  # cam_feats
    ctypes.c_int,     # cam_feats_dtype (1 = float16)
    ctypes.c_int,     # C_bev
    ctypes.c_void_p,  # camera_bev
    ctypes.c_void_p,  # geom_feats_out
    ctypes.c_void_p,  # interval_starts_out
    ctypes.c_void_p,  # interval_lengths_out
    ctypes.c_void_p,  # out_K
    ctypes.c_void_p,  # out_M
    ctypes.c_void_p,  # workspace
    ctypes.c_size_t,  # workspace_size
]
_lib.vtransform_post_depthnet_cuda.restype = ctypes.c_int


def compute_depth_map_cuda(
    points_ptr, points_prefix_sum_ptr, total_points,
    inv_lidar_aug_rot_ptr, inv_lidar_aug_trans_ptr,
    lidar2image_rot_ptr, lidar2image_trans_ptr,
    img_aug_rot_ptr, img_aug_trans_ptr,
    B, N, iH, iW, depth_map_ptr
):
    ret = _lib.compute_depth_map_cuda(
        ctypes.c_void_p(points_ptr),
        ctypes.c_void_p(points_prefix_sum_ptr),
        total_points,
        ctypes.c_void_p(inv_lidar_aug_rot_ptr),
        ctypes.c_void_p(inv_lidar_aug_trans_ptr),
        ctypes.c_void_p(lidar2image_rot_ptr),
        ctypes.c_void_p(lidar2image_trans_ptr),
        ctypes.c_void_p(img_aug_rot_ptr),
        ctypes.c_void_p(img_aug_trans_ptr),
        B, N, iH, iW,
        ctypes.c_void_p(depth_map_ptr),
    )
    if ret != 0:
        raise RuntimeError(f"compute_depth_map_cuda failed: CUDA error {ret}")
    return ret


def vtransform_gpu_workspace_size(B, N, D, fH, fW, grid_x, grid_y, grid_z, C_bev):
    return _lib.vtransform_gpu_workspace_size(B, N, D, fH, fW, grid_x, grid_y, grid_z, C_bev)


def vtransform_post_depthnet_cuda(
    frustum_ptr, inv_post_rots_ptr, post_trans_ptr,
    combine_rots_ptr, camera2lidar_trans_ptr,
    extra_rots_ptr, extra_trans_ptr,
    B, N, D, fH, fW,
    voxel_x, voxel_y, voxel_z,
    coors_x_min, coors_y_min, coors_z_min,
    grid_x, grid_y, grid_z,
    cam_feats_ptr, cam_feats_dtype, C_bev,
    camera_bev_ptr,
    geom_feats_out_ptr, interval_starts_out_ptr, interval_lengths_out_ptr,
    out_K_ptr, out_M_ptr,
    workspace_ptr, workspace_size,
):
    ret = _lib.vtransform_post_depthnet_cuda(
        ctypes.c_void_p(frustum_ptr),
        ctypes.c_void_p(inv_post_rots_ptr),
        ctypes.c_void_p(post_trans_ptr),
        ctypes.c_void_p(combine_rots_ptr),
        ctypes.c_void_p(camera2lidar_trans_ptr),
        ctypes.c_void_p(extra_rots_ptr) if extra_rots_ptr is not None else ctypes.c_void_p(0),
        ctypes.c_void_p(extra_trans_ptr) if extra_trans_ptr is not None else ctypes.c_void_p(0),
        B, N, D, fH, fW,
        ctypes.c_float(voxel_x), ctypes.c_float(voxel_y), ctypes.c_float(voxel_z),
        ctypes.c_float(coors_x_min), ctypes.c_float(coors_y_min), ctypes.c_float(coors_z_min),
        grid_x, grid_y, grid_z,
        ctypes.c_void_p(cam_feats_ptr),
        cam_feats_dtype,
        C_bev,
        ctypes.c_void_p(camera_bev_ptr),
        ctypes.c_void_p(geom_feats_out_ptr),
        ctypes.c_void_p(interval_starts_out_ptr),
        ctypes.c_void_p(interval_lengths_out_ptr),
        ctypes.c_void_p(out_K_ptr),
        ctypes.c_void_p(out_M_ptr),
        ctypes.c_void_p(workspace_ptr),
        workspace_size,
    )
    if ret != 0:
        raise RuntimeError(f"vtransform_post_depthnet_cuda failed: CUDA error {ret}")
    return ret
