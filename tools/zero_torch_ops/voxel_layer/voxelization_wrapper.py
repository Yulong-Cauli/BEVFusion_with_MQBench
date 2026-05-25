"""
ctypes wrapper for libbevfusion_voxel_layer.so
No torch import required at runtime.
"""
import ctypes
import os

_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_voxel_layer.so")
_lib = ctypes.CDLL(_SO_PATH)

_lib.hard_voxelize_gpu_cuda.argtypes = [
    ctypes.c_void_p,  # points
    ctypes.c_void_p,  # voxels
    ctypes.c_void_p,  # coors
    ctypes.c_void_p,  # num_points_per_voxel
    ctypes.c_int,     # num_points
    ctypes.c_int,     # num_features
    ctypes.c_float,   # voxel_x
    ctypes.c_float,   # voxel_y
    ctypes.c_float,   # voxel_z
    ctypes.c_float,   # coors_x_min
    ctypes.c_float,   # coors_y_min
    ctypes.c_float,   # coors_z_min
    ctypes.c_float,   # coors_x_max
    ctypes.c_float,   # coors_y_max
    ctypes.c_float,   # coors_z_max
    ctypes.c_int,     # grid_x
    ctypes.c_int,     # grid_y
    ctypes.c_int,     # grid_z
    ctypes.c_int,     # max_points
    ctypes.c_int,     # max_voxels
    ctypes.c_int,     # NDim
    ctypes.c_int,     # device_id
]
_lib.hard_voxelize_gpu_cuda.restype = ctypes.c_int

_lib.dynamic_voxelize_gpu_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.dynamic_voxelize_gpu_cuda.restype = ctypes.c_int


def hard_voxelize_gpu(points_ptr, voxels_ptr, coors_ptr, num_points_per_voxel_ptr,
                      num_points, num_features,
                      voxel_size, coors_range, max_points, max_voxels,
                      NDim=3, device_id=0):
    """
    Deterministic hard voxelize on GPU.

    Args:
        points_ptr: GPU pointer, float [num_points, num_features]
        voxels_ptr: GPU pointer, float [max_voxels, max_points, num_features]
        coors_ptr: GPU pointer, int32 [max_voxels, NDim]
        num_points_per_voxel_ptr: GPU pointer, int32 [max_voxels]
        num_points, num_features, max_points, max_voxels: int
        voxel_size: list/tuple of 3 floats
        coors_range: list/tuple of 6 floats [x_min, y_min, z_min, x_max, y_max, z_max]
        NDim: int, typically 3
        device_id: int

    Returns:
        voxel_num: int (number of generated voxels)
    """
    voxel_x, voxel_y, voxel_z = voxel_size
    coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max = coors_range
    grid_x = round((coors_x_max - coors_x_min) / voxel_x)
    grid_y = round((coors_y_max - coors_y_min) / voxel_y)
    grid_z = round((coors_z_max - coors_z_min) / voxel_z)

    ret = _lib.hard_voxelize_gpu_cuda(
        ctypes.c_void_p(points_ptr),
        ctypes.c_void_p(voxels_ptr),
        ctypes.c_void_p(coors_ptr),
        ctypes.c_void_p(num_points_per_voxel_ptr),
        num_points,
        num_features,
        voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min,
        coors_x_max, coors_y_max, coors_z_max,
        grid_x, grid_y, grid_z,
        max_points, max_voxels,
        NDim,
        device_id,
    )
    if ret < 0:
        raise RuntimeError(f"hard_voxelize_gpu_cuda failed (CUDA error code={ret})")
    return ret


def dynamic_voxelize_gpu(points_ptr, coors_ptr, num_points, num_features,
                         voxel_size, coors_range, NDim=3, device_id=0):
    voxel_x, voxel_y, voxel_z = voxel_size
    coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max, coors_z_max = coors_range
    grid_x = round((coors_x_max - coors_x_min) / voxel_x)
    grid_y = round((coors_y_max - coors_y_min) / voxel_y)
    grid_z = round((coors_z_max - coors_z_min) / voxel_z)

    ret = _lib.dynamic_voxelize_gpu_cuda(
        ctypes.c_void_p(points_ptr),
        ctypes.c_void_p(coors_ptr),
        num_points,
        num_features,
        voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min,
        coors_x_max, coors_y_max, coors_z_max,
        grid_x, grid_y, grid_z,
        NDim,
        device_id,
    )
    if ret != 0:
        raise RuntimeError(f"dynamic_voxelize_gpu_cuda failed (CUDA error code={ret})")
    return ret
