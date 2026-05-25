"""
Functional correctness test for libbevfusion_voxel_layer.so
Zero-torch: uses ctypes + libcudart for GPU memory.
"""
import os
import ctypes
import numpy as np

from voxelization_wrapper import hard_voxelize_gpu, dynamic_voxelize_gpu

cudart = ctypes.CDLL("libcudart.so.11.0")
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cudart.cudaMemset.restype = ctypes.c_int
cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaSetDevice.restype = ctypes.c_int

CUDA_MEMCPY_H2D = 1
CUDA_MEMCPY_D2H = 2


def gpu_alloc(nbytes):
    p = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(p), nbytes)
    if ret != 0:
        raise RuntimeError(f"cudaMalloc failed: {ret}")
    return p


def gpu_free(p):
    if p.value:
        cudart.cudaFree(p)


def memcpy_h2d(dst, src, nbytes):
    ret = cudart.cudaMemcpy(dst, src.ctypes.data, nbytes, CUDA_MEMCPY_H2D)
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed: {ret}")


def memcpy_d2h(dst, src, nbytes):
    ret = cudart.cudaMemcpy(dst.ctypes.data, src, nbytes, CUDA_MEMCPY_D2H)
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed: {ret}")


def make_points(num, feat_dim=4, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.random((num, feat_dim), dtype=np.float32)
    # scale xyz to a reasonable range, e.g. [0, 70] x [-40, 40] x [-3, 1]
    pts[:, 0] = pts[:, 0] * 70.0
    pts[:, 1] = pts[:, 1] * 80.0 - 40.0
    pts[:, 2] = pts[:, 2] * 4.0 - 3.0
    return pts


def test_hard_voxelize_basic():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0.0, -40.0, -3.0, 70.0, 40.0, 1.0]
    max_points = 10
    max_voxels = 200
    num_points = 1024
    feat_dim = 4
    NDim = 3

    points = make_points(num_points, feat_dim, seed=7)

    d_points = gpu_alloc(points.nbytes)
    d_voxels = gpu_alloc(max_voxels * max_points * feat_dim * 4)
    d_coors = gpu_alloc(max_voxels * NDim * 4)
    d_num_per_voxel = gpu_alloc(max_voxels * 4)

    try:
        memcpy_h2d(d_points, points, points.nbytes)
        cudart.cudaMemset(d_voxels, 0, max_voxels * max_points * feat_dim * 4)
        cudart.cudaMemset(d_coors, 0, max_voxels * NDim * 4)
        cudart.cudaMemset(d_num_per_voxel, 0, max_voxels * 4)

        voxel_num = hard_voxelize_gpu(
            d_points.value, d_voxels.value, d_coors.value, d_num_per_voxel.value,
            num_points, feat_dim,
            voxel_size, point_cloud_range, max_points, max_voxels,
            NDim=NDim, device_id=0
        )

        voxels = np.zeros((max_voxels, max_points, feat_dim), dtype=np.float32)
        coors = np.zeros((max_voxels, NDim), dtype=np.int32)
        num_per_voxel = np.zeros(max_voxels, dtype=np.int32)

        memcpy_d2h(voxels, d_voxels, voxels.nbytes)
        memcpy_d2h(coors, d_coors, coors.nbytes)
        memcpy_d2h(num_per_voxel, d_num_per_voxel, num_per_voxel.nbytes)

        assert 0 <= voxel_num <= max_voxels, f"voxel_num out of range: {voxel_num}"
        assert np.all(num_per_voxel[:voxel_num] > 0), "all valid voxels must have >0 points"
        assert np.all(num_per_voxel[:voxel_num] <= max_points), "points per voxel exceed max_points"

        grid_x = round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        grid_y = round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        grid_z = round((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])

        for i in range(voxel_num):
            c = coors[i]
            assert 0 <= c[0] < grid_x
            assert 0 <= c[1] < grid_y
            assert 0 <= c[2] < grid_z

        # Verify that points inside each voxel share the same computed coordinate
        for i in range(voxel_num):
            n = num_per_voxel[i]
            for j in range(n):
                p = voxels[i, j]
                cx = int(np.floor((p[0] - point_cloud_range[0]) / voxel_size[0]))
                cy = int(np.floor((p[1] - point_cloud_range[1]) / voxel_size[1]))
                cz = int(np.floor((p[2] - point_cloud_range[2]) / voxel_size[2]))
                assert cx == coors[i, 0] and cy == coors[i, 1] and cz == coors[i, 2], \
                    f"point coordinate mismatch in voxel {i}"

        print(f"✅ hard_voxelize basic test PASSED (voxel_num={voxel_num})")
    finally:
        gpu_free(d_points)
        gpu_free(d_voxels)
        gpu_free(d_coors)
        gpu_free(d_num_per_voxel)


def test_dynamic_voxelize_basic():
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0.0, -40.0, -3.0, 70.0, 40.0, 1.0]
    num_points = 512
    feat_dim = 4
    NDim = 3

    points = make_points(num_points, feat_dim, seed=9)
    d_points = gpu_alloc(points.nbytes)
    d_coors = gpu_alloc(num_points * NDim * 4)

    try:
        memcpy_h2d(d_points, points, points.nbytes)
        dynamic_voxelize_gpu(
            d_points.value, d_coors.value, num_points, feat_dim,
            voxel_size, point_cloud_range, NDim=NDim, device_id=0
        )
        coors = np.zeros((num_points, NDim), dtype=np.int32)
        memcpy_d2h(coors, d_coors, coors.nbytes)

        grid_x = round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        grid_y = round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
        grid_z = round((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])

        for i in range(num_points):
            c = coors[i]
            if c[0] == -1:
                continue
            assert 0 <= c[0] < grid_x
            assert 0 <= c[1] < grid_y
            assert 0 <= c[2] < grid_z
            p = points[i]
            cx = int(np.floor((p[0] - point_cloud_range[0]) / voxel_size[0]))
            cy = int(np.floor((p[1] - point_cloud_range[1]) / voxel_size[1]))
            cz = int(np.floor((p[2] - point_cloud_range[2]) / voxel_size[2]))
            assert cx == c[0] and cy == c[1] and cz == c[2]

        print("✅ dynamic_voxelize basic test PASSED")
    finally:
        gpu_free(d_points)
        gpu_free(d_coors)


if __name__ == "__main__":
    cudart.cudaSetDevice(0)
    test_hard_voxelize_basic()
    test_dynamic_voxelize_basic()
