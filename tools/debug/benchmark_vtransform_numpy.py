"""
Benchmark NumpyVTransformGeometry on x86 CPU.
Tests whether VTransform is feasible on Orin A78 CPU.
x86 threshold: < 10ms total (accounting for 3~5x slower Orin CPU).
"""
import sys
import os
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.trt_infer_zero_torch import NumpyVTransformGeometry


def random_intrinsics(N):
    intrins = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        intrins[i] = np.eye(3, dtype=np.float32)
        intrins[i, 0, 0] = 1200.0  # fx
        intrins[i, 1, 1] = 800.0   # fy
        intrins[i, 0, 2] = 800.0   # cx
        intrins[i, 1, 2] = 450.0   # cy
    return intrins


def random_affine_matrices(B, N=None):
    if N is not None:
        mats = np.zeros((B, N, 4, 4), dtype=np.float32)
        for b in range(B):
            for n in range(N):
                m = np.eye(4, dtype=np.float32)
                angle = np.random.uniform(-0.1, 0.1)
                m[:3, :3] = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                m[:3, 3] = np.random.randn(3).astype(np.float32) * 0.5
                mats[b, n] = m
    else:
        mats = np.zeros((B, 4, 4), dtype=np.float32)
        for b in range(B):
            m = np.eye(4, dtype=np.float32)
            angle = np.random.uniform(-0.1, 0.1)
            m[:3, :3] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            m[:3, 3] = np.random.randn(3).astype(np.float32) * 0.5
            mats[b] = m
    return mats


def random_lidar_points(B, num_points=20000):
    pts = []
    for b in range(B):
        p = np.random.randn(num_points, 5).astype(np.float32)
        p[:, 0] = p[:, 0] * 20.0  # x
        p[:, 1] = p[:, 1] * 10.0  # y
        p[:, 2] = p[:, 2] * 2.0   # z
        pts.append(p)
    return pts


def benchmark():
    B = 1
    N = 6
    image_size = (900, 1600)
    feature_size = (32, 88)
    xbound = [0.0, 51.2, 0.4]
    ybound = [-25.6, 25.6, 0.4]
    zbound = [-2.0, 4.4, 0.4]
    dbound = [1.0, 60.0, 0.5]

    geom = NumpyVTransformGeometry(image_size, feature_size, xbound, ybound, zbound, dbound)
    print(f"Frustum shape: {geom.frustum.shape}, D={geom.D}")

    camera2lidar = random_affine_matrices(B, N)
    camera_intrinsics = random_intrinsics(N)
    camera2lidar_rots = camera2lidar[..., :3, :3]
    camera2lidar_trans = camera2lidar[..., :3, 3]
    img_aug_matrix = random_affine_matrices(B, N)
    lidar_aug_matrix = random_affine_matrices(B, None)
    lidar2image = random_affine_matrices(B, N)
    points = random_lidar_points(B)

    post_rots = img_aug_matrix[..., :3, :3]
    post_trans = img_aug_matrix[..., :3, 3]
    extra_rots = lidar_aug_matrix[..., :3, :3]
    extra_trans = lidar_aug_matrix[..., :3, 3]

    # Warmup
    for _ in range(5):
        g = geom.get_geometry(camera2lidar_rots, camera2lidar_trans, camera_intrinsics,
                              post_rots, post_trans, extra_rots, extra_trans)
        idx = geom.precompute_bev_indices(g, B)
        d = geom.compute_depth_map(points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N)

    times_geo = []
    times_idx = []
    times_depth = []
    for _ in range(50):
        t0 = time.time()
        g = geom.get_geometry(camera2lidar_rots, camera2lidar_trans, camera_intrinsics,
                              post_rots, post_trans, extra_rots, extra_trans)
        t1 = time.time()
        times_geo.append((t1 - t0) * 1000.0)

        t0 = time.time()
        idx = geom.precompute_bev_indices(g, B)
        t1 = time.time()
        times_idx.append((t1 - t0) * 1000.0)

        t0 = time.time()
        d = geom.compute_depth_map(points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N)
        t1 = time.time()
        times_depth.append((t1 - t0) * 1000.0)

    avg_geo = sum(times_geo) / len(times_geo)
    avg_idx = sum(times_idx) / len(times_idx)
    avg_depth = sum(times_depth) / len(times_depth)
    total_avg = avg_geo + avg_idx + avg_depth

    print(f"\n--- Numpy VTransform Benchmark (x86 CPU) ---")
    print(f"  get_geometry    : {avg_geo:.3f}ms")
    print(f"  precompute_bev  : {avg_idx:.3f}ms")
    print(f"  compute_depth   : {avg_depth:.3f}ms")
    print(f"  TOTAL           : {total_avg:.3f}ms")
    print(f"\n  Projected Orin (3x): {total_avg*3:.1f}ms")
    print(f"  Projected Orin (5x): {total_avg*5:.1f}ms")

    if total_avg < 10.0:
        print("  ✅ PASS — x86 < 10ms, likely safe on Orin A78")
    else:
        print("  ❌ FAIL — x86 >= 10ms, Orin A78 CPU will be the bottleneck")
        print("  Recommendation: optimize with Numba/Cython or move to GPU kernel")


if __name__ == "__main__":
    benchmark()
