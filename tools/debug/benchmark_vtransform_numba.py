"""
Benchmark Numba-accelerated VTransform on x86 CPU.
"""
import sys, os, time
import numpy as np
from numba import njit, prange


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


def random_intrinsics(N):
    intrins = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        intrins[i] = np.eye(3, dtype=np.float32)
        intrins[i, 0, 0] = 1200.0
        intrins[i, 1, 1] = 800.0
        intrins[i, 0, 2] = 800.0
        intrins[i, 1, 2] = 450.0
    return intrins


def random_lidar_points(B, num_points=20000):
    pts = []
    for b in range(B):
        p = np.random.randn(num_points, 5).astype(np.float32)
        p[:, 0] = p[:, 0] * 20.0
        p[:, 1] = p[:, 1] * 10.0
        p[:, 2] = p[:, 2] * 2.0
        pts.append(p)
    return pts


@njit(parallel=True, fastmath=True)
def _transform_frustum_numba(frustum, M1, t1, M2, t2, out):
    B, N, P = M1.shape[0], M1.shape[1], frustum.shape[0]
    for b in prange(B):
        for n in range(N):
            m1 = M1[b, n]
            tt1 = t1[b, n]
            m2 = M2[b, n]
            tt2 = t2[b, n]
            for i in range(P):
                x = frustum[i, 0]
                y = frustum[i, 1]
                z = frustum[i, 2]

                # p1 = m1 @ p + t1
                x1 = m1[0, 0] * x + m1[0, 1] * y + m1[0, 2] * z + tt1[0]
                y1 = m1[1, 0] * x + m1[1, 1] * y + m1[1, 2] * z + tt1[1]
                z1 = m1[2, 0] * x + m1[2, 1] * y + m1[2, 2] * z + tt1[2]

                # nonlinear: xy *= z
                x1 *= z1
                y1 *= z1

                # p2 = m2 @ [x1, y1, z1] + t2
                ox = m2[0, 0] * x1 + m2[0, 1] * y1 + m2[0, 2] * z1 + tt2[0]
                oy = m2[1, 0] * x1 + m2[1, 1] * y1 + m2[1, 2] * z1 + tt2[1]
                oz = m2[2, 0] * x1 + m2[2, 1] * y1 + m2[2, 2] * z1 + tt2[2]

                out[b, n, i, 0] = ox
                out[b, n, i, 1] = oy
                out[b, n, i, 2] = oz


def benchmark():
    B, N = 1, 6
    image_size = (900, 1600)
    feature_size = (32, 88)
    xbound = [0.0, 51.2, 0.4]
    ybound = [-25.6, 25.6, 0.4]
    zbound = [-2.0, 4.4, 0.4]
    dbound = [1.0, 60.0, 0.5]

    # Build frustum
    iH, iW = image_size
    fH, fW = feature_size
    ds = np.arange(*dbound, dtype=np.float32).reshape(-1, 1, 1)
    ds = np.broadcast_to(ds, (ds.shape[0], fH, fW))
    D = ds.shape[0]
    xs = np.linspace(0, iW - 1, fW, dtype=np.float32).reshape(1, 1, fW)
    xs = np.broadcast_to(xs, (D, fH, fW))
    ys = np.linspace(0, iH - 1, fH, dtype=np.float32).reshape(1, fH, 1)
    ys = np.broadcast_to(ys, (D, fH, fW))
    frustum = np.stack((xs, ys, ds), axis=-1)  # [D, fH, fW, 3]
    P = D * fH * fW

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

    # Test fused numba kernel for geometry only
    frustum_flat = frustum.reshape(P, 3).astype(np.float32)
    out = np.empty((B, N, P, 3), dtype=np.float32)

    # Precompute per-(b,n) transforms: M1, t1, M2, t2
    M1 = np.empty((B, N, 3, 3), dtype=np.float32)
    t1 = np.empty((B, N, 3), dtype=np.float32)
    M2 = np.empty((B, N, 3, 3), dtype=np.float32)
    t2 = np.empty((B, N, 3), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            _M1 = np.linalg.inv(post_rots[b, n]).astype(np.float32)
            _t1 = -_M1 @ post_trans[b, n]
            _inv_intrins = np.linalg.inv(camera_intrinsics[n]).astype(np.float32)
            _M2 = camera2lidar_rots[b, n].astype(np.float32) @ _inv_intrins
            _t2 = camera2lidar_trans[b, n].astype(np.float32)
            if extra_rots is not None:
                _M2 = extra_rots[b].astype(np.float32) @ _M2
                _t2 = extra_rots[b].astype(np.float32) @ _t2 + extra_trans[b].astype(np.float32)
            M1[b, n] = _M1
            t1[b, n] = _t1
            M2[b, n] = _M2
            t2[b, n] = _t2

    # Warmup + JIT compile
    _transform_frustum_numba(frustum_flat, M1, t1, M2, t2, out)

    times = []
    for _ in range(50):
        t0 = time.time()
        _transform_frustum_numba(frustum_flat, M1, t1, M2, t2, out)
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    avg = sum(times) / len(times)
    print(f"Numba get_geometry (fused): {avg:.3f}ms")

    # verify shape
    print(f"out shape: {out.shape}")


# ------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _transform_frustum_numba(frustum, M1, t1, M2, t2, out):
    B, N, P = M1.shape[0], M1.shape[1], frustum.shape[0]
    for b in prange(B):
        for n in range(N):
            m1 = M1[b, n]
            tt1 = t1[b, n]
            m2 = M2[b, n]
            tt2 = t2[b, n]
            for i in range(P):
                x = frustum[i, 0]
                y = frustum[i, 1]
                z = frustum[i, 2]

                # p1 = m1 @ p + t1
                x1 = m1[0, 0] * x + m1[0, 1] * y + m1[0, 2] * z + tt1[0]
                y1 = m1[1, 0] * x + m1[1, 1] * y + m1[1, 2] * z + tt1[1]
                z1 = m1[2, 0] * x + m1[2, 1] * y + m1[2, 2] * z + tt1[2]

                # nonlinear: xy *= z
                x1 *= z1
                y1 *= z1

                # p2 = m2 @ [x1, y1, z1] + t2
                ox = m2[0, 0] * x1 + m2[0, 1] * y1 + m2[0, 2] * z1 + tt2[0]
                oy = m2[1, 0] * x1 + m2[1, 1] * y1 + m2[1, 2] * z1 + tt2[1]
                oz = m2[2, 0] * x1 + m2[2, 1] * y1 + m2[2, 2] * z1 + tt2[2]

                out[b, n, i, 0] = ox
                out[b, n, i, 1] = oy
                out[b, n, i, 2] = oz


if __name__ == "__main__":
    benchmark()
