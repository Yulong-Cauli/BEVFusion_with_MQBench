"""
Benchmark optimized numpy VTransform on x86 CPU.
"""
import sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.trt_infer_zero_torch import NumpyVTransformGeometry


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


def fast_get_geometry(frustum, camera2lidar_rots, camera2lidar_trans, intrins,
                      post_rots, post_trans, extra_rots=None, extra_trans=None):
    """
    Optimized get_geometry using einsum to avoid giant broadcast copies.
    """
    B, N = camera2lidar_trans.shape[:2]
    D, fH, fW, _ = frustum.shape
    P = D * fH * fW
    # Flatten spatial dims to P
    points = frustum.reshape(1, 1, P, 3)  # [1, 1, P, 3]

    # Step 1: subtract post_trans
    points = points - post_trans.reshape(B, N, 1, 3)  # [B, N, P, 3]

    # Step 2: apply inv_post_rots
    inv_post_rots = np.linalg.inv(post_rots)  # [B, N, 3, 3]
    points = np.einsum('bnij,bnpj->bnpi', inv_post_rots, points)

    # Step 3: (x,y) *= z
    points_xy = points[..., :2] * points[..., 2:3]
    points = np.concatenate([points_xy, points[..., 2:3]], axis=-1)

    # Step 4: camera2lidar_rots @ inv_intrins
    inv_intrins = np.linalg.inv(intrins)  # [B, N, 3, 3]
    combine = np.matmul(camera2lidar_rots, inv_intrins)  # [B, N, 3, 3]
    points = np.einsum('bnij,bnpj->bnpi', combine, points)
    points += camera2lidar_trans.reshape(B, N, 1, 3)

    if extra_rots is not None:
        # extra_rots: [B, 3, 3] -> [B, 1, 3, 3]
        er = extra_rots[:, None, :, :]  # [B, 1, 3, 3]
        points = np.einsum('bnij,bnpj->bnpi', er, points)
    if extra_trans is not None:
        points += extra_trans.reshape(B, 1, 1, 3)

    return points.reshape(B, N, D, fH, fW, 3)


def fast_precompute_bev_indices(geom, dx, bx, nx, B):
    N_per_batch = geom.shape[1] * geom.shape[2] * geom.shape[3] * geom.shape[4]
    Nprime = B * N_per_batch
    D_val, H_val, W_val = int(nx[2]), int(nx[0]), int(nx[1])

    # reshape in-place view if possible
    geom_feats = ((geom - (bx - dx / 2.0)) / dx).astype(np.int32)
    geom_feats = geom_feats.reshape(Nprime, 3)

    # build batch index without list concat
    batch_ix = np.empty((Nprime, 1), dtype=np.int32)
    for ix in range(B):
        batch_ix[ix * N_per_batch:(ix + 1) * N_per_batch, 0] = ix
    geom_feats = np.concatenate([geom_feats, batch_ix], axis=1)

    kept = (
        (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) &
        (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) &
        (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
    )
    if not kept.any():
        return {
            "kept": kept, "sort_indices": np.empty(0, dtype=np.int32),
            "geom_feats": np.empty((0, 4), dtype=np.int32),
            "interval_starts": np.empty(0, dtype=np.int32),
            "interval_lengths": np.empty(0, dtype=np.int32),
            "B": B, "D": D_val, "H": H_val, "W": W_val,
        }
    geom_feats = geom_feats[kept]

    # compute ranks in int64
    ranks = (
        geom_feats[:, 0].astype(np.int64) * (W_val * D_val * B)
        + geom_feats[:, 1].astype(np.int64) * (D_val * B)
        + geom_feats[:, 2].astype(np.int64) * B
        + geom_feats[:, 3].astype(np.int64)
    )
    sort_indices = np.argsort(ranks)
    geom_feats = geom_feats[sort_indices]
    ranks = ranks[sort_indices]

    if ranks.size == 0:
        return {
            "kept": kept, "sort_indices": sort_indices,
            "geom_feats": geom_feats, "interval_starts": np.empty(0, dtype=np.int32),
            "interval_lengths": np.empty(0, dtype=np.int32),
            "B": B, "D": D_val, "H": H_val, "W": W_val,
        }

    kept_intervals = np.ones(ranks.shape[0], dtype=bool)
    kept_intervals[1:] = ranks[1:] != ranks[:-1]
    interval_starts = np.where(kept_intervals)[0].astype(np.int32)
    interval_lengths = np.zeros_like(interval_starts)
    if interval_lengths.shape[0] > 1:
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks.shape[0] - interval_starts[-1]

    return {
        "kept": kept, "sort_indices": sort_indices.astype(np.int32),
        "geom_feats": geom_feats.astype(np.int32),
        "interval_starts": interval_starts,
        "interval_lengths": interval_lengths,
        "B": B, "D": D_val, "H": H_val, "W": W_val,
    }


def benchmark():
    B, N = 1, 6
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
        g = fast_get_geometry(geom.frustum, camera2lidar_rots, camera2lidar_trans,
                              camera_intrinsics, post_rots, post_trans,
                              extra_rots, extra_trans)
        idx = fast_precompute_bev_indices(g, geom.dx, geom.bx, geom.nx, B)
        d = geom.compute_depth_map(points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N)

    times_geo = []
    times_idx = []
    times_depth = []
    for _ in range(50):
        t0 = time.time()
        g = fast_get_geometry(geom.frustum, camera2lidar_rots, camera2lidar_trans,
                              camera_intrinsics, post_rots, post_trans,
                              extra_rots, extra_trans)
        t1 = time.time()
        times_geo.append((t1 - t0) * 1000.0)

        t0 = time.time()
        idx = fast_precompute_bev_indices(g, geom.dx, geom.bx, geom.nx, B)
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

    print(f"\n--- Optimized Numpy VTransform Benchmark (x86 CPU) ---")
    print(f"  get_geometry    : {avg_geo:.3f}ms")
    print(f"  precompute_bev  : {avg_idx:.3f}ms")
    print(f"  compute_depth   : {avg_depth:.3f}ms")
    print(f"  TOTAL           : {total_avg:.3f}ms")
    print(f"\n  Projected Orin (3x): {total_avg*3:.1f}ms")
    print(f"  Projected Orin (5x): {total_avg*5:.1f}ms")

    if total_avg < 10.0:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL — still too slow for Orin A78")


if __name__ == "__main__":
    benchmark()
