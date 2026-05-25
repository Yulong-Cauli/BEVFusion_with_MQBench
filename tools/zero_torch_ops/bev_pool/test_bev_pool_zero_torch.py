"""
Numerical equivalence test for zero-torch bev_pool.
Uses pure-numpy reference (no torch imports) vs ctypes-based libbevfusion_bev_pool.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from tools.trt_infer_zero_torch import np_bev_pool_v2


def ref_bev_pool_v2(x_np, geom_feats_np, interval_starts_np, interval_lengths_np, B, D, H, W):
    """Pure numpy reference for bev_pool_v2 matching CUDA kernel layout [B,D,H,W,C].

    geom_feats columns: [H, W, D, B] (matches real BEVFusion precompute_bev_indices).
    """
    C = x_np.shape[1]
    out = np.zeros((B, D, H, W, C), dtype=np.float32)
    for s, l in zip(interval_starts_np, interval_lengths_np):
        chunk = x_np[s:s + l]
        feat = chunk.sum(axis=0)  # [C]
        h_coord, w_coord, d_coord, b_coord = geom_feats_np[s]
        out[b_coord, d_coord, h_coord, w_coord, :] += feat
    return out.transpose(0, 4, 1, 2, 3)  # -> [B,C,D,H,W]


def test_bev_pool_small():
    b, d, h, w = 1, 4, 8, 8
    n = 100
    c = 16
    n_intervals = 20

    np.random.seed(42)
    x = np.random.randn(n, c).astype(np.float32)
    # geom columns: [H, W, D, B]
    h_coords = np.random.randint(0, h, size=n)
    w_coords = np.random.randint(0, w, size=n)
    d_coords = np.random.randint(0, d, size=n)
    b_coords = np.zeros(n, dtype=np.int32)
    geom = np.stack([h_coords, w_coords, d_coords, b_coords], axis=1).astype(np.int32)

    # Sort by rank (H-major) so identical coords are contiguous
    rank = d_coords * h * w + h_coords * w + w_coords
    sort_idx = np.argsort(rank)
    geom = geom[sort_idx]
    x = x[sort_idx]

    # Build intervals ensuring each interval has identical geom coords
    starts = []
    lengths = []
    i = 0
    while i < n and len(starts) < n_intervals:
        j = i + 1
        while j < n and np.array_equal(geom[j], geom[i]):
            j += 1
        starts.append(i)
        lengths.append(j - i)
        i = j
    starts = np.array(starts, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.int32)

    out_ref = ref_bev_pool_v2(x, geom, starts, lengths, b, d, h, w)
    out_ct = np_bev_pool_v2(x, geom, starts, lengths, b, d, h, w)

    max_diff = np.abs(out_ref - out_ct).max()
    cos_sim = np.dot(out_ref.flatten(), out_ct.flatten()) / (
        np.linalg.norm(out_ref.flatten()) * np.linalg.norm(out_ct.flatten()) + 1e-12
    )
    print(f"[Small] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.8f}")
    assert max_diff < 1e-4, f"Mismatch! max_diff={max_diff}"
    assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"
    print("✅ bev_pool small zero-torch test PASSED")


def test_bev_pool_realistic():
    b, d, h, w = 1, 59, 180, 180
    n = 50000
    c = 80
    n_intervals = 12000

    np.random.seed(0)
    geom_list = []
    for _ in range(n):
        hh = np.random.randint(0, h)
        ww = np.random.randint(0, w)
        dd = np.random.randint(0, d)
        rank = dd * h * w + hh * w + ww
        geom_list.append((rank, hh, ww, dd))
    geom_list.sort(key=lambda t: t[0])
    geom = np.array([[t[1], t[2], t[3], 0] for t in geom_list], dtype=np.int32)

    # Build intervals: contiguous chunks of identical (H,W,D,B)
    starts = []
    lengths = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.array_equal(geom[j], geom[i]):
            j += 1
        starts.append(i)
        lengths.append(j - i)
        i = j
    starts = np.array(starts, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.int32)
    n_intervals = len(starts)

    x = np.random.randn(n, c).astype(np.float32)

    out_ref = ref_bev_pool_v2(x, geom, starts, lengths, b, d, h, w)
    out_ct = np_bev_pool_v2(x, geom, starts, lengths, b, d, h, w)

    max_diff = np.abs(out_ref - out_ct).max()
    cos_sim = np.dot(out_ref.flatten(), out_ct.flatten()) / (
        np.linalg.norm(out_ref.flatten()) * np.linalg.norm(out_ct.flatten()) + 1e-12
    )
    print(f"[Realistic] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.8f}")
    assert max_diff < 1e-4, f"Mismatch! max_diff={max_diff}"
    assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"
    print("✅ bev_pool realistic zero-torch test PASSED")


if __name__ == "__main__":
    test_bev_pool_small()
    test_bev_pool_realistic()
