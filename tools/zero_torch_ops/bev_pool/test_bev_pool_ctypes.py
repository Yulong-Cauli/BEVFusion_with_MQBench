"""
Numerical equivalence test:
  torch-based bev_pool_ext  vs  ctypes-based libbevfusion_bev_pool
"""
import os
import sys
import subprocess
import numpy as np

# 1. Build the .so if it doesn't exist
SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_bev_pool.so")
if not os.path.exists(SO_PATH):
    print("Building libbevfusion_bev_pool.so...")
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "build_bev_pool.py")])

import torch

# Pre-load the original extension (same as trt_infer_standalone.py does)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "build_sp39"))
import bev_pool_ext as _bev_pool_ext

from bev_pool_wrapper import bev_pool_forward


def test_bev_pool():
    torch.manual_seed(42)

    # Small synthetic test case
    b, d, h, w = 1, 4, 8, 8
    n = 100
    c = 16
    n_intervals = 20

    x = torch.randn(n, c, device="cuda", dtype=torch.float32)
    geom_feats = torch.randint(0, min(d, h, w), (n, 4), device="cuda", dtype=torch.int32)
    interval_starts = torch.arange(0, n, n // n_intervals, device="cuda", dtype=torch.int32)[:n_intervals]
    interval_lengths = torch.full((n_intervals,), n // n_intervals, device="cuda", dtype=torch.int32)

    # Torch reference
    out_ref = _bev_pool_ext.bev_pool_forward(
        x, geom_feats, interval_lengths, interval_starts, b, d, h, w
    )
    # Reference output shape: [b, d, h, w, c]

    # Ctypes version: allocate output buffer externally
    out_ct = torch.zeros(b, d, h, w, c, device="cuda", dtype=torch.float32)
    bev_pool_forward(
        x.data_ptr(),
        geom_feats.data_ptr(),
        interval_starts.data_ptr(),
        interval_lengths.data_ptr(),
        out_ct.data_ptr(),
        b, d, h, w, n, c, n_intervals,
    )

    diff = (out_ref - out_ct).abs()
    max_diff = diff.max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.flatten(), out_ct.flatten(), dim=0
    ).item()

    print(f"max_diff={max_diff:.6e}, cosine_sim={cos_sim:.8f}")
    assert max_diff < 1e-5, f"Mismatch! max_diff={max_diff}"
    assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"
    print("✅ bev_pool ctypes test PASSED")


def test_bev_pool_realistic():
    """Use dimensions close to the real BEVFusion config.
    Input mimics actual precompute_bev_indices output: geom_feats sorted by rank,
    and intervals cover contiguous sorted ranges.
    """
    torch.manual_seed(0)
    b, d, h, w = 1, 59, 180, 180
    n = 50000
    c = 80
    n_intervals = 12000

    # Generate valid (x, y, z, batch) with batch=0 and bounded coords
    geom_list = []
    for _ in range(n):
        gx = torch.randint(0, w, (1,)).item()
        gy = torch.randint(0, h, (1,)).item()
        gz = torch.randint(0, d, (1,)).item()
        rank = gz * h * w + gy * w + gx  # simple z-major ordering
        geom_list.append((rank, gx, gy, gz))
    # Sort by rank so identical (x,y,z) are contiguous
    geom_list.sort(key=lambda t: t[0])
    geom_np = np.array([[t[1], t[2], t[3], 0] for t in geom_list], dtype=np.int32)
    geom_feats = torch.from_numpy(geom_np).cuda()

    # Build intervals: contiguous chunks of identical rank
    step = max(1, n // n_intervals)
    starts = []
    lengths = []
    for i in range(n_intervals):
        s = i * step
        e = min(n, s + step)
        if s >= n:
            break
        starts.append(s)
        lengths.append(e - s)
    interval_starts = torch.tensor(starts, device="cuda", dtype=torch.int32)
    interval_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    n_intervals = len(starts)

    x = torch.randn(n, c, device="cuda", dtype=torch.float32)

    out_ref = _bev_pool_ext.bev_pool_forward(
        x, geom_feats, interval_lengths, interval_starts, b, d, h, w
    )

    out_ct = torch.zeros(b, d, h, w, c, device="cuda", dtype=torch.float32)
    bev_pool_forward(
        x.data_ptr(),
        geom_feats.data_ptr(),
        interval_starts.data_ptr(),
        interval_lengths.data_ptr(),
        out_ct.data_ptr(),
        b, d, h, w, n, c, n_intervals,
    )

    max_diff = (out_ref - out_ct).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.flatten(), out_ct.flatten(), dim=0
    ).item()

    print(f"[Realistic] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.8f}")
    assert max_diff < 1e-5, f"Mismatch! max_diff={max_diff}"
    assert cos_sim > 0.99999, f"Cosine similarity too low: {cos_sim}"
    print("✅ bev_pool realistic test PASSED")


if __name__ == "__main__":
    test_bev_pool()
    test_bev_pool_realistic()
