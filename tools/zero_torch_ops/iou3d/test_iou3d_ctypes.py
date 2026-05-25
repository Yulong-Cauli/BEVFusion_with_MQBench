"""
Numerical equivalence test:
  torch-based iou3d_cuda.nms_gpu  vs  ctypes-based libbevfusion_iou3d.nms_gpu_cuda
"""
import os
import sys
import subprocess
import time

# Build the .so if it doesn't exist
SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_iou3d.so")
if not os.path.exists(SO_PATH):
    print("Building libbevfusion_iou3d.so...")
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), "build_iou3d.py")])

import torch
import numpy as np

# Pre-load the original extension
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "build_sp39"))
import iou3d_cuda as _iou3d_cuda

from iou3d_wrapper import nms_gpu


def make_rotated_boxes(num, device="cuda"):
    """Generate realistic oriented BEV boxes [x1, y1, x2, y2, ry]."""
    centers = torch.randn(num, 2, device=device) * 20.0
    sizes = torch.rand(num, 2, device=device) * 4.0 + 1.0
    angles = (torch.rand(num, device=device) * 3.14159) - 1.570795
    x1 = centers[:, 0] - sizes[:, 0] / 2
    y1 = centers[:, 1] - sizes[:, 1] / 2
    x2 = centers[:, 0] + sizes[:, 0] / 2
    y2 = centers[:, 1] + sizes[:, 1] / 2
    boxes = torch.stack([x1, y1, x2, y2, angles], dim=1).float().contiguous()
    return boxes


def test_nms_gpu_small():
    torch.manual_seed(42)
    boxes = make_rotated_boxes(64)
    keep_ref = torch.zeros(boxes.size(0), dtype=torch.long)
    num_ref = _iou3d_cuda.nms_gpu(boxes, keep_ref, 0.5, boxes.device.index)
    keep_ref = keep_ref[:num_ref].cpu().numpy()

    keep_ct = torch.zeros(boxes.size(0), dtype=torch.long)
    num_ct = nms_gpu(boxes.data_ptr(), keep_ct.data_ptr(), boxes.size(0), 0.5, boxes.device.index)
    keep_ct = keep_ct[:num_ct].cpu().numpy()

    assert num_ref == num_ct, f"num mismatch: ref={num_ref}, ct={num_ct}"
    assert np.array_equal(keep_ref, keep_ct), f"keep mismatch: ref={keep_ref}, ct={keep_ct}"
    print(f"✅ nms_gpu small test PASSED (kept {num_ref}/{boxes.size(0)})")


def test_nms_gpu_large():
    """Stress test with ~2000 boxes — typical TransFusionHead output magnitude."""
    torch.manual_seed(0)
    boxes = make_rotated_boxes(2048)
    keep_ref = torch.zeros(boxes.size(0), dtype=torch.long)

    t0 = time.time()
    num_ref = _iou3d_cuda.nms_gpu(boxes, keep_ref, 0.3, boxes.device.index)
    t1 = time.time()
    dt_ref = (t1 - t0) * 1000.0
    keep_ref = keep_ref[:num_ref].cpu().numpy()

    keep_ct = torch.zeros(boxes.size(0), dtype=torch.long)
    t0 = time.time()
    num_ct = nms_gpu(boxes.data_ptr(), keep_ct.data_ptr(), boxes.size(0), 0.3, boxes.device.index)
    t1 = time.time()
    dt_ct = (t1 - t0) * 1000.0
    keep_ct = keep_ct[:num_ct].cpu().numpy()

    assert num_ref == num_ct, f"num mismatch: ref={num_ref}, ct={num_ct}"
    assert np.array_equal(keep_ref, keep_ct), f"keep mismatch"
    print(f"✅ nms_gpu large test PASSED (kept {num_ref}/{boxes.size(0)})")
    print(f"   ref={dt_ref:.3f}ms  ctypes={dt_ct:.3f}ms")


def benchmark_nms_gpu():
    """Benchmark representative sizes for Orin feasibility assessment."""
    torch.manual_seed(0)
    print("\n--- NMS GPU Benchmark (x86 RTX 3090 / A100) ---")
    for n in [256, 512, 1024, 2048, 4096]:
        boxes = make_rotated_boxes(n)
        keep_ct = torch.zeros(n, dtype=torch.long)

        # Warmup
        for _ in range(3):
            nms_gpu(boxes.data_ptr(), keep_ct.data_ptr(), n, 0.3, boxes.device.index)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            t0 = time.time()
            nms_gpu(boxes.data_ptr(), keep_ct.data_ptr(), n, 0.3, boxes.device.index)
            torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)

        avg = sum(times) / len(times)
        print(f"  N={n:5d}: {avg:.3f}ms  (projected Orin: {avg*3:.1f}~{avg*5:.1f}ms)")


if __name__ == "__main__":
    test_nms_gpu_small()
    test_nms_gpu_large()
    benchmark_nms_gpu()
