"""
Functional correctness test for libbevfusion_iou3d.so
Zero-torch: uses ctypes + libcudart for GPU memory.
"""
import os
import ctypes
import numpy as np

from iou3d_wrapper import nms_gpu, nms_normal_gpu, boxes_overlap_bev_gpu, boxes_iou_bev_gpu

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


def make_rotated_boxes(num, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=20.0, size=(num, 2))
    sizes = rng.random((num, 2)) * 4.0 + 1.0
    angles = rng.random(num) * np.pi - np.pi / 2
    x1 = centers[:, 0] - sizes[:, 0] / 2
    y1 = centers[:, 1] - sizes[:, 1] / 2
    x2 = centers[:, 0] + sizes[:, 0] / 2
    y2 = centers[:, 1] + sizes[:, 1] / 2
    boxes = np.stack([x1, y1, x2, y2, angles], axis=1).astype(np.float32)
    return boxes


def make_axis_aligned_boxes(num, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=20.0, size=(num, 2))
    sizes = rng.random((num, 2)) * 4.0 + 1.0
    x1 = centers[:, 0] - sizes[:, 0] / 2
    y1 = centers[:, 1] - sizes[:, 1] / 2
    x2 = centers[:, 0] + sizes[:, 0] / 2
    y2 = centers[:, 1] + sizes[:, 1] / 2
    boxes = np.stack([x1, y1, x2, y2, np.zeros(num)], axis=1).astype(np.float32)
    return boxes


def rotated_iou_simple(a, b):
    """Very rough simplified axis-aligned IoU for sanity checks only."""
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    union = sa + sb - inter
    return inter / max(union, 1e-8)


def test_boxes_overlap_and_iou():
    num_a, num_b = 16, 8
    boxes_a = make_rotated_boxes(num_a, seed=0)
    boxes_b = make_rotated_boxes(num_b, seed=1)

    d_a = gpu_alloc(boxes_a.nbytes)
    d_b = gpu_alloc(boxes_b.nbytes)
    d_overlap = gpu_alloc(num_a * num_b * 4)
    d_iou = gpu_alloc(num_a * num_b * 4)
    try:
        memcpy_h2d(d_a, boxes_a, boxes_a.nbytes)
        memcpy_h2d(d_b, boxes_b, boxes_b.nbytes)

        boxes_overlap_bev_gpu(num_a, d_a.value, num_b, d_b.value, d_overlap.value)
        boxes_iou_bev_gpu(num_a, d_a.value, num_b, d_b.value, d_iou.value)

        overlap = np.zeros((num_a, num_b), dtype=np.float32)
        iou = np.zeros((num_a, num_b), dtype=np.float32)
        memcpy_d2h(overlap, d_overlap, overlap.nbytes)
        memcpy_d2h(iou, d_iou, iou.nbytes)

        # sanity checks
        assert np.all(overlap >= 0), "overlap should be >= 0"
        assert np.all(iou >= -1e-5) and np.all(iou <= 1 + 1e-5), "IoU should be in [0,1]"
        assert np.all(iou <= overlap / (np.minimum((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]), 1e-8)[:, None] +
                                          (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])[None, :] - overlap) + 1e-3), "IoU consistency rough check"
        # self-iou on identical boxes should be ~1
        boxes_self = make_rotated_boxes(4, seed=7)
        d_self = gpu_alloc(boxes_self.nbytes)
        d_self_iou = gpu_alloc(4 * 4 * 4)
        try:
            memcpy_h2d(d_self, boxes_self, boxes_self.nbytes)
            boxes_iou_bev_gpu(4, d_self.value, 4, d_self.value, d_self_iou.value)
            self_iou = np.zeros((4, 4), dtype=np.float32)
            memcpy_d2h(self_iou, d_self_iou, self_iou.nbytes)
            for i in range(4):
                assert abs(self_iou[i, i] - 1.0) < 1e-3, f"self IoU should be ~1, got {self_iou[i, i]}"
        finally:
            gpu_free(d_self)
            gpu_free(d_self_iou)

        print(f"✅ boxes_overlap / boxes_iou functional test PASSED")
    finally:
        gpu_free(d_a)
        gpu_free(d_b)
        gpu_free(d_overlap)
        gpu_free(d_iou)


def test_nms_normal_gpu():
    boxes = make_axis_aligned_boxes(64, seed=3)
    d_boxes = gpu_alloc(boxes.nbytes)
    keep = np.zeros(64, dtype=np.int64)
    try:
        memcpy_h2d(d_boxes, boxes, boxes.nbytes)
        num_keep = nms_normal_gpu(d_boxes.value, keep.ctypes.data, 64, 0.5, 0)
        keep = keep[:num_keep]
        assert num_keep > 0 and num_keep <= 64
        # NMS property: for any kept box, no earlier kept box has IoU > thresh
        for i in range(num_keep):
            for j in range(i):
                ii = keep[i]
                jj = keep[j]
                iou = rotated_iou_simple(boxes[ii], boxes[jj])
                assert iou <= 0.5 + 1e-3, f"nms_normal violation: IoU={iou:.3f} between {jj} and {ii}"
        print(f"✅ nms_normal_gpu functional test PASSED (kept {num_keep}/64)")
    finally:
        gpu_free(d_boxes)


def test_nms_gpu():
    boxes = make_rotated_boxes(64, seed=4)
    d_boxes = gpu_alloc(boxes.nbytes)
    keep = np.zeros(64, dtype=np.int64)
    try:
        memcpy_h2d(d_boxes, boxes, boxes.nbytes)
        num_keep = nms_gpu(d_boxes.value, keep.ctypes.data, 64, 0.5, 0)
        keep = keep[:num_keep]
        assert num_keep > 0 and num_keep <= 64
        # Strict numerical equivalence can't be checked without the old extension,
        # but we verify basic validity.
        assert np.all(keep >= 0) and np.all(keep < 64)
        assert len(np.unique(keep)) == len(keep), "keep indices must be unique"
        print(f"✅ nms_gpu functional test PASSED (kept {num_keep}/64)")
    finally:
        gpu_free(d_boxes)


def benchmark_nms_gpu():
    import time
    print("\n--- NMS GPU Benchmark (zero-torch, ctypes + libcudart) ---")
    for n in [256, 512, 1024, 2048, 4096]:
        boxes = make_rotated_boxes(n, seed=0)
        d_boxes = gpu_alloc(boxes.nbytes)
        keep = np.zeros(n, dtype=np.int64)
        try:
            memcpy_h2d(d_boxes, boxes, boxes.nbytes)
            # warmup
            for _ in range(3):
                nms_gpu(d_boxes.value, keep.ctypes.data, n, 0.3, 0)
            cudart.cudaMemset(0, 0, 0)  # dummy sync?  cudart doesn't expose synchronize easily via ctypes without symbol.
            # We'll just rely on H2D/D2H memcpy for coarse timing or measure raw call overhead.
            times = []
            for _ in range(20):
                t0 = time.time()
                nms_gpu(d_boxes.value, keep.ctypes.data, n, 0.3, 0)
                t1 = time.time()
                times.append((t1 - t0) * 1000.0)
            avg = sum(times) / len(times)
            print(f"  N={n:5d}: {avg:.3f}ms  (projected Orin: {avg*3:.1f}~{avg*5:.1f}ms)")
        finally:
            gpu_free(d_boxes)


if __name__ == "__main__":
    cudart.cudaSetDevice(0)
    test_boxes_overlap_and_iou()
    test_nms_normal_gpu()
    test_nms_gpu()
    benchmark_nms_gpu()
