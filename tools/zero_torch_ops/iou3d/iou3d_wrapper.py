"""
ctypes wrapper for libbevfusion_iou3d.so
No torch import required at runtime.
"""
import ctypes
import os

_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_iou3d.so")
_lib = ctypes.CDLL(_SO_PATH)

_lib.nms_gpu_cuda.argtypes = [
    ctypes.c_void_p,   # boxes_data (float*)
    ctypes.c_void_p,   # keep_data (int64_t*)
    ctypes.c_int,      # boxes_num
    ctypes.c_float,    # nms_overlap_thresh
    ctypes.c_int,      # device_id
]
_lib.nms_gpu_cuda.restype = ctypes.c_int

_lib.nms_normal_gpu_cuda.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
]
_lib.nms_normal_gpu_cuda.restype = ctypes.c_int

_lib.boxes_overlap_bev_gpu_cuda.argtypes = [
    ctypes.c_int,      # num_a
    ctypes.c_void_p,   # boxes_a (float*)
    ctypes.c_int,      # num_b
    ctypes.c_void_p,   # boxes_b (float*)
    ctypes.c_void_p,   # ans_overlap (float*)
]
_lib.boxes_overlap_bev_gpu_cuda.restype = ctypes.c_int

_lib.boxes_iou_bev_gpu_cuda.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_lib.boxes_iou_bev_gpu_cuda.restype = ctypes.c_int


def nms_gpu(boxes_ptr, keep_ptr, boxes_num, thresh, device_id=0):
    """
    Args:
        boxes_ptr: CUdeviceptr / int, float [N, 5] on GPU (x1,y1,x2,y2,ry)
        keep_ptr: host pointer (CPU), int64 [N]
        boxes_num: int
        thresh: float
        device_id: int
    Returns:
        num_to_keep: int
    """
    ret = _lib.nms_gpu_cuda(
        ctypes.c_void_p(boxes_ptr),
        ctypes.c_void_p(keep_ptr),
        boxes_num,
        ctypes.c_float(thresh),
        device_id,
    )
    if ret < 0:
        raise RuntimeError(f"nms_gpu_cuda failed (CUDA error code={ret})")
    return ret


def nms_normal_gpu(boxes_ptr, keep_ptr, boxes_num, thresh, device_id=0):
    ret = _lib.nms_normal_gpu_cuda(
        ctypes.c_void_p(boxes_ptr),
        ctypes.c_void_p(keep_ptr),
        boxes_num,
        ctypes.c_float(thresh),
        device_id,
    )
    if ret < 0:
        raise RuntimeError(f"nms_normal_gpu_cuda failed (CUDA error code={ret})")
    return ret


def boxes_overlap_bev_gpu(num_a, boxes_a_ptr, num_b, boxes_b_ptr, ans_overlap_ptr):
    """
    Args:
        num_a: int
        boxes_a_ptr: GPU pointer, float [num_a, 5]
        num_b: int
        boxes_b_ptr: GPU pointer, float [num_b, 5]
        ans_overlap_ptr: GPU pointer, float [num_a, num_b]
    Returns:
        0 on success, negative on CUDA error
    """
    ret = _lib.boxes_overlap_bev_gpu_cuda(
        num_a, ctypes.c_void_p(boxes_a_ptr),
        num_b, ctypes.c_void_p(boxes_b_ptr),
        ctypes.c_void_p(ans_overlap_ptr)
    )
    if ret < 0:
        raise RuntimeError(f"boxes_overlap_bev_gpu_cuda failed (CUDA error code={ret})")
    return ret


def boxes_iou_bev_gpu(num_a, boxes_a_ptr, num_b, boxes_b_ptr, ans_iou_ptr):
    ret = _lib.boxes_iou_bev_gpu_cuda(
        num_a, ctypes.c_void_p(boxes_a_ptr),
        num_b, ctypes.c_void_p(boxes_b_ptr),
        ctypes.c_void_p(ans_iou_ptr)
    )
    if ret < 0:
        raise RuntimeError(f"boxes_iou_bev_gpu_cuda failed (CUDA error code={ret})")
    return ret
