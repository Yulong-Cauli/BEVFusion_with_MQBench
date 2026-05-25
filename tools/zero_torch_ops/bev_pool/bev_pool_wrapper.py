"""
ctypes wrapper for libbevfusion_bev_pool.so
No torch import required at runtime.
"""
import ctypes
import os

_SO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libbevfusion_bev_pool.so")
_lib = ctypes.CDLL(_SO_PATH)

_lib.bev_pool_forward_cuda.argtypes = [
    ctypes.c_int,  # b
    ctypes.c_int,  # d
    ctypes.c_int,  # h
    ctypes.c_int,  # w
    ctypes.c_int,  # n
    ctypes.c_int,  # c
    ctypes.c_int,  # n_intervals
    ctypes.c_void_p,  # x (float*)
    ctypes.c_void_p,  # geom_feats (int*)
    ctypes.c_void_p,  # interval_starts (int*)
    ctypes.c_void_p,  # interval_lengths (int*)
    ctypes.c_void_p,  # out (float*)
]
_lib.bev_pool_forward_cuda.restype = ctypes.c_int

_lib.bev_pool_forward_cuda_half.argtypes = [
    ctypes.c_int,  # b
    ctypes.c_int,  # d
    ctypes.c_int,  # h
    ctypes.c_int,  # w
    ctypes.c_int,  # n
    ctypes.c_int,  # c
    ctypes.c_int,  # n_intervals
    ctypes.c_void_p,  # x (__half*)
    ctypes.c_void_p,  # geom_feats (int*)
    ctypes.c_void_p,  # interval_starts (int*)
    ctypes.c_void_p,  # interval_lengths (int*)
    ctypes.c_void_p,  # out (float*)
]
_lib.bev_pool_forward_cuda_half.restype = ctypes.c_int


def bev_pool_forward(x_ptr, geom_feats_ptr, interval_starts_ptr,
                     interval_lengths_ptr, out_ptr,
                     b, d, h, w, n, c, n_intervals):
    """
    Args:
        x_ptr: CUdeviceptr / int, float [n, c]
        geom_feats_ptr: CUdeviceptr / int, int [n, 4]
        interval_starts_ptr: CUdeviceptr / int, int [n_intervals]
        interval_lengths_ptr: CUdeviceptr / int, int [n_intervals]
        out_ptr: CUdeviceptr / int, float [b, d, h, w, c]
        b,d,h,w,n,c,n_intervals: ints
    Returns:
        0 on success, negative on CUDA error.
    """
    ret = _lib.bev_pool_forward_cuda(
        b, d, h, w, n, c, n_intervals,
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(geom_feats_ptr),
        ctypes.c_void_p(interval_starts_ptr),
        ctypes.c_void_p(interval_lengths_ptr),
        ctypes.c_void_p(out_ptr),
    )
    if ret != 0:
        raise RuntimeError("bev_pool_forward_cuda failed (CUDA error)")
    return ret


def bev_pool_forward_half(x_ptr, geom_feats_ptr, interval_starts_ptr,
                          interval_lengths_ptr, out_ptr,
                          b, d, h, w, n, c, n_intervals):
    """
    Half-precision input variant.
    Args:
        x_ptr: CUdeviceptr / int, __half [n, c]
        (other args same as bev_pool_forward)
    Returns:
        0 on success.
    """
    ret = _lib.bev_pool_forward_cuda_half(
        b, d, h, w, n, c, n_intervals,
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(geom_feats_ptr),
        ctypes.c_void_p(interval_starts_ptr),
        ctypes.c_void_p(interval_lengths_ptr),
        ctypes.c_void_p(out_ptr),
    )
    if ret != 0:
        raise RuntimeError("bev_pool_forward_cuda_half failed (CUDA error)")
    return ret
