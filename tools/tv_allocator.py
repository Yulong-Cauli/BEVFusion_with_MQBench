"""
TVAllocator + TVSpconvMatmul: PyTorch-free memory allocator and matmul for spconv 2.3.

Uses cumm.tensorview (tv.Tensor) for GPU memory and cuBLAS via ctypes for GEMM.
Replaces spconv.pytorch.cppcore.TorchAllocator and TorchSpconvMatmul.
"""

import ctypes
import numpy as np
from cumm import tensorview as tv
from spconv.core_cc.csrc.sparse.alloc import ExternalAllocator
from spconv.core_cc.csrc.sparse.convops import ExternalSpconvMatmul
from spconv.constants import AllocKeys

# ============================================================================
# cuBLAS via ctypes
# ============================================================================

_cublas = ctypes.cdll.LoadLibrary("libcublas.so")

# cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
_cublas.cublasCreate_v2.restype = ctypes.c_int
_cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

# cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
_cublas.cublasDestroy_v2.restype = ctypes.c_int
_cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

# cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream)
_cublas.cublasSetStream_v2.restype = ctypes.c_int
_cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# cublasStatus_t cublasGemmEx(handle, transa, transb, m, n, k,
#   alpha, A, Atype, lda, B, Btype, ldb,
#   beta, C, Ctype, ldc, computeType, algo)
_cublas.cublasGemmEx.restype = ctypes.c_int
_cublas.cublasGemmEx.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # transa
    ctypes.c_int,     # transb
    ctypes.c_int,     # m
    ctypes.c_int,     # n
    ctypes.c_int,     # k
    ctypes.c_void_p,  # alpha
    ctypes.c_void_p,  # A
    ctypes.c_int,     # Atype (cudaDataType)
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # B
    ctypes.c_int,     # Btype
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # beta
    ctypes.c_void_p,  # C
    ctypes.c_int,     # Ctype
    ctypes.c_int,     # ldc
    ctypes.c_int,     # computeType (cublasComputeType_t)
    ctypes.c_int,     # algo (cublasGemmAlgo_t)
]

# cuBLAS constants
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUDA_R_16F = 2
CUDA_R_32F = 0
CUBLAS_COMPUTE_16F = 64
CUBLAS_COMPUTE_32F = 68
CUBLAS_COMPUTE_32F_FAST_16F = 74
CUBLAS_GEMM_DEFAULT = -1

# cublasAxpyEx for FP16 residual add (y = alpha*x + y)
_cublas.cublasAxpyEx.restype = ctypes.c_int
_cublas.cublasAxpyEx.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # n
    ctypes.c_void_p,  # alpha
    ctypes.c_int,     # alphaType
    ctypes.c_void_p,  # x
    ctypes.c_int,     # xType
    ctypes.c_int,     # incx
    ctypes.c_void_p,  # y
    ctypes.c_int,     # yType
    ctypes.c_int,     # incy
    ctypes.c_int,     # executionType
]

# cublasSaxpy_v2 for FP32 residual add
_cublas.cublasSaxpy_v2.restype = ctypes.c_int
_cublas.cublasSaxpy_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
]

_cublas_handle = ctypes.c_void_p()
_cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))


def cublas_gemm_fp16(A_ptr, B_ptr, C_ptr, M, N, K, stream_int=0):
    """C[M,N] = A[M,K] @ B[N,K].T  (all FP16, row-major)

    cuBLAS uses column-major. Row-major A[M,K] is col-major [K,M] with ld=K.
    C_row = A_row @ B_row^T  →  C_col[N,M] = B_col_view^T[N,K] @ A_col_view[K,M]
    = gemm(OP_T, OP_N, N, M, K, alpha, B, K, A, K, beta, C, N)
    """
    if stream_int:
        _cublas.cublasSetStream_v2(_cublas_handle, ctypes.c_void_p(stream_int))
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    status = _cublas.cublasGemmEx(
        _cublas_handle,
        ctypes.c_int(CUBLAS_OP_T),
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
        ctypes.byref(alpha),
        B_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(K),
        A_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(K),
        ctypes.byref(beta),
        C_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(N),
        ctypes.c_int(CUBLAS_COMPUTE_32F),
        ctypes.c_int(CUBLAS_GEMM_DEFAULT),
    )
    assert status == 0, f"cublasGemmEx failed with status {status}"


def cublas_gemm_fp32(A_ptr, B_ptr, C_ptr, M, N, K, stream_int=0):
    """C[M,N] = A[M,K] @ B[N,K].T  (all FP32, row-major)"""
    if stream_int:
        _cublas.cublasSetStream_v2(_cublas_handle, ctypes.c_void_p(stream_int))
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    status = _cublas.cublasGemmEx(
        _cublas_handle,
        ctypes.c_int(CUBLAS_OP_T),
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
        ctypes.byref(alpha),
        B_ptr, ctypes.c_int(CUDA_R_32F), ctypes.c_int(K),
        A_ptr, ctypes.c_int(CUDA_R_32F), ctypes.c_int(K),
        ctypes.byref(beta),
        C_ptr, ctypes.c_int(CUDA_R_32F), ctypes.c_int(N),
        ctypes.c_int(CUBLAS_COMPUTE_32F),
        ctypes.c_int(CUBLAS_GEMM_DEFAULT),
    )
    assert status == 0, f"cublasGemmEx failed with status {status}"


def cublas_gemm_fp16_nn(A_ptr, B_ptr, C_ptr, M, N, K, stream_int=0):
    """C[M,N] = A[M,K] @ B[K,N]  (all FP16, row-major, no transpose on B)

    cuBLAS column-major: C_col[N,M] = B_col[N,K] @ A_col[K,M]
    = gemm(OP_N, OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    """
    if stream_int:
        _cublas.cublasSetStream_v2(_cublas_handle, ctypes.c_void_p(stream_int))
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    status = _cublas.cublasGemmEx(
        _cublas_handle,
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
        ctypes.byref(alpha),
        B_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(N),
        A_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(K),
        ctypes.byref(beta),
        C_ptr, ctypes.c_int(CUDA_R_16F), ctypes.c_int(N),
        ctypes.c_int(CUBLAS_COMPUTE_32F),
        ctypes.c_int(CUBLAS_GEMM_DEFAULT),
    )
    assert status == 0, f"cublasGemmEx failed with status {status}"


def cublas_axpy_fp16(y_ptr, x_ptr, n, stream_int=0):
    """y += x (both FP16). Uses cublasAxpyEx with FP32 execution type."""
    if stream_int:
        _cublas.cublasSetStream_v2(_cublas_handle, ctypes.c_void_p(stream_int))
    alpha = ctypes.c_float(1.0)
    status = _cublas.cublasAxpyEx(
        _cublas_handle, ctypes.c_int(n),
        ctypes.byref(alpha), CUDA_R_32F,
        x_ptr, CUDA_R_16F, ctypes.c_int(1),
        y_ptr, CUDA_R_16F, ctypes.c_int(1),
        CUDA_R_32F,
    )
    assert status == 0, f"cublasAxpyEx FP16 failed: {status}"


def cublas_axpy_fp32(y_ptr, x_ptr, n, stream_int=0):
    """y += x (both FP32). Uses cublasSaxpy_v2."""
    if stream_int:
        _cublas.cublasSetStream_v2(_cublas_handle, ctypes.c_void_p(stream_int))
    alpha = ctypes.c_float(1.0)
    status = _cublas.cublasSaxpy_v2(
        _cublas_handle, ctypes.c_int(n),
        ctypes.byref(alpha),
        x_ptr, ctypes.c_int(1),
        y_ptr, ctypes.c_int(1),
    )
    assert status == 0, f"cublasSaxpy_v2 failed: {status}"


# ============================================================================
# TVAllocator
# ============================================================================

class TVAllocator(ExternalAllocator):
    """GPU memory allocator using torch.empty + tv.from_blob.

    Uses PyTorch's CUDA caching allocator for GPU memory (compatible with TRT
    execution contexts), then wraps as tv.Tensor for spconv C++ API.
    Mirrors TorchAllocator from spconv.pytorch.cppcore but stores tv.Tensor
    in the public interface.
    """

    # tv dtype → torch dtype mapping (PyTorch lacks uint32/uint16/uint64)
    _TV_TO_TORCH = {
        tv.float32: 'torch.float32', tv.float16: 'torch.float16',
        tv.float64: 'torch.float64',
        tv.int32: 'torch.int32', tv.int64: 'torch.int64',
        tv.int16: 'torch.int16', tv.int8: 'torch.int8',
        tv.uint8: 'torch.uint8',
        tv.uint32: 'torch.int32', tv.uint16: 'torch.int16',
        tv.uint64: 'torch.int64',
    }

    def __init__(self, device=0):
        super().__init__()
        self.device = device
        self.allocated = {}
        self._torch_refs = {}  # keep torch.Tensor alive to prevent GC
        import torch as _torch
        self._torch = _torch
        self._gpu = _torch.device(f'cuda:{device}')
        self._cpu = _torch.device('cpu')
        # Build dtype map using actual torch dtype objects
        self._dtype_map = {
            tv.float32: _torch.float32, tv.float16: _torch.float16,
            tv.float64: _torch.float64,
            tv.int32: _torch.int32, tv.int64: _torch.int64,
            tv.int16: _torch.int16, tv.int8: _torch.int8,
            tv.uint8: _torch.uint8,
            tv.uint32: _torch.int32, tv.uint16: _torch.int16,
            tv.uint64: _torch.int64,
        }

    def _torch_to_tv(self, th_ten, tv_dtype):
        """Wrap a torch.Tensor as tv.Tensor with correct dtype."""
        ptr = th_ten.data_ptr()
        dev = 0 if th_ten.is_cuda else -1
        return tv.from_blob(ptr, list(th_ten.shape), tv_dtype, dev)

    def zeros(self, name, shape, dtype, device, stream=0,
              is_temp_memory=False, scale=1.0):
        th_dtype = self._dtype_map[dtype]
        dev = self._cpu if device == -1 else self._gpu
        th_ten = self._torch.empty(shape, dtype=th_dtype, device=dev).zero_()
        ten = self._torch_to_tv(th_ten, dtype)
        self._torch_refs[ten.byte_pointer()] = th_ten
        self.allocated[ten.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten

    def empty(self, name, shape, dtype, device, stream=0,
              is_temp_memory=False, scale=1.0):
        th_dtype = self._dtype_map[dtype]
        dev = self._cpu if device == -1 else self._gpu
        th_ten = self._torch.empty(shape, dtype=th_dtype, device=dev)
        ten = self._torch_to_tv(th_ten, dtype)
        self._torch_refs[ten.byte_pointer()] = th_ten
        self.allocated[ten.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten

    def full_int(self, name, shape, value, dtype, device, stream=0,
                 is_temp_memory=False):
        th_dtype = self._dtype_map[dtype]
        dev = self._cpu if device == -1 else self._gpu
        th_ten = self._torch.full(shape, value, dtype=th_dtype, device=dev)
        ten = self._torch_to_tv(th_ten, dtype)
        self._torch_refs[ten.byte_pointer()] = th_ten
        self.allocated[ten.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten

    def full_float(self, name, shape, value, dtype, device, stream=0,
                   is_temp_memory=False):
        th_dtype = self._dtype_map[dtype]
        dev = self._cpu if device == -1 else self._gpu
        th_ten = self._torch.full(shape, value, dtype=th_dtype, device=dev)
        ten = self._torch_to_tv(th_ten, dtype)
        self._torch_refs[ten.byte_pointer()] = th_ten
        self.allocated[ten.byte_pointer()] = ten
        if name and not is_temp_memory:
            self.allocated[name] = ten
        return ten

    def get_tensor_by_name(self, name):
        return self.allocated[name]

    def free(self, ten):
        bp = ten.byte_pointer()
        if bp in self.allocated:
            self.allocated.pop(bp)
        self._torch_refs.pop(bp, None)

    def free_noexcept(self, ten):
        bp = ten.byte_pointer()
        self.allocated.pop(bp, None)
        self._torch_refs.pop(bp, None)


# ============================================================================
# TVSpconvMatmul
# ============================================================================

class TVSpconvMatmul(ExternalSpconvMatmul):
    """SubM conv init GEMM using cuBLAS. No PyTorch dependency.

    Replaces TorchSpconvMatmul which uses torch.mm().
    Only forward (inference) is supported.
    """

    def __init__(self, alloc):
        super().__init__()
        self.alloc = alloc

    def indice_conv_init_gemm(self, features_n, filters_n,
                              all_weight_is_krsc, is_kc_not_ck,
                              kv_center, out_channel, stream_int=0):
        """out = features @ filters[center].T

        spconv 2.3 uses KRSC layout (all_weight_is_krsc=True):
          filters shape: [out_ch, k, k, k, in_ch]
          reshaped: [out_ch, KV, in_ch]
          center slice: [out_ch, in_ch]
        """
        features = self.alloc.allocated[features_n]
        filters = self.alloc.allocated[filters_n]

        M = features.dim(0)
        K = features.dim(1)
        N = out_channel

        if all_weight_is_krsc:
            # [out_ch, KV, in_ch] → select center → [out_ch, in_ch]
            filters_r = filters.view([out_channel, -1, filters.dim(-1)])
            filter_center = filters_r[:, kv_center]  # [N, K], non-contiguous
            if not filter_center.is_contiguous():
                # Copy strided slice to contiguous buffer via CPU
                fc_np = filter_center.cpu().numpy().copy()
                filter_center = tv.from_numpy(fc_np).cuda()
        else:
            filters_r = filters.view([-1, filters.dim(-2), filters.dim(-1)])
            filter_center = filters_r[kv_center]
            if not filter_center.is_contiguous():
                fc_np = filter_center.cpu().numpy().copy()
                filter_center = tv.from_numpy(fc_np).cuda()

        out = tv.zeros([M, N], features.dtype, device=features.device)

        A_ptr = ctypes.c_void_p(features.byte_pointer())
        B_ptr = ctypes.c_void_p(filter_center.byte_pointer())
        C_ptr = ctypes.c_void_p(out.byte_pointer())

        if features.dtype == tv.float16:
            if all_weight_is_krsc:
                # C[M,N] = A[M,K] @ B[N,K].T
                cublas_gemm_fp16(A_ptr, B_ptr, C_ptr, M, N, K, stream_int)
            elif not is_kc_not_ck:
                # filter_center is [K, N], need A @ B (no transpose)
                # Use: C^T[N,M] = B^T[N,K] @ A^T[K,M] → col-major trick
                # Actually simpler: just call with swapped args
                cublas_gemm_fp16_nn(A_ptr, B_ptr, C_ptr, M, N, K, stream_int)
            else:
                cublas_gemm_fp16(A_ptr, B_ptr, C_ptr, M, N, K, stream_int)
        else:
            if all_weight_is_krsc:
                cublas_gemm_fp32(A_ptr, B_ptr, C_ptr, M, N, K, stream_int)
            else:
                cublas_gemm_fp32(A_ptr, B_ptr, C_ptr, M, N, K, stream_int)

        self.alloc.allocated[AllocKeys.OutFeatures] = out
        self.alloc.allocated[out.byte_pointer()] = out
        return out

    def indice_conv_cpu_gemm(self, inp_buffer_n, out_buffer_n, filters_n,
                             all_weight_is_krsc, is_kc_not_ck, nhot, index):
        """CPU GEMM fallback — not needed for GPU inference."""
        raise NotImplementedError("CPU GEMM not supported in TVSpconvMatmul")

    def indice_conv_bwd_init_gemm(self, features_n, filters_n, out_bp_n,
                                  dfilters_n, all_weight_is_krsc, is_kc_not_ck,
                                  kv_center, stream_int=0):
        raise NotImplementedError("Backward not supported in TVSpconvMatmul")

    def indice_conv_bwd_cpu_gemm(self, inp_buffer_n, out_buffer_n, filters_n,
                                 dfilters_n, all_weight_is_krsc, is_kc_not_ck,
                                 nhot, index):
        raise NotImplementedError("Backward not supported in TVSpconvMatmul")
