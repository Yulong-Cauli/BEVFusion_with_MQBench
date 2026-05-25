"""
TVSparseEncoder: LiDAR sparse convolution backbone.

Uses cumm.tensorview (tv.Tensor) + spconv core_cc API directly.
No torch.nn, no autograd in the forward path.
GPU memory allocated via PyTorch CUDA caching allocator (required for
TRT execution context coexistence), wrapped as tv.Tensor.

Reference: temp/spconv/example/libspconv/main.cu
"""

import ctypes
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
from cumm import tensorview as tv
from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.core_cc.csrc.sparse.convops.spops import ConvGemmOps
from spconv.core_cc.csrc.sparse.inference import InferenceOps
from spconv.constants import AllocKeys
import spconv.algo as spconv_algo

from tools.tv_allocator import TVAllocator, TVSpconvMatmul, cublas_axpy_fp16, cublas_axpy_fp32

_cudart = ctypes.cdll.LoadLibrary("libcudart.so")

# Load Log2 fake quant CUDA kernel
_libtv_log2 = ctypes.cdll.LoadLibrary(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtv_log2_quant.so")
)
_libtv_log2.tv_log2_fake_quant_fp16.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p]
_libtv_log2.tv_log2_fake_quant_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p]

_libtv_log2.tv_bn_forward_inplace_fp16.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libtv_log2.tv_bn_forward_inplace_fp32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

def log2_fake_quant_inplace(features: tv.Tensor, log2_base: float, stream: int = 0, eps: float = 1e-6):
    """Log2 fake quantization on GPU (inplace).

    Matches SparseLog2FakeQuantize forward (inference only):
      x_dq = sign(x) * 2^round(log2(|x|) - base).clamp(-127,127) + base
      zero if |x| < eps
    """
    n = features.dim(0) * features.dim(1)
    ptr = ctypes.c_void_p(features.byte_pointer())
    if features.dtype == tv.float16:
        _libtv_log2.tv_log2_fake_quant_fp16(ptr, n, log2_base, eps, ctypes.c_void_p(stream))
    elif features.dtype == tv.float32:
        _libtv_log2.tv_log2_fake_quant_fp32(ptr, n, log2_base, eps, ctypes.c_void_p(stream))
    else:
        raise TypeError(f"log2_fake_quant_inplace unsupported dtype {features.dtype}")

def bn_forward_inplace(features: tv.Tensor, scale_tv: tv.Tensor, shift_tv: tv.Tensor, stream: int = 0):
    """BN forward on GPU (inplace): x = x * scale + shift (per-channel)."""
    n = features.dim(0) * features.dim(1)
    c = features.dim(1)
    data_ptr = ctypes.c_void_p(features.byte_pointer())
    scale_ptr = ctypes.c_void_p(scale_tv.byte_pointer())
    shift_ptr = ctypes.c_void_p(shift_tv.byte_pointer())
    if features.dtype == tv.float16:
        _libtv_log2.tv_bn_forward_inplace_fp16(data_ptr, scale_ptr, shift_ptr, n, c, ctypes.c_void_p(stream))
    elif features.dtype == tv.float32:
        _libtv_log2.tv_bn_forward_inplace_fp32(data_ptr, scale_ptr, shift_ptr, n, c, ctypes.c_void_p(stream))
    else:
        raise TypeError(f"bn_forward_inplace unsupported dtype {features.dtype}")

# Module-level list to keep torch.Tensor refs alive during forward pass.
# Cleared at the start of each forward() call.
_torch_refs = []

# tv dtype → torch dtype
_TV_TO_TORCH = {
    tv.float32: torch.float32, tv.float16: torch.float16,
    tv.int32: torch.int32, tv.int64: torch.int64,
    tv.uint8: torch.uint8, tv.int8: torch.int8,
}

def _np_to_tv_cuda(arr: np.ndarray, tv_dtype=None) -> tv.Tensor:
    """Create a PyTorch-backed tv.Tensor on GPU from numpy array.

    Uses torch.from_numpy().cuda() for memory allocation (PyTorch CUDA
    caching allocator), then wraps as tv.Tensor via tv.from_blob.
    This ensures compatibility with TRT execution contexts.
    """
    th = torch.from_numpy(arr).cuda()
    if tv_dtype is None:
        _NP_TO_TV = {
            np.float32: tv.float32, np.float16: tv.float16,
            np.int32: tv.int32, np.int64: tv.int64,
        }
        tv_dtype = _NP_TO_TV.get(arr.dtype.type, tv.float32)
    return tv.from_blob(th.data_ptr(), list(th.shape), tv_dtype, 0), th

def _tv_zeros_cuda(shape, tv_dtype, th_ref_list=None) -> tv.Tensor:
    """Create a zeroed PyTorch-backed tv.Tensor on GPU."""
    th_dtype = _TV_TO_TORCH[tv_dtype]
    th = torch.zeros(shape, dtype=th_dtype, device='cuda:0')
    ten = tv.from_blob(th.data_ptr(), list(th.shape), tv_dtype, 0)
    if th_ref_list is not None:
        th_ref_list.append(th)
    return ten, th

def _tv_empty_cuda(shape, tv_dtype, th_ref_list=None) -> tv.Tensor:
    """Create an uninitialized PyTorch-backed tv.Tensor on GPU."""
    th_dtype = _TV_TO_TORCH[tv_dtype]
    th = torch.empty(shape, dtype=th_dtype, device='cuda:0')
    ten = tv.from_blob(th.data_ptr(), list(th.shape), tv_dtype, 0)
    if th_ref_list is not None:
        th_ref_list.append(th)
    return ten, th


# ============================================================================
# CUDA arch detection (no PyTorch)
# ============================================================================

def get_cuda_arch(device=0):
    major = ctypes.c_int()
    minor = ctypes.c_int()
    _cudart.cudaDeviceGetAttribute(ctypes.byref(major), 75, device)
    _cudart.cudaDeviceGetAttribute(ctypes.byref(minor), 76, device)
    return (major.value, minor.value)


# ============================================================================
# Conv output size (copied from spconv.pytorch.ops, no torch dependency)
# ============================================================================

def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    output_size = []
    for i in range(len(input_size)):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        output_size.append(size)
    return output_size


# ============================================================================
# TVSparseConvTensor — lightweight replacement for spconv.SparseConvTensor
# ============================================================================

@dataclass
class TVSparseConvTensor:
    features: tv.Tensor          # [N, C] GPU fp16/fp32
    indices: tv.Tensor           # [N, ndim+1] int32 GPU
    spatial_shape: List[int]
    batch_size: int
    indice_dict: dict = field(default_factory=dict)
    _allocators: list = field(default_factory=list)  # keep allocators alive

    def replace_feature(self, new_features):
        ret = TVSparseConvTensor(
            new_features, self.indices, self.spatial_shape,
            self.batch_size, self.indice_dict, self._allocators)
        return ret


# ============================================================================
# Sparse conv forward (single layer) — follows main.cu implicit gemm path
# ============================================================================

def sparse_conv_forward(
    inp: TVSparseConvTensor,
    weight: tv.Tensor,       # [out_ch, *ksize, in_ch] KRSC layout
    ksize: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    subm: bool,
    stream: int,
    arch: Tuple[int, int],
    conv_tuner,
    indice_key: Optional[str] = None,
) -> TVSparseConvTensor:
    """Single sparse conv forward without PyTorch.

    Follows spconv.pytorch.ops.py dynamic allocator path exactly:
    - TVAllocator (dynamic) for get_indice_pairs_implicit_gemm
    - Read back tensors from alloc.allocated (no tight-pack slicing needed
      because dynamic allocator allocates exact sizes)
    - Use mask_tensor returned by get_indice_pairs_implicit_gemm
    - auto_fp32_accum=True for implicit_gemm (matches Python default)
    """
    ndim = len(ksize)
    KV = 1
    for k in ksize:
        KV *= k

    features = inp.features
    indices = inp.indices
    spatial_shape = inp.spatial_shape
    batch_size = inp.batch_size
    num_act_in = features.dim(0)
    out_channels = weight.dim(0)

    # Check indice cache
    if indice_key and indice_key in inp.indice_dict:
        cached = inp.indice_dict[indice_key]
        pair_fwd = cached["pair_fwd"]
        pair_mask_splits = cached["pair_mask_splits"]
        mask_argsort_splits = cached["mask_argsort_splits"]
        mask_tv = cached["mask_tv"]
        num_act_out = cached["num_act_out"]
        out_inds = cached["out_inds"]
        out_spatial_shape = cached["out_spatial_shape"]
    else:
        if subm:
            out_spatial_shape = spatial_shape
        else:
            out_spatial_shape = get_conv_output_size(
                spatial_shape, ksize, stride, padding, dilation)

        # Dynamic allocator — matches spconv.pytorch.ops.py path
        alloc = TVAllocator(device=0)
        conv_algo_val = 1  # kMaskImplicitGemm
        use_direct_table = False  # spconv Python default; keep consistent

        # get_indice_pairs_implicit_gemm returns (mask_tensor, num_act_out)
        # mask_tensor is the proper mask from C++, use it directly
        mask_tensor_from_cpp, num_act_out = SpconvOps.get_indice_pairs_implicit_gemm(
            alloc, indices, batch_size, spatial_shape,
            conv_algo_val, ksize, stride, padding, dilation,
            [0] * ndim,
            subm, False, False,
            stream, -1,
            tv.CUDAKernelTimer(False),
            use_direct_table, True)

        mask_split_cnt = mask_tensor_from_cpp.dim(0)

        # Use mask from C++ return value (matches ops.py line 389/1503-1504)
        # ops.py: masks = [mask_tensor[i:i+1].numpy() for i in range(mask_split_count)]
        # ops.py: mask = np.concatenate(masks); mask_tv = tv.from_numpy(mask).clone()
        masks_np = [mask_tensor_from_cpp[i:i+1].numpy() for i in range(mask_split_cnt)]
        mask_np = np.concatenate(masks_np)
        mask_tv = tv.from_numpy(mask_np).clone()

        # Read back C++-allocated tensors from alloc.allocated
        # Dynamic allocator allocates exact sizes — NO tight-pack slicing needed
        # (matches ops.py lines 393-440)
        if subm:
            out_inds = indices
            pair = alloc.allocated[AllocKeys.PairFwd]
            # SubM: PairFwd shape [1, KV, N] → take [0] → [KV, N]
            pair_fwd = pair[0] if pair.ndim == 3 else pair
        else:
            out_inds = alloc.allocated[AllocKeys.OutIndices]
            pair_fwd = alloc.allocated[AllocKeys.PairFwd]

        pair_mask = alloc.allocated[AllocKeys.PairMask]
        mask_argsort = alloc.allocated[AllocKeys.MaskArgSort]

        # Split by mask_split_cnt — use [i] indexing (matches ops.py)
        pair_mask_splits = [pair_mask[i] for i in range(mask_split_cnt)]
        mask_argsort_splits = [mask_argsort[i] for i in range(mask_split_cnt)]

        # Cache for reuse (keep alloc alive to prevent GC)
        if indice_key:
            inp.indice_dict[indice_key] = {
                "pair_fwd": pair_fwd,
                "pair_mask_splits": pair_mask_splits,
                "mask_argsort_splits": mask_argsort_splits,
                "mask_tv": mask_tv,
                "num_act_out": num_act_out,
                "out_inds": out_inds,
                "out_spatial_shape": out_spatial_shape,
                "_alloc": alloc,
            }

    # Run implicit GEMM — matches ops.py SPCONV_CPP_GEMM path exactly.
    # Let C++ allocate output via allocator callback; read back from alloc.allocated.
    alloc2 = TVAllocator(device=0)

    ConvGemmOps.implicit_gemm(
        alloc2, conv_tuner, features, weight, pair_fwd,
        pair_mask_splits, mask_argsort_splits, num_act_out,
        mask_tv, arch, False, subm, stream,
        tv.CUDAKernelTimer(False),
        True, False,  # auto_fp32_accum=True, fp32_accum=False (matches ops.py default)
        tv.Tensor(), 0.0, 0.0,
        tv.gemm.Activation.None_)

    out_features = alloc2.allocated[AllocKeys.OutFeatures]

    out = TVSparseConvTensor(
        out_features, out_inds, out_spatial_shape,
        batch_size, inp.indice_dict)
    # Pin allocators so their underlying torch tensors aren't GC'd while
    # out_features / pair_fwd etc are still in use by subsequent kernels.
    if not (indice_key and indice_key in inp.indice_dict):
        out._allocators.append(alloc)
    out._allocators.append(alloc2)
    return out


# ============================================================================
# ReLU inplace (via InferenceOps)
# ============================================================================

def relu_inplace(features: tv.Tensor, stream: int = 0):
    """ReLU via spconv InferenceOps CUDA kernel."""
    InferenceOps.activation_inplace(
        features, tv.gemm.Activation.ReLU, 0.0, 0.0, stream)


# ============================================================================
# scatter_nd: sparse → dense (no PyTorch)
# ============================================================================

def scatter_nd_numpy(indices_np, features_np, shape):
    """Scatter sparse features into dense tensor via numpy."""
    ret = np.zeros(shape, dtype=features_np.dtype)
    ndim = indices_np.shape[-1]
    slices = tuple(indices_np[:, i] for i in range(ndim))
    ret[slices] = features_np
    return ret


# ============================================================================
# TVSparseEncoder — full BEVFusion LiDAR backbone without PyTorch
# ============================================================================

class TVSparseEncoder:
    """SparseEncoder using tv.Tensor + core_cc API. Zero PyTorch in forward."""

    def __init__(self, arch=(8, 6), stream=0):
        self.arch = arch
        self.stream = stream
        self.conv_tuner = spconv_algo.CONV_CPP
        self.sparse_shape = [1440, 1440, 41]

        # Layer params: {conv_name: {"weight": tv.Tensor, "bias": tv.Tensor|None}}
        # BN is fused into conv weight+bias at load time.
        self.layer_params = {}

        # Network structure definition
        # (layer_name, conv_name, ksize, stride, padding, subm, in_ch, out_ch, indice_key)
        self.conv_layers = [
            # conv_input
            ("conv_input", "conv_input.0", [3,3,3], [1,1,1], [1,1,1], True, 5, 16, "subm1"),
            # encoder_layer1: block0, block1, downsample
            ("encoder_layer1.0.conv1", "encoder_layer1.0.conv1", [3,3,3], [1,1,1], [1,1,1], True, 16, 16, None),
            ("encoder_layer1.0.conv2", "encoder_layer1.0.conv2", [3,3,3], [1,1,1], [1,1,1], True, 16, 16, None),
            ("encoder_layer1.1.conv1", "encoder_layer1.1.conv1", [3,3,3], [1,1,1], [1,1,1], True, 16, 16, None),
            ("encoder_layer1.1.conv2", "encoder_layer1.1.conv2", [3,3,3], [1,1,1], [1,1,1], True, 16, 16, None),
            ("encoder_layer1.2.0", "encoder_layer1.2.0", [3,3,3], [2,2,2], [1,1,1], False, 16, 32, "spconv1"),
            # encoder_layer2
            ("encoder_layer2.0.conv1", "encoder_layer2.0.conv1", [3,3,3], [1,1,1], [1,1,1], True, 32, 32, None),
            ("encoder_layer2.0.conv2", "encoder_layer2.0.conv2", [3,3,3], [1,1,1], [1,1,1], True, 32, 32, None),
            ("encoder_layer2.1.conv1", "encoder_layer2.1.conv1", [3,3,3], [1,1,1], [1,1,1], True, 32, 32, None),
            ("encoder_layer2.1.conv2", "encoder_layer2.1.conv2", [3,3,3], [1,1,1], [1,1,1], True, 32, 32, None),
            ("encoder_layer2.2.0", "encoder_layer2.2.0", [3,3,3], [2,2,2], [1,1,1], False, 32, 64, "spconv2"),
            # encoder_layer3
            ("encoder_layer3.0.conv1", "encoder_layer3.0.conv1", [3,3,3], [1,1,1], [1,1,1], True, 64, 64, None),
            ("encoder_layer3.0.conv2", "encoder_layer3.0.conv2", [3,3,3], [1,1,1], [1,1,1], True, 64, 64, None),
            ("encoder_layer3.1.conv1", "encoder_layer3.1.conv1", [3,3,3], [1,1,1], [1,1,1], True, 64, 64, None),
            ("encoder_layer3.1.conv2", "encoder_layer3.1.conv2", [3,3,3], [1,1,1], [1,1,1], True, 64, 64, None),
            ("encoder_layer3.2.0", "encoder_layer3.2.0", [3,3,3], [2,2,2], [1,1,0], False, 64, 128, "spconv3"),
            # encoder_layer4
            ("encoder_layer4.0.conv1", "encoder_layer4.0.conv1", [3,3,3], [1,1,1], [1,1,1], True, 128, 128, None),
            ("encoder_layer4.0.conv2", "encoder_layer4.0.conv2", [3,3,3], [1,1,1], [1,1,1], True, 128, 128, None),
            ("encoder_layer4.1.conv1", "encoder_layer4.1.conv1", [3,3,3], [1,1,1], [1,1,1], True, 128, 128, None),
            ("encoder_layer4.1.conv2", "encoder_layer4.1.conv2", [3,3,3], [1,1,1], [1,1,1], True, 128, 128, None),
            # conv_out
            ("conv_out", "conv_out.0", [1,1,3], [1,1,2], [0,0,0], False, 128, 128, "spconv_down2"),
        ]

        # BN layers follow each conv (same naming but with .1 suffix for conv_input/encoder_layerX.2)
        # Residual connections in BasicBlocks

    def load_weights(self, ckpt_path, dtype=tv.float16):
        """Load weights from BEVFusion checkpoint into tv.Tensor.

        BN is fused into conv weights and bias at load time to eliminate
        per-layer GPU↔CPU roundtrips during inference.

        All GPU tensors are allocated via PyTorch CUDA caching allocator
        (torch.from_numpy().cuda() + tv.from_blob) for TRT compatibility.
        """
        import torch  # Only used for loading checkpoint, not in forward
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        prefix = "encoders.lidar.backbone."

        self._torch_weight_refs = []  # keep torch.Tensor alive

        for layer_name, conv_name, ksize, stride, padding, subm, in_ch, out_ch, ikey in self.conv_layers:
            # Checkpoint key: encoder layers have extra "encoder_layers." prefix
            if conv_name.startswith("encoder_layer"):
                ckpt_conv_name = f"encoder_layers.{conv_name}"
            else:
                ckpt_conv_name = conv_name

            # Conv weight
            src_key = f"{prefix}{ckpt_conv_name}.weight"
            if src_key not in state_dict:
                continue

            w = state_dict[src_key].numpy()
            # spconv 2.1 [k,k,k,in,out] → spconv 2.3 [out,k,k,k,in]
            w = np.ascontiguousarray(np.transpose(w, (4, 0, 1, 2, 3)))

            # BN params — determine BN key
            if ckpt_conv_name.endswith(".0") and (".2.0" in ckpt_conv_name
                                             or "conv_input" in ckpt_conv_name or "conv_out" in ckpt_conv_name):
                bn_name = ckpt_conv_name[:-1] + "1"
            else:
                bn_name = ckpt_conv_name.replace("conv", "bn")

            bn_src = f"{prefix}{bn_name}"
            bn_keys = ["weight", "bias", "running_mean", "running_var"]
            has_bn = all(f"{bn_src}.{k}" in state_dict for k in bn_keys)

            fused_w = w
            fused_b = None
            if has_bn:
                gamma, beta, running_mean, running_var = (
                    state_dict[f"{bn_src}.{k}"].numpy().astype(np.float32)
                    for k in bn_keys
                )
                eps = 1e-3
                scale = gamma / np.sqrt(running_var + eps)
                shift = beta - running_mean * scale
                # Fuse scale into weight [out_ch, 1, 1, 1, 1]
                fused_w = w * scale.reshape(-1, 1, 1, 1, 1)
                fused_b = shift

            if dtype == tv.float16:
                fused_w = fused_w.astype(np.float16)
                if fused_b is not None:
                    fused_b = fused_b.astype(np.float16)

            w_tv, w_th = _np_to_tv_cuda(fused_w, dtype)
            self._torch_weight_refs.append(w_th)

            bias_tv = tv.Tensor()
            if fused_b is not None:
                bias_tv, b_th = _np_to_tv_cuda(fused_b, dtype)
                self._torch_weight_refs.append(b_th)

            self.layer_params[conv_name] = {
                "weight": w_tv,
                "bias": bias_tv if has_bn else None,
            }

        print(f"  Loaded {len(self.layer_params)} fused conv layers")

    def load_ptq_weights(self, ptq_ckpt_path, dtype=tv.float16):
        """Load PTQ INT8 weights with Log2 activation quantization.

        Weight fake quantization (symmetric INT8) is applied at load time:
          w_dq = round(w / scale).clamp(-127, 127) * scale

        For INT8 path, BN is kept UNFUSED (separate GPU kernel) to exactly
        match the PyTorch FakeQuant computation graph.
        Per-layer log2_base is stored for runtime Log2 fake quantization.
        """
        import torch
        ckpt = torch.load(ptq_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        prefix = "encoders.lidar.backbone."

        self._torch_weight_refs = []

        for layer_name, conv_name, ksize, stride, padding, subm, in_ch, out_ch, ikey in self.conv_layers:
            if conv_name.startswith("encoder_layer"):
                ckpt_conv_name = f"encoder_layers.{conv_name}"
            else:
                ckpt_conv_name = conv_name

            src_key = f"{prefix}{ckpt_conv_name}.conv.weight"
            if src_key not in state_dict:
                src_key = f"{prefix}{ckpt_conv_name}.weight"
            if src_key not in state_dict:
                continue

            w = state_dict[src_key].numpy()
            w = np.ascontiguousarray(np.transpose(w, (4, 0, 1, 2, 3)))

            # Weight fake quantization scale (per-channel symmetric)
            wfq_scale = state_dict.get(
                f"{prefix}{ckpt_conv_name}.weight_fake_quant.scale")
            if wfq_scale is not None:
                s = wfq_scale.numpy().astype(np.float32).reshape(-1, 1, 1, 1, 1)
                w = np.clip(np.round(w.astype(np.float32) / s), -127, 127) * s

            # BN fuse
            if ckpt_conv_name.endswith(".0") and (".2.0" in ckpt_conv_name
                                             or "conv_input" in ckpt_conv_name or "conv_out" in ckpt_conv_name):
                bn_name = ckpt_conv_name[:-1] + "1"
            else:
                bn_name = ckpt_conv_name.replace("conv", "bn")
            bn_src = f"{prefix}{bn_name}"
            bn_keys = ["weight", "bias", "running_mean", "running_var"]
            has_bn = all(f"{bn_src}.{k}" in state_dict for k in bn_keys)

            if dtype == tv.float16:
                w = w.astype(np.float16)

            w_tv, w_th = _np_to_tv_cuda(w, dtype)
            self._torch_weight_refs.append(w_th)

            # BN params — kept separate for INT8 path
            bn_scale_tv = tv.Tensor()
            bn_shift_tv = tv.Tensor()
            if has_bn:
                gamma, beta, running_mean, running_var = (
                    state_dict[f"{bn_src}.{k}"].numpy().astype(np.float32)
                    for k in bn_keys
                )
                eps = 1e-3
                scale_np = (gamma / np.sqrt(running_var + eps)).astype(np.float32)
                shift_np = (beta - running_mean * scale_np).astype(np.float32)
                bn_scale_tv, s_th = _np_to_tv_cuda(scale_np, tv.float32)
                bn_shift_tv, sh_th = _np_to_tv_cuda(shift_np, tv.float32)
                self._torch_weight_refs.extend([s_th, sh_th])

            # Log2 activation quantization base
            log2_base = state_dict.get(
                f"{prefix}{ckpt_conv_name}.act_fake_quant.log2_base")
            log2_base_val = log2_base.item() if log2_base is not None else None

            self.layer_params[conv_name] = {
                "weight": w_tv,
                "bias": None,
                "bn_scale": bn_scale_tv if has_bn else None,
                "bn_shift": bn_shift_tv if has_bn else None,
                "log2_base": log2_base_val,
            }

        print(f"  Loaded {len(self.layer_params)} PTQ conv layers (BN separate)")

    def _conv_bn_relu(self, x: TVSparseConvTensor, conv_name: str,
                      ksize, stride, padding, subm, indice_key) -> TVSparseConvTensor:
        """Conv + BN + ReLU.

        BN scale is fused into conv weight at load time.
        BN shift + ReLU are applied via InferenceOps on GPU (matches spconv Python path).
        Log2 fake quantization is applied before conv when log2_base is present.
        """
        params = self.layer_params[conv_name]
        if params.get("log2_base") is not None:
            log2_fake_quant_inplace(x.features, params["log2_base"], self.stream)
        x = sparse_conv_forward(
            x, params["weight"], ksize, stride, padding, [1,1,1], subm,
            self.stream, self.arch, self.conv_tuner, indice_key)
        if params.get("bn_scale") is not None:
            bn_forward_inplace(x.features, params["bn_scale"], params["bn_shift"], self.stream)
            relu_inplace(x.features, self.stream)
        elif params["bias"] is not None:
            InferenceOps.bias_add_act_inplace(
                x.features, params["bias"], tv.gemm.Activation.ReLU,
                0.0, 0.0, self.stream)
        return x

    def _conv_bn(self, x: TVSparseConvTensor, conv_name: str,
                 ksize, stride, padding, subm, indice_key) -> TVSparseConvTensor:
        """Conv + BN (no ReLU, for residual blocks)."""
        params = self.layer_params[conv_name]
        if params.get("log2_base") is not None:
            log2_fake_quant_inplace(x.features, params["log2_base"], self.stream)
        x = sparse_conv_forward(
            x, params["weight"], ksize, stride, padding, [1,1,1], subm,
            self.stream, self.arch, self.conv_tuner, indice_key)
        if params.get("bn_scale") is not None:
            bn_forward_inplace(x.features, params["bn_scale"], params["bn_shift"], self.stream)
        elif params["bias"] is not None:
            InferenceOps.bias_add_inplace(x.features, params["bias"], self.stream)
        return x

    def _basic_block(self, x: TVSparseConvTensor, block_prefix: str) -> TVSparseConvTensor:
        """SparseBasicBlock: conv1+bn1+relu → conv2+bn2 → +identity → relu."""
        identity = x.features.clone()  # deep copy; _conv_bn_relu modifies x.features in-place via log2 quant

        out = self._conv_bn_relu(x, f"{block_prefix}.conv1",
                                 [3,3,3], [1,1,1], [1,1,1], True, None)
        out = self._conv_bn(out, f"{block_prefix}.conv2",
                            [3,3,3], [1,1,1], [1,1,1], True, None)

        # Residual add on GPU via cuBLAS axpy (no CPU roundtrip, no torch)
        n = out.features.dim(0) * out.features.dim(1)
        out_ptr = ctypes.c_void_p(out.features.byte_pointer())
        id_ptr = ctypes.c_void_p(identity.byte_pointer())
        if out.features.dtype == tv.float16:
            cublas_axpy_fp16(out_ptr, id_ptr, n, self.stream)
        else:
            cublas_axpy_fp32(out_ptr, id_ptr, n, self.stream)

        relu_inplace(out.features, self.stream)
        return out

    def forward(self, voxel_features: tv.Tensor, coors: tv.Tensor,
                batch_size: int,
                feature_ref=None,
                coors_ref=None) -> np.ndarray:
        """Full SparseEncoder forward. Returns dense BEV as numpy [N,C*D,H,W].

        Args:
            voxel_features: [num_voxels, 5] fp16 GPU (must be PyTorch-backed)
            coors: [num_voxels, 4] int32 GPU (batch_idx, z, y, x)
            batch_size: int
            feature_ref: optional torch.Tensor to keep voxel_features alive
            coors_ref: optional torch.Tensor to keep coors alive

        Returns:
            numpy array [batch, 256, 180, 180] fp16
        """
        global _torch_refs
        _torch_refs = []  # clear refs from previous forward
        if feature_ref is not None:
            _torch_refs.append(feature_ref)
        if coors_ref is not None:
            _torch_refs.append(coors_ref)

        x = TVSparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)

        # conv_input: SubMConv3d(5→16) + BN + ReLU
        x = self._conv_bn_relu(x, "conv_input.0", [3,3,3], [1,1,1], [1,1,1], True, "subm1")

        # encoder_layer1: 2 BasicBlocks + downsample
        x = self._basic_block(x, "encoder_layer1.0")
        x = self._basic_block(x, "encoder_layer1.1")
        x = self._conv_bn_relu(x, "encoder_layer1.2.0", [3,3,3], [2,2,2], [1,1,1], False, "spconv1")

        # encoder_layer2
        x = self._basic_block(x, "encoder_layer2.0")
        x = self._basic_block(x, "encoder_layer2.1")
        x = self._conv_bn_relu(x, "encoder_layer2.2.0", [3,3,3], [2,2,2], [1,1,1], False, "spconv2")

        # encoder_layer3
        x = self._basic_block(x, "encoder_layer3.0")
        x = self._basic_block(x, "encoder_layer3.1")
        x = self._conv_bn_relu(x, "encoder_layer3.2.0", [3,3,3], [2,2,2], [1,1,0], False, "spconv3")

        # encoder_layer4
        x = self._basic_block(x, "encoder_layer4.0")
        x = self._basic_block(x, "encoder_layer4.1")

        # conv_out: SparseConv3d(128→128, k=(1,1,3), s=(1,1,2)) + BN + ReLU
        x = self._conv_bn_relu(x, "conv_out.0", [1,1,3], [1,1,2], [0,0,0], False, "spconv_down2")

        # Sparse → Dense
        # _cudart.cudaDeviceSynchronize()  # Removed: cpu().numpy() already syncs implicitly
        indices_np = x.indices.cpu().numpy()  # [N, 4]
        features_np = x.features.cpu().numpy()  # [N, 128]

        # dense shape: [batch, *spatial_shape, C]
        # spatial_shape after all downsamples = [180, 180, 2] (H, W, D)
        dense_shape = [batch_size] + x.spatial_shape + [features_np.shape[1]]
        dense = scatter_nd_numpy(indices_np, features_np, dense_shape)
        # dense: [batch, H, W, D, C]
        # spconv dense() returns [N, C, *spatial_shape] = [N, C, H, W, D]
        # We need [N, C, D, H, W] → [N, C*D, H, W]
        # So: [batch, H, W, D, C] → [batch, C, D, H, W]
        dense = np.transpose(dense, (0, 4, 3, 1, 2))
        N, C, D, H, W = dense.shape
        dense = dense.reshape(N, C * D, H, W)
        return dense
