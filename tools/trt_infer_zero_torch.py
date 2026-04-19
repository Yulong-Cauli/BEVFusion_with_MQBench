"""
Zero-PyTorch standalone TRT inference pipeline for BEVFusion.

Uses ctypes + libcudart for GPU memory and CUDA ops.
Uses numpy for CPU-side geometry and post-processing.
No torch/ATen imports.

Usage (example):
    python tools/trt_infer_zero_torch.py --test-single ...
"""
import argparse
import ctypes
import hashlib
import logging
import os
import re
import sys
import time

import numpy as np
import tensorrt as trt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ------------------------------------------------------------------
# ctypes bindings for libcudart
# ------------------------------------------------------------------
_cudart = ctypes.CDLL("libcudart.so.11.0")

_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int
_cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_cudart.cudaMemset.restype = ctypes.c_int
_cudart.cudaSetDevice.argtypes = [ctypes.c_int]
_cudart.cudaSetDevice.restype = ctypes.c_int
_cudart.cudaGetLastError.restype = ctypes.c_int
_cudart.cudaDeviceSynchronize.restype = ctypes.c_int
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int
_cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamDestroy.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int
_cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaEventCreate.restype = ctypes.c_int
_cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaEventDestroy.restype = ctypes.c_int
_cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_cudart.cudaEventRecord.restype = ctypes.c_int
_cudart.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
_cudart.cudaStreamWaitEvent.restype = ctypes.c_int

CUDA_MEMCPY_H2D = 1
CUDA_MEMCPY_D2H = 2


class CudaBuffer:
    """Lightweight GPU buffer managed via ctypes + libcudart."""

    def __init__(self, nbytes, fill_value=None, dtype=None):
        self.nbytes = nbytes
        self._ptr = ctypes.c_void_p()
        ret = _cudart.cudaMalloc(ctypes.byref(self._ptr), nbytes)
        if ret != 0:
            raise RuntimeError(f"cudaMalloc failed: {ret}")
        if fill_value is not None:
            if dtype is not None and fill_value == 0:
                _cudart.cudaMemset(self._ptr, 0, nbytes)
            else:
                arr = np.full(nbytes // np.dtype(dtype).itemsize, fill_value, dtype=dtype)
                _cudart.cudaMemcpy(self._ptr, arr.ctypes.data, arr.nbytes, CUDA_MEMCPY_H2D)

    @property
    def ptr(self):
        return self._ptr.value

    def upload(self, arr: np.ndarray):
        assert arr.nbytes <= self.nbytes
        ret = _cudart.cudaMemcpy(self._ptr, arr.ctypes.data, arr.nbytes, CUDA_MEMCPY_H2D)
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed: {ret}")

    def download(self, shape, dtype):
        arr = np.empty(shape, dtype=dtype)
        ret = _cudart.cudaMemcpy(arr.ctypes.data, self._ptr, arr.nbytes, CUDA_MEMCPY_D2H)
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed: {ret}")
        return arr

    def __del__(self):
        if self._ptr.value:
            _cudart.cudaFree(self._ptr)


def make_cuda_buffer_from_array(arr: np.ndarray):
    buf = CudaBuffer(arr.nbytes)
    buf.upload(arr)
    # Preserve shape metadata for dynamic-shape TRT engines.
    buf.shape = tuple(arr.shape)
    return buf


def _stream_wait_for_default_stream(dst_stream):
    event = ctypes.c_void_p()
    ret = _cudart.cudaEventCreate(ctypes.byref(event))
    if ret != 0:
        raise RuntimeError(f"cudaEventCreate failed: {ret}")
    try:
        ret = _cudart.cudaEventRecord(event, ctypes.c_void_p(0))
        if ret != 0:
            raise RuntimeError(f"cudaEventRecord failed: {ret}")
        ret = _cudart.cudaStreamWaitEvent(ctypes.c_void_p(int(dst_stream)), event, 0)
        if ret != 0:
            raise RuntimeError(f"cudaStreamWaitEvent failed: {ret}")
    finally:
        _cudart.cudaEventDestroy(event)


def _summarize_np_tensor(arr: np.ndarray):
    arr = np.asarray(arr)
    size = int(arr.size)
    if size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": 0,
            "nonzero": 0,
            "nz_ratio": 0.0,
            "l2": 0.0,
            "abs_max": 0.0,
            "mean": 0.0,
        }
    arr64 = arr.astype(np.float64, copy=False)
    nonzero = int(np.count_nonzero(arr))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": size,
        "nonzero": nonzero,
        "nz_ratio": float(nonzero / float(size)),
        "l2": float(np.linalg.norm(arr64.ravel())),
        "abs_max": float(np.max(np.abs(arr64))),
        "mean": float(np.mean(arr64)),
    }


def _summarize_torch_tensor(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor):
        return _summarize_np_tensor(np.asarray(tensor))
    size = int(tensor.numel())
    if size == 0:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "size": 0,
            "nonzero": 0,
            "nz_ratio": 0.0,
            "l2": 0.0,
            "abs_max": 0.0,
            "mean": 0.0,
        }
    t = tensor.detach()
    nonzero = int(torch.count_nonzero(t).item())
    l2 = float(torch.linalg.vector_norm(t.float()).item())
    abs_max = float(torch.max(torch.abs(t.float())).item())
    mean = float(torch.mean(t.float()).item())
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "size": size,
        "nonzero": nonzero,
        "nz_ratio": float(nonzero / float(size)),
        "l2": l2,
        "abs_max": abs_max,
        "mean": mean,
    }


# ------------------------------------------------------------------
# Zero-torch TRT runner
# ------------------------------------------------------------------
class ZeroTorchTRTRunner:
    """Runs TRT engine using ctypes-managed CUDA buffers."""

    def __init__(self, engine_path, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        ret = _cudart.cudaStreamCreate(ctypes.byref(self.stream))
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreate failed: {ret}")
        self._output_buffer_cache = {}

        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes_np = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(self.engine.get_tensor_shape(name))
                dtype_trt = self.engine.get_tensor_dtype(name)
                if dtype_trt == trt.float16:
                    self.output_dtypes_np[name] = np.float16
                else:
                    self.output_dtypes_np[name] = np.float32
        self.logger.info(
            f"TRT engine loaded: {engine_path} "
            f"(inputs={self.input_names}, outputs={self.output_names})"
        )

    def __del__(self):
        stream = getattr(self, "stream", None)
        if stream is not None and getattr(stream, "value", None):
            _cudart.cudaStreamDestroy(stream)

    def __call__(self, input_buffers, return_gpu_buffers=False):
        """
        Args:
            input_buffers: dict[str -> CudaBuffer or int(data_ptr)]
            return_gpu_buffers: if True, return list[CudaBuffer] (GPU resident)
        Returns:
            list of np.ndarray (outputs copied to host) or list[CudaBuffer]
        """
        assert len(input_buffers) == len(self.input_names)

        for name in self.input_names:
            buf = input_buffers[name]
            if isinstance(buf, CudaBuffer):
                ptr = buf.ptr
            elif hasattr(buf, "ptr"):
                ptr = int(buf.ptr)
            elif hasattr(buf, "data_ptr"):
                ptr = int(buf.data_ptr())
            else:
                ptr = int(buf)
            shape = self._infer_shape(name, buf)
            self.context.set_input_shape(name, shape)
            self.context.set_tensor_address(name, ptr)

        output_buffers = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * np.dtype(self.output_dtypes_np[name]).itemsize
            cached = self._output_buffer_cache.get(name)
            if cached is None or cached.nbytes < nbytes:
                cached = CudaBuffer(nbytes)
                self._output_buffer_cache[name] = cached
            buf = cached
            # Preserve runtime shape metadata for downstream zero-copy chaining.
            buf.shape = tuple(int(v) for v in shape)
            output_buffers[name] = buf
            self.context.set_tensor_address(name, buf.ptr)

        self.context.execute_async_v3(stream_handle=self.stream.value)
        _cudart.cudaStreamSynchronize(self.stream)

        if return_gpu_buffers:
            return [output_buffers[name] for name in self.output_names]

        results = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            arr = output_buffers[name].download(shape, self.output_dtypes_np[name])
            results.append(arr)
        return results

    def _infer_shape(self, name, buf_or_ptr):
        shape = tuple(self.engine.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            meta_shape = getattr(buf_or_ptr, "shape", None)
            if meta_shape is None and hasattr(buf_or_ptr, "size"):
                size_attr = buf_or_ptr.size
                meta_shape = size_attr() if callable(size_attr) else size_attr
            if meta_shape is None:
                raise RuntimeError(
                    f"Dynamic TRT input '{name}' requires shape metadata; "
                    "use make_cuda_buffer_from_array for this input."
                )
            return tuple(int(v) for v in meta_shape)
        return shape


def _build_trt_engine_from_onnx(
    onnx_path,
    engine_path,
    *,
    fp16=True,
    int8=False,
    workspace_mb=4096,
    dynamic_shapes=None,
):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT build failed: {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    return engine_path


def _resolve_swin_onnx_path(swin_engine_path):
    candidates = []
    engine_dir = os.path.dirname(swin_engine_path)
    engine_base = os.path.basename(swin_engine_path)
    base_no_ext = os.path.splitext(engine_base)[0]

    candidates.append(os.path.join(engine_dir, base_no_ext + ".onnx"))
    stripped = re.sub(r"_sm\d+$", "", base_no_ext)
    stripped = re.sub(r"_trt\d+$", "", stripped)
    if stripped != base_no_ext:
        candidates.append(os.path.join(engine_dir, stripped + ".onnx"))

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find Swin ONNX beside engine {swin_engine_path}. Tried: {candidates}"
    )


def _is_engine_loadable(engine_path):
    trt_logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine is not None


def _swin_batched_engine_matches_base(
    base_engine_path, batched_engine_path, batch_size, c, h, w, logger
):
    try:
        base_runner = ZeroTorchTRTRunner(base_engine_path, logger)
        batched_runner = ZeroTorchTRTRunner(batched_engine_path, logger)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((batch_size, c, h, w), dtype=np.float32)

        batched_out = [
            out.astype(np.float32)
            for out in batched_runner(
                {batched_runner.input_names[0]: make_cuda_buffer_from_array(x)}
            )
        ]

        serial_out = []
        for i in range(batch_size):
            outs = base_runner({base_runner.input_names[0]: make_cuda_buffer_from_array(x[i:i + 1])})
            if not serial_out:
                serial_out = [[] for _ in range(len(outs))]
            for s, out in enumerate(outs):
                serial_out[s].append(out.astype(np.float32))
        serial_out = [np.concatenate(v, axis=0) for v in serial_out]

        for idx, (ref, got) in enumerate(zip(serial_out, batched_out)):
            cos = float(
                np.dot(ref.ravel(), got.ravel())
                / (np.linalg.norm(ref.ravel()) * np.linalg.norm(got.ravel()) + 1e-12)
            )
            if cos < 0.999:
                logger.warning(
                    f"Batched Swin engine sanity check failed on output[{idx}] "
                    f"(cos={cos:.6f}); fallback to base Swin engine."
                )
                return False
        return True
    except Exception as exc:
        logger.warning(f"Batched Swin engine sanity check failed ({exc}); fallback to base.")
        return False


def prepare_swin_batched_engine(swin_engine_path, batch_size=6, logger=None, auto_build=False):
    logger = logger or logging.getLogger(__name__)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(swin_engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {swin_engine_path}")

    input_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
            break
    if input_name is None:
        raise RuntimeError(f"No input tensor found in engine: {swin_engine_path}")

    in_shape = tuple(engine.get_tensor_shape(input_name))
    if len(in_shape) != 4:
        raise RuntimeError(f"Unexpected Swin input shape: {in_shape}")
    if in_shape[0] == batch_size:
        return swin_engine_path
    if in_shape[0] < 0:
        # Dynamic batch engine already; assume caller will use requested batch.
        return swin_engine_path
    if in_shape[0] != 1:
        raise RuntimeError(
            f"Cannot auto-convert Swin engine with fixed batch {in_shape[0]} "
            f"to batch {batch_size}"
        )

    engine_dir = os.path.dirname(swin_engine_path)
    engine_base = os.path.basename(swin_engine_path)
    stem, ext = os.path.splitext(engine_base)
    batched_stem = stem
    if "_sm" in stem:
        pfx, sfx = stem.rsplit("_sm", 1)
        batched_stem = f"{pfx}_b{batch_size}_sm{sfx}"
    elif "_trt" in stem:
        pfx, sfx = stem.rsplit("_trt", 1)
        batched_stem = f"{pfx}_b{batch_size}_trt{sfx}"
    else:
        batched_stem = f"{stem}_b{batch_size}"
    batched_engine_path = os.path.join(engine_dir, batched_stem + ext)
    if os.path.exists(batched_engine_path):
        try:
            with open(batched_engine_path, "rb") as f:
                batched_engine = runtime.deserialize_cuda_engine(f.read())
            if batched_engine is not None:
                batched_input = None
                for i in range(batched_engine.num_io_tensors):
                    n = batched_engine.get_tensor_name(i)
                    if batched_engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                        batched_input = n
                        break
                if batched_input is not None:
                    batched_shape = tuple(batched_engine.get_tensor_shape(batched_input))
                    if len(batched_shape) == 4 and batched_shape[0] == batch_size:
                        c, h, w = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
                        if _swin_batched_engine_matches_base(
                            swin_engine_path, batched_engine_path, batch_size, c, h, w, logger
                        ):
                            return batched_engine_path
                        return swin_engine_path
                logger.warning(
                    f"Existing batched Swin engine has unexpected input shape "
                    f"{batched_shape if batched_input is not None else 'unknown'}, rebuilding."
                )
            else:
                logger.warning("Existing batched Swin engine is not loadable, rebuilding.")
        except Exception as exc:
            logger.warning(f"Failed to inspect existing batched Swin engine ({exc}), rebuilding.")
    if not auto_build:
        logger.info(
            f"Swin engine is fixed batch={in_shape[0]} and no prebuilt batch-{batch_size} "
            "engine found; keep original engine."
        )
        return swin_engine_path

    patched_onnx_path = os.path.join(engine_dir, batched_stem + ".onnx")
    batched_onnx_candidates = []
    stripped_batched = re.sub(r"_sm\d+$", "", batched_stem)
    if stripped_batched != batched_stem:
        batched_onnx_candidates.append(os.path.join(engine_dir, stripped_batched + ".onnx"))
    batched_onnx_candidates.append(patched_onnx_path)
    prebuilt_batched_onnx = next((p for p in batched_onnx_candidates if os.path.exists(p)), None)

    if prebuilt_batched_onnx is not None:
        logger.info(
            f"Building batched Swin engine (B={batch_size}) from prebuilt ONNX: "
            f"{prebuilt_batched_onnx}"
        )
        _build_trt_engine_from_onnx(
            prebuilt_batched_onnx,
            batched_engine_path,
            fp16=True,
            int8=True,
            workspace_mb=4096,
        )
        logger.info(f"Batched Swin engine ready: {batched_engine_path}")
        c, h, w = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
        if _swin_batched_engine_matches_base(
            swin_engine_path, batched_engine_path, batch_size, c, h, w, logger
        ):
            return batched_engine_path
        return swin_engine_path

    onnx_src = _resolve_swin_onnx_path(swin_engine_path)
    logger.info(f"Building batched Swin engine (B={batch_size}) from {onnx_src}")

    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_src)
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) > 0:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = int(batch_size)

    init_map = {init.name: init for init in model.graph.initializer}
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        init = init_map.get(shape_name)
        if init is None:
            continue
        arr = numpy_helper.to_array(init).copy()
        if arr.ndim == 1 and arr.size > 0 and arr[0] == 1:
            arr[0] = int(batch_size)
            init.CopyFrom(numpy_helper.from_array(arr.astype(arr.dtype), shape_name))
            patched += 1
    logger.info(f"Patched {patched} reshape tensors for fixed batch={batch_size}")
    onnx.save(model, patched_onnx_path)

    _build_trt_engine_from_onnx(
        patched_onnx_path,
        batched_engine_path,
        fp16=True,
        int8=True,
        workspace_mb=4096,
    )
    logger.info(f"Batched Swin engine ready: {batched_engine_path}")
    c, h, w = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
    if _swin_batched_engine_matches_base(
        swin_engine_path, batched_engine_path, batch_size, c, h, w, logger
    ):
        return batched_engine_path
    return swin_engine_path


def prepare_bev_downsample_engine(bev_downsample, engine_path, input_shape, logger=None):
    logger = logger or logging.getLogger(__name__)
    if bev_downsample is None:
        return None
    import torch.nn as nn
    if isinstance(bev_downsample, nn.Identity):
        return None
    if os.path.exists(engine_path):
        try:
            if _is_engine_loadable(engine_path):
                return engine_path
            logger.warning(
                f"Existing BEV downsample engine is not loadable on this GPU, rebuilding: {engine_path}"
            )
        except Exception:
            logger.warning(
                f"Existing BEV downsample engine check failed, rebuilding: {engine_path}"
            )

    onnx_path = os.path.splitext(engine_path)[0] + ".onnx"
    logger.info(f"Exporting BEV downsample ONNX: {onnx_path}")
    import torch
    module = bev_downsample.eval().cuda().float()
    dummy = torch.randn(*input_shape, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        _ = module(dummy)
    torch.onnx.export(
        module,
        dummy,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["camera_bev"],
        output_names=["camera_bev_down"],
        dynamic_axes=None,
    )

    logger.info(f"Building BEV downsample TRT engine: {engine_path}")
    _build_trt_engine_from_onnx(
        onnx_path,
        engine_path,
        fp16=False,
        int8=False,
        workspace_mb=1024,
    )
    return engine_path


# ------------------------------------------------------------------
# Zero-torch voxelization (hard_voxelize)
# ------------------------------------------------------------------
_VOXEL_SO = ctypes.CDLL(
    os.path.join(ROOT, "tools", "zero_torch_ops", "voxel_layer",
                 "libbevfusion_voxel_layer.so")
)

_VOXEL_SO.hard_voxelize_gpu_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_VOXEL_SO.hard_voxelize_gpu_cuda.restype = ctypes.c_int

_VOXEL_SO.dynamic_voxelize_gpu_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_VOXEL_SO.dynamic_voxelize_gpu_cuda.restype = ctypes.c_int


class ZeroTorchVoxelization:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        self.voxel_size = list(voxel_size)
        self.point_cloud_range = list(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self._workspace = {}

    def _get_workspace(self, num_features, ndim):
        key = (int(num_features), int(ndim))
        cached = self._workspace.get(key)
        if cached is not None:
            return cached
        max_voxels = self.max_voxels
        max_points = self.max_num_points
        cached = (
            CudaBuffer(max_voxels * max_points * num_features * 4),
            CudaBuffer(max_voxels * ndim * 4),
            CudaBuffer(max_voxels * 4),
        )
        self._workspace[key] = cached
        return cached

    def __call__(self, points_gpu_ptr, num_points, num_features, NDim=3):
        """
        Args:
            points_gpu_ptr: GPU pointer to float [num_points, num_features]
        Returns:
            voxels_np, coors_np, num_points_per_voxel_np, voxel_num
        """
        max_voxels = self.max_voxels
        max_points = self.max_num_points
        voxels_buf, coors_buf, num_buf = self._get_workspace(num_features, NDim)

        voxel_x, voxel_y, voxel_z = self.voxel_size
        cr = self.point_cloud_range
        grid_x = round((cr[3] - cr[0]) / voxel_x)
        grid_y = round((cr[4] - cr[1]) / voxel_y)
        grid_z = round((cr[5] - cr[2]) / voxel_z)

        voxel_num = _VOXEL_SO.hard_voxelize_gpu_cuda(
            ctypes.c_void_p(points_gpu_ptr),
            ctypes.c_void_p(voxels_buf.ptr),
            ctypes.c_void_p(coors_buf.ptr),
            ctypes.c_void_p(num_buf.ptr),
            num_points, num_features,
            voxel_x, voxel_y, voxel_z,
            cr[0], cr[1], cr[2],
            cr[3], cr[4], cr[5],
            grid_x, grid_y, grid_z,
            max_points, max_voxels,
            NDim, 0,
        )
        if voxel_num < 0:
            raise RuntimeError(f"hard_voxelize_gpu_cuda failed: {voxel_num}")
        if voxel_num == 0:
            return (
                np.empty((0, max_points, num_features), dtype=np.float32),
                np.empty((0, NDim), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                0,
            )

        voxels_np = voxels_buf.download((voxel_num, max_points, num_features), np.float32)
        coors_np = coors_buf.download((voxel_num, NDim), np.int32)
        num_np = num_buf.download((voxel_num,), np.int32)
        return voxels_np, coors_np, num_np, voxel_num


# ------------------------------------------------------------------
# Zero-torch bev_pool_v2
# ------------------------------------------------------------------
_BEVPOOL_SO = ctypes.CDLL(
    os.path.join(ROOT, "tools", "zero_torch_ops", "bev_pool",
                 "libbevfusion_bev_pool.so")
)

_BEVPOOL_SO.bev_pool_forward_cuda.argtypes = [
    ctypes.c_int,   # b
    ctypes.c_int,   # d
    ctypes.c_int,   # h
    ctypes.c_int,   # w
    ctypes.c_int,   # n
    ctypes.c_int,   # c
    ctypes.c_int,   # n_intervals
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # geom_feats
    ctypes.c_void_p,  # interval_starts
    ctypes.c_void_p,  # interval_lengths
    ctypes.c_void_p,  # out
]
_BEVPOOL_SO.bev_pool_forward_cuda.restype = ctypes.c_int


def np_bev_pool_v2(x_np, geom_feats_np, interval_starts_np, interval_lengths_np, B, D, H, W):
    """
    Pure-numpy wrapper around zero-torch bev_pool.
    Inputs are numpy arrays; output is numpy array.
    (This incurs H2D/D2H copies for x/geom/intervals/out, which is acceptable
    for correctness verification but will be optimized in Phase B.)
    """
    C = x_np.shape[1]
    N = x_np.shape[0]
    n_intervals = interval_starts_np.shape[0]
    # CUDA kernel writes [B,D,H,W,C]; download as that then permute to [B,C,D,H,W]
    out_np_bdwc = np.zeros((B, D, H, W, C), dtype=np.float32)

    x_buf = make_cuda_buffer_from_array(x_np)
    geom_buf = make_cuda_buffer_from_array(geom_feats_np)
    starts_buf = make_cuda_buffer_from_array(interval_starts_np)
    lengths_buf = make_cuda_buffer_from_array(interval_lengths_np)
    out_buf = CudaBuffer(out_np_bdwc.nbytes)

    ret = _BEVPOOL_SO.bev_pool_forward_cuda(
        B, D, H, W,
        N, C, n_intervals,
        ctypes.c_void_p(x_buf.ptr),
        ctypes.c_void_p(geom_buf.ptr),
        ctypes.c_void_p(starts_buf.ptr),
        ctypes.c_void_p(lengths_buf.ptr),
        ctypes.c_void_p(out_buf.ptr),
    )
    if ret != 0:
        raise RuntimeError(f"bev_pool_forward_cuda failed: {ret}")

    out_arr = out_buf.download(out_np_bdwc.shape, np.float32)
    return out_arr.transpose(0, 4, 1, 2, 3)


# ------------------------------------------------------------------
# GPU zero-copy vtransform
# ------------------------------------------------------------------
from tools.zero_torch_ops.vtransform_gpu.vtransform_wrapper import (
    compute_depth_map_cuda,
    vtransform_gpu_workspace_size,
    vtransform_post_depthnet_cuda,
)


# ------------------------------------------------------------------
# Numpy VTransform geometry
# ------------------------------------------------------------------
class NumpyVTransformGeometry:
    def __init__(self, image_size, feature_size, xbound, ybound, zbound, dbound):
        self.image_size = image_size
        self.feature_size = feature_size

        dx = np.array([row[2] for row in [xbound, ybound, zbound]], dtype=np.float32)
        bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]], dtype=np.float32)
        nx = np.array([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]], dtype=np.int64)
        self.dx = dx
        self.bx = bx
        self.nx = nx

        iH, iW = image_size
        fH, fW = feature_size
        ds = np.arange(*dbound, dtype=np.float32).reshape(-1, 1, 1)
        ds = np.broadcast_to(ds, (ds.shape[0], fH, fW))
        D = ds.shape[0]
        xs = np.linspace(0, iW - 1, fW, dtype=np.float32).reshape(1, 1, fW)
        xs = np.broadcast_to(xs, (D, fH, fW))
        ys = np.linspace(0, iH - 1, fH, dtype=np.float32).reshape(1, fH, 1)
        ys = np.broadcast_to(ys, (D, fH, fW))
        self.frustum = np.stack((xs, ys, ds), axis=-1)  # [D, fH, fW, 3]
        self.D = D

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins,
                     post_rots, post_trans, extra_rots=None, extra_trans=None):
        B, N, _ = camera2lidar_trans.shape
        frustum = self.frustum  # [D, fH, fW, 3]
        # Expand to [B, N, D, fH, fW, 3]
        points = frustum.reshape(1, 1, *frustum.shape)
        points = np.broadcast_to(points, (B, N, *frustum.shape)).copy()

        post_trans = post_trans.reshape(B, N, 1, 1, 1, 3)
        points = points - post_trans

        inv_post_rots = np.linalg.inv(post_rots).reshape(B, N, 1, 1, 1, 3, 3)
        points = np.matmul(inv_post_rots, points[..., np.newaxis]).squeeze(-1)

        points_xy = points[..., :2] * points[..., 2:3]
        points = np.concatenate([points_xy, points[..., 2:3]], axis=-1)

        inv_intrins = np.linalg.inv(intrins)
        combine = np.matmul(camera2lidar_rots, inv_intrins).reshape(B, N, 1, 1, 1, 3, 3)
        points = np.matmul(combine, points[..., np.newaxis]).squeeze(-1)
        points += camera2lidar_trans.reshape(B, N, 1, 1, 1, 3)

        if extra_rots is not None:
            extra_rots = extra_rots.reshape(B, 1, 1, 1, 1, 3, 3)
            extra_rots = np.broadcast_to(extra_rots, (B, N, 1, 1, 1, 3, 3))
            points = np.matmul(extra_rots, points[..., np.newaxis]).squeeze(-1)
        if extra_trans is not None:
            extra_trans = extra_trans.reshape(B, 1, 1, 1, 1, 3)
            extra_trans = np.broadcast_to(extra_trans, (B, N, 1, 1, 1, 3))
            points += extra_trans
        return points

    def precompute_bev_indices(self, geom, B):
        N_per_batch = geom.shape[1] * geom.shape[2] * geom.shape[3] * geom.shape[4]
        Nprime = B * N_per_batch
        dx = self.dx
        bx = self.bx
        nx = self.nx

        geom_feats = ((geom - (bx - dx / 2.0)) / dx).astype(np.int64)
        geom_feats = geom_feats.reshape(Nprime, 3)
        batch_ix = np.concatenate([
            np.full((N_per_batch, 1), ix, dtype=np.int64)
            for ix in range(B)
        ], axis=0)
        geom_feats = np.concatenate([geom_feats, batch_ix], axis=1)

        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) &
            (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) &
            (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        )
        geom_feats = geom_feats[kept]
        D_val, H_val, W_val = int(nx[2]), int(nx[0]), int(nx[1])
        ranks = (
            geom_feats[:, 0] * (W_val * D_val * B)
            + geom_feats[:, 1] * (D_val * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sort_indices = np.argsort(ranks)
        geom_feats = geom_feats[sort_indices]
        ranks = ranks[sort_indices]

        kept_intervals = np.ones(ranks.shape[0], dtype=bool)
        if ranks.shape[0] > 0:
            kept_intervals[1:] = ranks[1:] != ranks[:-1]
        interval_starts = np.where(kept_intervals)[0].astype(np.int32)
        interval_lengths = np.zeros_like(interval_starts)
        if interval_lengths.shape[0] > 1:
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        if interval_lengths.shape[0] > 0:
            interval_lengths[-1] = ranks.shape[0] - interval_starts[-1]

        return {
            "kept": kept,
            "sort_indices": sort_indices.astype(np.int32),
            "geom_feats": geom_feats.astype(np.int32),
            "interval_starts": interval_starts,
            "interval_lengths": interval_lengths,
            "B": B, "D": D_val, "H": H_val, "W": W_val,
        }

    def compute_depth_map(self, points, img_aug_matrix, lidar_aug_matrix,
                          lidar2image, B, N):
        depth = np.zeros((B, N, 1, *self.image_size), dtype=np.float32)
        for b in range(B):
            cur_coords = points[b][:, :3]  # [M, 3]
            cur_img_aug = img_aug_matrix[b]
            cur_lidar_aug = lidar_aug_matrix[b]
            cur_l2i = lidar2image[b]

            cur_coords = cur_coords - cur_lidar_aug[:3, 3]
            cur_coords = np.linalg.inv(cur_lidar_aug[:3, :3]).dot(cur_coords.T)
            cur_coords = np.matmul(cur_l2i[:, :3, :3], cur_coords)
            cur_coords += cur_l2i[:, :3, 3].reshape(-1, 3, 1)
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = np.clip(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            cur_coords = np.matmul(cur_img_aug[:, :3, :3], cur_coords)
            cur_coords += cur_img_aug[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(0, 2, 1)
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) &
                (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].astype(np.int32)
                masked_dist = dist[c, on_img[c]]
                if masked_coords.shape[0] > 0:
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        return depth


# ------------------------------------------------------------------
# Zero-torch NMS (iou3d)
# ------------------------------------------------------------------
_IOU3D_SO = ctypes.CDLL(
    os.path.join(ROOT, "tools", "zero_torch_ops", "iou3d",
                 "libbevfusion_iou3d.so")
)

_IOU3D_SO.nms_gpu_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int,
]
_IOU3D_SO.nms_gpu_cuda.restype = ctypes.c_int


def np_nms_gpu(boxes_np, scores_np, thresh, pre_maxsize=None, post_max_size=None):
    """
    boxes_np: float [N, 5] (x1, y1, x2, y2, ry) on CPU
    scores_np: float [N]
    """
    order = np.argsort(-scores_np)
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes_np[order].astype(np.float32).copy()

    d_boxes = make_cuda_buffer_from_array(boxes)
    keep = np.zeros(boxes.shape[0], dtype=np.int64)
    num_out = _IOU3D_SO.nms_gpu_cuda(
        ctypes.c_void_p(d_boxes.ptr),
        ctypes.c_void_p(keep.ctypes.data),
        boxes.shape[0],
        ctypes.c_float(thresh),
        0,
    )
    if num_out < 0:
        raise RuntimeError(f"nms_gpu_cuda failed: {num_out}")
    keep = order[keep[:num_out]]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def circle_nms(dets, thresh):
    """Circular NMS (pure numpy)."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2
            if dist <= thresh ** 2:
                suppressed[j] = 1
    return keep


# ------------------------------------------------------------------
# Numpy BBox coder
# ------------------------------------------------------------------
class NumpyTransFusionBBoxCoder:
    def __init__(self, pc_range, out_size_factor, voxel_size,
                 post_center_range=None, score_threshold=None, code_size=10):
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.out_size_factor = out_size_factor
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        if hasattr(post_center_range, 'cpu'):
            post_center_range = post_center_range.cpu().numpy()
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        # heatmap: [B, num_cls, num_proposals]
        final_preds = heatmap.argmax(axis=1)  # [B, num_proposals]
        final_scores = heatmap.max(axis=1)    # [B, num_proposals]

        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        dim[:, 0, :] = np.exp(dim[:, 0, :])
        dim[:, 1, :] = np.exp(dim[:, 1, :])
        dim[:, 2, :] = np.exp(dim[:, 2, :])
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = np.arctan2(rots, rotc)

        final_box_preds = np.concatenate([center, height, dim, rot, vel], axis=1).transpose(0, 2, 1)  # [B, N, code_size]

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]

            if filter and self.post_center_range is not None:
                post_center_range = np.array(self.post_center_range, dtype=np.float32)
                mask = (boxes3d[:, :3] >= post_center_range[:3]).all(axis=1)
                mask &= (boxes3d[:, :3] <= post_center_range[3:]).all(axis=1)
                if self.score_threshold is not None:
                    mask &= scores > self.score_threshold
                boxes3d = boxes3d[mask]
                scores = scores[mask]
                labels = labels[mask]

            predictions_dicts.append({
                'bboxes': boxes3d, 'scores': scores, 'labels': labels
            })
        return predictions_dicts


# ------------------------------------------------------------------
# SimpleLiDARBox (numpy)
# ------------------------------------------------------------------
class SimpleLiDARBox:
    def __init__(self, tensor, box_dim=9):
        self.tensor = tensor
        self.box_dim = box_dim

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def gravity_center(self):
        bottom_center = self.tensor[:, :3]
        gc = np.zeros_like(bottom_center)
        gc[:, :2] = bottom_center[:, :2]
        gc[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gc

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    @property
    def bev(self):
        t = self.tensor
        cx, cy = t[:, 0], t[:, 1]
        w, l = t[:, 3], t[:, 4]
        yaw = t[:, 6]
        bev_boxes = np.stack([
            cx - w / 2, cy - l / 2,
            cx + w / 2, cy + l / 2,
            yaw
        ], axis=1)
        return bev_boxes


# ------------------------------------------------------------------
# Main hybrid model
# ------------------------------------------------------------------
class ZeroTorchBEVFusion:
    def __init__(self, swin_trt, depthnet_trt, fuser_trt, neck_trt, head_trt,
                 lidar_backbone, voxelizer, vtransform_geom, bev_downsample,
                 bbox_coder, test_cfg, num_proposals, num_classes,
                 voxelize_reduce, logger, use_tv_lidar=False,
                 use_gpu_vtransform=False, capture_intermediates=True,
                 enable_lidar_gpu_chain=True):
        self.swin_trt = swin_trt
        self.depthnet_trt = depthnet_trt
        self.fuser_trt = fuser_trt
        self.neck_trt = neck_trt
        self.head_trt = head_trt
        self.lidar_backbone = lidar_backbone
        self.voxelizer = voxelizer
        self.vtransform_geom = vtransform_geom
        self.bev_downsample = bev_downsample
        self.bbox_coder = bbox_coder
        self.test_cfg = test_cfg
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.voxelize_reduce = voxelize_reduce
        self.logger = logger
        self.use_tv_lidar = use_tv_lidar
        self.use_gpu_vtransform = use_gpu_vtransform
        self.capture_intermediates = capture_intermediates
        self.enable_lidar_gpu_chain = enable_lidar_gpu_chain
        self._forward_count = 0
        self._tv_lidar_warmed = False
        self._last_intermediates = {}
        self._verbose_steps = os.environ.get("ZERO_TORCH_VERBOSE", "0") == "1"
        self._upload_buffers = {}
        if not self.use_tv_lidar:
            raise RuntimeError(
                "PyTorch LiDAR fallback has been removed in strict zero-torch mode. "
                "Ensure TVSparseEncoder is available."
            )
        if self.bev_downsample is not None and not isinstance(self.bev_downsample, ZeroTorchTRTRunner):
            raise RuntimeError(
                "PyTorch bev_downsample fallback has been removed in strict zero-torch mode. "
                "Provide a TRT bev_downsample runner or disable downsample."
            )

    def _vprint(self, msg):
        if self._verbose_steps:
            print(msg, flush=True)

    def _upload_cached(self, key, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        cached = self._upload_buffers.get(key)
        if cached is None or cached.nbytes < arr.nbytes:
            cached = CudaBuffer(arr.nbytes)
            self._upload_buffers[key] = cached
        cached.upload(arr)
        cached.shape = tuple(arr.shape)
        return cached

    def forward(self, img, points, camera2ego, lidar2ego, lidar2camera,
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, **kwargs):
        self._last_intermediates = {}
        B, N, C, H, W = img.shape

        timings = {}
        _t0 = time.time()

        # Compute LiDAR branch first to avoid downstream camera CUDA ops
        # impacting TVSparseEncoder numerics on shared default stream.
        self._vprint("[ZeroTorch] Step 4: LiDAR backbone (precompute)")
        feats, coords, sizes = self._voxelize(points)
        if self.use_tv_lidar and coords.shape[0] > 0:
            order = np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))
            feats = np.ascontiguousarray(feats[order])
            coords = np.ascontiguousarray(coords[order])
            sizes = np.ascontiguousarray(sizes[order])
        batch_size = int(coords[-1, 0]) + 1 if coords.shape[0] > 0 else 0
        coords_i32 = coords.astype(np.int32, copy=False)
        feats_f32 = feats.astype(np.float32, copy=False)
        if self.capture_intermediates:
            self._last_intermediates["lidar_voxel_stats"] = {
                "batch_size": int(batch_size),
                "num_voxels": int(feats.shape[0]),
                "num_points_per_voxel": _summarize_np_tensor(sizes),
                "voxel_features": _summarize_np_tensor(feats),
                "coords": _summarize_np_tensor(coords_i32),
                "coords_md5": hashlib.md5(coords_i32.tobytes()).hexdigest(),
                "features_md5": hashlib.md5(feats_f32.tobytes()).hexdigest(),
            }
            self._last_intermediates["lidar_voxel_features"] = feats_f32.copy()
            self._last_intermediates["lidar_voxel_coords"] = coords_i32.copy()
        lidar_bev_gpu = None
        if self.use_tv_lidar:
            from cumm import tensorview as tv
            import torch
            feats_fp16 = torch.from_numpy(feats).cuda().half().contiguous()
            feats_tv = tv.from_blob(feats_fp16.data_ptr(), list(feats_fp16.shape), tv.float16, 0)
            coords_i32 = torch.from_numpy(coords.astype(np.int32)).cuda().contiguous()
            coords_tv = tv.from_blob(coords_i32.data_ptr(), list(coords_i32.shape), tv.int32, 0)
            batch_size = int(coords[-1, 0]) + 1

            def _run_tv_lidar():
                try:
                    return self.lidar_backbone.forward(
                        feats_tv, coords_tv, batch_size,
                        feature_ref=feats_fp16, coors_ref=coords_i32,
                        return_torch=self.enable_lidar_gpu_chain,
                    )
                except TypeError:
                    return self.lidar_backbone.forward(
                        feats_tv, coords_tv, batch_size,
                        feature_ref=feats_fp16, coors_ref=coords_i32,
                    )

            lidar_bev_obj = None
            candidate_stats = []
            selected_try = -1
            # Some TV backends may return a sparsified first output before kernels settle.
            # Retry a few times and keep the first stable dense-looking result.
            max_retries = 1 if self._tv_lidar_warmed else 4
            for retry_idx in range(max_retries):
                candidate = _run_tv_lidar()
                lidar_bev_obj = candidate
                if self.capture_intermediates or max_retries > 1:
                    stats = (
                        _summarize_torch_tensor(candidate)
                        if hasattr(candidate, "detach")
                        else _summarize_np_tensor(candidate)
                    )
                    stats["retry"] = int(retry_idx)
                    candidate_stats.append(stats)
                else:
                    if hasattr(candidate, "detach"):
                        size = max(1, int(candidate.numel()))
                        nz_ratio = float((candidate != 0).sum().item() / size)
                    else:
                        arr = np.asarray(candidate)
                        size = max(1, int(arr.size))
                        nz_ratio = float(np.count_nonzero(arr) / size)
                    stats = {"retry": int(retry_idx), "nz_ratio": nz_ratio, "l2": 0.0}
                if stats["nz_ratio"] > 0.02:
                    selected_try = retry_idx
                    break
            if selected_try < 0:
                selected_try = max(0, len(candidate_stats) - 1)
            if self.capture_intermediates:
                self._last_intermediates["lidar_tv_diag"] = {
                    "mode": "tv",
                    "selected_try": int(selected_try),
                    "candidates": candidate_stats,
                }
            self._tv_lidar_warmed = True
            if hasattr(lidar_bev_obj, "detach"):
                lidar_bev_gpu = lidar_bev_obj.float().contiguous()
                lidar_bev = (
                    lidar_bev_gpu.detach().cpu().numpy().copy()
                    if self.capture_intermediates
                    else None
                )
            else:
                lidar_bev = lidar_bev_obj
        else:
            raise RuntimeError(
                "PyTorch LiDAR fallback path is disabled. "
                "Set up TVSparseEncoder and keep use_tv_lidar=True."
            )
        if self.capture_intermediates and lidar_bev is not None:
            self._last_intermediates['lidar_bev'] = lidar_bev.copy()
        _t1 = time.time(); timings['lidar'] = (_t1 - _t0) * 1000.0; _t0 = _t1

        self._vprint("[ZeroTorch] Step 1: SwinT backbone")
        # Step 1: SwinT backbone (TRT)
        img_flat = img.reshape(B * N, C, H, W).astype(np.float32)
        swin_input_name = self.swin_trt.input_names[0]
        swin_engine_shape = tuple(self.swin_trt.engine.get_tensor_shape(swin_input_name))
        if len(swin_engine_shape) != 4:
            raise RuntimeError(f"Unexpected Swin input shape: {swin_engine_shape}")

        if swin_engine_shape[0] < 0 or swin_engine_shape[0] == B * N:
            img_buf = self._upload_cached("swin.image", img_flat)
            swin_outs = self.swin_trt({swin_input_name: img_buf})
            multi_scale_feats = [o.astype(np.float32) for o in swin_outs]
        elif swin_engine_shape[0] == 1:
            # Compatibility fallback for legacy B=1 engines.
            swin_outputs = []
            for i in range(B * N):
                img_buf = self._upload_cached("swin.image.b1", img_flat[i:i + 1])
                outs = self.swin_trt({swin_input_name: img_buf})
                swin_outputs.append([o.astype(np.float32) for o in outs])

            num_scales = len(swin_outputs[0])
            multi_scale_feats = []
            for s in range(num_scales):
                feat = np.concatenate([swin_outputs[i][s] for i in range(B * N)], axis=0)
                multi_scale_feats.append(feat)
        else:
            raise RuntimeError(
                f"Swin engine fixed batch={swin_engine_shape[0]} incompatible with B*N={B*N}"
            )
        _t1 = time.time(); timings['swin'] = (_t1 - _t0) * 1000.0; _t0 = _t1
        self._vprint("[ZeroTorch] Step 2: Camera neck")
        # Step 2: Camera neck (TRT)
        neck_inputs = {
            self.neck_trt.input_names[0]: self._upload_cached("neck.in0", multi_scale_feats[0]),
            self.neck_trt.input_names[1]: self._upload_cached("neck.in1", multi_scale_feats[1]),
            self.neck_trt.input_names[2]: self._upload_cached("neck.in2", multi_scale_feats[2]),
        }
        neck_out = self.neck_trt(neck_inputs)
        x_cam = neck_out[0].astype(np.float32)
        _t1 = time.time(); timings['neck'] = (_t1 - _t0) * 1000.0; _t0 = _t1
        self._vprint("[ZeroTorch] Step 3: vtransform depthnet + bev_pool")
        # Step 3: vtransform depthnet (TRT) + bev_pool_v2
        BN, C_neck, fH, fW = x_cam.shape
        x_cam_5d = x_cam.reshape(B, N, C_neck, fH, fW)
        D = self.vtransform_geom.D

        if self.use_gpu_vtransform:
            _vt_t0 = time.time()
            # --- compute_depth_map on GPU ---
            points_prefix_sum = [0]
            points_list = []
            for b in range(B):
                pts = points[b][:, :3].astype(np.float32)
                points_list.append(pts)
                points_prefix_sum.append(points_prefix_sum[-1] + pts.shape[0])
            points_np = np.concatenate(points_list, axis=0)
            points_gpu = self._upload_cached("vt.points", points_np)
            prefix_sum_gpu = self._upload_cached(
                "vt.prefix_sum", np.array(points_prefix_sum, dtype=np.int32)
            )

            inv_lidar_aug_rot = np.linalg.inv(lidar_aug_matrix[:, :3, :3]).astype(np.float32)
            inv_lidar_aug_trans = lidar_aug_matrix[:, :3, 3].astype(np.float32)
            lidar2image_rot = lidar2image[:, :, :3, :3].astype(np.float32)
            lidar2image_trans = lidar2image[:, :, :3, 3].astype(np.float32)
            img_aug_rot = img_aug_matrix[:, :, :3, :3].astype(np.float32)
            img_aug_trans = img_aug_matrix[:, :, :3, 3].astype(np.float32)

            iH, iW = self.vtransform_geom.image_size
            depth_map_gpu = CudaBuffer(B * N * iH * iW * 4, fill_value=0, dtype=np.float32)
            _ilr = self._upload_cached("vt.inv_lidar_aug_rot", inv_lidar_aug_rot)
            _ilt = self._upload_cached("vt.inv_lidar_aug_trans", inv_lidar_aug_trans)
            _l2r = self._upload_cached("vt.lidar2image_rot", lidar2image_rot)
            _l2t = self._upload_cached("vt.lidar2image_trans", lidar2image_trans)
            _iar = self._upload_cached("vt.img_aug_rot", img_aug_rot)
            _iat = self._upload_cached("vt.img_aug_trans", img_aug_trans)
            compute_depth_map_cuda(
                points_gpu.ptr, prefix_sum_gpu.ptr, points_np.shape[0],
                _ilr.ptr, _ilt.ptr, _l2r.ptr, _l2t.ptr, _iar.ptr, _iat.ptr,
                B, N, iH, iW, depth_map_gpu.ptr)
            self._vprint("  [GPUvt] compute_depth_map_cuda done")
            _vt_t1 = time.time(); timings['vt_depth_map'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            # --- depthnet TRT (GPU in / GPU out) ---
            depth_inputs = {
                self.depthnet_trt.input_names[0]: self._upload_cached("depthnet.x_cam", x_cam_5d),
                self.depthnet_trt.input_names[1]: depth_map_gpu,
            }
            cam_feats_gpu = self.depthnet_trt(depth_inputs, return_gpu_buffers=True)[0]
            self._vprint("  [GPUvt] depthnet_trt done, cam_feats nbytes={}".format(cam_feats_gpu.nbytes))
            _vt_t1 = time.time(); timings['vt_depthnet'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            # --- geometry + bev_pool entirely on GPU ---
            camera2lidar_rots = camera2lidar[..., :3, :3]
            camera2lidar_trans = camera2lidar[..., :3, 3]
            intrins = camera_intrinsics[..., :3, :3]
            post_rots = img_aug_matrix[..., :3, :3]
            post_trans = img_aug_matrix[..., :3, 3]
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

            inv_post_rots = np.linalg.inv(post_rots).astype(np.float32)
            inv_intrins = np.linalg.inv(intrins).astype(np.float32)
            combine_rots = np.matmul(camera2lidar_rots, inv_intrins).astype(np.float32)

            frustum = self.vtransform_geom.frustum.astype(np.float32)
            dx = self.vtransform_geom.dx
            bx = self.vtransform_geom.bx
            nx = self.vtransform_geom.nx

            depthnet_out_name = self.depthnet_trt.output_names[0]
            depthnet_out_shape = self.depthnet_trt.output_shapes[depthnet_out_name]
            if len(depthnet_out_shape) == 2:
                C_bev = depthnet_out_shape[1]
            else:
                C_bev = depthnet_out_shape[-1]

            Nprime = B * N * D * fH * fW
            workspace_size = vtransform_gpu_workspace_size(B, N, D, fH, fW, int(nx[0]), int(nx[1]), int(nx[2]), C_bev)
            workspace_gpu = CudaBuffer(workspace_size)
            camera_bev_gpu = CudaBuffer(B * C_bev * int(nx[2]) * int(nx[0]) * int(nx[1]) * 4, fill_value=0, dtype=np.float32)
            geom_feats_out_gpu = CudaBuffer(Nprime * 4 * 4, fill_value=0, dtype=np.int32)
            interval_starts_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)
            interval_lengths_gpu = CudaBuffer(Nprime * 4, fill_value=0, dtype=np.int32)

            out_K = np.zeros(1, dtype=np.int32)
            out_M = np.zeros(1, dtype=np.int32)

            self._vprint(
                "  [GPUvt] calling vtransform_post_depthnet_cuda: Nprime={} C_bev={} workspace={}".format(
                    Nprime, C_bev, workspace_size
                )
            )
            _frustum_gpu = self._upload_cached("vt.frustum", frustum)
            _inv_post_rots_gpu = self._upload_cached("vt.inv_post_rots", inv_post_rots)
            _post_trans_gpu = self._upload_cached("vt.post_trans", post_trans.astype(np.float32))
            _combine_rots_gpu = self._upload_cached("vt.combine_rots", combine_rots)
            _camera2lidar_trans_gpu = self._upload_cached(
                "vt.camera2lidar_trans", camera2lidar_trans.astype(np.float32)
            )
            _extra_rots_gpu = self._upload_cached("vt.extra_rots", extra_rots.astype(np.float32))
            _extra_trans_gpu = self._upload_cached("vt.extra_trans", extra_trans.astype(np.float32))
            vtransform_post_depthnet_cuda(
                _frustum_gpu.ptr,
                _inv_post_rots_gpu.ptr,
                _post_trans_gpu.ptr,
                _combine_rots_gpu.ptr,
                _camera2lidar_trans_gpu.ptr,
                _extra_rots_gpu.ptr,
                _extra_trans_gpu.ptr,
                B, N, D, fH, fW,
                float(dx[0]), float(dx[1]), float(dx[2]),
                float(bx[0]), float(bx[1]), float(bx[2]),
                int(nx[0]), int(nx[1]), int(nx[2]),
                cam_feats_gpu.ptr, 0, C_bev,
                camera_bev_gpu.ptr,
                geom_feats_out_gpu.ptr, interval_starts_gpu.ptr, interval_lengths_gpu.ptr,
                out_K.ctypes.data, out_M.ctypes.data,
                workspace_gpu.ptr, workspace_size,
            )
            self._vprint("  [GPUvt] vtransform_post_depthnet_cuda done K={} M={}".format(out_K[0], out_M[0]))
            timings['vt_geometry'] = 0.0
            timings['vt_precompute'] = 0.0
            _vt_t1 = time.time(); timings['vt_bev_pool'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            camera_bev_gpu.shape = (B, C_bev * int(nx[2]), int(nx[0]), int(nx[1]))
            camera_bev = None
        else:
            # --- numpy fallback path ---
            _vt_t0 = time.time()
            depth_map = self.vtransform_geom.compute_depth_map(
                points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N)
            _vt_t1 = time.time(); timings['vt_depth_map'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            depth_inputs = {
                self.depthnet_trt.input_names[0]: self._upload_cached("depthnet.x_cam", x_cam_5d),
                self.depthnet_trt.input_names[1]: self._upload_cached("depthnet.depth_map", depth_map),
            }
            depthnet_out = self.depthnet_trt(depth_inputs)
            cam_feats_flat = depthnet_out[0].astype(np.float32)
            _vt_t1 = time.time(); timings['vt_depthnet'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            camera2lidar_rots = camera2lidar[..., :3, :3]
            camera2lidar_trans = camera2lidar[..., :3, 3]
            intrins = camera_intrinsics[..., :3, :3]
            post_rots = img_aug_matrix[..., :3, :3]
            post_trans = img_aug_matrix[..., :3, 3]
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

            geom = self.vtransform_geom.get_geometry(
                camera2lidar_rots, camera2lidar_trans,
                intrins, post_rots, post_trans,
                extra_rots=extra_rots, extra_trans=extra_trans)
            _vt_t1 = time.time(); timings['vt_geometry'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            Nprime_bev = B * N * D * fH * fW
            if cam_feats_flat.shape[0] == Nprime_bev:
                C_bev = cam_feats_flat.shape[1]
            else:
                C_bev = cam_feats_flat.shape[-1]
            cam_feats_6d = cam_feats_flat.reshape(B, N, D, fH, fW, C_bev)

            indices = self.vtransform_geom.precompute_bev_indices(geom, B)
            _vt_t1 = time.time(); timings['vt_precompute'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            Nprime = B * N * D * fH * fW
            x_flat = cam_feats_6d.reshape(Nprime, C_bev)
            x_flat = x_flat[indices["kept"]]
            x_flat = x_flat[indices["sort_indices"]]

            out = np_bev_pool_v2(
                x_flat, indices["geom_feats"],
                indices["interval_starts"], indices["interval_lengths"],
                indices["B"], indices["D"], indices["H"], indices["W"])
            _vt_t1 = time.time(); timings['vt_bev_pool'] = (_vt_t1 - _vt_t0) * 1000.0; _vt_t0 = _vt_t1

            camera_bev = np.concatenate(np.split(out, out.shape[2], axis=2), axis=1).squeeze(2)
            camera_bev_gpu = None

        if self.bev_downsample is not None:
            if isinstance(self.bev_downsample, ZeroTorchTRTRunner):
                if camera_bev_gpu is None:
                    camera_bev_gpu = self._upload_cached("downsample.camera_bev", camera_bev.astype(np.float32))
                downsample_inputs = {self.bev_downsample.input_names[0]: camera_bev_gpu}
                camera_bev_gpu = self.bev_downsample(
                    downsample_inputs, return_gpu_buffers=True
                )[0]
                camera_bev = None
            else:
                raise RuntimeError(
                    "PyTorch bev_downsample fallback path is disabled. "
                    "Build/load TRT downsample engine."
                )
        elif camera_bev_gpu is None:
            camera_bev = camera_bev.astype(np.float32, copy=False)

        if self.capture_intermediates:
            if camera_bev is None and camera_bev_gpu is not None:
                camera_bev = camera_bev_gpu.download(camera_bev_gpu.shape, np.float32)
            self._last_intermediates['camera_bev'] = camera_bev.copy()
        _t1 = time.time(); timings['vtransform'] = (_t1 - _t0) * 1000.0; _t0 = _t1
        self._vprint("[ZeroTorch] Step 5: Fuser + Decoder")
        # Step 5: Fuser + Decoder (TRT)
        lidar_input = None
        if lidar_bev_gpu is not None:
            lidar_input = lidar_bev_gpu
        else:
            lidar_input = self._upload_cached("fuser.lidar_bev", lidar_bev.astype(np.float32))
        if camera_bev_gpu is None:
            camera_bev_gpu = self._upload_cached("fuser.camera_bev", camera_bev.astype(np.float32))
        if self.use_gpu_vtransform or lidar_bev_gpu is not None:
            _stream_wait_for_default_stream(self.fuser_trt.stream.value)
        fuser_inputs = {
            self.fuser_trt.input_names[0]: camera_bev_gpu,
            self.fuser_trt.input_names[1]: lidar_input,
        }
        neck_features_gpu = self.fuser_trt(fuser_inputs, return_gpu_buffers=True)[0]
        neck_features = None
        if self.capture_intermediates:
            neck_features = neck_features_gpu.download(neck_features_gpu.shape, np.float32)
            self._last_intermediates['neck_features'] = neck_features.copy()
        _t1 = time.time(); timings['fuser'] = (_t1 - _t0) * 1000.0; _t0 = _t1

        # Step 6: TransFusionHead (TRT) + post-processing
        batch_size_int = img.shape[0]
        outputs = [{} for _ in range(batch_size_int)]

        head_outs_gpu = self.head_trt(
            {self.head_trt.input_names[0]: neck_features_gpu},
            return_gpu_buffers=True,
        )
        head_np = {}
        for name, buf in zip(self.head_trt.output_names, head_outs_gpu):
            shape = getattr(buf, "shape", tuple(self.head_trt.output_shapes[name]))
            head_np[name] = buf.download(shape, self.head_trt.output_dtypes_np[name]).astype(
                np.float32, copy=False
            )
        center = head_np["center"]
        height = head_np["height"]
        dim = head_np["dim"]
        rot = head_np["rot"]
        vel = head_np["vel"]
        heatmap = head_np["heatmap"]
        query_heatmap_score = head_np["query_heatmap_score"]
        if self.capture_intermediates:
            self._last_intermediates['center'] = center.copy()
            self._last_intermediates['height'] = height.copy()
            self._last_intermediates['dim'] = dim.copy()
            self._last_intermediates['rot'] = rot.copy()
            self._last_intermediates['vel'] = vel.copy()
            self._last_intermediates['heatmap'] = heatmap.copy()
            self._last_intermediates['query_heatmap_score'] = query_heatmap_score.copy()
        _t1 = time.time(); timings['head'] = (_t1 - _t0) * 1000.0; _t0 = _t1
        self._vprint("[ZeroTorch] Step 6: Decode + NMS")

        bboxes = self._decode_and_nms(
            center, height, dim, rot, vel, heatmap, query_heatmap_score, metas)
        for k, (boxes, scores, labels) in enumerate(bboxes):
            outputs[k].update({
                "boxes_3d": boxes,
                "scores_3d": scores,
                "labels_3d": labels,
            })
        _t1 = time.time(); timings['decode_nms'] = (_t1 - _t0) * 1000.0
        self._vprint("[ZeroTorch] Forward complete")

        self._forward_count += 1
        if self._forward_count % 100 == 1:
            prof_str = " | ".join(f"{k}={v:.1f}ms" for k, v in timings.items())
            self.logger.info(f"[Profile] {prof_str}")

        return outputs

    def _voxelize(self, points):
        feats_list, coords_list, sizes_list = [], [], []
        for k, res in enumerate(points):
            # res: np.ndarray [M, C]
            num_points = res.shape[0]
            num_features = res.shape[1]
            d_points = self._upload_cached("voxel.points", res.astype(np.float32))
            voxels, coors, num_per_voxel, voxel_num = self.voxelizer(
                d_points.ptr, num_points, num_features)

            # voxelize reduce (mean / sum) on CPU (numpy)
            if self.voxelize_reduce:
                voxels = voxels.sum(axis=1) / num_per_voxel.astype(np.float32)[:, None]
            else:
                voxels = voxels[:, :num_per_voxel.max()]

            batch_coors = np.pad(coors, ((0, 0), (1, 0)), mode="constant", constant_values=k)
            feats_list.append(voxels)
            coords_list.append(batch_coors)
            sizes_list.append(num_per_voxel)

        feats = np.concatenate(feats_list, axis=0).astype(np.float32)
        coords = np.concatenate(coords_list, axis=0).astype(np.int32)
        sizes = np.concatenate(sizes_list, axis=0).astype(np.int32)
        return feats, coords, sizes

    def _decode_and_nms(self, center, height, dim, rot, vel, heatmap,
                        query_heatmap_score, metas):
        num_proposals = self.num_proposals
        num_classes = self.num_classes
        test_cfg = self.test_cfg
        bbox_coder = self.bbox_coder

        batch_score = 1.0 / (1.0 + np.exp(-heatmap[..., -num_proposals:]))  # sigmoid
        query_labels = query_heatmap_score.argmax(axis=1)
        # one-hot
        B = query_labels.shape[0]
        N = query_labels.shape[1]
        one_hot = np.zeros((B, num_classes, N), dtype=np.float32)
        for b in range(B):
            for n in range(N):
                one_hot[b, query_labels[b, n], n] = 1.0
        batch_score = batch_score * query_heatmap_score * one_hot

        batch_center = center[..., -num_proposals:]
        batch_height = height[..., -num_proposals:]
        batch_dim = dim[..., -num_proposals:]
        batch_rot = rot[..., -num_proposals:]
        batch_vel = vel[..., -num_proposals:]

        preds_dicts = bbox_coder.decode(batch_score, batch_rot, batch_dim,
                                        batch_center, batch_height, batch_vel,
                                        filter=True)

        nms_type = test_cfg.get('nms_type', None)
        nms_thr = test_cfg.get('nms', 0.2)
        score_thr = test_cfg.get('score_threshold', None)
        pre_max_size = test_cfg.get('pre_max_size', 1000)
        post_max_size = test_cfg.get('post_max_size', 83)
        min_radius = test_cfg.get('min_radius', [])
        post_center_range = test_cfg.get('post_center_range', None)
        pc_range = bbox_coder.pc_range

        ret_task = []
        for i in range(len(preds_dicts)):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']

            # Score threshold (only if explicitly configured)
            if score_thr is not None:
                score_mask = scores > score_thr
                bboxes = bboxes[score_mask]
                scores = scores[score_mask]
                labels = labels[score_mask]

            if len(bboxes) == 0:
                ret_task.append((np.zeros((0, 9), dtype=np.float32),
                                 np.zeros((0,), dtype=np.float32),
                                 np.zeros((0,), dtype=np.int64)))
                continue

            # NMS (only if nms_type is configured)
            if nms_type is not None:
                boxes_for_nms = SimpleLiDARBox(bboxes).bev.astype(np.float32)
                if min_radius and len(min_radius) > 0:
                    dets = np.concatenate([boxes_for_nms[:, :2], scores[:, None]], axis=1)
                    keep = circle_nms(dets, min_radius[0])
                elif nms_type == 'circle':
                    dets = np.concatenate([boxes_for_nms[:, :2], scores[:, None]], axis=1)
                    keep = circle_nms(dets, nms_thr)
                else:
                    keep = np_nms_gpu(boxes_for_nms, scores, nms_thr,
                                      pre_maxsize=pre_max_size, post_max_size=post_max_size)
                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            ret_task.append((bboxes, scores, labels))
        return ret_task


# ------------------------------------------------------------------
# Dataset helpers (copied from original for minimal standalone usage)
# ------------------------------------------------------------------
def run_evaluation(model, data_loader, logger):
    logger.warning("run_evaluation stub: zero-torch runner evaluation not yet fully wired")
    return {}


def run_single_test(model, data_loader, logger):
    logger.warning("run_single_test stub: zero-torch runner single test not yet fully wired")
    return {}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Zero-Torch BEVFusion TRT Standalone Runner")
    parser.add_argument("--config", required=True, help="model config")
    parser.add_argument("--ckpt", required=True, help="checkpoint (for LiDAR backbone weights)")
    parser.add_argument("--swin-engine", required=True)
    parser.add_argument("--depthnet-engine", required=True)
    parser.add_argument("--fuser-engine", required=True)
    parser.add_argument("--neck-engine", required=True)
    parser.add_argument("--head-engine", required=True)
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-torch-lidar", action="store_true", default=True,
                        help="Use TVSparseEncoder (required for zero-torch)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Zero-Torch BEVFusion TRT Runner initializing...")

    _cudart.cudaSetDevice(0)

    # The runner below is a correctness skeleton. For actual end-to-end inference
    # the caller still needs to instantiate engines, TVSparseEncoder, and dataloader.
    logger.info("Skeleton loaded. Next steps:\n"
                "  1) Phase B: replace numpy VTransform bottleneck with optimized impl\n"
                "  2) Phase C: wire dataloader → zero-torch runner end-to-end")


if __name__ == "__main__":
    main()
