"""
Build TensorRT engine from ONNX with configurable precision and target SM.
Supports INT8/FP16 and can target SM 8.7 for Jetson Orin from x86 build hosts.

Usage example:
    python tools/build_trt_engine.py \
        --onnx artifacts/vtransform_depthnet_int8.onnx \
        --engine vtransform_depthnet_int8_sm87.engine \
        --fp16 --int8 --workspace 4096 \
        --timing-cache timing.cache
"""
import argparse
import os
import sys

import tensorrt as trt


class TrtLogger(trt.Logger):
    def __init__(self):
        super().__init__(trt.Logger.INFO)

    def log(self, severity, msg):
        if severity <= self.severity:
            print(f"[TRT] {msg}")


def build_engine(onnx_path, engine_path, fp16=False, int8=False, workspace_mb=4096,
                 timing_cache_path=None, hardware_compat=False):
    logger = TrtLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    print(f"ONNX parsed: {onnx_path}")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Enabled FP16")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("  Enabled INT8")

    # Hardware compatibility within Ampere family (SM 8.x) can help x86-built
    # engines run on Orin (SM 8.7) without rebuilding on the device itself.
    # Available in TensorRT 10.x as a preview feature.
    if hardware_compat:
        try:
            config.set_preview_feature(trt.PreviewFeature.HARDWARE_COMPATIBLE_AMAX, True)
            print("  Enabled Hardware Compatibility (Ampere)")
        except AttributeError:
            print("  WARNING: HARDWARE_COMPATIBLE_AMAX not available in this TRT version")

    if timing_cache_path and os.path.exists(timing_cache_path):
        with open(timing_cache_path, "rb") as f:
            timing_cache = config.create_timing_cache(f.read())
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
            print(f"  Loaded timing cache: {timing_cache_path}")

    # Builder optimization level (5 = max)
    config.builder_optimization_level = 5

    print("Building engine... (this may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    file_size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"Engine saved: {engine_path} ({file_size_mb:.2f} MB)")

    # Save timing cache for incremental builds
    if timing_cache_path:
        timing_cache = config.get_timing_cache()
        if timing_cache is not None:
            with open(timing_cache_path, "wb") as f:
                f.write(timing_cache.serialize())
            print(f"Timing cache saved: {timing_cache_path}")
        else:
            print("  Timing cache not available from this builder config")


def main():
    parser = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--engine", required=True, help="Output engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--int8", action="store_true", help="Enable INT8")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace MB")
    parser.add_argument("--timing-cache", type=str, default="timing.cache")
    parser.add_argument("--hardware-compat", action="store_true",
                        help="Enable Ampere hardware compatibility for Orin")
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace_mb=args.workspace,
        timing_cache_path=args.timing_cache,
        hardware_compat=args.hardware_compat,
    )


if __name__ == "__main__":
    main()
