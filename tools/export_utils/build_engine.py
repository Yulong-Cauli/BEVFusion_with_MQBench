"""
替代 trtexec 的引擎构建工具，适配 TRT Python API 10.15。
用法：
    python tools/export_utils/build_engine.py \
        --onnx swin_int8.onnx --engine swin_int8.engine --int8 --fp16

    python tools/export_utils/build_engine.py \
        --onnx lidar_backbone.onnx --engine lidar_backbone.engine --fp16 \
        --plugins tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
"""
import argparse, ctypes, os
import tensorrt as trt


def build_engine(onnx_path, engine_path,
                 use_int8=False, use_fp16=False,
                 plugin_paths=None, workspace_gb=4):

    if plugin_paths:
        for p in plugin_paths:
            ctypes.CDLL(p)
            print(f"[Plugin] 已加载: {p}")

    logger  = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    config  = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Q/DQ 模式：scale 已内嵌在 ONNX 中，不需要额外 calibrator

    print(f"[ONNX] 解析: {onnx_path}")
    with open(onnx_path, "rb") as f:
        success = parser.parse(f.read())

    if not success:
        print("ERROR: ONNX 解析失败：")
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        return False

    print(f"ONNX 解析成功，共 {network.num_layers} 层")
    print("构建引擎中（首次约 5~15 分钟）...")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: 引擎构建失败，请查看上方 VERBOSE 日志")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"引擎已保存: {engine_path}  "
          f"({os.path.getsize(engine_path)/1024/1024:.1f} MB)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",      required=True)
    p.add_argument("--engine",    required=True)
    p.add_argument("--int8",      action="store_true")
    p.add_argument("--fp16",      action="store_true")
    p.add_argument("--plugins",   default="", help="逗号分隔的 .so 路径")
    p.add_argument("--workspace", type=int, default=4, help="显存 GB，默认 4")
    args = p.parse_args()
    plugins = [x.strip() for x in args.plugins.split(",") if x.strip()]
    ok = build_engine(args.onnx, args.engine,
                      args.int8, args.fp16, plugins, args.workspace)
    exit(0 if ok else 1)

if __name__ == "__main__":
    main()
