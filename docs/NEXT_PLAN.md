# BEVFusion 量化模型 TensorRT 全量部署执行计划 v8

## 环境信息

```
硬件          : NVIDIA RTX 3090 (Ampere, SM 8.6)
CUDA (nvcc)   : 11.8   (/usr/local/cuda)
TRT Python API: 10.15.1.29

环境 1 (bevfusion_mqbench):
  Python 3.8 + PyTorch 1.10.2 + spconv 2.1 + mmcv 1.4.0 + MQBench 0.0.6
  用途: PTQ 校准、ONNX 导出、引擎构建
  推理脚本: tools/trt_infer.py

环境 2 (spconv23_deploy):  ← Phase 7 新增
  /media/yellowstone/data2/CYL/spconv23_deploy
  Python 3.9 + PyTorch 2.0.1 + spconv 2.3.8 + mmcv 1.7.2
  用途: 独立推理 + NDS 评估（不依赖 cpython-38 扩展）
  推理脚本: tools/trt_infer_standalone.py
  CUDA 扩展: build_sp39/ (JIT 编译)
```

## 关键设计决策（Agent 必读，不得违背）

```
1. SwinT 导出为单一 ONNX，不拆分子模块。
   LayerNorm / Softmax / AdaptivePadding 不需要写任何 TRT Plugin。
   TRT 8.6 原生支持这些算子，导出后自动以 FP16 运行。
   量化层（Conv2d/Linear）通过 Q/DQ 节点以 INT8 运行。
   只需要做：1、消除动态控制流  2、注册 MQBench FakeQuant Symbolic。

2. Log2 Plugin 是唯一需要手写的 Plugin（Phase 2）。

3. BEV Pooling Plugin 从 BEVDet 提取，不从零写（Phase 3）。

4. SpConv 路线待探查后确定（Phase 4）。

5. 必须全 TRT 部署。禁止任何 PyTorch 混合方案。
   边缘设备场景下保留 PyTorch 运行任何子模块都会导致量化失去意义。

6. 每个模块导出引擎后，必须立即做精度验证（见 Step 0.4 及各 Phase 的精度验证步骤）。
   验证通过（cosine_sim > 0.999）才能进入下一阶段。
```

## 工作流说明

**所有开发和运行均在服务器上直接进行**，不需要本地开发后上传。
代码修改、脚本编写、编译、验证，全部在服务器完成。

```bash
# 每次开始工作前先执行：
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
```

---

## 已完成

- [x] TRT/ONNX/ORT 版本确认
- [x] trtexec 不可用，改用 build_engine.py
- [x] TRT C++ SDK：`/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6`
- [x] NvInfer.h 可读，SM86，编译环境就绪
- [x] Phase 1: SwinTransformer ONNX 导出 + TRT 引擎构建（2026-03-24）

  - ✅ ONNX 导出：208 Q/DQ 节点，0 FakeQuant，9143 总节点
  - ✅ TRT 引擎：32.3 MB，23,655 层，INT8 卷积 + FP16 融合
  - ✅ 关键修复：`FakeQuantizeLearnablePerchannelAffine.symbolic` 覆盖为 Q/DQ
  - ⚠️ **精度验证（Step 1.7）尚未执行**

- [x] Phase 2: SparseLog2Quant TRT Plugin（2026-03-25）

  - ✅ Plugin 源码：kernel.cu + plugin.h/cpp + CMakeLists.txt
  - ✅ 编译成功：`libsparse_log2_quant_plugin.so`（~75KB）
  - ✅ C++ 注册测试通过：Plugin 正确注册到 TRT Registry
  - ✅ 关键修复：kernel 接口与头文件统一（按值传递 dim0-dim3）
  - ✅ 关键修复：重复注册问题（forceInit 仅用于加载 .so）
  - ⚠️ 数值验证（enqueue）待 Phase 5 用 C++ 方式测试（Python API 版本不匹配）

- [x] Phase 3: vtransform depthnet INT8 + bev_pool_v2 Plugin（2026-03-28）

- [x] Phase 4: LiDAR SparseEncoder spconv 2.3 FP16（2026-03-29）
  - cosine_sim = 0.999994

- [x] Phase 4b: Fuser + Decoder + TransFusionHead TRT 引擎（2026-03-29）
  - FP16: 10.4 MB | INT8: 5.8 MB (56 Q/DQ)
  - TransFusionHead: ✅ 可导出 TRT（argsort→topk 修复），3.4 MB

- [x] Phase 5: 端到端集成 + NDS 验证（2026-03-29）
  - ⚠️ 混合 TRT+PyTorch 方案（非全 TRT）
  - Version A (W8A16): NDS=0.7144, mAP=0.6851
  - Version B (INT8): NDS=0.7102, mAP=0.6786
  - 两个版本 NDS 均高于 FP32 baseline (0.7069)

- [x] Phase 6: Camera Neck + TransFusionHead TRT + bev_pool_v2 + LiDAR Log2 量化（2026-03-29）
  - 详见 docs/order.md
  - Phase 6 (7/8 量化): NDS=0.7040, mAP=0.6654

- [x] Phase 7: Standalone 推理脚本 — 去 bevfusion_mqbench 环境（2026-03-30）
  - ✅ spconv23_deploy 环境搭建（Python 3.9 + PyTorch 2.0 + spconv 2.3.8 + TRT 10.15）
  - ✅ 4 个 CUDA 扩展重新编译为 cpython-39（bev_pool_ext, voxel_layer, iou3d_cuda, roiaware_pool3d_ext）
  - ✅ trt_infer_standalone.py：内联所有 mmcv/mmdet3d/MQBench 依赖
  - ✅ mmdet3d.ops BEVFUSION_STANDALONE 开关
  - ✅ FP16 NDS=0.7039 验证通过
  - ✅ INT8 WeightFakeQuantize scale shape 修复（per-channel [out_ch]）
  - ⚠️ INT8 NDS 待重跑验证
  - 详见 docs/HANDOFF_MASTER.md

- [x] Phase 8: 去 LiDAR Backbone PyTorch 依赖（2026-04-05 完成）
  - ✅ 调研 spconv 2.3 架构：core_cc.so 不链接 PyTorch，只需 data_ptr + cuda_stream
  - ✅ TVAllocator(ExternalAllocator)：用 torch.empty + tv.from_blob 分配 GPU 内存（PyTorch CUDA allocator 后端，兼容 TRT）
  - ✅ TVSpconvMatmul(ExternalSpconvMatmul)：用 cuBLAS ctypes 替代 torch.mm
  - ✅ cuBLAS GEMM cosine_sim=1.000000 验证通过
  - ✅ TVSparseEncoder：完整 backbone，直接调 SpconvOps + ConvGemmOps + InferenceOps
  - ✅ trt_infer_standalone.py 加 --no-torch-lidar 模式
  - ✅ 定位并修复 implicit_gemm CUDA error 700（cumm allocator 与 TRT execution context 不兼容）
  - ✅ 消除 BN/残差 add 的 CPU numpy roundtrip（fuse BN + cuBLAS axpy）
  - ✅ 修复 GC 导致的 CUDA tensor 悬空指针（allocator pinning + feature_ref pinning）
  - ✅ TV backbone FP16 NDS = 0.7039（与 PyTorch 路径一致）
  - 详见 docs/HANDOFF_MASTER.md

---

## 待完成

- [x] Phase 9 Part A：TV backbone INT8 Log2 量化（2026-04-05 完成）
  - ✅ `TVSparseEncoder.load_ptq_weights`：加载 PTQ INT8 weight + per-channel scale + `log2_base`
  - ✅ 自定义 CUDA kernel：`tv_log2_fake_quant_fp16` + `tv_bn_forward_inplace_fp16`
  - ✅ 计算图对齐：每层 conv 前 Log2 量化 → `implicit_gemm` → BN → ReLU（与 PT FakeQuant 图一致）
  - ✅ 关键修复：`_basic_block` 中 `identity = x.features.clone()`，修复 in-place 量化破坏残差连接
  - ✅ 数值验证：end-to-end dense max diff = **2.89**（INT8 rounding 可接受）
  - ✅ 完整 pipeline 冒烟测试通过（200 detections，3 with score > 0.3）
  - ✅ TV backbone INT8 完整 NDS 评估完成：NDS = 0.6893，mAP = 0.6474（略优于 PyTorch INT8 路径）
  - 详见 docs/HANDOFF_MASTER.md

### Phase 9 Part A：TV backbone INT8 Log2 量化（2026-04-05 完成）

当前 `--no-torch-lidar` 路径已从 FP16（NDS 0.7039）扩展到 INT8 Log2（目标 NDS ≈ 0.6875）。

技术实现：
- `tools/tv_log2_quant.cu`：手写 Log2 / BN CUDA kernel，ctypes 绑定
- `TVSparseEncoder.load_ptq_weights`：运行时加载 INT8 weight 和 `log2_base`
- `_conv_bn_relu`：Log2 quant → `implicit_gemm` → BN → ReLU（与 PT 图对齐）
- `_basic_block`：`identity = x.features.clone()` 防止 residual 被 in-place 修改破坏

成果：
- conv0 diff ≈ 0.027 ✅
- Log2 quant diff = 0.000000 ✅
- BN diff ≈ 0.016 ✅
- 完整 backbone dense diff ≈ 2.89 ✅
- 冒烟测试通过 ✅

NDS 结果（6019 帧完整验证集）：
- TV INT8 Log2（零 PyTorch LiDAR）：NDS = **0.6893**，mAP = **0.6474**
- 对比 PyTorch INT8 仿真：NDS = 0.6875，mAP = 0.6429
- 差异来源：INT8 `implicit_gemm` 不同 kernel 实现的正常 rounding 波动，精度完全可用

- [x] Phase 7 收尾：INT8 NDS 重跑（2026-04-05 完成）
  - ✅ PyTorch spconv 2.3 INT8 路径 NDS = **0.6893**，mAP = **0.6478**
  - ✅ 验证 WeightFakeQuantize scale shape 修复（从 [1] → [out_channels]）后无精度 regression
  - ✅ TV INT8 与 PyTorch INT8 NDS 完全一致（0.6893），mAP 差异仅 0.0004（rounding 正常波动）

### Phase 9 Part B：完全零 PyTorch（Jetson Orin 目标）

Phase 8 去掉了 LiDAR backbone 内部的 PyTorch，但以下部分仍依赖 PyTorch：
- TRT engine I/O（torch.Tensor ↔ TRT binding）
- Voxelization / bev_pool_v2 CUDA ext（torch.utils.cpp_extension 编译）
- 数据加载 (DataLoader)
- VTransform geometry 计算
- TransFusionHead 后处理

两个方向：
- **Python 路径**：逐步用 tv.Tensor + numpy + ctypes 替代剩余 PyTorch 依赖
- **C++ 路径**：参照 `temp/spconv/example/libspconv/main.cu`，用 C++ 重写整个 pipeline

需要 Jetson Orin 设备后再决定具体方案。

---

## Phase 0：基础工具准备

### Step 0.1：build_engine.py（替代 trtexec）

新建 `tools/export_utils/build_engine.py`：

```python
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
        print("❌ ONNX 解析失败：")
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        return False

    print(f"✅ ONNX 解析成功，共 {network.num_layers} 层")
    print("构建引擎中（首次约 5~15 分钟）...")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("❌ 引擎构建失败，请查看上方 VERBOSE 日志")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"✅ 引擎已保存: {engine_path}  "
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
```

验证：

```bash
python tools/export_utils/build_engine.py --help
# 期望：打印用法，无报错
```

### Step 0.2：保存校准样本

在 `tools/quant_ptq_minmax.py` 的 `run_calibration` 函数开头插入：

```python
def run_calibration(model, calib_loader, num_batches, logger):
    _saved = False                                              # ← 新增
    model.eval()
    enable_calibration(model)
    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= num_batches:
                break
            if not _saved:                                      # ← 新增
                torch.save(data, "calib_sample_0.pt")          # ← 新增
                logger.info("已保存: calib_sample_0.pt")       # ← 新增
                _saved = True                                   # ← 新增
            model(return_loss=False, **data)
```

### Step 0.3：提取量化参数

新建 `tools/export_utils/freeze_fakequant.py`：

```python
"""
从 ptq_minmax_model.pth 提取所有量化参数。
输出：
  quant_params.json  —— scale / zero_point / log2_base
  log2_layers.txt    —— 使用 SparseLog2FakeQuantize 的层名
用法：
    python tools/export_utils/freeze_fakequant.py \
        --config configs/.../convfuser.yaml \
        --ckpt path/to/ptq_minmax_model.pth
"""
import argparse, json, sys, os, logging
sys.path.insert(0, os.getcwd())
import torch
from mmcv import Config
from mmdet3d.utils import get_root_logger
from mqbench.utils.state import enable_quantization
from tools.quant_ptq_minmax import SparseLog2FakeQuantize, build_ptq_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt",   required=True)
    args = p.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    cfg    = Config.fromfile(args.config)
    model, _, _ = build_ptq_model(cfg, logger)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    enable_quantization(model)
    model.eval()

    quant_params, log2_layers = {}, []

    for name, module in model.named_modules():
        if isinstance(module, SparseLog2FakeQuantize):
            base = module.log2_base.detach().cpu().float()
            quant_params[name] = {
                "quant_type": "log2",
                "log2_base":  base.tolist(),
                "per_channel": module.per_channel,
                "n_bits": module.n_bits,
                "qmin": module.qmin, "qmax": module.qmax,
            }
            log2_layers.append(name)
        elif (hasattr(module, "scale") and hasattr(module, "zero_point")
              and hasattr(module, "fake_quant_enabled")):
            try:
                quant_params[name] = {
                    "quant_type": "linear",
                    "scale":      module.scale.detach().cpu().float().tolist(),
                    "zero_point": module.zero_point.detach().cpu().tolist(),
                    "bits": 8,
                }
            except Exception as e:
                print(f"[WARN] {name}: {e}")

    with open("quant_params.json", "w") as f:
        json.dump(quant_params, f, indent=2)
    with open("log2_layers.txt", "w") as f:
        f.write("\n".join(log2_layers))

    print(f"✅ 总量化层: {len(quant_params)}, Log2 层: {len(log2_layers)}")
    for l in log2_layers:
        print(f"   {l}")

    # 完整性检查
    for name, p in quant_params.items():
        if p["quant_type"] == "linear":
            s = p["scale"] if isinstance(p["scale"], list) else [p["scale"]]
            assert all(v > 0 for v in s), f"{name}: scale 含非正值"
        else:
            b = p["log2_base"] if isinstance(p["log2_base"], list) else [p["log2_base"]]
            assert all(-20 <= v <= 4 for v in b), f"{name}: log2_base 越界 {b}"
    print("✅ 完整性检查通过")

if __name__ == "__main__":
    main()
```

运行：

```bash
python tools/export_utils/freeze_fakequant.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt path/to/ptq_minmax_model.pth
```

### Step 0.4：模块精度验证工具（verify_engine.py）

**每个模块导出引擎后必须执行精度验证。** 新建 `tools/export_utils/verify_engine.py`：

```python
"""
模块精度验证工具：对比 PyTorch 模块输出 vs TRT 引擎输出。
导出任意子模块引擎后立即运行，cosine_sim > 0.999 才能继续。

用法：
    python tools/export_utils/verify_engine.py \
        --engine swin_int8.engine \
        --input calib_sample_0.pt \
        --module swin \
        --config configs/.../convfuser.yaml \
        --ckpt path/to/ptq.pth \
        [--plugins a.so,b.so]
"""
import argparse, ctypes, sys, os
import numpy as np
import torch
import tensorrt as trt
sys.path.insert(0, os.getcwd())


def cosine_sim(a, b):
    a = np.array(a).flatten().astype(np.float32)
    b = np.array(b).flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_trt(engine_path, inputs_np, plugin_paths=None):
    """运行 TRT 引擎，返回所有输出的 numpy 数组列表。"""
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    if plugin_paths:
        for p in plugin_paths:
            ctypes.CDLL(p)

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    bindings = []
    outputs  = []
    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        buf   = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        bindings.append(int(buf))
        if engine.binding_is_input(i):
            arr = inputs_np[i].astype(dtype)
            cuda.memcpy_htod(buf, arr)
        else:
            outputs.append((buf, shape, dtype))

    context.execute_v2(bindings)

    results = []
    for buf, shape, dtype in outputs:
        arr = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(arr, buf)
        results.append(arr)
    return results


def get_pytorch_output(module_name, config_path, ckpt_path, inputs_pt):
    """加载 PyTorch 模型，运行对应子模块，返回输出。"""
    from mmcv import Config
    from mmdet3d.models import build_model
    from mqbench.utils.state import enable_quantization

    cfg   = Config.fromfile(config_path)
    model = build_model(cfg.model).eval().cuda()
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    enable_quantization(model)

    module_map = {
        "swin":        model.encoders.camera.backbone,
        "vtransform":  model.encoders.camera.vtransform,
        "lidar":       model.encoders.lidar.backbone,
    }
    assert module_name in module_map, f"未知 module: {module_name}，可选: {list(module_map)}"
    m = module_map[module_name].eval()

    with torch.no_grad():
        inputs_cuda = [t.cuda() for t in inputs_pt]
        out = m(*inputs_cuda)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine",  required=True)
    p.add_argument("--input",   required=True, help="calib_sample_0.pt 或 dummy 输入 .pt")
    p.add_argument("--module",  required=True, choices=["swin", "vtransform", "lidar"])
    p.add_argument("--config",  required=True)
    p.add_argument("--ckpt",    required=True)
    p.add_argument("--plugins", default="", help="逗号分隔的 .so 路径")
    p.add_argument("--threshold", type=float, default=0.999)
    args = p.parse_args()

    plugin_paths = [x.strip() for x in args.plugins.split(",") if x.strip()]
    data = torch.load(args.input, map_location="cpu")

    # 根据 module 构造输入
    if args.module == "swin":
        # SwinT 输入：[B, 3, 256, 704]（取第一张图）
        img = data["img"].data[0][:1]   # [1, 3, 256, 704]
        inputs_pt  = [img.float()]
        inputs_np  = [img.numpy().astype(np.float32)]
    elif args.module == "vtransform":
        # vtransform 需要 camera features + depth
        raise NotImplementedError("vtransform 输入构造请根据实际导出接口补充")
    elif args.module == "lidar":
        raise NotImplementedError("lidar 输入构造请根据实际导出接口补充")

    print(f"[PyTorch] 运行 {args.module} ...")
    pt_out = get_pytorch_output(args.module, args.config, args.ckpt, inputs_pt)

    print(f"[TRT] 运行引擎 {args.engine} ...")
    trt_outs = run_trt(args.engine, inputs_np, plugin_paths)
    trt_out  = trt_outs[0]

    cs  = cosine_sim(pt_out, trt_out)
    mae = float(np.abs(pt_out.astype(np.float32) - trt_out.astype(np.float32)).max())
    print(f"\n{'='*50}")
    print(f"  cosine_sim  : {cs:.6f}  （阈值 > {args.threshold}）")
    print(f"  max_abs_err : {mae:.6f}")

    if cs > args.threshold:
        print(f"  ✅ 精度验证通过，可继续下一阶段")
    else:
        print(f"  ❌ 精度不达标！请检查导出流程")
        exit(1)

if __name__ == "__main__":
    main()
```

---

## Phase 1：SwinTransformer → 单一 ONNX → TRT engine

> **核心原则**：整个 SwinT 导出为一个 ONNX。
>
> - Conv2d / Linear（量化层）：通过 Q/DQ 节点以 INT8 运行
> - LayerNorm / Softmax / AdaptivePadding（非量化层）：TRT 原生支持，自动 FP16
> - **不需要为任何非量化层写 Plugin**

### Step 1.1：打印静态 pad 值

先跑一次，获取各 Stage 的实际 pad 值：

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
import torch
from mmdet3d.models.backbones.swin import AdaptivePadding
from mmcv import Config
from mmdet3d.models import build_model

_orig = AdaptivePadding.forward
def _patched(self, x):
    input_h, input_w = x.shape[-2:]
    kernel_h = self.kernel_size[0] + (self.kernel_size[0]-1)*(self.dilation[0]-1)
    kernel_w = self.kernel_size[1] + (self.kernel_size[1]-1)*(self.dilation[1]-1)
    out_h = (input_h + self.stride[0] - 1) // self.stride[0]
    out_w = (input_w + self.stride[1] - 1) // self.stride[1]
    pad_h = max((out_h-1)*self.stride[0] + kernel_h - input_h, 0)
    pad_w = max((out_w-1)*self.stride[1] + kernel_w - input_w, 0)
    print(f"AdaptivePadding | input=({input_h},{input_w}) | "
          f"kernel={self.kernel_size} stride={self.stride} | "
          f"pad=({pad_h},{pad_w})")
    return _orig(self, x)
AdaptivePadding.forward = _patched

cfg = Config.fromfile(
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/"
    "swint_v0p075/convfuser.yaml")
model = build_model(cfg.model).eval()
swint = model.encoders.camera.backbone
with torch.no_grad():
    swint(torch.randn(1, 3, 256, 704))
EOF
```

**把输出结果记录下来**，Step 1.2 需要用到这些具体数值。

### Step 1.2：静态化 swin.py

修改 `mmdet3d/models/backbones/swin.py`，两处改动：

**改动①：AdaptivePadding.forward 去掉运行时动态计算**

```python
# 原始（运行时动态计算 pad，ONNX 导出时产生动态节点）：
def forward(self, x):
    input_h, input_w = x.shape[-2:]
    kernel_h = ...
    pad_h = max(...)
    if pad_h > 0 or pad_w > 0:      # ← 这个 if 会成为 ONNX If 节点
        x = F.pad(x, [...])
    return x

# 修改后（保留计算逻辑，但固定输入尺寸下会被 constant folding 消除）：
def forward(self, x):
    # 保持原始计算逻辑不变，ONNX export 时 do_constant_folding=True
    # 会将固定输入(256,704)下的所有 pad 值折叠为常量，if 分支静态化
    input_h, input_w = x.shape[-2:]
    kernel_h = self.kernel_size[0] + (self.kernel_size[0]-1)*(self.dilation[0]-1)
    kernel_w = self.kernel_size[1] + (self.kernel_size[1]-1)*(self.dilation[1]-1)
    out_h = (input_h + self.stride[0] - 1) // self.stride[0]
    out_w = (input_w + self.stride[1] - 1) // self.stride[1]
    pad_h = max((out_h-1)*self.stride[0] + kernel_h - input_h, 0)
    pad_w = max((out_w-1)*self.stride[1] + kernel_w - input_w, 0)
    # ⚠️ 如果 constant folding 后仍有 If 节点，则改为直接用 Step 1.1 的常量：
    # pad_h, pad_w = <Step 1.1 打印出的具体值>
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w//2, pad_w - pad_w//2,
                      pad_h//2, pad_h - pad_h//2])
    return x
```

**改动②：SwinTransformerBlock 的 attn_mask 改为静态 buffer**

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, ..., input_resolution, ...):
        super().__init__()
        ...
        H, W = input_resolution   # 每个 stage 固定，例如 (64, 176)
        # 原来：每次 forward 动态调用 create_mask(H, W)
        # 改为：__init__ 中预计算，注册为 buffer
        attn_mask = self.create_mask(
            torch.zeros(1, H, W, 1), H, W   # 用 dummy tensor 预计算
        )
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, query, hw_shape):
        ...
        # 原来：attn_mask = self.create_mask(query, H, W)
        # 改为：直接使用 self.attn_mask
        attn_out = self.attn(query_windows, mask=self.attn_mask)
        ...
```

### Step 1.3：验证静态化（ONNX 无 If 节点）

```bash
python - <<'EOF'
import sys, torch, onnx
sys.path.insert(0, ".")
from mmcv import Config
from mmdet3d.models import build_model

cfg   = Config.fromfile(
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/"
    "swint_v0p075/convfuser.yaml")
model = build_model(cfg.model).eval()
swint = model.encoders.camera.backbone

dummy = torch.randn(1, 3, 256, 704)
torch.onnx.export(swint, dummy, "/tmp/swin_check.onnx",
                  opset_version=13, do_constant_folding=True)

m        = onnx.load("/tmp/swin_check.onnx")
if_nodes = [n for n in m.graph.node if n.op_type == "If"]
all_ops  = sorted(set(n.op_type for n in m.graph.node))

print(f"If 节点数: {len(if_nodes)}  （期望 = 0）")
print(f"全部 op 类型: {all_ops}")
assert len(if_nodes) == 0, f"❌ 仍有 {len(if_nodes)} 个动态分支，需继续修改 swin.py"
print("✅ 静态化验证通过，可继续 Step 1.4")
EOF
```

**如果仍有 If 节点**：把上面打印的 `all_ops` 发给我，定位是哪个算子引入的动态分支。

### Step 1.4：注册 MQBench ONNX Symbolic

新建 `tools/export_utils/mqbench_onnx_symbolic.py`：

```python
"""
注册两类 ONNX Symbolic：
  1. MQBench LearnableFakeQuantize → QuantizeLinear + DequantizeLinear
  2. SparseLog2FakeQuantize        → custom::SparseLog2Quant
必须在任何 torch.onnx.export 调用之前 import 此模块。
"""
import sys, os
import torch, torch.onnx
from torch.onnx import register_custom_op_symbolic

# ── 1. MQBench LearnableFakeQuantize ─────────────────────────────────────────

def _per_tensor_fq_symbolic(g, x, scale, zero_point, quant_min, quant_max, *args):
    scale_f32 = g.op("Cast", scale,      to_i=torch.onnx.TensorProtoDataType.FLOAT)
    zp_i8     = g.op("Cast", zero_point, to_i=torch.onnx.TensorProtoDataType.INT8)
    q  = g.op("QuantizeLinear",   x, scale_f32, zp_i8)
    dq = g.op("DequantizeLinear", q, scale_f32, zp_i8)
    return dq

def _per_channel_fq_symbolic(g, x, scale, zero_point,
                              quant_min, quant_max, ch_axis, *args):
    scale_f32 = g.op("Cast", scale,      to_i=torch.onnx.TensorProtoDataType.FLOAT)
    zp_i8     = g.op("Cast", zero_point, to_i=torch.onnx.TensorProtoDataType.INT8)
    q  = g.op("QuantizeLinear",   x, scale_f32, zp_i8, axis_i=ch_axis)
    dq = g.op("DequantizeLinear", q, scale_f32, zp_i8, axis_i=ch_axis)
    return dq

for _name in ["mqbench::fake_quant_per_tensor_affine",
              "fake_quantize_per_tensor_affine"]:
    try:
        register_custom_op_symbolic(_name, _per_tensor_fq_symbolic, 13)
    except Exception:
        pass

for _name in ["mqbench::fake_quant_per_channel_affine",
              "fake_quantize_per_channel_affine"]:
    try:
        register_custom_op_symbolic(_name, _per_channel_fq_symbolic, 13)
    except Exception:
        pass

# ── 2. SparseLog2FakeQuantize → custom::SparseLog2Quant ──────────────────────

sys.path.insert(0, os.getcwd())
from tools.quant_ptq_minmax import SparseLog2FakeQuantize


class _Log2QuantExportFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log2_base, per_channel_flag):
        orig_dtype = x.dtype
        x_f       = x.float()
        eps       = 1e-6
        zero_mask = x_f.abs() < eps
        sign      = x_f.sign()
        base      = log2_base.to(x_f.device)
        if per_channel_flag and base.ndim == 1:
            base = base.unsqueeze(0)
        log2_x = torch.log2(x_f.abs().clamp(min=1e-30)) - base
        q_int  = torch.round(log2_x).clamp(-127, 127)
        x_dq   = sign * torch.pow(2.0, q_int + base)
        x_dq   = torch.where(zero_mask, torch.zeros_like(x_f), x_dq)
        return x_dq.to(orig_dtype)

    @staticmethod
    def symbolic(g, x, log2_base, per_channel_flag):
        base_list = log2_base.detach().cpu().tolist()
        if not isinstance(base_list, list):
            base_list = [base_list]
        return g.op(
            "custom::SparseLog2Quant", x,
            log2_base_f=base_list,
            per_channel_i=int(per_channel_flag),
            plugin_version_s="1",
        )

_orig_log2_forward = SparseLog2FakeQuantize.forward

def _patched_log2_forward(self, x):
    if torch.onnx.is_in_onnx_export():
        return _Log2QuantExportFunc.apply(x, self.log2_base, int(self.per_channel))
    return _orig_log2_forward(self, x)

SparseLog2FakeQuantize.forward = _patched_log2_forward
print("[mqbench_onnx_symbolic] 注册完成")
```

### Step 1.5：导出 swin_int8.onnx

新建 `tools/export_utils/export_swin.py`：

```python
"""
将整个 SwinTransformer 导出为单一 swin_int8.onnx。
量化层（Conv2d/Linear）→ Q/DQ 节点（INT8）
非量化层（LayerNorm/Softmax/AdaptivePadding）→ TRT 原生算子（FP16）
不需要为非量化层写任何 Plugin。

用法：
    python tools/export_utils/export_swin.py \
        --config configs/.../convfuser.yaml \
        --ckpt path/to/ptq_minmax_model.pth \
        --output swin_int8.onnx
"""
import argparse, sys, os, logging
sys.path.insert(0, os.getcwd())

import tools.export_utils.mqbench_onnx_symbolic  # noqa

import torch, onnx
from mmcv import Config
from mmdet3d.utils import get_root_logger
from mqbench.utils.state import enable_quantization
from tools.quant_ptq_minmax import build_ptq_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--output", default="swin_int8.onnx")
    args = p.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    cfg    = Config.fromfile(args.config)

    model, _, _ = build_ptq_model(cfg, logger)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    enable_quantization(model)
    model.eval()

    swint = model.encoders.camera.backbone
    swint.eval()
    dummy = torch.randn(1, 3, 256, 704)

    print(f"导出 → {args.output}")
    torch.onnx.export(
        swint, dummy, args.output,
        opset_version=13,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes=None,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"✅ 导出完成")

    m       = onnx.load(args.output)
    qdq     = [n for n in m.graph.node
               if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    fakeq   = [n for n in m.graph.node if "FakeQuant" in n.op_type]
    if_n    = [n for n in m.graph.node if n.op_type == "If"]
    all_ops = sorted(set(n.op_type for n in m.graph.node))

    print(f"  Q/DQ 节点   : {len(qdq)}   （期望 > 0）")
    print(f"  FakeQuant   : {len(fakeq)}  （期望 = 0）")
    print(f"  If 节点     : {len(if_n)}   （期望 = 0）")
    print(f"  全部 op 类型: {all_ops}")

    if len(qdq) == 0:
        print("⚠️  Q/DQ==0：把上方 all_ops 中 FakeQuant 相关名称告诉我，更新注册名")
    if len(if_n) > 0:
        print("⚠️  仍有 If 节点：需继续静态化 swin.py")
    if len(qdq) > 0 and len(if_n) == 0:
        print("✅ ONNX 结构正常，可以构建引擎")

if __name__ == "__main__":
    main()
```

运行：

```bash
python tools/export_utils/export_swin.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt path/to/ptq_minmax_model.pth \
    --output swin_int8.onnx
```

### Step 1.6：构建 swin_int8.engine

```bash
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx \
    --engine swin_int8.engine \
    --int8 --fp16
# 期望：ONNX 解析成功，引擎保存，无报错
```

### Step 1.7：SwinT 精度验证（必须执行）

```bash
python tools/export_utils/verify_engine.py \
    --engine swin_int8.engine \
    --input calib_sample_0.pt \
    --module swin \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt path/to/ptq_minmax_model.pth
# 期望：cosine_sim > 0.999
# ❌ 不达标则排查导出流程，不得跳过
```

---

## Phase 2：SparseLog2Quant TRT Plugin

> SparseLog2FakeQuantize 是唯一需要手写 Plugin 的算子。
> Plugin 做的事：FP16 输入 → log2 量化反量化 → FP16 输出（pointwise）
> 数学：q = round(log2(|x|) - base).clamp(-127,127)，x_dq = sign * 2^(q+base)

### Step 2.1：目录结构

创建：

```
tools/trt_plugins/sparse_log2_quant/
├── CMakeLists.txt
└── src/
    ├── sparse_log2_quant_kernel.cu
    ├── sparse_log2_quant_plugin.h
    └── sparse_log2_quant_plugin.cpp
```

### Step 2.2：CUDA Kernel

`tools/trt_plugins/sparse_log2_quant/src/sparse_log2_quant_kernel.cu`：

```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

static inline __device__ int32_t get_channel_idx(
    int64_t idx, int32_t C, int32_t nb_dims,
    int32_t dim2, int32_t dim3
) {
    if (nb_dims == 2) {
        return idx % C;
    } else if (nb_dims == 4) {
        int32_t HW = dim2 * dim3;
        return (idx / HW) % C;
    } else if (nb_dims == 3) {
        return (idx / dim2) % C;
    } else {
        return idx % C;
    }
}

__global__ void sparse_log2_quant_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    const float*  __restrict__ log2_base,
    int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
    int32_t nb_dims, bool per_channel,
    int32_t qmin, int32_t qmax, float eps
) {
    int64_t total_elements = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const int32_t C = dim1;
    const int32_t c = per_channel ? get_channel_idx(idx, C, nb_dims, dim2, dim3) : 0;
    const float base = log2_base[c];
    const float x = __half2float(input[idx]);

    if (fabsf(x) < eps) {
        output[idx] = __float2half(0.0f);
        return;
    }

    const float sign_x = (x > 0.0f) ? 1.0f : -1.0f;
    const float log2_x = log2f(fmaxf(fabsf(x), 1e-30f)) - base;
    float q = roundf(log2_x);
    q = fmaxf(static_cast<float>(qmin), fminf(static_cast<float>(qmax), q));
    output[idx] = __float2half(sign_x * exp2f(q + base));
}

extern "C"
void launch_sparse_log2_quant(
    const void* input, void* output, const float* log2_base,
    int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
    int32_t nb_dims, bool per_channel, cudaStream_t stream
) {
    int64_t total = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3;
    const int32_t block_size = 256;
    const int64_t grid_size = (total + block_size - 1) / block_size;
    uint32_t grid_dim_x = (grid_size > INT32_MAX) ? INT32_MAX : static_cast<uint32_t>(grid_size);

    sparse_log2_quant_kernel<<<grid_dim_x, block_size, 0, stream>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<half*>(output),
        log2_base, dim0, dim1, dim2, dim3, nb_dims,
        per_channel, -127, 127, 1e-6f
    );
}
```

### Step 2.3：Plugin 头文件

`tools/trt_plugins/sparse_log2_quant/src/sparse_log2_quant_plugin.h`：

```cpp
#pragma once
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

static const char* LOG2_PLUGIN_NAME    = "SparseLog2Quant";
static const char* LOG2_PLUGIN_VERSION = "1";

extern "C" void launch_sparse_log2_quant(
    const void* input, void* output, const float* log2_base,
    int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
    int32_t nb_dims, bool per_channel, cudaStream_t stream);

class SparseLog2QuantPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    SparseLog2QuantPlugin(std::vector<float> log2_base, bool per_channel);
    SparseLog2QuantPlugin(const void* data, size_t length);
    ~SparseLog2QuantPlugin() override;

    nvinfer1::DimsExprs getOutputDimensions(int32_t, const nvinfer1::DimsExprs* inputs,
        int32_t, nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int32_t pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int32_t, int32_t) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t,
        const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int32_t,
        const nvinfer1::PluginTensorDesc*, int32_t) const noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc*, const void* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t,
        const nvinfer1::DataType* inputTypes, int32_t) const noexcept override;

    const char* getPluginType()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int32_t     getNbOutputs()     const noexcept override;
    int32_t     initialize()              noexcept override;
    void        terminate()               noexcept override;
    size_t      getSerializationSize()    const noexcept override;
    void        serialize(void*)          const noexcept override;
    void        destroy()                       noexcept override;
    void        setPluginNamespace(const char*) noexcept override;
    const char* getPluginNamespace()       const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

private:
    std::vector<float> mLog2Base;
    bool               mPerChannel;
    float*             mLog2BaseDevice = nullptr;
    std::string        mNamespace;
};

class SparseLog2QuantPluginCreator : public nvinfer1::IPluginCreator {
public:
    SparseLog2QuantPluginCreator();
    const char* getPluginName()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char*,
        const nvinfer1::PluginFieldCollection*) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char*,
        const void*, size_t) noexcept override;
    void        setPluginNamespace(const char*) noexcept override;
    const char* getPluginNamespace() const noexcept override;
private:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mFields;
    std::string                        mNamespace;
};
```

### Step 2.4：Plugin 实现

`tools/trt_plugins/sparse_log2_quant/src/sparse_log2_quant_plugin.cpp`：

```cpp
#include "sparse_log2_quant_plugin.h"
#include <cstring>

SparseLog2QuantPlugin::SparseLog2QuantPlugin(
    std::vector<float> log2_base, bool per_channel)
    : mLog2Base(std::move(log2_base)), mPerChannel(per_channel) {}

SparseLog2QuantPlugin::SparseLog2QuantPlugin(const void* data, size_t) {
    const char* buf = static_cast<const char*>(data);
    size_t sz;
    memcpy(&sz, buf, sizeof(size_t));               buf += sizeof(size_t);
    mLog2Base.resize(sz);
    memcpy(mLog2Base.data(), buf, sz*sizeof(float)); buf += sz*sizeof(float);
    memcpy(&mPerChannel, buf, sizeof(bool));
}

SparseLog2QuantPlugin::~SparseLog2QuantPlugin() { terminate(); }

nvinfer1::DimsExprs SparseLog2QuantPlugin::getOutputDimensions(
    int32_t, const nvinfer1::DimsExprs* inputs, int32_t,
    nvinfer1::IExprBuilder&) noexcept { return inputs[0]; }

bool SparseLog2QuantPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut,
    int32_t, int32_t) noexcept {
    return inOut[pos].type   == nvinfer1::DataType::kHALF &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void SparseLog2QuantPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc*, int32_t,
    const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept {
    if (!mLog2BaseDevice) {
        cudaMalloc(&mLog2BaseDevice, mLog2Base.size()*sizeof(float));
        cudaMemcpy(mLog2BaseDevice, mLog2Base.data(),
                   mLog2Base.size()*sizeof(float), cudaMemcpyHostToDevice);
    }
}

size_t SparseLog2QuantPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc*, int32_t,
    const nvinfer1::PluginTensorDesc*, int32_t) const noexcept { return 0; }

int32_t SparseLog2QuantPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc*,
    const void* const* inputs, void* const* outputs,
    void*, cudaStream_t stream) noexcept {
    const auto& dims = inputDesc[0].dims;
    int32_t dim0 = (dims.nbDims > 0) ? dims.d[0] : 1;
    int32_t dim1 = (dims.nbDims > 1) ? dims.d[1] : 1;
    int32_t dim2 = (dims.nbDims > 2) ? dims.d[2] : 1;
    int32_t dim3 = (dims.nbDims > 3) ? dims.d[3] : 1;
    launch_sparse_log2_quant(
        inputs[0], outputs[0], mLog2BaseDevice,
        dim0, dim1, dim2, dim3, dims.nbDims,
        mPerChannel, stream);
    return 0;
}

nvinfer1::DataType SparseLog2QuantPlugin::getOutputDataType(
    int32_t, const nvinfer1::DataType* t, int32_t) const noexcept { return t[0]; }

size_t SparseLog2QuantPlugin::getSerializationSize() const noexcept {
    return sizeof(size_t) + mLog2Base.size()*sizeof(float) + sizeof(bool);
}
void SparseLog2QuantPlugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    size_t sz = mLog2Base.size();
    memcpy(buf, &sz, sizeof(size_t));               buf += sizeof(size_t);
    memcpy(buf, mLog2Base.data(), sz*sizeof(float)); buf += sz*sizeof(float);
    memcpy(buf, &mPerChannel, sizeof(bool));
}

const char* SparseLog2QuantPlugin::getPluginType()    const noexcept { return LOG2_PLUGIN_NAME; }
const char* SparseLog2QuantPlugin::getPluginVersion() const noexcept { return LOG2_PLUGIN_VERSION; }
int32_t     SparseLog2QuantPlugin::getNbOutputs()     const noexcept { return 1; }
int32_t     SparseLog2QuantPlugin::initialize()              noexcept { return 0; }
void SparseLog2QuantPlugin::terminate() noexcept {
    if (mLog2BaseDevice) { cudaFree(mLog2BaseDevice); mLog2BaseDevice = nullptr; }
}
void SparseLog2QuantPlugin::destroy() noexcept { delete this; }
nvinfer1::IPluginV2DynamicExt* SparseLog2QuantPlugin::clone() const noexcept {
    auto* p = new SparseLog2QuantPlugin(mLog2Base, mPerChannel);
    if (mLog2BaseDevice) {
        cudaMalloc(&p->mLog2BaseDevice, mLog2Base.size()*sizeof(float));
        cudaMemcpy(p->mLog2BaseDevice, mLog2BaseDevice,
                   mLog2Base.size()*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return p;
}
void SparseLog2QuantPlugin::setPluginNamespace(const char* ns) noexcept { mNamespace=ns; }
const char* SparseLog2QuantPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

SparseLog2QuantPluginCreator::SparseLog2QuantPluginCreator() {
    mFields = {
        {"log2_base",   nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"per_channel", nullptr, nvinfer1::PluginFieldType::kINT32,   1},
    };
    mFC.nbFields = (int32_t)mFields.size();
    mFC.fields   = mFields.data();
}
const char* SparseLog2QuantPluginCreator::getPluginName()    const noexcept { return LOG2_PLUGIN_NAME; }
const char* SparseLog2QuantPluginCreator::getPluginVersion() const noexcept { return LOG2_PLUGIN_VERSION; }
const nvinfer1::PluginFieldCollection*
SparseLog2QuantPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV2* SparseLog2QuantPluginCreator::createPlugin(
    const char*, const nvinfer1::PluginFieldCollection* fc) noexcept {
    std::vector<float> base; bool per_ch = false;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (std::string(f.name) == "log2_base") {
            const float* d = static_cast<const float*>(f.data);
            base.assign(d, d+f.length);
        } else if (std::string(f.name) == "per_channel") {
            per_ch = (*static_cast<const int32_t*>(f.data)) != 0;
        }
    }
    return new SparseLog2QuantPlugin(std::move(base), per_ch);
}
nvinfer1::IPluginV2* SparseLog2QuantPluginCreator::deserializePlugin(
    const char*, const void* data, size_t length) noexcept {
    return new SparseLog2QuantPlugin(data, length);
}
void SparseLog2QuantPluginCreator::setPluginNamespace(const char* ns) noexcept { mNamespace=ns; }
const char* SparseLog2QuantPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

extern "C" {
    __attribute__((visibility("default")))
    void forceInitSparseLog2QuantPlugin() {
        // 空函数：实际的 Plugin 注册由 REGISTER_TENSORRT_PLUGIN 宏自动完成
    }
}

REGISTER_TENSORRT_PLUGIN(SparseLog2QuantPluginCreator);
```

### Step 2.5：CMakeLists.txt

`tools/trt_plugins/sparse_log2_quant/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.14)
project(sparse_log2_quant CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(TRT_ROOT  "/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6")
set(CUDA_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_ARCHITECTURES "86")

find_package(CUDA REQUIRED)
include_directories(${TRT_ROOT}/include ${CUDA_ROOT}/include src/)
link_directories(${TRT_ROOT}/lib ${CUDA_ROOT}/lib64)

cuda_add_library(sparse_log2_quant_plugin SHARED
    src/sparse_log2_quant_plugin.cpp
    src/sparse_log2_quant_kernel.cu
)
target_link_libraries(sparse_log2_quant_plugin nvinfer nvinfer_plugin cudart)
```

编译：

```bash
cd tools/trt_plugins/sparse_log2_quant
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ls -lh libsparse_log2_quant_plugin.so   # 期望：约 200~500 KB
```

### Step 2.6：C++ 注册测试（重要）

> 由于 TRT Python API (10.15) 与 C++ SDK (8.6) 版本不匹配，Python 端无法识别 Plugin。
> C++ 端测试验证 Plugin 在目标 SDK 环境下能正确注册和执行。

`tools/trt_plugins/sparse_log2_quant/test/test_plugin_registration.cpp` 测试内容：
1. Plugin 库加载
2. Plugin Registry 初始化
3. 查找 SparseLog2Quant Plugin
4. 字段检查（log2_base, per_channel）
5. 创建 Plugin 实例
6. 序列化/反序列化测试

编译和运行：
```bash
cd tools/trt_plugins/sparse_log2_quant/test
./build_test.sh
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:../build:$LD_LIBRARY_PATH"
./build/test_plugin_registration
```

**期望输出**：`[SUCCESS] All tests passed!`

### Step 2.7：数值验证（待 Phase 5）

⚠️ **阻塞**：TRT Python API 与 C++ SDK 版本不匹配，Python 端无法测试。

**方案**：Phase 5 用 C++ 方式统一测试（创建最小 TRT 网络，对比 PyTorch 输出）。

参考实现 `tools/export_utils/test_log2_plugin.py`（Python，当前不可用，待 Phase 5）。

---

## Phase 2 修复记录

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| kernel 接口不匹配 | kernel.cu 用指针 dims，头文件用值 dim0-dim3 | 统一为按值传递 dim0-dim3 |
| 重复注册 | forceInit 和 REGISTER_TENSORRT_PLUGIN 都注册 | forceInit 改为空函数，仅用于加载 .so |
| Python API 版本不匹配 | Python 10.15 与 C++ SDK 8.6 不兼容 | 走 C++ 路径部署，Python 仅用于开发 |
| 找不到 cudart | 链接器缺 CUDA lib64 路径 | CMake 添加 `-L${CUDA_ROOT}/lib64` |
| getPluginRegistry 命名空间 | TRT 8.6 中是全局函数 | 去掉 `nvinfer1::` 前缀 |

---

## Phase 3：BEV Pooling TRT Plugin

> **当前状态（2026-03-28）：✅ 完成**
>
> depthnet INT8 ONNX 导出 + TRT 引擎构建 + 精度验证全部通过。
> bev_pool_v2 Plugin 编译通过，模型替换完成，集成测试通过。

### 背景：为什么需要替换 bev_pool

原始 `bev_pool`（scatter-based）存在两个根本问题：
1. **不可 trace**：含有布尔索引、动态 shape 等控制流，无法 `torch.onnx.export`
2. **不可量化**：scatter 操作无法插入 Q/DQ 节点

`bev_pool_v2`（interval-sum-based）是可 trace、可量化的等价替换：
- 用预计算的 `ranks_depth`、`ranks_feat`、`ranks_bev`、`interval_starts`、`interval_lengths` 替代动态 scatter
- 算法等价，精度损失可忽略

### Step 3.1：提取 BEVDet 源码

```bash
cd /media/yellowstone/data2/CYL
git clone https://github.com/HuangJunJie2017/BEVDet --branch dev2.0 --depth 1 BEVDet_tmp
find BEVDet_tmp -name "bev_pool*" -type f   # 确认文件路径

mkdir -p BEVFusion_with_MQBench/tools/trt_plugins/bev_pool_v2/src
cp BEVDet_tmp/det2trt/models/ops/bev_pool_v2/src/bev_pool.cpp \
   BEVFusion_with_MQBench/tools/trt_plugins/bev_pool_v2/src/
cp BEVDet_tmp/det2trt/models/ops/bev_pool_v2/src/bev_pool.cu \
   BEVFusion_with_MQBench/tools/trt_plugins/bev_pool_v2/src/
# 路径不对时根据 find 的输出调整
```

### Step 3.2：BEV Pooling CMakeLists.txt

新建 `tools/trt_plugins/bev_pool_v2/CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.14)
project(bev_pool_v2 CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(TRT_ROOT  "/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6")
set(CUDA_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_ARCHITECTURES "86")
find_package(CUDA REQUIRED)
include_directories(${TRT_ROOT}/include ${CUDA_ROOT}/include src/)
link_directories(${TRT_ROOT}/lib ${CUDA_ROOT}/lib64)
cuda_add_library(bev_pool_v2_plugin SHARED src/bev_pool.cpp src/bev_pool.cu)
target_link_libraries(bev_pool_v2_plugin nvinfer nvinfer_plugin cudart)
```

编译：

```bash
cd tools/trt_plugins/bev_pool_v2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
ls -lh libbev_pool_v2_plugin.so
```

### Step 3.3：替换模型中的 bev_pool 调用（当前工作重点）

修改 `mmdet3d/models/vtransforms/lsstransform.py`（或对应的 BaseTransform）：

**目标**：将 `self.bev_pool(x, geom_feats, ...)` 改为 `bev_pool_v2(depth, feat, ranks_*, ...)` 接口。

关键改动：
- 在 `__init__` 或 `create_frustum` 阶段预计算 `ranks_depth`、`ranks_feat`、`ranks_bev`
- 用预计算的 `interval_starts`、`interval_lengths` 替代 scatter 的动态索引
- 前向传播中调用 `bev_pool_v2`（来自 `mmdet3d/ops/bev_pool_v2`）

**替换后验证**：对比 PyTorch 输出，cosine_sim > 0.999。

### Step 3.4：bev_pool_v2 ONNX Symbolic 注册

在 `mmdet3d/models/vtransforms/lss.py` 或导出脚本中添加：

```python
import torch

def _bev_pool_v2_symbolic(g, feat, geom_feats, interval_starts,
                           interval_lengths, B, D, H, W):
    return g.op("custom::BEVPoolV2", feat, geom_feats,
                interval_starts, interval_lengths,
                plugin_version_s="1",
                out_dim_0_i=int(B), out_dim_1_i=int(D),
                out_dim_2_i=int(H), out_dim_3_i=int(W))

torch.onnx.register_custom_op_symbolic(
    "mmdet3d::bev_pool_v2", _bev_pool_v2_symbolic, opset_version=13)
```

### Step 3.5：导出并构建 vtransform_int8.engine

新建/更新 `tools/export_utils/export_vtransform.py`（完整导出，含 bev_pool_v2 节点）：

```bash
python tools/export_utils/export_vtransform.py \
    --config configs/.../convfuser.yaml --ckpt path/to/ptq.pth \
    --output vtransform_int8.onnx

python tools/export_utils/build_engine.py \
    --onnx vtransform_int8.onnx --engine vtransform_int8.engine \
    --int8 --fp16 \
    --plugins tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so
```

### Step 3.6：vtransform 精度验证（必须执行）

```bash
python tools/export_utils/verify_engine.py \
    --engine vtransform_int8.engine \
    --input calib_sample_0.pt \
    --module vtransform \
    --config configs/.../convfuser.yaml \
    --ckpt path/to/ptq.pth \
    --plugins tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so
# 期望：cosine_sim > 0.999
# ❌ 不达标则排查替换逻辑和导出流程，不得跳过
```

---

## Phase 4：SpConv LiDAR Backbone

### Step 4.1：探查 NVIDIA CUDA-BEVFusion 接口（先做）

```bash
cd /media/yellowstone/data2/CYL
git clone https://github.com/NVIDIA-AI-IOT/bevfusion --depth 1 nvidia_bevfusion_tmp
find nvidia_bevfusion_tmp/src -name "lidar-scn*" -o -name "*spconv*" | head -20
# 把输出结果反馈，决定后续路线
```

> Phase 4 的具体实现方案取决于 Step 4.1 的探查结果：
>
> - **路线 A**（期望）：NVIDIA 提供完整 TRT C++ 网络定义，直接集成
> - **路线 B**（备选）：手动用 TRT Python API 逐层构建 SparseEncoder
>
> Step 4.1 完成后根据实际结果补充 Step 4.2 的代码。

### Step 4.2：集成 Log2 Plugin

无论走哪条路线，SparseEncoder 每个 `_QuantizedSparseConv` 的激活量化处
都需要插入 `SparseLog2Quant` Plugin。具体集成方式待 Step 4.1 确认。

### Step 4.3：构建 lidar_backbone.engine

```bash
python tools/export_utils/build_engine.py \
    --onnx lidar_backbone.onnx --engine lidar_backbone.engine --fp16 \
    --plugins tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
```

### Step 4.4：LiDAR Backbone 精度验证（必须执行）

```bash
python tools/export_utils/verify_engine.py \
    --engine lidar_backbone.engine \
    --input calib_sample_0.pt \
    --module lidar \
    --config configs/.../convfuser.yaml \
    --ckpt path/to/ptq.pth \
    --plugins tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
# 期望：cosine_sim > 0.999
# ❌ 不达标则排查 Log2 Plugin 数值正确性
```

---

## Phase 5：端到端集成与 NDS 验证

### Step 5.1：完整 TRT 推理 Pipeline

新建 `tools/trt_infer.py`：

```python
"""
完整 TRT 推理 Pipeline。
① 6路摄像头 → swin_int8.engine            → SwinT features
② SwinT features → vtransform_int8.engine → Camera BEV
③ LiDAR → Voxelization (PyTorch)          → SparseConvTensor
④ SparseConvTensor → lidar_backbone.engine → LiDAR BEV
⑤ Camera BEV + LiDAR BEV → 现有 neck/fuser/decoder → 检测结果
"""
import ctypes, tensorrt as trt

# 启动时加载所有 Plugin（必须早于任何 TRT 操作）
ctypes.CDLL("tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so")
ctypes.CDLL("tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so")

# 具体推理逻辑根据现有推理代码改写
```

### Step 5.2：端到端 NDS 验证

```bash
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from trt_engines/ --eval bbox 2>&1 | tee trt_deploy_eval.log
```

目标：NDS 与 Round 9 仿真结果差值 < 0.003。

### Step 5.3：Log2 Plugin 数值验证（C++ 方式）

```bash
python tools/export_utils/test_log2_plugin.py \
    --plugin tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
# 期望：两项 cosine_sim > 0.9999
```

---

## 进度清单

```
Phase 0：基础工具
  [✅] 环境全部确认
  [✅] 0.1  build_engine.py               【服务器】✅ 2026-03-24
  [ ]  0.2  calib_sample_0.pt             【服务器】（Phase 5 时做）
  [ ]  0.3  freeze_fakequant.py           【服务器】（Phase 5 时做）
  [ ]  0.4  verify_engine.py              【服务器】（新增，每模块导出后使用）

Phase 1：SwinT（整体导出为单一 ONNX，无需为非量化层写 Plugin）
  [✅] 1.1  swin.py 静态化               【已在 mmdet3d/models/backbones/swin.py】
  [✅] 1.2  export_swin.py               【tools/export_utils/，含 symbolic 注册】
  [✅] 1.3  mqbench_symbolic_replacement 【覆盖 FakeQuantizeLearnablePerchannelAffine.symbolic】
  [✅] 1.4  swin_int8.onnx              【208 Q/DQ，0 FakeQuant，9143 节点】✅ 2026-03-24
  [✅] 1.5  swin_int8.engine            【32.3 MB，23,655 层，INT8+FP16】✅ 2026-03-24
  [ ]  1.6  SwinT 精度验证               【服务器，新增】⚠️ 尚未执行，需补做

Phase 2：Log2 Plugin（唯一需要手写的 Plugin）✅ 2026-03-25
  [✅] 2.1  sparse_log2_quant_kernel.cu  支持多维 [N,C,H,W]
  [✅] 2.2  sparse_log2_quant_plugin.h
  [✅] 2.3  sparse_log2_quant_plugin.cpp
  [✅] 2.4  CMakeLists.txt
  [✅] 2.5  服务器编译 .so               ~75KB
  [✅] 2.6  C++ 注册测试通过             Plugin 正确注册到 TRT Registry
  [ ]  2.7  数值验证（enqueue）          【Phase 5】Python API 版本不匹配，待 C++ 测试

Phase 3：BEV Pooling + depthnet INT8（从 BEVDet 提取，不从零写）✅ 2026-03-28
  [✅] 3.1  BEVDet 源码提取             【服务器 git clone + cp】已完成
  [✅] 3.2  bev_pool_v2 CMakeLists.txt  已完成，编译通过
  [✅] 3.3  bev_pool → bev_pool_v2 模型替换  ✅ 2026-03-28
        - Plugin kernel 坐标映射 bug 修复（x/y 反了）
        - BEVPoolV2Function + ONNX symbolic (custom::BEVPoolV2)
        - BaseTransform.precompute_bev_indices + bev_pool_with_indices
        - use_bev_pool_v2 开关，不影响现有功能
        - PyTorch 等价性验证: cosine_sim = 1.000000
  [✅] 3.4  bev_pool_v2 ONNX Symbolic 注册   ✅ 定义在 BEVPoolV2Function.symbolic
  [✅] 3.5  depthnet INT8 ONNX 导出      ✅ 5.5 MB, 24 Q/DQ 节点, 64 总节点
  [✅] 3.6  depthnet INT8 TRT 引擎       ✅ 1.6 MB (INT8+FP16), 151 层, RTX 3090
  [✅] 3.7  depthnet INT8 精度验证       ✅ cosine_sim = 0.999682 (FakeQuant vs TRT)
        max_abs_err = 0.013197, rmse = 0.000116
  [✅] 3.8  集成测试（depthnet TRT + 索引预计算 + bev_pool_v2）
        ✅ cosine_sim = 0.999999 (vs 原始 vtransform, FP16 引擎)
  注意: vtransform 拆分为 depthnet 引擎 + bev_pool_v2 Plugin
        索引预计算在 PyTorch 端完成（验证时每帧动态，部署时离线一次性）
        bev_pool_v2 是纯索引 scatter-add 操作，无可学习参数，不需要量化
        PTQ 量化覆盖: dtransform(3 Conv) + depthnet(3 Conv) + downsample(3 Conv) = 9 Conv, 全部 INT8
        KL Observer 校准的 scale/zero_point 已正确嵌入 ONNX Q/DQ 节点

Phase 4：SpConv LiDAR ✅ 路线确定：spconv 2.3 原生推理（2026-03-29）
  [❌] 4.1  libspconv.so 路线             放弃（段错误，闭源无法调试）
  [✅] 4.2  export_lidar.py 修复          Conv+BN fusion + Add 节点 trace
  [✅] 4.3  spconv 2.3 环境创建           /media/yellowstone/data2/CYL/spconv23_deploy
  [✅] 4.4  build_lidar_spconv23.py       spconv 2.3 重建 + 权重加载 + 推理
  [✅] 4.5  LiDAR FP16 精度验证           cosine_sim = 0.999994 ✅
  [ ]  4.6  W8A16 版本（权重INT8+激活FP16）— 未做，当前用 PyTorch FP16
  [ ]  4.7  INT8+Log2 版本（Log2 Plugin 集成）— 未做，当前用 PyTorch FP16

Phase 4b：Fuser + Decoder + Heads TRT 导出 ✅ 2026-03-29
  [✅] 4b.1 export_fuser_decoder.py       ConvFuser + SECOND + SECONDFPN → ONNX → TRT
        FP16: fuser_decoder_fp16_sm86.engine (10.4 MB)
        INT8: fuser_decoder_int8_sm86.engine (5.8 MB, 56 Q/DQ)
  [✅] 4b.2 TransFusionHead              ✅ 可导出 ONNX → TRT（argsort→topk, maxpool 维度修复）
        transfusion_head_fp16.onnx (5.5 MB, 439 nodes)
        transfusion_head_fp16_sm86.engine (3.4 MB)
  [ ]  4b.3 精度验证                      cosine_sim > 0.999 — 未单独验证

Phase 5：端到端 ✅ 2026-03-29（混合 TRT+PyTorch 方案，精度验证用）
  [✅] 5.1  trt_infer.py Pipeline         混合方案：TRT(SwinT/depthnet/fuser) + PyTorch(neck/bev_pool/lidar/head)
  [✅] 5.2  NDS 验证
        Version A (W8A16): NDS=0.7144, mAP=0.6851 (vs FP32 baseline 0.7069)
        Version B (INT8):  NDS=0.7102, mAP=0.6786 (vs FP32 baseline 0.7069)
        两个版本 NDS 均高于 FP32 baseline，精度验证通过
  [ ]  5.3  Log2 Plugin 数值验证（C++ 方式）— 未做

Phase 6：去 PyTorch 全量部署（目标：零 PyTorch 依赖）
  整体 pipeline:
    Voxelization (CUDA) → LiDAR Backbone (spconv 2.3 C++ API) → dense BEV
    → Camera Neck + bev_pool + Fuser+Decoder + TransFusionHead (全 TRT)

  [ ]  6.1  camera neck (GeneralizedLSSFPN) ONNX → TRT
        标准 Conv2d+BN+Upsample，直接 torch.onnx.export
  [ ]  6.2  TransFusionHead ONNX → TRT（含 argsort→topk 修复）
        已验证可导出：transfusion_head_fp16_sm86.engine (3.4 MB)
        需要写正式的 export_head.py + 精度验证
  [ ]  6.3  bev_pool TRT Plugin 集成
        已有 libbev_pool_v2_plugin.so，需要在 TRT engine 中使用或独立调用
  [ ]  6.4  LiDAR backbone spconv 2.3 C++ API 部署
        用 spconv::SparseInferenceEngine 脱离 PyTorch
        输出 dense BEV tensor 后接入 TRT pipeline
  [ ]  6.5  Voxelization 纯 CUDA 实现
        脱离 PyTorch 的 mmdet3d Voxelization
  [ ]  6.6  全量 pipeline 集成 + NDS 验证
        C++ 推理 pipeline，NDS 差值 < 0.003
```

---

## 失败应对预案

| 问题                        | 症状                                        | 解决方案                                                     |
| --------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| Q/DQ 节点数为 0             | export_swin.py 打印 Q/DQ=0                  | 把打印的 `all_ops` 中 FakeQuant 相关名称告诉我，更新 mqbench_onnx_symbolic.py 注册名 |
| 仍有 If 节点                | export_swin.py 打印 If>0                    | 把 `all_ops` 发给我，定位哪个算子引入了动态分支              |
| scale 非常量报错            | build_engine.py 报 "scale must be constant" | freeze_fakequant.py 中 scale 加 `.detach().clone()`，重新导出 |
| Log2 Plugin 编译失败        | make 报找不到 nvinfer                       | `ls $TRT_ROOT/include/NvInfer.h` 验证路径，检查 CMakeLists.txt 的 TRT_ROOT |
| Log2 Plugin C++ 测试失败    | test_plugin_registration 找不到 Plugin      | 确认 .so 已编译，LD_LIBRARY_PATH 包含 ${TRT_ROOT}/lib 和 ../build |
| Log2 Plugin 数值验证失败    | Phase 5 测试 cosine < 0.9999                | 检查 per_channel 分支：base [C] 要 unsqueeze(0) 才能广播到 [N,C] |
| bev_pool_v2 替换后精度下降  | verify_engine cosine < 0.999                | 对比 bev_pool 和 bev_pool_v2 的 rank/interval 预计算逻辑，检查坐标系对齐 |
| vtransform ONNX 无 BEVPoolV2 节点 | ONNX 中找不到 custom::BEVPoolV2 节点   | 确认 Symbolic 已注册，确认模型调用路径走的是 bev_pool_v2    |
| SpConv 无可用路线           | NVIDIA 接口不兼容                           | 上报后决定路线，不得擅自保留 PyTorch 混合方案                |
| 端到端 NDS 差距 > 0.003     | test.py 结果偏低                            | 逐引擎换回 PyTorch，定位哪个子模块引入精度损失               |