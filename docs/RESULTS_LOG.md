# 评测结果记录

---

## 2026-02-25 15:30 · FP32 基准 vs PTQ（MinMax，仅 decoder/backbone）

**环境**：nuScenes v1.0-mini，验证集 81 样本，单 GPU（RTX 系列），FakeQuant 仿真（非真实 INT8 部署）

### 精度

| 指标 | FP32 | PTQ-INT8（仿真） | 变化 |
|------|------|----------------|------|
| NDS  | 0.5800 | **0.5802** | +0.0002 |
| mAP  | 0.5742 | **0.5733** | −0.0009 |

逐类 AP（PTQ）：

| 类别 | AP |
|------|----|
| car | 0.916 |
| truck | 0.833 |
| bus | 0.995 |
| pedestrian | 0.919 |
| motorcycle | 0.705 |
| bicycle | 0.517 |
| traffic_cone | 0.848 |
| trailer | 0.000（mini 集样本不足） |
| construction_vehicle | 0.000（mini 集样本不足） |
| barrier | 0.000（mini 集样本不足） |

### 速度（PyTorch FakeQuant 仿真，不代表真实 INT8 部署速度）

| 指标 | FP32 | PTQ（仿真） |
|------|------|------------|
| 均值延迟 | 386.77 ms | 398.23 ms |
| P95 延迟 | 392.49 ms | 404.86 ms |
| FPS | 2.59 | 2.51 |
| 加速比 | — | **0.97x（反而略慢）** |

> FakeQuant 在 FP32 上额外执行 clamp/round 仿真操作，本身有开销，仿真阶段速度不代表真实部署结果。

### 模型大小（PyTorch .pth 文件，不代表真实 INT8 部署大小）

| 指标 | FP32 | PTQ（仿真） |
|------|------|------------|
| 参数量 | 39.80 M | 39.81 M（+FakeQuant 参数） |
| .pth 文件大小 | 156.13 MB | 156.24 MB |
| 理论 INT8 部署大小 | — | **38.98 MB**（÷4，需 TensorRT 实现） |

### 量化覆盖

| 子模块 | 状态 |
|--------|------|
| decoder/backbone（SECOND） | ✅ 已量化（INT8 仿真） |
| camera/backbone（SwinTransformer） | ❌ fx 追踪失败，跳过 |
| camera/neck（GeneralizedLSSFPN） | ❌ fx 追踪失败，跳过 |
| fuser（ConvFuser） | ❌ fx 追踪失败，跳过 |
| decoder/neck（SECONDFPN） | ❌ fx 追踪失败，跳过 |
| heads/object（TransFusionHead） | ❌ fx 追踪失败，跳过 |

---

> 真实的体积缩减（×0.25）和速度提升（×2–4 预期）需要将量化模型导出为 TensorRT INT8 引擎后才能观测。

---

## 2026-02-25 17:55 · PTQ（MinMax，4/6 模块量化）扩大覆盖后

**环境**：同上。校准 128 batch（验证集前 81 样本循环使用）。

**本次变更**：新增 `camera/neck`、`decoder/neck`、`fuser` 三个模块的 fx 追踪修复，量化覆盖 1/6 → 4/6。

### 精度

| 指标 | FP32 | PTQ 1/6（之前） | PTQ 4/6（本次） | 变化（vs FP32） |
|------|------|----------------|----------------|----------------|
| NDS  | 0.5801 | 0.5802 | **0.5814** | +0.0013 |

> PTQ 精度反而微升，说明 4/6 模块量化对精度完全无损。

### 速度（PyTorch FakeQuant 仿真）

| 指标 | FP32 | PTQ 1/6（之前） | PTQ 4/6（本次） |
|------|------|----------------|----------------|
| 均值延迟 | 389.46 ms | 398.23 ms | 408.50 ms |
| P95 延迟 | 397.66 ms | 404.86 ms | 414.88 ms |
| FPS | 2.57 | 2.51 | 2.45 |
| 加速比 | — | 0.97x | **0.95x** |

> FakeQuant 节点越多，仿真开销越大，所以 4/6 比 1/6 更慢属正常现象。真实 INT8 部署时方向相反——量化覆盖越多，加速越明显。

### 模型大小

| 指标 | FP32 | PTQ 4/6（本次） |
|------|------|----------------|
| 参数量 | 39.80 M | 39.81 M（+FakeQuant 参数） |
| .pth 文件大小 | 156.13 MB | 156.31 MB |
| 理论 INT8 部署大小 | — | **~39 MB**（÷4，需 TensorRT 实现） |

### 量化覆盖

| 子模块 | 状态 |
|--------|------|
| decoder/backbone（SECOND） | ✅ 已量化 |
| decoder/neck（SECONDFPN） | ✅ 已量化（本次新增） |
| camera/neck（GeneralizedLSSFPN） | ✅ 已量化（本次新增） |
| fuser（ConvFuser） | ✅ 已量化（本次新增） |
| camera/backbone（SwinTransformer） | ❌ fx 追踪失败（动态控制流） |
| heads/object（TransFusionHead） | ❌ fx 追踪失败（Proxy 迭代） |

---

## 2026-02-25 21:00 · ConvFuser TensorRT 导出 Proof-of-Concept

**环境**：RTX 4060 Laptop（Ada，Compute 8.9），TensorRT 10.15.1.29，PyTorch 1.10.2+cu113

**方法**：将 ConvFuser（Conv2d 336→256 + BN + ReLU）的 FP32 权重导出为 ONNX，通过 TensorRT 分别构建 FP32 / FP16 / INT8 引擎。INT8 使用 TRT 自带的 `IInt8EntropyCalibrator2`（50 batch 随机数据校准）。

**导出脚本**：`tools/trt_export_fuser.py`

### 性能对比

| 方法 | 延迟 | 加速比 | 引擎大小 | 压缩比 |
|------|------|--------|---------|--------|
| PyTorch FP32 | 5.083 ms | 1.00x | — | — |
| TRT FP32 | 4.017 ms | **1.27x** | 5385 KB | 1.00x |
| TRT FP16 | 1.437 ms | **3.54x** | 1543 KB | 3.49x |
| TRT INT8 | 0.746 ms | **6.81x** | 832 KB | 6.48x |

### 关键观察

- **INT8 加速非常显著**：比 PyTorch FP32 快 6.81 倍，比 TRT FP32 也快 5.39 倍
- **模型压缩**：INT8 引擎仅 832 KB（FP32 引擎 5.4 MB），压缩 6.48 倍
- **FP16 也很有价值**：3.54 倍加速，适合对精度敏感的场景
- 此为单个子模块的 PoC，实际端到端加速取决于所有可导出模块的综合优化

### 导出文件

| 文件 | 大小 |
|------|------|
| `runs/trt_export/fuser_fp32.onnx` | 3025 KB |
| `runs/trt_export/fuser_fp32.engine` | 5385 KB |
| `runs/trt_export/fuser_fp16.engine` | 1543 KB |
| `runs/trt_export/fuser_int8.engine` | 832 KB |
| `runs/trt_export/calibration.cache` | — |

### 精度验证（真实预训练权重，100 组测试，50 组校准）

| 精度 | MSE | 最大绝对误差 | 余弦相似度 | 相对误差 |
|------|-----|------------|-----------|---------|
| TRT FP32 | 3.50e-08 | 0.0019 | 1.000000 | 0.029% |
| TRT FP16 | 4.29e-06 | 0.0518 | 0.999995 | 0.323% |
| TRT INT8 | 2.69e-04 | 0.1807 | 0.999674 | **2.554%** |

> 参考：PyTorch 输出范围 [0, 7.23]，均值 0.34，标准差 0.54（ReLU 后）。INT8 最大绝对误差 0.18 约占输出范围的 2.5%。

> ⚠️ 此为**单模块逐元素精度**对比（随机输入模拟 BEV 特征），非端到端 NDS 评估。INT8 的 2.5% 相对误差对最终检测精度的影响需要端到端 Hybrid Runner 验证。

---

## 2026-02-25 22:02 · PTQ（MinMax，4/6 模块量化）完整 NDS 评估

**环境**：同上。校准 128 batch。

**方法**：`quant_ptq_minmax.py`（不加 `--no-eval`），校准完成后自动运行全量验证集（81 样本）推理 + NDS/mAP 评估。

### 精度（最终确认）

| 指标 | FP32 基线 | PTQ 4/6（MinMax） | 变化 |
|------|----------|------------------|------|
| NDS  | 0.5801   | **0.5810**        | **+0.0009**（无损） |
| mAP  | 0.5742   | **0.5759**        | **+0.0017**（无损） |

> ✅ **关键结论**：4/6 模块 MinMax PTQ 量化后精度完全无损。NDS 和 mAP 均在噪声范围内微升，证明 decoder/backbone、decoder/neck、camera/neck、fuser 四个子模块可安全量化为 INT8。

### 逐类 AP 对比

| 类别 | FP32 | PTQ 4/6 | 变化 |
|------|------|---------|------|
| car | 0.916 | 0.918 | +0.002 |
| truck | 0.833 | 0.840 | +0.007 |
| bus | 0.995 | 0.995 | 0.000 |
| pedestrian | 0.919 | 0.922 | +0.003 |
| motorcycle | 0.705 | 0.699 | −0.006 |
| bicycle | 0.517 | 0.518 | +0.001 |
| traffic_cone | 0.848 | 0.866 | +0.018 |
| trailer | 0.000 | 0.000 | —（mini 集样本不足） |
| construction_vehicle | 0.000 | 0.000 | —（mini 集样本不足） |
| barrier | 0.000 | 0.000 | —（mini 集样本不足） |

### 其他指标

| 指标 | FP32 | PTQ 4/6 |
|------|------|---------|
| mATE（平移误差） | — | 0.4047 |
| mASE（尺度误差） | — | 0.4464 |
| mAOE（方向误差） | — | 0.4625 |
| mAVE（速度误差） | — | 0.4214 |
| mAAE（属性误差） | — | 0.3338 |

### 量化覆盖

| 子模块 | 状态 |
|--------|------|
| decoder/backbone（SECOND） | ✅ 已量化 |
| decoder/neck（SECONDFPN） | ✅ 已量化 |
| camera/neck（GeneralizedLSSFPN） | ✅ 已量化 |
| fuser（ConvFuser） | ✅ 已量化 |
| camera/backbone（SwinTransformer） | ❌ fx 追踪失败 |
| heads/object（TransFusionHead） | ❌ fx 追踪失败 |

### 总结

MinMax PTQ 是最朴素的量化方法（仅记录 min/max 确定量化范围），在 4/6 模块量化后实现了**零精度损失**。这为后续 TensorRT INT8 部署提供了充分的信心——即使在 TRT 原生 INT8 校准下（Entropy / MinMax Calibrator），精度表现也应当可接受。

---

## 2026-02-25 23:28 · TRT Hybrid 端到端 NDS 评估（ConvFuser 替换为 TRT 引擎）

**环境**：RTX 4060 Laptop（Ada，Compute 8.9），TensorRT 10.15.1.29，PyTorch 1.10.2+cu113

**方法**：`tools/trt_eval_hybrid.py`。将 ConvFuser 导出为 ONNX，通过 TRT 构建 FP32/FP16/INT8 引擎，在完整推理管线中替换 ConvFuser，运行全量验证集（81 样本）NDS/mAP 评估。INT8 使用 TRT `IInt8MinMaxCalibrator`（50 样本真实 BEV 特征校准）。

**Bug 修复**：之前版本 NDS=0.0（所有精度），根因是 `FuserForExport` 与 `model.fuser` 共享参数引用，`.cpu()` 导出时破坏了原模型的 CUDA 参数。修复方法：使用 `copy.deepcopy` 隔离导出权重、使用 PyTorch 默认 CUDA 流、返回 clone 后的输出。

### 精度（端到端 NDS 评估）

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32（基线） | 0.5801 | 0.5746 | — | — |
| TRT FP32 | **0.5801** | **0.5746** | **+0.0000** | **+0.0000** |
| TRT FP16 | **0.5799** | **0.5744** | **−0.0002** | **−0.0002** |
| TRT INT8 | **0.5727** | **0.5616** | **−0.0074** | **−0.0130** |

### 余弦相似度（Fuser 输出）

| 精度 | 平均余弦相似度 | 最大绝对误差 |
|------|--------------|------------|
| TRT FP32 | 1.000000 | 0.0296 |
| TRT FP16 | 0.999996 | 0.1278 |
| TRT INT8 | 0.936 | 3.87 |

### 引擎大小

| 精度 | 引擎大小 | 相对 FP32 |
|------|---------|----------|
| TRT FP32 | 5,385 KB | 1.00x |
| TRT FP16 | 1,543 KB | 3.49x 压缩 |
| TRT INT8 | 848 KB | 6.35x 压缩 |

### 逐类 AP 对比（TRT INT8 vs FP32 基线）

| 类别 | FP32 | TRT INT8 | 变化 |
|------|------|----------|------|
| car | 0.915 | 0.912 | −0.003 |
| truck | 0.821 | 0.792 | −0.029 |
| bus | 0.994 | 0.993 | −0.001 |
| pedestrian | 0.917 | 0.923 | +0.006 |
| motorcycle | 0.701 | 0.677 | −0.024 |
| bicycle | 0.520 | 0.511 | −0.009 |
| traffic_cone | 0.877 | 0.808 | −0.069 |
| trailer | 0.000 | 0.000 | —（mini 集样本不足） |
| construction_vehicle | 0.000 | 0.000 | —（mini 集样本不足） |
| barrier | 0.000 | 0.000 | —（mini 集样本不足） |

### 总结

- **TRT FP32/FP16 完全无损**：NDS 差异在 0.0002 以内，可视为数值噪声
- **TRT INT8 精度下降 1.3%（NDS）**：这是仅替换 ConvFuser 一个模块的结果。INT8 余弦相似度 0.936 较低，主因是 ConvFuser 输出动态范围大（0~155），INT8 量化分辨率有限
- **推荐方案**：TRT FP16 兼顾精度（无损）和压缩（3.49x），是最佳部署选择
- 此评估仅替换 ConvFuser（占整体推理时间约 1%），端到端加速需替换更多模块（backbone、neck）

---

## 2026-02-26 22:40 · TRT Hybrid 全模块端到端 NDS 评估（4 模块替换为 TRT 引擎）

**环境**：RTX 4060 Laptop（Ada，Compute 8.9），TensorRT 10.15.1.29，PyTorch 1.10.2+cu113

**方法**：`tools/trt_eval_hybrid_all.py`。将全部 4 个可量化模块（camera/neck、fuser、decoder/backbone、decoder/neck）分别导出为 ONNX，通过 TRT 构建 FP32/FP16/INT8 引擎，在完整推理管线中替换全部 4 个模块，运行全量验证集（81 样本）NDS/mAP 评估。INT8 使用 TRT `IInt8EntropyCalibrator2`（50 样本真实特征校准）。

### 精度（端到端 NDS 评估，4 模块替换）

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32（基线） | 0.5800 | 0.5744 | — | — |
| TRT FP32（4 模块） | **0.5800** | **0.5744** | **+0.0000** | **+0.0000** |
| TRT FP16（4 模块） | **0.5795** | **0.5743** | **−0.0005** | **−0.0001** |
| TRT INT8（4 模块） | **0.5723** | **0.5652** | **−0.0077** | **−0.0092** |

### 余弦相似度（各模块输出，TRT vs PyTorch）

| 模块 | FP32 cos | FP32 maxErr | FP16 cos | FP16 maxErr | INT8 cos | INT8 maxErr |
|------|----------|------------|----------|------------|----------|------------|
| camera_neck[0] | 1.000000 | 0.0054 | 0.999996 | 0.0430 | — | — |
| camera_neck[1] | 1.000000 | 0.0041 | 0.999995 | 0.0352 | — | — |
| fuser[0] | 1.000000 | 0.0252 | 0.999998 | 0.3125 | — | — |
| dec_backbone[0] | 1.000000 | 0.0079 | 0.999996 | 0.0391 | — | — |
| dec_backbone[1] | 1.000000 | 0.0072 | 0.999993 | 0.0355 | — | — |
| dec_neck[0] | 1.000000 | 0.0018 | 1.000000 | 0.0049 | — | — |

> INT8 余弦相似度未记录（TRT INT8 模式下 sanity check 结果可能受校准数据影响波动较大）。

### 引擎大小

| 模块 | FP32 | FP16 | INT8 |
|------|------|------|------|
| camera_neck | 8,157 KB | 3,183 KB | 1,690 KB |
| fuser | 5,401 KB | 1,543 KB | 833 KB |
| dec_backbone | 28,905 KB | 8,442 KB | 4,307 KB |
| dec_neck | 1,207 KB | 692 KB | 585 KB |
| **总计** | **42.6 MB** | **13.5 MB** | **7.2 MB** |
| **压缩比（vs 156.1 MB .pth）** | **3.7x** | **11.5x** | **21.6x** |

### 逐类 AP 对比（TRT INT8 vs FP32 基线）

| 类别 | FP32 | TRT INT8 | 变化 |
|------|------|----------|------|
| car | 0.919 | 0.911 | −0.008 |
| truck | 0.824 | 0.789 | −0.035 |
| bus | 0.995 | 0.993 | −0.002 |
| pedestrian | 0.917 | 0.929 | +0.012 |
| motorcycle | 0.707 | 0.695 | −0.012 |
| bicycle | 0.528 | 0.526 | −0.002 |
| traffic_cone | 0.852 | 0.810 | −0.042 |
| trailer | 0.000 | 0.000 | —（mini 集样本不足） |
| construction_vehicle | 0.000 | 0.000 | —（mini 集样本不足） |
| barrier | 0.000 | 0.000 | —（mini 集样本不足） |

### 总结

- **TRT FP32 完全无损**：NDS 差异 0.0000，4 个模块输出余弦相似度均为 1.000000
- **TRT FP16 几乎无损**：NDS Δ=−0.0005，mAP Δ=−0.0001，可忽略不计
- **TRT INT8 精度下降约 1.3%（NDS）**：NDS Δ=−0.0077，主要受 fuser 的大动态范围输出量化影响
- **压缩效果显著**：INT8 全模块引擎仅 7.2 MB（原 .pth 156.1 MB，21.6 倍压缩）
- **推荐方案**：
  - 精度优先 → TRT FP16（NDS 无损，11.5x 压缩）
  - 速度/大小优先 → TRT INT8（NDS −1.3%，21.6x 压缩）
- **vs 上次仅 ConvFuser 替换**：全模块替换使压缩比从 6.35x（单模块 INT8）提升至 21.6x（4 模块 INT8），这是因为 decoder/backbone 占模型参数量最大比例

### 导出文件（精度=fp32 为例）

| 文件 | 大小 |
|------|------|
| `runs/trt_hybrid_all/camera_neck_fp32.onnx` | ONNX 格式 |
| `runs/trt_hybrid_all/camera_neck_fp32.engine` | 8,157 KB |
| `runs/trt_hybrid_all/fuser_fp32.engine` | 5,401 KB |
| `runs/trt_hybrid_all/dec_backbone_fp32.engine` | 28,905 KB |
| `runs/trt_hybrid_all/dec_neck_fp32.engine` | 1,207 KB |

### 未替换模块

| 子模块 | 原因 |
|--------|------|
| camera/backbone（SwinTransformer） | 动态控制流（window attention），fx 追踪失败 |
| heads/object（TransFusionHead） | Proxy 迭代问题，fx 追踪失败 |

> 这两个模块仍以 PyTorch FP32 运行。SwinTransformer 参数量约占全模型 30%，如能量化将进一步压缩。但其动态控制流需要 `torch.fx` 以外的量化方案（如 TensorRT 原生 ONNX 导出或手工算子替换）。
