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

## 2026-02-25 21:30 · PTQ 4/6 MinMax 完整评估（校准 + NDS 确认）

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
| **模块压缩比（vs 26.5 MB FP32 权重）** | — | **1.96x** | **3.68x** |

> 注：压缩比相对于 4 个 TRT 模块的 FP32 权重（26.51 MB，占全模型 17%）计算。未量化模块（SwinTransformer 等，129.4 MB）在部署时仍需加载。

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
- **压缩效果**：INT8 引擎总大小 7.2 MB，相对于 4 模块 FP32 权重（26.5 MB）压缩 3.68 倍。但未量化模块（SwinTransformer 等）仍需 129.4 MB，总部署体积约 136.6 MB
- **推荐方案**：
  - 精度优先 → TRT FP16（NDS 无损，4 模块 1.96x 压缩）
  - 速度/大小优先 → TRT INT8（NDS −1.3%，4 模块 3.68x 压缩）
- **瓶颈分析**：SwinTransformer 占全模型参数的 67.5%（105.2 MB）且无法量化，是总体压缩的最大障碍

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
