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
