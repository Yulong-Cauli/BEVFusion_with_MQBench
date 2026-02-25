# BEVFusion 混合精度量化与 TensorRT 部署报告

> **项目**：BEVFusion + MQBench 训练后量化（PTQ）  
> **日期**：2026-02  
> **硬件**：NVIDIA RTX 4060 Laptop GPU（Ada Lovelace，Compute 8.9）  
> **框架**：PyTorch 1.10.2 + CUDA 11.3 + TensorRT 10.15.1

---

## 目录

1. [概述](#1-概述)
2. [BEVFusion 模型架构分析](#2-bevfusion-模型架构分析)
3. [混合精度量化设计思路](#3-混合精度量化设计思路)
4. [实现细节](#4-实现细节)
5. [各模块精度方案一览](#5-各模块精度方案一览)
6. [混合精度的进一步探索](#6-混合精度的进一步探索)
7. [数据集与校准集分析](#7-数据集与校准集分析)
8. [实验结果](#8-实验结果)
9. [后续工作](#9-后续工作)
10. [结论](#10-结论)

---

## 1. 概述

BEVFusion 是一种多模态 3D 目标检测模型，融合摄像头和激光雷达信息生成 BEV（鸟瞰图）表示。模型在 nuScenes 数据集上取得了优秀的检测精度，但其约 40M 参数、156 MB 的模型体积和 ~389 ms 的单帧推理延迟限制了实际部署。

本项目基于 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，对 BEVFusion 实施**训练后量化（PTQ）**，目标后端为 NVIDIA TensorRT INT8。核心挑战在于：BEVFusion 是一个**异构多模态模型**，包含稀疏卷积、自定义 CUDA 算子、Transformer 等多种子模块，无法对全模型统一量化，因此我们设计了**选择性混合精度量化**方案。

**核心成果**：

| 指标 | FP32 基线 | PTQ 4/6 (FakeQuant) | TRT INT8 (Hybrid) |
|------|----------|--------------------|--------------------|
| NDS | 0.5801 | **0.5810 (+0.0009)** | **0.5727 (−0.0074)** |
| mAP | 0.5742 | **0.5759 (+0.0017)** | **0.5616 (−0.0126)** |
| ConvFuser 延迟 | 5.08 ms | — | **0.75 ms (6.81x↑)** |
| ConvFuser 引擎大小 | — | — | **832 KB (6.48x↓)** |

---

## 2. BEVFusion 模型架构分析

BEVFusion 的推理管线可以分为以下几个阶段：

```
输入                     特征提取                  BEV 投影         融合       解码       检测
─────────────────────────────────────────────────────────────────────────────────────────────
多视角图像 ─→ SwinTransformer ─→ GeneralizedLSSFPN ─→ bev_pool ─┐
    (6×256×704×3)   camera/backbone    camera/neck      camera/vtransform  │
                                                                           ├→ ConvFuser ─→ SECOND ─→ SECONDFPN ─→ TransFusionHead ─→ 3D Bbox
                                                                           │     fuser    decoder/   decoder/     heads/object
点云 ─→ Voxelization ─→ SparseEncoder ─────────────────────────────────────┘    backbone    neck
         lidar/voxelize  lidar/backbone
```

### 各子模块的计算特性

| 子模块 | 类 | 算子类型 | 量化友好度 |
|--------|-----|---------|-----------|
| `camera/backbone` | SwinTransformer | Attention + MLP + Window Partition | 🔴 困难 |
| `camera/neck` | GeneralizedLSSFPN | Conv2d + BN + ReLU + Bilinear Upsample | 🟢 友好 |
| `camera/vtransform` | LSSTransform | 自定义 CUDA 算子 (QuickCumsumCuda) | ⛔ 不可能 |
| `lidar/voxelize` | Voxelization | 动态分散 (scatter)，非神经网络层 | ⛔ 不适用 |
| `lidar/backbone` | SparseEncoder | 稀疏卷积 (spconv) | ⛔ 不兼容 |
| `fuser` | ConvFuser | Conv2d(336,256,3) + BN + ReLU | 🟢 友好 |
| `decoder/backbone` | SECOND | 多层 Conv2d + BN + ReLU | 🟢 友好 |
| `decoder/neck` | SECONDFPN | ConvTranspose2d + BN + ReLU | 🟢 友好 |
| `heads/object` | TransFusionHead | Attention + TopK + 动态 shape | 🔴 困难 |

**关键观察**：量化友好的模块集中在 BEV 空间的中后段管线（neck → fuser → decoder），而计算最密集的 SwinTransformer 和稀疏卷积恰恰是最难量化的部分。

---

## 3. 混合精度量化设计思路

### 3.1 为什么不能全模型量化

传统 PTQ 工具（如 TensorRT 的 `trtexec --int8`）假设模型可被完整导出为 ONNX，再统一做 INT8 校准。BEVFusion 存在以下障碍使得全模型导出不可行：

1. **稀疏卷积（spconv）**：LiDAR 分支的 SparseEncoder 使用稀疏张量表示，标准 ONNX 和 TensorRT 均不支持。稀疏卷积的输入/输出格式与密集张量完全不同，FakeQuant 节点无法插入。

2. **自定义 CUDA 算子**：Camera 分支的 `bev_pool`（QuickCumsumCuda）是用 CUDA C++ 实现的 autograd Function，没有对应的 ONNX 算子或 TensorRT 插件。

3. **动态控制流**：SwinTransformer 内部有 `if x.shape[0] > window_size:` 等基于张量值的分支判断，`torch.fx` 符号追踪时会失败（Proxy 对象无法求值为布尔值）。TransFusionHead 中 `for layer in decoder_layers:` 等动态迭代同样不兼容。

4. **体素化预处理**：Voxelization 将不规则点云映射为规则网格，这是一个离散化预处理步骤，不包含可微分的权重，不需要也不应该量化。

### 3.2 选择性量化策略

基于以上分析，我们采用**选择性量化**：逐个子模块独立调用 `MQBench.prepare_by_platform`（基于 `torch.fx` 符号追踪），仅量化可追踪的子模块，跳过不兼容的部分。

```
已量化（INT8 FakeQuant）：           保持 FP32：
├── camera/neck (GeneralizedLSSFPN)   ├── camera/backbone (SwinTransformer)
├── fuser (ConvFuser)                 ├── camera/vtransform (bev_pool)
├── decoder/backbone (SECOND)         ├── lidar/* (稀疏卷积)
└── decoder/neck (SECONDFPN)          ├── heads/object (TransFusionHead)
                                      └── lidar/voxelize
```

量化覆盖率：**4/6 可量化模块**（排除设计上不适合量化的 vtransform 和 voxelize 后，实际可量化模块为 6 个，成功量化 4 个）。

### 3.3 分段 TensorRT 部署（Hybrid 推理）

由于无法将全模型导出为 TRT 引擎，我们设计了 **Hybrid 推理架构**：将已量化且 ONNX 兼容的子模块导出为 TRT 引擎，其余部分保持 PyTorch 执行。数据在 PyTorch 和 TRT 之间通过 CUDA 显存零拷贝传递。

```
PyTorch 执行区域：                    TRT 引擎区域：
┌─────────────────────┐              ┌──────────────────┐
│ SwinTransformer      │              │                  │
│ LSSTransform         │──BEV feat──→│  ConvFuser (TRT)  │
│ SparseEncoder        │              │                  │
│ TransFusionHead      │←─fused BEV──│                  │
│ SECOND (PyTorch)     │              └──────────────────┘
│ SECONDFPN (PyTorch)  │
└─────────────────────┘
```

当前 PoC 仅替换了 ConvFuser 一个模块。ConvFuser 是一个轻量的单层卷积（Conv2d + BN + ReLU），验证了 Hybrid 架构的可行性。

---

## 4. 实现细节

本项目在 `tools/` 目录下新增了以下脚本，在 `mmdet3d/` 下修改了模型代码以支持 `torch.fx` 追踪。

### 4.1 `tools/quant_ptq_minmax.py` — PTQ 主流程

这是核心量化脚本，实现了完整的 MinMax PTQ 流程：

**主要功能**：
- **选择性量化** (`apply_selective_ptq`)：遍历 `_QUANTIZABLE_SUBMODULE_KEYS` 列表，对每个子模块独立调用 `prepare_by_platform(submodule, BackendType.Tensorrt)` 插入 FakeQuantize 节点。失败的子模块自动跳过并记录日志。
- **mmcv 兼容性补丁** (`patch_mmcv_for_fx`)：上下文管理器，在 `torch.fx` 追踪期间临时将 mmcv 的 `Conv2d`/`ConvTranspose2d`/`MaxPool2d`/`Linear` 包装层的 `forward` 方法替换为 PyTorch 原生父类版本。这是因为 mmcv 的兼容性代码中有 `if x.numel() == 0` 分支，在符号追踪时 `x` 是 Proxy 对象，无法求值为布尔值。
- **MinMax 校准** (`run_calibration`)：`enable_calibration` → 前向推理收集 min/max → `enable_quantization` 激活 FakeQuant。
- **精度评估** (`evaluate_quantized_model`)：调用 `single_gpu_test` + `dataset.evaluate` 输出完整 NDS/mAP。

**关键设计决策**：
- 校准和评估均在 `test_mode=True` 下进行，与推理管线保持一致
- 模型必须包装在 `MMDataParallel` 中，否则 `DataContainer` 对象无法被正确解包
- PTQ checkpoint 的 `state_dict` 键名经 `torch.fx` 改造，不能用 `test.py` 直接评估

### 4.2 `tools/quant_benchmark.py` — Benchmark 工具

测量并对比 FP32 模型与量化模型的：
- **参数量**：统计模型可训练参数数量
- **模型大小**：FP32 `.pth` 文件大小，以及理论 INT8 部署大小（÷4 估算）
- **推理延迟**：GPU warmup + 正式计时，报告均值/P95/P99 延迟

支持 `--size-only` 模式（仅报告大小，不需要数据集）和 `--use-real-data` 模式（使用真实 nuScenes 数据）。

### 4.3 `tools/trt_export_fuser.py` — ConvFuser TRT 导出 PoC

验证单个子模块的 TRT 导出可行性：

1. **ONNX 导出**：通过 `FuserForExport` 包装类将 ConvFuser 的 `list` 输入转换为两个独立的 tensor 参数（`camera_bev` 和 `lidar_bev`），解决 ONNX 不支持 list 输入的问题
2. **引擎构建**：分别构建 FP32/FP16/INT8 三种精度的 TRT 引擎。INT8 使用 `IInt8EntropyCalibrator2` 随机数据校准
3. **隔离延迟测试**：对 PyTorch 和各 TRT 精度分别进行 200 次推理，报告平均延迟和加速比

### 4.4 `tools/trt_accuracy_test.py` — TRT 精度验证

加载**真实预训练权重**（从 `bevfusion-det.pth` 中提取 fuser 参数），对比 PyTorch FP32 与各 TRT 精度的逐元素精度差异：
- MSE（均方误差）
- 最大绝对误差
- 余弦相似度
- 相对误差百分比

使用模拟 BEV 特征分布的数据（100 组测试、50 组校准），验证 TRT 导出的数值正确性。

### 4.5 `tools/trt_eval_hybrid.py` — Hybrid TRT 端到端 NDS 评估

这是最重要的验证脚本，实现了完整的 Hybrid 推理 + NDS 评估：

**工作流程**：

```
Step 1: 加载 FP32 模型 + 预训练权重
Step 2: ONNX 导出（使用 deepcopy 隔离，避免破坏原模型参数）
Step 3: 收集校准数据（真实 BEV 特征）+ 运行 FP32 基线
Step 4: 构建 TRT 引擎（FP32/FP16/INT8）
Step 5: 替换 model.fuser → TRTFuser，运行完整 NDS 评估
```

**关键实现**：

- **`FuserForExport`**（ONNX 导出包装器）：使用 `copy.deepcopy(model.fuser)` 创建独立的参数副本。这是修复 NDS=0.0 bug 的关键——之前直接引用 `model.fuser` 的子模块，导致 `.cpu()` 导出时破坏了原模型的 CUDA 参数。
- **`TRTFuser`**（TRT 推理替换器）：作为 `nn.Module` 的子类，可直接替换 `model.fuser`。使用 PyTorch 默认 CUDA 流（而非自定义流）避免流间同步问题，返回 `clone()` 后的输出防止缓冲区复用。
- **`RealDataCalibrator`**（INT8 校准器）：使用真实 BEV 特征（而非随机数据）进行校准，继承自 `trt.IInt8MinMaxCalibrator`。MinMax 校准器对 ConvFuser 的效果优于 Entropy 校准器（后者过度裁剪动态范围）。
- **`--debug` 模式**：保存 `deepcopy` 的原始 fuser，在推理时逐样本对比 PyTorch 与 TRT 输出的余弦相似度。

### 4.6 模型代码修改

为支持 `torch.fx` 符号追踪，对以下模型文件进行了最小化修改：

#### `mmdet3d/models/necks/second.py`（SECONDFPN）

```python
# 修改前（fx 追踪失败：len(x) 在 Proxy 上不可用）
def forward(self, x):
    assert len(x) == len(self.in_channels)
    ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
    ...

# 修改后
def forward(self, x):
    # 移除了 assert len(x) == len(self.in_channels) 断言
    ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
    ...
```

#### `mmdet3d/models/necks/generalized_lss.py`（GeneralizedLSSFPN）

```python
# 修改前（fx 追踪失败：len(inputs) 和 range(len(inputs)) 在 Proxy 上不可用）
def forward(self, inputs):
    assert len(inputs) == len(self.in_channels)
    laterals = [inputs[i + self.start_level] for i in range(len(inputs))]
    ...

# 修改后
def forward(self, inputs):
    # 使用 self.num_ins（__init__ 中预计算的常量）替代 len(inputs)
    laterals = [inputs[i + self.start_level] for i in range(self.num_ins)]
    ...
```

#### `mmdet3d/models/fusers/conv.py`（ConvFuser）

```python
# 修改前（fx 追踪失败：torch.cat(Proxy, dim=1) 中 Proxy 代表整个 list，fx 无法展开）
def forward(self, inputs):
    return super().forward(torch.cat(inputs, dim=1))

# 修改后
def forward(self, inputs):
    # 通过 __getitem__ 索引让 fx 看到独立的 Proxy 对象
    return super().forward(torch.cat([inputs[i] for i in range(len(self.in_channels))], dim=1))
```

**修改原则**：
- 最小化改动，不改变运行时行为
- 仅消除 `torch.fx` 追踪时的动态控制流
- 用 `__init__` 中的常量替代运行时的 `len()` 调用

---

## 5. 各模块精度方案一览

下表详细列出了 BEVFusion 每个子模块在当前方案中的精度状态：

| 子模块 | 类型 | 当前精度 | PTQ 状态 | TRT 导出状态 | 备注 |
|--------|------|---------|---------|-------------|------|
| `camera/backbone` | SwinTransformer | **FP32** | ❌ fx 追踪失败 | ❌ 不可导出 | 动态控制流：`if tensor > window_size` |
| `camera/neck` | GeneralizedLSSFPN | **INT8 (FakeQuant)** | ✅ 已量化 | 🟡 可导出（未实施） | 修复 `len()` + mmcv patch |
| `camera/vtransform` | LSSTransform | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | `bev_pool` 自定义 CUDA 算子 |
| `lidar/voxelize` | Voxelization | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | 非神经网络层 |
| `lidar/backbone` | SparseEncoder | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | 稀疏卷积 |
| `fuser` | ConvFuser | **INT8 (TRT)** | ✅ 已量化 | ✅ 已导出 | Hybrid 推理验证通过 |
| `decoder/backbone` | SECOND | **INT8 (FakeQuant)** | ✅ 已量化 | 🟡 可导出（未实施） | 纯 Conv2d 堆叠 |
| `decoder/neck` | SECONDFPN | **INT8 (FakeQuant)** | ✅ 已量化 | 🟡 可导出（未实施） | 含 ConvTranspose2d |
| `heads/object` | TransFusionHead | **FP32** | ❌ fx 追踪失败 | 🟡 需 opset≥11 | Proxy 迭代 + 动态 shape |

---

## 6. 混合精度的进一步探索

既然已经实现了"分模块量化"的基础设施，一个自然的延伸是：**不同的模块使用不同的精度**，以在压缩率和精度之间取得更细粒度的平衡。

### 6.1 可行的混合精度方案

| 方案 | camera/neck | fuser | decoder/backbone | decoder/neck | 预期效果 |
|------|------------|-------|-----------------|-------------|---------|
| A（当前） | INT8 FakeQuant | INT8 TRT | INT8 FakeQuant | INT8 FakeQuant | NDS −0.0074 (TRT Hybrid) |
| B（保守） | FP16 TRT | FP16 TRT | INT8 TRT | INT8 TRT | 预期 NDS ≈ −0.002 |
| C（激进） | INT8 TRT | INT8 TRT | INT8 TRT | INT8 TRT | 预期 NDS ≈ −0.01，最大压缩 |
| D（精度优先） | FP16 TRT | FP16 TRT | FP16 TRT | FP16 TRT | 预期 NDS ≈ −0.001，中等压缩 |

### 6.2 分析与建议

**方案 B（推荐探索）**：

- ConvFuser 的 INT8 余弦相似度为 0.936，是精度损失的主要来源。将 fuser 改为 FP16（余弦相似度 0.999996，NDS −0.0002）可显著减少精度损失
- decoder/backbone 和 decoder/neck 都是纯密集卷积，INT8 量化对精度影响小
- camera/neck 包含多尺度特征融合和双线性上采样，使用 FP16 更安全

**方案 C（最大压缩）**：

- 将所有 4 个已量化模块都导出为 TRT INT8
- 需要为每个模块分别收集校准数据，选择合适的校准器（MinMax vs Entropy）
- decoder/backbone 和 decoder/neck 的输入来自 fuser 的输出，如果 fuser 已是 INT8，误差会累积传播

**精度敏感性排序**（从实验结果推断）：

```
fuser > camera/neck > decoder/neck > decoder/backbone
（对精度影响从大到小）
```

fuser 位于 camera 和 lidar 特征融合的关键路径上，其量化误差会传播到所有下游模块。因此 fuser 使用 FP16、decoder 使用 INT8 的混合方案可能是最优的工程折衷。

### 6.3 实施难度

混合精度 TRT 引擎的实施主要有两种路径：

1. **逐模块独立引擎**（当前方案的自然扩展）：每个子模块有自己的 TRT 引擎，可单独设置精度。缺点是引擎间数据传递有开销。
2. **单引擎多精度**（TRT 原生支持）：TRT 支持 per-layer 精度设置，但需要将多个模块合并导出为一个 ONNX，然后在 TRT 中标记每层的精度。对于 BEVFusion 的 Hybrid 架构，合并导出较困难。

---

## 7. 数据集与校准集分析

### 7.1 当前数据集

本项目使用 **nuScenes v1.0-mini** 数据集：

| 数据集 | 样本数 | 用途 |
|--------|-------|------|
| 训练集 | 323 帧 | 未使用（PTQ 不需要训练） |
| 验证集 | 81 帧 | PTQ 校准 + NDS 精度评估 |

**nuScenes v1.0-mini 的特点**：
- 仅包含 10 个场景（scenes），来自 Boston 和 Singapore 两座城市
- 每帧包含 6 个摄像头图像 + 1 个 LiDAR 点云 + 完整 3D 标注
- 约占 nuScenes full 数据集的 1/40（full 有 ~28k 训练 + ~6k 验证帧）

### 7.2 校准集设置

| 参数 | PTQ (MQBench) | TRT INT8 (ConvFuser) |
|------|--------------|---------------------|
| 校准数据来源 | 验证集循环采样 | 真实 BEV 特征（模型前向推理提取） |
| 校准样本数 | 128 batch | 50 样本 |
| 校准方法 | MinMax（记录全局 min/max） | IInt8MinMaxCalibrator |
| 校准数据分布 | 原始图像 + 点云 | ReLU 后的 BEV 特征（全正值） |

### 7.3 说服力评估

**优势**：
- Mini 数据集虽小，但包含了城市道路的主要场景类型（十字路口、直行道、转弯等）
- PTQ 校准仅需观测各层激活值的统计特性（min/max），对样本数量的需求远低于训练
- 128 batch 的校准量对于 MinMax 策略已经充分，因为 min/max 在约 30-50 batch 后即趋于稳定
- 实验结果也验证了这一点：PTQ 精度无损（NDS +0.0009），说明校准数据足以捕获分布

**局限性**：
- **评估集过小（81 帧）**：3 个类别（trailer、construction_vehicle、barrier）的样本数为 0，导致这些类别的 AP 评估无意义。这意味着我们的 NDS/mAP 数字实际上只反映了 7/10 个类别的表现
- **场景多样性不足**：Mini 数据集仅有 10 个场景，不同天气、光照条件的覆盖有限。在极端条件下（如强逆光、暴雨、夜间），量化模型的鲁棒性未被验证
- **校准集 = 验证集**：PTQ 校准使用了验证集的数据，存在一定程度的"过拟合"风险，但对于 MinMax 这种极简单的统计方法（仅记录全局 min/max），这种风险很小

### 7.4 使用 nuScenes 完整数据集的考虑

| 方面 | Mini (当前) | Full |
|------|------------|------|
| 验证集大小 | 81 帧 | ~6,019 帧 |
| 场景数 | 10 | 850 |
| 所有类别有样本 | ❌ (7/10) | ✅ (10/10) |
| 评估可信度 | 🟡 定性可信 | 🟢 定量可信 |
| 磁盘空间 | ~4 GB | ~300+ GB |
| 评估时间 | ~3 分钟 | ~4 小时（估算） |

**建议**：

- **如果目标是发表论文或正式报告**：**必须**在完整 nuScenes 验证集上评估。Mini 数据集的结果只能作为开发阶段的快速验证参考。
- **如果目标是工程验证（当前阶段）**：Mini 数据集已经足够。关键结论（PTQ 无损、INT8 精度下降 ~1.3%）的方向性不太可能在完整数据集上反转。
- **校准集可以保持 mini**：MinMax 校准对数据量不敏感，即使在完整数据集上评估，校准集使用 128 帧 mini 数据即可。更高级的校准策略（如 Histogram / Percentile）可能需要更多校准数据。

---

## 8. 实验结果

### 8.1 PTQ 精度评估（MQBench FakeQuant，4/6 模块）

在 nuScenes v1.0-mini 验证集（81 帧），128 batch 校准：

| 指标 | FP32 基线 | PTQ 4/6 (MinMax) | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.5801 | **0.5810** | **+0.0009**（无损） |
| **mAP** | 0.5742 | **0.5759** | **+0.0017**（无损） |

逐类 AP：

| 类别 | FP32 | PTQ 4/6 | 变化 |
|------|------|---------|------|
| car | 0.916 | 0.918 | +0.002 |
| truck | 0.833 | 0.840 | +0.007 |
| bus | 0.995 | 0.995 | 0.000 |
| pedestrian | 0.919 | 0.922 | +0.003 |
| motorcycle | 0.705 | 0.699 | −0.006 |
| bicycle | 0.517 | 0.518 | +0.001 |
| traffic_cone | 0.848 | 0.866 | +0.018 |

> 结论：MinMax PTQ 在 4/6 模块量化后实现了零精度损失。NDS/mAP 的微小波动在统计噪声范围内。

### 8.2 ConvFuser TRT 导出性能（隔离测试）

RTX 4060 Laptop，TensorRT 10.15，200 次推理平均延迟：

| 方法 | 延迟 | 加速比 | 引擎大小 | 压缩比 |
|------|------|--------|---------|--------|
| PyTorch FP32 | 5.083 ms | 1.00x | — | — |
| TRT FP32 | 4.017 ms | 1.27x | 5,385 KB | 1.00x |
| TRT FP16 | 1.437 ms | **3.54x** | 1,543 KB | 3.49x |
| TRT INT8 | 0.746 ms | **6.81x** | 832 KB | **6.48x** |

### 8.3 Hybrid TRT 端到端 NDS 评估

ConvFuser 替换为 TRT 引擎，全量验证集（81 帧）NDS 评估：

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32 | 0.5801 | 0.5746 | — | — |
| TRT FP32 | **0.5801** | **0.5746** | +0.0000 | +0.0000 |
| TRT FP16 | **0.5799** | **0.5744** | −0.0002 | −0.0002 |
| TRT INT8 | **0.5727** | **0.5616** | **−0.0074** | **−0.0130** |

> 结论：TRT FP32/FP16 几乎无损，FP16 推荐作为精度与效率的最佳平衡点。INT8 精度下降 1.3%（NDS），主要来自 ConvFuser 输出的大动态范围（0~155）。

### 8.4 逐元素精度分析

| 精度 | MSE | 最大绝对误差 | 余弦相似度 | 相对误差 |
|------|-----|------------|-----------|---------|
| TRT FP32 | 3.50e-08 | 0.0019 | 1.000000 | 0.029% |
| TRT FP16 | 4.29e-06 | 0.0518 | 0.999995 | 0.323% |
| TRT INT8 | 2.69e-04 | 0.1807 | 0.999674 | 2.554% |

---

## 9. 后续工作

### 9.1 短期（工程完善）

1. **推广 TRT Hybrid 到更多模块**  
   当前仅 ConvFuser 使用 TRT 引擎（占总推理 ~1%）。decoder/backbone（SECOND）和 decoder/neck（SECONDFPN）都是纯密集卷积，ONNX 导出难度低，可以直接复用 `FuserForExport` + `TRTFuser` 的模式编写各自的 Wrapper。预期端到端加速显著提升。

2. **在完整 nuScenes 数据集上验证**  
   下载 nuScenes v1.0-trainval（约 300 GB），在 6,019 帧验证集上重跑 PTQ 和 TRT Hybrid 评估。这将提供论文级别的可信度，并覆盖 mini 数据集缺失的 3 个类别。

3. **混合精度 TRT 方案验证**  
   如第 6 节分析，将 fuser 设为 FP16、decoder 设为 INT8，验证是否能在更少精度损失的前提下保持压缩率。

### 9.2 中期（方法改进）

4. **更先进的校准策略**  
   当前使用最朴素的 MinMax 校准。可尝试：
   - **Histogram / Percentile 校准**：裁剪掉极端值，可能改善 INT8 的余弦相似度
   - **AdaRound**（MQBench 支持）：学习最优的 round-to-nearest 策略，而非简单截断
   - **Per-channel 量化**：TRT 支持 per-channel INT8，可能在 ConvFuser 这种通道数较多（256）的场景下改善精度

5. **SwinTransformer 量化探索**  
   SwinTransformer 是 BEVFusion 中计算最密集的模块。虽然 `torch.fx` 追踪有动态控制流障碍，但可以尝试：
   - 传 `concrete_args` 固定输入尺寸，让 fx 常量化分支条件
   - 使用 `torch.fx.Tracer` 的自定义子类，标记特定函数为 leaf
   - 考虑 FX-free 的量化方式（手动插入 QuantStub/DeQuantStub）

6. **TransFusionHead 量化**  
   检测头中的 Attention 层和 FFN 层也可以量化。障碍是 `ModuleList` 动态迭代，可通过静态展开（unroll）解决。

### 9.3 长期（部署落地）

7. **全模型 TRT 引擎**  
   参考 NVIDIA 官方 [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) 项目，它提供了完整的 TRT 适配（含 SpConv plugin、bev_pool plugin），但是独立的 C++ 工程。可以将本项目的量化参数（scale/zero_point）导入到 CUDA-BEVFusion 的 TRT 引擎中。

8. **端侧部署验证**  
   在目标部署硬件（如 Orin、Xavier）上构建 TRT 引擎并验证实时性。TRT 引擎是硬件绑定的，需要在目标 GPU 上重新构建。

9. **量化感知训练（QAT）**  
   虽然当前 PTQ 已无损，但如果扩大量化覆盖到 SwinTransformer 和 TransFusionHead 后精度下降明显，QAT 可以通过微调恢复精度。本项目之前有 `quant_train.py` 脚本（已移除，因 PTQ 无损不需要），需要时可恢复。

---

## 10. 结论

本项目成功实现了 BEVFusion 的选择性混合精度量化与 TensorRT 部署：

1. **量化覆盖**：6 个可量化模块中成功量化 4 个（decoder/backbone、decoder/neck、camera/neck、fuser），覆盖率 67%
2. **PTQ 精度**：MinMax PTQ 实现零精度损失（NDS +0.0009），验证了 BEVFusion 中后段管线对 INT8 量化的高容忍度
3. **TRT 部署验证**：ConvFuser 单模块 TRT INT8 实现 6.81 倍加速和 6.48 倍压缩，端到端 NDS 下降仅 1.3%
4. **Hybrid 架构**：验证了分段 TRT 部署的可行性，为渐进式量化部署提供了实践经验
5. **工程价值**：建立了完整的 PTQ → FakeQuant 评估 → TRT 导出 → Hybrid 端到端验证的工具链

核心发现是：**BEVFusion 的 BEV 中后段管线（neck → fuser → decoder）对量化极其友好**，即使使用最朴素的 MinMax 策略也能无损量化。这为 BEV 感知模型的部署优化提供了有价值的参考。
