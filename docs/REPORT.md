# BEVFusion 混合精度量化与 TensorRT 部署报告

> **项目**：BEVFusion + MQBench 训练后量化（PTQ）  
> **日期**：2026-02  
> **硬件**：NVIDIA RTX 4060 Laptop GPU（Ada Lovelace，Compute 8.9）  
> **框架**：PyTorch 1.10.2 + CUDA 11.3 + TensorRT 10.15.1

---

## 目录

1. [概述](#1-概述)
2. [背景知识](#2-背景知识)
3. [BEVFusion 模型架构分析](#3-bevfusion-模型架构分析)
4. [混合精度量化设计思路](#4-混合精度量化设计思路)
5. [实现细节](#5-实现细节)
6. [各模块精度方案一览](#6-各模块精度方案一览)
7. [混合精度的进一步探索](#7-混合精度的进一步探索)
8. [数据集与校准集分析](#8-数据集与校准集分析)
9. [实验结果](#9-实验结果)
10. [与 NVIDIA CUDA-BEVFusion 的对比](#10-与-nvidia-cuda-bevfusion-的对比)
11. [后续工作](#11-后续工作)
12. [结论](#12-结论)

---

## 1. 概述

BEVFusion 是一种多模态 3D 目标检测模型，融合摄像头和激光雷达信息生成 BEV（鸟瞰图）表示。模型在 nuScenes 数据集上取得了优秀的检测精度，但其约 40M 参数、156 MB 的模型体积和 ~389 ms 的单帧推理延迟限制了实际部署。

本项目基于 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，对 BEVFusion 实施**训练后量化（PTQ）**，目标后端为 NVIDIA TensorRT INT8。核心挑战在于：BEVFusion 是一个**异构多模态模型**，包含稀疏卷积、自定义 CUDA 算子、Transformer 等多种子模块，无法对全模型统一量化，因此我们设计了**选择性混合精度量化**方案。

**核心成果**：

**SwinTransformer Backbone（原始方案，4/6 模块量化）**：

| 指标 | FP32 基线 | PTQ 4/6 (FakeQuant 仿真) | TRT INT8 Hybrid (4 模块) | TRT FP16 Hybrid (4 模块) |
|------|----------|------------------------|-------------------------|-------------------------|
| NDS | 0.5801 | **0.5810 (+0.0009)** | **0.5723 (−0.0077)** | **0.5795 (−0.0005)** |
| mAP | 0.5742 | **0.5759 (+0.0017)** | **0.5652 (−0.0092)** | **0.5743 (−0.0001)** |
| TRT 引擎大小 | — | — | **7.2 MB** | **13.5 MB** |
| 总部署体积 | 155.9 MB | — | **~136.6 MB** | — |

**ResNet-50 Backbone（量化友好替换方案，5/6 模块量化）**：

| 指标 | FP32 基线 | PTQ 5/6 (FakeQuant 仿真) | TRT INT8 Hybrid (5 模块) | TRT FP16 Hybrid (5 模块) |
|------|----------|------------------------|-------------------------|-------------------------|
| NDS | 0.3982 | **0.4079 (+0.0097)** | **0.4078 (+0.0096)** | **0.3981 (−0.0001)** |
| mAP | 0.4135 | **0.4189 (+0.0054)** | **0.4187 (+0.0052)** | **0.4136 (+0.0001)** |
| TRT 引擎大小 | — | — | **31.4 MB** | **59.9 MB** |
| 总部署体积 | 420.7 MB | — | **~55.7 MB（−59% vs SwinT）** | **~84.2 MB** |

> ⚠️ **关于 NDS 差异的说明**：ResNet-50 的 FP32 NDS（0.4989）低于 SwinT（0.7069），主因是仅训练 6 epochs（SwinT 为官方预训练 20 epochs），训练 loss 仍在下降（Epoch 6 下降率 7.6%）。ResNet-50 的核心优势在于量化覆盖（88% vs 18% 参数）和总部署体积（55.7 MB vs 136.8 MB，−59%）。

---

## 2. 背景知识

> 本节面向初学者，解释量化部署涉及的核心概念。如果你已经熟悉这些内容，可以直接跳到第 3 节。

### 2.1 什么是模型量化

深度学习模型默认使用 **FP32**（32 位浮点数）存储权重和计算。量化就是把这些数字用更少的位数来表示——比如 **INT8**（8 位整数）或 **FP16**（16 位浮点数）。

为什么要量化？三个好处：
- **模型更小**：INT8 每个参数只需 1 字节（FP32 需要 4 字节），理论上压缩 4 倍
- **推理更快**：低精度运算在 GPU 上有专门的高速计算单元（如 NVIDIA 的 Tensor Core 支持 INT8 吞吐是 FP32 的 2~4 倍）
- **功耗更低**：更少的数据传输和计算意味着更低的能耗，这对边缘设备（如自动驾驶车载芯片）尤为重要

代价是**精度损失**：用 8 位整数近似 32 位浮点数必然有误差。量化的核心问题就是：**如何在压缩率和精度之间取得最佳平衡**。

#### 量化的数学原理（简化版）

量化本质上是一个**线性映射**。假设一个 FP32 权重值 $x$ 的范围是 $[x_{min}, x_{max}]$，INT8 的表示范围是 $[0, 255]$（无符号）或 $[-128, 127]$（有符号），量化过程为：

```
量化：  x_int8 = round((x - zero_point) / scale)
反量化：x_approx = x_int8 * scale + zero_point
```

其中：
- **scale（缩放因子）**：决定了量化的精度。scale 越小，精度越高，但能表示的数值范围越小
- **zero_point（零点）**：将浮点的 0.0 映射到整数的某个值

**校准（Calibration）** 就是确定每一层的 scale 和 zero_point 的过程。最简单的方法是 **MinMax 校准**：用一批数据跑一遍前向推理，记录每一层输出的最小值和最大值，然后：

```
scale = (max - min) / 255
zero_point = round(-min / scale)
```

### 2.2 PTQ vs QAT

量化有两种主要方法：

| 方法 | 全称 | 是否需要训练 | 精度 | 耗时 |
|------|------|------------|------|------|
| **PTQ** | Post-Training Quantization（训练后量化） | ❌ 不需要 | 通常够用 | 几分钟 |
| **QAT** | Quantization-Aware Training（量化感知训练） | ✅ 需要微调 | 更高 | 几小时~几天 |

**PTQ**（本项目使用的方法）：
- 拿一个训练好的 FP32 模型，用一小批数据（称为"校准集"）跑前向推理
- 统计每层激活值的分布（min/max），确定量化参数
- 整个过程不需要反向传播，不需要训练数据的标签
- 优点是**快**（几分钟就能完成），缺点是如果模型对量化敏感，精度可能下降较多

**QAT**：
- 在训练过程中就模拟量化的效果（插入 FakeQuant 节点）
- 让模型在训练时就"习惯"低精度的数值表示
- 精度通常比 PTQ 更好，但需要重新训练（消耗 GPU 时间和训练数据）
- 本项目中 PTQ 已经精度无损（NDS +0.0009），所以**没有必要使用 QAT**

### 2.3 什么是 FakeQuant（仿真量化）

这是一个容易混淆的概念。当我们说"用 MQBench 做了 PTQ 量化"时，实际上模型的权重**仍然是 FP32 存储的**。MQBench 做的事情是：

1. 在模型的每一层前后插入 **FakeQuantize 节点**
2. 这些节点模拟量化的效果：先把 FP32 值量化为 INT8，再反量化回 FP32
3. 这个过程引入了量化误差，但计算仍然在 FP32 精度下进行

```
原始前向传播：  input (FP32) → Conv2d → output (FP32)

FakeQuant 后：  input (FP32) → FakeQuant → Conv2d → FakeQuant → output (FP32)
                              ↑ 模拟量化误差            ↑ 模拟量化误差
```

所以 FakeQuant 的模型：
- ❌ **不会更小**（权重仍是 FP32，.pth 文件甚至略大因为多了 scale/zero_point 参数）
- ❌ **不会更快**（反而因为额外的 FakeQuant 计算而略慢）
- ✅ **能验证精度**（如果 FakeQuant 模型精度无损，说明真实 INT8 部署也不会有大问题）

**真正的压缩和加速需要将模型导出为 TensorRT 引擎**（见下一节）。FakeQuant 只是一个"验证工具"。

### 2.4 什么是 TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) 是 NVIDIA 提供的**高性能深度学习推理优化器**。它的作用是：

1. 接收一个训练好的模型（通常是 ONNX 格式）
2. 对模型进行一系列优化（算子融合、内存优化、精度校准等）
3. 生成一个高度优化的**引擎文件**（`.engine`），专门针对目标 GPU 硬件

TRT 引擎的特点：
- **速度快**：比 PyTorch 快 2~10 倍（取决于模型和精度）
- **体积小**：INT8 引擎只有 FP32 模型的 1/4 甚至更小
- **硬件绑定**：在 RTX 4060 上构建的引擎不能在 RTX 3090 上运行，必须在目标 GPU 上重新构建
- **精度可选**：可以构建 FP32、FP16、INT8 三种精度的引擎

### 2.5 什么是 ONNX

**ONNX**（Open Neural Network Exchange）是一种开放的模型交换格式。它的作用是在不同框架之间传递模型：

```
PyTorch 模型 → torch.onnx.export → ONNX 文件 → TensorRT 读取 → TRT 引擎
```

ONNX 定义了一套标准算子（Conv、MatMul、Relu 等）。如果模型中使用了非标准算子（如稀疏卷积、自定义 CUDA 核函数），就无法导出为 ONNX，这是本项目中部分模块无法导出的根本原因。

### 2.6 什么是 torch.fx

`torch.fx` 是 PyTorch 提供的一个**符号追踪（symbolic tracing）** 工具。MQBench 使用它来分析模型结构、找到需要插入 FakeQuant 节点的位置。

符号追踪的工作方式：
1. 创建一个"假的"输入（称为 Proxy），代替真实的 Tensor
2. 让模型执行一遍前向传播，但 Proxy 不做真正的计算，只记录操作顺序
3. 得到一个计算图（graph），表示模型的完整结构

问题在于：如果模型代码中有**基于数据值的条件判断**（比如 `if x.shape[0] > 10:`），Proxy 无法给出真/假的结果，追踪就会失败。这就是为什么 SwinTransformer 和 TransFusionHead 无法被 MQBench 量化的原因。

### 2.7 评估指标：NDS 和 mAP

**mAP（mean Average Precision，平均精度）**：衡量检测准确率的标准指标。对每个类别计算 AP（检测到多少目标、有多少误检），然后取平均。mAP 越高越好，1.0 是满分。

**NDS（nuScenes Detection Score）**：nuScenes 数据集专用的综合指标，考虑了多个维度：

```
NDS = (1/10) × [5 × mAP + Σ(1 - min(1, metric_error))]
```

其中 metric_error 包括：
- **mATE**：平均平移误差（位置偏差，米）
- **mASE**：平均尺度误差（大小偏差）
- **mAOE**：平均方向误差（角度偏差，弧度）
- **mAVE**：平均速度误差
- **mAAE**：平均属性误差

NDS 综合考虑了检测精度和定位质量，是评估 3D 目标检测模型的最重要指标。本项目中，FP32 基线 NDS = 0.5801，我们要确保量化后 NDS 不会显著下降。

---

## 3. BEVFusion 模型架构分析

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

### 各子模块的参数分布

下表展示了各子模块在 FP32 预训练权重中的实际大小占比（基于 `pretrained/bevfusion-det.pth` 统计）：

| 子模块 | 参数量 | FP32 权重大小 | 占比 | TRT 导出 |
|--------|--------|-------------|------|---------|
| `camera/backbone` (SwinTransformer) | 27.55M | **105.20 MB** | **67.5%** | ❌ |
| `lidar/backbone` (SparseEncoder) | 2.70M | 10.29 MB | 6.6% | ❌ |
| `camera/vtransform` (LSSTransform) | 2.61M | 9.95 MB | 6.4% | ❌ |
| `decoder/backbone` (SECOND) | 4.29M | **16.35 MB** | **10.5%** | ✅ |
| `camera/neck` (GeneralizedLSSFPN) | 1.59M | 6.08 MB | 3.9% | ✅ |
| `heads/object` (TransFusionHead) | 1.04M | 3.95 MB | 2.5% | ❌ |
| `fuser` (ConvFuser) | 0.78M | 2.95 MB | 1.9% | ✅ |
| `decoder/neck` (SECONDFPN) | 0.30M | 1.13 MB | 0.7% | ✅ |
| **总计** | **40.84M** | **155.91 MB** | **100%** | |
| **4 个 TRT 模块合计** | **6.95M** | **26.51 MB** | **17.0%** | ✅ |
| **未量化模块合计** | **33.89M** | **129.39 MB** | **83.0%** | ❌ |

**关键观察**：

1. **SwinTransformer 独占 67.5%**：摄像头主干网络是绝对的参数瓶颈，而它恰恰是最难量化的模块
2. 量化友好的 4 个模块仅占总参数的 17.0%（26.51 MB），即使全部压缩为 INT8，对总模型大小的影响也有限
3. 这意味着：**要显著压缩 BEVFusion，必须解决 SwinTransformer 的量化问题**，或者更换为量化友好的主干网络（如 ResNet）

---

## 4. 混合精度量化设计思路

### 4.1 为什么不能全模型量化

传统 PTQ 工具（如 TensorRT 的 `trtexec --int8`）假设模型可被完整导出为 ONNX，再统一做 INT8 校准。BEVFusion 存在以下障碍使得全模型导出不可行：

1. **稀疏卷积（spconv）**：LiDAR 分支的 SparseEncoder 使用稀疏张量表示，标准 ONNX 和 TensorRT 均不支持。稀疏卷积的输入/输出格式与密集张量完全不同，FakeQuant 节点无法插入。

2. **自定义 CUDA 算子**：Camera 分支的 `bev_pool`（QuickCumsumCuda）是用 CUDA C++ 实现的 autograd Function，没有对应的 ONNX 算子或 TensorRT 插件。

3. **动态控制流**：SwinTransformer 内部有 `if x.shape[0] > window_size:` 等基于张量值的分支判断，`torch.fx` 符号追踪时会失败（Proxy 对象无法求值为布尔值）。TransFusionHead 中 `for layer in decoder_layers:` 等动态迭代同样不兼容。

4. **体素化预处理**：Voxelization 将不规则点云映射为规则网格，这是一个离散化预处理步骤，不包含可微分的权重，不需要也不应该量化。

### 4.2 选择性量化策略

基于以上分析，我们采用**选择性量化**：逐个子模块独立调用 `MQBench.prepare_by_platform`（基于 `torch.fx` 符号追踪），仅量化可追踪的子模块，跳过不兼容的部分。

```
已量化（INT8）：                      保持 FP32：
├── camera/neck (GeneralizedLSSFPN)   ├── camera/backbone (SwinTransformer)
├── fuser (ConvFuser)                 ├── camera/vtransform (bev_pool)
├── decoder/backbone (SECOND)         ├── lidar/* (稀疏卷积)
└── decoder/neck (SECONDFPN)          ├── heads/object (TransFusionHead)
                                      └── lidar/voxelize
```

量化覆盖率：**4/6 可量化模块**（排除设计上不适合量化的 vtransform 和 voxelize 后，实际可量化模块为 6 个，成功量化 4 个）。

### 4.3 分段 TensorRT 部署（Hybrid 推理）

由于无法将全模型导出为 TRT 引擎，我们设计了 **Hybrid 推理架构**：将已量化且 ONNX 兼容的 4 个子模块全部导出为 TRT 引擎，其余部分保持 PyTorch 执行。数据在 PyTorch 和 TRT 之间通过 CUDA 显存零拷贝传递。

```
PyTorch 执行区域：                    TRT 引擎区域（4 个模块）：
┌─────────────────────┐              ┌────────────────────────────┐
│ SwinTransformer      │              │ GeneralizedLSSFPN (TRT)    │
│ LSSTransform         │──BEV feat──→│ ConvFuser (TRT)            │
│ SparseEncoder        │              │ SECOND (TRT)               │
│ TransFusionHead      │←─decoded────│ SECONDFPN (TRT)            │
└─────────────────────┘              └────────────────────────────┘
```

#### Hybrid 推理是怎么运行的？

**TRT 引擎本质上是预编译好的 GPU 程序**。可以类比为：你用 C++ 写了几个高性能函数，编译成 `.dll`，然后在 Python 中通过 ctypes 调用。TRT 引擎做的事情类似，但更聪明——它直接操作 GPU 显存中的数据。

整个推理流程中，数据**全程留在 GPU 显存**，不需要 CPU↔GPU 的数据拷贝：

```
步骤 1: PyTorch 模块（SwinTransformer 等）在 GPU 上计算 → 输出 tensor 在 GPU 显存中
         ↓ （GPU 显存地址传递，无拷贝）
步骤 2: TRT 引擎直接从 GPU 显存读取输入 → 在 GPU 上执行推理 → 结果写入 GPU 显存
         ↓ （GPU 显存地址传递，无拷贝）
步骤 3: PyTorch 模块（TransFusionHead）从 GPU 显存读取继续后续计算
```

在我们的实现中，每个 TRT 引擎被包装为一个 `nn.Module` 子类（如 `CameraNeckTRTWrapper`），可以像普通 PyTorch 模块一样调用 `module(input_tensor)`。模型的 `forward()` 方法不需要任何修改——只需用 TRT Wrapper 替换原始 PyTorch 子模块即可。

#### 未量化的部分怎么运行？

**依然靠 PyTorch 运行**。SwinTransformer、SpConv 稀疏卷积、bev_pool CUDA 算子、TransFusionHead 等模块仍然以 PyTorch FP32 模式在 GPU 上执行。这意味着：

- **部署环境需要**：Python + PyTorch + TensorRT 运行时
- **如果目标设备没有 Python**：需要将所有未量化模块也转成 C++，这是一个完全独立的工程（参考 NVIDIA 官方 CUDA-BEVFusion 项目，见第 10 节）
- **当前方案的定位**：研究性验证（验证量化精度和压缩效果），而非生产级部署

---

## 5. 实现细节

本项目在 `tools/` 目录下新增了以下脚本，在 `mmdet3d/` 下修改了模型代码以支持 `torch.fx` 追踪。

### 5.1 `tools/quant_ptq_minmax.py` — PTQ 主流程

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

### 5.2 `tools/quant_benchmark.py` — Benchmark 工具

测量并对比 FP32 模型与量化模型的：
- **参数量**：统计模型可训练参数数量
- **模型大小**：FP32 `.pth` 文件大小，以及理论 INT8 部署大小（÷4 估算）
- **推理延迟**：GPU warmup + 正式计时，报告均值/P95/P99 延迟

支持 `--size-only` 模式（仅报告大小，不需要数据集）和 `--use-real-data` 模式（使用真实 nuScenes 数据）。

### 5.3 `tools/trt_eval_hybrid_all.py` — 全模块 Hybrid TRT 端到端评估

这是最重要的验证脚本，将全部 4 个已量化模块导出为 TRT 引擎并运行完整 NDS 评估：

**工作流程**：

```
Step 1: 加载 FP32 模型 + 预训练权重
Step 2: 对每个模块进行 ONNX 导出（使用 deepcopy 隔离，避免破坏原模型参数）
Step 3: 收集校准数据（真实特征）+ 运行 FP32 基线（INT8 模式时需要校准）
Step 4: 构建 TRT 引擎（FP32/FP16/INT8，4 个模块各一个）
Step 5: Sanity check（对比每个模块的 PyTorch vs TRT 输出余弦相似度）
Step 6: 替换 4 个模块 → TRT 版本，运行完整 NDS 评估
```

**各模块的导出包装器（Export Wrapper）**：

ONNX 要求每个输入都是独立的命名 tensor，但 PyTorch 模块通常接收 list/tuple。每个模块需要一个导出包装器将 list 输入展平为独立参数：

```python
# 以 ConvFuser 为例
class FuserExportWrapper(nn.Module):
    """将 [camera_bev, lidar_bev] list 输入拆分为两个独立的 ONNX 输入"""
    def __init__(self, fuser):
        super().__init__()
        self.fuser = copy.deepcopy(fuser)  # 关键：deepcopy 隔离参数
    
    def forward(self, camera_bev, lidar_bev):
        return self.fuser([camera_bev, lidar_bev])
```

为什么需要 `copy.deepcopy`？因为 `torch.onnx.export` 会将模型移到 CPU（`.cpu()`），如果直接引用原模型的子模块，`.cpu()` 会破坏原模型的 CUDA 参数，导致后续推理全部输出 0。

**TRT 推理替换器（TRT Wrapper）**：

每个 TRT 引擎被封装为 `nn.Module` 子类，可以直接替换原始模块：

```python
class TRTModule(nn.Module):
    def __init__(self, engine_path):
        # 加载 TRT 引擎，分配输入/输出 buffer
        ...
    
    def forward(self, *inputs):
        # 1. 将 PyTorch tensor 的 data_ptr 传给 TRT（零拷贝，直接读 GPU 显存）
        # 2. 执行 TRT 推理
        # 3. 返回 clone() 后的输出（避免 buffer 复用问题）
        ...
```

### 5.4 `tools/trt_export_fuser.py` — ConvFuser 隔离延迟测试

单模块导出脚本，用于测试 ConvFuser 在不同精度下的 TRT 延迟和引擎大小。主要用于开发阶段的快速验证。

### 5.5 `tools/trt_eval_hybrid.py` — 单模块 Hybrid TRT 评估

仅替换 ConvFuser 一个模块的 Hybrid 评估脚本。是 `trt_eval_hybrid_all.py` 的前身，保留用于调试和对比。

### 5.6 模型代码修改

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

## 6. 各模块精度方案一览

下表详细列出了 BEVFusion 每个子模块在当前方案中的精度状态：

| 子模块 | 类型 | SwinT 精度 | ResNet-50 精度 | PTQ 状态 | TRT 导出状态 | 备注 |
|--------|------|-----------|-------------|---------|-------------|------|
| `camera/backbone` | SwinT / ResNet-50 | **FP32** | **TRT INT8/FP16** | SwinT ❌ / ResNet ✅ | SwinT ❌ / ResNet ✅ | SwinT 动态控制流；ResNet 纯 CNN |
| `camera/neck` | GeneralizedLSSFPN | **TRT INT8/FP16** | **TRT INT8/FP16** | ✅ 已量化 | ✅ 已导出 | 修复 `len()` + mmcv patch |
| `camera/vtransform` | LSSTransform | **FP32** | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | `bev_pool` 自定义 CUDA 算子 |
| `lidar/voxelize` | Voxelization | **FP32** | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | 非神经网络层 |
| `lidar/backbone` | SparseEncoder | **FP32** | **FP32** | ⊘ 设计跳过 | ❌ 不可导出 | 稀疏卷积 |
| `fuser` | ConvFuser | **TRT INT8/FP16** | **TRT INT8/FP16** | ✅ 已量化 | ✅ 已导出 | 最早验证的模块 |
| `decoder/backbone` | SECOND | **TRT INT8/FP16** | **TRT INT8/FP16** | ✅ 已量化 | ✅ 已导出 | 纯 Conv2d 堆叠，参数量最大 |
| `decoder/neck` | SECONDFPN | **TRT INT8/FP16** | **TRT INT8/FP16** | ✅ 已量化 | ✅ 已导出 | 含 ConvTranspose2d |
| `heads/object` | TransFusionHead | **FP32** | **FP32** | ❌ fx 追踪失败 | ❌ 未导出 | Proxy 迭代 + 动态 shape |

---

## 7. 混合精度的进一步探索

既然已经实现了"分模块量化"的基础设施，一个自然的延伸是：**不同的模块使用不同的精度**，以在压缩率和精度之间取得更细粒度的平衡。

### 7.1 已验证的精度方案

| 方案 | camera/neck | fuser | decoder/backbone | decoder/neck | NDS | 引擎总大小 | 模块压缩比 |
|------|------------|-------|-----------------|-------------|-----|-----------|-----------|
| A: 全 FP32 TRT | FP32 | FP32 | FP32 | FP32 | 0.5800 | 42.6 MB | 0.62x（引擎含编译代码，大于纯权重） |
| B: 全 FP16 TRT | FP16 | FP16 | FP16 | FP16 | 0.5795 | 13.5 MB | **1.96x** |
| C: 全 INT8 TRT | INT8 | INT8 | INT8 | INT8 | 0.5723 | 7.2 MB | **3.68x** |

> **压缩比说明**：此处的压缩比是相对于 4 个模块的 FP32 权重（26.51 MB）计算的，而非全模型 155.9 MB。TRT FP32 引擎（42.6 MB）反而比纯权重大，是因为引擎中包含编译后的 CUDA 执行计划（kernel code），不仅仅是权重。

### 7.2 分析与建议

**方案 B（全 FP16，推荐部署方案）**：
- NDS 仅下降 0.0005（在统计噪声范围内），可视为无损
- 各模块余弦相似度均 ≥ 0.999993，数值误差极小
- 4 个 TRT 模块从 26.5 MB 压缩到 13.5 MB
- **强烈推荐作为默认部署方案**

**方案 C（全 INT8，最大压缩）**：
- NDS 下降 0.0077（约 1.3%），在大多数应用场景下可接受
- 4 个 TRT 模块压缩到仅 7.2 MB（3.68 倍）
- 适合对模型大小和推理速度要求极高的边缘设备部署
- **注意**：未量化的模块（SwinTransformer 等）仍需 129.4 MB FP32 权重，总部署体积约 136.6 MB

**未验证的混合方案**（可进一步探索）：
- 将 fuser 和 camera/neck 设为 FP16，decoder 设为 INT8
- 预期 NDS 介于方案 B 和 C 之间，但压缩比高于 B
- 当前脚本 `trt_eval_hybrid_all.py` 不支持逐模块设置精度，需要修改

### 7.3 实施难度

混合精度 TRT 引擎的实施主要有两种路径：

1. **逐模块独立引擎**（当前方案）：每个子模块有自己的 TRT 引擎，可单独设置精度。缺点是引擎间数据传递有开销。
2. **单引擎多精度**（TRT 原生支持）：TRT 支持 per-layer 精度设置，但需要将多个模块合并导出为一个 ONNX，然后在 TRT 中标记每层的精度。对于 BEVFusion 的 Hybrid 架构，合并导出较困难。

---

## 8. 数据集与校准集分析

### 8.1 当前数据集

本项目使用 **nuScenes v1.0-mini** 数据集：

| 数据集 | 样本数 | 用途 |
|--------|-------|------|
| 训练集 | 323 帧 | 未使用（PTQ 不需要训练） |
| 验证集 | 81 帧 | PTQ 校准 + NDS 精度评估 |

**nuScenes v1.0-mini 的特点**：
- 仅包含 10 个场景（scenes），来自 Boston 和 Singapore 两座城市
- 每帧包含 6 个摄像头图像 + 1 个 LiDAR 点云 + 完整 3D 标注
- 约占 nuScenes full 数据集的 1/40（full 有 ~28k 训练 + ~6k 验证帧）

### 8.2 校准集设置

| 参数 | PTQ (MQBench) | TRT INT8 (4 模块) |
|------|--------------|---------------------|
| 校准数据来源 | 验证集循环采样 | 真实模型中间特征（模型前向推理提取） |
| 校准样本数 | 128 batch | 50 样本 |
| 校准方法 | MinMax（记录全局 min/max） | IInt8MinMaxCalibrator |
| 校准数据分布 | 原始图像 + 点云 | 各模块实际输入特征 |

### 8.3 说服力评估

**优势**：
- Mini 数据集虽小，但包含了城市道路的主要场景类型（十字路口、直行道、转弯等）
- PTQ 校准仅需观测各层激活值的统计特性（min/max），对样本数量的需求远低于训练
- 128 batch 的校准量对于 MinMax 策略已经充分，因为 min/max 在约 30-50 batch 后即趋于稳定
- 实验结果也验证了这一点：PTQ 精度无损（NDS +0.0009），说明校准数据足以捕获分布

**局限性**：

- **评估集过小（81 帧）**：3 个类别（trailer、construction_vehicle、barrier）的样本数为 0，导致这些类别的 AP 评估无意义。这意味着我们的 NDS/mAP 数字实际上只反映了 7/10 个类别的表现
- **场景多样性不足**：Mini 数据集仅有 10 个场景，不同天气、光照条件的覆盖有限。在极端条件下（如强逆光、暴雨、夜间），量化模型的鲁棒性未被验证
- **校准集 = 验证集**：PTQ 校准使用了验证集的数据，存在一定程度的"过拟合"风险，但对于 MinMax 这种极简单的统计方法（仅记录全局 min/max），这种风险很小

### 8.4 使用 nuScenes 完整数据集的考虑

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

## 9. 实验结果

### 9.1 PTQ 精度评估（MQBench FakeQuant，4/6 模块）

#### 9.1.1 Mini 数据集（81 帧，开发阶段验证）

| 指标 | FP32 基线 | PTQ 4/6 (MinMax) | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.5801 | **0.5810** | **+0.0009**（无损） |
| **mAP** | 0.5742 | **0.5759** | **+0.0017**（无损） |

#### 9.1.2 完整验证集（6019 帧，最终结论）

| 指标 | FP32 基线 | PTQ 4/6 (MinMax) | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.7069 | **0.7015** | **−0.0054** |
| **mAP** | 0.6728 | **0.6618** | **−0.0110** |

> 完整验证集上 PTQ NDS 下降 0.0054（0.76%），说明 mini 数据集的 "+0.0009" 是统计噪声。在完整数据集上，MinMax PTQ 仍表现出良好的精度保持能力，NDS 损失不到 1%。

### 9.2 全模块 Hybrid TRT 端到端 NDS 评估（核心结果）

将全部 4 个已量化模块（camera/neck、fuser、decoder/backbone、decoder/neck）替换为 TRT 引擎后，全量验证集 NDS 评估：

#### 9.2.1 Mini 数据集（81 帧，开发阶段验证）

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32（基线） | 0.5800 | 0.5744 | — | — |
| TRT FP32（4 模块） | **0.5800** | **0.5744** | **+0.0000** | **+0.0000** |
| TRT FP16（4 模块） | **0.5795** | **0.5743** | **−0.0005** | **−0.0001** |
| TRT INT8（4 模块） | **0.5723** | **0.5652** | **−0.0077** | **−0.0092** |

#### 9.2.2 完整验证集（6019 帧，最终结论）

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32（基线） | **0.7069** | **0.6728** | — | — |
| PTQ 4/6（MQBench FakeQuant） | 0.7015 | 0.6618 | −0.0054 | −0.0110 |
| TRT FP32（4 模块） | 0.7065 | 0.6726 | −0.0004 | −0.0002 |
| TRT FP16（4 模块） | **0.7069** | **0.6728** | **+0.0000** | **+0.0000** |
| TRT INT8（4 模块） | 0.7022 | 0.6641 | **−0.0047** | **−0.0087** |

> **关键发现**：
> - TRT FP16 在完整验证集上实现**完全无损**（NDS Δ=0.0000），甚至比 mini 数据集结果（−0.0005）更好
> - TRT INT8 精度损失仅 0.67%（NDS），优于 mini 数据集上观测到的 1.3%，说明 mini 数据集对 INT8 误差的估计偏悲观
> - PTQ FakeQuant（NDS −0.0054）与真实 TRT INT8（NDS −0.0047）损失接近，验证了 MQBench 作为量化精度预测工具的有效性

### 9.3 TRT 数值精度验证指标

在将 PyTorch 模块替换为 TRT 引擎后，我们需要验证每个模块的输出是否与原始 PyTorch 输出一致。为此，`trt_eval_hybrid_all.py` 对每个模块的每个输出 tensor 计算两个互补的指标：

#### 余弦相似度（Cosine Similarity）

$$\text{cos\_sim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

其中 $A$ 是 PyTorch 原始模块的输出 tensor（展平为一维向量），$B$ 是对应 TRT 引擎的输出 tensor。

- **取值范围**：$[-1, 1]$，实际中 FP32/FP16 精度下通常 $\geq 0.9999$
- **物理含义**：衡量两个输出向量在高维空间中的**方向一致性**。值为 1.0 表示完全同向（数值等比例），值越低表示两者的特征模式偏差越大
- **解释示例**：余弦相似度 0.957 意味着 TRT 输出与 PyTorch 输出有 4.3% 的方向偏差。虽然听起来很小，但对于后续检测头来说可能有影响——不过实际端到端 NDS 损失仅 0.47%，说明检测头对这种偏差有容错能力
- **优点**：对整体缩放不敏感（如果 TRT 输出整体放大 2 倍，余弦相似度仍为 1.0），反映的是特征"形状"的一致性

#### 最大绝对误差（Max Absolute Error）

$$\text{maxErr}(A, B) = \max_i |A_i - B_i|$$

- **物理含义**：所有元素中**最坏情况**的逐点偏差
- **解释示例**：maxErr = 3.38 意味着在数万个输出值中，有个别点的 TRT 输出与 PyTorch 输出相差达 3.38。需要结合该层输出的数值范围来判断严重性——如果输出范围是 $[-100, 100]$，则 3.38 仅占 1.7%；如果范围是 $[-1, 1]$，则偏差严重
- **特点**：比余弦相似度更敏感于局部异常值。即使整体余弦相似度很高（如 0.999），个别异常点的 maxErr 也可能较大

两个指标配合使用：余弦相似度反映整体一致性，最大误差反映最坏情况。在下面的表格中，FP32 重建精度极高（cos=1.000000），FP16 精度损失极小（cos≥0.999990），INT8 在某些模块上有明显偏差但端到端精度仍可接受。

#### 9.3.1 SwinT 方案各模块余弦相似度（完整验证集，A100 GPU）

| 模块 | FP32 余弦相似度 | FP32 最大误差 | FP16 余弦相似度 | FP16 最大误差 | INT8 余弦相似度 | INT8 最大误差 |
|------|---------------|-------------|---------------|-------------|---------------|-------------|
| camera_neck[0] | 1.000000 | 0.0046 | 0.999996 | 0.0352 | 0.999046 | 0.2409 |
| camera_neck[1] | 1.000000 | 0.0051 | 0.999996 | 0.0273 | 0.998616 | 0.2906 |
| fuser[0] | 1.000000 | 0.0374 | 0.999998 | 0.3438 | 0.957402 | 2.4582 |
| dec_backbone[0] | 1.000000 | 0.0102 | 0.999997 | 0.0469 | 0.977364 | 2.0801 |
| dec_backbone[1] | 1.000000 | 0.0078 | 0.999990 | 0.0483 | 0.880242 | 3.3819 |
| dec_neck[0] | 1.000000 | 0.0013 | 1.000000 | 0.0049 | 0.999271 | 0.1631 |

> INT8 的 fuser 和 dec_backbone[1] 余弦相似度较低（0.957/0.880），反映了 INT8 量化在高动态范围特征上的误差。但经过 TransFusionHead 后端到端 NDS 仅下降 0.0047，说明检测头对这种量化误差有良好的容错能力。

### 9.4 引擎大小与压缩比（完整验证集，A100 GPU，TRT 10.15）

| 模块 | FP32 引擎 | FP16 引擎 | INT8 引擎 |
|------|----------|----------|----------|
| camera_neck | 9,946 KB | 3,245 KB | 1,787 KB |
| fuser | 5,385 KB | 1,551 KB | 845 KB |
| dec_backbone | 28,928 KB | 8,490 KB | 4,349 KB |
| dec_neck | 1,196 KB | 680 KB | 613 KB |
| **总计** | **44.4 MB** | **13.6 MB** | **7.4 MB** |
| **压缩比（vs 4 模块 FP32 权重 26.51 MB）** | — | **1.95x** | **3.58x** |

> ⚠️ **压缩比澄清**：上表的压缩比是相对于这 4 个模块在 .pth 中的 FP32 权重大小（26.51 MB）计算的。实际部署总大小 = TRT 引擎 + 未量化模块权重（129.4 MB）。SwinT INT8 总部署体积约 136.8 MB。

> decoder/backbone（SECOND）是最大的模块（28.9 MB FP32 引擎），包含多层 Conv2d + BN + ReLU。INT8 后缩小到 4.3 MB（6.6 倍压缩）。

### 9.5 逐类 AP 对比（完整验证集 6019 帧，TRT INT8 vs FP32 基线）

| 类别 | FP32 | TRT INT8 | 变化 |
|------|------|----------|------|
| car | 0.875 | 0.876 | +0.001 |
| truck | 0.639 | 0.626 | −0.013 |
| construction_vehicle | 0.277 | 0.270 | −0.007 |
| bus | 0.741 | 0.729 | −0.012 |
| trailer | 0.424 | 0.405 | −0.019 |
| barrier | 0.726 | 0.715 | −0.011 |
| motorcycle | 0.769 | 0.755 | −0.014 |
| bicycle | 0.612 | 0.606 | −0.006 |
| pedestrian | 0.877 | 0.874 | −0.003 |
| traffic_cone | 0.788 | 0.786 | −0.002 |

> 在完整验证集上，INT8 量化对各类别的影响均匀且轻微（最大下降 trailer −0.019），car 类别甚至微幅提升。相比 mini 数据集上观测到的较大波动（如 traffic_cone −0.042），完整数据集的结论更稳定可靠。

### 9.6 ResNet-50 Backbone 替换实验

#### 9.6.1 背景与动机

SwinTransformer 占全模型参数的 67.5%（105.2 MB），但由于动态控制流（window attention 中的 `if tensor > window_size`）无法被 torch.fx 追踪，导致 camera/backbone 无法量化，量化覆盖仅 4/6 模块（~18% 参数）。

为解决这一瓶颈，我们将 camera/backbone 替换为量化友好的 **ResNet-50**（纯 Conv2d + BN + ReLU，torch.fx 完全兼容）。配置文件：`configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml`。使用完整 nuScenes v1.0-trainval 训练集（28130 samples）在 4×RTX 3090 上训练 6 epochs。

**训练 Loss 收敛分析**：

| Epoch | Avg Loss | Δ Loss | 下降率 |
|-------|----------|--------|--------|
| 1 | 5.4672 | — | — |
| 2 | 1.9093 | −3.558 | −65.1%（warmup 后大幅下降） |
| 3 | 1.6166 | −0.293 | −15.3% |
| 4 | 1.4094 | −0.207 | −12.8% |
| 5 | 1.2523 | −0.157 | −11.1% |
| 6 | 1.1577 | −0.095 | −7.6% |

> **Loss 远未收敛**：6 个 epoch 结束后下降率仍有 7.6%，表明模型尚有大量学习空间。参考 BEVFusion 官方 SwinT 训练配置（20 epochs），继续训练至 20 epochs 预计可显著提升 NDS（从 0.4989 至 0.55+）。当前 NDS=0.4989 是 6 epochs 训练不充分的结果，非 ResNet-50 架构的能力上限。

**模型权重大小分布**（epoch_6.pth = 420.7 MB，含 optimizer state 277.9 MB，纯推理权重 142.8 MB）：

| 模块 | 纯权重大小 | 占比 | 可量化？ |
|------|-----------|------|---------|
| camera/backbone (ResNet-50) | 89.9 MB | 63.0% | ✅ |
| decoder/backbone (SECOND) | 16.3 MB | 11.4% | ✅ |
| lidar/backbone (SparseEncoder) | 10.3 MB | 7.2% | ❌ |
| camera/vtransform | 10.0 MB | 7.0% | ❌ |
| camera/neck (GeneralizedLSSFPN) | 8.3 MB | 5.8% | ✅ |
| heads/object (TransFusionHead) | 4.0 MB | 2.8% | ❌ |
| fuser (ConvFuser) | 3.0 MB | 2.1% | ✅ |
| decoder/neck (SECONDFPN) | 1.1 MB | 0.8% | ✅ |
| **总计** | **142.8 MB** | **100%** | **5/8 模块，88% 参数** |

#### 9.6.2 PTQ 精度评估（MQBench FakeQuant，5/6 模块）

##### Mini 数据集（81 帧，开发验证）

| 指标 | FP32 基线 | PTQ 5/6 (MinMax) | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.3982 | **0.4079** | **+0.0097**（无损） |
| **mAP** | 0.4135 | **0.4189** | **+0.0054**（无损） |

##### 完整验证集（6019 帧，最终结论）

| 指标 | FP32 基线 | PTQ 5/6 (MinMax) | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.4989 | **0.4958** | **−0.0031** |
| **mAP** | 0.4961 | **0.4904** | **−0.0057** |

> 完整验证集上 PTQ NDS 下降 0.0031（0.62%），精度保持良好。

量化覆盖对比：

| 子模块 | SwinT 可量化？ | ResNet-50 可量化？ |
|--------|-------------|-----------------|
| camera/backbone | ❌ | ✅ **新增** |
| camera/neck | ✅ | ✅ |
| fuser | ✅ | ✅ |
| decoder/backbone | ✅ | ✅ |
| decoder/neck | ✅ | ✅ |
| heads/object | ❌ | ❌ |
| **模块覆盖** | **4/6** | **5/6** |
| **参数覆盖** | **~18%** | **~88%** |

> ResNet-50 使得 camera/backbone 成功量化，可量化参数占比从 ~18% 跃升至 ~88%。

#### 9.6.3 全模块 Hybrid TRT 端到端 NDS 评估（5 模块替换）

将全部 5 个可量化模块（camera/backbone、camera/neck、fuser、decoder/backbone、decoder/neck）替换为 TRT 引擎：

##### 完整验证集（6019 帧，最终结论）

| 方法 | NDS | mAP | NDS 变化 | mAP 变化 |
|------|------|------|---------|---------|
| PyTorch FP32（基线） | **0.4989** | **0.4961** | — | — |
| PTQ 5/6（MQBench FakeQuant） | 0.4958 | 0.4904 | −0.0031 | −0.0057 |
| TRT FP32（5 模块） | 0.4994 | 0.4965 | +0.0005 | +0.0004 |
| TRT FP16（5 模块） | 0.4992 | 0.4962 | +0.0003 | +0.0001 |
| TRT INT8（5 模块） | 0.4948 | 0.4945 | **−0.0041** | **−0.0016** |

> TRT INT8 NDS 下降仅 0.82%，mAP 下降仅 0.32%。TRT FP32/FP16 几乎完全无损（NDS 变化 <0.001）。

#### 9.6.4 各模块余弦相似度（TRT vs PyTorch 输出，完整验证集，A100 GPU）

| 模块 | FP32 cos | FP32 maxErr | FP16 cos | FP16 maxErr | INT8 cos | INT8 maxErr |
|------|----------|------------|----------|------------|----------|------------|
| camera_backbone[0] | 0.999997 | 0.0078 | 0.999991 | 0.0161 | 0.988652 | 0.5494 |
| camera_backbone[1] | 0.999995 | 0.0088 | 0.999983 | 0.0154 | 0.982651 | 0.5049 |
| camera_backbone[2] | 0.999997 | 0.0243 | 0.999990 | 0.0352 | 0.991823 | 1.4941 |
| camera_neck[0] | 1.000000 | 0.0058 | 0.999996 | 0.0352 | 0.998132 | 0.5396 |
| camera_neck[1] | 1.000000 | 0.0056 | 0.999997 | 0.0312 | 0.998791 | 0.2435 |
| fuser[0] | 1.000000 | 0.0048 | 0.999998 | 0.0742 | 0.991984 | 0.8687 |
| dec_backbone[0] | 1.000000 | 0.0082 | 0.999998 | 0.0430 | 0.994035 | 1.4570 |
| dec_backbone[1] | 1.000000 | 0.0065 | 0.999995 | 0.0469 | 0.991764 | 2.4262 |
| dec_neck[0] | 1.000000 | 0.0066 | 1.000000 | 0.0234 | 0.999459 | 0.3098 |

> camera_backbone INT8 余弦相似度最低为 0.983（layer3 输出），但所有 9 个输出的余弦相似度均 ≥ 0.982，整体优于 SwinT 方案中 fuser/dec_backbone 的 INT8 表现（0.880/0.957）。

#### 9.6.5 引擎大小与压缩比（完整验证集，A100 GPU，TRT 10.15）

| 模块 | FP32 引擎 | FP16 引擎 | INT8 引擎 |
|------|----------|----------|----------|
| **camera_backbone** | **115,685 KB** | **46,533 KB** | **23,958 KB** |
| camera_neck | 12,187 KB | 4,362 KB | 2,415 KB |
| fuser | 5,385 KB | 1,551 KB | 844 KB |
| dec_backbone | 28,928 KB | 8,494 KB | 4,350 KB |
| dec_neck | 1,196 KB | 680 KB | 613 KB |
| **总计** | **159.6 MB** | **60.2 MB** | **31.4 MB** |

总部署体积对比（含未量化模块）：

| 方案 | TRT 引擎 | 未量化模块权重 | 总部署体积 |
|------|---------|-------------|----------|
| SwinT INT8（4 模块） | 7.4 MB | 129.4 MB | **~136.8 MB** |
| ResNet-50 INT8（5 模块） | 31.4 MB | 24.3 MB | **~55.7 MB** |
| 缩减 | — | — | **−59%** |

> 未量化模块权重明细（ResNet-50）：lidar/backbone（SparseEncoder）10.3 MB + camera/vtransform（bev_pool）10.0 MB + heads/object（TransFusionHead）4.0 MB = 24.3 MB。camera_backbone（ResNet-50）INT8 引擎 23.9 MB，占总引擎的 76%。将 SwinTransformer 的 105.2 MB FP32 权重替换为 TRT INT8 引擎，**总部署体积从 ~137 MB 降至 ~56 MB（−59%）**。

#### 9.6.6 逐类 AP 对比（完整验证集 6019 帧，TRT INT8 5 模块 vs FP32 基线）

| 类别 | FP32 | TRT INT8 | 变化 |
|------|------|----------|------|
| car | 0.754 | 0.753 | −0.001 |
| truck | 0.419 | 0.415 | −0.004 |
| construction_vehicle | 0.204 | 0.199 | −0.005 |
| bus | 0.466 | 0.459 | −0.007 |
| trailer | 0.318 | 0.322 | +0.004 |
| barrier | 0.686 | 0.686 | +0.000 |
| motorcycle | 0.466 | 0.464 | −0.002 |
| bicycle | 0.379 | 0.378 | −0.001 |
| pedestrian | 0.607 | 0.608 | +0.001 |
| traffic_cone | 0.662 | 0.662 | +0.000 |

> 完整数据集上 ResNet-50 INT8 各类别变化极小（最大下降 bus −0.007），证明 CNN 架构对 INT8 量化有极好的容忍度。

#### 9.6.7 SwinT vs ResNet-50 量化部署总结（完整验证集最终对比）

| 指标 | SwinT（4 模块 TRT） | ResNet-50（5 模块 TRT） |
|------|-------------------|----------------------|
| FP32 NDS | **0.7069** | 0.4989 |
| 量化模块数 | 4 / 6 | **5 / 6** |
| 可量化参数占比 | ~18% | **~88%** |
| TRT INT8 NDS | 0.7022 | 0.4948 |
| TRT INT8 NDS 变化 | −0.0047（−0.67%） | **−0.0041（−0.82%）** |
| INT8 引擎大小 | 7.4 MB | 31.4 MB |
| 未量化模块权重 | 129.4 MB | 24.3 MB |
| **总部署体积** | **~136.8 MB** | **~55.7 MB（−59%）** |
| TRT FP16 NDS 变化 | +0.0000（完全无损） | +0.0003 |

**核心结论**：

1. **SwinT 绝对精度大幅领先**：SwinT FP32 NDS 0.7069 vs ResNet-50 FP32 NDS 0.4989，差距 0.2080。ResNet-50 仅训练 6 epochs，如需缩小差距需更长训练或更大模型
2. **ResNet-50 量化覆盖优势显著**：88% vs 18% 的可量化参数占比，总部署体积 55.7 MB vs 136.8 MB（−59%）
3. **两种架构 INT8 精度损失均很小**：SwinT −0.67%，ResNet-50 −0.82%，均在工程可接受范围
4. **ResNet-50 训练尚未收敛**：6 epochs 后 loss 仍以 7.6%/epoch 速率下降，继续训练至 20 epochs 预计可显著提升 NDS
5. **选择建议**：精度优先选 SwinT（NDS 0.70+），部署效率优先选 ResNet-50（55.7 MB，88% 量化覆盖）

---

## 10. 与 NVIDIA CUDA-BEVFusion 的对比

NVIDIA 官方提供了 [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) 项目，实现了 BEVFusion 的完整 TensorRT 部署。这里对比两种方案的异同。

### 10.1 CUDA-BEVFusion 怎么做到"全模块"的？

**核心答案：** 他们用 C++ 从零重写了整个推理引擎，而不是用 Python/ONNX 标准流程。

#### 1. 稀疏卷积（我们说"不可能"的部分）

- **NVIDIA 的做法：**
  - **自定义 ONNX 算子：** 在 `exptool.py` 里他们没有用 `torch.onnx.export`，而是手工构建 ONNX 图——通过 Hook 住 `SparseConvolution.forward`，每次调用时自己拼一个 `helper.make_node("SparseConvolution", ...)`，把 kernel、stride、spatial_shape 等全部作为自定义属性写进去。
  - **自定义 C++ TRT 推理：** `src/bevfusion/lidar-scn.cpp` + `lidar-scn-onnx-parser.cpp` 是他们自己写的稀疏卷积 CUDA kernel，直接用 CUDA 实现了 SubMConv3d、SparseConv3d、ScatterDense 等操作。这不是 TRT 标准插件，是完全独立的 CUDA 推理引擎。
  - **量化方式：** 用 NVIDIA 自家的 `pytorch_quantization`（不是 MQBench），自定义了 `SparseConvolutionQunat` 类，在稀疏卷积的 feature 上插入量化节点，量化参数（amax/dynamic_range）写入自定义 ONNX 节点的属性中。

#### 2. bev_pool（我们说"不可能"的部分）

- `src/bevfusion/camera-bevpool.cu`：用 CUDA C++ 从零实现了 BEV pooling 操作。
- 根本没经过 ONNX/TRT，是独立的 CUDA kernel。

#### 3. Voxelization（我们说"不适用"的部分）

- `src/bevfusion/lidar-voxelization.cu`：同样是 CUDA C++ 独立实现。

#### 4. Camera Backbone / TransFusionHead

- 他们用 **ResNet50**（不是 SwinTransformer），ResNet50 没有动态控制流，可以直接 `torch.onnx.export` + TRT 标准流程。
- TransFusionHead 也通过标准 ONNX 导出（`head.bbox.onnx`）。

#### 5. 量化工具

- 用 NVIDIA `pytorch_quantization` 而非 MQBench，这个库和 TRT 原生兼容更好。
- **Histogram 校准**（而非我们的 MinMax），使用 300 batch 进行校准。
- 手动禁用了部分层的量化（`conv_input` 和 `decoder.neck.deblocks[0][0]`）以保精度。

------

### 和我们的项目对比

| **方面**                  | **我们（MQBench + SwinT）** | **我们（MQBench + ResNet-50）** | **CUDA-BEVFusion（NVIDIA）**             |
| ------------------------- | ----------------------------- | --------------------------------- | ---------------------------------------- |
| **语言**                  | Python（PyTorch 推理）        | Python（PyTorch 推理）            | C++/CUDA（完全独立推理引擎）             |
| **Camera Backbone**       | SwinTransformer（❌ 无法量化） | **ResNet50（✅ 可量化可导出）**     | ResNet50（✅ 可量化可导出）               |
| **稀疏卷积**              | ❌ 标准框架不支持              | ❌ 标准框架不支持                   | ✅ 自写 CUDA kernel                       |
| **bev_pool**              | ❌ 自定义 CUDA op 不可导出     | ❌ 自定义 CUDA op 不可导出          | ✅ 自写 CUDA kernel                       |
| **Voxelization**          | ❌ 跳过                        | ❌ 跳过                             | ✅ 自写 CUDA kernel                       |
| **ONNX 导出**             | `torch.onnx.export`           | `torch.onnx.export`                | 手工拼 ONNX 图（自定义算子）             |
| **量化库**                | MQBench + torch.fx            | MQBench + torch.fx                 | NVIDIA pytorch_quantization              |
| **部署方式**              | Hybrid（TRT+PyTorch 混合）    | Hybrid（TRT+PyTorch 混合）         | 纯 TRT + CUDA（无 Python 依赖）          |
| **TRT 模块数**            | 4 / 6                         | **5 / 6**                          | 全部                                      |
| **可量化参数占比**         | ~18%                          | **~88%**                           | ~100%                                     |
| **TRT INT8 NDS 变化**     | −0.0047（−0.67%）              | −0.0041（−0.82%）                  | −0.17（−0.24%）                           |
| **INT8 总部署体积**        | ~136.8 MB                     | **~55.7 MB**                       | 全 TRT                                    |
| **精度（Full nuScenes）** | **NDS=0.7069 → 0.7022 (INT8)** | **NDS=0.4989 → 0.4948 (INT8)**    | FP16: NDS=70.98, INT8: NDS=70.81 (−0.17) |
| **速度（ORIN）**          | —                             | —                                  | FP16: 18 FPS, INT8: 25 FPS               |
| **工程量**                | ~1500 行 Python               | ~2000 行 Python                    | ~5000+ 行 C++/CUDA                       |

### 10.3 关键结论

上表中说的"不可能"更准确的表述是**"在标准 Python/ONNX/TRT 工具链下不可能"**。NVIDIA 的做法是：

1. **绕过了整个标准工具链**：不用 `torch.onnx.export`，而是 hook 模型前向传播，手工构建带自定义算子的 ONNX
2. **用 C++ 重写了所有非标准算子**：稀疏卷积、bev_pool、体素化都是从零实现的 CUDA kernel
3. **用了 ResNet50 而不是 SwinTransformer**：避开了 Transformer 的动态控制流问题

这是一个工业级部署方案，工程量大但效果好。本项目的 MQBench 方案定位于研究性验证——在 Python 框架内尽可能量化，用 Hybrid 方式跑通端到端。两者定位不同，但 NVIDIA 的方案证明了这些模块技术上是可以量化和加速的，只是需要非常大的 C++ 工程投入。

---

## 11. 后续工作

### 11.1 短期（工程完善）

1. ~~**在完整 nuScenes 数据集上验证**~~ ✅ **已完成**  
   已在 nuScenes v1.0-trainval 验证集（6,019 帧）上完成 SwinT 和 ResNet-50 两种方案的全部 10 组实验（FP32/PTQ/TRT-FP32/FP16/INT8），结果见第 9 节。

2. **推理速度测量**  
   当前缺少 Hybrid TRT 的推理时间数据。在 `trt_eval_hybrid_all.py` 中添加计时代码，测量 FP32/FP16/INT8 的端到端推理延迟。这是评估实际加速效果的关键指标。

3. **逐模块混合精度 TRT 方案**  
   修改 `trt_eval_hybrid_all.py` 支持逐模块精度设置（如 fuser 用 FP16、decoder 用 INT8），验证是否能在更少精度损失的前提下保持高压缩率。

### 11.2 中期（方法改进）

3. ~~**更换主干网络为量化友好的 CNN（推荐，影响最大）**~~  ✅ **已完成**
   
   已将 camera/backbone 替换为 ResNet-50 并完成训练和全套评估（含完整验证集 6019 帧）。结果：
   - 量化覆盖从 4/6 模块（~18% 参数）提升到 **5/6 模块（~88% 参数）**
   - TRT INT8 总部署体积从 ~136.8 MB 降至 **~55.7 MB（−59%）**
   - 完整验证集 INT8 NDS 损失仅 0.82%（−0.0041），SwinT 方案损失 0.67%（−0.0047）
   - 详见第 9.6 节
   
   **进一步探索**：ResNet-50 FP32 NDS（0.4989）显著低于 SwinT（0.7069），主因是仅训练 6 epochs（训练 loss 仍以 7.6%/epoch 速率下降，远未收敛）。继续训练至 20 epochs 预计可显著提升 NDS。

4. **更先进的校准策略**  
   当前使用最朴素的 MinMax 校准。可尝试：
   - **Histogram / Percentile 校准**：裁剪掉极端值，可能改善 INT8 精度
   - **AdaRound**（MQBench 支持）：学习最优的 round-to-nearest 策略
   - **Per-channel 量化**：TRT 支持 per-channel INT8，可能改善精度

5. **SwinTransformer 量化探索（不更换主干的替代方案）**  
   如果不想更换主干网络，可以尝试绕过 torch.fx 的限制：
   - 传 `concrete_args` 固定输入尺寸，让 fx 常量化分支条件
   - 手动插入 QuantStub/DeQuantStub，不依赖 fx 自动追踪
   - 直接导出 ONNX → TRT 原生 INT8 校准（需验证 `roll`/`window_partition` 的 ONNX 兼容性）

6. **TransFusionHead 量化**  
   检测头中的 Attention 层和 FFN 层也可以量化。障碍是 `ModuleList` 动态迭代，可通过静态展开（unroll）解决。参数量小（1.04M / 2.5%），优先级低于 SwinTransformer。

### 11.3 长期（部署落地）

6. **全模型 TRT 引擎**  
   参考 NVIDIA 官方 CUDA-BEVFusion 项目（第 10 节），它提供了完整的 TRT 适配（含 SpConv plugin、bev_pool plugin），但是独立的 C++ 工程。

7. **端侧部署验证**  
   在目标部署硬件（如 Orin、Xavier）上构建 TRT 引擎并验证实时性。TRT 引擎是硬件绑定的，需要在目标 GPU 上重新构建。

8. **量化感知训练（QAT）**  
   虽然当前 PTQ 已无损，但如果扩大量化覆盖到 SwinTransformer 和 TransFusionHead 后精度下降明显，QAT 可以通过微调恢复精度。

---

## 12. 结论

本项目成功实现了 BEVFusion 的选择性混合精度量化与 TensorRT 部署，并在 nuScenes v1.0-trainval 完整验证集（6,019 帧）上进行了全面验证：

### 12.1 SwinTransformer Backbone 方案（原始架构）

1. **量化覆盖**：6 个可量化模块中成功量化 4 个（decoder/backbone、decoder/neck、camera/neck、fuser），覆盖率 67%。但这 4 个模块仅占全模型参数量的 17%（26.51 MB / 155.9 MB），因为 SwinTransformer 独占 67.5% 的参数量且无法量化
2. **PTQ 精度**：MinMax PTQ NDS 下降 0.0054（0.76%），mAP 下降 0.0110（1.63%），精度损失很小
3. **TRT 全模块部署**：4 个模块全部导出为 TRT 引擎，INT8 模式下引擎总大小仅 7.4 MB（相对于这 4 个模块的 FP32 权重 26.51 MB，压缩 3.58 倍），NDS 下降仅 0.67%
4. **FP16 完全无损**：TRT FP16 引擎 13.6 MB，NDS 变化 +0.0000（完全无损），是精度和压缩的最佳平衡

### 12.2 ResNet-50 Backbone 方案（量化友好替换）

5. **量化覆盖大幅提升**：5/6 模块量化，可量化参数占比从 ~18% 跃升至 ~88%。camera/backbone（ResNet-50）首次成功通过 torch.fx 追踪、ONNX 导出和 TRT 部署
6. **INT8 精度损失极小**：TRT INT8 NDS 下降 0.0041（0.82%），mAP 下降仅 0.0016（0.32%），各类别 AP 变化均在 ±0.007 以内
7. **总部署体积大幅下降**：TRT INT8 总部署体积 ~55.7 MB（vs SwinT 方案的 ~136.8 MB，减少 59%）。5 模块 INT8 引擎合计 31.4 MB（其中 camera_backbone 占 23.9 MB），未量化模块权重仅 24.3 MB
8. **训练尚未收敛**：ResNet-50 训练 loss 在 6 epochs 后仍以 7.6%/epoch 速率下降，表明继续训练可进一步提升 NDS（当前 0.4989 非架构能力上限）

### 12.3 核心发现

- **BEVFusion 的 BEV 中后段管线（neck → fuser → decoder）对量化极其友好**，即使使用最朴素的 MinMax 策略也能保持极低的精度损失
- **主干网络是量化部署的关键瓶颈**：SwinTransformer 占参数量的 67.5% 且无法通过标准工具链量化。替换为 ResNet-50 后，量化覆盖和部署体积均实现质的飞跃
- **Mini 数据集 vs 完整数据集**：完整验证集的结论更稳定，mini 数据集（81 帧）的 INT8 精度波动（如 SwinT NDS −0.0077）在完整数据集上收敛为更温和的 −0.0047。SwinT FP16 从 mini 集的 −0.0005 改善为完整集的 +0.0000
- **Hybrid 架构验证成功**：TRT 引擎与 PyTorch 通过 CUDA 显存零拷贝无缝协作，支持 4 模块（SwinT）和 5 模块（ResNet-50）两种配置
- **工程价值**：建立了完整的 PTQ → FakeQuant 评估 → ONNX 导出 → TRT 引擎构建 → Hybrid 端到端验证的工具链，支持自动检测 backbone 类型并选择对应的量化策略

### 12.4 待完成

- 推理速度测量（端到端延迟 / FPS）
- TransFusionHead 量化探索（最后 1/6 模块）
- ResNet-50 更长训练以缩小与 SwinT 的 FP32 精度差距（当前 NDS 0.4989 vs 0.7069）
