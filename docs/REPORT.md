# BEVFusion 全模型 INT8 量化研究报告

> **项目**：BEVFusion + MQBench 训练后量化（PTQ）
> **硬件**：NVIDIA RTX 4060 Laptop GPU（Ada Lovelace，Compute 8.9）
> **框架**：PyTorch 1.10.2 + CUDA 11.3 + MQBench 0.0.6

---

## 目录

1. [概述](#1-概述)
2. [BEVFusion 模型架构分析](#2-bevfusion-模型架构分析)
3. [量化策略与实现](#3-量化策略与实现)
4. [量化实验](#4-量化实验)
   - [4.1 瓶颈定位：6/8 消融实验](#41-瓶颈定位68-消融实验)
   - [4.2 KL Observer 解决 vtransform 瓶颈](#42-kl-observer-解决-vtransform-瓶颈)
   - [4.3 Log2 对数域量化解决 lidar 瓶颈](#43-log2-对数域量化解决-lidar-瓶颈)
   - [4.4 实验总结](#44-实验总结)
5. [最终结果汇总](#5-最终结果汇总)
6. [结论](#6-结论)

---

## 1. 概述

BEVFusion 是一种多模态 3D 目标检测模型，融合摄像头和激光雷达信息生成 BEV（鸟瞰图）表示。本项目基于 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，对 BEVFusion 实施**训练后量化（PTQ）**，目标后端为 NVIDIA TensorRT INT8。

核心挑战在于：BEVFusion 是一个**异构多模态模型**，包含稀疏卷积、自定义 CUDA 算子、Transformer 等多种子模块，无法对全模型统一量化。我们设计了**三条选择性量化路径**，最终实现了 **8/8 全模块 INT8 量化**，精度损失仅 **−2.7%**。

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

| 子模块              | 类                | 算子类型                               | 量化路径       |
| ------------------- | ----------------- | -------------------------------------- | -------------- |
| `camera/backbone`   | SwinTransformer   | Attention + MLP + Window Partition     | 路径二（手动） |
| `camera/neck`       | GeneralizedLSSFPN | Conv2d + BN + ReLU + Bilinear Upsample | 路径一（fx）   |
| `camera/vtransform` | LSSTransform      | 自定义 CUDA 算子 (QuickCumsumCuda)     | 路径二（手动） |
| `lidar/voxelize`    | Voxelization      | 动态分散 (scatter)，非神经网络层       | 跳过（非 NN）  |
| `lidar/backbone`    | SparseEncoder     | 稀疏卷积 (spconv)                      | 路径三（稀疏） |
| `fuser`             | ConvFuser         | Conv2d(336,256,3) + BN + ReLU          | 路径一（fx）   |
| `decoder/backbone`  | SECOND            | 多层 Conv2d + BN + ReLU                | 路径一（fx）   |
| `decoder/neck`      | SECONDFPN         | ConvTranspose2d + BN + ReLU            | 路径一（fx）   |
| `heads/object`      | TransFusionHead   | Attention + TopK + 动态 shape          | 路径二（手动） |

### 各子模块的参数分布

| 子模块                              | 参数量     | FP32 权重大小 | 占比     |
| ----------------------------------- | ---------- | ------------- | -------- |
| `camera/backbone` (SwinTransformer) | 27.55M     | 105.20 MB     | 67.5%    |
| `lidar/backbone` (SparseEncoder)    | 2.70M      | 10.29 MB      | 6.6%     |
| `camera/vtransform` (LSSTransform)  | 2.61M      | 9.95 MB       | 6.4%     |
| `decoder/backbone` (SECOND)         | 4.29M      | 16.35 MB      | 10.5%    |
| `camera/neck` (GeneralizedLSSFPN)   | 1.59M      | 6.08 MB       | 3.9%     |
| `heads/object` (TransFusionHead)    | 1.04M      | 3.95 MB       | 2.5%     |
| `fuser` (ConvFuser)                 | 0.78M      | 2.95 MB       | 1.9%     |
| `decoder/neck` (SECONDFPN)          | 0.30M      | 1.13 MB       | 0.7%     |
| **总计**                            | **40.84M** | **155.91 MB** | **100%** |

---

## 3. 量化策略与实现

### 3.1 为什么需要三条量化路径

传统 PTQ 工具假设模型可被完整导出为 ONNX，再统一做 INT8 校准。BEVFusion 存在以下障碍：

1. **稀疏卷积（spconv）**：LiDAR 分支的 SparseEncoder 使用稀疏张量表示，标准 ONNX 和 TensorRT 均不支持。
2. **自定义 CUDA 算子**：Camera 分支的 `bev_pool`（QuickCumsumCuda）是用 CUDA C++ 实现的 autograd Function，没有对应的 ONNX 算子。
3. **动态控制流**：SwinTransformer 内部有 `if x.shape[0] > window_size:` 等基于张量值的分支判断，`torch.fx` 符号追踪时会失败。TransFusionHead 中的动态迭代同样不兼容。
4. **体素化预处理**：Voxelization 是离散化预处理步骤，不包含可微分的权重，不需要量化。

### 3.2 三条量化路径

| 路径                              | 适用模块                                           | 方法                          |
| --------------------------------- | -------------------------------------------------- | ----------------------------- |
| **路径一**（torch.fx 自动插桩）   | camera/neck, fuser, decoder/backbone, decoder/neck | `MQBench.prepare_by_platform` |
| **路径二**（手动 FakeQuant 包装） | camera/backbone, camera/vtransform, heads          | 手动替换 Conv2d/Linear        |
| **路径三**（稀疏卷积专用）        | lidar/backbone                                     | 手动替换 SparseConvolution    |

**设计跳过**：`lidar/voxelize`（体素化预处理，非神经网络层）

### 3.3 核心工具脚本

| 脚本                        | 功能                                         |
| --------------------------- | -------------------------------------------- |
| `tools/quant_ptq_minmax.py` | **核心 PTQ 工具**（MinMax/KL Observer/Log2） |
| `tools/test.py`             | FP32 基线评估                                |
| `tools/quant_benchmark.py`  | 性能基准测试                                 |

### 3.4 mmdet3d 代码修改

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
# 修改前（fx 追踪失败：len(inputs) 在 Proxy 上不可用）
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
# 修改前（fx 追踪失败：torch.cat(Proxy, dim=1) 中 Proxy 代表整个 list）
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

## 4. 量化实验

### 4.1 瓶颈定位：6/8 消融实验

在完整验证集（6019 帧）上运行逐模块消融实验，定位量化瓶颈：

| 实验配置                            | 量化模块                               | NDS        | ΔNDS       |
| ----------------------------------- | -------------------------------------- | ---------- | ---------- |
| FP32 基线                           | —                                      | 0.7069     | —          |
| **PTQ 6/8** (skip vtransform+lidar) | SwinT+neck+fuser+dec_bb+dec_neck+heads | **0.7010** | **−0.83%** |
| PTQ 7/8 +vtransform                 | 6/8 + vtransform                       | 0.6179     | −12.6%     |
| PTQ 7/8 +lidar                      | 6/8 + lidar/backbone                   | 0.5751     | −18.5%     |
| PTQ 8/8 全模型 (MinMax)             | 全部 8 模块                            | 0.4562     | −35.5%     |

**关键发现**：

1. **PTQ 6/8 是良好基线**：NDS 仅下降 0.83%，说明 SwinT、neck、fuser、decoder、heads 的量化是可控的。

2. **vtransform 量化损失巨大**（−12.6%）：其 `bev_pool` 操作产生的激活值呈长尾分布，EMAMinMax 导致 98.3% range waste。

3. **lidar/backbone 量化损失更大**（−18.5%）：SparseEncoder 的稀疏激活呈二模态分布（大多数为 0，少数有大值），per-tensor INT8 量化粒度不足。

4. **8/8 全量化不可用**（−35.5%）：vtransform + lidar 累积误差导致模型性能崩溃。

**结论**：vtransform 和 lidar/backbone 是量化瓶颈，需要专门设计的量化策略。

---

### 4.2 KL Observer 解决 vtransform 瓶颈

#### 4.2.1 方法：KLDivergenceObserver

**动机**：EMAMinMaxObserver 使用 [min, max] 全范围映射到 INT8，当激活分布呈长尾时，大量 INT8 量化级别被浪费在几乎没有激活值的区间。

**算法**：受 TensorRT 的 IInt8EntropyCalibrator2 启发：

1. **直方图收集**：校准阶段累积 2048-bin 直方图
2. **最优截断搜索**：遍历截断阈值 T，计算截断分布 P 与量化分布 Q 的 KL 散度
3. **对称范围设置**：选择 KL 最小的 T 作为量化范围 [-T, T]

#### 4.2.2 实验结果

| 配置            | Observer  | NDS        | ΔNDS      | 改善          |
| --------------- | --------- | ---------- | --------- | ------------- |
| 7/8 +vtransform | EMAMinMax | 0.6179     | −12.6%    | 基线          |
| 7/8 +vtransform | **KL**    | **0.7033** | **−0.5%** | **+12.1 pts** |
| 8/8 全量化      | EMAMinMax | 0.4562     | −35.5%    | 基线          |
| 8/8 全量化      | KL(both)  | 0.5750     | −18.7%    | +16.8 pts     |

**关键发现**：

- **KL Observer 完全解决 vtransform 量化瓶颈**：从 −12.6% 改善至 −0.5%（+12.1 pts）
- **KL Observer 对 lidar 几乎无效**：仅改善 +0.07 pts（0.5751 → 0.5758）

**原因分析**：

- vtransform 的问题是"长尾 range waste"，KL 截断策略直接解决
- lidar 的问题是"空间稀疏 + 值域二模态"，per-tensor 量化粒度不足，非 Observer 策略所能解决

---

### 4.3 Log2 对数域量化解决 lidar 瓶颈

#### 4.3.1 W8A16 控制实验：确认激活是瓶颈

在尝试解决 lidar 瓶颈前，先通过 W8A16 控制实验定量区分权重和激活量化的各自贡献：

| 实验                 | 配置                            | NDS        | ΔNDS       |
| -------------------- | ------------------------------- | ---------- | ---------- |
| FP32                 | 基线                            | 0.7069     | 0%         |
| 7/8 +lidar EMA       | W8A8                            | 0.5751     | −18.5%     |
| **7/8 +lidar W8A16** | **W8A16（权重 int8，激活 FP）** | **0.7009** | **−0.85%** |

**结论**：

- **激活量化是 lidar 损失的根本来源**（−18.5% vs −0.85%）
- 权重 int8 几乎无损（仅 0.02%）
- 解决方案应聚焦于**激活量化方法**

#### 4.3.2 方法：SparseLog2FakeQuantize（对数域量化）

**动机**：lidar 稀疏激活呈二模态分布——大多数 voxel 值为 0，少数有较大值。传统线性量化在零点附近浪费 90%+ 的 INT8 级别，而在有值区域精度不足。

**算法**：对数域量化

$$y = \text{sign}(x) \cdot 2^{\text{round}(\log_2(|x| + 1) / \text{scale}) \cdot \text{scale}}$$

**与 INT8 均匀量化的关键区别**：

- 均匀量化：绝对误差恒定，小幅值信号相对误差 >> 100%
- Log2 量化：相邻格点比例恒为 2，相对误差恒定 ≈ 41%

**实现**：在 `tools/quant_ptq_minmax.py` 中新增 `SparseLog2FakeQuantize` 类。

#### 4.3.3 实验结果（训练集校准，128 batch）

| 配置                          | NDS        | ΔNDS      | 对比 EMA 基线   |
| ----------------------------- | ---------- | --------- | --------------- |
| FP32 基线                     | 0.7069     | 0%        | —               |
| 7/8 +lidar EMA                | 0.5751     | −18.5%    | 基线            |
| 7/8 +lidar KL                 | 0.5758     | −18.5%    | +0.07 pts       |
| **7/8 +lidar Log2 PT**        | **0.6849** | **−3.1%** | **+15.4 pts** ✨ |
| 7/8 +lidar Log2 PC + LWC      | 0.6878     | −2.7%     | +16.3 pts       |
| **8/8 +vt KL +lidar Log2 PT** | **0.6875** | **−2.7%** | —               |

**关键发现**：

1. **Log2 对数域量化对 lidar 极其有效**：从 −18.5% 改善至 −3.1%（+15.4 pts）
2. **Per-Tensor 优于 Per-Channel**：PT −3.1% vs PC −4.9%（per-channel 在小数据集上容易过拟合）
3. **全量化（8/8）突破**：KL + Log2 组合使 8/8 精度从 −35.5% 提升到 −2.7%

---

### 4.4 实验总结

| 瓶颈模块       | 问题                                 | 解决方案             | 改善                       |
| -------------- | ------------------------------------ | -------------------- | -------------------------- |
| vtransform     | bev_pool 长尾分布，98.3% range waste | KL Observer 最优截断 | −12.6% → −0.5% (+12.1 pts) |
| lidar/backbone | 稀疏二模态分布，线性量化零点附近浪费 | Log2 对数域量化      | −18.5% → −3.1% (+15.4 pts) |

**最终突破**：KL Observer + Log2 量化组合，使 8/8 全模块量化精度损失从 −35.5% 降至 −2.7%。

---

## 5. 最终结果汇总

### 5.1 量化精度（完整验证集 6019 帧）

| 配置                       | NDS        | mAP        | ΔNDS      | 量化模块 | 说明                       |
| -------------------------- | ---------- | ---------- | --------- | -------- | -------------------------- |
| **FP32 基线**              | **0.7069** | **0.6728** | **0%**    | 0/8      | 原始模型                   |
| **8/8 全量化（最终最优）** | **0.6875** | **0.6429** | **−2.7%** | **8/8**  | vtransform KL + lidar Log2 |
| 7/8 +vt KL                 | 0.7033     | 0.6657     | −0.5%     | 7/8      | skip lidar                 |
| PTQ 6/8                    | 0.7010     | 0.6614     | −0.83%    | 6/8      | skip vt+lidar              |
| PTQ 8/8 MinMax 基线        | 0.4562     | 0.3536     | −35.5%    | 8/8      | 传统 MinMax                |

### 5.2 部署验证结果（spconv23_deploy 环境，2026-04-05）

在量化算法突破后，进一步完成了**独立部署环境**（Python 3.9 + PyTorch 2.0 + spconv 2.3）的精度验证，以及**去 PyTorch LiDAR backbone** 的落地。

| 路径 | LiDAR 实现 | 量化 | NDS | mAP | 备注 |
|------|-----------|------|-----|-----|------|
| PyTorch FP16 | `SparseEncoder23` (spconv 2.3) | 无 | 0.7040 | 0.6654 | Phase 7 基准 |
| **TV FP16** | **`TVSparseEncoder` (去 PyTorch)** | 无 | **0.7039** | — | Phase 8，与 PyTorch 一致 |
| PyTorch INT8 | `SparseEncoder23` (spconv 2.3) | Log2 | **0.6893** | **0.6478** | Phase 7 控制组 |
| **TV INT8** | **`TVSparseEncoder` (去 PyTorch)** | Log2 | **0.6893** | **0.6474** | Phase 9 Part A |

**关键结论**：
- TV FP16 与 PyTorch FP16 NDS 几乎一致（0.7039 vs 0.7040），证明 `tv.Tensor` + `core_cc` 稀疏卷积实现数值正确。
- PyTorch INT8 控制组 NDS 为 **0.6893**，验证 PTQ checkpoint 在 spconv 2.3 环境下无精度 regression。
- TV INT8 与 PyTorch INT8 NDS **完全一致**（0.6893），mAP 差异仅 0.0004，属于 INT8 不同 `implicit_gemm` kernel 实现的正常 rounding 波动。
- 这标志着 **Log2 对数量化不仅在 PyTorch 仿真中有效，也在去 PyTorch 的 TV backbone 部署中完全保持精度**。

### 关键技术贡献

| 技术                | 解决的问题                   | 效果           |
| ------------------- | ---------------------------- | -------------- |
| **KL Observer**     | vtransform bev_pool 长尾分布 | −12.6% → −0.5% |
| **Log2 对数域量化** | lidar 稀疏二模态分布         | −18.5% → −3.1% |

---

## 6. 结论

本项目实现了 BEVFusion **8/8 全模块 INT8 量化**，精度损失仅 **−2.7%**（NDS 0.6875 vs FP32 0.7069）。

**核心贡献**：

1. **三条量化路径**：针对不同模块特性（密集卷积、稀疏卷积、fx 追踪失败模块），设计了 torch.fx 自动插桩、手动 FakeQuant 包装、稀疏卷积专用三条量化路径，实现了全模型覆盖。

2. **KL Observer**：基于 KL 散度的最优截断校准器，解决了 vtransform bev_pool 长尾分布导致的 range waste 问题，将 vtransform 量化损失从 −12.6% 降至 −0.5%。

3. **Log2 对数域量化**：针对 lidar 稀疏激活的二模态分布特性，设计了 Log2 对数域量化，为小值区域提供更多量化级别，将 lidar 量化损失从 −18.5% 降至 −3.1%。

4. **W8A16 控制实验**：确认了 lidar 量化损失的主要来源是激活量化而非权重量化，为 Log2 量化方案提供了理论依据。

**项目状态**：✅ 核心研究完成，8/8 全量化精度损失仅 −2.7%

**部署状态**：✅ 已完成 `spconv23_deploy` 独立环境迁移及去 PyTorch LiDAR backbone 部署验证
- TV FP16 NDS = 0.7039（与 PyTorch 路径一致）
- TV INT8 Log2 NDS = 0.6893（与 PyTorch INT8 路径一致）

**下一步方向**：Jetson Orin 完全零 PyTorch 部署（Phase 9 Part B），待设备到位后推进。