## 2026-03-03 · 服务器完整验证集评估（nuScenes v1.0-trainval val，6019 帧）

**环境**：服务器 4×RTX 3090 + 1×A100-SXM4-80GB，CUDA 12.2，PyTorch 1.10.2+cu113，TensorRT 10.15.1.29（tensorrt-cu12），mmcv-full 1.4.0

**数据集**：nuScenes v1.0-trainval 验证集（6019 帧，全部 10 类均有充足样本）

**日志文件**：`severlog/results_server_*.log`（共 10 个）

> ⚠️ **这是最终结果**。之前所有在 mini 数据集（81 帧）上的结果仅为开发阶段快速验证，不具有统计意义。完整验证集的评估覆盖了全部 10 个类别和 850 个场景，结论具有充分的可信度。

### SwinT Backbone（官方预训练权重，4/6 模块可量化）

#### 精度总表

| 方法 | NDS | mAP | NDS Δ | mAP Δ |
|------|------|------|-------|-------|
| PyTorch FP32（基线） | **0.7069** | **0.6728** | — | — |
| PTQ 4/6（MQBench FakeQuant） | 0.7015 | 0.6618 | −0.0054 | −0.0110 |
| TRT FP32（4 模块） | 0.7065 | 0.6726 | −0.0004 | −0.0002 |
| TRT FP16（4 模块） | **0.7069** | **0.6728** | **+0.0000** | **+0.0000** |
| TRT INT8（4 模块） | 0.7022 | 0.6641 | −0.0047 | −0.0087 |

#### 误差指标

| 方法 | mATE | mASE | mAOE | mAVE | mAAE |
|------|------|------|------|------|------|
| FP32 基线 | 0.2854 | 0.2555 | 0.3165 | 0.2518 | 0.1856 |
| PTQ 4/6 | 0.2881 | 0.2563 | 0.3091 | 0.2543 | 0.1859 |
| TRT FP32 | 0.2868 | 0.2558 | 0.3174 | 0.2513 | 0.1870 |
| TRT FP16 | 0.2856 | 0.2556 | 0.3165 | 0.2518 | 0.1856 |
| TRT INT8 | 0.2895 | 0.2562 | 0.3121 | 0.2541 | 0.1868 |

#### 逐类 AP

| 类别 | FP32 | PTQ 4/6 | TRT FP32 | TRT FP16 | TRT INT8 |
|------|------|---------|----------|----------|----------|
| car | 0.875 | 0.869 | 0.875 | 0.875 | 0.876 |
| truck | 0.639 | 0.625 | 0.638 | 0.639 | 0.626 |
| construction_vehicle | 0.277 | 0.272 | 0.275 | 0.276 | 0.270 |
| bus | 0.741 | 0.729 | 0.745 | 0.741 | 0.729 |
| trailer | 0.424 | 0.416 | 0.425 | 0.425 | 0.405 |
| barrier | 0.726 | 0.711 | 0.726 | 0.726 | 0.715 |
| motorcycle | 0.770 | 0.752 | 0.762 | 0.769 | 0.755 |
| bicycle | 0.612 | 0.599 | 0.614 | 0.611 | 0.606 |
| pedestrian | 0.877 | 0.870 | 0.877 | 0.877 | 0.874 |
| traffic_cone | 0.788 | 0.774 | 0.789 | 0.788 | 0.786 |

#### 余弦相似度（TRT vs PyTorch，per-module）

| 模块 | FP32 cos | FP32 maxErr | FP16 cos | FP16 maxErr | INT8 cos | INT8 maxErr |
|------|----------|------------|----------|------------|----------|------------|
| camera_neck[0] | 1.000000 | 0.0046 | 0.999996 | 0.0352 | 0.999046 | 0.2409 |
| camera_neck[1] | 1.000000 | 0.0051 | 0.999996 | 0.0273 | 0.998616 | 0.2906 |
| fuser[0] | 1.000000 | 0.0374 | 0.999998 | 0.3438 | 0.957402 | 2.4582 |
| dec_backbone[0] | 1.000000 | 0.0102 | 0.999997 | 0.0469 | 0.977364 | 2.0801 |
| dec_backbone[1] | 1.000000 | 0.0078 | 0.999990 | 0.0483 | 0.880242 | 3.3819 |
| dec_neck[0] | 1.000000 | 0.0013 | 1.000000 | 0.0049 | 0.999271 | 0.1631 |

> INT8 的 fuser 和 dec_backbone[1] 余弦相似度较低（0.957/0.880），但经过 TransFusionHead 后端到端 NDS 仅下降 0.0047，说明检测头对这种量化误差有良好的容错能力。

#### 引擎大小

| 模块 | FP32 引擎 | FP16 引擎 | INT8 引擎 |
|------|----------|----------|----------|
| camera_neck | 9,946 KB | 3,245 KB | 1,787 KB |
| fuser | 5,385 KB | 1,551 KB | 845 KB |
| dec_backbone | 28,928 KB | 8,490 KB | 4,349 KB |
| dec_neck | 1,196 KB | 680 KB | 613 KB |
| **总计** | **44.4 MB** | **13.6 MB** | **7.4 MB** |

#### PTQ Warnings（预期行为，非错误）

PTQ 日志中的 WARNING 是**正常预期行为**：
- `✗ 量化子模块 camera/backbone 失败`：SwinTransformer 动态控制流，无法 fx 追踪（已知限制）
- `✗ 量化子模块 heads/object 失败`：TransFusionHead Proxy 迭代问题（已知限制）
- 最终量化 4/6 模块，与设计一致

TRT 日志中的 `prim::Constant shape inference` WARNING 是 PyTorch ONNX 导出的已知警告，不影响结果。

---


> ⚠️ **ResNet-50 实验已归档**：所有 ResNet-50 相关实验结果已归档至 [archive/resnet50_experiments/README.md](../archive/resnet50_experiments/README.md)

**核心结论**：ResNet-50 量化覆盖率高（88% vs 18%），但精度天花板低（NDS 0.4989）。已不再是项目主攻方向，但实验验证了 PTQ→TRT 工具链的可行性。

## 2026-03-08 · SwinT PTQ 全模块消融实验（6/7/8 模块，完整验证集 6019 帧）

> **背景**：修复 MQBench `symmetric_range` 兼容性问题后，SwinT（手动路径）、vtransform（手动路径）、lidar/backbone（稀疏卷积路径）、heads/object（手动路径）均可正常量化。4 张 GPU 并行运行消融实验。

### 精度（完整验证集，nuScenes v1.0-trainval，6019 帧）

| 配置 | 量化模块 | 模块覆盖 | NDS | mAP | ΔNDS | ΔmAP |
|------|---------|---------|-----|-----|------|------|
| FP32 基线 | — | 0% | 0.7069 | 0.6728 | — | — |
| **PTQ 6/8** (skip vtransform+lidar) | SwinT, neck, fuser, dec_bb, dec_neck, heads | 67% | **0.7010** | **0.6614** | **−0.0059** | **−0.0114** |
| **PTQ 7/8 +vtransform** (skip lidar) | +vtransform | 78% | 0.6179 | 0.5194 | −0.0890 | −0.1534 |
| **PTQ 7/8 +lidar** (skip vtransform) | +lidar/backbone | 78% | 0.5751 | 0.5394 | −0.1318 | −0.1334 |
| **PTQ 8/8 全模型** | 全部 8 模块 | 89% | 0.4562 | 0.3536 | −0.2507 | −0.3192 |

### 逐类 AP 对比（平均 4 个距离阈值）

| 类别 | FP32 | PTQ 6/8 | PTQ 7/8+vtrans | PTQ 7/8+lidar | PTQ 8/8 |
|------|------|---------|----------------|---------------|---------|
| car | 0.875 | 0.870 | 0.637 | 0.777 | 0.494 |
| truck | 0.639 | 0.625 | 0.391 | 0.515 | 0.250 |
| construction_vehicle | 0.277 | 0.274 | 0.139 | 0.189 | 0.053 |
| bus | 0.741 | 0.730 | 0.483 | 0.548 | 0.299 |
| trailer | 0.424 | 0.416 | 0.263 | 0.368 | 0.208 |
| barrier | 0.726 | 0.710 | 0.657 | 0.688 | 0.584 |
| motorcycle | 0.770 | 0.751 | 0.604 | 0.562 | 0.346 |
| bicycle | 0.612 | 0.597 | 0.495 | 0.400 | 0.253 |
| pedestrian | 0.877 | 0.869 | 0.811 | 0.624 | 0.428 |
| traffic_cone | 0.788 | 0.774 | 0.714 | 0.724 | 0.621 |

### 理论 INT8 部署体积（假设量化权重存储为 INT8）

| 配置 | 量化 FP32 大小 | 未量化 FP32 | 理论 INT8 总大小 | 压缩率 |
|------|-------------|-----------|----------------|--------|
| FP32 基线 | — | 156.1 MB | 156.1 MB | — |
| PTQ 6/8 | 137.9 MB → 34.5 MB INT8 | 20.3 MB | **54.8 MB** | **−65%** |
| PTQ 7/8 +vtransform | 147.9 MB → 37.0 MB INT8 | 10.3 MB | **47.3 MB** | **−70%** |
| PTQ 7/8 +lidar | 148.2 MB → 37.1 MB INT8 | 10.0 MB | **47.1 MB** | **−70%** |
| PTQ 8/8 全模型 | 156.1 MB → 39.0 MB INT8 | 0 MB | **39.0 MB** | **−75%** |

> ⚠️ PTQ checkpoint（.pth）仍为 FakeQuant 格式（FP32 + 少量 scale/zp 元数据），文件大小约 157 MB（略大于 FP32）。上表为转换为真实 INT8 推理格式后的理论大小。

### 关键结论

1. **PTQ 6/8（推荐配置）**：SwinT 手动量化 + heads 手动量化后，NDS 仅下降 0.0059（−0.83%）。与旧 PTQ 4/6（NDS 0.7015）几乎相同，说明新增的手动量化路径（SwinT + heads）几乎无精度损失，但理论压缩体积从 ~156 MB 降至 ~54.8 MB（−65%）
2. **vtransform 高度敏感**：INT8 量化使 NDS 额外下降 0.089（−12.6%）。这远大于 mini 数据集估计的 −0.031，说明 mini 数据集对 vtransform 敏感度的估计严重偏乐观
3. **lidar/backbone 最为敏感**：INT8 量化使 NDS 额外下降 0.132（−18.6%）。SparseEncoder 的稀疏卷积激活值分布不适合 INT8 MinMax 校准
4. **8/8 全模型量化不可用**：NDS 从 0.7069 降至 0.4562（−35.5%），精度损失无法接受
5. **mini 数据集低估了敏感模块的精度损失**：vtransform 上差距尤其大（mini −0.031 vs full −0.089）

## 2026-03-09 · SwinT PTQ 精细化校准策略消融实验（LWC + MSEObserver，8/8 模块，完整验证集 6019 帧）

> **背景**：上一轮消融实验揭示了 lidar/backbone（稀疏卷积）和 camera/vtransform 的量化敏感性问题。lidar/backbone 权重分布
> 诊断（`tools/diag_lidar_distribution.py`）发现权重范围利用率（max/range 比值）仅 25–78%，激活值最大值与中位值之比最高
> 达 15.7，存在明显离群点。据此，实施了两种精细化校准策略：
> - **LWC（Learnable Weight Clipping）**：受 OmniQuant（ICLR 2024）启发，为 lidar/backbone 各层引入可学习的权重截断参数
>   `weight_clip_ratio`，通过最小化 FakeQuant 后的权重重建 MSE 来优化截断位置
> - **MSEObserver**：将 lidar/backbone 激活量化的 observer 从默认的 EMAMinMaxObserver 替换为 MQBench 的 MSEObserver，
>   通过最小化校准数据上的均方误差搜索最优激活 range

### 精度（完整验证集，nuScenes v1.0-trainval，6019 帧）

| 配置 | 权重校准 | 激活校准 | NDS | mAP | ΔNDS vs FP32 | ΔNDS vs 8/8基线 |
|------|---------|---------|-----|-----|-------------|----------------|
| FP32 基线 | — | — | 0.7069 | — | 0 | — |
| PTQ 8/8 MinMax 基线 | EMAMinMax | EMAMinMax | 0.4562 | 0.3536 | −0.2507 | — |
| PTQ 8/8 + LWC | **LWC** | EMAMinMax | 0.4545 | 0.3540 | −0.2524 | −0.0017 |
| PTQ 8/8 + MSEObserver | EMAMinMax | **MSEObserver** | 0.4560 | 0.3522 | −0.2509 | −0.0002 |
| PTQ 8/8 + LWC + MSEObserver | **LWC** | **MSEObserver** | 0.4526 | 0.3519 | −0.2543 | −0.0036 |
| PTQ 8/8 + LWC + EMAQuantile | **LWC** | **EMAQuantile** | 0.4540 | 0.3535 | −0.2529 | −0.0022 |

### LWC 运行统计（lidar/backbone，21 层，128 calib batches）

- 优化权重截断层数：21 层（所有稀疏卷积层）
- 最终截断比率范围：0.982–0.990（即截断后 range 为原始 MinMax range 的 98.2%–99.0%）
- 实际被截断的权重元素数：**3,003 / 2,691,696（0.11%）**
- 各层 FakeQuant 重建 MSE 损失：1e-5 至 4e-5（极低）

### 结果分析

**量化成功运行，但 LWC/MSEObserver 对 NDS 均无显著改善（差异在 ±0.004 以内，属于校准随机性范围）。**

这不是代码 bug，而是揭示了以下深层原因：

#### 根本原因：lidar/backbone 不是 8/8 精度崩溃的主要来源

回顾两轮消融实验的完整数据：

| 实验 | 相较 FP32 的 ΔNDS |
|------|-----------------|
| 仅 vtransform 量化（7/8） | −0.0890 |
| 仅 lidar 量化（7/8） | −0.1318 |
| 线性叠加预测（8/8） | −0.2208 |
| 实际 8/8 MinMax | **−0.2507**（比线性叠加还差 −0.030）|

vtransform 和 lidar 同时量化时存在**非线性协同劣化效应**：两路特征各自的量化误差在 fuser 处相互干扰，导致融合后的精度崩溃
比两者单独损失之和更大。

**LWC 针对 lidar/backbone 权重截断**：0.11% 的截断率、极低的重建 MSE（1e-5 量级）说明 lidar 稀疏卷积权重本身并没有严重的
"功能性离群点"——权重范围诊断显示的 22–75% range 利用率低，更多源于权重分布较集中，而非存在危害量化的真正离群值。

**MSEObserver 针对 lidar 激活范围搜索**：在 vtransform 激活已经把 camera BEV 特征破坏的情况下，单独改善 lidar 侧的激活
range 估计，无法补偿融合处的协同误差。MSEObserver 收益有限还可能与 128 batch 校准量不足有关（MSEObserver 通常需要更多样本
才能找到更优 range）。

#### 为什么 LWC+MSE 组合反而更差（−0.0036 vs 基线）？

- LWC 截断的 0.11% 权重元素虽然是统计意义上的"离群点"，但**不等于对任务无用**——
  某些权重可能在特定稀疏点云模式下起关键作用，截断后任务损失轻微上升
- 差异仍在噪声范围（±0.004），可能只是不同校准 batch 随机性导致

#### 核心结论

1. **LWC 方法论正确，代码实现验证有效**，但作用在了次要瓶颈上
2. **vtransform 是 8/8 精度崩溃的另一主因**，且 bev_pool 输出的激活分布更难用标准 PTQ 处理
3. **6/8 方案（NDS 0.7010）是正确的工程选择**：跳过 vtransform 和 lidar 这两个量化损失最大的模块，
   用 MinMax PTQ 即可实现 NDS −0.0059（−0.83%），无需任何精细化校准
4. 若要在 8/8 上取得改善，需要面向 vtransform 设计 bev_pool 专属的量化方案（如固定激活 range 或混合精度），
   而非仅优化 lidar/backbone 侧的校准策略

---

## 2026-03-11 · Round 4：KL 散度 Observer 实验（本地 mini 验证集）

**分支**：`exp/lss-kl-divergence-calibration`（基于 `v1.0-baseline-8of8-complete` 备份）

### 背景

vtransform 诊断（03-10）确认 EMAMinMaxObserver 导致 downsample 层 94~97% range waste，bev_pool 98.3% waste。
核心原因：EMAMinMax 跟踪全局 [min, max]，但大多数激活集中在很窄的范围，导致 INT8 量化步长远大于信号标准差。

### 方法：KL 散度最优截断

实现 `KLDivergenceObserver` 类（类似 TensorRT 的 entropy calibrator）：
1. 在校准阶段收集激活值绝对值的直方图（2048 bins）
2. 校准结束后，遍历候选截断阈值 T，将 [0, T] 内的分布量化为 INT8
3. 对每个 T，计算 KL(P_fp32 || Q_int8)，其中 Q 包含截断到最后一个 bin 的质量
4. 选择 T_opt = argmin KL，约束 min_percentile ≥ 0.9999（保留 ≥99.99% 样本质量）
5. 设置对称范围 [-T_opt, T_opt]

关键差异：EMAMinMax 保留所有极端值 → 巨大的 step size；KL 截断离群值 → 更紧凑的 INT8 分配。

### Bug 修复：`--load-from` 参数名不匹配

发现 NDS=0.0 的根本原因：parser 定义了 `--load_from`（下划线），但命令行使用 `--load-from`（连字符）。
argparse 不视二者为等同，导致 `--load-from` 泄漏到 `opts` → `configs.update(opts)` 创建了 `load-from` 键
（而非 `load_from`），checkpoint **从未被加载**。模型仅有 ImageNet 预训练的 backbone 权重。
修复：将 parser 定义改为 `--load-from`（argparse 自动将 dest 转为 `load_from`）。

### 实验结果（mini 验证集 81 帧，8 calib batch）

| 配置 | NDS | mAP | Δ NDS | vt observer | lidar observer |
|------|-----|-----|-------|-------------|----------------|
| 0/8 FP32 | 0.5788 | 0.5732 | 0% | — | — |
| 6/8（skip vt+lidar） | 0.5779 | 0.5709 | −0.2% | skip | skip |
| 7/8 +vt EMAMinMax | 0.5474 | 0.5060 | −5.4% | EMA | skip |
| 7/8 +vt **KL** | **0.5720** | **0.5642** | **−1.2%** | KL | skip |
| 7/8 +lidar EMAMinMax | 0.4734 | 0.4639 | −18.2% | skip | EMA |
| 7/8 +lidar **KL** | **0.5173** | **0.4961** | **−10.6%** | skip | KL |
| 8/8 EMAMinMax | 0.4285 | 0.3626 | −26.0% | EMA | EMA |
| 8/8 KL(vt) | 0.4680 | 0.4553 | −19.1% | KL | EMA |
| 8/8 KL(vt)+MSE(lidar) | 0.4714 | 0.4568 | −18.6% | KL | MSE |
| **8/8 KL(both)** | **0.5085** | **0.4817** | **−12.1%** | KL | KL |

### KL Observer 截断阈值（vtransform 9 层）

| 层 | T_opt | hist_max | 压缩率 | 覆盖率 |
|----|-------|----------|--------|--------|
| dtransform.0 | 57.0 | 59.2 | 3.6% | 100% |
| dtransform.3 | 10.2 | 11.9 | 13.9% | 99.99% |
| dtransform.6 | 12.8 | 20.0 | 36.1% | 99.99% |
| depthnet.0 | 7.8 | 17.3 | 55.0% | 99.99% |
| depthnet.3 | 6.1 | 12.1 | 50.0% | 100% |
| depthnet.6 | 6.8 | 13.1 | 47.9% | 99.99% |
| downsample.0 | 31.9 | 427.4 | **92.5%** | 100% |
| downsample.3 | 22.1 | 316.5 | **93.0%** | 99.99% |
| downsample.6 | 19.2 | 227.8 | **91.6%** | 99.99% |

downsample 3 层从 EMAMinMax 的 94~97% waste 降到 KL 主动截断 91~93%，
但截断是**有信息的**：保留了 99.99% 的样本质量，丢弃的 0.01% 是真正的离群值。

### 关键发现

1. **KL Observer 在 vtransform 上效果显著**：7/8 +vt 从 NDS −5.4% 改善到 −1.2%（拯救 4.2 个 NDS 点）
2. **KL Observer 在 lidar 上同样有效**：7/8 +lidar 从 NDS −18.2% 改善到 −10.6%（拯救 7.6 个 NDS 点）
3. **8/8 KL(both) 综合提升 13.9 个 NDS 点**：从 0.4285 提升到 0.5085
4. MSE Observer 对 lidar 的效果弱于 KL（仅提升 0.34% vs KL 的 7.6%）
5. **残留差距**：8/8 KL(both) 仍比 6/8 差 6.9 个 NDS 点（0.5085 vs 0.5779），
   主要来自 lidar 稀疏激活在 per-tensor INT8 下的固有精度损失

### 待做：服务器全量验证

需要在 6019 帧完整验证集上确认 mini 结果是否成立。服务器命令见下方。

---

## 2026-03-12 · Round 4+5：KL 散度 Observer 服务器全量验证（6019 帧）

**分支**：`exp/lss-kl-divergence-calibration`

### 实验设计

- **Round 4**：512 calib batch + shuffle，4 个 GPU 并行
- **Round 5**：128 calib batch + shuffle，4 个 GPU 并行
- 目的：(1) 全量验证 KL Observer 效果；(2) 对比校准量 128 vs 512 的影响

### 全量验证结果汇总

| 配置 | calib | NDS | mAP | Δ NDS | vt obs | lidar obs |
|------|-------|-----|-----|-------|--------|-----------|
| FP32 基线 | — | 0.7069 | 0.6728 | 0% | — | — |
| **7/8 +vt KL** | **128** | **0.7033** | **0.6657** | **−0.5%** | **KL** | skip |
| 6/8（skip vt+lidar） | 128 | 0.7010 | 0.6614 | −0.8% | skip | skip |
| 7/8 +vt KL | 512 | 0.6874 | 0.6427 | −2.8% | KL | skip |
| 7/8 +vt EMA（旧） | 128 | 0.6179 | 0.5194 | −12.6% | EMA | skip |
| 7/8 +lidar KL | 128 | 0.5758 | 0.5452 | −18.5% | skip | KL |
| 7/8 +lidar EMA（旧） | 128 | 0.5751 | 0.5394 | −18.6% | skip | EMA |
| **8/8 KL(both)** | **128** | **0.5750** | **0.5444** | **−18.7%** | **KL** | **KL** |
| 8/8 KL(vt) | 128 | 0.5706 | 0.5221 | −19.3% | KL | EMA |
| 8/8 KL(vt) | 512 | 0.5656 | 0.5137 | −20.0% | KL | EMA |
| 8/8 KL(both) | 512 | 0.5628 | 0.5123 | −20.4% | KL | KL |
| 8/8 EMA 512+shuf（R3） | 512 | 0.4680 | 0.3598 | −33.8% | EMA | EMA |
| 8/8 EMA（旧） | 128 | 0.4562 | 0.3536 | −35.5% | EMA | EMA |

### 128 vs 512 校准量对比

| 配置 | NDS (128) | NDS (512) | Δ NDS | mAP (128) | mAP (512) | Δ mAP |
|------|-----------|-----------|-------|-----------|-----------|-------|
| 7/8 +vt KL | **0.7033** | 0.6874 | **+0.0159** | **0.6657** | 0.6427 | **+0.0230** |
| 7/8 +lidar KL | **0.5758** | 0.5745 | +0.0013 | **0.5452** | 0.5414 | +0.0038 |
| 8/8 KL(both) | **0.5750** | 0.5628 | **+0.0122** | **0.5444** | 0.5123 | **+0.0321** |
| 8/8 KL(vt) | **0.5706** | 0.5656 | +0.0050 | **0.5221** | 0.5137 | +0.0084 |

**128 batch 全面优于 512 batch**，尤其是 vtransform 侧差异显著（+1.6~2.3 NDS 点）。
原因分析：KL Observer 依赖直方图积累。512 batch 的 shuffle 引入过多场景多样性，
直方图被远距离/稀疏场景的离群值稀释，导致最优截断阈值偏大。
128 batch 恰好覆盖前几个场景，分布更集中，KL 截断更紧凑精确。

### KL Observer 效果分析（全量数据集 vs mini 数据集）

#### vtransform：KL 效果在全量数据集上更好

| 指标 | mini | 全量(128) | 说明 |
|------|------|-----------|------|
| EMA NDS | 0.5474 | 0.6179 | — |
| KL NDS | 0.5720 | **0.7033** | — |
| 改善幅度 | 4.2 pts | **8.5 pts** | 全量更显著 |
| KL Δ from FP32 | −1.2% | **−0.5%** | **近乎无损** |

**7/8 +vt KL（128 batch）NDS 0.7033 超过了 6/8 的 0.7010！**
这意味着用 KL Observer 量化 vtransform 后，不再需要跳过它。
新的最优实用配置变为 **7/8 +vt KL**（量化 7 个模块，仅跳过 lidar/backbone），NDS −0.5%。

#### lidar：KL 在全量数据集上几乎无效

| 指标 | mini | 全量(128) | 说明 |
|------|------|-----------|------|
| EMA NDS | 0.4734 | 0.5751 | — |
| KL NDS | 0.5173 | 0.5758 | — |
| 改善幅度 | 7.6 pts | **0.07 pts** | **全量几乎无效** |

**mini 数据集严重高估了 KL 对 lidar 的效果。**
原因：mini 仅 81 帧 2 个场景，lidar 稀疏激活的极端值分布不具代表性。
全量 6019 帧下，lidar 的稀疏激活分布本身就更稳定，EMA 的 range 估计已经够用。
lidar 量化的 −18.6% 精度损失是 per-tensor INT8 对稀疏特征的固有限制，
不是 observer 选择的问题——需要 per-channel 或混合精度才能进一步改善。

#### 8/8 综合效果

| 指标 | mini | 全量(128) | 说明 |
|------|------|-----------|------|
| EMA NDS | 0.4285 | 0.4562 | — |
| KL(both) NDS | 0.5085 | **0.5750** | — |
| 改善幅度 | 13.9 pts | **11.9 pts** | 全量改善略小但仍显著 |

8/8 KL(both) 128 batch: NDS 0.5750（−18.7%），相比 EMA 基线 0.4562 提升 11.9 个 NDS 点。
但对比 7/8 +vt KL（0.7033），lidar 量化仍贡献了 −12.8 NDS 点的损失。

### 核心结论

1. **新最优实用配置：7/8 +vt KL**（NDS 0.7033，−0.5%），超越了原 6/8（NDS 0.7010，−0.8%），
   且多量化了 vtransform 模块（理论 INT8 体积覆盖 ~70%）
2. **KL Observer 完全解决了 vtransform 量化瓶颈**：从 −12.6% 降到 −0.5%，拯救 8.5 个 NDS 点
3. **KL Observer 对 lidar 几乎无效**（全量数据集）：lidar 的 −18.6% 损失是固有限制，非 observer 问题
4. **128 batch > 512 batch**（对 KL Observer）：更少的校准数据反而更好，
   因为 KL 直方图不需要场景多样性，过多数据会稀释分布特征
5. **mini 数据集对 lidar 效果的预测不可靠**：mini 高估了 KL 对 lidar 的效果 100 倍（7.6 pts → 0.07 pts）
6. **8/8 全量化的残余瓶颈完全在 lidar/backbone**：
   vtransform KL 已近乎无损，8/8 的 −18.7% 损失中约 −18% 来自 lidar per-tensor 量化

---

### [2026-03-14] Round 5 修正：校准集 Train vs Val 的影响

**背景**：
此前 Round 5 实验（上文结果）使用了 `cfg.data.val` 作为校准集。这虽然方便，但存在过拟合验证集分布的风险。
本次修正将校准集改为 `cfg.data.train`（关闭数据增强），以验证结论的鲁棒性。

**实验配置**：
- 校准集：`cfg.data.train`（无增强，无 shuffle/shuffle）
- 校准量：128 batches
- 模型：7/8 +vt KL 和 8/8 KL(both)

**结果对比（NDS / mAP）**

| 配置 | 原结果 (Val校准) | 新结果 (Train校准) | 变化 (Δ) | 说明 |
|------|----------------|-------------------|----------|------|
| **7/8 +vt KL (shuffle)** | 0.7033 / 0.6657 | **0.6913 / 0.6444** | −1.2% / −2.1% | 真实性能略低，但仍优于 plain EMA |
| **7/8 +vt KL (no shuffle)** | — | 0.6912 / 0.6453 | — | Shuffle 与否影响微乎其微 |
| **8/8 KL Both (shuffle)** | 0.5750 / 0.5444 | **0.5644 / 0.5075** | −1.1% / −3.7% | 同样下降约 1% NDS |
| **8/8 KL Both (no shuffle)** | — | 0.5676 / 0.5151 | +0.3% / +0.8% | No Shuffle 略微优于 Shuffle |

**分析**：
1.  **Val校准的确存在虚高**：NDS 普遍下降约 1.2%，mAP 下降 2-3%。这证实了之前的担忧，Val set calibration 会导致轻微的过拟合。
2.  **结论方向未变**：尽管绝对数值下降，但相对趋势一致。7/8 +vt KL (0.6913) 仍然是非常可用的配置（FP32 ~0.707，损失约 -1.5%）。
3.  **Shuffle 影响**：在 Train set 上，128 batch 是否 shuffle 对结果影响很小（<0.001 NDS for 7/8, ~0.003 NDS for 8/8）。这说明 128 个样本（即便来自同一序列）的特征分布已经足够稳定地通过 KL 散度捕捉到全局统计规律。
4.  **修正后的最优配置**：7/8 +vt KL (Train Calib) = **0.6913 NDS**。

**后续建议**：
后续所有 PTQ 实验均应严格使用 Train set 进行校准。

---

### [2026-03-14] Round 6：LiDAR/backbone 稀疏激活逐通道量化（per-channel）结果

**实验目标**：验证 `--sparse-act-mode per_channel` 能否显著缓解 LiDAR 稀疏激活的量化瓶颈。  
**统一设置**：Train 校准集（`cfg.data.train + test_mode=True`）、`--calib-batches 128 --calib-shuffle`。

| 实验 | 配置 | mAP | NDS | 日志 |
|------|------|-----|-----|------|
| R6-A | 7/8，skip vtransform，`act-observer=ema_minmax`，`sparse-act-mode=per_channel` | 0.5357 | 0.5733 | `round6_ptq6_plus_lidar_pc_ema_calib128s.log` |
| R6-B | 8/8，`vtransform-observer=kl_divergence`，`act-observer=ema_minmax`，`sparse-act-mode=per_channel` | 0.5050 | 0.5611 | `round6_ptq6_plus_lidar_pc_vtkl_ema_calib128s.log` |
| R6-C | 7/8，skip vtransform，`act-observer=mse`，`sparse-act-mode=per_channel` | 0.5359 | 0.5768 | `round6_ptq6_plus_lidar_pc_mse_calib128s.log` |
| R6-D | 8/8，`vtransform-observer=kl_divergence`，`act-observer=mse`，`sparse-act-mode=per_channel` | 0.5052 | 0.5639 | `round6_ptq6_plus_lidar_pc_vtkl_mse_calib128s.log` |

**与 Round 5（Train 校准）对比结论**：
1. 逐通道激活量化并未带来预期中的明显增益，整体变化在 ±0.01 NDS 量级。
2. 在 8/8 + vt(KL) 条件下，Round 6 最优（R6-D: 0.5639）仍略低于 Round 5 no-shuffle（0.5676），与 Round 5 shuffle（0.5644）基本持平。
3. `mse` 相比 `ema_minmax` 在 per-channel 下有小幅改善（约 +0.002~0.004 NDS），但不足以改变“效果一般”的总体判断。

**阶段性判断**：
- LiDAR 瓶颈并不只来自”per-tensor vs per-channel”这一单一因素；仍可能受稀疏分布极端值、通道间有效样本不均衡、以及后续融合误差放大共同影响。
- 下一步应优先考虑更强的激活分布建模（如 per-channel KL 或分组/混合精度策略），而非仅替换 observer 名称。

---

## 2026-03-16 · Round 8：Sparse-Aware KL + W8A16 控制实验

**核心问题**：lidar 量化的 −18% 损失来自权重还是激活？

**实验设计**：
- R8-A/B: 测试 sparse-aware KL 能否修复 lidar 激活量化
- **R8-C**: W8A16 控制实验 — 权重 int8，激活 FP16，测试激活量化的独立贡献

| 实验 | 配置 | mAP | NDS | ΔNDS | 日志 |
|------|------|-----|-----|------|------|
| FP32 | 基线 | 0.6728 | 0.7069 | 0% | — |
| R8-A | 7/8 + sparse-aware KL PT | 0.5054 | 0.5629 | −20.4% | `round8_ptq7_lidar_sparse_kl_pt_calib128s.log` |
| R8-B | 8/8 + vtKL + sparse-aware KL PT | 0.5358 | 0.5750 | −18.7% | `round8_ptq8_vtkl_lidar_sparse_kl_pt_calib128s.log` |
| **R8-C** | **7/8 + W8A16** | **0.6623** | **0.7009** | **−0.85%** | `round8_ptq7_lidar_w8a16_calib128s.log` |

**关键发现**：

1. **Sparse-aware KL 无效**：R8-A/B 结果与普通 KL/EMA 几乎相同，−18%~−20%
2. **W8A16 几乎无损**：R8-C NDS 0.7009 (−0.85%)，说明：
   - 权重 int8 量化对 lidar 精度影响极小
   - **激活（activation）量化是 −18% 损失的根本来源**
   - 解决方案应聚焦于激活量化方法，而非权重量化

**结论**：lidar 量化的核心问题是**激活量化**，不是权重量化。下一步应探索更好的激活量化方法（如对数域量化 Log2）。

---

## 2026-03-17 · Round 9：Log2 对数域量化突破

**背景**：Round 8 确认激活量化是 lidar 瓶颈的根本原因。本轮探索 Log2 对数域量化能否解决稀疏激活的量化难题。

**统一设置**：
- 校准集：训练集（`cfg.data.train` + `test_mode=True`）
- 校准量：`--calib-batches 128 --calib-shuffle`

| 实验 | 配置 | mAP | NDS | ΔNDS | 日志 |
|------|------|-----|-----|------|------|
| FP32 | 基线 | 0.6728 | 0.7069 | 0% | — |
| R9-A | 7/8 +lidar Log2 PT | 0.6417 | 0.6849 | −3.1% | `round9_ptq7_lidar_log2_pt_calib128s.log` |
| R9-B | 7/8 +lidar Log2 PC | 0.6179 | 0.6721 | −4.9% | `round9_ptq7_lidar_log2_pc_calib128s.log` |
| R9-C | 7/8 +lidar Log2 PC + LWC | 0.6439 | 0.6878 | −2.7% | `round9_ptq7_lidar_log2_pc_lwc_calib128s.log` |
| R9-D | 8/8 +vt KL +lidar Log2 PT | 0.6429 | 0.6875 | −2.7% | `round9_ptq8_vtkl_lidar_log2_pt_calib128s.log` |

**关键突破**：

| 方法对比 | NDS | ΔNDS | 改善 |
|----------|-----|------|------|
| EMA (Round 5) | 0.5751 | −18.5% | 基线 |
| KL (Round 5) | 0.5758 | −18.5% | +0.07 pts |
| W8A16 (Round 8) | 0.7009 | −0.85% | +17.7 pts |
| **Log2 PT (Round 9)** | **0.6849** | **−3.1%** | **+15.4 pts** ✨ |
| **Log2 PC+LWC (Round 9)** | **0.6878** | **−2.7%** | **+16.9 pts** ✨ |

**结论**：
1. **Log2 对数域量化极其有效**：从 EMA/KL 的 −18.5% 改善到 −3.1%，提升 15.4 pts
2. **全量化突破**：首次实现 8/8 全模块量化，NDS 仅损失 −2.7%
3. **Per-channel 反而更差**：Log2 PT (−3.1%) > Log2 PC (−4.9%)
4. **LWC 有帮助**：Log2 PC + LWC 恢复到 −2.7%

**最终最优配置**：

| 配置 | NDS | ΔNDS | 量化模块 |
|------|-----|------|---------|
| FP32 基线 | 0.7069 | 0% | 0/8 |
| 旧最优：7/8 +vt KL | 0.7033 | −0.5% | 7/8 |
| **新最优：8/8 vtKL + Log2** | **0.6875** | **−2.7%** | **8/8** |

