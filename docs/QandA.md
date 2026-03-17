# BEVFusion + MQBench 技术问答文档

> 记录对项目关键技术点的深入分析，便于复盘与汇报。

---

## Q1：SwinTransformer 的量化策略是什么？数据以什么形状在其中流通？

**提问时间**：2026-03-12

---

### 1. 为什么 torch.fx 不能追踪 SwinT？

SwinT 的 Window Attention 里存在**动态控制流**：

```python
# 伪代码（AdaptivePadding / window_partition 等）
if H % window_size != 0:
    x = F.pad(x, ...)   # 依赖运行时 tensor 值的动态分支
windows = x.view(B, H//Wh, Wh, W//Ww, Ww, C)  # 动态 reshape
```

torch.fx 做的是**符号追踪**（Symbolic Trace）：遇到 `if <Proxy>:` 时无法确定走哪条分支，直接抛出 `TraceError`。因此 SwinT 走**路径 2（手动包装）**，而非路径 1（torch.fx 自动插桩）。

---

### 2. 量化策略：手动 FakeQuantize 包装

代码逻辑（`tools/quant_ptq_minmax.py` line 628–660）：

```python
def _try_quantize(submodule, display_name, ...):
    is_vtransform = (display_name == "camera/vtransform")
    manual_obs = vtransform_observer_cls if is_vtransform else None
    # ↑ SwinT backbone → manual_obs = None → 强制 EMAMinMaxObserver

    # 先试 FX → 失败 → 试手动包装
    manual_quantize_nontraceable(submodule, act_observer_cls=manual_obs)
```

`manual_quantize_nontraceable` 遍历整个 SwinT 模块树，把所有 `nn.Conv2d` 和 `nn.Linear` 替换为带 FakeQuant 的包装器：

| 被包装的层类型 | 在 SwinT 里的位置 |
|---|---|
| `nn.Conv2d` | PatchEmbed（`patch_embed.proj`） |
| `nn.Linear` | WindowAttention 的 Q/K/V 投影（`attn.qkv`）、输出投影（`attn.proj`） |
| `nn.Linear` | MLP 的 `fc1`、`fc2`（每个 SwinT Block 都有） |

**不被量化**：LayerNorm、Dropout、Softmax、ReLU 等，继续以 FP32 运行。

---

### 3. 量化参数（TensorRT INT8 标准配置）

| 对象 | 位宽 | 对称 | 粒度 | Observer |
|---|---|---|---|---|
| **权重 (Weight)** | 8-bit | ✅ 对称 `[-127, 127]` | **Per-Channel** | MinMaxObserver |
| **激活 (Activation)** | 8-bit | ✅ 对称 `[-127, 127]` | **Per-Tensor** | **EMAMinMaxObserver（硬编码，不受 `--act-observer` 影响！）** |

> ⚠️ **关键细节**：`--act-observer kl_divergence` 只能控制 vtransform 和 torch.fx 路径的模块。SwinT backbone 的激活 Observer 在代码里被硬编码为 `None`（即 EMAMinMaxObserver），无法通过 CLI 改变。这是刻意设计——SwinT 量化本身几乎无损（6/8 验证集 NDS 仅 −0.0002），EMA 已经足够。

每个被包装的层插入 2 个 FakeQuant 节点，forward 顺序：

```python
def forward(self, x):
    x = self.act_fake_quant(x)           # 1. 先量化输入激活
    w = self.weight_fake_quant(self.conv.weight)  # 2. 再量化权重
    return F.conv2d(x, w, ...)           # 3. 正常算子计算（模拟 INT8）
```

---

### 4. 数据在整个相机流水线中的形状流转

```
┌─────────────────────────────────────────────────────────────────────────┐
│  原始输入                                                                │
│  img: [B, N, 3, H, W] = [1, 6, 3, 256, 704]                           │
│  ↓ reshape（合并 batch × cameras）                                       │
│  [B*N, 3, H, W]      = [6, 3, 256, 704]                               │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────────┐
│  SwinTransformer（camera/backbone）  ← 手动量化路径                      │
│  配置：embed_dims=96, depths=[2,2,6,2], num_heads=[3,6,12,24]           │
│                                                                         │
│  PatchEmbed:                                                            │
│    Conv2d(3→96, kernel=4, stride=4)                                    │
│    [6, 3, 256, 704] → [6, 96, 64, 176]          ← ✅ 量化               │
│                                                                         │
│  Stage 0（depth=2, heads=3, 96-ch）:                                    │
│    tokens: [6, 64×176=11264, 96]（内部序列形式）                         │
│    Q/K/V Linear: 96→96×3，proj Linear: 96→96    ← ✅ 量化               │
│    MLP fc1/fc2: 96→384→96                       ← ✅ 量化               │
│    PatchMerging: [6, 96, 64, 176] → [6, 192, 32, 88]                  │
│                                                                         │
│  Stage 1（depth=2, heads=6, 192-ch）:                                   │
│    Q/K/V Linear: 192→192×3，MLP: 192→768→192    ← ✅ 量化               │
│    输出: [6, 192, 32, 88]    ← out_indices[1]                          │
│    PatchMerging → [6, 384, 16, 44]                                     │
│                                                                         │
│  Stage 2（depth=6, heads=12, 384-ch）:                                  │
│    Q/K/V Linear: 384→384×3，MLP: 384→1536→384   ← ✅ 量化              │
│    输出: [6, 384, 16, 44]    ← out_indices[2]                          │
│    PatchMerging → [6, 768, 8, 22]                                      │
│                                                                         │
│  Stage 3（depth=2, heads=24, 768-ch）:                                  │
│    Q/K/V Linear: 768→768×3，MLP: 768→3072→768   ← ✅ 量化              │
│    输出: [6, 768, 8, 22]     ← out_indices[3]                          │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │ 3路多尺度输出（stages 1, 2, 3）
┌───────────────────────▼─────────────────────────────────────────────────┐
│  GeneralizedLSSFPN（camera/neck）  ← torch.fx 自动量化路径               │
│  in_channels: [192, 384, 768] → out_channels: 256                      │
│                                                                         │
│  Top-down FPN 融合（上采样 + concat + conv）:                            │
│    Level 0（stride 8）:  → [6, 256, 32, 88]                            │
│    Level 1（stride 16）: → [6, 256, 16, 44]                            │
│    Level 2（stride 32）: → [6, 256, 8,  22]                            │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │ 取最细粒度（stride 8）: [6, 256, 32, 88]
┌───────────────────────▼─────────────────────────────────────────────────┐
│  DepthLSSTransform（camera/vtransform）← 手动量化路径，KL Observer 关键  │
│                                                                         │
│  depthnet: 预测 D=118 个深度 bin 概率（dbound=[1.0, 60.0, 0.5]）        │
│  feature_size = [256//8, 704//8] = [32, 88]                            │
│                                                                         │
│  Lift:  [6, 256, 32, 88] + 深度概率 → 投影到 3D voxel（稠密 4D tensor） │
│  Splat: Pool 到 BEV：xbound=[-54,54,0.3]→360格，downsample=2           │
│                                                                         │
│  输出: [1, 80, 180, 180]  ← 360/2=180（相机 BEV 特征）                  │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────────┐
│  ConvFuser（fuser）  ← torch.fx 自动量化路径                             │
│                                                                         │
│  相机 BEV:   [1,  80, 180, 180]（来自 vtransform）                      │
│  LiDAR BEV: [1, 256, 180, 180]（SparseEncoder 8×下采样：1440→180）      │
│  Concat → Conv2d(336→256) → [1, 256, 180, 180]                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 5. 一句话总结

> SwinT 使用**手动逐层替换**（Path 2），将所有 Conv2d 和 Linear 包装为带 FakeQuant 的等价层。权重 INT8 per-channel，激活 INT8 per-tensor，均使用对称量化 `[-127, 127]`，激活 Observer 固定为 EMAMinMaxObserver。SwinT 量化本身几乎无损（仅 −0.0002 NDS），瓶颈在 vtransform（深度 bin 激活分布极端，已通过 KL Observer 解决，−12.6% → −0.5%）和 lidar（稀疏激活 per-tensor 粒度太粗，尚未解决）。

---

## Q2：SwinT 到底量化了哪些层？这种量化程度有实际部署价值吗？

**提问时间**：2026-03-12

### SwinT 里被量化的层（仅此而已）

| 位置 | 层 | 数量 |
|---|---|---|
| PatchEmbed | Conv2d(3→96) | 1 |
| 每个 Block 的 Attention | `qkv` Linear、`proj` Linear | 2 × 12块 = 24 |
| 每个 Block 的 MLP | `fc1` Linear、`fc2` Linear | 2 × 12块 = 24 |
| Stage 间 PatchMerging | Linear | 3 |

**共约 52 个层，每个插 2 个 FakeQuant（权重 + 激活）。**

未量化，全程 FP32：`Q×K^T`、`Attn×V`、`Softmax`、`LayerNorm`、`GELU`、`Dropout`。

---

### 这种量化程度有实际部署价值吗？

**结论：理论验证价值有，实际部署目前没有。**

#### 为什么部署不了？

SwinT 有动态控制流（window partition、AdaptivePadding 等），**无法导出 ONNX/TRT**。本项目 TRT Hybrid 只部署了 4 个 fx 兼容模块（neck/fuser/decoder），SwinT 始终跑在 PyTorch FP32。FakeQuant 的 −0.0002 NDS 是纸面数字，不对应任何真实 INT8 部署。

#### 即使将来能导出，覆盖率够吗？

**够的——SwinT 的计算瓶颈恰好在 MLP，不在 Attention。**

原因：SwinT 用的是**局部窗口 Attention**，每个窗口只有 7×7=49 个 token，`Q×K^T` 是 49×49 的小矩阵乘法，计算量极小。MLP（fc1/fc2 有 4× channel 膨胀）才是主体计算：

```
Stage 0 举例（64×176=11264 tokens）：
  MLP:      11264 × (96×384 + 384×96) ≈ 830M FLOPs  ← 已量化 ✅
  Attention: (11264/49) 窗口 × 49²×96 ≈  53M FLOPs  ← 未量化 ❌
```

MLP 计算量约是窗口 Attention 的 **15×**。量化 Linear 层实际覆盖了 SwinT ~85% 的计算量。

#### 总结

| 维度 | 结论 |
|---|---|
| 精度验证 | ✅ 有价值：证明量化误差几乎为零（−0.0002 NDS），为未来完整部署奠定基础 |
| 当前实际部署 | ❌ 无法实现：SwinT 动态控制流不可 TRT 导出 |
| 计算量覆盖率 | ✅ 足够：Linear/Conv2d 覆盖 SwinT ~85% 计算（MLP 主导，窗口 Attention 很小） |
| 内存节省 | ✅ 理论上权重压缩 ~50%（如果能部署） |
| 瓶颈在哪 | 不是量化粒度，是**导出工程问题**（动态形状、自定义 op） |

---


---

## Q3：LiDAR/backbone 量化诊断 ——"权重 waste 22~45%，激活最大/中位比 15.7"从何而来？

**提问时间**：2026-03-12

> 诊断脚本：`tools/diag_lidar_distribution.py`；
> 诊断图像：`results_vis/lidar_diag_server/`（已存档，服务器 03-09 跑出）。
> 终端表格输出未保存为文件，下列数值均从存档图像中读出。

---

### 1. 诊断指标定义

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `W_waste%` | `1 - p99.9(|w|) / max(|w|)` | MinMax 校准下权重量化范围中被最大离群点"浪费"掉的比例 |
| `A_waste%` | `1 - p99.9(act) / max(act)` | 同上，用于激活 |
| `A_out%`   | `|x| > 3σ` 的比例 | 激活重尾程度（高斯下期望 0.27%） |
| `max/std`  | `max(|act|) / std(act)` | 离群点放大倍数，即文档中"激活最大/中位比" |

---

### 2. 四个具体层的完整数字（从诊断图读出）

#### Layer: `conv_input.0`（第一层输入 SparseConv，权重 n=2,160）

**权重分布**
- mean=0.012, std=0.405
- min=**−15.239**, max=5.483，p99.9=3.430，p99.99=13.133
- `W_out%`=0.37%（权重本身离群不多）
- **`W_waste%` = 77.5%**（原因：单个权重值 −15.24 是 p99.9=3.43 的 4.4×，把 MinMax scale 撑大）

**激活分布**（n=200,000 active voxel 特征值）
- mean=0.134, std=**17.55**
- min=−292.2, max=219.5，p99.9=109.8，p99.99=152.5
- `A_out%`=**2.47%**（最高，重尾最严重）
- **`A_waste%` = 62.4%**
- `max/std` = 292.2 / 17.55 = **16.7** ← 文档里"最大/中位比 15.7"的来源层

> **结论**：`conv_input.0` 的权重有一个极端离群值（−15.24），但这是真实训练权重，LWC 裁掉它会破坏模型功能。激活的 max=292 vs std=17.5，62% 量化范围被 0.1% 的极端激活占据。

---

#### Layer: `encoder_layers.encoder_layer1.0.conv1`（权重 n=6,912）

**权重分布**
- min=−1.743, max=1.711，p99.9=0.808，`W_out%`=1.39%，**`W_waste%`=53.6%**

**激活分布**
- mean=−0.528, std=2.482
- min=−16.897, max=16.886，p99.9=10.596，`A_out%`=1.38%，**`A_waste%`=37.3%**
- **激活呈明显双峰分布**（zoom 图可见两个隆起）

> **双峰分布是 per-tensor 量化失败的根本原因之一**：
> - 第一峰：背景/稀疏区域的低激活值（|x| ≈ 0–5）
> - 第二峰：近物体 voxel 的高激活值（|x| ≈ 10–15）
> - per-tensor scale 被 max=16.9 决定，第一峰的 127 个 INT8 级别只有 `5/16.9 × 127 ≈ 37` 个实际用到，有效精度不足 6-bit

---

#### Layer: `encoder_layers.encoder_layer2.2.0`（下采样层，权重 n=55,296）

**权重分布**
- min=−1.122, max=0.986，p99.9=0.672，`W_out%`=1.18%，**`W_waste%`=40.1%**

**激活分布**
- mean=0.411, std=2.764
- min=−32.079, max=21.349，p99.9=14.577，`A_out%`=1.80%，**`A_waste%`=54.6%**
- max/std = 32.1/2.76 = **11.6**，单峰但重尾

---

#### Layer: `encoder_layers.encoder_layer3.0.conv1`（深层，权重 n=110,592）

**权重分布**
- min=−1.693, max=0.937，p99.9=0.697，`W_out%`=1.05%，**`W_waste%`=58.8%**

**激活分布**
- mean=−2.084, std=4.435
- min=**−73.722**, max=21.544，p99.9=29.113，`A_out%`=1.27%，**`A_waste%`=60.5%**
- 激活负方向偏斜（|min| = 73.7 >> max = 21.5），非对称重尾

---

### 3. 全网统计汇总（~25 层，来自 summary.png 图表）

| 指标 | 最小值 | 最大值 | 均值（估计） | 阈值（代码内） |
|------|-------|-------|------------|--------------|
| **W_waste%** | 25.6% | 77.5% | **~48%** | 10%（几乎全部超标） |
| **A_waste%** | 37.3% | 82.6% | **~55%** | 10%（全部超标） |
| **W_out%**   | 0.37% | 1.5%  | ~1.0% | 1%（约半数超标） |
| **A_out%**   | 0.6%  | 2.5%  | ~1.3% | 1%（大多数超标） |

关键观察：
- **全部 ~25 层的 A_waste 均超过 10% 阈值**（最低也有 37.3%），说明 EMAMinMaxObserver 对整个 lidar backbone 都是不适配的
- **最高激活 waste 达 82.6%**（在 encoder_layer2 区域某层），该层的有效 INT8 精度不足 5-bit
- **权重 waste 均值 48%** 对应权重范围利用率均值约 52%，意味着平均 48% 的 INT8 权重表示空间被离群点占据

---

### 4. 为什么 KL Observer 对 lidar 几乎无效？（+0.0007 NDS）

KL Observer 对 vtransform 有奇效（−12.6% → −0.5%），对 lidar 却只有 +0.0007。原因在于两者分布的根本差异：

| 模块 | 激活分布特征 | KL 优化效果 |
|------|------------|------------|
| **vtransform** | **离散双峰**（D=118 depth bins，大多数接近 0，极少数接近 1.0），两峰之间有清晰间隔 | ✅ KL 找到最优裁剪点，把 scale 收窄到有用区间，−12.6% → −0.5% |
| **lidar** | **连续重尾分布**（近似 Laplacian/heavy-Gaussian），无清晰的"主峰"与"离群点"边界 | ❌ KL 散度找不到理想裁剪点，因为裁掉任何值都会损失信息，提升极微 |

简言之：**KL 需要"主体分布"与"离群点"之间有明显分离**，vtransform 有（深度概率分布天然二极化），lidar 没有（SparseConv 特征是连续分布）。

---

### 5. 为什么 LWC（Learned Weight Clipping）也无效？

LWC 的理论假设：**权重量化误差是主要瓶颈**，通过学习裁剪比率 `clip_ratio` 减少权重 waste，提升精度。

实际数据：
- **权重 waste 均值 48%**：确实存在，但每层 p99.9/max 之比说明 outlier 是"少量极端值"
- **真正瓶颈是激活 waste 37–83%**：`conv_input.0` 激活 waste 62%，即使权重量化做到完美，激活量化误差仍然主导输出误差
- LWC 只优化权重 clip，不碰激活 scale → **治标不治本**
- Round 3 实测：LWC 从 NDS 0.4562 → 0.4671，仅 +0.0109（约 2.4% 改善），与随机波动相近

---

### 6. 结论与待攻克方向

```
lidar/backbone 量化损失 ≈ −18.5% NDS
根本原因：per-tensor INT8 激活量化 + 重尾/双峰分布

已尝试失效的方案：
  ❌ EMAMinMaxObserver：被极端激活值主导 scale
  ❌ KL Observer：分布连续无清晰间隔，提升 +0.0007
  ❌ LWC（权重裁剪）：瓶颈在激活，权重优化无济于事
  ❌ LWC + MSEObserver：同上，且未单独测试 MSE 激活 Observer

待尝试的方案（按预期收益排序）：
  🔵 Per-channel 激活量化：对 [N_voxels, C_out] 特征逐通道估计 scale，不同通道独立校准
  🔵 AdaRound/BRECQ：联合优化权重-激活量化误差，超越贪心逐层校准
  🔵 MSEObserver（仅激活）：裁掉 A_out% 1-2.5% 的激活离群点，预期改善 A_waste 10-20%
  🔵 Mixed precision：lidar 用 INT16 或 FP16，其余 INT8
```

---
