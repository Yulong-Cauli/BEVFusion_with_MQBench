# BEVFusion + MQBench 项目说明（Agent Instructions）

## 环境

- **操作系统**：Windows，PowerShell（本地）/ Linux bash（服务器）
- **本地 Conda 环境**：`bevfusion`（路径：`D:\aconda\envs\bevfusion`）
- **服务器 Conda 环境**：`bevfusion_mqbench`（路径：`/home/yellowstone/anaconda3/envs/bevfusion_mqbench`）
- **服务器地址**：`yellowstone@10.129.51.101`，项目路径：`/media/yellowstone/data2/CYL/BEVFusion_with_MQBench`
- **服务器 GPU**：`CUDA_DEVICE_ORDER=PCI_BUS_ID` 下 GPU#0/1/3/4 = RTX 3090，GPU#2 = A100（共享，不用）
- **关键依赖**：PyTorch 1.10.2+cu113，mmdet3d 0.0.0（本地安装），MQBench 0.0.6
- **数据集（本地）**：`data/nuscenes`（v1.0-mini，81 帧，2 个场景）
- **数据集（服务器）**：`data/nuscenes`（v1.0-trainval，6019 帧，~150 个场景）
- **预训练权重**：`pretrained/bevfusion-det.pth`（SwinT BEVFusion 原始权重）
- **所有 Python 命令（本地）**必须加 `$env:PYTHONUTF8="1"`，否则 Windows 会报 GBK codec 错误

本地运行任何脚本前的标准前置设置：
```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

## 项目文档（开始任务前必读）

- `docs/REPORT.md`：完整技术报告（含所有实验结果、方法分析）
- `docs/RESULTS_LOG.md`：实验结果时间线记录（02-25 至今，含最新消融结果）
- `docs/SERVER_DEPLOY.md`：服务器部署手册（含所有 Round 的完整运行命令）
- `docs/PTQ_BENCHMARK_NOTES.md`：量化覆盖问题分析、TensorRT 导出方案
- `docs/RUNBOOK.md`：可复现运行手册

## 核心研究方向（导师最新指导：03-09）

**只做量化，不改模型架构**。ResNet-50/ConvNeXt 替换方案已降为辅助验证，不是主线。
主线是：在原始 SwinT BEVFusion 上，做尽可能完整和精细的 PTQ INT8 量化研究。

## 当前实验状态（03-11）

### 已完成的核心实验（全部 6019 帧完整验证集）

| 实验 | 量化模块 | NDS | mAP | 说明 |
|------|---------|-----|-----|------|
| FP32 基线 | 0/8 | 0.7069 | 0.6728 | 基准 |
| PTQ 6/8（推荐配置） | 6/8 | 0.7010 | 0.6614 | skip vtransform+lidar，**−0.83%** |
| PTQ 7/8 +vtransform | 7/8 | 0.6179 | 0.5194 | **−12.6%**，bev_pool 激活量化 |
| PTQ 7/8 +lidar | 7/8 | 0.5751 | 0.5394 | **−18.6%**，稀疏激活量化 |
| PTQ 8/8 MinMax 基线 | 8/8 | 0.4562 | 0.3536 | **−35.5%**，vtransform+lidar 协同劣化 |
| PTQ 8/8 + LWC | 8/8 | 0.4545 | 0.3540 | LWC 无效（0.11% 截断率，MSE 1e-5） |
| PTQ 8/8 + MSEObserver | 8/8 | 0.4560 | 0.3522 | MSEObserver 无效 |
| PTQ 8/8 + LWC+MSE | 8/8 | 0.4526 | 0.3519 | 组合反而略差 |
| PTQ 8/8 + 512+shuffle | 8/8 | 0.4680 | 0.3598 | 校准多样性有帮助（+2.6%） |

### Round 4：KL 散度 Observer（本地 mini 验证集初步结果）

**分支**：`exp/lss-kl-divergence-calibration`

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
| **8/8 KL(both)** | **0.5085** | **0.4817** | **−12.1%** | KL | KL |

**KL Observer 总结**：
- vtransform: −5.4% → −1.2%（拯救 4.2 NDS 点）
- lidar: −18.2% → −10.6%（拯救 7.6 NDS 点）
- 8/8 综合: −26.0% → −12.1%（拯救 13.9 NDS 点）
- **待服务器全量验证**

**已确认结论**：
1. **6/8 是当前最优实用配置**（NDS −0.83%，理论 INT8 体积 −65%）
2. **KL Observer 大幅改善 8/8 精度**（mini 数据集 +13.9 NDS 点），待服务器确认
3. vtransform 量化瓶颈本质是 EMAMinMax 的 range waste，KL 截断几乎消除（−1.2% vs −5.4%）
4. lidar 量化瓶颈更深层（稀疏激活 per-tensor 量化固有困难），KL 有效但残留 −10.6%

### Round 3 结果（已完成，2026-03-10）

| GPU | 实验 | NDS | mAP |
|-----|------|-----|-----|
| GPU#0 | PTQ 8/8 + shuffle + 512 batch | 0.4680 | 0.3598 |
| GPU#1 | PTQ 6/8 + shuffle + 512 batch（对照） | 0.6998 | 0.6594 |
| GPU#3 | PTQ 8/8 + shuffle + 512 + LWC | 0.4671 | 0.3602 |
| GPU#4 | PTQ 8/8 + shuffle + 512 + LWC + MSE | 0.4672 | 0.3623 |

## 已验证可工作的脚本

| 脚本 | 结果 |
|------|------|
| `tools/test.py` | SwinT FP32：NDS = 0.5800 (mini) / 0.7069 (full) |
| `tools/quant_ptq_minmax.py` | 支持 --calib-batches/--calib-shuffle/--lwc/--act-observer/--skip-modules |
| `tools/quant_benchmark.py` | 可运行 |
| `tools/trt_eval_hybrid_all.py` | SwinT TRT INT8：NDS 0.7022（4模块，full） |

## 关键约束（务必遵守）

### 1. 不能破坏已有工作状态
- 修改代码后必须跑 `tools/test.py` 确认 NDS 仍在 0.578 以上
- PTQ 脚本的已有量化逻辑不能被破坏，扩大覆盖必须是向后兼容的**追加**

### 2. torch.fx 追踪兼容性
- **不要用 `torch.fx.wrap('len')`**：全局拦截所有 `len()` 调用导致连锁失败。改用 `self.num_ins` 等预计算常量。
- **mmcv 层包装器**：mmcv `Conv2d`/`ConvTranspose2d` 的 `if x.numel() == 0` 检查会触发 `TraceError`。已通过 `patch_mmcv_for_fx()` 解决。

### 3. PTQ checkpoint 不能用 test.py 直接评估
fx 改造后 state_dict key 变化，`strict=False` 会静默跳过量化模块权重。PTQ 精度评估只能通过 `quant_ptq_minmax.py` 内部流程。

### 4. MMDataParallel 包装
PTQ 校准或推理时，模型必须包在 `MMDataParallel(model, device_ids=[0])` 里。

### 5. calib-shuffle 注意事项
mmdet `build_dataloader(shuffle=True)` 需要 `dataset.flag`，但 `test_mode=True` 的数据集没有。
代码已修复（`calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.uint8)`），但如遇到新的 dataset 类型仍可能报错。

### 6. TRT 校准器
代码中实际使用 `trt.IInt8MinMaxCalibrator`（**不是** `IInt8EntropyCalibrator2`，文档已修正）。

## 下一步待做任务（按优先级）

### 【高优先级】服务器全量验证 KL Observer（Round 4）
在 6019 帧完整验证集上确认 mini 数据集的 KL Observer 结果。
需要先将分支推送到服务器，然后运行 4 个实验。

### 【中优先级】KL + 其他优化组合
- KL + LWC：KL 解决激活瓶颈后，LWC 的权重优化可能有更大发挥空间
- KL + 不同 min_percentile（0.999 vs 0.9999 vs 0.99999）

### 【中优先级】Scale Factor Trick（用户提出的方案 2）
在 `DepthLSSTransform.get_cam_feats()` 中，对 depth*S 和 features/S 做缩放。
需要在 depthnet 最后一层的 Conv2d 输出上添加 FakeQuant 节点才能生效。
详见 plan.md 的 Phase 3 分析。

### 【低优先级】AdaRound / INT4 实验
见之前文档描述。

## 当前模块量化状态

| 模块 | 状态 | 参数量 | 备注 |
|------|------|--------|------|
| camera/backbone (SwinT) | ✅ 手动路径 | ~27.5M (67.5%) | window attention 动态控制流，手动遍历 Conv2d/Linear |
| camera/neck | ✅ fx 路径 | ~2.1M (5.2%) | 无问题 |
| camera/vtransform | ✅ 手动路径 | ~1.0M (2.5%) | EMA: −12.6%, **KL: −1.2%**（mini） |
| lidar/backbone | ✅ 稀疏卷积路径 | ~5.0M (12.3%) | EMA: −18.6%, **KL: −10.6%**（mini） |
| fuser | ✅ fx 路径 | ~1.4M (3.4%) | 无问题 |
| decoder/backbone | ✅ fx 路径 | ~7.4M (18.2%) | 无问题 |
| decoder/neck | ✅ fx 路径 | ~0.3M (0.7%) | 无问题 |
| heads/object | ✅ 手动路径 | ~1.0M (2.6%) | TransFusionHead，精度几乎无影响 |
| **最优实用配置（6/8）** | skip vtransform+lidar | ~11.5M (28%) | **NDS 0.7010，−0.83%** |
| **全量化（8/8）EMA** | 全部 8 模块 | ~40.7M (100%) | **NDS 0.4562，−35.5%** |
| **全量化（8/8）KL** | 全部 8 模块 | ~40.7M (100%) | **NDS 0.5085，−12.1%**（mini，待全量验证） |

## 重要技术备忘

### 校准集结构（本地 vs 服务器）
- **本地**：mini 验证集 81 帧，2 个场景，全部数据覆盖 100%（128 batch 已够）
- **服务器**：完整验证集 6019 帧，~150 个场景，顺序排列（同一场景帧连续）
  - `shuffle=False + 128 batch`：仅覆盖前 3~4 个场景（约 2%），不具代表性
  - **推荐**：`--calib-batches 512 --calib-shuffle`，均匀覆盖所有场景

### LWC 实现细节
- 位置：`tools/quant_ptq_minmax.py` 的 `optimize_lwc()` 函数（约第 560 行）
- 针对 lidar/backbone（`_QuantizedSparseConv` 包装的 SparseConvolution 层）
- 每层独立优化 `weight_clip_ratio` 标量参数，Adam lr=0.01，500 iters
- 用校准数据前向推理计算 FakeQuant 后的权重重建 MSE 作为 loss
- **关键**：LWC 的效果高度依赖校准数据的多样性

### MQBench 的定位
- MQBench 是**核心量化工具**：PTQ 仿真（FakeQuant）+ 精度评估
- TRT 是**部署验证**：证明 FakeQuant 预测与真实 INT8 部署精度一致
- FakeQuant 的 NDS/mAP 就是量化结果，**不依赖 TRT**
- MQBench 支持 AdaRound/BRECQ/QDrop 等高级 PTQ 方法（`advanced_ptq.py`）

### LiDAR/vtransform 量化障碍分析
- **SparseEncoder（lidar/backbone）**：稀疏激活分布（大多数 voxel=0，少数靠近物体 voxel 值很大），EMAMinMax 被极端值主导，大多数激活精度丧失。Per-tensor 量化对稀疏 [N,C] 特征粒度太粗
- **vtransform（camera/vtransform）**：depthnet 输出 D=118 个 depth bin 的概率分布，INT8 量化步长 ~0.02，量化误差 ~15%，导致特征投影到错误 BEV voxel，空间位移严重
- **两者协同劣化**：8/8 时 fuser 接收两路已劣化特征，误差在融合层放大，比线性叠加还差 −0.030

### 分布式训练
- 服务器使用 `torchrun --nproc_per_node=N --standalone` 启动（已替代 torchpack dist-run）
- `tools/train.py` 已修改支持 OMPI + torchrun 双路径
- 训练脚本：`tools/scripts/train_resnet50_server.sh`

### 历史实验汇总（已完成，无需重测）
- SwinT FP32 6019帧：NDS=0.7069 ✅
- SwinT TRT FP16 6019帧：NDS=0.7069（无损）✅
- SwinT TRT INT8 6019帧：NDS=0.7022（−0.67%）✅
- ResNet-50 FP32 6019帧：NDS=0.4989 ✅
- ResNet-50 TRT INT8 6019帧：NDS=0.4948（−0.82%）✅
