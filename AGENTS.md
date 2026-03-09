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

## 当前实验状态（03-09）

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

**已确认结论**：
1. **6/8 是当前最优实用配置**（NDS −0.83%，理论 INT8 体积 −65%）
2. vtransform（depthnet 深度概率 INT8 量化）是 8/8 精度崩溃的主因（−12.6%）
3. LWC/MSEObserver 无效的根因：校准数据仅 128 帧顺序采样（约 3~4 个场景），lidar 权重本身无功能性离群点

### 正在服务器上运行（Round 3，03-09 晚）

| GPU | 实验 | 日志文件 |
|-----|------|---------|
| GPU#0 | PTQ 8/8 + shuffle + 512 batch | `logs/results_server_ptq_8of8_calib512_shuffle.log` |
| GPU#1 | PTQ 6/8 + shuffle + 512 batch（对照） | `logs/results_server_ptq_6of8_calib512_shuffle.log` |
| GPU#3 | PTQ 8/8 + shuffle + 512 + LWC（核心） | `logs/results_server_ptq_8of8_lwc_calib512_shuffle.log` |
| GPU#4 | PTQ 8/8 + shuffle + 512 + LWC + MSE | `logs/results_server_ptq_8of8_lwc_mse_calib512_shuffle.log` |

**假设**：旧实验 shuffle=False 仅覆盖前 3~4 个场景（约 2% 场景多样性），LWC 优化目标受限。
512 batch + shuffle 覆盖全部 ~150 个场景，LWC 的 MSE 优化目标更准确。

传回结果后，运行以下命令分析：
```bash
# 服务器打包
tar czf server_calib512_results.tar.gz \
    logs/results_server_ptq_8of8_calib512_shuffle.log \
    logs/results_server_ptq_6of8_calib512_shuffle.log \
    logs/results_server_ptq_8of8_lwc_calib512_shuffle.log \
    logs/results_server_ptq_8of8_lwc_mse_calib512_shuffle.log
```
```powershell
# 本地拉取
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/server_calib512_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\
cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf server_calib512_results.tar.gz
```

### 结果解读指引（收到结果后）

| 对比组 | 预期 | 意义 |
|--------|------|------|
| 8/8 shuffle512 vs 旧 8/8 (0.4562) | 可能有 +0.01~0.05 改善 | 校准多样性的独立效果 |
| 6/8 shuffle512 vs 旧 6/8 (0.7010) | 应基本不变（±0.002） | 对照：6/8 已近无损，不应变 |
| 8/8 LWC+shuffle512 vs 旧 LWC (0.4545) | **核心观测**，如有改善说明校准多样性对 LWC 有帮助 | LWC 的有效性依赖校准集质量 |
| 8/8 LWC+MSE+shuffle512 vs 旧 0.4526 | 同上 | 组合策略上限 |

**若 shuffle 后 LWC 仍无效**：说明 8/8 精度损失确实由 vtransform 主导，lidar 侧任何优化都杯水车薪。
**若 shuffle 后 LWC 有效**：更新文档结论，并考虑更大迭代次数 LWC（`--lwc-iters 2000`）。

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

### 【当前进行中】Round 3 calib-shuffle 实验
服务器正在跑，等结果即可。参见"正在服务器上运行"一节。

### 【高优先级，结果回来后】分析并更新文档
1. 解读 4 个实验结果，写入 `docs/RESULTS_LOG.md`（参考已有 Round 2 格式）
2. 更新 `docs/REPORT.md` §9.9 节（新建），更新 §12.3/12.4
3. 如果 LWC+shuffle 有改善，尝试 `--lwc-iters 2000` 进一步优化

### 【中优先级】AdaRound 实验
MQBench 内置 `ptq_reconstruction`，对 lidar/backbone 做基于任务损失的权重重建（比 LWC 更强）。
LWC 优化重建 MSE，AdaRound 直接优化量化后的任务输出误差。

实现要点：
- 在 `quant_ptq_minmax.py` 中加 `--adaround` flag
- 调用 `mqbench.advanced_ptq.ptq_reconstruction`
- 需要一批校准数据（建议 512 batch + shuffle）
- 实验配置：`--adaround --calib-batches 512 --calib-shuffle`

### 【中优先级】vtransform 细粒度量化探索
vtransform 内部 depthnet 最后一层（直接输出深度概率的 Conv2d，维度 D+C=118+80）对量化最敏感。
**可以在 `_manual_quantize_vtransform()` 函数中，把 depthnet 最后一层排除在量化之外**，
观察 NDS 能从 0.6179 恢复多少。代价只是这一层保持 FP32。

### 【低优先级】INT4 W4A8 对比实验
修改 `quant_ptq_minmax.py` 加 `--bit-width 4` 选项，用 `BackendType.Academic` + `QuantizeScheme(bit=4)`。
不需要 TRT，FakeQuant NDS 即为结果。可做 INT4/INT8 对比图。

## 当前模块量化状态

| 模块 | 状态 | 参数量 | 备注 |
|------|------|--------|------|
| camera/backbone (SwinT) | ✅ 手动路径 | ~27.5M (67.5%) | window attention 动态控制流，手动遍历 Conv2d/Linear |
| camera/neck | ✅ fx 路径 | ~2.1M (5.2%) | 无问题 |
| camera/vtransform | ✅ 手动路径 | ~1.0M (2.5%) | **量化后 NDS −12.6%**，depthnet 深度概率精度损失 |
| lidar/backbone | ✅ 稀疏卷积路径 | ~5.0M (12.3%) | **量化后 NDS −18.6%**，稀疏激活动态范围大 |
| fuser | ✅ fx 路径 | ~1.4M (3.4%) | 无问题 |
| decoder/backbone | ✅ fx 路径 | ~7.4M (18.2%) | 无问题 |
| decoder/neck | ✅ fx 路径 | ~0.3M (0.7%) | 无问题 |
| heads/object | ✅ 手动路径 | ~1.0M (2.6%) | TransFusionHead，精度几乎无影响 |
| **最优实用配置（6/8）** | skip vtransform+lidar | ~11.5M (28%) | **NDS 0.7010，−0.83%** |
| **全量化（8/8）** | 全部 8 模块 | ~40.7M (100%) | **NDS 0.4562，−35.5%**，不可用 |

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
