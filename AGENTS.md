# BEVFusion + MQBench 项目说明（Agent Instructions）

## 环境

- **操作系统**：Windows，PowerShell
- **Conda 环境**：`bevfusion`（路径：`D:\aconda\envs\bevfusion`）
- **关键依赖**：PyTorch 1.10.2+cu113，mmdet3d 0.0.0（本地安装），MQBench 0.0.6
- **数据集**：`data/nuscenes`（v1.0-mini，Junction 符号链接）
- **预训练权重**：`pretrained/bevfusion-det.pth`（SwinT），`server_artifacts/resnet50_fulldata/epoch_6.pth`（ResNet-50）
- **所有 Python 命令**必须加 `$env:PYTHONUTF8="1"`，否则 Windows 会报 GBK codec 错误

运行任何脚本前的标准前置设置：
```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

## 项目文档

- `docs/REPORT.md`：完整技术报告
- `docs/RESULTS_LOG.md`：评测结果记录（按时间线，02-25 ~ 03-03 的全部实验）
- `docs/PTQ_BENCHMARK_NOTES.md`：量化覆盖问题分析、Benchmark 命令、TensorRT 导出方案
- `docs/RUNBOOK.md`：可复现运行手册（所有命令 + 服务器部署）

**开始任务前请先阅读以上文档。**

## 已验证可工作的脚本

| 脚本 | 结果（Mini 81 帧 → 完整 6019 帧） |
|------|------|
| `tools/test.py` | SwinT FP32：NDS = 0.5800 (mini) / 0.7069 (full) |
| `tools/quant_ptq_minmax.py` | SwinT PTQ 4/6：NDS = 0.5810 (mini) / 0.7015 (full) |
| `tools/quant_benchmark.py` | 可运行 |
| `tools/trt_eval_hybrid_all.py` | SwinT 4模块 TRT：FP16 NDS=0.7069, INT8 NDS=0.7022 (full) |
| `tools/trt_eval_hybrid_all.py` | ResNet-50 5模块 TRT：FP16 NDS=0.4992, INT8 NDS=0.4948 (full) |
| `tools/make_ppt.py` | PPT 生成脚本（19 页） |

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

### 5. TRT 校准器
代码中实际使用 `trt.IInt8MinMaxCalibrator`（**不是** `IInt8EntropyCalibrator2`，文档已修正）。

## 已完成的工作

### 阶段一：MQBench PTQ INT8 仿真（02-25）
- ✅ SwinT 4/6 模块量化（decoder/backbone + decoder/neck + camera/neck + fuser）
- ✅ torch.fx 兼容性修复（len(Proxy)、torch.cat 列表参数、mmcv patch）
- ✅ PTQ 精度无损：NDS +0.0009（mini），NDS −0.0054（full 6019帧）

### 阶段二：TensorRT 真实 INT8 部署（02-26 ~ 03-03）
- ✅ 逐模块 ONNX 导出 → TRT FP32/FP16/INT8 引擎构建
- ✅ Hybrid 推理：TRT 模块 + PyTorch 模块混合运行
- ✅ SwinT 4模块 TRT INT8：NDS 0.7022（−0.67%），引擎 7.4 MB
- ✅ FP16 完全无损验证（NDS 0.7069 = FP32）

### 阶段三：ResNet-50 替换方案（03-01 ~ 03-03）
- ✅ camera/backbone 替换为 ResNet-50（量化友好的纯 CNN）
- ✅ 服务器训练 6 epochs（Loss 仍在下降，未收敛）
- ✅ 量化覆盖 4/6 → 5/6（18% → 88% 参数）
- ✅ ResNet-50 5模块 TRT INT8：NDS 0.4948（−0.82%），总部署 55.7 MB
- ✅ 完整验证集（6019帧）评估，两种方案均已验证

### 阶段四：分布式训练环境适配（02-28 ~ 03-01）
- ✅ 修复 torchrun/torchpack/mpirun 兼容性
- ✅ 重写 `_init_distributed()` 支持 OMPI + torchrun 双路径
- ✅ 修正 GPU 设备编排脚本

## 待完成任务

### 任务 1：ConvNeXt 替换 camera/backbone（高优先级）

**目标**：用 ConvNeXt-Tiny 替换 ResNet-50 作为 camera/backbone，提升 FP32 精度同时保持量化友好。

**背景**：
- ConvNeXt 是纯 CNN（无 attention/动态控制流），torch.fx 完全兼容，可量化
- ConvNeXt-Tiny ~28M 参数，ImageNet 精度优于 SwinT-Tiny
- 当前代码库无 ConvNeXt，需从 mmcls/timm 引入并注册

**步骤**：
1. 引入 ConvNeXt backbone 并注册到 mmdet3d
2. 编写 config yaml（参考 ResNet-50 config）
3. 用 ImageNet 预训练权重初始化
4. 服务器训练 20 epochs
5. MQBench PTQ INT8 验证 + TRT 部署

**注意**：ConvNeXt 使用 LayerNorm（非 BN），INT8 量化精度需实测验证。

### 任务 2：PointPillars 替换 lidar/backbone（高优先级）

**目标**：用 PointPillars 替换 SparseEncoder，解决 LiDAR 流不可量化的问题。

**背景**：
- SparseEncoder 用稀疏卷积（spconv），**三条路全堵**：fx 追踪失败、ONNX 无法导出、TRT 无原生支持
- PointPillars 已有代码和 config：`mmdet3d/models/backbones/pillar_encoder.py`，`configs/nuscenes/det/transfusion/secfpn/lidar/pointpillars.yaml`
- PillarFeatureNet 使用全连接层（可量化），PointPillarsScatter 使用 scatter 索引操作
- PointPillarsScatter 的动态索引阻止 fx 追踪，但 ONNX 可导出（ScatterND opset 11），可直接走 TRT 原生 INT8

**步骤**：
1. 用已有 PointPillars config 验证 FP32 精度
2. 服务器训练
3. 尝试 MQBench PTQ（PillarFeatureNet 部分），scatter 部分跳过
4. ONNX 导出 → TRT INT8 部署
5. 端到端 NDS 评估

**预期**：精度可能略低于 SparseEncoder，但解锁了 LiDAR 流的量化部署。

### 任务 3：INT4 量化实验（中优先级）

**目标**：探索 INT4（W4A8）量化，对比 INT8 的精度-体积权衡。

**背景**：
- MQBench `QuantizeScheme` 支持 `bit=4`
- 使用 `BackendType.Academic` + `extra_qconfig_dict` 配置 W4A8
- INT4 精度通常需要 AdaRound/BRECQ 等高级 PTQ 方法（MQBench `advanced_ptq.py` 已实现）
- **不需要 TRT 部署**——MQBench FakeQuant 仿真即为完整的量化实验结果

**步骤**：
1. 修改 `quant_ptq_minmax.py`，支持 `BackendType.Academic` + 可配置 bit width
2. W4A8 MinMax PTQ → 评估 NDS（预计有明显下降）
3. W4A8 AdaRound PTQ → 评估 NDS（预计显著改善）
4. 可选：W4A4、W8A8 AdaRound 对比
5. 绘制 bit width vs NDS vs 理论模型大小 的权衡曲线

**MQBench INT4 配置示例**：
```python
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.scheme import QuantizeScheme

prepare_by_platform(model, BackendType.Academic,
    prepare_custom_config_dict={
        'extra_qconfig_dict': {
            'w_qscheme': QuantizeScheme(symmetry=True, per_channel=True, bit=4),
            'a_qscheme': QuantizeScheme(symmetry=True, per_channel=False, bit=8),
        }
    })
```

**MQBench AdaRound 使用**：
```python
from mqbench.advanced_ptq import ptq_reconstruction

ptq_reconstruction(model, cali_data, config={
    'pattern': 'block',
    'scale_lr': 4e-5,
    'warm_up': 0.2,
    'weight': 0.01,
    'max_count': 20000,
    'b_range': [20, 2],
    'keep_gpu': True,
    'round_mode': 'learned_hard_sigmoid',
})
```

### 任务 4：推理速度测量（低优先级）

**目标**：在 `trt_eval_hybrid_all.py` 中添加计时代码，测量 FP32/FP16/INT8 的端到端推理延迟。

### 任务 5：TransFusionHead 量化（低优先级）

**目标**：通过静态展开 ModuleList 迭代或手动插入量化节点，解决 heads/object 的 fx 追踪问题。参数量小（1.04M / 4.0 MB），体积收益有限，但可提升覆盖率。

## 当前模块量化状态与目标

| 模块 | SwinT 方案 | ResNet-50 方案 | ConvNeXt+PointPillars 目标 |
|------|-----------|---------------|--------------------------|
| camera/backbone | ❌ SwinT 动态控制流 | ✅ ResNet-50 | ✅ ConvNeXt（纯 CNN） |
| camera/neck | ✅ | ✅ | ✅ |
| camera/vtransform | ❌ bev_pool CUDA 算子 | ❌ | ❌ |
| lidar/backbone | ❌ 稀疏卷积 | ❌ | ✅ PointPillars（FC 层） |
| fuser | ✅ | ✅ | ✅ |
| decoder/backbone | ✅ | ✅ | ✅ |
| decoder/neck | ✅ | ✅ | ✅ |
| heads/object | ❌ Proxy 迭代 | ❌ | ❌（低优先级） |
| **可量化模块** | **4/8** | **5/8** | **6/8 目标** |

## 重要技术备忘

### MQBench 的定位
- MQBench 是**核心量化工具**：PTQ 仿真（FakeQuant）+ 精度评估
- TRT 是**部署验证**：证明 FakeQuant 预测与真实 INT8 部署精度一致
- FakeQuant 的 NDS/mAP 就是量化结果，**不依赖 TRT**
- MQBench 可做 INT4/INT8 等任意 bit width 的 FakeQuant 仿真
- MQBench 支持 AdaRound/BRECQ/QDrop 等高级 PTQ 方法（`advanced_ptq.py`）

### 分布式训练
- 服务器使用 `torchrun --nproc_per_node=N --standalone` 启动（已替代 torchpack dist-run）
- `tools/train.py` 已修改支持 OMPI + torchrun 双路径
- 训练脚本：`tools/scripts/train_resnet50_server.sh`

### 模型大小对比（公平比较用纯推理权重）
- SwinT：155.91 MB（纯推理权重 .pth）
- ResNet-50：142.8 MB（纯推理权重）/ 420.7 MB（训练 .pth 含 optimizer 277.9 MB）

### LiDAR 流量化障碍分析
- **SparseEncoder**：使用 `SubMConv3d`/`SparseConv3d`（自定义 spconv），`SparseConvTensor` 非标准张量格式 → fx 追踪失败 + ONNX 无法导出 + TRT 无原生支持
- **PointPillars 替代方案**：`PillarFeatureNet`（FC 层，可量化）+ `PointPillarsScatter`（scatter 索引，fx 不兼容但 ONNX 可导出）
- **bev_pool**：`QuickCumsumCuda.apply()` 自定义 CUDA kernel + `argsort`/`torch.where` 动态索引 → 不可量化，不可 ONNX 导出
