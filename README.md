# BEVFusion + MQBench 量化工具集

> 本项目在 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion)（基于 mmdetection3d）的基础上，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具，实现了面向 TensorRT INT8 后端的**训练后量化（PTQ）**与**量化感知训练（QAT）**。

---

## 目录

- [项目简介](#项目简介)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [预训练模型](#预训练模型)
- [PTQ：训练后量化](#ptq训练后量化)
- [QAT：量化感知训练](#qat量化感知训练)
- [Benchmark 工具](#benchmark-工具)
- [推荐工作流](#推荐工作流)
- [TensorRT 部署](#tensorrt-部署)
- [致谢](#致谢)

---

## 项目简介

本项目以 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion)（多传感器融合自动驾驶感知模型）为基础，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，新增以下量化脚本：

| 脚本 | 功能 |
|------|------|
| `tools/quant_ptq_minmax.py` | **PTQ** — MinMax 校准，训练后量化 |
| `tools/quant_train.py` | **QAT** — 量化感知训练，端到端微调 |
| `tools/quant_benchmark.py` | **Benchmark** — 模型大小与推理延迟测量 |

目标后端：**NVIDIA TensorRT INT8**

---

## 环境安装

### 基础依赖

```bash
# Python 3.8（推荐）
# CUDA 11.1 或 11.3，PyTorch 1.9.0 ~ 1.10.2

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0
pip install mmdet==2.20.0
pip install nuscenes-devkit
pip install torchpack
```

### 安装本项目

```bash
python setup.py develop
```

> Windows 用户注意：如需重新编译 CUDA 扩展，需确保 Visual Studio 编译器环境可见。详见 [CLIlog.md](CLIlog.md)。

### 安装 MQBench

```bash
# 从 PyPI 安装
pip install mqbench

# 验证安装
python -c "import mqbench; print('MQBench 安装成功')"
```

---

## 数据准备

本项目使用 [nuScenes](https://www.nuscenes.org/) 数据集。请按照 [mmdetection3d 文档](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) 下载并预处理数据集。

预处理完成后，目录结构如下：

```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_dbinfos_train.pkl
```

生成 info 文件：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes --extra-tag nuscenes
```

---

## 预训练模型

下载 BEVFusion 预训练权重：

```bash
./tools/download_pretrained.sh
```

或手动下载到 `pretrained/` 目录：

| 文件 | 说明 |
|------|------|
| `bevfusion-det.pth` | BEVFusion 检测模型（Camera+LiDAR，nuScenes val mAP=68.52） |
| `swint-nuimages-pretrained.pth` | Swin Transformer Backbone 预训练权重 |

---

## PTQ：训练后量化

**脚本**：`tools/quant_ptq_minmax.py`

### 原理

PTQ（Post-Training Quantization）无需重新训练，仅需少量校准数据（几十到几百个 batch）即可确定量化参数。本项目采用 **MinMax 校准**策略：

1. 对可量化子模块逐一调用 `prepare_by_platform`，插入 FakeQuantize 节点
2. `enable_calibration`：在校准数据上前向推理，记录各层激活值的 min/max
3. `enable_quantization`：冻结 Observer，激活 FakeQuant，进入量化推理模式

### 选择性量化策略

BEVFusion 包含自定义 CUDA 算子，不能对全模型直接量化，因此采用**选择性量化**：

**✅ 可量化部分**（标准密集卷积，`torch.fx` 可追踪）：

| 子模块 | 说明 |
|--------|------|
| `encoders.camera.backbone` | 相机骨干网络（SwinTransformer / ResNet） |
| `encoders.camera.neck` | 相机 Neck（GeneralizedLSSFPN / FPN） |
| `fuser` | 多模态融合模块（ConvFuser） |
| `decoder.backbone` | 解码器骨干（SECOND） |
| `decoder.neck` | 解码器 Neck（SECONDFPN） |
| `heads.*` | 检测 / 分割 Head |

**❌ 跳过部分**（含自定义 CUDA 算子或稀疏卷积）：

| 子模块 | 跳过原因 |
|--------|----------|
| `encoders.camera.vtransform` | 内含 `bev_pool`（QuickCumsumCuda）自定义 CUDA autograd Function |
| `encoders.lidar.voxelize` | 点云体素化预处理，非神经网络层 |
| `encoders.lidar.backbone` | 稀疏卷积（SparseEncoder），FakeQuant 节点无法插入 |

### 使用方法

```bash
# 单 GPU（默认 128 个校准 batch）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth

# 自定义校准 batch 数量
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --calib-batches 256

# 跳过精度评估（仅校准并保存）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --no-eval

# 多 GPU 分布式
torchpack dist-run -np 8 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth
```

### 输出

量化模型保存至 `runs/<run_dir>/ptq_minmax_model.pth`，包含量化后的权重与 scale/zero_point 参数。

### 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calib-batches` | 128 | 校准 batch 数量，32~512 通常足够 |
| `--no-eval` | False | 是否跳过量化后的精度评估 |
| `--run-dir` | 自动生成 | 输出目录 |

---

## QAT：量化感知训练

**脚本**：`tools/quant_train.py`

### 原理

QAT（Quantization-Aware Training）在训练时模拟量化效果，通过梯度反传更新权重，相比 PTQ 通常能获得更高精度。

**量化策略**（TensorRT INT8 后端）：
- 权重：Per-channel 对称量化
- 激活：Per-tensor 量化
- 使用 Straight-Through Estimator (STE) 近似量化梯度

### 关键概念：Leaf Modules

`torch.fx` 无法追踪包含自定义 CUDA 扩展的模块，必须将其标记为 **leaf modules**，让 `torch.fx` 将其视为不可分割的原子操作。

BEVFusion 中需要设为 leaf 的核心模块：

| 类别 | 模块 |
|------|------|
| BEV Pooling | `QuickCumsum`、`QuickCumsumCuda` |
| 稀疏卷积 | `SparseModule`、`SparseConvolution`、`SparseMaxPool` 等 |
| 体素化 | `Voxelization`、`DynamicScatter` |
| 视图变换 | `BaseTransform`、`LSSTransform` |
| 其他 | ROI Pooling、Point Sampling、PAConv、Group Points |

完整列表见 `tools/quant_train.py` 中的 `get_leaf_modules_for_mmdet3d()` 函数。

### QAT 训练流程

```
浮点预训练模型
    ↓ prepare_by_platform（插入 FakeQuantize 节点）
    ↓ enable_calibration（收集激活统计信息）
    ↓ enable_quantization（激活 FakeQuant）
    ↓ QAT 微调（端到端梯度更新）
    ↓ 保存量化模型
```

### 使用方法

```bash
# 单 GPU
python tools/quant_train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth

# 多 GPU 分布式（推荐）
torchpack dist-run -np 8 python tools/quant_train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/qat_bevfusion
```

### 训练超参数建议

| 参数 | 浮点训练 | QAT 建议值 | 说明 |
|------|----------|-----------|------|
| 学习率 | 2e-3 | **2e-4** | QAT 使用约 1/10 的学习率，避免破坏量化参数的稳定性 |
| Epoch | 20 | **10~20** | QAT 收敛较快 |
| Batch Size | 保持一致 | 保持一致 | 无需调整 |
| 权重衰减 | 0.01 | 0.01 | 无需调整 |

### 精度预期

- 精度下降：绝对精度下降应控制在 **1~2 个百分点**以内（mAP / NDS）
- 训练时间：8× A100 约 **2~4 小时**（20 epoch）
- 显存占用：与浮点训练相当

### 故障排除

**`torch.fx` 追踪错误**（`RuntimeError: Could not run 'aten::xxx'`）  
→ 在 `get_leaf_modules_for_mmdet3d()` 中添加对应模块类

**精度下降过大（> 5%）**  
→ 降低学习率至 1e-4；增加训练周期至 30 epoch；增加校准数据量

**显存不足（OOM）**  
→ 减小 `samples_per_gpu`；使用梯度累积；启用 gradient checkpointing

---

## Benchmark 工具

**脚本**：`tools/quant_benchmark.py`

用于报告模型大小（参数量、FP32 内存、估算 INT8 大小）及测量 GPU 推理延迟，支持 FP32 与量化模型的横向对比。

```bash
# 仅报告 FP32 模型大小（无需数据集）
python tools/quant_benchmark.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint pretrained/bevfusion-det.pth \
    --size-only

# 使用真实验证集数据测量推理延迟
python tools/quant_benchmark.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint pretrained/bevfusion-det.pth \
    --use-real-data --num-iters 50

# 对比 FP32 与量化模型
python tools/quant_benchmark.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint pretrained/bevfusion-det.pth \
    --quant-checkpoint runs/ptq_minmax/ptq_minmax_model.pth \
    --use-real-data --num-iters 50
```

输出示例：

```
============================================================
  模型大小报告 [FP32]
============================================================
  可训练参数量:  69,838,336  (69.84 M)
  FP32 内存占用: 266.45 MB
  估算 INT8 大小: 66.61 MB  (FP32 / 4，仅供参考)
============================================================
  推理延迟报告 [FP32] (共 50 次)
============================================================
  均值:   125.34 ms
  P95:    138.20 ms
  FPS:    7.98
============================================================
```

---

## 推荐工作流

```
① 下载预训练浮点模型
        ↓
② PTQ MinMax（tools/quant_ptq_minmax.py）
   快速获取量化基线，只需校准数据，无需训练
        ↓
③ 评估精度（tools/test.py）
   检查 mAP / NDS 是否满足要求
        ↓
   精度可接受 ──→ ④ Benchmark（tools/quant_benchmark.py）
                     测量推理速度，准备 TensorRT 部署
        ↓
   精度不足   ──→ ⑤ QAT（tools/quant_train.py）
                     端到端微调恢复精度，再回到 ③
```

| 方法 | 脚本 | 精度 | 耗时 | 所需数据 |
|------|------|------|------|----------|
| PTQ MinMax | `quant_ptq_minmax.py` | ★★☆ | 最快（分钟级） | 少量校准集 |
| QAT | `quant_train.py` | ★★★ | 较慢（小时级） | 完整训练集 |

---

## TensorRT 部署

量化模型可通过 [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) 部署到 TensorRT，在 Jetson Orin 上实现约 **25 FPS** 的 INT8 推理。

---

## 致谢

- **BEVFusion**：[论文](https://arxiv.org/abs/2205.13542) | [代码](https://github.com/mit-han-lab/bevfusion)
- **MQBench**：[代码](https://github.com/ModelTC/MQBench) | [文档](https://mqbench.readthedocs.io/)
- **mmdetection3d**：[代码](https://github.com/open-mmlab/mmdetection3d)
- **CUDA-BEVFusion**：[NVIDIA TensorRT 部署方案](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
