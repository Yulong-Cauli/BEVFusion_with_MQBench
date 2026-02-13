# BEVFusion Quantization-Aware Training (QAT) with MQBench

## 概述 (Overview)

本文档介绍如何使用 MQBench 对 BEVFusion 模型进行量化感知训练（Quantization-Aware Training, QAT），以便将模型部署到 TensorRT 进行 INT8 推理。

This document describes how to use MQBench for Quantization-Aware Training (QAT) of BEVFusion models for deployment to TensorRT with INT8 inference.

---

## 环境准备 (Environment Setup)

### 1. 安装 BEVFusion 依赖 (Install BEVFusion Dependencies)

首先，按照 [BEVFusion README](../README.md) 的说明安装基础环境。

### 2. 安装 MQBench (Install MQBench)

```bash
# 从 PyPI 安装最新版本
pip install mqbench

# 或从源码安装
git clone https://github.com/ModelTC/MQBench.git
cd MQBench
pip install -e .
```

---

## 快速开始 (Quick Start)

### 运行 QAT 训练 (Run QAT Training)

#### 单 GPU 训练 (Single GPU)

```bash
python tools/quant_train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth
```

#### 多 GPU 分布式训练 (Multi-GPU Distributed)

```bash
torchpack dist-run -np 8 python tools/quant_train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/qat_bevfusion
```

---

## 关键概念 (Key Concepts)

### 1. Leaf Modules（叶子模块）

`torch.fx` 无法追踪包含 CUDA 扩展的自定义模块，必须将这些模块标记为 "leaf modules"。

**BEVFusion 中的关键 Leaf Modules：**
- BEV Pooling (QuickCumsum, QuickCumsumCuda)
- Sparse Convolution (SparseModule, SparseConvolution)
- Voxelization (Voxelization, DynamicScatter)
- View Transformers (BaseTransform, LSSTransform)

详见 `tools/quant_train.py` 中的 `get_leaf_modules_for_mmdet3d()` 函数。

### 2. 量化感知训练流程 (QAT Pipeline)

```
浮点预训练模型 → 插入量化节点 → Calibration → QAT Fine-tuning → 导出 INT8 模型
```

---

## 自定义算子列表 (Custom Operators List)

以下是需要设置为 leaf modules 的 mmdetection3d 自定义算子：

### 1. BEV Pooling 相关
- `mmdet3d.ops.bev_pool.bev_pool.QuickCumsum`
- `mmdet3d.ops.bev_pool.bev_pool.QuickCumsumCuda`

### 2. 稀疏卷积 (Sparse Convolution)
- `mmdet3d.ops.spconv.SparseModule`
- `mmdet3d.ops.spconv.SparseConvolution`
- `mmdet3d.ops.spconv.SparseMaxPool`
- 等等

### 3. 体素化 (Voxelization)
- `mmdet3d.ops.voxel.Voxelization`
- `mmdet3d.ops.voxel.scatter_points.DynamicScatter`

### 4. 其他算子
- ROI Pooling, Point Sampling, PAConv, Group Points, View Transformers

完整列表请查看 `tools/quant_train.py` 脚本。

---

## 训练建议 (Training Tips)

- **学习率**: QAT 使用较小的学习率（通常是预训练的 1/10），推荐 1e-4 到 2e-4
- **训练周期**: 通常 10-20 个 epoch 足够恢复精度
- **精度下降**: 通过 QAT，精度下降通常可控制在 1-2% 以内

---

## 参考资料 (References)

- **BEVFusion**: [论文](https://arxiv.org/abs/2205.13542) | [代码](https://github.com/mit-han-lab/bevfusion)
- **MQBench**: [代码](https://github.com/ModelTC/MQBench) | [文档](https://mqbench.readthedocs.io/)
- **NVIDIA 部署**: [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)

---

**最后更新**: 2026-02-13
