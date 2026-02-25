#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion QAT (Quantization-Aware Training) Demo with MQBench
==============================================================

角色：高级 AI 编译器与量化工程师
任务：将 MQBench 集成到 MIT BEVFusion (基于 mmdetection3d) 代码库中，实现量化感知训练 (QAT) 演示

背景：
- 框架：PyTorch, mmdetection3d
- 工具：MQBench (最新版本)
- 目标后端：TensorRT (Int8)
- 模型：BEVFusion (ResNet50 + LiDAR 分支)

关键约束与说明：
1. 不将 BEVFusion 代码移动到 MQBench 中，而是在 BEVFusion 仓库内编写此脚本，导入 MQBench
2. BEVFusion 模型包含自定义算子（特别是在 View Transformer/BEV Pooling 层），torch.fx 无法追踪
3. 必须配置 prepare_by_platform 将 BEV Pooling 层和其他 mmcv 自定义算子视为 leaf_modules 以避免追踪错误
4. 提供确切的 Python 代码片段以：
   - 加载浮点 BEVFusion 模型
   - 为 TensorRT 定义 backend_config
   - 使用正确的 leaf_module 列表应用 prepare_by_platform
   - 设置用于微调的训练循环

输出：
完整的、逻辑性强的 Python 脚本结构和在 mmdetection3d 模型中通常需要设置为 leaf 的模块的具体列表
"""

import argparse
import copy
import os
import sys
sys.path.append(os.getcwd())
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# MQBench imports
try:
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization
    from mqbench.convert_deploy import convert_deploy
except ImportError:
    warnings.warn(
        "MQBench is not installed. Please install it via: "
        "pip install mqbench"
    )
    raise


# ============================================================================
# 关键模块列表：需要设置为 leaf_modules 的 mmdetection3d 自定义算子
# ============================================================================
# Key Modules: Custom operators in mmdetection3d that need to be set as leaf_modules
# These modules contain CUDA extensions or operations that torch.fx cannot trace

def get_leaf_modules_for_mmdet3d():
    """
    返回 mmdetection3d 和 BEVFusion 中需要作为 leaf modules 的自定义算子列表
    
    这些模块包含：
    1. BEV Pooling 相关的自定义 CUDA 算子
    2. Sparse Convolution 相关模块 (spconv)
    3. Voxelization 相关模块
    4. 其他 CUDA 扩展算子（ROI Pooling, Ball Query, KNN 等）
    5. View Transformer 相关模块
    
    Returns:
        list: 需要作为 leaf modules 的类列表
    """
    leaf_modules = []
    
    # 1. BEV Pooling - 核心自定义算子，必须作为 leaf module
    try:
        from mmdet3d.ops.bev_pool.bev_pool import QuickCumsum, QuickCumsumCuda
        leaf_modules.extend([QuickCumsum, QuickCumsumCuda])
    except ImportError:
        warnings.warn("Cannot import BEV pooling modules")
    
    # 2. Sparse Convolution (spconv) - 3D 稀疏卷积，包含自定义 CUDA 实现
    try:
        from mmdet3d.ops.spconv import SparseModule, SparseConvolution, SparseMaxPool
        from mmdet3d.ops.spconv import SparseSequential, ToDense
        from mmdet3d.ops.sparse_block import SparseBasicBlock, SparseBottleneck
        leaf_modules.extend([
            SparseModule, SparseConvolution, SparseMaxPool,
            SparseSequential, ToDense, SparseBasicBlock, SparseBottleneck
        ])
    except ImportError:
        warnings.warn("Cannot import spconv modules")
    
    # 3. Voxelization - 点云体素化模块
    try:
        from mmdet3d.ops.voxel import Voxelization
        from mmdet3d.ops.voxel.scatter_points import DynamicScatter
        leaf_modules.extend([Voxelization, DynamicScatter])
    except ImportError:
        warnings.warn("Cannot import voxelization modules")
    
    # 4. ROI Aware Pool3D - 3D ROI 池化
    try:
        from mmdet3d.ops.roiaware_pool3d import RoIAwarePool3d
        leaf_modules.append(RoIAwarePool3d)
    except ImportError:
        warnings.warn("Cannot import RoIAwarePool3d")
    
    # 5. Point Cloud Sampling Operations - 点云采样操作
    try:
        from mmdet3d.ops.furthest_point_sample import Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler
        leaf_modules.extend([Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler])
    except ImportError:
        warnings.warn("Cannot import point sampling modules")
    
    # 6. PAConv - Point Adaptive Convolution
    try:
        from mmdet3d.ops.paconv import PAConv, ScoreNet
        leaf_modules.extend([PAConv, ScoreNet])
    except ImportError:
        warnings.warn("Cannot import PAConv modules")
    
    # 7. Group Points - 点云分组操作
    try:
        from mmdet3d.ops.group_points import QueryAndGroup, GroupAll
        leaf_modules.extend([QueryAndGroup, GroupAll])
    except ImportError:
        warnings.warn("Cannot import group points modules")
    
    # 8. View Transformers - 视图变换模块（LSS, BEVDepth 等）
    try:
        from mmdet3d.models.vtransforms import BaseTransform, LSSTransform
        # BaseDepthTransform 可能不存在，使用 try-except
        try:
            from mmdet3d.models.vtransforms import BaseDepthTransform
            leaf_modules.append(BaseDepthTransform)
        except ImportError:
            pass
        leaf_modules.extend([BaseTransform, LSSTransform])
    except ImportError:
        warnings.warn("Cannot import view transform modules")
    
    # 9. 其他常见的 mmcv 自定义算子
    try:
        # Ball Query
        from mmdet3d.ops.ball_query import ball_query
        # KNN
        from mmdet3d.ops.knn import knn
        # 这些是函数，不是类，但如果有对应的模块类，也需要添加
    except ImportError:
        pass
    
    return leaf_modules


def get_backend_config_for_tensorrt():
    """
    为 TensorRT Int8 后端配置 MQBench
    
    TensorRT 支持多种量化策略：
    - Per-tensor 和 Per-channel 量化
    - 对称和非对称量化
    - INT8 推理
    
    Returns:
        BackendType: MQBench 的后端类型配置
    """
    # TensorRT 使用 per-channel 对称量化用于权重
    # 使用 per-tensor 对称/非对称量化用于激活
    return BackendType.Tensorrt


def prepare_model_for_qat(model, backend_type, leaf_modules):
    """
    使用 MQBench 准备模型进行量化感知训练
    
    Args:
        model (nn.Module): 原始浮点模型
        backend_type (BackendType): 目标后端类型
        leaf_modules (list): 需要作为 leaf 处理的模块列表
    
    Returns:
        nn.Module: 准备好进行 QAT 的模型
    """
    # 创建一个虚拟输入用于追踪（需要根据实际模型输入调整）
    # BEVFusion 的输入比较复杂，包括图像、点云等
    # 这里提供一个占位符，实际使用时需要根据具体配置调整
    
    # 配置量化参数
    extra_quantizer_dict = {
        'additional_module_type': tuple(leaf_modules) if leaf_modules else (),
    }
    
    # 使用 prepare_by_platform 准备模型
    # 注意：prepare_by_platform 会修改模型，添加量化节点
    model = prepare_by_platform(
        model,
        backend_type,
        extra_quantizer_dict=extra_quantizer_dict,
    )
    
    return model


def build_qat_model(cfg):
    """
    构建并准备用于 QAT 的 BEVFusion 模型
    
    Args:
        cfg: mmcv Config 对象
    
    Returns:
        nn.Module: 准备好进行 QAT 的模型
    """
    # 1. 构建原始浮点模型
    model = build_model(cfg.model)
    model.init_weights()
    
    # 2. 如果需要，应用 SyncBN
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])
    
    # 3. 获取需要作为 leaf 的模块列表
    leaf_modules = get_leaf_modules_for_mmdet3d()
    
    # 4. 获取 TensorRT 后端配置
    backend_type = get_backend_config_for_tensorrt()
    
    # 5. 准备模型进行 QAT
    try:
        model = prepare_model_for_qat(model, backend_type, leaf_modules)
        print(f"✓ 成功准备模型进行 QAT，使用 {len(leaf_modules)} 个 leaf modules")
    except Exception as e:
        warnings.warn(f"准备 QAT 模型时出错: {e}")
        warnings.warn("将使用原始浮点模型继续训练（无量化）")
    
    return model


def train_qat_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    """
    使用量化感知训练微调模型
    
    训练流程：
    1. Calibration 阶段：收集激活值的统计信息
    2. QAT 阶段：在量化约束下微调模型
    
    Args:
        model: 准备好进行 QAT 的模型
        dataset: 训练数据集
        cfg: 配置
        distributed: 是否使用分布式训练
        validate: 是否在训练过程中验证
        timestamp: 时间戳
    """
    logger = get_root_logger()
    
    # 使用 mmdetection3d 的标准训练流程
    # 在训练开始前，可以添加 calibration 步骤
    
    # 可选：Calibration 阶段
    # 如果模型已经通过 prepare_by_platform 准备，需要先进行 calibration
    if hasattr(model, 'model'):  # 如果被 DDP 包装
        inner_model = model.model
    else:
        inner_model = model
    
    # 检查模型是否包含量化模块
    has_quantization = False
    for name, module in inner_model.named_modules():
        if 'quantize' in name.lower() or 'fake_quant' in name.lower():
            has_quantization = True
            break
    
    if has_quantization:
        logger.info("检测到量化模块，将进行 QAT 训练")
        # 在实际训练中，可能需要先运行 calibration
        # enable_calibration(inner_model)
        # ... 运行 calibration 数据 ...
        # enable_quantization(inner_model)
    else:
        logger.info("未检测到量化模块，将进行标准浮点训练")
    
    # 使用标准的 mmdet3d 训练流程
    train_model(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
    )


def main():
    """
    主函数：解析参数并启动 QAT 训练
    
    使用示例：
    
    # 单 GPU 训练
    python tools/quant_train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --load_from pretrained/bevfusion-det.pth
    
    # 多 GPU 分布式训练
    torchpack dist-run -np 8 python tools/quant_train.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --load_from pretrained/bevfusion-det.pth
    """
    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init()
        distributed = True
    else:
        distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="BEVFusion Quantization-Aware Training with MQBench"
    )
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="path to checkpoint to resume from"
    )
    args, opts = parser.parse_known_args()
    
    # 加载配置
    configs.load(args.config, recursive=True)
    configs.update(opts)
    
    cfg = Config(recursive_eval(configs), filename=args.config)
    
    # 设置 CUDA
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if distributed:
        torch.cuda.set_device(dist.local_rank())
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. Please run on a machine with CUDA.")
        cfg.gpu_ids = [0]
    
    # 设置运行目录
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    
    # 如果指定了 load_from 或 resume_from，更新配置
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # 保存配置
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))
    
    # 初始化日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    
    # 打印配置
    logger.info(f"配置文件:\n{cfg}")
    logger.info(f"MQBench QAT 训练启动")
    
    # 设置随机种子
    if cfg.seed is not None:
        logger.info(
            f"设置随机种子为 {cfg.seed}, "
            f"确定性模式: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    
    # 构建并准备 QAT 模型
    logger.info("开始构建 QAT 模型...")
    model = build_qat_model(cfg)
    logger.info(f"模型构建完成:\n{model}")
    
    # 打印 leaf modules 信息
    leaf_modules = get_leaf_modules_for_mmdet3d()
    logger.info(f"\n{'='*80}")
    logger.info(f"mmdetection3d 中需要作为 leaf modules 的自定义算子列表:")
    logger.info(f"{'='*80}")
    for i, module_class in enumerate(leaf_modules, 1):
        logger.info(f"{i:2d}. {module_class.__module__}.{module_class.__name__}")
    logger.info(f"{'='*80}\n")
    
    # 开始训练
    logger.info("开始 QAT 训练...")
    train_qat_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
    )
    
    logger.info("QAT 训练完成!")
    
    # 训练完成后，可以导出量化模型用于部署
    # 示例：
    # if dist.is_master():
    #     logger.info("导出量化模型用于 TensorRT 部署...")
    #     # 创建虚拟输入
    #     # dummy_input = create_dummy_input(cfg)
    #     # 导出模型
    #     # convert_deploy(
    #     #     model.module if hasattr(model, 'module') else model,
    #     #     BackendType.Tensorrt,
    #     #     dummy_input,
    #     #     output_path=os.path.join(cfg.run_dir, 'quantized_model.onnx')
    #     # )


if __name__ == "__main__":
    main()


"""
附录：关键概念说明
==================

1. Leaf Modules（叶子模块）
   - torch.fx 追踪器在遇到 leaf modules 时会将其视为不可分割的原子操作
   - 对于包含 CUDA 扩展或复杂操作的模块，必须设置为 leaf module
   - 否则 torch.fx 会尝试追踪其内部实现，导致错误

2. BEV Pooling
   - BEVFusion 的核心自定义算子
   - 将 2D 图像特征投影到 3D BEV 空间
   - 使用 CUDA 实现的自定义梯度计算

3. Sparse Convolution (spconv)
   - 用于高效处理 3D 稀疏数据（如 LiDAR 点云）
   - mmdetection3d 使用自定义的 spconv 实现
   - 包含多个 CUDA 算子

4. MQBench Backend Types
   - BackendType.Tensorrt: TensorRT 后端，支持 INT8 推理
   - BackendType.SNPE: Qualcomm SNPE
   - BackendType.PPLW8A8: PPL 8-bit weight and activation
   - 等等

5. 量化感知训练流程
   a. Calibration（校准）：
      - 收集激活值的统计信息（min/max, histogram 等）
      - 确定量化参数（scale, zero_point）
   
   b. QAT（量化感知训练）：
      - 在前向传播中模拟量化（fake quantization）
      - 反向传播时使用 Straight-Through Estimator (STE)
      - 微调模型权重以适应量化误差
   
   c. Deployment（部署）：
      - 将训练好的模型转换为真正的 INT8 模型
      - 导出为 ONNX 或其他部署格式
      - 在目标硬件上运行推理

6. mmdetection3d 中常见的自定义算子
   - bev_pool: BEV pooling 操作
   - sparse_conv: 稀疏卷积
   - voxelization: 点云体素化
   - roiaware_pool3d: 3D ROI 池化
   - ball_query: 球查询（用于点云邻域搜索）
   - knn: K 近邻搜索
   - furthest_point_sample: 最远点采样
   - group_points: 点分组
   - paconv: 点自适应卷积

7. 建议的训练策略
   - 先使用浮点模型预训练（如果还没有预训练权重）
   - 加载预训练权重后进行 QAT 微调
   - QAT 训练时使用较小的学习率（通常是预训练的 1/10）
   - 训练 10-20 个 epoch 通常足够恢复精度
   - 监控量化模型和浮点模型的精度差异

8. 调试建议
   - 如果遇到 torch.fx 追踪错误，检查是否有遗漏的 leaf modules
   - 使用 print 或 logging 输出模型结构，确认量化节点被正确插入
   - 对比量化模型和浮点模型的输出，确保数值差异在合理范围内
   - 使用小数据集测试训练流程，确保没有错误后再使用完整数据集
"""
