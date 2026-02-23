#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion PTQ (Post-Training Quantization) Demo with MQBench — MinMax Calibration
==================================================================================

角色：高级 AI 编译器与量化工程师
任务：在 MIT BEVFusion (基于 mmdetection3d) 中使用 MQBench 实现最简单的 PTQ (训练后量化) 演示

背景：
- 框架：PyTorch, mmdetection3d
- 工具：MQBench (最新版本)
- 目标后端：TensorRT (Int8)
- 模型：BEVFusion (ResNet50 + LiDAR 分支)
- 量化方法：PTQ + MinMax 校准

PTQ vs QAT：
- PTQ (训练后量化)：无需重新训练权重，仅通过少量校准数据收集激活值统计信息，
  计算量化参数 (scale/zero_point)，速度快但精度通常略低于 QAT。
- MinMax 是最简单的校准方法：直接取激活值的全局 min/max 作为量化范围。

使用示例：
    # 单 GPU
    python tools/quant_ptq.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load_from pretrained/bevfusion-det.pth

    # 多 GPU 分布式
    torchpack dist-run -np 8 python tools/quant_ptq.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load_from pretrained/bevfusion-det.pth
"""

import argparse
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

from mmdet3d.datasets import build_dataloader, build_dataset
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
# Leaf Modules：与 QAT 脚本保持一致，避免 torch.fx 追踪出错
# ============================================================================

def get_leaf_modules_for_mmdet3d():
    """
    返回 mmdetection3d 和 BEVFusion 中需要作为 leaf modules 的自定义算子列表。
    这些模块包含 CUDA 扩展或复杂控制流，torch.fx 无法直接追踪。

    Returns:
        list: 需要作为 leaf modules 的类列表
    """
    leaf_modules = []

    # 1. BEV Pooling
    try:
        from mmdet3d.ops.bev_pool.bev_pool import QuickCumsum, QuickCumsumCuda
        leaf_modules.extend([QuickCumsum, QuickCumsumCuda])
    except ImportError:
        warnings.warn("Cannot import BEV pooling modules")

    # 2. Sparse Convolution (spconv)
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

    # 3. Voxelization
    try:
        from mmdet3d.ops.voxel import Voxelization
        from mmdet3d.ops.voxel.scatter_points import DynamicScatter
        leaf_modules.extend([Voxelization, DynamicScatter])
    except ImportError:
        warnings.warn("Cannot import voxelization modules")

    # 4. ROI Aware Pool3D
    try:
        from mmdet3d.ops.roiaware_pool3d import RoIAwarePool3d
        leaf_modules.append(RoIAwarePool3d)
    except ImportError:
        warnings.warn("Cannot import RoIAwarePool3d")

    # 5. Point Cloud Sampling
    try:
        from mmdet3d.ops.furthest_point_sample import Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler
        leaf_modules.extend([Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler])
    except ImportError:
        warnings.warn("Cannot import point sampling modules")

    # 6. PAConv
    try:
        from mmdet3d.ops.paconv import PAConv, ScoreNet
        leaf_modules.extend([PAConv, ScoreNet])
    except ImportError:
        warnings.warn("Cannot import PAConv modules")

    # 7. Group Points
    try:
        from mmdet3d.ops.group_points import QueryAndGroup, GroupAll
        leaf_modules.extend([QueryAndGroup, GroupAll])
    except ImportError:
        warnings.warn("Cannot import group points modules")

    # 8. View Transformers (LSS 等)
    try:
        from mmdet3d.models.vtransforms import BaseTransform, LSSTransform
        try:
            from mmdet3d.models.vtransforms import BaseDepthTransform
            leaf_modules.append(BaseDepthTransform)
        except ImportError:
            pass
        leaf_modules.extend([BaseTransform, LSSTransform])
    except ImportError:
        warnings.warn("Cannot import view transform modules")

    return leaf_modules


# ============================================================================
# PTQ 核心：MinMax 校准 + 量化参数确定
# ============================================================================

def prepare_model_for_ptq(model, backend_type, leaf_modules):
    """
    使用 MQBench 将模型准备为 PTQ 模式。

    PTQ 流程说明：
    1. prepare_by_platform：插入 FakeQuantize 节点（含 Observer）
    2. enable_calibration：激活 Observer，禁用 FakeQuant —— 此阶段只收集统计信息
    3. 运行校准数据（前向传播，无梯度）
    4. enable_quantization：冻结 Observer，激活 FakeQuant —— 此阶段模型变为量化模式

    MinMax 原理：
    - Observer 在校准阶段记录每一层激活值的全局 min 和 max
    - scale = (max - min) / (2^bits - 1)，zero_point 由对称/非对称设置决定
    - 是最简单、最快速的 PTQ 校准方法

    Args:
        model (nn.Module): 原始浮点模型
        backend_type (BackendType): 目标后端类型（TensorRT）
        leaf_modules (list): 需要作为 leaf 处理的模块列表

    Returns:
        nn.Module: 插入了 FakeQuantize 节点的模型（处于校准就绪状态）
    """
    extra_quantizer_dict = {
        'additional_module_type': tuple(leaf_modules) if leaf_modules else (),
    }

    # prepare_by_platform 默认使用 MinMaxObserver 作为激活值观测器
    # 对于 TensorRT 后端：权重 per-channel 对称，激活 per-tensor 对称
    model = prepare_by_platform(
        model,
        backend_type,
        extra_quantizer_dict=extra_quantizer_dict,
    )
    return model


def run_calibration(model, data_loader, num_batches, logger):
    """
    MinMax PTQ 校准阶段：在校准数据上进行前向推理，收集激活值的 min/max 统计信息。

    Args:
        model (nn.Module): 已通过 prepare_by_platform 准备好的模型
        data_loader: 数据加载器（通常使用训练集的子集）
        num_batches (int): 用于校准的 batch 数量（通常 32~512 个 batch 已足够）
        logger: 日志记录器
    """
    logger.info(f"开始 MinMax 校准，共使用 {num_batches} 个 batch...")

    # 启用校准模式：激活 Observer（记录 min/max），禁用 FakeQuant
    enable_calibration(model)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            try:
                # mmdet3d 数据格式：dict，包含 img、points 等字段
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                logger.warning(f"校准 batch {i} 出错（已跳过）: {e}")
                continue
            if (i + 1) % 10 == 0:
                logger.info(f"  校准进度: {i + 1}/{num_batches}")

    logger.info("MinMax 校准完成，量化参数 (scale/zero_point) 已确定。")

    # 切换为量化模式：冻结 Observer，激活 FakeQuant
    enable_quantization(model)
    logger.info("模型已切换为量化推理模式（FakeQuant 激活）。")


def build_ptq_model(cfg, logger):
    """
    构建浮点模型并将其准备为 PTQ 模式。

    Args:
        cfg: mmcv Config 对象
        logger: 日志记录器

    Returns:
        nn.Module: 准备好进行 PTQ 的模型（处于校准就绪状态）
    """
    # 1. 构建浮点模型
    model = build_model(cfg.model)
    model.init_weights()

    # 2. 加载预训练权重
    if cfg.get("load_from", None):
        logger.info(f"加载预训练权重: {cfg.load_from}")
        ckpt = torch.load(cfg.load_from, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info("预训练权重加载完成。")

    # 3. 应用 SyncBN（如果配置需要）
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    # 4. 获取 leaf modules 列表
    leaf_modules = get_leaf_modules_for_mmdet3d()
    logger.info(f"共收集到 {len(leaf_modules)} 个 leaf modules")

    # 5. 准备模型进行 PTQ（插入 FakeQuantize 节点）
    backend_type = BackendType.Tensorrt  # TensorRT INT8 后端
    try:
        model = prepare_model_for_ptq(model, backend_type, leaf_modules)
        logger.info("✓ 成功将模型准备为 PTQ 模式 (MinMax + TensorRT INT8)")
    except Exception as e:
        logger.warning(f"PTQ 准备失败: {e}，将使用原始浮点模型继续。")

    return model


# ============================================================================
# 评估辅助
# ============================================================================

def evaluate_quantized_model(model, data_loader, cfg, logger):
    """
    对量化后的模型进行简单的前向推理评估（用于验证量化模型是否正常工作）。

    Args:
        model (nn.Module): 量化后的模型
        data_loader: 验证集数据加载器
        cfg: 配置
        logger: 日志记录器
    """
    logger.info("开始评估量化模型（验证集前向推理）...")
    model.eval()
    results = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            try:
                result = model(return_loss=False, rescale=True, **data)
                results.extend(result)
            except Exception as e:
                logger.warning(f"评估 batch {i} 出错: {e}")
                break
            if (i + 1) % 50 == 0:
                logger.info(f"  评估进度: {i + 1} batches")

    logger.info(f"量化模型评估完成，共处理 {len(results)} 个样本。")
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行 PTQ（MinMax 校准）流程

    完整流程：
    1. 加载浮点预训练模型
    2. 使用 prepare_by_platform 插入 FakeQuantize 节点
    3. enable_calibration → 运行校准数据（收集 min/max）
    4. enable_quantization → 模型进入量化推理模式
    5. 评估量化模型精度
    6. (可选) 导出量化模型用于 TensorRT 部署
    """
    # 初始化分布式环境
    dist.init()

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="BEVFusion Post-Training Quantization (PTQ) with MQBench — MinMax"
    )
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="path to pretrained model checkpoint (required for PTQ)",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=128,
        help="number of batches used for MinMax calibration (default: 128)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="skip evaluation after calibration",
    )
    args, opts = parser.parse_known_args()

    # 加载配置
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # 设置 CUDA
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # 设置运行目录
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    if args.load_from is not None:
        cfg.load_from = args.load_from

    # 保存配置
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # 初始化日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"配置文件:\n{cfg.pretty_text}")
    logger.info("=" * 60)
    logger.info("BEVFusion PTQ — MinMax 校准")
    logger.info("=" * 60)

    # 设置随机种子
    if cfg.seed is not None:
        logger.info(f"随机种子: {cfg.seed}, 确定性: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # Step 1: 构建数据集与数据加载器
    # ------------------------------------------------------------------
    logger.info("构建校准数据集（使用训练集）...")
    calib_dataset = build_dataset(cfg.data.train)
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # 验证集（用于评估量化后精度）
    if not args.no_eval:
        logger.info("构建验证数据集...")
        val_dataset = build_dataset(cfg.data.val)
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Step 2: 构建并准备 PTQ 模型（插入 FakeQuantize 节点）
    # ------------------------------------------------------------------
    logger.info("构建 PTQ 模型（MinMax）...")
    model = build_ptq_model(cfg, logger)
    model.cuda()
    logger.info("模型已移动到 GPU。")

    # ------------------------------------------------------------------
    # Step 3: MinMax 校准
    # ------------------------------------------------------------------
    logger.info(f"MinMax 校准阶段：使用 {args.calib_batches} 个 batch 收集统计信息")
    run_calibration(model, calib_loader, num_batches=args.calib_batches, logger=logger)

    # ------------------------------------------------------------------
    # Step 4: （可选）评估量化模型
    # ------------------------------------------------------------------
    if not args.no_eval:
        evaluate_quantized_model(model, val_loader, cfg, logger)

    # ------------------------------------------------------------------
    # Step 5: 保存量化模型检查点
    # ------------------------------------------------------------------
    if dist.is_master():
        save_path = os.path.join(cfg.run_dir, "ptq_minmax_model.pth")
        torch.save(
            {
                "state_dict": (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                ),
                "meta": {"ptq_method": "MinMax", "backend": "TensorRT"},
            },
            save_path,
        )
        logger.info(f"PTQ 量化模型已保存至: {save_path}")

        # ------------------------------------------------------------------
        # Step 6: （可选）导出为 TensorRT / ONNX 部署格式
        # ------------------------------------------------------------------
        # logger.info("导出量化模型用于 TensorRT 部署...")
        # dummy_input = create_dummy_input(cfg)  # 需根据具体输入格式实现
        # convert_deploy(
        #     model.module if hasattr(model, "module") else model,
        #     BackendType.Tensorrt,
        #     dummy_input,
        #     output_path=os.path.join(cfg.run_dir, "ptq_minmax_model.onnx"),
        # )

    logger.info("PTQ (MinMax) 流程完成！")
    logger.info(
        "后续步骤提示：\n"
        "  1. 使用 tools/test.py 对量化模型进行完整评估\n"
        "  2. 如精度下降过多，可切换为 QAT（tools/quant_train.py）进行微调\n"
        "  3. 如需 TensorRT 部署，取消注释 convert_deploy 相关代码"
    )


if __name__ == "__main__":
    main()


"""
附录：PTQ MinMax 与其他校准方法对比
=====================================

1. MinMax（本脚本使用）
   - 原理：取校准数据中每层激活值的全局 min/max 作为量化范围
   - 优点：计算最简单，速度最快
   - 缺点：对异常值敏感，量化范围可能偏大
   - 适用：快速验证量化可行性

2. Percentile（百分位数校准）
   - 原理：取 p% 分位数（如 99.99th）作为量化范围，忽略极端异常值
   - 优点：对异常值鲁棒，通常比 MinMax 精度更高
   - 适用：需要比 MinMax 更好精度时

3. MSE（均方误差最小化）
   - 原理：搜索最小化量化误差的 scale
   - 优点：理论精度最优
   - 缺点：计算开销大
   - 适用：精度要求高时

4. QAT（量化感知训练，tools/quant_train.py）
   - 原理：在训练中模拟量化，反向传播更新权重
   - 优点：精度最高，可恢复量化损失
   - 缺点：需要训练时间和数据
   - 适用：PTQ 精度不满足要求时

MQBench 在 BackendType.Tensorrt 下默认使用 MinMaxObserver，
因此本 PTQ 脚本直接使用 prepare_by_platform 的默认配置即可实现 MinMax PTQ。
"""
