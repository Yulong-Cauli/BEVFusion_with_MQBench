#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion 量化模型 Benchmark 工具
===================================

功能：
  1. 报告模型大小（参数量、FP32 磁盘大小、估算 INT8 大小）
  2. 测量推理延迟（GPU warmup + 正式计时）
  3. 对比浮点模型与量化模型（如同时提供两个 checkpoint）

使用示例：
    # 报告 FP32 模型大小（不计时，不需要数据集）
    python tools/quant_benchmark.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --checkpoint pretrained/bevfusion-det.pth \\
        --size-only

    # 同时对比量化模型
    python tools/quant_benchmark.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --checkpoint pretrained/bevfusion-det.pth \\
        --quant-checkpoint runs/ptq_minmax/ptq_minmax_model.pth

    # 使用真实数据（需要数据集）
    python tools/quant_benchmark.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --checkpoint pretrained/bevfusion-det.pth \\
        --use-real-data --num-iters 50
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


# ============================================================================
# 模型大小统计
# ============================================================================

def count_parameters(model):
    """统计模型可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """统计模型全部参数总量（含 buffer）。"""
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return total, buffers


def get_model_size_mb(model):
    """
    计算模型 FP32 参数所占内存（MB）。
    每个 float32 参数占 4 字节。
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def get_disk_size_mb(path):
    """获取文件实际磁盘大小（MB）。"""
    if path and os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    return None


def print_model_size_report(model, checkpoint_path=None, label="FP32"):
    """打印模型大小报告。"""
    total_params, buffers = count_all_parameters(model)
    trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    estimated_int8_mb = size_mb / 4  # INT8 大约是 FP32 的 1/4

    print(f"\n{'='*60}")
    print(f"  模型大小报告 [{label}]")
    print(f"{'='*60}")
    print(f"  可训练参数量:  {trainable_params:,}  ({trainable_params / 1e6:.2f} M)")
    print(f"  全部参数量:    {total_params:,}  ({total_params / 1e6:.2f} M)")
    print(f"  Buffer 量:     {buffers:,}")
    print(f"  FP32 内存占用: {size_mb:.2f} MB")
    print(f"  估算 INT8 大小: {estimated_int8_mb:.2f} MB  (FP32 / 4，仅供参考)")

    if checkpoint_path:
        disk_mb = get_disk_size_mb(checkpoint_path)
        if disk_mb is not None:
            print(f"  Checkpoint 文件大小: {disk_mb:.2f} MB  ({checkpoint_path})")

    print(f"{'='*60}")


# ============================================================================
# 推理速度测量
# ============================================================================

def measure_inference_time(
    model,
    data_loader,
    num_warmup=10,
    num_iters=50,
    device="cuda",
    label="FP32",
):
    """
    测量模型推理延迟。

    Args:
        model: 待测模型（已移至 device）
        data_loader: 数据加载器（每 batch = 1 个样本）
        num_warmup (int): GPU 预热 iteration 数
        num_iters (int): 正式计时 iteration 数
        device (str): 计算设备
        label (str): 报告中显示的标签

    Returns:
        dict: 包含 mean/std/min/max 延迟（ms）
    """
    model.eval()
    latencies = []

    data_iter = iter(data_loader)

    def _get_next_data():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            return next(data_iter)

    print(f"\n[{label}] 开始推理速度测量 (warmup={num_warmup}, iters={num_iters}) ...")

    with torch.no_grad():
        # Warmup
        for i in range(num_warmup):
            data = _get_next_data()
            try:
                model(return_loss=False, rescale=True, **data)
                if device == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:
                warnings.warn(f"  Warmup {i} 出错: {e}")
                break

        # 正式计时
        for i in range(num_iters):
            data = _get_next_data()
            try:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(return_loss=False, rescale=True, **data)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # 转换为毫秒
            except Exception as e:
                warnings.warn(f"  计时 iter {i} 出错: {e}")
                break

    if not latencies:
        print(f"[{label}] 未能完成推理计时（请检查模型和数据）。")
        return {}

    latencies_arr = np.array(latencies)
    stats = {
        "mean_ms":   float(np.mean(latencies_arr)),
        "std_ms":    float(np.std(latencies_arr)),
        "min_ms":    float(np.min(latencies_arr)),
        "max_ms":    float(np.max(latencies_arr)),
        "p50_ms":    float(np.percentile(latencies_arr, 50)),
        "p95_ms":    float(np.percentile(latencies_arr, 95)),
        "p99_ms":    float(np.percentile(latencies_arr, 99)),
    }

    print(f"\n{'='*60}")
    print(f"  推理延迟报告 [{label}] (共 {len(latencies)} 次)")
    print(f"{'='*60}")
    print(f"  均值:   {stats['mean_ms']:.2f} ms")
    print(f"  标准差: {stats['std_ms']:.2f} ms")
    print(f"  最小值: {stats['min_ms']:.2f} ms")
    print(f"  最大值: {stats['max_ms']:.2f} ms")
    print(f"  P50:    {stats['p50_ms']:.2f} ms")
    print(f"  P95:    {stats['p95_ms']:.2f} ms")
    print(f"  P99:    {stats['p99_ms']:.2f} ms")
    print(f"  FPS:    {1000 / stats['mean_ms']:.2f}")
    print(f"{'='*60}")

    return stats


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BEVFusion 量化模型 Benchmark — 模型大小 & 推理速度"
    )
    parser.add_argument("config", metavar="FILE", help="配置文件路径")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="浮点模型 checkpoint 路径（.pth）",
    )
    parser.add_argument(
        "--quant-checkpoint",
        type=str,
        default=None,
        help="量化模型 checkpoint 路径（由 quant_ptq_minmax.py 生成）",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="使用真实验证集数据计时（需要配置好数据集路径；未指定时仍尝试加载验证集）",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="GPU 预热 iteration 数（默认 10）",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="正式计时 iteration 数（默认 50）",
    )
    parser.add_argument(
        "--size-only",
        action="store_true",
        help="只报告模型大小，跳过推理速度测量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda）",
    )
    args = parser.parse_args()
    return args


# ============================================================================
# 辅助：构建模型
# ============================================================================

def build_fp32_model(cfg, checkpoint_path=None, device="cuda"):
    """构建浮点模型并加载权重。"""
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    if checkpoint_path:
        print(f"  加载浮点权重: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model


def build_quant_model(cfg, quant_checkpoint_path, device="cuda"):
    """
    构建量化模型：先恢复模型结构，再重新应用 PTQ，最后加载量化权重。

    注意：量化模型的 state_dict 包含 FakeQuantize 参数（scale/zero_point），
    因此需要先用 prepare_by_platform 重建量化结构，再加载 state_dict。
    """
    try:
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_quantization
    except ImportError:
        print("  MQBench 未安装，跳过量化模型加载。")
        return None

    # 动态导入选择性量化函数
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from quant_ptq_minmax import apply_selective_ptq
    except ImportError:
        print("  无法导入 quant_ptq_minmax，跳过量化模型加载。")
        return None

    import logging
    logger = logging.getLogger("benchmark")

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.init_weights()

    # 重新应用选择性量化（重建 FakeQuantize 节点）
    model = apply_selective_ptq(model, BackendType.Tensorrt, logger)

    # 加载量化 checkpoint
    print(f"  加载量化权重: {quant_checkpoint_path}")
    ckpt = torch.load(quant_checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    # 切换到量化推理模式
    enable_quantization(model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 加载配置
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    print(f"\n配置文件: {args.config}")
    print(f"计算设备: {args.device}")

    # ----------------------------------------------------------------
    # 1. 浮点模型报告
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  构建浮点模型 ...")
    print("=" * 60)
    fp32_model = build_fp32_model(cfg, args.checkpoint, device=args.device)
    print_model_size_report(fp32_model, args.checkpoint, label="FP32")

    # ----------------------------------------------------------------
    # 2. 量化模型报告（可选）
    # ----------------------------------------------------------------
    quant_model = None
    if args.quant_checkpoint:
        print("\n" + "=" * 60)
        print("  构建量化模型 ...")
        print("=" * 60)
        quant_model = build_quant_model(cfg, args.quant_checkpoint, device=args.device)
        if quant_model is not None:
            print_model_size_report(quant_model, args.quant_checkpoint, label="PTQ-INT8(simulated)")

    # ----------------------------------------------------------------
    # 3. 推理速度测量
    # ----------------------------------------------------------------
    if not args.size_only:
        if args.use_real_data:
            print("\n使用真实验证集数据计时 ...")
            val_dataset = build_dataset(cfg.data.val)
            data_loader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False,
            )
        else:
            # 也使用验证集（BEVFusion 输入格式复杂，无法简单生成虚拟数据）
            print("\n使用验证集构建数据加载器 ...")
            print("（提示：如无数据集，请使用 --size-only 只查看模型大小）")
            try:
                val_dataset = build_dataset(cfg.data.val)
                data_loader = build_dataloader(
                    val_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False,
                )
            except Exception as e:
                print(f"  构建数据加载器失败: {e}")
                print("  跳过推理速度测量。若需测速，请确保数据集已配置，或使用 --size-only 。")
                data_loader = None

        if data_loader is not None:
            fp32_stats = measure_inference_time(
                fp32_model,
                data_loader,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
                device=args.device,
                label="FP32",
            )

            if quant_model is not None:
                quant_stats = measure_inference_time(
                    quant_model,
                    data_loader,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                    device=args.device,
                    label="PTQ-INT8(simulated)",
                )

                # 对比摘要
                if fp32_stats and quant_stats:
                    speedup = fp32_stats["mean_ms"] / quant_stats["mean_ms"]
                    fp32_size = get_model_size_mb(fp32_model)
                    quant_size = get_model_size_mb(quant_model)
                    print(f"\n{'='*60}")
                    print("  FP32 vs PTQ-INT8 对比摘要")
                    print(f"{'='*60}")
                    print(f"  FP32  均值延迟:   {fp32_stats['mean_ms']:.2f} ms")
                    print(f"  PTQ   均值延迟:   {quant_stats['mean_ms']:.2f} ms")
                    print(f"  加速比 (FP32/PTQ): {speedup:.2f}x")
                    print(f"  FP32  内存大小:   {fp32_size:.2f} MB")
                    print(f"  PTQ   内存大小:   {quant_size:.2f} MB  (FakeQuant参数略有增加)")
                    print(f"  估算 INT8 部署大小: {fp32_size / 4:.2f} MB")
                    print(f"{'='*60}")

    print("\nBenchmark 完成。")


if __name__ == "__main__":
    main()
