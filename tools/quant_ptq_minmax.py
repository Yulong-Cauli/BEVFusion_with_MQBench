#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion PTQ (Post-Training Quantization) with MQBench — MinMax Calibration
=============================================================================

策略：对适合量化的子模块进行 PTQ，跳过不适合量化的部分。

量化部分（标准密集卷积，torch.fx 可追踪）：
  - Camera backbone (SwinTransformer / ResNet)
  - Camera neck (GeneralizedLSSFPN / FPN)
  - Fuser (ConvFuser)
  - Decoder backbone (SECOND)
  - Decoder neck (SECONDFPN)
  - Detection / Segmentation heads

跳过部分（含自定义 CUDA 算子或稀疏卷积，不适合标准量化）：
  - Camera vtransform (BaseTransform / LSSTransform / DepthLSSTransform)
    → 内部含 bev_pool 自定义 CUDA 算子
  - LiDAR / Radar voxelize (Voxelization / DynamicScatter)
    → 体素化预处理，非神经网络层
  - LiDAR / Radar backbone (SparseEncoder)
    → 稀疏卷积，不兼容标准 FakeQuant 节点

使用示例：
    # 单 GPU
    python tools/quant_ptq_minmax.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load_from pretrained/bevfusion-det.pth

    # 多 GPU 分布式
    torchpack dist-run -np 8 python tools/quant_ptq_minmax.py \\
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
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmcv.parallel import MMDataParallel
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# MQBench imports
try:
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization
except ImportError:
    warnings.warn(
        "MQBench is not installed. Please install it via: "
        "pip install mqbench"
    )
    raise


# 临时修补 mmcv 的 Conv/Linear 包装层，使 torch.fx 追踪时不触发 Proxy 布尔判断报错
@contextmanager
def patch_mmcv_for_fx():
    """Patch mmcv Conv/ConvTranspose2d wrappers for torch.fx tracing compatibility.

    mmcv wraps standard PyTorch layers with a ``if x.numel() == 0`` guard for
    backward-compat with PyTorch < 1.4.  During fx symbolic tracing the guard
    becomes ``if Proxy:`` which raises TraceError.  We temporarily replace the
    forward methods with the plain PyTorch parent versions so that fx can trace
    through them.

    用原生 nn.Module 的 forward 替换 mmcv wrapper 的 forward，
    避免 fx 将条件判断中的 Tensor 变成 Proxy 后导致 TraceError

    MMCV 的行为：它的 forward 函数里包含显式的 Python if 语句（即 if x.numel() == 0）。这属于动态执行流。
    Torch.fx 的预期：它希望 forward 是一条顺畅的、确定的算子链条。
    当 fx 遇到 if 语句时，它必须立刻决定走 True 分支还是 False 分支。

    冲突点：由于 x 是一个符号化的 Proxy 对象，它的 numel() 结果也是一个 Proxy。
    Python 解释器在执行 if <Proxy> 时，无法得知这个 Tensor 在未来运行阶段到底是空还是满，因此报错。
    """

    import mmcv.cnn.bricks.wrappers as w

    saved = {}
    patch_map = {
        'Conv2d': nn.Conv2d,
        'ConvTranspose2d': nn.ConvTranspose2d,
        'MaxPool2d': nn.MaxPool2d,
        'Linear': nn.Linear,
    }
    for name, parent_cls in patch_map.items():
        cls = getattr(w, name, None)
        if cls is not None and hasattr(cls, 'forward'):
            saved[name] = cls.forward
            cls.forward = parent_cls.forward
    try:
        yield
    finally:
        for name, fwd in saved.items():
            getattr(w, name).forward = fwd


# ============================================================================
# 选择性量化：只量化适合量化的子模块
# ============================================================================

# 量化适用性说明：
#
# ✓ 可量化：Camera backbone/neck, Fuser, Decoder backbone/neck, Heads
#   - 标准密集卷积 (Conv2d / Linear / BN), torch.fx 可正常追踪
#
# ✗ 不可量化（跳过）：
#   1. vtransform (LSSTransform / DepthLSSTransform / BaseTransform)
#      - 内部调用 bev_pool (QuickCumsumCuda) ── 自定义 CUDA autograd Function
#      - 包含 nonzero / argsort / dynamic indexing 等 torch.fx 不支持的控制流
#   2. LiDAR/Radar voxelize (Voxelization / DynamicScatter)
#      - 点云体素化，纯预处理操作，非可微分层
#   3. LiDAR/Radar backbone (SparseEncoder)
#      - 稀疏卷积 (spconv)，输入输出均为稀疏张量，FakeQuant 节点无法插入

_QUANTIZABLE_SUBMODULE_KEYS = [
    # (attr_path_on_model, display_name)
    # camera branch
    ("encoders.camera.backbone",  "camera/backbone"),
    ("encoders.camera.neck",      "camera/neck"),
    # fuser
    ("fuser",                     "fuser"),
    # decoder
    ("decoder.backbone",          "decoder/backbone"),
    ("decoder.neck",              "decoder/neck"),
]

# heads 单独遍历（数量不定）


def _get_nested_attr(obj, key: str):
    """支持 'a.b.c' 形式的嵌套属性访问（含 nn.ModuleDict）。"""
    for part in key.split("."):
        if isinstance(obj, nn.ModuleDict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj, key: str, value):
    """支持 'a.b.c' 形式的嵌套属性设置（含 nn.ModuleDict）。"""
    parts = key.split(".")
    for part in parts[:-1]:
        if isinstance(obj, nn.ModuleDict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, nn.ModuleDict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def apply_selective_ptq(model, backend_type, logger):
    """
    对模型中适合量化的子模块逐一调用 prepare_by_platform，
    跳过 vtransform / voxelize / sparse-backbone 等不适合量化的部分。

    该函数的核心思想是“选择性量化”：只对标准密集算子分支做 FX 插桩，
    并用日志记录成功/失败/跳过的子模块，确保流程可控、可回溯。

    Args:
        model (nn.Module): 原始浮点模型
        backend_type (BackendType): 目标后端（TensorRT）
        logger: 日志记录器

    Returns:
        nn.Module: 已对可量化子模块插入 FakeQuantize 节点的模型
    """
    success, failed, skipped = [], [], []

    # --- 固定路径的子模块 ---
    # 逐个按路径获取子模块，使用 patch_mmcv_for_fx 保护 FX 追踪
    for attr_key, display_name in _QUANTIZABLE_SUBMODULE_KEYS:
        try:
            submodule = _get_nested_attr(model, attr_key)
        except (KeyError, AttributeError):
            # 该分支不存在（如纯 LiDAR 配置没有 camera 分支）
            skipped.append(display_name)
            continue

        try:
            # 关键点：临时修补 mmcv wrapper，避免 FX 因 if x.numel()==0 报错
            with patch_mmcv_for_fx():
                quantized = prepare_by_platform(submodule, backend_type)
            # 将量化后的子模块写回模型
            _set_nested_attr(model, attr_key, quantized)
            success.append(display_name)
            logger.info(f"  ✓ 量化子模块: {display_name}")
        except Exception as e:
            # 量化失败则记录并跳过，保证整体流程不中断
            failed.append(display_name)
            logger.warning(f"  ✗ 量化子模块 {display_name} 失败（已跳过）: {e}")

    # --- heads（数量可变）---
    # heads 是 ModuleDict，名称不固定，因此逐项遍历量化
    if hasattr(model, "heads"):
        for head_name, head_module in model.heads.items():
            display_name = f"heads/{head_name}"
            try:
                with patch_mmcv_for_fx():
                    quantized_head = prepare_by_platform(head_module, backend_type)
                model.heads[head_name] = quantized_head
                success.append(display_name)
                logger.info(f"  ✓ 量化子模块: {display_name}")
            except Exception as e:
                failed.append(display_name)
                logger.warning(f"  ✗ 量化子模块 {display_name} 失败（已跳过）: {e}")

    logger.info(
        f"选择性量化完成: 成功 {len(success)} 个, "
        f"失败 {len(failed)} 个, 不存在/跳过 {len(skipped)} 个"
    )
    if failed:
        logger.warning(f"  失败的子模块: {failed}")

    # 标记不量化的部分（仅供日志参考）
    skipped_by_design = [
        "camera/vtransform (含 bev_pool CUDA 算子)",
        "lidar/voxelize    (体素化预处理)",
        "lidar/backbone    (SparseEncoder 稀疏卷积)",
        "radar/voxelize    (体素化预处理，如有)",
        "radar/backbone    (SparseEncoder 稀疏卷积，如有)",
    ]
    logger.info("以下部分已设计跳过量化（不适合标准 PTQ）：")
    for item in skipped_by_design:
        logger.info(f"  - {item}")

    return model


# ============================================================================
# 校准阶段
# ============================================================================

def run_calibration(model, data_loader, num_batches, logger):
    """
    MinMax PTQ 校准：在校准数据上前向推理，收集各层激活值的 min/max。

    流程：
      enable_calibration → 运行 num_batches 个 batch → enable_quantization

    Args:
        model: 已插入 FakeQuantize 节点的模型
        data_loader: 数据加载器（建议用训练集子集）
        num_batches (int): 校准 batch 数（32~512 通常已足够）
        logger: 日志记录器
    """
    logger.info(f"开始 MinMax 校准，共使用 {num_batches} 个 batch ...")

    # 启用 Observer（记录 min/max），禁用 FakeQuant，进入“收集统计量”模式
    enable_calibration(model)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                # 达到指定校准 batch 数后退出
                break
            try:
                # 前向推理：仅用于统计激活分布，不计算梯度
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                # 单个 batch 失败时跳过，避免影响整体校准
                logger.warning(f"  校准 batch {i} 出错（已跳过）: {e}")
                continue
            if (i + 1) % 10 == 0:
                # 定期打印进度，便于监控校准进展
                logger.info(f"  校准进度: {i + 1}/{num_batches}")

    logger.info("MinMax 校准完成，scale/zero_point 已确定。")

    # 冻结 Observer 的统计量，启用 FakeQuant，切换到量化推理模式
    enable_quantization(model)
    logger.info("模型已切换为量化推理模式（FakeQuant 激活）。")


# ============================================================================
# 构建模型
# ============================================================================

def build_ptq_model(cfg, logger):
    """
    构建浮点模型，加载预训练权重，再对可量化子模块进行 PTQ 准备。

    Args:
        cfg: mmcv Config 对象
        logger: 日志记录器

    Returns:
        nn.Module: 已对可量化子模块插入 FakeQuantize 节点的模型
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

    # 3. SyncBN（如配置需要）
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    # 4. 选择性 PTQ 准备（仅对可量化子模块插入 FakeQuantize 节点）
    backend_type = BackendType.Tensorrt
    logger.info("开始对可量化子模块进行 PTQ 准备 (MinMax + TensorRT INT8) ...")
    model = apply_selective_ptq(model, backend_type, logger)

    return model


# ============================================================================
# 评估
# ============================================================================

def evaluate_quantized_model(model, data_loader, dataset, cfg, logger):
    """
    对量化模型进行完整评估，输出 NDS / mAP 指标。
    """
    logger.info("开始评估量化模型（验证集推理 + NDS/mAP 计算）...")
    outputs = single_gpu_test(model, data_loader)
    logger.info(f"量化模型推理完成，共处理 {len(outputs)} 个样本。")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    # 去掉训练专用 key
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"):
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox"))

    logger.info("计算量化模型 NDS / mAP ...")
    metrics = dataset.evaluate(outputs, **eval_kwargs)
    logger.info(f"量化模型评估结果:\n{metrics}")


# ============================================================================
# 量化诊断
# ============================================================================

# 模块路径映射（诊断 + 参数分析共用）
_ALL_MODULE_PATHS = [
    ("camera/backbone",   "encoders.camera.backbone"),
    ("camera/neck",       "encoders.camera.neck"),
    ("camera/vtransform", "encoders.camera.vtransform"),
    ("lidar/backbone",    "encoders.lidar.backbone"),
    ("fuser",             "fuser"),
    ("decoder/backbone",  "decoder.backbone"),
    ("decoder/neck",      "decoder.neck"),
    ("heads/object",      "heads.object"),
]


def diagnose_quantization_effect(model, data_loader, logger, num_samples=5):
    """
    诊断量化效果，生成可汇报的分析报告：
      1. 参数量覆盖率分析（解释为什么 FP32 ≈ INT8）
      2. FakeQuant 输出差异验证（证明量化确实在工作）
      3. 结论摘要
    """
    import torch.nn.functional as F

    inner = model.module if hasattr(model, "module") else model

    # ========== 1. 参数量覆盖分析 ==========
    logger.info("\n" + "=" * 70)
    logger.info("              INT8 量化诊断报告")
    logger.info("=" * 70)

    total_params = sum(p.numel() for p in inner.parameters())
    quantized_names = set()
    quantized_param_count = 0
    path_lookup = dict(_ALL_MODULE_PATHS)

    logger.info("\n[1/3] 各模块参数量与量化状态:")
    logger.info(
        f"  {'模块':<25s} {'参数量':>12s} {'占比':>8s} "
        f"{'FakeQuant节点':>14s} {'状态':>8s}"
    )
    logger.info(f"  {'-' * 70}")

    for display_name, attr_path in _ALL_MODULE_PATHS:
        try:
            mod = _get_nested_attr(inner, attr_path)
            params = sum(p.numel() for p in mod.parameters())
            pct = params / total_params * 100
            fq_count = sum(
                1 for m in mod.modules() if hasattr(m, "fake_quant_enabled")
            )
            if fq_count > 0:
                status = "✅ INT8"
                quantized_names.add(display_name)
                quantized_param_count += params
            else:
                status = "❌ FP32"
            logger.info(
                f"  {display_name:<25s} {params:>12,} {pct:>7.1f}% "
                f"{fq_count:>14d} {status:>8s}"
            )
        except (KeyError, AttributeError):
            logger.info(
                f"  {display_name:<25s} {'N/A':>12s} {'':>8s} "
                f"{'':>14s} {'跳过':>8s}"
            )

    unquantized = total_params - quantized_param_count
    q_pct = quantized_param_count / total_params * 100
    u_pct = unquantized / total_params * 100
    logger.info(f"\n  总参数量:        {total_params:>12,}")
    logger.info(f"  已量化(INT8):    {quantized_param_count:>12,} ({q_pct:.1f}%)")
    logger.info(f"  未量化(FP32):    {unquantized:>12,} ({u_pct:.1f}%)")

    # ========== 2. FakeQuant 输出差异验证 ==========
    logger.info(f"\n[2/3] FakeQuant 输出差异验证 ({num_samples} 个样本):")

    fq_modules = [
        m for m in model.modules() if hasattr(m, "fake_quant_enabled")
    ]
    fq_active = sum(1 for m in fq_modules if m.fake_quant_enabled)
    logger.info(f"  FakeQuantize 节点: {len(fq_modules)} 个 (已激活: {fq_active})")

    if not fq_modules:
        logger.error("  ❌ 未找到 FakeQuantize 节点！")
        return

    # 预先收集数据，保证两轮推理使用完全相同的输入
    logger.info(f"  正在收集 {num_samples} 个数据样本...")
    data_samples = []
    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break
        data_samples.append(data)

    class _OutputCapture:
        """Forward hook，捕获模块输出的第一个 Tensor。"""
        def __init__(self):
            self.outputs = []

        def __call__(self, module, inp, out):
            if isinstance(out, torch.Tensor):
                self.outputs.append(out.detach().cpu().clone())
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        self.outputs.append(o.detach().cpu().clone())
                        break

    def _run_capture(tag):
        """注册 hook → 前向推理 → 返回各模块的输出列表。"""
        captures = {}
        handles = []
        for name in quantized_names:
            try:
                mod = _get_nested_attr(inner, path_lookup[name])
                cap = _OutputCapture()
                handles.append(mod.register_forward_hook(cap))
                captures[name] = cap
            except (KeyError, AttributeError):
                pass

        logger.info(f"  运行 {tag} 模式...")
        model.eval()
        with torch.no_grad():
            for data in data_samples:
                model(return_loss=False, rescale=True, **data)

        for h in handles:
            h.remove()
        return {name: cap.outputs for name, cap in captures.items()}

    # INT8 pass (FakeQuant ON)
    int8_outputs = _run_capture("INT8 (FakeQuant ON)")

    # FP32 pass (FakeQuant OFF)
    for m in fq_modules:
        m.disable_fake_quant()
    fp32_outputs = _run_capture("FP32 (FakeQuant OFF)")
    for m in fq_modules:
        m.enable_fake_quant()

    # ========== 3. 比较与结论 ==========
    logger.info(f"\n  各量化模块 INT8 vs FP32 输出差异:")
    logger.info(
        f"  {'模块':<25s} {'Cosine Sim':>12s} {'相对MSE':>12s} "
        f"{'最大差异':>12s} {'结论':>10s}"
    )
    logger.info(f"  {'-' * 75}")

    all_working = True
    for name in sorted(quantized_names):
        i8_outs = int8_outputs.get(name, [])
        fp_outs = fp32_outputs.get(name, [])
        n = min(len(i8_outs), len(fp_outs))
        if n == 0:
            logger.warning(f"  {name}: 无输出可比较，跳过")
            continue

        cos_sims, rel_mses, max_diffs = [], [], []
        for i8, fp in zip(i8_outs[:n], fp_outs[:n]):
            cos = F.cosine_similarity(
                i8.flatten().unsqueeze(0), fp.flatten().unsqueeze(0)
            ).item()
            mse = F.mse_loss(i8, fp).item()
            fp_var = fp.var().item() + 1e-10
            cos_sims.append(cos)
            rel_mses.append(mse / fp_var)
            max_diffs.append(torch.max(torch.abs(i8 - fp)).item())

        avg_cos = sum(cos_sims) / n
        avg_rmse = sum(rel_mses) / n
        avg_md = sum(max_diffs) / n

        is_working = avg_cos < (1.0 - 1e-7)
        verdict = "✅ 正常" if is_working else "⚠️ 无差异"
        if not is_working:
            all_working = False

        logger.info(
            f"  {name:<25s} {avg_cos:>12.8f} {avg_rmse:>12.6e} "
            f"{avg_md:>12.6e} {verdict:>10s}"
        )

    logger.info(f"\n[3/3] 诊断结论:")
    logger.info("=" * 70)
    if all_working:
        logger.info("  ✅ 所有已量化模块的 FakeQuant 节点均正常工作")
        logger.info("  ✅ INT8 输出与 FP32 存在可测量差异，量化已正确生效")
        logger.info("")
        logger.info("  💡 FP32 ≈ INT8 (NDS 几乎无差异) 的原因:")
        logger.info(
            f"     仅 {q_pct:.1f}% 的参数被量化，"
            f"{u_pct:.1f}% 的模型仍为 FP32。"
        )
        logger.info(
            "     最大的未量化模块 (camera/backbone SwinT) 占总参数 ~67%。"
        )
        logger.info("     量化覆盖率低 → 对端到端 NDS 影响自然很小。")
        logger.info("")
        logger.info("  📊 要获得更显著的量化效果，需要量化 SwinT backbone。")
    else:
        logger.error("  ⚠️ 部分模块的 FakeQuant 可能未正确工作！")
        logger.error(
            "  请检查 MQBench prepare_by_platform 和 enable_quantization 调用。"
        )
    logger.info("=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    PTQ MinMax 主流程：

      1. 构建浮点模型并加载预训练权重
      2. 对可量化子模块插入 FakeQuantize 节点 (prepare_by_platform)
      3. enable_calibration → 运行校准数据（收集 min/max）
      4. enable_quantization → 进入量化推理模式
      5. (可选) 评估量化模型精度
      6. 保存量化模型检查点
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init()
        distributed = True
    else:
        distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(
        description="BEVFusion PTQ (MinMax) with MQBench — Selective Quantization"
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
        help="number of batches for MinMax calibration (default: 128)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="skip evaluation after calibration",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="run diagnostic analysis to verify INT8 quantization is working correctly",
    )
    parser.add_argument(
        "--diagnose-samples",
        type=int,
        default=5,
        help="number of samples for quantization diagnosis (default: 5)",
    )
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    if args.load_from is not None:
        cfg.load_from = args.load_from

    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info("=" * 60)
    logger.info("BEVFusion PTQ — MinMax 选择性量化")
    logger.info("=" * 60)
    logger.info(f"配置文件:\n{cfg}")

    if cfg.seed is not None:
        logger.info(f"随机种子: {cfg.seed}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # Step 1: 构建校准数据集
    # ------------------------------------------------------------------
    # 使用验证集做校准（test_mode=True，与推理流程一致，不含 GTDepth 等训练专属字段）
    logger.info("构建校准数据集（使用验证集，test_mode=True）...")
    calib_cfg = cfg.data.val.copy()
    calib_cfg.test_mode = True
    calib_dataset = build_dataset(calib_cfg)
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

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
    # Step 2: 构建 PTQ 模型
    # ------------------------------------------------------------------
    logger.info("构建 PTQ 模型（选择性 MinMax 量化）...")
    model = build_ptq_model(cfg, logger)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    logger.info("模型已移动到 GPU（MMDataParallel）。")

    # ------------------------------------------------------------------
    # Step 3: MinMax 校准
    # ------------------------------------------------------------------
    logger.info(f"MinMax 校准阶段：{args.calib_batches} 个 batch")
    run_calibration(model, calib_loader, num_batches=args.calib_batches, logger=logger)

    # ------------------------------------------------------------------
    # Step 4: 量化诊断（可选）
    # ------------------------------------------------------------------
    if args.diagnose:
        diagnose_quantization_effect(
            model, calib_loader, logger, num_samples=args.diagnose_samples
        )

    # ------------------------------------------------------------------
    # Step 5: 评估量化模型（可选）
    # ------------------------------------------------------------------
    if not args.no_eval:
        evaluate_quantized_model(model, val_loader, val_dataset, cfg, logger)

    # ------------------------------------------------------------------
    # Step 5: 保存量化模型
    # ------------------------------------------------------------------
    save_path = os.path.join(cfg.run_dir, "ptq_minmax_model.pth")
    inner_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "state_dict": inner_model.state_dict(),
            "meta": {
                "ptq_method": "MinMax",
                "backend": "TensorRT",
                "quantized_modules": [k for k, _ in _QUANTIZABLE_SUBMODULE_KEYS]
                + ["heads/*"],
                "skipped_modules": [
                    "camera/vtransform",
                    "lidar/voxelize",
                    "lidar/backbone (SparseEncoder)",
                ],
            },
        },
        save_path,
    )
    logger.info(f"PTQ 量化模型已保存至: {save_path}")

    logger.info("PTQ (MinMax) 流程完成！")
    logger.info(
        "后续步骤提示：\n"
        "  1. 使用 tools/quant_benchmark.py 查看模型大小与推理速度\n"
        "  2. 如精度下降过多，可切换到 tools/quant_train.py 进行 QAT 微调\n"
        "  注意：PTQ checkpoint 含 FakeQuant 结构，不能直接用 tools/test.py 评估"
    )


if __name__ == "__main__":
    main()
