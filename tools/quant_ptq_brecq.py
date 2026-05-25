#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion BRECQ (Block Reconstruction PTQ) — DDP 4-GPU
========================================================

实验目的：
    与 quant_ptq_minmax.py（KL/Log2 calibration-only PTQ）做对照，验证
    在同一校准数据下 BRECQ 是否优于 KL+Log2 组合（A3 方案 NDS=0.6875）。

与 quant_ptq_minmax.py 的关系：
    复用： _get_nested_attr / _set_nested_attr / patch_mmcv_for_fx /
           _replace_feature / _QUANTIZABLE_SUBMODULE_KEYS
    新写： _BRECQConv2d / _BRECQLinear / _BRECQSparseConv 包装类
           apply_selective_brecq （平行于 apply_selective_ptq）
           cache_submodule_io / reconstruct_submodule（BRECQ 优化循环）
    替换： 权重 LearnableFakeQuantize → AdaRoundFakeQuantize
           激活 LearnableFakeQuantize → QDropFakeQuantize

BRECQ 粒度：
    fx 可追踪区     — prepare_by_platform 自动插桩 AdaRound+QDrop
    fx 不可追踪区   — 手动包装为 _BRECQConv2d/Linear/SparseConv
    一律按"子模块级 reconstruction"做（每子模块一次端到端 fp32-aligned 优化）。
    fx 不可追踪区无法做更细粒度的 block recon，子模块级是这套架构能做到的最细粒度。

强制要求：
    必须 4 卡 DDP 启动；脚本前端会跑 nvidia-smi 检查 GPU 占用，任一可见卡
    显存>5GB 或 util>10% 视为占用，直接 sys.exit(1) 拒绝运行（避免抢占）。

启动命令（标准）：
    torchpack dist-run -np 4 python tools/quant_ptq_brecq.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load-from pretrained/bevfusion-det.pth \\
        --calib-batches 256 \\
        --recon-iters 2000 \\
        --run-dir runs/brecq_full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.apis import multi_gpu_test
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval

# MQBench
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all
from mqbench.fake_quantize import (
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
)
from mqbench.observer import MinMaxObserver
from mqbench.scheme import QuantizeScheme

# spconv
from mmdet3d.ops.spconv.conv import SparseConvolution
from mmdet3d.ops.spconv.modules import SparseModule
from mmdet3d.ops.spconv.structure import SparseConvTensor

# 复用 quant_ptq_minmax 的非状态依赖工具
from tools.quant_ptq_minmax import (
    _get_nested_attr,
    _set_nested_attr,
    patch_mmcv_for_fx,
    _replace_feature,
    _QUANTIZABLE_SUBMODULE_KEYS,
)


# ============================================================================
# Pre-flight: GPU 占用检查（拒绝抢占别人的卡）
# ============================================================================

def check_gpus_idle(visible_indices: Optional[List[int]] = None,
                    min_free_mem_mib: int = 5000,
                    max_util_pct: int = 10) -> None:
    """启动前检查 nvidia-smi，任一可见 GPU 占用超阈值则退出。

    Args:
        visible_indices: 要检查的 GPU 索引列表。None 时检查 CUDA_VISIBLE_DEVICES
            指定的所有卡；CUDA_VISIBLE_DEVICES 未设置则检查全部物理卡。
        min_free_mem_mib: 已用显存 ≥ 此值视为占用（默认 5000 MiB）
        max_util_pct: GPU 利用率 ≥ 此值视为占用（默认 10%）

    本函数在所有 rank 上独立执行；任一 rank 检测到忙碌都会自杀，整体 DDP
    启动会因为 NCCL 握手失败而退出，无副作用。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[brecq][pre-flight] nvidia-smi 调用失败: {e}", flush=True)
        sys.exit(2)

    if visible_indices is None:
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env:
            visible_indices = [int(x) for x in env.split(",") if x.strip().isdigit()]

    busy = []
    parsed = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, mem_used, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        parsed.append((idx, mem_used, util))
        if visible_indices is not None and idx not in visible_indices:
            continue
        if mem_used >= min_free_mem_mib or util >= max_util_pct:
            busy.append((idx, mem_used, util))

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    if rank == 0:
        print("[brecq][pre-flight] GPU 状态:", flush=True)
        for idx, mem, util in parsed:
            mark = " ⚠ BUSY" if (idx, mem, util) in busy else ""
            print(f"  GPU{idx}: {mem:>5d} MiB used, {util:>3d}% util{mark}", flush=True)

    if busy:
        if rank == 0:
            print(
                f"[brecq][pre-flight] ✗ 检测到 {len(busy)} 张 GPU 被占用，"
                f"拒绝启动以避免抢占。\n"
                f"     阈值：mem<{min_free_mem_mib} MiB 且 util<{max_util_pct}%",
                flush=True,
            )
        sys.exit(1)


# ============================================================================
# DDP 初始化（兼容 torchpack/mpirun 与 torchrun 两种启动器）
# ============================================================================

def _init_distributed() -> bool:
    """与 tools/train.py 的 _init_distributed 保持一致。"""
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        dist.init()
        return True
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if not torch_dist.is_initialized():
            torch_dist.init_process_group(
                backend='nccl', init_method='env://',
                world_size=world_size, rank=rank,
            )
        import torchpack.distributed.context as _tp_ctx
        _tp_ctx._world_size = world_size
        _tp_ctx._world_rank = rank
        _tp_ctx._local_size = world_size
        _tp_ctx._local_rank = local_rank
        return True
    return False


def _world_size() -> int:
    return torch_dist.get_world_size() if torch_dist.is_initialized() else 1


def _is_main_rank() -> bool:
    return (not torch_dist.is_initialized()) or torch_dist.get_rank() == 0


# ============================================================================
# BRECQ FakeQuant 配置（AdaRound 权重 + QDrop 激活）
# ============================================================================

def _make_brecq_dense_pair() -> Tuple[nn.Module, nn.Module]:
    """Conv2d / Linear 用：AdaRound 权重（per-channel sym INT8）+ QDrop 激活。"""
    w_params = QuantizeScheme(
        symmetry=True, per_channel=True, pot_scale=False, bit=8
    ).to_observer_params()
    w_params['quant_min'] = -127
    a_params = QuantizeScheme(
        symmetry=True, per_channel=False, pot_scale=False, bit=8
    ).to_observer_params()
    a_params['quant_min'] = -127
    weight_fq = AdaRoundFakeQuantize(observer=MinMaxObserver, **w_params)
    act_fq = QDropFakeQuantize(observer=MinMaxObserver, **a_params)
    return weight_fq, act_fq


def _make_brecq_sparse_pair(act_per_channel: bool = False) -> Tuple[nn.Module, nn.Module]:
    """SparseConv 用：5D 权重 ch_axis=4。

    注意：AdaRoundFakeQuantize 的 init_alpha 内部会调用 weight.reshape，
    依赖 ch_axis 正确指向输出通道维度。spconv v1 权重 [K,K,K,Cin,Cout] → ch_axis=4。
    """
    w_params = QuantizeScheme(
        symmetry=True, per_channel=True, pot_scale=False, bit=8
    ).to_observer_params()
    w_params['quant_min'] = -127
    w_params['ch_axis'] = 4
    a_params = QuantizeScheme(
        symmetry=True, per_channel=act_per_channel, pot_scale=False, bit=8
    ).to_observer_params()
    a_params['quant_min'] = -127
    if act_per_channel:
        a_params['ch_axis'] = 1
    weight_fq = AdaRoundFakeQuantize(observer=MinMaxObserver, **w_params)
    act_fq = QDropFakeQuantize(observer=MinMaxObserver, **a_params)
    return weight_fq, act_fq


# ============================================================================
# BRECQ 包装类（fx 不可追踪区使用）
# ============================================================================

class _BRECQConv2d(nn.Module):
    """Conv2d + AdaRound 权重 + QDrop 激活。"""

    def __init__(self, original: nn.Conv2d):
        super().__init__()
        self.conv = original
        self.weight_fake_quant, self.act_fake_quant = _make_brecq_dense_pair()

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        # QDropFakeQuantize 的 torch.rand_like 不支持 FP16，先转 float
        if x.dtype == torch.float16:
            x = x.float()
        x = self.act_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        return F.conv2d(
            x, weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class _BRECQLinear(nn.Module):
    def __init__(self, original: nn.Linear):
        super().__init__()
        self.linear = original
        self.weight_fake_quant, self.act_fake_quant = _make_brecq_dense_pair()

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        x = self.act_fake_quant(x)
        weight = self.weight_fake_quant(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


class _BRECQSparseConv(SparseModule):
    """SparseConvolution + AdaRound 5D 权重 + QDrop 激活。

    注意：AdaRound 在 reconstruction 期间用 alpha 学习 round；snap 之后
    把 hard_value 写入 conv.weight.data，关闭 adaround flag。
    本类的 forward 与 _QuantizedSparseConv 类似（in-place copy_ 替代 data 指针重写）。
    """

    def __init__(self, original: SparseConvolution, act_per_channel: bool = False):
        super().__init__()
        self.conv = original
        self.weight_fake_quant, self.act_fake_quant = _make_brecq_sparse_pair(act_per_channel)
        self.register_buffer('_weight_dirty', torch.tensor(0, dtype=torch.uint8))

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        if self._weight_dirty.item():
            raise RuntimeError(
                "_BRECQSparseConv detected dirty weight state from a previous "
                "interrupted forward. Please reload the model checkpoint."
            )
        # QDropFakeQuantize 的 torch.rand_like 不支持 FP16
        if input.features.dtype == torch.float16:
            input = _replace_feature(input, input.features.float())
        if self.act_fake_quant is not None:
            quant_feats = self.act_fake_quant(input.features)
            input = _replace_feature(input, quant_feats)
        saved_weight = self.conv.weight.data.clone()
        self.conv.weight.data.copy_(self.weight_fake_quant(saved_weight))
        self._weight_dirty.fill_(1)
        try:
            output = self.conv(input)
        finally:
            self.conv.weight.data.copy_(saved_weight)
            self._weight_dirty.fill_(0)
        return output


def manual_brecq_dense(module: nn.Module, logger, module_name: str = "unknown") -> nn.Module:
    replacements = []
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, _BRECQConv2d):
            replacements.append((name, _BRECQConv2d(child)))
        elif isinstance(child, nn.Linear) and not isinstance(child, _BRECQLinear):
            replacements.append((name, _BRECQLinear(child)))
    for name, replacement in replacements:
        _set_nested_attr(module, name, replacement)
    if _is_main_rank():
        logger.info(
            f"  ↪ 手动 BRECQ {module_name}: 替换 {len(replacements)} 个 Conv2d/Linear"
        )
    return module


def manual_brecq_sparse(module: nn.Module, logger, module_name: str = "unknown",
                        act_per_channel: bool = False) -> nn.Module:
    replacements = []
    for name, child in list(module.named_modules()):
        if isinstance(child, SparseConvolution) and not isinstance(child, _BRECQSparseConv):
            replacements.append((name, _BRECQSparseConv(child, act_per_channel=act_per_channel)))
    for name, replacement in replacements:
        _set_nested_attr(module, name, replacement)
    if _is_main_rank():
        logger.info(
            f"  ↪ SparseConv BRECQ {module_name}: 替换 {len(replacements)} 个 SparseConv "
            f"({'per-channel' if act_per_channel else 'per-tensor'} 激活)"
        )
    return module


# ============================================================================
# 选择性 BRECQ：对全模型 8/8 子模块插入 BRECQ FakeQuant
# ============================================================================

def _brecq_extra_qconfig() -> dict:
    """让 prepare_by_platform 在 fx 区直接产出 AdaRound+QDrop FakeQuant。"""
    return {
        'extra_qconfig_dict': {
            'w_fakequantize': 'AdaRoundFakeQuantize',
            'a_fakequantize': 'QDropFakeQuantize',
        }
    }


def apply_selective_brecq(model, logger, skip_modules=None,
                          sparse_act_per_channel: bool = False):
    """对全模型可量化子模块插入 BRECQ FakeQuant 节点。

    流程与 apply_selective_ptq 对称：
      路径 1: torch.fx 自动插桩（extra_qconfig_dict 切到 BRECQ FakeQuant）
      路径 2: 手动 BRECQ 包装（Conv2d/Linear）
      路径 3: 手动 SparseConv BRECQ 包装
    """
    success, failed, skipped = [], [], []
    backend_type = BackendType.Tensorrt
    extra_cfg = _brecq_extra_qconfig()

    def _has_sparse_conv(module):
        return any(isinstance(m, SparseConvolution) for m in module.modules())

    def _try(submodule, display_name, attr_key=None, set_back=None):
        # 路径 1
        try:
            with patch_mmcv_for_fx():
                quantized = prepare_by_platform(submodule, backend_type, extra_cfg)
            if attr_key:
                _set_nested_attr(model, attr_key, quantized)
            elif set_back:
                set_back(quantized)
            success.append(display_name)
            if _is_main_rank():
                logger.info(f"  ✓ BRECQ 子模块: {display_name} (fx)")
            return
        except Exception as e:
            if _is_main_rank():
                logger.warning(
                    f"  ✗ {display_name} fx 追踪失败: {type(e).__name__}: {str(e)[:80]}"
                )

        # 路径 3：稀疏卷积
        if _has_sparse_conv(submodule):
            try:
                manual_brecq_sparse(submodule, logger, display_name,
                                    act_per_channel=sparse_act_per_channel)
                success.append(f"{display_name} (sparse)")
                return
            except Exception as e2:
                failed.append(display_name)
                if _is_main_rank():
                    logger.warning(f"  ✗ {display_name} sparse BRECQ 失败: {e2}")
                return

        # 路径 2：手动 Conv2d/Linear
        try:
            manual_brecq_dense(submodule, logger, display_name)
            success.append(f"{display_name} (manual)")
        except Exception as e2:
            failed.append(display_name)
            if _is_main_rank():
                logger.warning(f"  ✗ {display_name} 手动 BRECQ 失败: {e2}")

    if skip_modules is None:
        skip_modules = []

    for attr_key, display_name in _QUANTIZABLE_SUBMODULE_KEYS:
        if display_name in skip_modules:
            skipped.append(display_name)
            if _is_main_rank():
                logger.info(f"  ⊘ 跳过子模块: {display_name}")
            continue
        try:
            submodule = _get_nested_attr(model, attr_key)
        except (KeyError, AttributeError):
            skipped.append(display_name)
            continue
        _try(submodule, display_name, attr_key=attr_key)

    if hasattr(model, "heads"):
        for head_name, head_module in model.heads.items():
            display_name = f"heads/{head_name}"
            if display_name in skip_modules:
                skipped.append(display_name)
                continue
            if any(display_name in s for s in success):
                continue
            _try(head_module, display_name,
                 set_back=lambda q, hn=head_name: model.heads.__setitem__(hn, q))

    if _is_main_rank():
        logger.info(
            f"BRECQ 插桩完成: 成功 {len(success)}, 失败 {len(failed)}, 跳过 {len(skipped)}"
        )
    return model, success, failed


# ============================================================================
# 初始 MinMax 校准（让 AdaRound/QDrop 拿到有意义的 scale 起点）
# ============================================================================

def run_initial_calibration(model, calib_loader, num_batches: int, logger) -> None:
    """observer ON, fake_quant OFF —— 走完 num_batches 用 MinMax 收集 scale。"""
    if _is_main_rank():
        logger.info(f"初始 MinMax 校准：{num_batches} 个 batch ...")
    enable_calibration(model)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= num_batches:
                break
            try:
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                if _is_main_rank():
                    logger.warning(f"  校准 batch {i} 失败（已跳过）: {e}")
                continue
            if _is_main_rank() and (i + 1) % 20 == 0:
                logger.info(f"  校准进度: {i + 1}/{num_batches}")
    enable_quantization(model)
    if _is_main_rank():
        logger.info("初始校准完成；scale/zero_point 已就绪，准备进入 BRECQ 优化。")


# ============================================================================
# 子模块 I/O 缓存（FP32 模型 hook 输入输出）
# ============================================================================

class _IOCacheHook:
    """forward_hook 收集输入 args 与输出 tensor，配合 stop_forward 终止上层链路。"""

    def __init__(self, store_inp: bool = True, store_oup: bool = True,
                 stop_forward: bool = True, keep_gpu: bool = False):
        self.store_inp = store_inp
        self.store_oup = store_oup
        self.stop_forward = stop_forward
        self.keep_gpu = keep_gpu
        self.input_store: Optional[Tuple] = None
        self.output_store = None

    def __call__(self, module, inp, out):
        if self.store_inp:
            self.input_store = inp
        if self.store_oup:
            self.output_store = out
        if self.stop_forward:
            raise _StopForward()


class _StopForward(Exception):
    pass


def _detach_and_offload(obj, keep_gpu: bool):
    """递归把 (Tensor / SparseConvTensor / list / tuple / dict) 中的张量 detach 并迁到 CPU/GPU。
    ★ 关键：把 FP16 统一转 FP32，避免 QDropFakeQuantize 的 torch.rand_like 不支持 half。"""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.dtype == torch.float16:
            t = t.float()
        return t if keep_gpu else t.cpu()
    if isinstance(obj, SparseConvTensor):
        feats = obj.features.detach()
        if feats.dtype == torch.float16:
            feats = feats.float()
        feats = feats if keep_gpu else feats.cpu()
        idx = obj.indices.detach()
        idx = idx if keep_gpu else idx.cpu()
        new = SparseConvTensor(feats, idx, obj.spatial_shape, obj.batch_size)
        return new
    if isinstance(obj, (list, tuple)):
        out = [_detach_and_offload(x, keep_gpu) for x in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    if isinstance(obj, dict):
        return {k: _detach_and_offload(v, keep_gpu) for k, v in obj.items()}
    return obj


def _to_device(obj, device):
    """递归把张量迁到目标设备；同时把 FP16 统一转 FP32。"""
    if isinstance(obj, torch.Tensor):
        t = obj.to(device, non_blocking=True)
        if t.dtype == torch.float16:
            t = t.float()
        return t
    if isinstance(obj, SparseConvTensor):
        feats = obj.features.to(device, non_blocking=True)
        if feats.dtype == torch.float16:
            feats = feats.float()
        idx = obj.indices.to(device, non_blocking=True)
        return SparseConvTensor(feats, idx, obj.spatial_shape, obj.batch_size)
    if isinstance(obj, (list, tuple)):
        out = [_to_device(x, device) for x in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def cache_submodule_io(fp32_model, calib_loader, target_module, num_batches: int,
                       logger, keep_gpu: bool = False) -> List[Tuple]:
    """在 FP32 模型上 hook target_module，跑 num_batches 个 calib batch，
    返回 [(input_args_tuple, output_tensor_or_list), ...]。"""
    hook = _IOCacheHook(store_inp=True, store_oup=True, stop_forward=True, keep_gpu=keep_gpu)
    handle = target_module.register_forward_hook(hook)
    cache = []
    fp32_model.eval()
    try:
        with torch.no_grad():
            for i, data in enumerate(calib_loader):
                if i >= num_batches:
                    break
                hook.input_store = None
                hook.output_store = None
                try:
                    fp32_model(return_loss=False, rescale=True, **data)
                except _StopForward:
                    pass
                except Exception as e:
                    if _is_main_rank():
                        logger.warning(f"  cache batch {i} 异常（跳过）: {e}")
                    continue
                if hook.input_store is None or hook.output_store is None:
                    continue
                inp = _detach_and_offload(hook.input_store, keep_gpu)
                oup = _detach_and_offload(hook.output_store, keep_gpu)
                cache.append((inp, oup))
    finally:
        handle.remove()
    return cache


# ============================================================================
# AdaRound 工具：init alpha / 计算 round loss / snap to hard
# ============================================================================

def _all_adaround_quantizers(module: nn.Module) -> List[AdaRoundFakeQuantize]:
    return [m for m in module.modules() if isinstance(m, AdaRoundFakeQuantize)]


def _all_qdrop_quantizers(module: nn.Module) -> List[QDropFakeQuantize]:
    return [m for m in module.modules() if isinstance(m, QDropFakeQuantize)]


def _find_paired_weight_for_adaround(submodule: nn.Module,
                                     quantizer: AdaRoundFakeQuantize) -> Optional[torch.Tensor]:
    """AdaRound 需要在 init 时拿到对应的 weight tensor。三种来源：
       1. 父模块是 _BRECQConv2d/_BRECQLinear/_BRECQSparseConv → weight = self.conv/.linear.weight
       2. fx 区：父模块持有 weight 参数（ConvBnFusion 等），quantizer 是 .weight_fake_quant
    """
    for parent in submodule.modules():
        for child_name, child in parent.named_children():
            if child is not quantizer:
                continue
            if child_name == 'weight_fake_quant':
                if isinstance(parent, _BRECQConv2d):
                    return parent.conv.weight
                if isinstance(parent, _BRECQLinear):
                    return parent.linear.weight
                if isinstance(parent, _BRECQSparseConv):
                    return parent.conv.weight
                if hasattr(parent, 'weight') and isinstance(parent.weight, torch.Tensor):
                    return parent.weight
    return None


def init_adaround_alphas(submodule: nn.Module, logger) -> int:
    n = 0
    for q in _all_adaround_quantizers(submodule):
        w = _find_paired_weight_for_adaround(submodule, q)
        if w is None:
            continue
        q.init(w.data, round_mode='learned_hard_sigmoid')
        n += 1
    return n


def adaround_round_loss(quantizers: List[AdaRoundFakeQuantize], beta: float) -> torch.Tensor:
    """sum_i (1 - |2 * sigmoid_rect(alpha_i) - 1| ^ beta).sum()，BRECQ/AdaRound 标准公式。"""
    losses = []
    for q in quantizers:
        if not getattr(q, 'adaround', False) or not hasattr(q, 'alpha'):
            continue
        h = q.rectified_sigmoid()
        l = (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()
        losses.append(l)
    if not losses:
        return torch.tensor(0.0, requires_grad=False)
    return torch.stack(losses).sum()


def snap_adaround_alphas(submodule: nn.Module) -> int:
    """把 alpha 学到的软 round 固化为 hard round，写回 conv.weight，关闭 adaround 标志。"""
    n = 0
    for parent in submodule.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, AdaRoundFakeQuantize):
                continue
            if child_name != 'weight_fake_quant':
                continue
            if not getattr(child, 'adaround', False) or not hasattr(child, 'alpha'):
                continue
            if isinstance(parent, _BRECQConv2d):
                target_w = parent.conv.weight
            elif isinstance(parent, _BRECQLinear):
                target_w = parent.linear.weight
            elif isinstance(parent, _BRECQSparseConv):
                target_w = parent.conv.weight
            elif hasattr(parent, 'weight') and isinstance(parent.weight, torch.Tensor):
                target_w = parent.weight
            else:
                continue
            with torch.no_grad():
                hard = child.get_hard_value(target_w.data.clone())
                target_w.data.copy_(hard)
            child.adaround = False
            n += 1
    return n


# ============================================================================
# BRECQ reconstruction loop（DDP 同步梯度）
# ============================================================================

class _LinearTempDecay:
    def __init__(self, t_max, warm_up=0.2, start_b=20.0, end_b=2.0):
        self.t_max = t_max
        self.warm_up = warm_up
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.warm_up * self.t_max:
            return self.start_b
        rel = (t - self.warm_up * self.t_max) / (self.t_max * (1 - self.warm_up))
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel)


def _allreduce_grads(params: List[torch.nn.Parameter]) -> None:
    if not torch_dist.is_initialized() or torch_dist.get_world_size() == 1:
        return
    for p in params:
        if p.grad is not None:
            torch_dist.all_reduce(p.grad.data)


def reconstruct_submodule(submodule: nn.Module,
                          cache: List[Tuple],
                          iters: int,
                          w_lr: float,
                          a_lr: float,
                          drop_prob: float,
                          weight_round_loss: float,
                          warm_up: float,
                          logger,
                          tag: str = "") -> None:
    """BRECQ 子模块级 reconstruction：
       loss = lp_loss(quant_out, fp32_out) / world_size + lambda * round_loss
       backward → all_reduce(grad) → step。
    """
    device = next(submodule.parameters()).device
    adaround_qs = _all_adaround_quantizers(submodule)
    qdrop_qs = _all_qdrop_quantizers(submodule)
    if not adaround_qs and not qdrop_qs:
        if _is_main_rank():
            logger.info(f"  [{tag}] 无可学参数，跳过 reconstruction")
        return
    if not cache:
        if _is_main_rank():
            logger.warning(f"  [{tag}] 缓存为空，跳过 reconstruction")
        return

    # 准备参数
    n_init = init_adaround_alphas(submodule, logger)
    w_params = [q.alpha for q in adaround_qs if hasattr(q, 'alpha')]
    a_params = [q.scale for q in qdrop_qs]

    if not w_params and not a_params:
        if _is_main_rank():
            logger.info(f"  [{tag}] 学习参数为空（n_init={n_init}），跳过")
        return

    # QDrop on
    for q in qdrop_qs:
        q.prob = drop_prob

    w_opt = torch.optim.Adam(w_params) if w_params else None
    a_opt = torch.optim.Adam(a_params, lr=a_lr) if a_params else None
    a_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.0)
               if a_opt is not None else None)
    beta_decay = _LinearTempDecay(t_max=iters, warm_up=warm_up, start_b=20.0, end_b=2.0)
    world = _world_size()

    submodule.train(False)  # eval mode（不开 BN running stats 更新；包装层禁止 train）
    # 但 alpha/scale 是 Parameter，反向需要梯度
    rng = np.random.default_rng(seed=0 + (torch_dist.get_rank() if torch_dist.is_initialized() else 0))
    n_cache = len(cache)

    if _is_main_rank():
        logger.info(
            f"  [{tag}] 开始 reconstruction: iters={iters}, "
            f"adaround={len(w_params)}, qdrop={len(a_params)}, world={world}, cache={n_cache}"
        )

    for it in range(iters):
        idx = int(rng.integers(0, n_cache))
        inp, fp32_oup = cache[idx]
        inp_d = _to_device(inp, device)
        fp32_oup_d = _to_device(fp32_oup, device)

        if w_opt: w_opt.zero_grad()
        if a_opt: a_opt.zero_grad()

        # 子模块 forward：input 通常是 args tuple
        if isinstance(inp_d, tuple):
            quant_out = submodule(*inp_d)
        elif isinstance(inp_d, list):
            quant_out = submodule(*inp_d)
        else:
            quant_out = submodule(inp_d)

        # lp_loss：递归处理 list/tuple/SparseConvTensor.features
        rec_loss = _lp_loss_recursive(quant_out, fp32_oup_d, p=2.0)
        rec_loss = rec_loss / world

        beta = beta_decay(it)
        round_loss = adaround_round_loss(adaround_qs, beta=beta).to(rec_loss.device) if adaround_qs else torch.zeros_like(rec_loss)
        loss = rec_loss + weight_round_loss * round_loss
        loss.backward()

        _allreduce_grads(w_params)
        _allreduce_grads(a_params)

        if w_opt: w_opt.step()
        if a_opt: a_opt.step()
        if a_sched: a_sched.step()

        if _is_main_rank() and ((it + 1) % max(1, iters // 10) == 0 or it == iters - 1):
            logger.info(
                f"  [{tag}] it={it + 1:>5d}/{iters} "
                f"rec={float(rec_loss):.5f} round={float(round_loss):.3f} beta={beta:.2f}"
            )

    # snap：alpha → hard round 写回 weight；prob → 1.0 关闭 drop
    n_snap = snap_adaround_alphas(submodule)
    for q in qdrop_qs:
        q.prob = 1.0
    if _is_main_rank():
        logger.info(f"  [{tag}] reconstruction 完成；snap {n_snap} 个 AdaRound")


def _lp_loss_recursive(pred, tgt, p: float = 2.0) -> torch.Tensor:
    """对 (Tensor / SparseConvTensor / list / tuple) 递归累加 lp loss。"""
    if isinstance(pred, torch.Tensor) and isinstance(tgt, torch.Tensor):
        if pred.shape != tgt.shape:
            # 尺寸不一致时退化为 element-wise mean 兜底（某些 head 输出形状会浮动）
            n = min(pred.numel(), tgt.numel())
            pred_f = pred.reshape(-1)[:n]
            tgt_f = tgt.reshape(-1)[:n]
            return (pred_f - tgt_f).abs().pow(p).mean()
        return (pred - tgt).abs().pow(p).mean()
    if isinstance(pred, SparseConvTensor) and isinstance(tgt, SparseConvTensor):
        return _lp_loss_recursive(pred.features, tgt.features, p)
    if isinstance(pred, (list, tuple)) and isinstance(tgt, (list, tuple)) and len(pred) == len(tgt):
        losses = [_lp_loss_recursive(p_, t_, p) for p_, t_ in zip(pred, tgt)]
        return torch.stack([l for l in losses if torch.is_tensor(l)]).sum() if losses else torch.tensor(0.0, device=_first_device(pred))
    if isinstance(pred, dict) and isinstance(tgt, dict):
        keys = set(pred.keys()) & set(tgt.keys())
        losses = [_lp_loss_recursive(pred[k], tgt[k], p) for k in keys]
        return torch.stack([l for l in losses if torch.is_tensor(l)]).sum() if losses else torch.tensor(0.0, device=_first_device(pred))
    # 不可比较 → 0
    return torch.tensor(0.0, device=_first_device(pred), requires_grad=False)


def _first_device(obj):
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, SparseConvTensor):
        return obj.features.device
    if isinstance(obj, (list, tuple)):
        for x in obj:
            d = _first_device(x)
            if d is not None:
                return d
    if isinstance(obj, dict):
        for v in obj.values():
            d = _first_device(v)
            if d is not None:
                return d
    return torch.device('cpu')


# ============================================================================
# 评估（DDP multi_gpu_test 优先）
# ============================================================================

def evaluate_brecq_model(model, val_loader, val_dataset, cfg, logger, distributed: bool):
    if _is_main_rank():
        logger.info(f"开始评估（{'DDP multi_gpu_test' if distributed else '单卡 single_gpu_test'}）...")
    if distributed:
        outputs = multi_gpu_test(model, val_loader, tmpdir=None, gpu_collect=True)
    else:
        outputs = single_gpu_test(model, val_loader)

    if not _is_main_rank():
        return

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"):
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox"))
    metrics = val_dataset.evaluate(outputs, **eval_kwargs)
    logger.info(f"BRECQ 模型评估结果:\n{metrics}")
    for key in sorted(metrics.keys()):
        if 'nds' in key.lower() or 'map' in key.lower():
            print(f"{key}: {metrics[key]}", flush=True)


# ============================================================================
# 模型构建（与 quant_ptq_minmax.build_ptq_model 平行）
# ============================================================================

def build_brecq_model(cfg, logger, skip_modules=None,
                     sparse_act_per_channel: bool = False):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
        if _is_main_rank():
            logger.info("已启用 FP16 混合精度（与 test.py 对齐）。")

    if cfg.get("load_from", None):
        if _is_main_rank():
            logger.info(f"加载预训练权重: {cfg.load_from}")
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, cfg.load_from, map_location="cpu")

    return apply_selective_brecq(
        model, logger, skip_modules=skip_modules,
        sparse_act_per_channel=sparse_act_per_channel,
    )


# ============================================================================
# 主流程
# ============================================================================

# 子模块优化顺序（必须按依赖关系：上游 → 下游）
_RECON_ORDER = [
    ("encoders.camera.backbone", "camera/backbone"),
    ("encoders.camera.neck", "camera/neck"),
    ("encoders.camera.vtransform", "camera/vtransform"),
    ("encoders.lidar.backbone", "lidar/backbone"),
    ("fuser", "fuser"),
    ("decoder.backbone", "decoder/backbone"),
    ("decoder.neck", "decoder/neck"),
]


def main():
    # 1. CLI（先解析，--help 不应被 GPU 检查拦截）
    parser = argparse.ArgumentParser(
        description="BEVFusion BRECQ PTQ — DDP 4-GPU"
    )
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--load-from", type=str, default=None,
                        help="path to pretrained model checkpoint")
    parser.add_argument("--calib-batches", type=int, default=256,
                        help="calibration batch count (init MinMax + recon cache)")
    parser.add_argument("--recon-iters", type=int, default=2000,
                        help="BRECQ reconstruction iterations per submodule")
    parser.add_argument("--cache-batches", type=int, default=64,
                        help="how many batches to cache as fp32 input/output per submodule")
    parser.add_argument("--w-lr", type=float, default=4e-4,
                        help="learning rate for AdaRound alpha (Adam default if 0)")
    parser.add_argument("--a-lr", type=float, default=4e-5,
                        help="learning rate for QDrop scale (Adam)")
    parser.add_argument("--drop-prob", type=float, default=0.5,
                        help="QDrop activation drop probability during reconstruction")
    parser.add_argument("--round-loss-weight", type=float, default=0.01,
                        help="lambda for adaround round_loss term")
    parser.add_argument("--warm-up", type=float, default=0.2,
                        help="beta linear-decay warmup ratio")
    parser.add_argument("--sparse-act-mode", type=str, default="per_tensor",
                        choices=["per_tensor", "per_channel"])
    parser.add_argument("--skip-modules", type=str, nargs="+", default=[])
    parser.add_argument("--no-eval", action="store_true",
                        help="skip post-BRECQ evaluation")
    parser.add_argument("--keep-cache-gpu", action="store_true",
                        help="keep IO cache on GPU (faster, uses VRAM); default offload to CPU")
    parser.add_argument("--smoke", action="store_true",
                        help="dry-run smoke test: skip data loading & training; verify wiring only")
    parser.add_argument("--bypass-gpu-check", action="store_true",
                        help="(测试用) 跳过 nvidia-smi 占用检查；正式跑禁用")
    args, opts = parser.parse_known_args()

    # 2. GPU pre-flight check（在 argparse 后、DDP init 前执行；--smoke/--bypass-gpu-check 跳过）
    if args.smoke:
        # smoke 模式不接触 GPU，仅做 wiring 验证
        pass
    elif args.bypass_gpu_check:
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0"))) == 0:
            print("[brecq] ⚠ --bypass-gpu-check 启用，跳过 GPU 占用检查（仅供测试）。", flush=True)
    else:
        check_gpus_idle()

    # 3. DDP init
    distributed = _init_distributed()
    if distributed:
        torch.cuda.set_device(dist.local_rank())
    elif torch.cuda.is_available() and not args.smoke:
        torch.cuda.set_device(0)

    # 重新跑一次 GPU 检查（如果用户 --bypass-gpu-check 则跳过）
    if args.bypass_gpu_check and _is_main_rank():
        pass  # 已在上面打印

    # 4. 配置加载
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

    if _is_main_rank():
        cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}_rank{dist.rank() if distributed else 0}.log")
    logger = get_root_logger(log_file=log_file)

    if _is_main_rank():
        logger.info("=" * 60)
        logger.info("BEVFusion BRECQ — DDP 4-GPU")
        logger.info("=" * 60)
        logger.info(f"world_size={_world_size()}, rank={dist.rank() if distributed else 0}")

    # 5. smoke 模式：仅做 wiring 验证，不加载数据/不跑训练
    if args.smoke:
        if _is_main_rank():
            logger.info("→ SMOKE 模式：仅验证 import + 模型 BRECQ 插桩，不跑 calib/recon/eval")
        cfg.load_from = None  # 不加载预训练；smoke 只看代码路径
        model, success, failed = build_brecq_model(cfg, logger,
                                                   skip_modules=args.skip_modules)
        if _is_main_rank():
            logger.info(f"smoke 通过：成功 {len(success)}, 失败 {len(failed)}")
            logger.info("smoke 完成。")
        return

    # 6. 校准与验证 dataloader
    calib_cfg = cfg.data.train.copy()
    calib_cfg.test_mode = True
    calib_dataset = build_dataset(calib_cfg)
    if not hasattr(calib_dataset, 'flag'):
        calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.uint8)
    calib_loader = build_dataloader(
        calib_dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed, shuffle=True, seed=0,
    )

    val_dataset = None
    val_loader = None
    if not args.no_eval:
        val_dataset = build_dataset(cfg.data.test)
        val_loader = build_dataloader(
            val_dataset, samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed, shuffle=False,
        )

    # 7. 构建 BRECQ 模型 + DDP 包装
    model, success, failed = build_brecq_model(
        cfg, logger, skip_modules=args.skip_modules,
        sparse_act_per_channel=(args.sparse_act_mode == "per_channel"),
    )
    if _is_main_rank():
        logger.info(f"BRECQ 插桩: 成功 {len(success)}, 失败 {len(failed)}")

    model.cuda()
    if distributed:
        model = MMDistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False, find_unused_parameters=True,
        )
    else:
        model = MMDataParallel(model, device_ids=[0])

    # 8. 初始 MinMax 校准（observer 阶段）
    run_initial_calibration(model, calib_loader, args.calib_batches, logger)

    # 9. 子模块级 BRECQ reconstruction
    inner = model.module if hasattr(model, 'module') else model
    if _is_main_rank():
        logger.info("=" * 60)
        logger.info("BRECQ reconstruction 阶段")
        logger.info("=" * 60)

    for attr_path, display in _RECON_ORDER:
        if display in args.skip_modules:
            continue
        try:
            sub = _get_nested_attr(inner, attr_path)
        except (KeyError, AttributeError):
            if _is_main_rank():
                logger.info(f"  ⊘ {display} 不存在，跳过")
            continue

        # 缓存 (input, fp32_output)：用整个 model（含 quant 模块）跑 fp32 路径需要先 disable_all
        # 这里采用 trick：临时 disable_all → forward 收集 → 恢复 enable_quantization
        disable_all(inner)
        if _is_main_rank():
            logger.info(f"[{display}] 缓存 FP32 I/O ({args.cache_batches} batch) ...")
        cache = cache_submodule_io(
            model, calib_loader, sub, args.cache_batches, logger,
            keep_gpu=args.keep_cache_gpu,
        )
        enable_quantization(inner)

        if _is_main_rank():
            logger.info(f"[{display}] 缓存完成: {len(cache)} 条样本")

        if not cache:
            if _is_main_rank():
                logger.warning(f"[{display}] 缓存为空，跳过 reconstruction")
            continue

        reconstruct_submodule(
            sub, cache,
            iters=args.recon_iters,
            w_lr=args.w_lr, a_lr=args.a_lr,
            drop_prob=args.drop_prob,
            weight_round_loss=args.round_loss_weight,
            warm_up=args.warm_up,
            logger=logger, tag=display,
        )
        # 释放 cache
        del cache
        torch.cuda.empty_cache()

    # 10. heads（与 fx 区结构平行，单独处理）
    if hasattr(inner, 'heads'):
        for head_name, head_mod in inner.heads.items():
            display = f"heads/{head_name}"
            if display in args.skip_modules:
                continue
            disable_all(inner)
            if _is_main_rank():
                logger.info(f"[{display}] 缓存 FP32 I/O ...")
            cache = cache_submodule_io(
                model, calib_loader, head_mod, args.cache_batches, logger,
                keep_gpu=args.keep_cache_gpu,
            )
            enable_quantization(inner)
            if not cache:
                continue
            reconstruct_submodule(
                head_mod, cache,
                iters=args.recon_iters,
                w_lr=args.w_lr, a_lr=args.a_lr,
                drop_prob=args.drop_prob,
                weight_round_loss=args.round_loss_weight,
                warm_up=args.warm_up,
                logger=logger, tag=display,
            )
            del cache
            torch.cuda.empty_cache()

    # 11. 评估
    if not args.no_eval:
        evaluate_brecq_model(model, val_loader, val_dataset, cfg, logger, distributed)

    # 12. 保存（rank 0）
    if _is_main_rank():
        save_path = os.path.join(cfg.run_dir, "ptq_brecq_model.pth")
        meta = {
            "ptq_method": "BRECQ (AdaRound + QDrop, submodule-level reconstruction)",
            "backend": "TensorRT",
            "calib_batches": args.calib_batches,
            "cache_batches": args.cache_batches,
            "recon_iters": args.recon_iters,
            "w_lr": args.w_lr,
            "a_lr": args.a_lr,
            "drop_prob": args.drop_prob,
            "round_loss_weight": args.round_loss_weight,
            "warm_up": args.warm_up,
            "sparse_act_mode": args.sparse_act_mode,
            "world_size": _world_size(),
        }
        torch.save({"state_dict": inner.state_dict(), "meta": meta}, save_path)
        logger.info(f"BRECQ 模型已保存至: {save_path}")
        logger.info("BRECQ 流程完成。")


if __name__ == "__main__":
    main()
