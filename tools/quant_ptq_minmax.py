#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEVFusion PTQ (Post-Training Quantization) with MQBench — MinMax Calibration
=============================================================================

策略：对全模型 8/8 子模块进行 PTQ，采用三种量化路径覆盖所有模块：

量化路径一（torch.fx 自动插桩）：
  - Camera neck (GeneralizedLSSFPN / FPN)
  - Fuser (ConvFuser)
  - Decoder backbone (SECOND)
  - Decoder neck (SECONDFPN)

量化路径二（手动 FakeQuant 包装 Conv2d/Linear）：
  - Camera backbone (SwinTransformer) — fx 失败于 AdaptivePadding 动态控制流
  - Camera vtransform (DepthLSSTransform) — fx 失败于 bev_pool CUDA 算子
  - Detection / Segmentation heads — fx 失败于 Proxy 迭代

量化路径三（手动 FakeQuant 包装 SparseConvolution）：
  - LiDAR backbone (SparseEncoder) — 稀疏卷积 features 量化 + weight 临时替换

跳过部分（非神经网络层）：
  - LiDAR / Radar voxelize (Voxelization / DynamicScatter) — 体素化预处理

使用示例：
    # 单 GPU
    python tools/quant_ptq_minmax.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load-from pretrained/bevfusion-det.pth

    # 多 GPU 分布式
    torchpack dist-run -np 8 python tools/quant_ptq_minmax.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        --load-from pretrained/bevfusion-det.pth
"""

import argparse
import math
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
import torch.nn.functional as F
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
    from mqbench.fake_quantize import LearnableFakeQuantize
    from mqbench.observer import MinMaxObserver, EMAMinMaxObserver, MSEObserver, EMAQuantileObserver
    from mqbench.scheme import QuantizeScheme
except ImportError:
    warnings.warn(
        "MQBench is not installed. Please install it via: "
        "pip install mqbench"
    )
    raise

# spconv imports（用于稀疏卷积量化）
from mmdet3d.ops.spconv.conv import SparseConvolution
from mmdet3d.ops.spconv.modules import SparseModule
from mmdet3d.ops.spconv.structure import SparseConvTensor

from mqbench.observer import ObserverBase


# ============================================================================
# KL 散度 Observer：基于直方图的最优截断阈值搜索
# ============================================================================

class KLDivergenceObserver(ObserverBase):
    """基于 KL 散度的激活校准 Observer（类似 TensorRT Entropy Calibrator）。

    与 EMAMinMaxObserver 不同，本 Observer 不直接使用 [min, max] 作为量化范围，
    而是在校准完成后通过 KL 散度搜索最优截断阈值 T_opt，丢弃导致 INT8 范围浪费的离群值。

    算法：
    1. 校准阶段：累积所有校准 batch 的激活值直方图（高分辨率，默认 2048 bins）
    2. 校准结束后（calculate_qparams 被调用时）：
       - 遍历候选截断点 T（从 128 bins 到 num_bins）
       - 对截断后的分布 P 量化为 num_quantized_bins 个 bin，得到 Q
       - 计算 KL(P || Q)，选择使 KL 最小的 T 作为 T_opt
    3. 用 T_opt 设置 min_val/max_val（对称：[-T_opt, T_opt]）

    参数：
        num_bins: 直方图分辨率（默认 2048，越高越精确但越慢）
        num_quantized_bins: INT8 量化 bin 数（默认 255，对应 quant_min~quant_max）
        ema_ratio: EMA 平滑系数（0 = 不平滑，纯累积直方图）
        sparse_mode: 稀疏感知模式（Round 8 新增，默认 False）。
            True 时构建直方图只统计 |x| > 1e-6 的非零元素，消除 ReLU 零值在
            bin[0] 产生的巨大尖峰对 KL 搜索的偏差。用于稀疏卷积激活量化。
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None,
                 ch_axis=-1, pot_scale=False,
                 num_bins=2048, ema_ratio=0.0, min_percentile=0.9999,
                 sparse_mode=False,
                 factory_kwargs=None):
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         ch_axis, pot_scale, factory_kwargs)
        self.num_bins = num_bins
        self.ema_ratio = ema_ratio
        self.min_percentile = min_percentile
        self.sparse_mode = sparse_mode  # ★ Round 8: Sparse-Aware Calibration 开关
        # 对称量化时，|x| 的量化级数 = quant_max + 1（例如 [-127,127] → 128 个正级）
        # 这是 KL 搜索时的 "目标 bin 数"
        if self.quant_min < 0:
            self.num_quantized_bins = self.quant_max + 1  # 128 for [-127, 127]
        else:
            self.num_quantized_bins = self.quant_max - self.quant_min + 1
        # per-tensor: [num_bins]
        # per-channel: [C, num_bins]（C 在第一次 forward 动态确定）
        self.register_buffer("histogram", torch.zeros(num_bins))
        # per-tensor: scalar
        # per-channel: [C]
        self.register_buffer("hist_max", torch.tensor(float(0)))
        self.register_buffer("_calibrated", torch.tensor(0, dtype=torch.long))

    def _ensure_per_channel_buffers(self, num_channels: int, device, dtype):
        """按通道模式初始化/对齐 histogram 与 hist_max。"""
        if self.histogram.ndim != 2 or self.histogram.size(0) != num_channels:
            self.histogram = torch.zeros(
                (num_channels, self.num_bins), device=device, dtype=dtype
            )
            self.hist_max = torch.zeros(num_channels, device=device, dtype=dtype)

    def _update_histogram_1d(self, abs_x: torch.Tensor, old_hist: torch.Tensor,
                             old_max: torch.Tensor, cur_max: torch.Tensor):
        """更新单通道（或 per-tensor）直方图。"""
        eps = 1e-8
        cur_max_val = max(float(cur_max.item()), eps)
        old_max_val = float(old_max.item())

        if old_max_val <= 0:
            new_hist = torch.histc(abs_x, bins=self.num_bins, min=0, max=cur_max_val)
            return new_hist, abs_x.new_tensor(cur_max_val)

        # 需要扩展直方图范围：先将旧直方图重映射到新范围，再与新 batch 合并
        if cur_max_val > old_max_val:
            new_hist = torch.histc(abs_x, bins=self.num_bins, min=0, max=cur_max_val)
            ratio = old_max_val / cur_max_val
            rescaled = torch.zeros_like(old_hist)
            if ratio > 0:
                for i in range(self.num_bins):
                    target_bin = int(i * ratio)
                    if target_bin < self.num_bins:
                        rescaled[target_bin] += old_hist[i]
            if self.ema_ratio > 0:
                merged = rescaled * self.ema_ratio + new_hist * (1.0 - self.ema_ratio)
            else:
                merged = rescaled + new_hist
            return merged, abs_x.new_tensor(cur_max_val)

        # 范围不变，直接累加
        new_hist = torch.histc(abs_x, bins=self.num_bins, min=0, max=old_max_val)
        if self.ema_ratio > 0:
            merged = old_hist * self.ema_ratio + new_hist * (1.0 - self.ema_ratio)
        else:
            merged = old_hist + new_hist
        return merged, old_max

    def forward(self, x_orig):
        """累积激活值 |x| 直方图。支持 per-tensor / per-channel。

        ★ Round 8 Sparse-Aware Calibration：当 sparse_mode=True 时，
          只统计 |x| > 1e-6 的非零元素，消除 ReLU 零值在 bin[0] 的巨大尖峰。
          对称量化下零值始终被精确映射到 INT8 的 0，不需要占用动态范围。
        """
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach().to(self.min_val.dtype)

        if self.ch_axis == -1:
            # ★ Sparse-Aware: 非零值专用直方图
            if self.sparse_mode:
                nz_mask = x.abs() > 1e-6
                x_nz = x[nz_mask]
                if x_nz.numel() < 10:
                    # 非零值太少（极端稀疏帧），跳过本 batch 直方图更新
                    min_val_cur, max_val_cur = torch._aminmax(x)
                    if self.max_val.numel() <= 1 and self.max_val.isinf():
                        self.min_val = min_val_cur
                        self.max_val = max_val_cur
                    else:
                        self.min_val = torch.min(self.min_val, min_val_cur)
                        self.max_val = torch.max(self.max_val, max_val_cur)
                    return x_orig
                x_for_hist = x_nz
            else:
                x_for_hist = x

            cur_max = x_for_hist.abs().max()
            old_hist = self.histogram if self.histogram.ndim == 1 else self.histogram.reshape(-1)
            old_max = self.hist_max if self.hist_max.ndim == 0 else self.hist_max.reshape(-1)[0]
            self.histogram, self.hist_max = self._update_histogram_1d(
                x_for_hist.abs(), old_hist, old_max, cur_max
            )
            min_val_cur, max_val_cur = torch._aminmax(x)  # min/max 始终基于全量 x
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)  # [C, N]
            num_channels = y.size(0)
            self._ensure_per_channel_buffers(
                num_channels, device=y.device, dtype=self.min_val.dtype
            )

            abs_y = y.abs()
            for c in range(num_channels):
                if self.sparse_mode:
                    # ★ Sparse-Aware per-channel：逐通道只统计非零值
                    nz_mask_c = abs_y[c] > 1e-6
                    vals_c = abs_y[c][nz_mask_c]
                    if vals_c.numel() < 10:
                        continue
                    cur_max_c = vals_c.max()
                else:
                    vals_c = abs_y[c]
                    cur_max_c = abs_y[c].max()
                self.histogram[c], self.hist_max[c] = self._update_histogram_1d(
                    vals_c, self.histogram[c], self.hist_max[c], cur_max_c
                )
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        # 同时记录 min/max（供 fallback 使用）
        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = torch.min(self.min_val, min_val_cur)
            self.max_val = torch.max(self.max_val, max_val_cur)

        return x_orig

    def _find_optimal_threshold_single(self, hist: np.ndarray, hist_max_val: float):
        """单通道 KL 最优阈值搜索。"""
        hist = np.asarray(hist, dtype=np.float64)
        num_bins = len(hist)
        num_q = self.num_quantized_bins  # 128 for symmetric INT8
        total = hist.sum()
        if total <= 0 or hist_max_val <= 0:
            return 0.0, 0.0

        p = hist / total  # 完整参考分布
        hist_nonzero = (hist > 0).astype(np.float64)

        # 百分位数约束：阈值必须覆盖至少 min_percentile 的样本质量
        cumsum = np.cumsum(hist)
        min_mass = self.min_percentile * total
        min_threshold_bin = int(np.searchsorted(cumsum, min_mass)) + 1
        min_threshold_bin = max(min_threshold_bin, num_q)
        min_threshold_bin = min(min_threshold_bin, num_bins)

        best_kl = float('inf')
        best_threshold_bin = num_bins

        # bin 中心坐标 [0.5, 1.5, ..., num_bins-0.5]
        bin_centers = np.arange(num_bins, dtype=np.float64) + 0.5

        for i in range(min_threshold_bin, num_bins + 1):
            # 每个 bin 的量化 level：center_k * (num_q-1) / i
            # 超出 [0, num_q-1] 的被 clip（模拟 INT8 饱和）
            levels = np.clip(
                np.round(bin_centers * (num_q - 1) / i).astype(np.int64),
                0, num_q - 1
            )

            # 每个 level 的总概率质量和非零 bin 计数
            level_mass = np.bincount(levels, weights=hist, minlength=num_q)
            level_nz = np.bincount(levels, weights=hist_nonzero, minlength=num_q)

            # Q[k]: 同一 level 内非零 bin 均分该 level 的总质量
            safe_nz = np.maximum(level_nz[levels], 1.0)
            q = np.where(hist > 0, level_mass[levels] / safe_nz, 0.0)

            q_sum = q.sum()
            if q_sum == 0:
                continue
            q_norm = q / q_sum

            # KL(P || Q) — 仅在 P>0 且 Q>0 的位置计算
            valid = (p > 0) & (q_norm > 0)
            if not valid.any():
                continue
            kl = float((p[valid] * np.log(p[valid] / q_norm[valid])).sum())

            if kl < best_kl:
                best_kl = kl
                best_threshold_bin = i

        threshold = (best_threshold_bin / num_bins) * hist_max_val
        return float(threshold), float(best_kl)

    def _find_optimal_threshold(self):
        """KL 最优阈值搜索（per-tensor / per-channel）。"""
        if self.ch_axis == -1:
            hist = self.histogram.float().cpu().numpy().astype(np.float64)
            hist_max_val = float(self.hist_max.item()) if self.hist_max.numel() == 1 else float(
                self.hist_max.max().item())
            return self._find_optimal_threshold_single(hist, hist_max_val)

        hists = self.histogram.float().cpu().numpy().astype(np.float64)
        hist_max = self.hist_max.float().cpu().numpy().astype(np.float64)
        num_channels = hists.shape[0]
        thresholds = np.zeros(num_channels, dtype=np.float64)
        kls = np.zeros(num_channels, dtype=np.float64)
        for c in range(num_channels):
            thresholds[c], kls[c] = self._find_optimal_threshold_single(hists[c], hist_max[c])
        return thresholds, kls

    def calculate_qparams(self):
        """在校准完成后计算量化参数。使用 KL 散度找到最优截断阈值。"""
        if self.histogram.sum() > 0:
            threshold, _ = self._find_optimal_threshold()
            if self.ch_axis == -1:
                t = torch.tensor(float(threshold), device=self.min_val.device, dtype=self.min_val.dtype)
            else:
                t = torch.as_tensor(threshold, device=self.min_val.device, dtype=self.min_val.dtype)
            # 对称量化：[-threshold, threshold]
            self.min_val = -t
            self.max_val = t
            self._calibrated.fill_(1)

        return super().calculate_qparams()


# ============================================================================
# Round 9：SparseLog2FakeQuantize — 对数域量化，针对稀疏卷积激活
# ============================================================================
#
# 动机（基于 Round 8 W8A16 控制实验的结论）：
#   - lidar 精度损失 95.5% 来自激活量化，权重量化代价仅 0.0001 NDS
#   - INT8 均匀量化假设值域内均匀分布，步长固定 ≈ 0.076（T=9.7 时）
#   - 稀疏激活（BN+ReLU 后）呈截断对数正态分布：大量有效信号集中在 (0, 0.5] 的
#     密集区，均匀量化在此区域仅有 ~13 个格点，相对误差高达 100-200%
#   - Log2 量化：相邻格点比例恒为 2，在 x < 1 密集区提供 ~60 个格点，
#     对任意量级信号保持恒定相对误差（≈ 41%）
#   - 零值仍被精确表示（稀疏卷积核心优势不变）

class SparseLog2FakeQuantize(nn.Module):
    """对数域激活量化，适用于 BN+ReLU 后的稀疏卷积特征。

    可表示的非零正值集合：{a^(log_a_base + k) : k ∈ [qmin, qmax]}
    其中 log_a_base 通过校准数据非零激活分布的低百分位自动估计。
    零值始终精确还原。

    与 INT8 均匀量化的关键区别：
      - 均匀：绝对误差恒定，小幅值信号相对误差 >> 100%
      - LogA：相对误差近似恒定，对对数正态稀疏分布更友好

    与 MQBench enable_calibration / enable_quantization 完全兼容：
      - enable_calibration:  observer ON,  fake_quant OFF → 更新 log2_base
      - enable_quantization: observer OFF, fake_quant ON  → 按 log2 量化

    Args:
        n_bits:        量化位宽（默认 8，正侧 127 个格点）
        per_channel:   是否逐通道估计 base（features 形状 [N,C]，按 C 维）
        ch_axis:       通道维度（默认 1，对应 [N,C] 的 C 维）
        ema_ratio:     log2_base EMA 更新系数（默认 0.9）
        percentile:    估计 base 用的分位数（默认 0.05 = 第 5 百分位）
        eps:           零值判断阈值（默认 1e-6）
        log_base:      对数底 a（默认 2.0 = 原始 Log2；可设为 1.25/1.5/e/...）
    """

    def __init__(self, n_bits=8, per_channel=False, ch_axis=1,
                 ema_ratio=0.9, percentile=0.05, eps=1e-6, log_base=2.0):
        super().__init__()
        if log_base <= 1.0:
            raise ValueError(f"log_base must be > 1.0, got {log_base}")
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.ema_ratio = ema_ratio
        self.percentile = percentile
        self.eps = eps
        self.register_buffer("log_base", torch.tensor(float(log_base), dtype=torch.float32))
        # 对称 INT8：[-127, 127]，正侧 127 个格点
        self.qmin = -(2 ** (n_bits - 1) - 1)
        self.qmax = (2 ** (n_bits - 1) - 1)

        # log2(base)：非零激活最小幅值的对数估计
        #   per-tensor: scalar；per-channel: [C]（第一次 forward 动态初始化）
        self.register_buffer('log2_base', torch.tensor(-8.0))
        # MQBench 兼容：enable_calibration / enable_quantization 查询这些标志
        self.register_buffer('_observer_enabled', torch.tensor(1, dtype=torch.uint8))
        self.register_buffer('_fake_quant_enabled', torch.tensor(1, dtype=torch.uint8))
        self.register_buffer('fake_quant_enabled', torch.tensor(1, dtype=torch.uint8))
        self.register_buffer('_initialized', torch.tensor(0, dtype=torch.uint8))

    # ── MQBench 兼容接口 ──────────────────────────────────────────────────
    def enable_observer(self, enabled=True):
        self._observer_enabled.fill_(1 if enabled else 0)

    def disable_observer(self):
        self._observer_enabled.fill_(0)

    def enable_fake_quant(self, enabled=True):
        self._fake_quant_enabled.fill_(1 if enabled else 0)
        self.fake_quant_enabled.fill_(1 if enabled else 0)

    def disable_fake_quant(self):
        self._fake_quant_enabled.fill_(0)
        self.fake_quant_enabled.fill_(0)

    def calculate_qparams(self):
        """Populate scale/zero_point buffers for MQBench compatibility.

        log2_base is updated via EMA during forward; this method only
        materializes the equivalent linear scale for downstream tools.
        """
        scale = torch.pow(self.log_base.float(), self.log2_base.float())
        if not hasattr(self, 'scale') or self.scale is None:
            self.register_buffer('scale', scale.detach().clone())
        else:
            self.scale.detach().copy_(scale)
        if not hasattr(self, 'zero_point') or self.zero_point is None:
            self.register_buffer('zero_point', torch.zeros_like(scale, dtype=torch.long))
        else:
            self.zero_point.detach().fill_(0)
        return self.scale, self.zero_point

    @property
    def scale(self):
        """等效线性 scale = a^log_a_base（供诊断 / 日志使用）。"""
        return torch.pow(self.log_base.float(), self.log2_base.float())

    @property
    def zero_point(self):
        return torch.zeros(1, device=self.log2_base.device, dtype=torch.long)

    # ── 校准：估计 log2_base ─────────────────────────────────────────────
    def _update_base_per_tensor(self, x_nz_abs: torch.Tensor):
        """用非零绝对值的第 p 百分位更新 log2_base（EMA）。"""
        x_safe = x_nz_abs.clamp(min=1e-30)
        if abs(self.log_base.item() - 2.0) < 1e-6:
            log2_vals = torch.log2(x_safe)
        else:
            log2_vals = torch.log(x_safe) / math.log(float(self.log_base.item()))
        k = max(1, int(self.percentile * x_nz_abs.numel()))
        # sort().values[k]：第 k 小的 log2 值 = 第 p 百分位
        new_log2_base = log2_vals.sort().values[k].clamp(-20.0, 2.0)
        if not self._initialized.item():
            self.log2_base.fill_(new_log2_base.item())
            self._initialized.fill_(1)
        else:
            self.log2_base.mul_(self.ema_ratio).add_(
                new_log2_base * (1.0 - self.ema_ratio)
            )

    def _update_base_per_channel(self, x: torch.Tensor):
        """逐通道更新 log2_base（EMA）。x 形状 [N, C]。"""
        C = x.shape[1]
        if self.log2_base.shape != torch.Size([C]):
            # 第一次前向：动态初始化 per-channel buffer
            self.log2_base = torch.full(
                (C,), -8.0, device=x.device, dtype=torch.float32
            )
            self._initialized.fill_(0)

        for c in range(C):
            x_c = x[:, c].detach().abs()
            nz_c = x_c[x_c > self.eps]
            if nz_c.numel() < 5:
                continue
            if abs(self.log_base.item() - 2.0) < 1e-6:
                log2_c = torch.log2(nz_c)
            else:
                log2_c = torch.log(nz_c) / math.log(float(self.log_base.item()))
            k = max(1, int(self.percentile * nz_c.numel()))
            new_base_c = log2_c.sort().values[k].clamp(-20.0, 2.0)
            if not self._initialized.item():
                self.log2_base[c] = new_base_c.item()
            else:
                self.log2_base[c] = (
                        self.ema_ratio * self.log2_base[c]
                        + (1.0 - self.ema_ratio) * new_base_c.item()
                )
        self._initialized.fill_(1)

    # ── 核心前向 ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 校准阶段：更新 log2_base ──────────────────────────────────────
        if self._observer_enabled.item():
            with torch.no_grad():
                nz_mask = x.abs() > self.eps
                x_nz = x[nz_mask]
                if x_nz.numel() >= 10:
                    if not self.per_channel:
                        self._update_base_per_tensor(x_nz.abs().float())  # ★ .float()
                    else:
                        if x.ndim == 2:
                            self._update_base_per_channel(x.detach().float())  # ★ .float()
                        else:
                            self._update_base_per_tensor(x_nz.abs().float())

        # ── 量化推理阶段 ─────────────────────────────────────────────────
        if not self._fake_quant_enabled.item():
            return x

        orig_dtype = x.dtype  # ★ 保存原始 dtype（可能是 float16）
        x_f = x.float()  # ★ 提升到 float32 做对数运算

        zero_mask = x_f.abs() < self.eps
        sign = x_f.sign()

        if self.per_channel and self.log2_base.ndim == 1:
            base = self.log2_base.to(x_f.device).unsqueeze(0)
        else:
            base = self.log2_base.to(x_f.device)

        x_abs = x_f.abs().clamp(min=1e-30)
        if abs(self.log_base.item() - 2.0) < 1e-6:
            log2_x = torch.log2(x_abs) - base
        else:
            log2_x = torch.log(x_abs) / math.log(float(self.log_base.item())) - base
        q_int = torch.round(log2_x).clamp(self.qmin, self.qmax)
        x_dq = sign * torch.pow(self.log_base.to(x_f.device), q_int + base)
        x_dq = torch.where(zero_mask, torch.zeros_like(x_f), x_dq)

        # STE：梯度直通，前向使用量化值
        out = x_f + (x_dq - x_f).detach()
        return out.to(orig_dtype)  # ★ cast 回原始 dtype

    def extra_repr(self) -> str:
        if self.log2_base.numel() > 1:
            base_str = (f"log2_base=[{self.log2_base.min().item():.2f}, "
                        f"{self.log2_base.max().item():.2f}]")
        else:
            base_str = f"log2_base={self.log2_base.item():.2f}"
        return (f"n_bits={self.n_bits}, per_channel={self.per_channel}, "
                f"qmin={self.qmin}, qmax={self.qmax}, log_base={self.log_base.item():.4g}, {base_str}, "
                f"percentile={self.percentile}")


def _report_log2_quantizer_results(model, logger):
    """报告所有 SparseLog2FakeQuantize 节点的校准结果。"""
    count = 0
    inner = model.module if hasattr(model, 'module') else model
    for name, module in inner.named_modules():
        if isinstance(module, SparseLog2FakeQuantize):
            b = module.log2_base
            if module._initialized.item():
                count += 1
                if b.numel() == 1:
                    log_base = float(module.log_base.item())
                    logger.info(
                        f"  Log2 [{name.split('.')[-2]}]: "
                        f"log2_base={b.item():.3f}  "
                        f"log_base={log_base:.4g}  "
                        f"base={log_base ** b.item():.4f}  "
                        f"range=[{log_base ** (b.item() + module.qmin):.4e}, "
                        f"{log_base ** (b.item() + module.qmax):.4f}]"
                    )
                else:
                    logger.info(
                        f"  Log2 [{name.split('.')[-2]}] per-channel C={b.numel()}: "
                        f"log_base={module.log_base.item():.4g}, "
                        f"log2_base(mean/min/max)=("
                        f"{b.mean().item():.2f}/"
                        f"{b.min().item():.2f}/"
                        f"{b.max().item():.2f})"
                    )
    if count:
        logger.info(f"  共 {count} 个 SparseLog2FakeQuantize 完成校准。")


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
# 手动量化：为无法 torch.fx 追踪的模块（SwinT、TransFusionHead 等）提供回退
# ============================================================================

def _create_tensorrt_fakeq_pair(act_observer_cls=None):
    """创建一对 (weight_fq, act_fq)，匹配 MQBench TensorRT INT8 配置。
    兼容不同版本 MQBench（不依赖 symmetric_range kwarg）。

    Args:
        act_observer_cls: 激活 Observer 类（默认 EMAMinMaxObserver）。
            可选：KLDivergenceObserver, MSEObserver, EMAQuantileObserver 等。
    """
    if act_observer_cls is None:
        act_observer_cls = EMAMinMaxObserver
    w_params = QuantizeScheme(
        symmetry=True, per_channel=True, pot_scale=False, bit=8
    ).to_observer_params()
    w_params['quant_min'] = -127  # TensorRT 标准对称范围 [-127, 127]
    a_params = QuantizeScheme(
        symmetry=True, per_channel=False, pot_scale=False, bit=8
    ).to_observer_params()
    a_params['quant_min'] = -127
    weight_fq = LearnableFakeQuantize(observer=MinMaxObserver, **w_params)
    act_fq = LearnableFakeQuantize(observer=act_observer_cls, **a_params)
    return weight_fq, act_fq


class _QuantizedConv2d(nn.Module):
    """Conv2d + MQBench FakeQuantize（适用于无法 fx 追踪的模块）。"""

    def __init__(self, original, act_observer_cls=None):
        super().__init__()
        self.conv = original
        self.weight_fake_quant, self.act_fake_quant = _create_tensorrt_fakeq_pair(act_observer_cls)

    # 代理常用属性，确保外部代码直接访问 .weight / .bias 等不会出错
    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = self.act_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        return F.conv2d(
            x, weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class _QuantizedLinear(nn.Module):
    """Linear + MQBench FakeQuantize（适用于无法 fx 追踪的模块）。"""

    def __init__(self, original, act_observer_cls=None):
        super().__init__()
        self.linear = original
        self.weight_fake_quant, self.act_fake_quant = _create_tensorrt_fakeq_pair(act_observer_cls)

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
        x = self.act_fake_quant(x)
        weight = self.weight_fake_quant(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)


def manual_quantize_nontraceable(module, logger, module_name="unknown", act_observer_cls=None):
    """对无法 torch.fx 追踪的模块手动插入 FakeQuantize 节点。

    逐层替换 Conv2d/Linear 为带有 FakeQuantize 的包装版本。
    与 MQBench enable_calibration / enable_quantization 完全兼容。

    Args:
        module: 待量化模块
        logger: 日志记录器
        module_name: 模块名（仅用于日志）
        act_observer_cls: 激活 Observer 类（默认 None → EMAMinMaxObserver）
    """
    replacements = []
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, _QuantizedConv2d):
            replacements.append((name, _QuantizedConv2d(child, act_observer_cls)))
        elif isinstance(child, nn.Linear) and not isinstance(child, _QuantizedLinear):
            replacements.append((name, _QuantizedLinear(child, act_observer_cls)))

    for name, replacement in replacements:
        _set_nested_attr(module, name, replacement)

    obs_name = act_observer_cls.__name__ if act_observer_cls else "EMAMinMaxObserver"
    fq_count = len(replacements) * 2
    logger.info(
        f"  ↪ 手动量化 {module_name}: 替换 {len(replacements)} 个 Conv2d/Linear，"
        f"插入 {fq_count} 个 FakeQuant 节点 (act_observer: {obs_name})"
    )
    return module


# ============================================================================
# 稀疏卷积量化：为 SparseEncoder (spconv v1.x) 提供 FakeQuant 支持
# ============================================================================

def _replace_feature(sparse_tensor: SparseConvTensor,
                     new_features: torch.Tensor) -> SparseConvTensor:
    """SparseConvTensor の features を新テンソルで置き換えて返す。

    Round 8 修复：避免 in-place 改写原始 SparseConvTensor。
    spconv v2 用 replace_feature()，v1.x 手动构造新 tensor。
    """
    if hasattr(sparse_tensor, 'replace_feature'):
        return sparse_tensor.replace_feature(new_features)
    new_tensor = SparseConvTensor(
        new_features,
        sparse_tensor.indices,
        sparse_tensor.spatial_shape,
        sparse_tensor.batch_size,
    )
    for attr in ('grid', 'voxel_num', 'indice_dict', 'benchmark', 'benchmark_record'):
        if hasattr(sparse_tensor, attr):
            setattr(new_tensor, attr, getattr(sparse_tensor, attr))
    return new_tensor


def _create_spconv_fakeq_pair(act_observer_cls=None, act_per_channel=False,
                              no_act_quant=False, log_base=2.0):
    """创建 FakeQuant 对，适用于稀疏卷积。

    Round 8 新增：
      - no_act_quant=True → W8A16 模式，只量化权重，激活保持 FP
      - KLDivergenceObserver 自动注入 sparse_mode=True（稀疏感知校准）

    Round 9 新增：
      - act_observer_cls=SparseLog2FakeQuantize → 直接构造 log2 量化节点，
        跳过 MQBench LearnableFakeQuantize 包装

    Args:
        act_observer_cls: 激活 Observer 类或哨兵（默认 EMAMinMaxObserver）
        act_per_channel:  是否逐通道量化激活（features [N,C]，按 C 维）
        no_act_quant:     W8A16 模式，激活不量化
    """
    if act_observer_cls is None:
        act_observer_cls = EMAMinMaxObserver

    # 权重始终 per-channel INT8（稀疏卷积权重 [K,K,K,C_in,C_out]，ch_axis=4）
    w_params = QuantizeScheme(
        symmetry=True, per_channel=True, pot_scale=False, bit=8
    ).to_observer_params()
    w_params['quant_min'] = -127
    w_params['ch_axis'] = 4
    weight_fq = LearnableFakeQuantize(observer=MinMaxObserver, **w_params)

    # W8A16：激活不量化
    if no_act_quant:
        return weight_fq, None

    # ★ Round 9：Log2 量化（哨兵路径）
    if act_observer_cls is SparseLog2FakeQuantize:
        act_fq = SparseLog2FakeQuantize(
            per_channel=act_per_channel,
            ch_axis=1,
            log_base=log_base,
        )
        return weight_fq, act_fq

    # 标准 MQBench 路径
    a_params = QuantizeScheme(
        symmetry=True, per_channel=act_per_channel, pot_scale=False, bit=8
    ).to_observer_params()
    a_params['quant_min'] = -127
    if act_per_channel:
        a_params['ch_axis'] = 1

    # ★ Round 8：KLDivergenceObserver 自动启用 sparse_mode
    if act_observer_cls is KLDivergenceObserver:
        a_params['sparse_mode'] = True

    act_fq = LearnableFakeQuantize(observer=act_observer_cls, **a_params)
    return weight_fq, act_fq


class _QuantizedSparseConv(SparseModule):
    """SparseConvolution + FakeQuantize（spconv v1.x 稀疏卷积量化）。

    Round 8 修复：
      1. features 更新通过 _replace_feature() 完成，避免 in-place 改写
      2. weight 还原用 try-finally 保证异常安全
      3. 支持 W8A16（act_fake_quant=None）

    Round 9 新增：
      4. act_fake_quant 可以是 SparseLog2FakeQuantize 实例
    """

    def __init__(self, original, act_observer_cls=None, act_per_channel=False,
                 no_act_quant=False, log_base=2.0):
        super().__init__()
        self.conv = original
        self.weight_fake_quant, self.act_fake_quant = _create_spconv_fakeq_pair(
            act_observer_cls=act_observer_cls,
            act_per_channel=act_per_channel,
            no_act_quant=no_act_quant,
            log_base=log_base,
        )
        self.register_buffer('_weight_dirty', torch.tensor(0, dtype=torch.uint8))

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)

        if self.training:
            raise RuntimeError(
                "_QuantizedSparseConv does not support training mode. "
                "Please use eval mode for PTQ inference."
            )

        if self._weight_dirty.item():
            raise RuntimeError(
                "_QuantizedSparseConv detected dirty weight state from a previous "
                "interrupted forward. Please reload the model checkpoint."
            )

        # ★ 激活量化（_replace_feature 避免 in-place 改写）
        if self.act_fake_quant is not None:
            quant_feats = self.act_fake_quant(input.features)
            input = _replace_feature(input, quant_feats)

        # ★ 权重量化：用 in-place copy_ 替代 data 指针重写，更稳定且不会替换 Parameter
        saved_weight = self.conv.weight.data.clone()
        self.conv.weight.data.copy_(self.weight_fake_quant(saved_weight))
        self._weight_dirty.fill_(1)
        try:
            output = self.conv(input)
        finally:
            self.conv.weight.data.copy_(saved_weight)
            self._weight_dirty.fill_(0)
        return output


def manual_quantize_sparse(
        module, logger, module_name="unknown", act_observer_cls=None,
        act_per_channel=False, no_act_quant=False, log_base=2.0,
):
    """对 SparseEncoder 中的稀疏卷积层插入 FakeQuantize 节点。

    替换所有 SparseConvolution (SubMConv3d/SparseConv3d) 为带 FakeQuant 的包装版本。
    BatchNorm1d / ReLU 等非稀疏层不受影响。

    Round 8+9 新增 Args:
        no_act_quant: W8A16 模式，只量化权重，激活保持 FP
    """
    replacements = []
    for name, child in list(module.named_modules()):
        if isinstance(child, SparseConvolution) and not isinstance(
                child, _QuantizedSparseConv
        ):
            replacements.append((
                name,
                _QuantizedSparseConv(
                    child,
                    act_observer_cls=act_observer_cls,
                    act_per_channel=act_per_channel,
                    no_act_quant=no_act_quant,
                    log_base=log_base,
                )
            ))

    for name, replacement in replacements:
        _set_nested_attr(module, name, replacement)

    is_log2 = (act_observer_cls is SparseLog2FakeQuantize)
    is_kl = (act_observer_cls is KLDivergenceObserver)
    obs_name = (act_observer_cls.__name__ if act_observer_cls else "EMAMinMaxObserver")
    act_scheme = "per-channel" if act_per_channel else "per-tensor"
    fq_count = len(replacements) * (1 if no_act_quant else 2)

    if no_act_quant:
        mode_str = "W8A16（权重only，激活FP）"
    elif is_log2:
        mode_str = f"LogA量化(a={log_base:g}), {act_scheme}"
    elif is_kl:
        mode_str = f"{obs_name} [sparse_aware=ON], {act_scheme}"
    else:
        mode_str = f"{obs_name}, {act_scheme}"

    logger.info(
        f"  ↪ 稀疏卷积量化 {module_name}: 替换 {len(replacements)} 个 SparseConv，"
        f"插入 {fq_count} 个 FakeQuant 节点 ({mode_str})"
    )
    return module


# ============================================================================
# 选择性量化：对全模型 8/8 子模块插入 FakeQuantize 节点
# ============================================================================

# 量化路径说明：
#
# 路径一（torch.fx 自动）：camera/neck, fuser, decoder/backbone, decoder/neck
# 路径二（手动 Conv2d/Linear 包装）：camera/backbone, camera/vtransform, heads
# 路径三（手动 SparseConv 包装）：lidar/backbone
#
# 设计跳过：lidar/voxelize (体素化预处理，非神经网络层)

_QUANTIZABLE_SUBMODULE_KEYS = [
    # (attr_path_on_model, display_name)
    # camera branch
    ("encoders.camera.backbone", "camera/backbone"),
    ("encoders.camera.neck", "camera/neck"),
    ("encoders.camera.vtransform", "camera/vtransform"),
    # lidar branch
    ("encoders.lidar.backbone", "lidar/backbone"),
    # fuser
    ("fuser", "fuser"),
    # decoder
    ("decoder.backbone", "decoder/backbone"),
    ("decoder.neck", "decoder/neck"),
    # heads (TransFusionHead)
    ("decoder.heads.object", "heads/object"),
]


# heads 单独遍历（数量不定）


def _get_nested_attr(obj, key: str):
    """支持 'a.b.c' 形式的嵌套属性访问（含 ModuleDict/ModuleList/Sequential）。"""
    for part in key.split("."):
        if isinstance(obj, nn.ModuleDict):
            obj = obj[part]
        elif isinstance(obj, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj, key: str, value):
    """支持 'a.b.c' 形式的嵌套属性设置（含 ModuleDict/ModuleList/Sequential）。"""
    parts = key.split(".")
    for part in parts[:-1]:
        if isinstance(obj, nn.ModuleDict):
            obj = obj[part]
        elif isinstance(obj, (nn.ModuleList, nn.Sequential)) and part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, nn.ModuleDict):
        obj[last] = value
    elif isinstance(obj, (nn.ModuleList, nn.Sequential)) and last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def apply_selective_ptq(
        model,
        backend_type,
        logger,
        skip_modules=None,
        act_observer_cls=None,
        vtransform_observer_cls=None,
        sparse_act_per_channel=False,
        no_lidar_act_quant=False,
        log_base=2.0,
):
    """
    对模型中全部可量化子模块插入 FakeQuantize 节点。

    三条量化路径：
      1. torch.fx 自动插桩（标准密集卷积模块）
      2. 手动 FakeQuant 包装 Conv2d/Linear（fx 追踪失败的密集模块）
      3. 手动 FakeQuant 包装 SparseConvolution（稀疏卷积模块）

    Round 8+9 新增 Args:
        no_lidar_act_quant: W8A16 控制实验 —— lidar 只量化权重，激活保持 FP
    """
    success, failed, skipped = [], [], []

    def _has_sparse_conv(module):
        return any(isinstance(m, SparseConvolution) for m in module.modules())

    def _try_quantize(submodule, display_name, attr_key=None, set_back=None):
        is_vtransform = (display_name == "camera/vtransform")
        is_lidar = (display_name == "lidar/backbone")
        manual_obs = vtransform_observer_cls if is_vtransform else None

        # 路径 1: torch.fx 自动插桩
        try:
            with patch_mmcv_for_fx():
                quantized = prepare_by_platform(submodule, backend_type)
            if attr_key:
                _set_nested_attr(model, attr_key, quantized)
            elif set_back:
                set_back(quantized)
            success.append(display_name)
            logger.info(f"  ✓ 量化子模块: {display_name} (fx)")
            return
        except Exception as e:
            logger.warning(
                f"  ✗ {display_name} torch.fx 追踪失败: "
                f"{type(e).__name__}: {str(e)[:80]}"
            )

        # 路径 3: 稀疏卷积量化（SparseEncoder 内无 Conv2d，需专用处理）
        if _has_sparse_conv(submodule):
            try:
                manual_quantize_sparse(
                    submodule,
                    logger,
                    display_name,
                    act_observer_cls=act_observer_cls,
                    act_per_channel=sparse_act_per_channel,
                    no_act_quant=(no_lidar_act_quant and is_lidar),
                    log_base=log_base,
                )
                success.append(f"{display_name} (稀疏)")
                return
            except Exception as e2:
                failed.append(display_name)
                logger.warning(f"  ✗ {display_name} 稀疏卷积量化失败: {e2}")
                return

        # 路径 2: 手动 FakeQuant 包装 Conv2d/Linear
        try:
            manual_quantize_nontraceable(submodule, logger, display_name, act_observer_cls=manual_obs)
            success.append(f"{display_name} (手动)")
        except Exception as e2:
            failed.append(display_name)
            logger.warning(f"  ✗ {display_name} 手动量化也失败: {e2}")

    if skip_modules is None:
        skip_modules = []

    # --- 固定路径的子模块 ---
    for attr_key, display_name in _QUANTIZABLE_SUBMODULE_KEYS:
        if display_name in skip_modules:
            skipped.append(f"{display_name} (--skip-modules)")
            logger.info(f"  ⊘ 跳过子模块: {display_name} (--skip-modules)")
            continue
        try:
            submodule = _get_nested_attr(model, attr_key)
        except (KeyError, AttributeError):
            skipped.append(display_name)
            continue
        _try_quantize(submodule, display_name, attr_key=attr_key)

    # --- heads（数量可变）---
    if hasattr(model, "heads"):
        for head_name, head_module in model.heads.items():
            display_name = f"heads/{head_name}"
            if display_name in skip_modules:
                skipped.append(f"{display_name} (--skip-modules)")
                logger.info(f"  ⊘ 跳过子模块: {display_name} (--skip-modules)")
                continue
            if display_name in success or any(display_name in s for s in success):
                continue
            _try_quantize(
                head_module, display_name,
                set_back=lambda q, hn=head_name: model.heads.__setitem__(hn, q),
            )

    logger.info(
        f"选择性量化完成: 成功 {len(success)} 个, "
        f"失败 {len(failed)} 个, 不存在/跳过 {len(skipped)} 个"
    )
    if failed:
        logger.warning(f"  失败的子模块: {failed}")

    skipped_by_design = [
        "lidar/voxelize  (体素化预处理，非神经网络层)",
        "radar/voxelize  (体素化预处理，如有)",
    ]
    if skipped_by_design:
        logger.info("以下部分已设计跳过量化（非神经网络层）：")
        for item in skipped_by_design:
            logger.info(f"  - {item}")

    return model, success, failed


# ============================================================================
# Learnable Weight Clipping (LWC)
# —— 对稀疏卷积权重学习最优截断范围，参考 OmniQuant (ICLR 2024)
# ============================================================================

def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding: forward=round, backward=identity."""
    return (x.round() - x).detach() + x


def _fake_quant_weight(w: torch.Tensor, clip_min: torch.Tensor, clip_max: torch.Tensor,
                       ch_axis: int = 4, qmin: int = -127, qmax: int = 127) -> torch.Tensor:
    """可微 per-channel fake quantization with learnable clipping.

    Args:
        w: 原始权重 [K,K,K,C_in,C_out]
        clip_min: per-channel 下界 [C_out]（负值）
        clip_max: per-channel 上界 [C_out]（正值）
        ch_axis: 输出通道维度（稀疏卷积=4）
        qmin, qmax: 量化范围（INT8 对称 = [-127, 127]）

    Returns:
        w_dequant: 经过 clamp → scale → round(STE) → dequant 的权重
    """
    # 把 clip_min/clip_max reshape 为可广播的形状
    ndim = w.ndim
    shape = [1] * ndim
    shape[ch_axis] = -1
    cmin = clip_min.view(shape)
    cmax = clip_max.view(shape)

    # clamp → compute scale → quantize → dequantize
    w_clamped = w.clamp(cmin, cmax)
    abs_max = torch.max(cmax.abs(), cmin.abs())
    scale = abs_max / qmax  # per-channel scale
    scale = scale.clamp(min=1e-8)

    w_int = _round_ste(w_clamped / scale)
    w_int = w_int.clamp(qmin, qmax)
    w_dequant = w_int * scale
    return w_dequant


def optimize_lwc_sparse(model, logger,
                        lr: float = 0.01,
                        num_iters: int = 500,
                        init_value: float = 4.0):
    """对模型中所有 _QuantizedSparseConv 的权重学习最优截断范围 (LWC)。

    原理（OmniQuant LWC, Method A — Weight Reconstruction）：
      对每个稀疏卷积层的权重 W，学习 per-channel 的截断参数 γ, δ：
        clip_max = sigmoid(γ) · max_W     (per output channel)
        clip_min = sigmoid(δ) · min_W     (per output channel)
      优化目标：min_{γ,δ} MSE(W, FakeQuant(clamp(W, clip_min, clip_max)))
      梯度通过 STE (Straight-Through Estimator) 穿过 round 操作。

    优化完成后，将截断后的权重写回原始 conv.weight.data，使得后续
    MinMax 校准器看到的权重已无离群点，自然得到更合理的 scale。

    Args:
        model: 已经过 apply_selective_ptq 的模型（含 _QuantizedSparseConv）
        logger: 日志记录器
        lr: Adam 学习率 (default: 0.01, 与 OmniQuant 一致)
        num_iters: 每层优化迭代次数 (default: 500)
        init_value: γ/δ 初始值 (default: 4.0, sigmoid(4)≈0.982)
    """
    inner = model.module if hasattr(model, "module") else model
    sigmoid = nn.Sigmoid()

    lwc_layers = []
    for name, mod in inner.named_modules():
        if isinstance(mod, _QuantizedSparseConv):
            lwc_layers.append((name, mod))

    if not lwc_layers:
        logger.warning("LWC: 未找到 _QuantizedSparseConv 层，跳过。")
        return

    logger.info(f"")
    logger.info(f"╔══════════════════════════════════════════════════════════════╗")
    logger.info(f"║   LWC — Learnable Weight Clipping (稀疏卷积权重截断优化)     ║")
    logger.info(f"╠══════════════════════════════════════════════════════════════╣")
    logger.info(f"║  目标层数: {len(lwc_layers):>3d}  │  lr={lr}  │  iters={num_iters}  │  init={init_value}")
    logger.info(f"╚══════════════════════════════════════════════════════════════╝")

    ch_axis = 4  # 稀疏卷积权重 [K,K,K,C_in,C_out] 的输出通道维度
    total_clipped = 0
    total_params = 0

    for layer_idx, (name, qconv) in enumerate(lwc_layers):
        W = qconv.conv.weight.data  # [K,K,K,C_in,C_out]
        C_out = W.shape[ch_axis]
        total_params += W.numel()

        # 统计每个输出通道的 min/max
        # 将非 ch_axis 维度 flatten，得到 [rest, C_out]
        perm = list(range(W.ndim))
        perm.remove(ch_axis)
        perm.append(ch_axis)
        W_perm = W.permute(*perm).contiguous().view(-1, C_out)  # [K³·C_in, C_out]
        min_W = W_perm.min(dim=0).values.detach()  # [C_out], 负值
        max_W = W_perm.max(dim=0).values.detach()  # [C_out], 正值

        # 初始化可学习截断参数
        gamma = nn.Parameter(torch.full((C_out,), init_value, device=W.device))  # 上界
        delta = nn.Parameter(torch.full((C_out,), init_value, device=W.device))  # 下界

        optimizer = torch.optim.Adam([gamma, delta], lr=lr)
        W_target = W.detach().clone()  # 优化目标：原始 FP32 权重

        best_loss = float("inf")
        best_gamma = gamma.data.clone()
        best_delta = delta.data.clone()

        for it in range(num_iters):
            optimizer.zero_grad()

            clip_max = sigmoid(gamma) * max_W  # [C_out]
            clip_min = sigmoid(delta) * min_W  # [C_out]

            W_q = _fake_quant_weight(W_target, clip_min, clip_max, ch_axis=ch_axis)
            loss = torch.nn.functional.mse_loss(W_q, W_target)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_gamma = gamma.data.clone()
                best_delta = delta.data.clone()

        # 用最优截断参数截断权重（原地修改）
        with torch.no_grad():
            clip_max = sigmoid(best_gamma) * max_W
            clip_min = sigmoid(best_delta) * min_W

            # 计算截断比例
            clip_ratio_up = sigmoid(best_gamma).mean().item()
            clip_ratio_lo = sigmoid(best_delta).mean().item()
            n_clipped = ((W < clip_min.view(1, 1, 1, 1, -1)) |
                         (W > clip_max.view(1, 1, 1, 1, -1))).sum().item()
            total_clipped += n_clipped

            # 原地截断权重 → 使后续 MinMax Observer 看到更紧凑的范围
            for c in range(C_out):
                W[..., c].clamp_(clip_min[c].item(), clip_max[c].item())

        logger.info(
            f"  [{layer_idx + 1:>2d}/{len(lwc_layers)}] {name:<45s} "
            f"loss={best_loss:.6f}  clip↑={clip_ratio_up:.4f}  clip↓={clip_ratio_lo:.4f}  "
            f"clipped={n_clipped:>6d}/{W.numel()}"
        )

    clip_pct = total_clipped / max(total_params, 1) * 100
    logger.info(f"")
    logger.info(f"LWC 优化完成：共截断 {total_clipped:,} / {total_params:,} 个权重元素 ({clip_pct:.2f}%)")
    logger.info(f"截断后的权重已写回模型，后续 MinMax 校准将基于截断后的权重范围。")
    logger.info(f"")


# ============================================================================
# 校准阶段
# ============================================================================

def _sync_log2_quantizer_state(model, observe: bool):
    """SparseLog2FakeQuantize と MQBench lifecycle の同期。（Round 9）

    MQBench の enable_calibration / enable_quantization は LearnableFakeQuantize
    しか認識しないため、SparseLog2FakeQuantize は別途状態を同期する必要がある。

    observe=True  → 校准: observer ON,  fake_quant OFF
    observe=False → 量化: observer OFF, fake_quant ON
    """
    for m in model.modules():
        if isinstance(m, SparseLog2FakeQuantize):
            if observe:
                m.enable_observer()
                m.disable_fake_quant()
            else:
                m.disable_observer()
                m.enable_fake_quant()


def run_calibration(model, data_loader, num_batches, logger):
    """
    PTQ 校准：在校准数据上前向推理，收集各层激活值统计量。

    Round 9 更新：SparseLog2FakeQuantize 节点与 MQBench 状态同步。

    流程：
      enable_calibration → sync_log2(observe=True) →
      运行 num_batches → enable_quantization → sync_log2(observe=False)
    """
    logger.info(f"开始校准，共使用 {num_batches} 个 batch ...")

    enable_calibration(model)
    _sync_log2_quantizer_state(model, observe=True)  # ★ Round 9
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            try:
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                logger.warning(f"  校准 batch {i} 出错（已跳过）: {e}")
                continue
            if (i + 1) % 10 == 0:
                logger.info(f"  校准进度: {i + 1}/{num_batches}")

    logger.info("校准完成，scale/zero_point 已确定。")

    _report_kl_observer_results(model, logger)
    _report_log2_quantizer_results(model, logger)  # ★ Round 9

    enable_quantization(model)
    _sync_log2_quantizer_state(model, observe=False)  # ★ Round 9
    logger.info("模型已切换为量化推理模式（FakeQuant 激活）。")


def _report_kl_observer_results(model, logger):
    """报告所有 KLDivergenceObserver 的校准结果（阈值、KL 散度、范围压缩比）。"""
    kl_count = 0
    inner = model.module if hasattr(model, "module") else model
    for name, module in inner.named_modules():
        if isinstance(module, KLDivergenceObserver):
            if module.histogram.sum() > 0:
                threshold, kl = module._find_optimal_threshold()
                name_parts = name.split('.')
                short_name = ".".join(name_parts[-2:]) if len(name_parts) >= 2 else name

                if module.ch_axis == -1:
                    hist_max = module.hist_max.item()
                    compression = 1.0 - (threshold / hist_max) if hist_max > 0 else 0
                    # Compute what percentile the threshold covers
                    hist_np = module.histogram.float().cpu().numpy()
                    cumsum = np.cumsum(hist_np)
                    total = cumsum[-1]
                    thresh_bin = int(threshold / hist_max * len(hist_np)) if hist_max > 0 else len(hist_np)
                    thresh_bin = min(thresh_bin, len(hist_np) - 1)
                    pct_covered = cumsum[thresh_bin] / total * 100 if total > 0 else 100.0
                    msg = (f"  KL [{short_name}]: "
                           f"T={threshold:.4f}, max={hist_max:.4f}, "
                           f"KL={kl:.6f}, compress={compression * 100:.1f}%, "
                           f"covers={pct_covered:.2f}%")
                    print(msg, flush=True)
                else:
                    # per-channel：打印统计摘要，避免刷屏
                    threshold_t = torch.as_tensor(threshold, dtype=torch.float32)
                    kl_t = torch.as_tensor(kl, dtype=torch.float32)
                    hist_max_t = module.hist_max.detach().float().cpu()
                    valid = hist_max_t > 0
                    compression = torch.zeros_like(hist_max_t)
                    compression[valid] = 1.0 - threshold_t[valid] / hist_max_t[valid]

                    msg = (
                        f"  KL [{short_name}] (per-channel, C={threshold_t.numel()}): "
                        f"T(mean/med/max)=({threshold_t.mean().item():.4f}/"
                        f"{threshold_t.median().item():.4f}/{threshold_t.max().item():.4f}), "
                        f"KL(mean/max)=({kl_t.mean().item():.6f}/{kl_t.max().item():.6f}), "
                        f"compress(mean/max)=({(compression.mean().item() * 100):.1f}%/"
                        f"{(compression.max().item() * 100):.1f}%)"
                    )
                    print(msg, flush=True)
                # Also show the qparams that will be set
                module.calculate_qparams()
                if module.min_val.numel() == 1:
                    print(
                        f"    -> range: [{module.min_val.item():.4f}, {module.max_val.item():.4f}]",
                        flush=True
                    )
                else:
                    t = module.max_val.detach().abs().float().cpu()
                    print(
                        f"    -> per-channel range: C={t.numel()}, "
                        f"T(mean/med/max)=({t.mean().item():.4f}/{t.median().item():.4f}/{t.max().item():.4f})",
                        flush=True
                    )
                kl_count += 1
    if kl_count > 0:
        logger.info(f"  共 {kl_count} 个 KLDivergenceObserver 完成阈值搜索。")


# ============================================================================
# 构建模型
# ============================================================================

def build_ptq_model(
        cfg,
        logger,
        skip_modules=None,
        act_observer_cls=None,
        vtransform_observer_cls=None,
        sparse_act_per_channel=False,
        no_lidar_act_quant=False,
        log_base=2.0,
):
    """
    构建浮点模型，加载预训练权重，再对可量化子模块进行 PTQ 准备。

    Round 8+9 新增 Args:
        no_lidar_act_quant: W8A16 控制实验，lidar 只量化权重，激活保持 FP
    """
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
        logger.info("已启用 FP16 混合精度（与 test.py 对齐）。")

    if cfg.get("load_from", None):
        logger.info(f"加载预训练权重: {cfg.load_from}")
        from mmcv.runner import load_checkpoint
        checkpoint = load_checkpoint(model, cfg.load_from, map_location="cpu")
        logger.info("预训练权重加载完成（load_checkpoint）。")

    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    backend_type = BackendType.Tensorrt
    logger.info("开始对可量化子模块进行 PTQ 准备 (TensorRT INT8) ...")
    model, success, failed = apply_selective_ptq(
        model, backend_type, logger, skip_modules=skip_modules,
        act_observer_cls=act_observer_cls,
        vtransform_observer_cls=vtransform_observer_cls,
        sparse_act_per_channel=sparse_act_per_channel,
        no_lidar_act_quant=no_lidar_act_quant,
        log_base=log_base,
    )

    return model, success, failed


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
    # Print key metrics
    for key in sorted(metrics.keys()):
        if 'nds' in key.lower() or 'map' in key.lower():
            print(f"{key}: {metrics[key]}", flush=True)


# ============================================================================
# 量化诊断
# ============================================================================

# 模块路径映射（诊断 + 参数分析共用）
_ALL_MODULE_PATHS = [
    ("camera/backbone", "encoders.camera.backbone"),
    ("camera/neck", "encoders.camera.neck"),
    ("camera/vtransform", "encoders.camera.vtransform"),
    ("lidar/backbone", "encoders.lidar.backbone"),
    ("fuser", "fuser"),
    ("decoder/backbone", "decoder.backbone"),
    ("decoder/neck", "decoder.neck"),
    ("heads/object", "heads.object"),
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
        if q_pct >= 99:
            logger.info(f"  📊 量化覆盖率 {q_pct:.1f}%（全模型 INT8）")
            logger.info("     NDS 下降取决于各模块的量化敏感度。")
        elif q_pct >= 80:
            logger.info(f"  📊 量化覆盖率 {q_pct:.1f}%（{u_pct:.1f}% 仍为 FP32）")
            logger.info("     覆盖率已较高，NDS 下降主要取决于各模块的量化敏感度。")
        else:
            logger.info("  💡 FP32 ≈ INT8 (NDS 几乎无差异) 的原因:")
            logger.info(
                f"     仅 {q_pct:.1f}% 的参数被量化，"
                f"{u_pct:.1f}% 的模型仍为 FP32。"
            )
            logger.info("     量化覆盖率低 → 对端到端 NDS 影响自然很小。")
            logger.info("")
            logger.info("  📊 要获得更显著的量化效果，需要量化更多模块。")
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
        "--load-from",
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
    parser.add_argument(
        "--skip-modules",
        type=str,
        nargs="+",
        default=[],
        help="display names of modules to skip (e.g. --skip-modules camera/vtransform lidar/backbone)",
    )
    # LWC (Learnable Weight Clipping) 参数
    parser.add_argument(
        "--lwc",
        action="store_true",
        help="enable Learnable Weight Clipping for sparse conv weights (OmniQuant-style)",
    )
    parser.add_argument(
        "--lwc-lr",
        type=float,
        default=0.01,
        help="LWC learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lwc-iters",
        type=int,
        default=500,
        help="LWC optimization iterations per layer (default: 500)",
    )
    # 稀疏卷积激活 Observer / 量化器选择
    parser.add_argument(
        "--act-observer",
        type=str,
        default="ema_minmax",
        choices=["ema_minmax", "mse", "ema_quantile", "kl_divergence", "log2"],
        help="activation observer for sparse conv (lidar/backbone) quantization. "
             "ema_minmax: EMA MinMax (default, baseline); "
             "mse: MSE-optimal range; "
             "ema_quantile: percentile-based clipping; "
             "kl_divergence: KL-divergence optimal truncation + sparse_mode (Round 8); "
             "log2: log2-domain quantization, constant relative error (Round 9, "
             "recommended for log-normal sparse activations near 0)",
    )
    parser.add_argument(
        "--log-base",
        type=float,
        default=2.0,
        help="log base 'a' for sparse log-domain activation quantization when "
             "--act-observer=log2. Default 2.0 (original Log2).",
    )
    parser.add_argument(
        "--sparse-act-mode",
        type=str,
        default="per_tensor",
        choices=["per_tensor", "per_channel"],
        help="activation quantization granularity for lidar/backbone sparse features [N,C]. "
             "per_tensor: one scale for all channels (baseline); "
             "per_channel: one scale per channel C.",
    )
    parser.add_argument(
        "--calib-shuffle",
        action="store_true",
        help="shuffle calibration data for better scene diversity (default: False, sequential).",
    )
    # vtransform 专用激活 Observer 选择
    parser.add_argument(
        "--vtransform-observer",
        type=str,
        default=None,
        choices=["ema_minmax", "mse", "ema_quantile", "kl_divergence"],
        help="activation observer specifically for camera/vtransform module. "
             "kl_divergence: recommended (Round 5, resolves vtransform quantization bottleneck).",
    )
    # Round 8+9：W8A16 控制实验
    parser.add_argument(
        "--no-lidar-act-quant",
        action="store_true",
        help="W8A16 control experiment: quantize lidar/backbone weights only, "
             "keep activations in FP (Round 8). Isolates weight vs activation "
             "quantization contribution to NDS degradation.",
    )
    args, opts = parser.parse_known_args()
    if args.log_base <= 1.0:
        raise ValueError(f"--log-base must be > 1.0, got {args.log_base}")

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
    # 使用训练集做校准（test_mode=True 关闭数据增强，避免过拟合到验证集）
    # PTQ 校准的目的是学习权重/激活的统计分布，应该使用训练集而非验证集
    logger.info("构建校准数据集（使用训练集，test_mode=True，关闭数据增强）...")
    calib_cfg = cfg.data.train.copy()
    calib_cfg.test_mode = True
    calib_dataset = build_dataset(calib_cfg)
    calib_shuffle = args.calib_shuffle
    if calib_shuffle and not hasattr(calib_dataset, 'flag'):
        # mmdet GroupSampler requires dataset.flag; test_mode datasets lack it.
        # Set a dummy flag (all zeros = single group) to allow shuffle.
        calib_dataset.flag = np.zeros(len(calib_dataset), dtype=np.uint8)
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=calib_shuffle,
    )
    logger.info(
        f"校准数据集构建完成：共 {len(calib_dataset)} 帧，"
        f"将使用前 {args.calib_batches} 个 batch，"
        f"{'随机采样（shuffle=True）' if calib_shuffle else '顺序采样（shuffle=False，仅覆盖前几个场景）'}"
    )

    if not args.no_eval:
        logger.info("构建验证数据集（test_mode=True，无数据增强）...")
        val_dataset = build_dataset(cfg.data.test)
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
    ACT_OBSERVER_MAP = {
        "ema_minmax": EMAMinMaxObserver,
        "mse": MSEObserver,
        "ema_quantile": EMAQuantileObserver,
        "kl_divergence": KLDivergenceObserver,
        "log2": SparseLog2FakeQuantize,  # ★ Round 9
    }
    VTRANSFORM_OBSERVER_MAP = {
        "ema_minmax": EMAMinMaxObserver,
        "mse": MSEObserver,
        "ema_quantile": EMAQuantileObserver,
        "kl_divergence": KLDivergenceObserver,
    }
    act_obs_cls = ACT_OBSERVER_MAP[args.act_observer]
    act_obs_name = {
        "ema_minmax": "EMAMinMaxObserver",
        "mse": "MSEObserver (MSE-optimal range)",
        "ema_quantile": "EMAQuantileObserver (percentile clipping)",
        "kl_divergence": "KLDivergenceObserver (KL 散度最优截断)",
        "log2": "SparseLog2FakeQuantize (对数域量化，Round 9)",
    }[args.act_observer]
    if args.act_observer == "log2":
        act_obs_name += f", base={args.log_base:g}"
    # vtransform observer
    vt_obs_cls = None
    vt_obs_name = "EMAMinMaxObserver (默认)"
    if args.vtransform_observer:
        vt_obs_cls = VTRANSFORM_OBSERVER_MAP[args.vtransform_observer]
        vt_obs_name = {
            "ema_minmax": "EMAMinMaxObserver",
            "mse": "MSEObserver (MSE-optimal range)",
            "ema_quantile": "EMAQuantileObserver (percentile clipping)",
            "kl_divergence": "KLDivergenceObserver (KL 散度最优截断)",
        }[args.vtransform_observer]
    sparse_act_per_channel = (args.sparse_act_mode == "per_channel")
    sparse_act_scheme_name = "per-channel" if sparse_act_per_channel else "per-tensor"
    is_log2 = (act_obs_cls is SparseLog2FakeQuantize)
    is_kl_sparse = (act_obs_cls is KLDivergenceObserver)

    logger.info(
        "构建 PTQ 模型（选择性量化，"
        f"稀疏激活: {act_obs_name}, 粒度: {sparse_act_scheme_name}）..."
    )
    if vt_obs_cls:
        logger.info(f"  vtransform 激活 Observer: {vt_obs_name}")
    if getattr(args, 'no_lidar_act_quant', False):
        logger.info("  ★ W8A16 控制实验：lidar/backbone 激活不量化（仅权重量化）")
    model, quant_success, quant_failed = build_ptq_model(
        cfg, logger, skip_modules=args.skip_modules,
        act_observer_cls=act_obs_cls,
        vtransform_observer_cls=vt_obs_cls,
        sparse_act_per_channel=sparse_act_per_channel,
        no_lidar_act_quant=getattr(args, 'no_lidar_act_quant', False),
        log_base=args.log_base,
    )
    model = MMDataParallel(model.cuda(), device_ids=[0])
    logger.info("模型已移动到 GPU（MMDataParallel）。")

    # ------------------------------------------------------------------
    # Step 2.5 (可选): LWC — Learnable Weight Clipping
    # ------------------------------------------------------------------
    if args.lwc:
        logger.info("启用 LWC (Learnable Weight Clipping)，优化稀疏卷积权重截断范围...")
        optimize_lwc_sparse(
            model, logger,
            lr=args.lwc_lr,
            num_iters=args.lwc_iters,
        )

    # ------------------------------------------------------------------
    # ★ 量化结果摘要（校准开始前，请确认后继续）
    # ------------------------------------------------------------------
    _EXPECTED_QUANT = {k for k, _ in _QUANTIZABLE_SUBMODULE_KEYS} | {"heads/object"}
    _skipped_set = set(args.skip_modules)
    total_possible = len(_EXPECTED_QUANT - _skipped_set)
    coverage_pct = len(quant_success) / max(total_possible, 1) * 100
    w8a16_flag = getattr(args, 'no_lidar_act_quant', False)

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║             ★ 量化摘要 — 校准即将开始，请检查！ ★              ║")
    logger.info("╠══════════════════════════════════════════════════════════════╣")
    logger.info(
        f"║  成功量化: {len(quant_success):>2d} 个模块  →  {', '.join(quant_success) if quant_success else '(无)'}")
    logger.info(
        f"║  量化失败: {len(quant_failed):>2d} 个模块  →  {', '.join(quant_failed) if quant_failed else '(无)'}")
    logger.info(f"║  主动跳过: {', '.join(args.skip_modules) if args.skip_modules else '(无)'}")
    logger.info(f"║  覆盖率  : {coverage_pct:.0f}%  ({len(quant_success)}/{total_possible} 个可量化模块)")
    logger.info(f"║  稀疏激活: {act_obs_name}{' [sparse_aware=ON]' if is_kl_sparse else ''}")
    if is_log2:
        logger.info(f"║  Log底数 : a={args.log_base:g} (a=2 等价原始 Log2)")
    logger.info(f"║  稀疏粒度: {sparse_act_scheme_name}")
    logger.info(f"║  VT激活  : {vt_obs_name}")
    logger.info(f"║  W8A16   : {'ON (lidar激活保持FP，仅权重量化)' if w8a16_flag else 'OFF (正常W8A8)'}")
    calib_desc = f"{args.calib_batches} batch, {'shuffle=True（多场景）' if args.calib_shuffle else 'shuffle=False（顺序，仅前几场景）'}"
    logger.info(f"║  校准配置: {calib_desc}")
    if args.lwc:
        logger.info(f"║  LWC     : ON (lr={args.lwc_lr}, iters={args.lwc_iters})")
    if quant_failed:
        logger.warning("║")
        logger.warning("║  ⚠️  有模块量化失败！如果结果不符合预期，请 Ctrl+C 停止。")
        logger.warning("║  ⚠️  失败模块将以 FP32 运行，不影响正确性，但会降低量化覆盖率。")
    else:
        logger.info("║  ✅  所有预期模块均量化成功！")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")

    if quant_failed:
        logger.warning(f"⏳ 5 秒后自动继续校准...（如需停止请按 Ctrl+C）")
        for i in range(5, 0, -1):
            logger.warning(f"   继续倒计时: {i}s ...")
            time.sleep(1)
    logger.info("→ 开始校准，预计耗时较长，请勿中断...")
    logger.info("")

    # ------------------------------------------------------------------
    # Step 3: MinMax 校准
    # ------------------------------------------------------------------
    logger.info(
        f"MinMax 校准阶段：{args.calib_batches} 个 batch，{'随机采样' if args.calib_shuffle else '顺序采样（仅前几个场景）'}")
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
    meta = {
        "ptq_method": "MinMax" + ("+LWC" if args.lwc else ""),
        "backend": "TensorRT",
        "sparse_act_observer": args.act_observer,
        "sparse_log_base": args.log_base if args.act_observer == "log2" else None,
        "sparse_act_mode": args.sparse_act_mode,
        "vtransform_act_observer": args.vtransform_observer or "ema_minmax",
        "quantized_modules": [k for k, _ in _QUANTIZABLE_SUBMODULE_KEYS]
                             + ["heads/*"],
        "skipped_modules": [
            "camera/vtransform",
            "lidar/voxelize",
            "lidar/backbone (SparseEncoder)",
        ],
    }
    if args.lwc:
        meta["lwc"] = {"lr": args.lwc_lr, "iters": args.lwc_iters}
    torch.save(
        {"state_dict": inner_model.state_dict(), "meta": meta},
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
