#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：可视化 LiDAR backbone (SparseEncoder) 权重和激活分布
=========================================================
在选择量化策略之前，分析各层的统计特性：
  - 离群点（outlier）比例
  - 动态范围浪费程度（max vs p99.9 的比值）
  - 通道间尺度差异（per-channel max 分布）
  - 激活分布形状（是否适合对称量化）

用法：
    python tools/diag_lidar_distribution.py \\
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\
        pretrained/bevfusion-det.pth \\
        [--num-batches 10] \\
        [--output-dir results_vis/lidar_diag]
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet3d.ops.spconv.conv import SparseConvolution
from mmdet3d.ops.spconv.structure import SparseConvTensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Optional


# ============================================================================
# 分布收集器
# ============================================================================

class LayerStats:
    """收集单层的权重和激活统计数据（跨多个 batch 累积）。"""

    MAX_SAMPLES = 200_000  # 最多保留的激活采样数

    def __init__(self, name: str):
        self.name = name
        self.weight_vals: Optional[np.ndarray] = None   # 收集一次权重就够了
        self.act_samples: List[np.ndarray] = []          # 每 batch 采样一批激活
        self._act_count = 0

    # ------------------------------------------------------------------
    # 权重收集
    # ------------------------------------------------------------------
    def collect_weight(self, weight: torch.Tensor):
        if self.weight_vals is None:
            self.weight_vals = weight.detach().cpu().float().numpy().flatten()

    # ------------------------------------------------------------------
    # 激活收集（通过 hook 调用）
    # ------------------------------------------------------------------
    def collect_act(self, feat: torch.Tensor):
        """feat: [N_voxels, C] 或任意 dense tensor（已从 SparseConvTensor.features 提取）"""
        if self._act_count >= self.MAX_SAMPLES:
            return
        flat = feat.detach().cpu().float().numpy().flatten()
        remaining = self.MAX_SAMPLES - self._act_count
        if flat.size > remaining:
            idx = np.random.choice(flat.size, remaining, replace=False)
            flat = flat[idx]
        self.act_samples.append(flat)
        self._act_count += flat.size

    @property
    def act_vals(self) -> Optional[np.ndarray]:
        if not self.act_samples:
            return None
        return np.concatenate(self.act_samples)


# ============================================================================
# 统计计算
# ============================================================================

def compute_stats(vals: np.ndarray) -> dict:
    """计算关键统计量，用于判断量化友好性。"""
    abs_vals = np.abs(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    p999 = float(np.percentile(abs_vals, 99.9))
    p9999 = float(np.percentile(abs_vals, 99.99))
    abs_max = float(np.max(abs_vals))

    # 离群点比例：|x| > mean + 3σ
    outlier_3std = float(np.mean(np.abs(vals - mean) > 3 * std) * 100)

    # 动态范围浪费：用 max 而非 p99.9 来设定量化范围时，分辨率损失多少
    # range_waste = 1 - p99.9 / abs_max，越高说明 MinMax 越浪费
    range_waste = float(1.0 - p999 / abs_max) if abs_max > 1e-8 else 0.0

    return {
        "mean": mean, "std": std, "min": mn, "max": mx,
        "abs_max": abs_max, "p99.9": p999, "p99.99": p9999,
        "outlier_3std%": outlier_3std,
        "range_waste%": range_waste * 100,
    }


# ============================================================================
# 绘图
# ============================================================================

_COLORS = {"weight": "#4C72B0", "act": "#DD8452"}


def _plot_hist_with_stats(ax, vals, color, title, xlim=None):
    """在 ax 上绘制直方图 + 垂直线标注关键分位点。"""
    abs_vals = np.abs(vals)
    p999 = np.percentile(abs_vals, 99.9)
    abs_max = np.max(abs_vals)

    counts, edges, _ = ax.hist(vals, bins=200, color=color, alpha=0.75,
                               density=True, rwidth=0.9)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=7)

    if xlim is not None:
        ax.set_xlim(xlim)

    # 标注 ±p99.9 和 ±abs_max
    for v, ls, label in [
        (p999,    "--", f"p99.9={p999:.3f}"),
        (abs_max, "-",  f"max={abs_max:.3f}"),
    ]:
        ax.axvline(v,  color="red",  linestyle=ls, linewidth=1.2, alpha=0.9, label=label)
        ax.axvline(-v, color="red",  linestyle=ls, linewidth=1.2, alpha=0.9)
    ax.legend(fontsize=6, loc="upper right")


def _stats_text(stats: dict) -> str:
    return (
        f"mean={stats['mean']:+.4f}  std={stats['std']:.4f}\n"
        f"min={stats['min']:.4f}  max={stats['max']:.4f}\n"
        f"p99.9={stats['p99.9']:.4f}  p99.99={stats['p99.99']:.4f}\n"
        f"outlier(>3σ): {stats['outlier_3std%']:.2f}%\n"
        f"range_waste:  {stats['range_waste%']:.1f}%"
    )


def plot_layer(layer: LayerStats, out_path: str):
    """为单层生成 2×2 的子图：权重直方图、激活直方图（全范围 + 裁剪）、统计文本。"""
    w = layer.weight_vals
    a = layer.act_vals

    n_cols = 2 + (1 if a is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    fig.suptitle(f"Layer: {layer.name}", fontsize=11, fontweight="bold")

    col = 0

    # ---- 权重直方图 ----
    if w is not None:
        ws = compute_stats(w)
        _plot_hist_with_stats(axes[col], w, _COLORS["weight"],
                              f"Weights  (n={w.size:,})")
        axes[col].text(0.02, 0.98, _stats_text(ws),
                       transform=axes[col].transAxes, fontsize=6,
                       va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        col += 1

    # ---- 激活直方图（全范围） ----
    if a is not None:
        as_ = compute_stats(a)
        _plot_hist_with_stats(axes[col], a, _COLORS["act"],
                              f"Activations  (n={a.size:,})")
        axes[col].text(0.02, 0.98, _stats_text(as_),
                       transform=axes[col].transAxes, fontsize=6,
                       va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        col += 1

        # ---- 激活直方图（裁剪到 p99.9 范围，看主体分布） ----
        p999 = as_["p99.9"]
        _plot_hist_with_stats(axes[col], a, _COLORS["act"],
                              f"Activations (zoom ≤p99.9={p999:.3f})",
                              xlim=(-p999 * 1.1, p999 * 1.1))
        col += 1

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_summary(layers: List[LayerStats], out_dir: str):
    """生成汇总对比图：各层的离群点比例 + 动态范围浪费。"""
    names_w, ow, rw = [], [], []
    names_a, oa, ra = [], [], []

    for l in layers:
        if l.weight_vals is not None:
            ws = compute_stats(l.weight_vals)
            names_w.append(l.name.split(".")[-1] or l.name)
            ow.append(ws["outlier_3std%"])
            rw.append(ws["range_waste%"])
        if l.act_vals is not None:
            as_ = compute_stats(l.act_vals)
            names_a.append(l.name.split(".")[-1] or l.name)
            oa.append(as_["outlier_3std%"])
            ra.append(as_["range_waste%"])

    if not names_w and not names_a:
        return

    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(names_w) * 0.6 + 4), 9))
    fig.suptitle("LiDAR Backbone — Quantization Friendliness Summary", fontsize=13, fontweight="bold")

    def _bar(ax, names, vals, title, ylabel, color, threshold=None):
        x = range(len(names))
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                       label=f"threshold={threshold}%")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    _bar(axes[0, 0], names_w, ow,
         "Weights: outlier ratio (|x|>3σ)",
         "% values", "#4C72B0", threshold=1.0)
    _bar(axes[0, 1], names_w, rw,
         "Weights: range waste by MinMax\n(1 - p99.9/max)",
         "% range wasted", "#4C72B0", threshold=10.0)
    _bar(axes[1, 0], names_a, oa,
         "Activations: outlier ratio (|x|>3σ)",
         "% values", "#DD8452", threshold=1.0)
    _bar(axes[1, 1], names_a, ra,
         "Activations: range waste by MinMax\n(1 - p99.9/max)",
         "% range wasted", "#DD8452", threshold=10.0)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] Summary plot → {os.path.join(out_dir, 'summary.png')}")


def plot_channel_range(layers: List[LayerStats], out_dir: str):
    """为每层绘制 per-channel abs-max，揭示通道间尺度差异（影响 per-tensor 量化质量）。"""
    # 只有激活数据是 [N_voxels, C_out] 时才有意义；
    # 这里用 act_vals（已扁平化），通道信息已丢失
    # 改为绘制权重的 per-output-channel L2 norm 分布
    fig_rows = [l for l in layers if l.weight_vals is not None]
    if not fig_rows:
        return

    ncols = 3
    nrows = (len(fig_rows) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("LiDAR Backbone — Per-output-channel Weight L2 Norm\n"
                 "(Spread indicates per-tensor quantization scale mismatch)",
                 fontsize=11, fontweight="bold")

    for i, l in enumerate(fig_rows):
        ax = axes[i]
        # weight shape = [K, K, K, C_in, C_out] → norm over all dims except last
        # We need the original weight tensor here, but we only stored flattened data.
        # Instead, show the distribution of |w| with annotations.
        w = np.abs(l.weight_vals)
        ws = compute_stats(l.weight_vals)
        ax.hist(w, bins=150, color="#4C72B0", alpha=0.8, density=True)
        ax.axvline(ws["p99.9"],  color="red",    linestyle="--", lw=1.2,
                   label=f"p99.9={ws['p99.9']:.3f}")
        ax.axvline(ws["abs_max"], color="darkred", linestyle="-",  lw=1.2,
                   label=f"max={ws['abs_max']:.3f}")
        ax.set_title(l.name, fontsize=8)
        ax.set_xlabel("|weight|", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(len(fig_rows), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, "weight_abs_per_layer.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] Weight |w| per-layer plot → {out}")


# ============================================================================
# 主逻辑
# ============================================================================

def register_hooks(lidar_backbone, stats_map: Dict) -> list:
    """为 SparseEncoder 中所有 SparseConvolution 注册 forward hook。"""
    handles = []
    for name, module in lidar_backbone.named_modules():
        if isinstance(module, SparseConvolution):
            stats = LayerStats(name if name else "conv_root")
            stats.collect_weight(module.weight.data)
            stats_map[name] = stats

            def make_hook(s):
                def hook(mod, inp, out):
                    if isinstance(out, SparseConvTensor) and out.features.numel() > 0:
                        s.collect_act(out.features)
                    elif isinstance(out, torch.Tensor) and out.numel() > 0:
                        s.collect_act(out)
                return hook

            handles.append(module.register_forward_hook(make_hook(stats)))
    return handles


def print_summary_table(stats_map: dict):
    """打印各层统计摘要（可直接 copy 到 report）。"""
    hdr = (
        f"\n{'Layer':<40s} {'W_max':>8s} {'W_p99.9':>8s} "
        f"{'W_waste%':>9s} {'A_max':>8s} {'A_p99.9':>8s} "
        f"{'A_waste%':>9s} {'A_out%':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, l in stats_map.items():
        ws = compute_stats(l.weight_vals) if l.weight_vals is not None else {}
        as_ = compute_stats(l.act_vals) if l.act_vals is not None else {}
        print(
            f"{name:<40s} "
            f"{ws.get('abs_max', float('nan')):8.4f} "
            f"{ws.get('p99.9', float('nan')):8.4f} "
            f"{ws.get('range_waste%', float('nan')):8.1f}% "
            f"{as_.get('abs_max', float('nan')):8.4f} "
            f"{as_.get('p99.9', float('nan')):8.4f} "
            f"{as_.get('range_waste%', float('nan')):8.1f}% "
            f"{as_.get('outlier_3std%', float('nan')):7.2f}%"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose LiDAR backbone weight/activation distributions"
    )
    parser.add_argument("config",  help="BEVFusion config yaml")
    parser.add_argument("checkpoint", help="Pretrained checkpoint (.pth)")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="Number of calibration batches to run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for data loader (default: 1)")
    parser.add_argument("--output-dir", default="results_vis/lidar_diag",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 构建模型
    # ------------------------------------------------------------------
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    logger = get_root_logger()

    # 禁止 build_model 触发网络下载（SwinT pretrained 等）
    cfg.model.pretrained = None
    if hasattr(cfg.model, "encoders"):
        for enc_cfg in cfg.model.encoders.values() if hasattr(cfg.model.encoders, "values") else []:
            if hasattr(enc_cfg, "backbone") and hasattr(enc_cfg.backbone, "init_cfg"):
                enc_cfg.backbone.init_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.eval()

    # ------------------------------------------------------------------
    # 2. 构建数据集
    # ------------------------------------------------------------------
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=0,   # 0 = main-process only，避免 Windows 多进程 pickle 问题
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    model = MMDataParallel(model, device_ids=[0])

    # ------------------------------------------------------------------
    # 3. 注册 hook
    # ------------------------------------------------------------------
    inner = model.module
    lidar_backbone = inner.encoders.lidar.backbone
    stats_map: Dict[str, LayerStats] = {}
    handles = register_hooks(lidar_backbone, stats_map)

    if not stats_map:
        print("[ERROR] No SparseConvolution layers found in lidar backbone!")
        return

    print(f"[diag] Found {len(stats_map)} SparseConvolution layers: "
          + ", ".join(list(stats_map.keys())[:5]) + "...")

    # ------------------------------------------------------------------
    # 4. 前向推理（calibration）
    # ------------------------------------------------------------------
    print(f"[diag] Running {args.num_batches} batches for activation collection...")
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.num_batches:
                break
            model(return_loss=False, rescale=True, **data)
            if (i + 1) % 5 == 0:
                print(f"  batch {i + 1}/{args.num_batches}")

    # hook 使命完成，移除
    for h in handles:
        h.remove()

    # ------------------------------------------------------------------
    # 5. 打印汇总表
    # ------------------------------------------------------------------
    print_summary_table(stats_map)

    # ------------------------------------------------------------------
    # 6. 生成每层详细图
    # ------------------------------------------------------------------
    layer_list = list(stats_map.values())
    print(f"[diag] Generating per-layer plots ({len(layer_list)} layers)...")
    for l in layer_list:
        safe_name = l.name.replace(".", "_").replace("/", "_")
        out_path = os.path.join(args.output_dir, f"layer_{safe_name}.png")
        try:
            plot_layer(l, out_path)
        except Exception as e:
            print(f"  [WARN] Failed to plot {l.name}: {e}")
    print(f"[diag] Per-layer plots saved to {args.output_dir}/")

    # ------------------------------------------------------------------
    # 7. 生成汇总图
    # ------------------------------------------------------------------
    plot_summary(layer_list, args.output_dir)
    plot_channel_range(layer_list, args.output_dir)

    # ------------------------------------------------------------------
    # 8. 解读指导（直接打印到终端）
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  解读指南")
    print("=" * 70)
    print("""
W_waste%  (权重动态范围浪费)
    < 5%  → MinMax 校准足够，权重量化无问题
    5-20% → 考虑 Percentile 或 MSE 校准
    > 20% → 存在严重离群点，强烈建议 MSEObserver 或 AdaRound

A_waste%  (激活动态范围浪费)
    < 5%  → EMAMinMaxObserver 足够
    5-30% → 换 EMAQuantileObserver(threshold=0.9999)
    > 30% → 需要 MSEObserver 或考虑 per-channel 激活量化

A_out%    (激活离群点 |x| > 3σ 的比例)
    < 0.5% → 分布接近高斯，MinMax 合理
    > 1%   → 重尾分布，Percentile/MSE 效果更好
    > 5%   → 分布严重偏斜，考虑 log-scale 量化
""")
    print(f"[diag] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
