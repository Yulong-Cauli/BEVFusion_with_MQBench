#!/usr/bin/env python
"""
Fast Lloyd-Max quantization analysis for LiDAR backbone activations.
Uses GPU-accelerated Lloyd-Max + 100-frame data collection.
Can parallelize across 5 GPUs.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
    python tools/analyze_lloydmax_fast.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --num-samples 100 \
        --output-dir runs/lloydmax_analysis_100 \
        --n-gpus 5
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

mp.set_start_method("spawn", force=True)

sys.path.append(os.getcwd())

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def configure_plot_font():
    # Force a CJK-capable font so Chinese legend/title renders correctly.
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Lloyd-Max analysis")
    parser.add_argument("config", help="model config file path")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--num-samples", type=int, default=100, help="number of samples")
    parser.add_argument("--n-gpus", type=int, default=5, help="number of GPUs for layer parallelization")
    parser.add_argument("--n-levels", type=int, default=256, help="quantization levels")
    parser.add_argument("--n-iters", type=int, default=50, help="Lloyd-Max iterations")
    parser.add_argument("--clip-pct", type=float, default=0.5, help="percentile clip for outlier removal")
    parser.add_argument("--output-dir", default="runs/lloydmax_analysis_100", help="output directory")
    parser.add_argument("--skip-plots", action="store_true", help="skip generating PNG figures")
    return parser.parse_args()


def build_model_and_loader(cfg):
    cfg = Config(recursive_eval(cfg._cfg_dict), filename=cfg.filename)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model = model.cuda().eval()

    calib_cfg = cfg.data.train.copy()
    calib_cfg.test_mode = True
    calib_dataset = build_dataset(calib_cfg)
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=True,  # random shuffle
    )
    return model, calib_loader


def register_hooks(backbone, activations):
    hooks = []
    target_names = []
    conv_classes = ("SubMConv3d", "SparseConv3d")

    def get_hook(name):
        def hook(module, input, output):
            feats = output.features.detach().cpu().float().numpy().reshape(-1)
            activations.setdefault(name, []).append(feats)
        return hook

    for name, module in backbone.named_modules():
        if type(module).__name__ in conv_classes:
            target_names.append(name)
            hooks.append(module.register_forward_hook(get_hook(name)))
    return hooks, target_names


def collect_activations(model, loader, num_samples):
    raw_model = model
    wrapped = MMDataParallel(raw_model, device_ids=[0])
    backbone = raw_model.encoders.lidar.backbone
    activations = {}
    hooks, names = register_hooks(backbone, activations)

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_samples:
                break
            try:
                wrapped(return_loss=False, rescale=True, **data)
            except Exception:
                if isinstance(data, list) and len(data) == 1:
                    wrapped(return_loss=False, rescale=True, **data[0])
                else:
                    raise

    for h in hooks:
        h.remove()

    concat = {name: np.concatenate(activations[name]) for name in names if name in activations}
    return concat



def lloyd_max_cpu(samples, n_levels=256, n_iters=50):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0 or (samples.max() - samples.min()) < 1e-12:
        levels = np.full(n_levels, samples.mean() if samples.size else 0.0)
        boundaries = np.linspace(samples.min() - 1.0, samples.max() + 1.0, n_levels + 1)
        return levels, boundaries
    max_samples = 10_000_000
    if samples.size > max_samples:
        perm = np.random.permutation(samples.size)[:max_samples]
        samples = samples[perm]
    boundaries = np.linspace(samples.min(), samples.max(), n_levels + 1)
    for _ in range(n_iters):
        idx = np.digitize(samples, boundaries[1:-1])
        counts = np.bincount(idx, minlength=n_levels)
        sums = np.bincount(idx, weights=samples, minlength=n_levels)
        levels = np.where(counts > 0, sums / counts, (boundaries[:-1] + boundaries[1:]) / 2.0)
        new_boundaries = boundaries.copy()
        new_boundaries[1:-1] = (levels[:-1] + levels[1:]) / 2.0
        if np.max(np.abs(new_boundaries - boundaries)) < 1e-7:
            boundaries = new_boundaries
            break
        boundaries = new_boundaries
    idx = np.digitize(samples, boundaries[1:-1])
    counts = np.bincount(idx, minlength=n_levels)
    sums = np.bincount(idx, weights=samples, minlength=n_levels)
    levels = np.where(counts > 0, sums / counts, (boundaries[:-1] + boundaries[1:]) / 2.0)
    return levels, boundaries


def rel_mse_uniform_cpu(x, n_levels=256):
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-6
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return 0.0
    step = (xmax - xmin) / (n_levels - 1)
    q = np.round((x - xmin) / step) * step + xmin
    q = np.clip(q, xmin, xmax)
    q = np.where(np.abs(x) < eps, 0.0, q)
    return float(np.mean(((x - q) / (np.abs(x) + eps)) ** 2))


def mse_uniform_cpu(x, n_levels=256):
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return 0.0
    step = (xmax - xmin) / (n_levels - 1)
    q = np.round((x - xmin) / step) * step + xmin
    q = np.clip(q, xmin, xmax)
    q = np.where(np.abs(x) < 1e-6, 0.0, q)
    return float(np.mean((x - q) ** 2))


def rel_mse_log2_cpu(x, n_levels=256):
    x_t = np.asarray(x, dtype=np.float64)
    eps = 1e-6
    best = float("inf")
    best_base = None
    for base in np.linspace(-6, 2, 41):
        x_dq = np.sign(x_t) * np.power(2.0, np.round(np.log2(np.maximum(np.abs(x_t), eps)) - base) + base)
        x_dq = np.where(np.abs(x_t) < eps, 0.0, x_dq)
        rmse = float(np.mean(((x_t - x_dq) / (np.abs(x_t) + eps)) ** 2))
        if rmse < best:
            best = rmse
            best_base = base
    return best, best_base


def mse_log2_cpu(x, best_base):
    x_t = np.asarray(x, dtype=np.float64)
    eps = 1e-6
    x_dq = np.sign(x_t) * np.power(2.0, np.round(np.log2(np.maximum(np.abs(x_t), eps)) - best_base) + best_base)
    x_dq = np.where(np.abs(x_t) < eps, 0.0, x_dq)
    return float(np.mean((x_t - x_dq) ** 2))


def rel_mse_lloydmax_cpu(x, levels):
    x = np.asarray(x, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    eps = 1e-6
    chunk_size = 1_000_000
    total = x.size
    if total == 0:
        return 0.0
    sq_sum = 0.0
    for i in range(0, total, chunk_size):
        chunk = x[i:i + chunk_size]
        idx = np.argmin(np.abs(chunk[:, None] - levels[None, :]), axis=1)
        q = levels[idx]
        q = np.where(np.abs(chunk) < eps, 0.0, q)
        sq_sum += float(np.sum(((chunk - q) / (np.abs(chunk) + eps)) ** 2))
    return sq_sum / total


def mse_lloydmax_cpu(x, levels):
    x = np.asarray(x, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    chunk_size = 1_000_000
    total = x.size
    if total == 0:
        return 0.0
    sq_sum = 0.0
    for i in range(0, total, chunk_size):
        chunk = x[i:i + chunk_size]
        idx = np.argmin(np.abs(chunk[:, None] - levels[None, :]), axis=1)
        q = levels[idx]
        q = np.where(np.abs(chunk) < 1e-6, 0.0, q)
        sq_sum += float(np.sum((chunk - q) ** 2))
    return sq_sum / total


def rel_mse_log_a_grid_cpu(x, n_levels=256, bases=(1.25, np.sqrt(2), 1.5, 2.0, 3.0, 4.0, np.e, 8.0, 10.0, 16.0)):
    x_t = np.asarray(x, dtype=np.float64)
    eps = 1e-6
    bases_arr = np.asarray(bases, dtype=np.float64)
    offsets = np.linspace(-6, 2, 41)
    B = len(bases_arr)
    S = len(offsets)
    sum_sq = np.zeros((B, S))
    count = 0
    chunk_size = 500_000
    total = x_t.size
    for i in range(0, total, chunk_size):
        chunk = x_t[i:i + chunk_size]
        C = chunk.size
        log_a_x = np.log(np.maximum(np.abs(chunk), eps)) / np.log(bases_arr)[:, None, None]
        q_int = np.round(log_a_x - offsets[None, :, None])
        x_dq = np.sign(chunk) * np.power(bases_arr[:, None, None], q_int + offsets[None, :, None])
        x_dq = np.where(np.abs(chunk) < eps, 0.0, x_dq)
        sq = ((chunk - x_dq) / (np.abs(chunk) + eps)) ** 2
        sum_sq += sq.sum(axis=2)
        count += C
    rel_mse = sum_sq / count
    best_per_base = rel_mse.min(axis=1)
    best_idx = rel_mse.argmin(axis=1)
    best_b = int(best_per_base.argmin())
    best_offset = offsets[best_idx[best_b]]
    best_rel = float(best_per_base[best_b])
    best_a = float(bases_arr[best_b])
    details = {}
    all_rel_mses = {}
    for bi, a_val in enumerate(bases_arr.tolist()):
        details[a_val] = {
            "rel_mse": float(best_per_base[bi]),
            "offset": float(offsets[best_idx[bi]]),
        }
        all_rel_mses[a_val] = float(best_per_base[bi])
    return best_rel, best_a, best_offset, details, all_rel_mses


def _pick_rel(rel_map, target):
    if not rel_map:
        return float("nan")
    key = min(rel_map.keys(), key=lambda k: abs(float(k) - float(target)))
    return float(rel_map[key])


def save_log_base_comparison_plots(results, output_dir):
    configure_plot_font()
    # 用户指定：仅保留 sqrt2, 2, 4, 8, 16
    bases = [float(np.sqrt(2)), 2.0, 4.0, 8.0, 16.0]
    xs = np.array(bases, dtype=np.float64)
    ys = []
    for b in bases:
        vals = []
        for r in results:
            rel_map = r.get("all_rel_mses", {})
            vals.append(_pick_rel(rel_map, b))
        ys.append(np.nanmean(vals))
    ys = np.array(ys, dtype=np.float64)

    # 1) 均匀刻度
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["sqrt2", "2", "4", "8", "16"])
    ax.set_xlabel("底数a")
    ax.set_ylabel("相对均方误差（Rel-MSE）")
    ax.set_title("LogA 底数与 Rel-MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "log_base_comparison_linear.png"), dpi=150)
    plt.close(fig)

    # 2) 对数刻度（base=2）
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1.0, 18.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(["sqrt2", "2", "4", "8", "16"])
    ax.set_xlabel("底数a（log2刻度）")
    ax.set_ylabel("相对均方误差（Rel-MSE）")
    ax.set_title("LogA 底数与 Rel-MSE")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "log_base_comparison_log2.png"), dpi=150)
    # 兼容旧文件名：默认指向对数刻度版本
    fig.savefig(os.path.join(output_dir, "log_base_comparison.png"), dpi=150)
    plt.close(fig)


def process_one_layer(args_dict):
    name = args_dict["name"]
    x = args_dict["x"]
    x = x[np.isfinite(x)]
    n_levels = args_dict["n_levels"]
    n_iters = args_dict["n_iters"]
    clip_pct = args_dict["clip_pct"]
    output_dir = args_dict["output_dir"]

    mu, b = stats.laplace.fit(x)

    low = float(np.percentile(x, clip_pct))
    high = float(np.percentile(x, 100.0 - clip_pct))
    x_clip = np.clip(x, low, high)

    levels, _ = lloyd_max_cpu(x_clip, n_levels, n_iters)

    rel_l2, best_base = rel_mse_log2_cpu(x_clip, n_levels)
    rel_lm = rel_mse_lloydmax_cpu(x_clip, levels)
    rel_u = rel_mse_uniform_cpu(x_clip, n_levels)

    mse_u = mse_uniform_cpu(x_clip, n_levels)
    mse_l2 = mse_log2_cpu(x_clip, best_base)
    mse_lm = mse_lloydmax_cpu(x_clip, levels)

    best_rel_a, best_a, best_offset, a_details, all_rel_mses = rel_mse_log_a_grid_cpu(x_clip, n_levels)

    qmin = -(2 ** (8 - 1) - 1)
    qmax = (2 ** (8 - 1) - 1)
    k = np.arange(qmin, qmax + 1)
    log2_levels = np.concatenate([-np.power(2.0, k + best_base)[::-1], [0], np.power(2.0, k + best_base)])
    log2_levels = log2_levels[(log2_levels >= low) & (log2_levels <= high)]

    if not args_dict.get("skip_plots", False):
        configure_plot_font()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(x_clip, bins=200, density=True, alpha=0.4, color="steelblue", label="激活直方图")
        xs = np.linspace(low, high, 2000)
        ax.plot(xs, stats.laplace.pdf(xs, loc=mu, scale=b), color="navy", lw=2, label=f"拉普拉斯拟合 (μ={mu:.3f}, b={b:.3f})")

        ax.vlines(log2_levels, 0, ax.get_ylim()[1] * 0.30, colors="green", lw=1.0, alpha=0.7, label="Log2 量化级")

        u_levels = np.linspace(low, high, 64)
        ax.vlines(u_levels, 0, ax.get_ylim()[1] * 0.12, colors="red", lw=1.0, alpha=0.7, label="均匀量化级")

        ax.set_xlim(low, high)
        ax.set_xlabel("激活值")
        ax.set_ylabel("概率密度")
        ax.set_title(f"层: {name}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{name.replace('.', '_')}.png"), dpi=150)
        plt.close(fig)

    return {
        "name": name,
        "N": int(x.size),
        "b": float(b),
        "low": float(low),
        "high": float(high),
        "mse_uniform": mse_u,
        "mse_log2": mse_l2,
        "mse_lloydmax": mse_lm,
        "rel_uniform": rel_u,
        "rel_log2": rel_l2,
        "rel_lloydmax": rel_lm,
        "best_base": float(best_base),
        "best_a": float(best_a),
        "best_a_rel": float(best_rel_a),
        "best_a_offset": float(best_offset),
        "all_rel_mses": all_rel_mses,
        "levels": levels,
        "log2_levels": log2_levels,
    }

def save_comparison_rug_plot(name, low, high, levels_uniform, levels_log2, levels_lloydmax, output_dir):
    fig, ax = plt.subplots(figsize=(12, 3))
    y_map = {"Lloyd-Max": 0, "Log2": 1, "Uniform": 2}
    colors = {"Lloyd-Max": "black", "Log2": "green", "Uniform": "red"}
    for label, levels in [("Lloyd-Max", levels_lloydmax), ("Log2", levels_log2), ("Uniform", levels_uniform)]:
        y = y_map[label]
        # keep only visible levels
        vis = levels[(levels >= low) & (levels <= high)]
        ax.scatter(vis, [y] * len(vis), marker='|', s=200, color=colors[label], label=label, alpha=0.8)
    ax.set_xlim(low, high)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Lloyd-Max", "Log2", "Uniform"])
    ax.set_xlabel("Activation value")
    ax.set_title(f"Comparison: {name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name.replace('.', '_')}_compare.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configure_plot_font()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    print("Building model and dataloader...")
    model, loader = build_model_and_loader(cfg)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    print(f"Model loaded from {args.ckpt}")

    print(f"Collecting activations from {args.num_samples} samples...")
    activations = collect_activations(model, loader, args.num_samples)
    print(f"Collected {len(activations)} layers, total activations per layer: {sum(v.size for v in activations.values())}")
    del model, loader
    torch.cuda.empty_cache()

    # Prepare work items
    work_items = []
    for i, (name, x) in enumerate(activations.items()):
        work_items.append({
            "name": name,
            "x": x,
            "n_levels": args.n_levels,
            "n_iters": args.n_iters,
            "clip_pct": args.clip_pct,
            "gpu_id": i % args.n_gpus,
            "output_dir": args.output_dir,
            "skip_plots": args.skip_plots,
        })

    print(f"Processing {len(work_items)} layers on {args.n_gpus} GPUs using imap_unordered...")
    start = time.time()
    with mp.Pool(processes=args.n_gpus) as pool:
        results = list(pool.imap_unordered(process_one_layer, work_items))
    print(f"Layer processing done in {time.time() - start:.1f}s")

    # sort by original layer order
    name_to_idx = {name: i for i, name in enumerate(activations.keys())}
    results.sort(key=lambda r: name_to_idx[r["name"]])

    if not args.skip_plots:
        # Select representative layer (largest Lloyd-Max Rel-MSE improvement)
        rep_layer = max(results, key=lambda r: (r["rel_uniform"] - r["rel_lloydmax"]) / (r["rel_uniform"] + 1e-12))
        rep_name = rep_layer["name"]
        print(f"Representative layer for rug plot: {rep_name}")
        save_comparison_rug_plot(
        rep_name,
        rep_layer["low"],
        rep_layer["high"],
        np.linspace(rep_layer["low"], rep_layer["high"], 64),
        rep_layer["log2_levels"],
        rep_layer["levels"],
        args.output_dir,
    )
        with open(os.path.join(args.output_dir, "representative_layer.txt"), "w") as f:
            f.write(rep_name + "\n")

    # Summary text
    header = (
        f"{'Layer':<40} {'N':>10} {'b':>10} "
        f"{'MSE_U':>10} {'MSE_L2':>10} {'MSE_LM':>10} "
        f"{'RelU':>10} {'RelL2':>10} {'RelLM':>10} "
        f"{'ImpMSE_L2':>10} {'ImpMSE_LM':>10} {'ImpRel_L2':>10} {'ImpRel_LM':>10} {'Best_a':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    csv_rows = []
    csv_header = [
        "layer", "N", "b",
        "mse_uniform", "mse_log2", "mse_lloydmax",
        "rel_uniform", "rel_log2", "rel_lloydmax",
        "imp_mse_log2_pct", "imp_mse_lloydmax_pct",
        "imp_rel_log2_pct", "imp_rel_lloydmax_pct",
        "best_a", "best_a_rel", "best_a_offset",
        "rel_a_1.25", "rel_a_sqrt2", "rel_a_1.50", "rel_a_2.00",
        "rel_a_3.00", "rel_a_4.00", "rel_a_e", "rel_a_8.00", "rel_a_10", "rel_a_16.00",
    ]
    csv_rows.append(csv_header)

    for r in results:
        mse_u = r["mse_uniform"]
        mse_l2 = r["mse_log2"]
        mse_lm = r["mse_lloydmax"]
        rel_u = r["rel_uniform"]
        rel_l2 = r["rel_log2"]
        rel_lm = r["rel_lloydmax"]
        imp_mse_l2 = (mse_u - mse_l2) / (mse_u + 1e-12) * 100.0
        imp_mse_lm = (mse_u - mse_lm) / (mse_u + 1e-12) * 100.0
        imp_rel_l2 = (rel_u - rel_l2) / (rel_u + 1e-12) * 100.0
        imp_rel_lm = (rel_u - rel_lm) / (rel_u + 1e-12) * 100.0
        lines.append(
            f"{r['name']:<40} {r['N']:>10} {r['b']:>10.4f} "
            f"{mse_u:>10.2e} {mse_l2:>10.2e} {mse_lm:>10.2e} "
            f"{rel_u:>10.2e} {rel_l2:>10.2e} {rel_lm:>10.2e} "
            f"{imp_mse_l2:>9.2f}% {imp_mse_lm:>9.2f}% {imp_rel_l2:>9.2f}% {imp_rel_lm:>9.2f}% {r['best_a']:>8.3f}"
        )
        rel_a = r.get("all_rel_mses", {})
        csv_rows.append([
            r["name"], r["N"], f"{r['b']:.4f}",
            f"{mse_u:.6e}", f"{mse_l2:.6e}", f"{mse_lm:.6e}",
            f"{rel_u:.6e}", f"{rel_l2:.6e}", f"{rel_lm:.6e}",
            f"{imp_mse_l2:.2f}", f"{imp_mse_lm:.2f}",
            f"{imp_rel_l2:.2f}", f"{imp_rel_lm:.2f}",
            f"{r['best_a']:.3f}", f"{r['best_a_rel']:.6e}", f"{r['best_a_offset']:.3f}",
            f"{rel_a.get(1.25, float('nan')):.6e}",
            f"{rel_a.get(float(np.sqrt(2)), float('nan')):.6e}",
            f"{rel_a.get(1.5, float('nan')):.6e}",
            f"{rel_a.get(2.0, float('nan')):.6e}",
            f"{rel_a.get(3.0, float('nan')):.6e}",
            f"{rel_a.get(4.0, float('nan')):.6e}",
            f"{rel_a.get(float(np.e), float('nan')):.6e}",
            f"{rel_a.get(8.0, float('nan')):.6e}",
            f"{rel_a.get(10.0, float('nan')):.6e}",
            f"{rel_a.get(16.0, float('nan')):.6e}",
        ])

    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")

    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    save_log_base_comparison_plots(results, args.output_dir)

    print(f"Done. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
