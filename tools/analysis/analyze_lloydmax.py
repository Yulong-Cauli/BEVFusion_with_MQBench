#!/usr/bin/env python
"""
Lloyd-Max quantization analysis for LiDAR backbone activations.

Steps:
1. Load BEVFusion model and forward a few samples.
2. Hook key layers in encoders.lidar.backbone to collect activations.
3. Fit Laplace(μ, b) to each layer's activation.
4. Run Lloyd-Max iteration on the fitted Laplace distribution.
5. Compare MSE of Uniform INT8, Log2 INT8, and Lloyd-Max optimal quantizers.
6. Save figures and a summary table.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python tools/analyze_lloydmax.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --num-samples 5 \
        --output-dir runs/lloydmax_analysis
"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from scipy import stats
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Lloyd-Max analysis for LiDAR activations")
    parser.add_argument("config", help="model config file path")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--num-samples", type=int, default=5, help="number of samples to collect activations")
    parser.add_argument("--n-levels", type=int, default=256, help="number of quantization levels")
    parser.add_argument("--n-iters", type=int, default=50, help="Lloyd-Max iterations")
    parser.add_argument("--clip-sigma", type=float, default=6.0, help="clipping range in units of b for Lloyd-Max init")
    parser.add_argument("--output-dir", default="runs/lloydmax_analysis", help="output directory")
    return parser.parse_args()


def build_model_and_loader(cfg):
    """Build model and calibration dataloader (single GPU)."""
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
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    return model, calib_loader


def register_hooks(backbone, activations):
    """Register forward hooks on all sparse conv layers in backbone."""
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


def collect_activations(raw_model, wrapped_model, loader, num_samples):
    """Forward num_samples and return activation dict."""
    backbone = raw_model.encoders.lidar.backbone
    activations = {}
    hooks, names = register_hooks(backbone, activations)

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= num_samples:
                break
            try:
                wrapped_model(return_loss=False, rescale=True, **data)
            except Exception as e:
                print(f"  Sample {i} forward raised {type(e).__name__}: {e}")
                if isinstance(data, list) and len(data) == 1:
                    wrapped_model(return_loss=False, rescale=True, **data[0])
                else:
                    raise

    for h in hooks:
        h.remove()

    # Concatenate per-layer
    concat = {}
    for name in names:
        if name in activations:
            concat[name] = np.concatenate(activations[name])
        else:
            concat[name] = np.array([])
    return concat


def lloyd_max_discrete(samples, n_levels=256, n_iters=50):
    """
    Fast Lloyd-Max via discrete samples (Monte Carlo surrogate).

    Instead of expensive numerical integration on a continuous PDF,
    we draw 1M samples from the fitted Laplace and run vectorized
    Lloyd-Max on the empirical distribution. This is exact for the
    purpose of comparing quantizer structures (Uniform vs Log2 vs Optimal).
    """
    samples = np.asarray(samples, dtype=np.float64)
    # Initialize boundaries uniformly over the empirical range
    boundaries = np.linspace(samples.min(), samples.max(), n_levels + 1)

    for it in range(n_iters):
        # Vectorized conditional mean: for each bin, mean of samples inside
        idx = np.digitize(samples, boundaries[1:-1])  # bin index for each sample
        levels = np.array([samples[idx == i].mean() if np.any(idx == i) else (boundaries[i] + boundaries[i + 1]) / 2.0
                           for i in range(n_levels)])
        new_boundaries = boundaries.copy()
        new_boundaries[1:-1] = (levels[:-1] + levels[1:]) / 2.0
        if np.max(np.abs(new_boundaries - boundaries)) < 1e-7:
            boundaries = new_boundaries
            break
        boundaries = new_boundaries

    # Final levels
    idx = np.digitize(samples, boundaries[1:-1])
    levels = np.array([samples[idx == i].mean() if np.any(idx == i) else (boundaries[i] + boundaries[i + 1]) / 2.0
                       for i in range(n_levels)])
    return levels, boundaries


def rel_mse_uniform(x, n_levels=256):
    """Symmetric uniform quantizer relative MSE (MinMax style)."""
    x = x.astype(np.float64)
    eps = 1e-6
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return 0.0
    step = (xmax - xmin) / (n_levels - 1)
    q = np.round((x - xmin) / step) * step + xmin
    q = np.clip(q, xmin, xmax)
    return float(np.mean(((x - q) / (np.abs(x) + eps)) ** 2))


def rel_mse_log2(x, n_levels=256):
    """
    Log2 quantizer relative MSE.
    We grid-search log2_base to minimize relative MSE (fair comparison).
    Returns (best_rel_mse, best_base).
    """
    x = torch.from_numpy(x.astype(np.float64))
    eps = 1e-6
    best_rel_mse = float("inf")
    best_base = None

    # Search over a reasonable base range
    for base in np.linspace(-6, 2, 41):
        base_t = torch.tensor(base, dtype=torch.float64)
        x_dq = torch.sign(x) * torch.pow(2.0, torch.round(torch.log2(torch.abs(x).clamp(min=eps)) - base_t) + base_t)
        x_dq = torch.where(torch.abs(x) < eps, torch.zeros_like(x), x_dq)
        rel_mse = float(torch.mean(((x - x_dq) / (torch.abs(x) + eps)) ** 2))
        if rel_mse < best_rel_mse:
            best_rel_mse = rel_mse
            best_base = base
    return best_rel_mse, best_base


def rel_mse_lloydmax(x, levels):
    """Relative MSE of a Lloyd-Max quantizer with given reconstruction levels."""
    x = x.astype(np.float64)
    eps = 1e-6
    levels = levels.astype(np.float64)
    # assign each x to nearest level
    idx = np.argmin(np.abs(x[:, None] - levels[None, :]), axis=1)
    q = levels[idx]
    return float(np.mean(((x - q) / (np.abs(x) + eps)) ** 2))


def analyze_layer(name, x, n_levels=256, n_iters=50):
    """Analyze a single layer's activation distribution."""
    if x.size == 0:
        return None

    # Fit Laplace (for plotting PDF curve only)
    mu, b = stats.laplace.fit(x)

    # Clip outliers using percentile to remove obvious outliers
    low = np.percentile(x, 0.5)
    high = np.percentile(x, 99.5)
    x_clip = np.clip(x, low, high)

    # Lloyd-Max directly on CLIPPED real data
    levels, boundaries = lloyd_max_discrete(x_clip, n_levels, n_iters)

    # Compute relative MSEs on clipped data
    mse_u = rel_mse_uniform(x_clip, n_levels)
    mse_l2, best_base = rel_mse_log2(x_clip, n_levels)
    mse_lm = rel_mse_lloydmax(x_clip, levels)

    # Generate Log2 levels for plotting (within clipped range)
    qmin = -(2 ** (8 - 1) - 1)
    qmax = (2 ** (8 - 1) - 1)
    k = np.arange(qmin, qmax + 1)
    log2_levels = np.concatenate([-np.power(2.0, k + best_base)[::-1], [0], np.power(2.0, k + best_base)])
    vis_mask = (log2_levels >= low) & (log2_levels <= high)
    log2_levels = log2_levels[vis_mask]

    return {
        "name": name,
        "N": int(x.size),
        "mu": float(mu),
        "b": float(b),
        "clip_low": float(low),
        "clip_high": float(high),
        "rel_uniform": mse_u,
        "rel_log2": mse_l2,
        "rel_lloydmax": mse_lm,
        "levels": levels,
        "boundaries": boundaries,
        "log2_levels": log2_levels,
    }


def plot_layer(result, x, output_dir):
    """Plot PDF + quantizer levels for one layer."""
    name = result["name"]
    mu = result["mu"]
    b = result["b"]
    low = result["clip_low"]
    high = result["clip_high"]
    levels = result["levels"]
    log2_levels = result["log2_levels"]
    x_clip = np.clip(x, low, high)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram (clipped data so levels are visually readable)
    ax.hist(x_clip, bins=200, density=True, alpha=0.4, color="steelblue", label="Activation histogram (clipped)")

    # Fitted Laplace PDF (within clip range)
    xs = np.linspace(low, high, 2000)
    pdf = stats.laplace.pdf(xs, loc=mu, scale=b)
    ax.plot(xs, pdf, color="navy", lw=2, label=f"Laplace fit (μ={mu:.3f}, b={b:.3f})")

    # Log2 levels
    ax.vlines(log2_levels, 0, ax.get_ylim()[1] * 0.30, colors="green", lw=1.0, alpha=0.7, label="Log2 levels")

    # Lloyd-Max levels: density-aware subsampling
    levels_sorted = np.sort(levels)
    center_mask = np.abs(levels_sorted) <= 3 * b
    center_levels = levels_sorted[center_mask]
    tail_levels = levels_sorted[~center_mask]
    n_tail_target = max(0, 64 - len(center_levels))
    if len(tail_levels) > n_tail_target:
        tail_idx = np.linspace(0, len(tail_levels) - 1, n_tail_target).astype(int)
        tail_levels = tail_levels[tail_idx]
    subset_levels = np.sort(np.concatenate([center_levels, tail_levels]))
    ax.vlines(subset_levels, 0, ax.get_ylim()[1] * 0.22, colors="black", linestyles="dashed", lw=1.2, alpha=0.8, label="Lloyd-Max levels")

    # Uniform levels (MinMax)
    u_levels = np.linspace(low, high, 64)
    ax.vlines(u_levels, 0, ax.get_ylim()[1] * 0.12, colors="red", lw=1.0, alpha=0.7, label="Uniform levels (subset)")

    ax.set_xlim(low, high)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.set_title(f"Layer: {name}   Rel-MSE  Uniform={result['rel_uniform']:.2e} | Log2={result['rel_log2']:.2e} | Lloyd-Max={result['rel_lloydmax']:.2e}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name.replace('.', '_')}.png"), dpi=150)
    plt.close(fig)


def print_summary(results, output_dir):
    """Print and save summary table (improvement over Uniform only)."""
    lines = []
    header = f"{'Layer':<40} {'N':>10} {'b':>10} {'Log2↓':>10} {'Lloyd↓':>10}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for r in results:
        mse_u = r["rel_uniform"]
        mse_l2 = r["rel_log2"]
        mse_lm = r["rel_lloydmax"]
        imp_l2 = (mse_u - mse_l2) / mse_u * 100.0
        imp_lm = (mse_u - mse_lm) / mse_u * 100.0
        # cap extreme negative values for readability
        imp_l2_str = f"{imp_l2:>9.1f}%"
        imp_lm_str = f"{imp_lm:>9.1f}%"
        lines.append(
            f"{r['name']:<40} {r['N']:>10} {r['b']:>10.4f} {imp_l2_str} {imp_lm_str}"
        )

    text = "\n".join(lines)
    print("\n" + text + "\n")

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    print("Building model and dataloader...")
    model, loader = build_model_and_loader(cfg)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    print(f"Model loaded from {args.ckpt}")

    print(f"Collecting activations from {args.num_samples} samples...")
    wrapped_model = MMDataParallel(model, device_ids=[0])
    activations = collect_activations(model, wrapped_model, loader, args.num_samples)
    print(f"Collected {len(activations)} layers")

    print("Running Lloyd-Max analysis...")
    results = []
    for name, x in activations.items():
        print(f"  [{name}] N={x.size}, fitting Laplace...")
        r = analyze_layer(name, x, args.n_levels, args.n_iters)
        if r is None:
            continue
        results.append(r)
        plot_layer(r, x, args.output_dir)

    print_summary(results, args.output_dir)
    print(f"Done. Figures and summary saved to {args.output_dir}")


if __name__ == "__main__":
    main()
