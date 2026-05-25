#!/usr/bin/env python3
"""Re-render only artifacts/vtransform_activation_hist_zh/vtransform_downsample_0_hist_zh.png
with Noto Serif CJK SC (Songti-style) for both Latin and CJK glyphs.

Hooks ONLY the requested vtransform Conv2d layer to keep GPU memory minimal.
Pin the GPU before importing torch so we never spill onto a busy card.

Caches the collected activations so subsequent plot tweaks don't re-run forward.
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    )
    parser.add_argument("--checkpoint", default="pretrained/bevfusion-det.pth")
    parser.add_argument("--layer", default="downsample.0",
                        help="vtransform sub-module name; default reproduces vtransform_downsample_0_hist_zh.png")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--sample-per-forward", type=int, default=20000)
    parser.add_argument("--bins", type=int, default=220)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--out-dir", default="artifacts/vtransform_activation_hist_zh")
    parser.add_argument("--font", default="Noto Serif CJK SC")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running forward pass even if cache exists")
    parser.add_argument("--xlim-full", type=float, nargs=2, default=[-200, 200],
                        help="Full-range x-axis limits (low high)")
    parser.add_argument("--xlim-zoom", type=float, nargs=2, default=[-0.25, 0.25],
                        help="Zoom x-axis limits (low high)")
    return parser.parse_args()


def collect_activations(args):
    """GPU-side: load model, hook target layer, run forward, return (vals, min_val, max_val)."""
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

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[GPU {args.gpu}] loading config + model ...", flush=True)
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    print(f"checkpoint loaded: {args.checkpoint}", flush=True)

    vt = model.module.encoders["camera"]["vtransform"]
    target_module = None
    for name, mod in vt.named_modules():
        if name == args.layer and isinstance(mod, nn.Conv2d):
            target_module = mod
            break
    if target_module is None:
        raise SystemExit(f"layer not found in vtransform (Conv2d): {args.layer}")

    st = {"min": float("inf"), "max": float("-inf"), "samples": []}

    def hook_fn(_m, _inp, out):
        if not isinstance(out, torch.Tensor):
            return
        t = out.detach().float()
        cur_min = t.min().item()
        cur_max = t.max().item()
        if cur_min < st["min"]:
            st["min"] = cur_min
        if cur_max > st["max"]:
            st["max"] = cur_max
        flat = t.reshape(-1)
        n = flat.numel()
        k = args.sample_per_forward
        if n > k:
            idx = torch.randint(0, n, (k,), device=flat.device)
            sampled = flat[idx]
        else:
            sampled = flat
        st["samples"].append(sampled.cpu().numpy())

    handle = target_module.register_forward_hook(hook_fn)
    print(f"hooked: vtransform.{args.layer}", flush=True)

    print(f"running forward on {args.num_samples} samples (GPU {args.gpu}) ...", flush=True)
    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for data in data_loader:
            _ = model(return_loss=False, rescale=True, **data)
            processed += 1
            if processed % 20 == 0:
                print(f"  forward {processed}/{args.num_samples} ({time.time() - t0:.1f}s)", flush=True)
            if processed >= args.num_samples:
                break
    handle.remove()
    print(f"forward done in {time.time() - t0:.1f}s", flush=True)

    del model, data_loader, dataset
    torch.cuda.empty_cache()

    if len(st["samples"]) == 0:
        raise SystemExit(f"hook collected nothing for {args.layer}")

    vals = np.concatenate(st["samples"], axis=0).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise SystemExit("no finite activations collected")
    return vals, float(st["min"]), float(st["max"])


def main():
    args = parse_args()

    # Pin GPU BEFORE torch import (only matters on cache miss).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.append(os.getcwd())

    import numpy as np
    try:
        np.long = int
        np.int = int
        np.float = float
        np.bool = bool
    except Exception:
        pass

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [args.font, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    np.random.seed(args.seed)

    root = Path(os.getcwd())
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / f"_cache_{args.layer.replace('.', '_')}_n{args.num_samples}.npz"

    if cache_path.exists() and not args.force:
        print(f"loading cached activations: {cache_path}", flush=True)
        cache = np.load(cache_path)
        vals = cache["vals"]
        min_val = float(cache["min_val"])
        max_val = float(cache["max_val"])
    else:
        vals, min_val, max_val = collect_activations(args)
        np.savez(cache_path, vals=vals, min_val=min_val, max_val=max_val)
        print(f"saved cache: {cache_path}", flush=True)

    p999 = float(np.percentile(vals, 99.9))
    p001 = float(np.percentile(vals, 0.1))
    if p999 <= min_val:
        p999 = float(np.percentile(vals, 99.99))

    # mmcv/mmdet imports during forward can clobber font rcParams; reapply before plotting.
    plt.rcParams["font.family"] = [args.font, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    color = "#D39572"
    layer_title = args.layer.replace(".", "")

    # shared ylim: compute both hists as counts (no density) first
    zoom_bound = max(abs(args.xlim_zoom[0]), abs(args.xlim_zoom[1]))
    zoom_vals = vals[np.abs(vals) <= zoom_bound]
    if zoom_vals.size < 10:
        zoom_vals = vals
    full_counts, _ = np.histogram(vals, bins=args.bins)
    zoom_counts, _ = np.histogram(zoom_vals, bins=args.bins)
    shared_ylim = max(int(full_counts.max()), int(zoom_counts.max())) * 1.05

    # ---- full-range single figure ----
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    ax1.hist(vals, bins=args.bins, color=color, alpha=0.90, edgecolor="white", linewidth=0.25)
    ax1.axvline(max_val, color="#B13C2E", linestyle="-", linewidth=2.0, label="最大值")
    ax1.axvline(p999, color="#2F4858", linestyle="--", linewidth=2.0, label="99.9分位")
    ax1.set_xlabel("激活值", fontsize=13)
    ax1.set_ylabel("频数", fontsize=13)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.legend(loc="upper left", frameon=True, fontsize=12)
    ax1.set_xlim(args.xlim_full[0], args.xlim_full[1])
    ax1.set_ylim(0, shared_ylim)
    ax1.xaxis.set_label_coords(0.5, -0.10)
    for spine in ax1.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    fig1.tight_layout()
    out_name_full = f"vtransform_{args.layer.replace('.', '_')}_hist_zh_full.png"
    out_path_full = out_dir / out_name_full
    fig1.savefig(out_path_full, bbox_inches="tight")
    plt.close(fig1)
    print(f"saved: {out_path_full}")

    # ---- zoomed-in single figure ----
    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
    ax2.hist(zoom_vals, bins=args.bins, color=color, alpha=0.90, edgecolor="white", linewidth=0.25)
    ax2.set_xlabel("激活值", fontsize=13)
    ax2.set_ylabel("频数", fontsize=13)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.grid(alpha=0.25, linestyle=":")
    ax2.set_xlim(args.xlim_zoom[0], args.xlim_zoom[1])
    ax2.set_ylim(0, shared_ylim)
    ax2.xaxis.set_label_coords(0.5, -0.10)
    for spine in ax2.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    fig2.tight_layout()
    out_name_zoom = f"vtransform_{args.layer.replace('.', '_')}_hist_zh_zoom.png"
    out_path_zoom = out_dir / out_name_zoom
    fig2.savefig(out_path_zoom, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved: {out_path_zoom}")

    print(f"[完成] min={min_val:.6f}, p0.1={p001:.6f}, p99.9={p999:.6f}, "
          f"max={max_val:.6f}, 采样点={vals.size}")


if __name__ == "__main__":
    main()
