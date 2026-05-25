#!/usr/bin/env python
"""Re-render a single Lloyd-Max activation histogram with custom fonts.

Targets one specific sparse-conv layer, runs 100-frame forward on a chosen
GPU, then re-plots the histogram using a Times-New-Roman-equivalent serif
font for Latin glyphs and a Songti-style CJK serif font for Chinese glyphs.

Default output:
    runs/lloydmax_analysis_100_new/encoder_layers_encoder_layer1_2_0.png

Defaults are tuned to match the original 100-sample analysis run. Pin to a
single GPU before any torch import so the model never lands on a busy card.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Re-plot one Lloyd-Max layer with TNR/Songti-style fonts")
    parser.add_argument("--config", default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
    parser.add_argument("--ckpt", default="pretrained/bevfusion-det.pth")
    parser.add_argument("--layer", default="encoder_layers.encoder_layer1.2.0",
                        help="Module name to hook; default matches encoder_layers_encoder_layer1_2_0.png")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n-levels", type=int, default=256)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--clip-pct", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=3, help="Physical GPU index to use")
    parser.add_argument("--output-dir", default="runs/lloydmax_analysis_100_new")
    parser.add_argument("--font", default="Noto Serif CJK SC",
                        help="Single Songti-style serif font used for all glyphs (Latin + CJK)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Pin GPU before importing torch so we never spill onto the training cards.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.analyze_lloydmax_fast import (
        lloyd_max_cpu,
        rel_mse_log2_cpu,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- font: single Songti-style serif (Noto Serif CJK SC) for both Latin and CJK
    plt.rcParams["font.family"] = [args.font, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- load model + dataloader (one process, no multiprocessing pool)
    print(f"[GPU {args.gpu}] loading config + model ...", flush=True)
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = Config(recursive_eval(cfg._cfg_dict), filename=cfg.filename)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model = model.cuda().eval()
    load_checkpoint(model, args.ckpt, map_location="cpu")
    print(f"checkpoint loaded: {args.ckpt}", flush=True)

    calib_cfg = cfg.data.train.copy()
    calib_cfg.test_mode = True
    calib_dataset = build_dataset(calib_cfg)
    calib_loader = build_dataloader(
        calib_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=True,
    )

    # ---- hook only the requested layer
    backbone = model.encoders.lidar.backbone
    activations = {}

    def hook_fn(_module, _inp, output):
        feats = output.features.detach().cpu().float().numpy().reshape(-1)
        activations.setdefault(args.layer, []).append(feats)

    target_module = None
    for name, module in backbone.named_modules():
        if name == args.layer:
            target_module = module
            break
    if target_module is None:
        raise SystemExit(f"layer not found in lidar.backbone: {args.layer}")
    handle = target_module.register_forward_hook(hook_fn)
    print(f"hooked: {args.layer} ({type(target_module).__name__})", flush=True)

    wrapped = MMDataParallel(model, device_ids=[0])
    print(f"running forward on {args.num_samples} samples (GPU {args.gpu}) ...", flush=True)
    import time
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= args.num_samples:
                break
            try:
                wrapped(return_loss=False, rescale=True, **data)
            except Exception:
                if isinstance(data, list) and len(data) == 1:
                    wrapped(return_loss=False, rescale=True, **data[0])
                else:
                    raise
            if (i + 1) % 20 == 0:
                print(f"  forward {i + 1}/{args.num_samples} ({time.time() - t0:.1f}s)", flush=True)
    handle.remove()
    print(f"forward done in {time.time() - t0:.1f}s", flush=True)

    # release GPU asap so it stays out of the training jobs' way
    del model, wrapped, calib_loader, calib_dataset
    torch.cuda.empty_cache()

    if args.layer not in activations:
        raise SystemExit(f"hook collected nothing for {args.layer}")
    x = np.concatenate(activations[args.layer]).astype(np.float64)
    x = x[np.isfinite(x)]
    print(f"collected {x.size} activation samples", flush=True)

    mu, b = stats.laplace.fit(x)
    low = float(np.percentile(x, args.clip_pct))
    high = float(np.percentile(x, 100.0 - args.clip_pct))
    x_clip = np.clip(x, low, high)

    print("running Lloyd-Max + log2 grid (CPU) ...", flush=True)
    levels, _ = lloyd_max_cpu(x_clip, args.n_levels, args.n_iters)
    _, best_base = rel_mse_log2_cpu(x_clip, args.n_levels)

    qmin = -(2 ** (8 - 1) - 1)
    qmax = (2 ** (8 - 1) - 1)
    k = np.arange(qmin, qmax + 1)
    log2_levels = np.concatenate([
        -np.power(2.0, k + best_base)[::-1],
        [0],
        np.power(2.0, k + best_base),
    ])
    log2_levels = log2_levels[(log2_levels >= low) & (log2_levels <= high)]
    u_levels = np.linspace(low, high, 64)

    # ---- re-plot with new fonts
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x_clip, bins=200, density=True, alpha=0.4, color="steelblue", label="激活直方图")
    xs = np.linspace(low, high, 2000)
    ax.plot(xs, stats.laplace.pdf(xs, loc=mu, scale=b), color="navy", lw=2,
            label=f"拉普拉斯拟合 (μ={mu:.3f}, b={b:.3f})")
    ax.vlines(log2_levels, 0, ax.get_ylim()[1] * 0.30, colors="green", lw=1.0, alpha=0.7,
              label="Log2 量化级")
    ax.vlines(u_levels, 0, ax.get_ylim()[1] * 0.12, colors="red", lw=1.0, alpha=0.7,
              label="均匀量化级")
    ax.set_xlim(low, high)
    ax.set_xlabel("激活值")
    ax.set_ylabel("概率密度")
    ax.set_title(f"层: {args.layer}")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, f"{args.layer.replace('.', '_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
