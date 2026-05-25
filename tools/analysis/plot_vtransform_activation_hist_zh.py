#!/usr/bin/env python3
import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.append(os.getcwd())

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
from matplotlib import font_manager
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 vtransform 各层激活分布图（中文标注）")
    parser.add_argument(
        "--config",
        default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
        help="配置文件",
    )
    parser.add_argument("--checkpoint", default="pretrained/bevfusion-det.pth", help="模型权重")
    parser.add_argument("--num-samples", type=int, default=100, help="统计样本数")
    parser.add_argument("--sample-per-forward", type=int, default=20000, help="每次前向每层随机采样点数")
    parser.add_argument("--bins", type=int, default=220, help="柱状图 bins")
    parser.add_argument("--workers-per-gpu", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out-dir", default="artifacts/vtransform_activation_hist_zh", help="输出目录")
    return parser.parse_args()


def setup_matplotlib_cn():
    noto_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if noto_path.exists():
        font_manager.fontManager.addfont(str(noto_path))
        fp = font_manager.FontProperties(fname=str(noto_path))
        font_name = fp.get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.sans-serif"] = [font_name]
    else:
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "SimHei",
            "Microsoft YaHei",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    plt.rcParams["axes.unicode_minus"] = False


def get_ordered_vtransform_conv_layers(vt_module: nn.Module):
    order = {"dtransform": 0, "depthnet": 1, "downsample": 2}
    convs = []
    for name, m in vt_module.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        prefix = name.split(".")[0] if "." in name else name
        if prefix not in order:
            continue
        try:
            idx = int(name.split(".")[1])
        except Exception:
            idx = 0
        convs.append((name, m, order[prefix], idx))
    convs.sort(key=lambda x: (x[2], x[3], x[0]))
    return [(name, m) for name, m, _, _ in convs]


def format_layer_title(layer_name: str) -> str:
    # 例如: "downsample.0" -> "downsample0"
    return layer_name.replace(".", "")


def main():
    args = parse_args()
    setup_matplotlib_cn()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    root = Path(os.getcwd())
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    vt = model.module.encoders["camera"]["vtransform"]
    conv_layers = get_ordered_vtransform_conv_layers(vt)
    if len(conv_layers) == 0:
        raise RuntimeError("未找到 vtransform 中的 Conv2d 层。")

    stats = OrderedDict()
    hooks = []
    for layer_name, layer in conv_layers:
        stats[layer_name] = {"min": float("inf"), "max": float("-inf"), "samples": []}

        def _hook(name):
            def fn(_, __, out):
                if not isinstance(out, torch.Tensor):
                    return
                t = out.detach().float()
                cur_min = t.min().item()
                cur_max = t.max().item()
                if cur_min < stats[name]["min"]:
                    stats[name]["min"] = cur_min
                if cur_max > stats[name]["max"]:
                    stats[name]["max"] = cur_max

                flat = t.reshape(-1)
                n = flat.numel()
                k = args.sample_per_forward
                if n > k:
                    idx = torch.randint(0, n, (k,), device=flat.device)
                    sampled = flat[idx]
                else:
                    sampled = flat
                stats[name]["samples"].append(sampled.cpu().numpy())

            return fn

        hooks.append(layer.register_forward_hook(_hook(layer_name)))

    processed = 0
    with torch.no_grad():
        for data in data_loader:
            _ = model(return_loss=False, rescale=True, **data)
            processed += 1
            if processed >= args.num_samples:
                break

    for h in hooks:
        h.remove()

    if processed < args.num_samples:
        print(f"[警告] 实际只处理了 {processed} 个样本（目标 {args.num_samples}）。")

    color = "#D39572"
    for layer_name, st in stats.items():
        if len(st["samples"]) == 0:
            continue
        vals = np.concatenate(st["samples"], axis=0).astype(np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        p999 = float(np.percentile(vals, 99.9))
        p001 = float(np.percentile(vals, 0.1))
        max_val = float(st["max"])
        min_val = float(st["min"])
        if p999 <= min_val:
            p999 = float(np.percentile(vals, 99.99))
        # 用中心区间 [0.1, 99.9] 构造对称显示范围，避免右图大面积留白
        sym_bound = max(abs(p001), abs(p999))
        if sym_bound <= 0:
            sym_bound = float(np.percentile(np.abs(vals), 99.0))
        if sym_bound <= 0:
            sym_bound = float(np.max(np.abs(vals)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)

        ax1.hist(vals, bins=args.bins, density=True, color=color, alpha=0.90, edgecolor="white", linewidth=0.25)
        ax1.axvline(max_val, color="#B13C2E", linestyle="-", linewidth=2.0, label="最大值")
        ax1.axvline(p999, color="#2F4858", linestyle="--", linewidth=2.0, label="99.9分位")
        ax1.set_title("全范围激活分布", fontsize=12)
        ax1.set_xlabel("激活值")
        ax1.set_ylabel("概率密度")
        ax1.grid(alpha=0.25, linestyle=":")
        ax1.legend(loc="upper right", frameon=True)

        # 右图做对称双侧截断，保证显示聚焦主体分布
        zoom_vals = vals[np.abs(vals) <= sym_bound]
        if zoom_vals.size < 10:
            zoom_vals = vals
        ax2.hist(zoom_vals, bins=args.bins, density=True, color=color, alpha=0.90, edgecolor="white", linewidth=0.25)
        ax2.set_title("局部放大", fontsize=12)
        ax2.set_xlabel("激活值")
        ax2.set_ylabel("概率密度")
        ax2.grid(alpha=0.25, linestyle=":")
        ax2.set_xlim(-sym_bound, sym_bound)

        layer_title = format_layer_title(layer_name)
        fig.suptitle(
            f"层:{layer_title}",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout()

        out_name = f"vtransform_{layer_name.replace('.', '_')}_hist_zh.png"
        fig.savefig(out_dir / out_name, bbox_inches="tight")
        plt.close(fig)
        print(
            f"[完成] {out_name} | min={min_val:.6f}, p0.1={p001:.6f}, p99.9={p999:.6f}, 对称边界={sym_bound:.6f}, max={max_val:.6f}, 采样点={vals.size}"
        )

    print(f"\n全部完成，输出目录：{out_dir}")


if __name__ == "__main__":
    main()
