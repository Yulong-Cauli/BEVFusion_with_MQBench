#!/usr/bin/env python
"""Re-render only log_base_relmse_nds_map_dualaxis_legend_upper_right.png
with the unified Noto Serif CJK SC (Songti-style) font for both Latin and CJK.

Pure CSV/log → matplotlib; no GPU needed.
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/lloydmax_analysis_100_new")
    parser.add_argument("--summary-csv", default="runs/lloydmax_analysis_100_new/summary.csv")
    parser.add_argument("--log-xmin", type=float, default=1.0)
    parser.add_argument("--log-xmax", type=float, default=18.0)
    parser.add_argument("--font", default="Noto Serif CJK SC")
    parser.add_argument("--loc", default="center right",
                        help="matplotlib legend location, e.g. 'upper right', 'center right'")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.append(os.getcwd())

    # Unified Songti-style serif font for everything.
    plt.rcParams["font.family"] = [args.font, "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    from tools.plot_loga_metrics import (
        BASES,
        BASE_LABELS,
        load_relmse,
        load_nds_map,
        _legend_top,
    )

    repo_root = os.getcwd()
    output_dir = os.path.join(repo_root, args.output_dir)
    summary_csv = os.path.join(repo_root, args.summary_csv)
    os.makedirs(output_dir, exist_ok=True)

    relmse = load_relmse(summary_csv)
    nds, map_v = load_nds_map(repo_root)

    out_suffix = args.loc.replace(" ", "_")
    out_path = os.path.join(output_dir, f"log_base_relmse_nds_map_dualaxis_legend_{out_suffix}.png")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1 = ax1.plot(BASES, relmse, color="#1f77b4", marker="o", lw=2, label="Rel-MSE")
    ax1.set_xscale("log", base=2)
    ax1.set_xlim(args.log_xmin, args.log_xmax)
    ax1.set_xticks(BASES)
    ax1.set_xticklabels(BASE_LABELS)
    ax1.set_xlabel("底数a（log2刻度）")
    ax1.set_ylabel("Rel-MSE", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    l2 = ax2.plot(BASES, nds, color="#2ca02c", marker="s", lw=2, label="NDS")
    l3 = ax2.plot(BASES, map_v, color="#d62728", marker="^", lw=2, label="mAP")
    ax2.set_ylabel("NDS / mAP（%）", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0, symbol=""))

    _legend_top(ax1, l1 + l2 + l3, ["Rel-MSE", "NDS", "mAP"], args.loc)
    ax1.set_title("Rel-MSE 与 NDS/mAP 对比")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
