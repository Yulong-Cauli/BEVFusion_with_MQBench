#!/usr/bin/env python
import argparse
import csv
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASES = np.array([float(np.sqrt(2)), 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
BASE_LABELS = ["sqrt2", "2", "4", "8", "16"]
LOGA_LOG_FILES = {
    float(np.sqrt(2)): "logs/ablation_LOGA_a1p41421356.log",
    2.0: "logs/ablation_LOGA_a2p0.log",
    4.0: "logs/ablation_LOGA_a4p0.log",
    8.0: "logs/ablation_LOGA_a8p0.log",
    16.0: "logs/ablation_LOGA_a16p0.log",
}


def configure_plot_font():
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _parse_last_float(pattern, text, name):
    vals = re.findall(pattern, text)
    if not vals:
        raise RuntimeError(f"Failed to parse {name}")
    return float(vals[-1])


def load_nds_map(repo_root):
    nds_vals = []
    map_vals = []
    for base in BASES:
        rel_path = LOGA_LOG_FILES[float(base)]
        path = os.path.join(repo_root, rel_path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        map_vals.append(_parse_last_float(r"mAP:\s*([0-9.]+)", text, f"mAP from {rel_path}"))
        nds_vals.append(_parse_last_float(r"NDS:\s*([0-9.]+)", text, f"NDS from {rel_path}"))
    return np.array(nds_vals, dtype=np.float64), np.array(map_vals, dtype=np.float64)


def load_relmse(summary_csv):
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key_map = [
        "rel_a_sqrt2",
        "rel_a_2.00",
        "rel_a_4.00",
        "rel_a_8.00",
        "rel_a_16.00",
    ]
    out = []
    for key in key_map:
        vals = [float(r[key]) for r in rows if r.get(key, "").strip()]
        out.append(float(np.mean(vals)))
    return np.array(out, dtype=np.float64)


def _legend_top(ax, handles, labels, loc):
    leg = ax.legend(
        handles,
        labels,
        loc=loc,
        framealpha=0.6,
        facecolor="white",
        edgecolor="black",
    )
    leg.set_zorder(10000)
    return leg


def _set_bottom_log2_xaxis(ax, log_xmin, log_xmax):
    ax.set_xscale("log", base=2)
    ax.set_xlim(log_xmin, log_xmax)
    ax.set_xticks(BASES)
    ax.set_xticklabels(BASE_LABELS)
    ax.set_xlabel("底数a（log2刻度）")


def plot_single_metric_linear(y, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BASES), dtype=np.float64)
    ax.plot(x, y, marker="o", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels(BASE_LABELS)
    ax.set_xlabel("底数a")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_metric_log2(y, ylabel, title, out_path, log_xmin, log_xmax):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(BASES, y, marker="o", lw=2)
    _set_bottom_log2_xaxis(ax, log_xmin, log_xmax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_relmse_vs_metric(relmse, metric, metric_name, out_path, log_xmin, log_xmax):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1 = ax1.plot(BASES, relmse, color="#1f77b4", marker="o", lw=2, label="Rel-MSE")
    _set_bottom_log2_xaxis(ax1, log_xmin, log_xmax)
    ax1.set_ylabel("Rel-MSE", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    l2 = ax2.plot(BASES, metric, color="#d62728", marker="s", lw=2, label=metric_name)
    ax2.set_ylabel(metric_name, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    _legend_top(ax1, l1 + l2, ["Rel-MSE", metric_name], "upper right")
    ax1.set_title(f"Rel-MSE 与 {metric_name} 对比")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_relmse_nds_map(relmse, nds, map_v, out_path, loc, log_xmin, log_xmax):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1 = ax1.plot(BASES, relmse, color="#1f77b4", marker="o", lw=2, label="Rel-MSE")
    _set_bottom_log2_xaxis(ax1, log_xmin, log_xmax)
    ax1.set_ylabel("Rel-MSE", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, which="both", alpha=0.3)

    ax2 = ax1.twinx()
    l2 = ax2.plot(BASES, nds, color="#2ca02c", marker="s", lw=2, label="NDS")
    l3 = ax2.plot(BASES, map_v, color="#d62728", marker="^", lw=2, label="mAP")
    ax2.set_ylabel("NDS / mAP", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    _legend_top(ax1, l1 + l2 + l3, ["Rel-MSE", "NDS", "mAP"], loc)
    ax1.set_title("Rel-MSE 与 NDS/mAP 对比")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_nds_map(nds, map_v, out_path, loc, log_xmin, log_xmax):
    fig, ax = plt.subplots(figsize=(8, 5))
    l1 = ax.plot(BASES, nds, color="#2ca02c", marker="s", lw=2, label="NDS")
    l2 = ax.plot(BASES, map_v, color="#d62728", marker="^", lw=2, label="mAP")
    _set_bottom_log2_xaxis(ax, log_xmin, log_xmax)
    ax.set_ylabel("指标值")
    ax.set_title("不同底数a的NDS和mAP对比")
    ax.grid(True, which="both", alpha=0.3)
    _legend_top(ax, l1 + l2, ["NDS", "mAP"], loc)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all(output_dir, summary_csv, repo_root, log_xmin, log_xmax):
    configure_plot_font()
    relmse = load_relmse(summary_csv)
    nds, map_v = load_nds_map(repo_root)

    plot_single_metric_linear(relmse, "Rel-MSE", "Rel-MSE 随底数 a 变化", os.path.join(output_dir, "log_base_comparison_linear.png"))
    plot_single_metric_log2(relmse, "Rel-MSE", "Rel-MSE 随底数 a 变化", os.path.join(output_dir, "log_base_comparison_log2.png"), log_xmin, log_xmax)
    plot_single_metric_log2(relmse, "Rel-MSE", "Rel-MSE 随底数 a 变化", os.path.join(output_dir, "log_base_comparison.png"), log_xmin, log_xmax)

    plot_single_metric_linear(nds, "NDS", "NDS 随底数 a 变化", os.path.join(output_dir, "log_base_nds_linear.png"))
    plot_single_metric_log2(nds, "NDS", "NDS 随底数 a 变化", os.path.join(output_dir, "log_base_nds_log2.png"), log_xmin, log_xmax)

    plot_single_metric_linear(map_v, "mAP", "mAP 随底数 a 变化", os.path.join(output_dir, "log_base_map_linear.png"))
    plot_single_metric_log2(map_v, "mAP", "mAP 随底数 a 变化", os.path.join(output_dir, "log_base_map_log2.png"), log_xmin, log_xmax)

    plot_relmse_vs_metric(relmse, nds, "NDS", os.path.join(output_dir, "log_base_relmse_vs_nds_dualaxis.png"), log_xmin, log_xmax)
    plot_relmse_vs_metric(relmse, map_v, "mAP", os.path.join(output_dir, "log_base_relmse_vs_map_dualaxis.png"), log_xmin, log_xmax)

    _plot_relmse_nds_map(
        relmse,
        nds,
        map_v,
        os.path.join(output_dir, "log_base_relmse_nds_map_dualaxis.png"),
        "upper right",
        log_xmin,
        log_xmax,
    )
    _plot_nds_map(nds, map_v, os.path.join(output_dir, "log_base_nds_vs_map.png"), "upper right", log_xmin, log_xmax)

    locs = [
        ("upper_right", "upper right"),
        ("upper_left", "upper left"),
        ("upper_center", "upper center"),
        ("center_right", "center right"),
        ("center_left", "center left"),
        ("center", "center"),
        ("lower_right", "lower right"),
        ("lower_left", "lower left"),
        ("lower_center", "lower center"),
    ]
    for suffix, loc in locs:
        _plot_relmse_nds_map(
            relmse,
            nds,
            map_v,
            os.path.join(output_dir, f"log_base_relmse_nds_map_dualaxis_legend_{suffix}.png"),
            loc,
            log_xmin,
            log_xmax,
        )
        _plot_nds_map(
            nds,
            map_v,
            os.path.join(output_dir, f"log_base_nds_vs_map_legend_{suffix}.png"),
            loc,
            log_xmin,
            log_xmax,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate LOGA metric figures.")
    parser.add_argument(
        "--output-dir",
        default="runs/lloydmax_analysis_100_new",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--summary-csv",
        default="runs/lloydmax_analysis_100_new/summary.csv",
        help="summary.csv from analyze_lloydmax_fast.py",
    )
    parser.add_argument("--log-xmin", type=float, default=1.0, help="Log-scale x-axis min")
    parser.add_argument("--log-xmax", type=float, default=18.0, help="Log-scale x-axis max")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.getcwd()
    output_dir = os.path.join(repo_root, args.output_dir)
    summary_csv = os.path.join(repo_root, args.summary_csv)
    generate_all(output_dir, summary_csv, repo_root, args.log_xmin, args.log_xmax)
    print(f"Regenerated LOGA figures in: {output_dir}")


if __name__ == "__main__":
    main()
