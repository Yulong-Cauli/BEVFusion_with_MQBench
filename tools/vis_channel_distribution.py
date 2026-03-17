#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script: LiDAR Backbone Channel-wise Distribution (Boxplot)
=======================================================================
This script analyzes the output feature distribution of LiDAR backbone layers per channel.
It generates a boxplot similar to RepQ-ViT Figure 2, showing the range and quartiles
of activation values for each channel.

Usage:
    python tools/vis_channel_distribution.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        pretrained/bevfusion-det.pth \
        --layer-name "encoder_layers.3" \
        --num-batches 10 \
        --output-dir results_vis/channel_dist
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hack to import local modules
sys.path.append(os.getcwd())

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.ops.spconv.structure import SparseConvTensor
from mmdet3d.ops.spconv.conv import SparseConvolution

# Increase font size for better readability
plt.rcParams.update({'font.size': 12})

def collect_channel_stats(model, data_loader, num_batches, target_layer_name=None):
    """
    Collects activations for the specified layer(s).
    Returns a dict: {layer_name: [list of (N, C) numpy arrays]}
    """
    activations = {}
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            # SparseConvTensor
            feats = None
            if isinstance(output, SparseConvTensor):
                if output.features.numel() > 0:
                    feats = output.features.detach().cpu().numpy()
            elif isinstance(output, torch.Tensor):
                if output.numel() > 0:
                    feats = output.detach().cpu().numpy()
                    # If 4D tensor (N, C, H, W), permute to (N, H, W, C) and flatten
                    if feats.ndim == 4:
                        feats = feats.transpose(0, 2, 3, 1).reshape(-1, feats.shape[1])
                    elif feats.ndim == 3:
                        feats = feats.transpose(0, 2, 1).reshape(-1, feats.shape[1])
            
            if feats is not None:
                if name not in activations:
                    activations[name] = []
                activations[name].append(feats)

        return hook

    # Register hooks
    # Assuming model is MMDataParallel -> module -> encoders -> lidar -> backbone
    try:
        if isinstance(model, MMDataParallel):
            backbone = model.module.encoders.lidar.backbone
        else:
            backbone = model.encoders.lidar.backbone
    except AttributeError:
        print("Error: Could not find lidar backbone in model.")
        return {}

    found_layers = 0
    for name, module in backbone.named_modules():
        # Filter by target_layer_name if provided
        if target_layer_name and target_layer_name not in name:
            continue
        
        # We are interested in SparseConvolution layers usually
        if isinstance(module, SparseConvolution):
            handles.append(module.register_forward_hook(get_hook(name)))
            found_layers += 1
            # print(f"Hooked layer: {name}")

    if found_layers == 0:
        print(f"Warning: No layers found matching '{target_layer_name}' in lidar backbone")
        # List available layers
        print("Available layers:")
        for name, module in backbone.named_modules():
            if isinstance(module, SparseConvolution):
                print(f"  {name}")
        return {}
    
    print(f"Hooked {found_layers} layers matching '{target_layer_name}'")

    # Run inference
    print(f"Running inference for {num_batches} batches...")
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            try:
                model(return_loss=False, rescale=True, **data)
            except Exception as e:
                print(f"Error during inference batch {i}: {e}")
                continue
            
            if (i+1) % 5 == 0:
                print(f"Batch {i+1}/{num_batches}")

    # Remove hooks
    for h in handles:
        h.remove()
        
    return activations

def plot_channel_boxplot(layer_name, feats_list, output_dir):
    """
    Generates a boxplot for a single layer.
    feats_list: list of (N_i, C) arrays.
    """
    if not feats_list:
        print(f"No features collected for {layer_name}")
        return

    # Concatenate all batches: (Total_N, C)
    try:
        data = np.concatenate(feats_list, axis=0)
    except ValueError as e:
        print(f"Error concatenating features for {layer_name}: {e}")
        return

    num_channels = data.shape[1]
    total_points = data.shape[0]
    
    print(f"Plotting {layer_name}: {total_points} points, {num_channels} channels")

    # Calculate figure size width based on number of channels
    width = max(10, num_channels * 0.15)
    
    # 1. Full Range Plot (with outliers cut off visually at 5*IQR or similar if crazy, but user wants range)
    # Boxplot with whiskers at min/max (whis=[0, 100]) shows true range.
    plt.figure(figsize=(width, 6))
    
    # Check if we have NaN or Inf
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"Warning: NaNs or Infs found in {layer_name}, removing them for plot.")
        data = data[~np.isnan(data).any(axis=1)]
        data = data[~np.isinf(data).any(axis=1)]
    
    plt.boxplot(data, whis=[0, 100], showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue', alpha=0.6),
                medianprops=dict(color='red'))
    
    plt.title(f"Channel Activation Distribution: {layer_name}\n(Whiskers = Min/Max, Box = IQR)\nMin={np.min(data):.4f}, Max={np.max(data):.4f}", fontsize=12)
    plt.xlabel("Channel Index")
    plt.ylabel("Activation Value")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.5) # Add zero line
    
    # Auto-scale Y with some margin
    plt.margins(y=0.1)

    # Set X-ticks
    step = max(1, num_channels // 20)
    locs = np.arange(1, num_channels+1, step)
    labels = np.arange(0, num_channels, step)
    plt.xticks(locs, labels, rotation=45)
    
    plt.tight_layout()
    
    safe_name = layer_name.replace(".", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f"boxplot_{safe_name}_full.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

    # 2. Zoomed Plot (p1 to p99)
    # This helps see the box (IQR) better if there are massive outliers
    # FIX: Use whis=[1, 99] so whiskers stay within the p1-p99 range (approximately)
    # If we use whis=[0, 100] but limit ylim, the whiskers will extend beyond the plot area.
    flat_data = data.flatten()
    if flat_data.size > 0:
        p1 = np.percentile(flat_data, 1)
        p99 = np.percentile(flat_data, 99)
        
        # Only plot zoom if range is significantly smaller than min-max
        mn, mx = np.min(flat_data), np.max(flat_data)
        
        # If the full range is much larger than p1-p99, generate a zoomed plot
        if (mx - mn) > 1.5 * (p99 - p1) and (p99 - p1) > 1e-6:
            plt.figure(figsize=(width, 6))
            
            # Use whiskers at 1% and 99% to match the visual range
            plt.boxplot(data, whis=[1, 99], showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', color='green', alpha=0.6),
                        medianprops=dict(color='red'))
            
            plt.title(f"Channel Activation Distribution (Zoomed p1-p99): {layer_name}\n(Whiskers = 1st/99th Percentile)", fontsize=14)
            plt.xlabel("Channel Index")
            plt.ylabel("Activation Value")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)
            
            # Set ylim to strictly contain the 1-99 range with small padding
            y_span = p99 - p1
            plt.ylim(p1 - 0.1 * y_span, p99 + 0.1 * y_span)
            
            plt.xticks(locs, labels, rotation=45)
            
            plt.tight_layout()
            out_path_zoom = os.path.join(output_dir, f"boxplot_{safe_name}_zoom.png")
            plt.savefig(out_path_zoom, dpi=150)
            plt.close()
            print(f"Saved zoomed plot to {out_path_zoom}")

def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR Channel Distributions")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--layer-name", type=str, default=None, help="Substring to filter layer names (e.g. 'encoder_layers.3')")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--output-dir", default="results_vis/channel_dist", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    
    # Disable pretrained loading
    cfg.model.pretrained = None
    if hasattr(cfg.model, "encoders"):
        for enc_cfg in cfg.model.encoders.values() if hasattr(cfg.model.encoders, "values") else []:
            if hasattr(enc_cfg, "backbone") and hasattr(enc_cfg.backbone, "init_cfg"):
                enc_cfg.backbone.init_cfg = None

    # Build Model
    print("Building model...")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)
    model.eval()
    
    model = MMDataParallel(model, device_ids=[0])

    # Build Dataloader
    print("Building dataloader...")
    dataset = build_dataset(cfg.data.val) 
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    # Collect Stats
    activations = collect_channel_stats(model, data_loader, args.num_batches, args.layer_name)

    # Plot
    print("Generating plots...")
    if not activations:
        print("No activations collected. Exiting.")
        return

    for name, feats in activations.items():
        plot_channel_boxplot(name, feats, args.output_dir)

if __name__ == "__main__":
    main()
