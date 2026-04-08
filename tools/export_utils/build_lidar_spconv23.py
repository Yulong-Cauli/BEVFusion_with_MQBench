"""
Phase 4: Build LiDAR SparseEncoder using spconv 2.3 for deployment.

This script runs in the spconv23_deploy conda environment (Python 3.9 + PyTorch 2.0 + spconv 2.3).
It rebuilds the SparseEncoder using pure spconv 2.3 API (no mmdet3d dependency),
loads FP32 weights from the BEVFusion checkpoint, and runs inference.

The output is saved as a .pt file for verification against PyTorch (spconv 2.1) output.

Usage:
    conda run --prefix /media/yellowstone/data2/CYL/spconv23_deploy python \
        tools/export_utils/build_lidar_spconv23.py \
        --ckpt pretrained/bevfusion-det.pth \
        --verify-dir lidar_verify_tensors \
        --output lidar_spconv23_output.pt
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

import spconv.pytorch as spconv


# ============================================================================
# Rebuild SparseEncoder with spconv 2.3 API
# ============================================================================

class SparseBasicBlock23(spconv.SparseModule):
    """SparseBasicBlock rebuilt with spconv 2.3 API.

    Matches mmdet3d SparseBasicBlock: conv1+bn1+relu -> conv2+bn2 -> add(residual) -> relu
    """
    def __init__(self, channels, norm_cfg_eps=1e-3, norm_cfg_momentum=0.01):
        super().__init__()
        # Note: spconv 2.3 weight shape is [out, k, k, k, in]
        # while spconv 2.1 is [k, k, k, in, out]
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels, eps=norm_cfg_eps, momentum=norm_cfg_momentum)
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels, eps=norm_cfg_eps, momentum=norm_cfg_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))
        return out


class SparseEncoder23(nn.Module):
    """SparseEncoder rebuilt with spconv 2.3 API.

    Matches the BEVFusion config:
        in_channels=5, sparse_shape=[1440,1440,41], output_channels=128
        encoder_channels=[[16,16,32],[32,32,64],[64,64,128],[128,128]]
        encoder_paddings=[[0,0,1],[0,0,1],[0,0,[1,1,0]],[0,0]]
        block_type=basicblock
    """
    def __init__(self):
        super().__init__()
        norm_eps = 1e-3
        norm_mom = 0.01

        # conv_input: SubMConv3d(5, 16, 3, padding=1) + BN + ReLU
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(5, 16, 3, padding=1, bias=False, indice_key="subm1"),
            nn.BatchNorm1d(16, eps=norm_eps, momentum=norm_mom),
            nn.ReLU(inplace=True),
        )

        # encoder_layer1: BasicBlock(16,16) x2 + SparseConv3d(16,32,3,stride=2,padding=1)
        self.encoder_layer1 = spconv.SparseSequential(
            SparseBasicBlock23(16, norm_eps, norm_mom),
            SparseBasicBlock23(16, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False, indice_key="spconv1"),
                nn.BatchNorm1d(32, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )

        # encoder_layer2: BasicBlock(32,32) x2 + SparseConv3d(32,64,3,stride=2,padding=1)
        self.encoder_layer2 = spconv.SparseSequential(
            SparseBasicBlock23(32, norm_eps, norm_mom),
            SparseBasicBlock23(32, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="spconv2"),
                nn.BatchNorm1d(64, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )

        # encoder_layer3: BasicBlock(64,64) x2 + SparseConv3d(64,128,3,stride=2,padding=[1,1,0])
        self.encoder_layer3 = spconv.SparseSequential(
            SparseBasicBlock23(64, norm_eps, norm_mom),
            SparseBasicBlock23(64, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(64, 128, 3, stride=2, padding=[1, 1, 0], bias=False, indice_key="spconv3"),
                nn.BatchNorm1d(128, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )

        # encoder_layer4: BasicBlock(128,128) x2 (no downsampling)
        self.encoder_layer4 = spconv.SparseSequential(
            SparseBasicBlock23(128, norm_eps, norm_mom),
            SparseBasicBlock23(128, norm_eps, norm_mom),
        )

        # conv_out: SparseConv3d(128, 128, (1,1,3), stride=(1,1,2)) + BN + ReLU
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, (1, 1, 3), stride=(1, 1, 2), padding=0, bias=False, indice_key="spconv_down2"),
            nn.BatchNorm1d(128, eps=norm_eps, momentum=norm_mom),
            nn.ReLU(inplace=True),
        )

        self.sparse_shape = [1440, 1440, 41]

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv_input(input_sp)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        out = self.conv_out(x)
        spatial_features = out.dense()
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features


# ============================================================================
# Weight mapping: spconv 2.1 state_dict -> spconv 2.3 state_dict
# ============================================================================

def build_weight_mapping():
    """Build mapping from BEVFusion checkpoint keys to SparseEncoder23 keys.

    BEVFusion (spconv 2.1) key pattern:
        encoders.lidar.backbone.conv_input.0.weight  (SubMConv3d, shape [k,k,k,in,out])
        encoders.lidar.backbone.conv_input.1.weight  (BN)
        encoders.lidar.backbone.encoder_layers.encoder_layer1.0.conv1.weight  (BasicBlock conv1)
        encoders.lidar.backbone.encoder_layers.encoder_layer1.0.norm1.weight  (BasicBlock bn1)
        ...

    SparseEncoder23 key pattern:
        conv_input.0.weight  (SubMConv3d, shape [out,k,k,k,in])
        conv_input.1.weight  (BN)
        encoder_layer1.0.conv1.weight  (BasicBlock conv1)
        encoder_layer1.0.bn1.weight    (BasicBlock bn1)
        ...

    Key differences:
    1. Prefix: "encoders.lidar.backbone." -> ""
    2. "encoder_layers.encoder_layerN" -> "encoder_layerN"
    3. spconv weight shape: [k,k,k,in,out] -> [out,k,k,k,in] (permute needed)
    4. BasicBlock: "norm1" -> "bn1", "norm2" -> "bn2"
    """
    mapping = {}
    prefix = "encoders.lidar.backbone."

    # conv_input: 0=SubMConv3d, 1=BN, 2=ReLU
    mapping[f"{prefix}conv_input.0.weight"] = ("conv_input.0.weight", True)  # needs permute
    mapping[f"{prefix}conv_input.1.weight"] = ("conv_input.1.weight", False)
    mapping[f"{prefix}conv_input.1.bias"] = ("conv_input.1.bias", False)
    mapping[f"{prefix}conv_input.1.running_mean"] = ("conv_input.1.running_mean", False)
    mapping[f"{prefix}conv_input.1.running_var"] = ("conv_input.1.running_var", False)
    mapping[f"{prefix}conv_input.1.num_batches_tracked"] = ("conv_input.1.num_batches_tracked", False)

    # encoder_layers
    layer_configs = [
        # (layer_idx, num_basicblocks, in_ch, out_ch, has_downsample)
        (1, 2, 16, 32, True),
        (2, 2, 32, 64, True),
        (3, 2, 64, 128, True),
        (4, 2, 128, None, False),
    ]

    for layer_idx, num_blocks, in_ch, out_ch, has_down in layer_configs:
        src_prefix = f"{prefix}encoder_layers.encoder_layer{layer_idx}"
        dst_prefix = f"encoder_layer{layer_idx}"

        for block_idx in range(num_blocks):
            # BasicBlock: conv1, norm1->bn1, conv2, norm2->bn2
            for conv_name in ["conv1", "conv2"]:
                src_key = f"{src_prefix}.{block_idx}.{conv_name}.weight"
                dst_key = f"{dst_prefix}.{block_idx}.{conv_name}.weight"
                mapping[src_key] = (dst_key, True)  # needs permute

            for bn_idx, (src_bn, dst_bn) in enumerate([("bn1", "bn1"), ("bn2", "bn2")]):
                for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
                    src_key = f"{src_prefix}.{block_idx}.{src_bn}.{param}"
                    dst_key = f"{dst_prefix}.{block_idx}.{dst_bn}.{param}"
                    mapping[src_key] = (dst_key, False)

        if has_down:
            # Downsampling SparseConvModule: last child in the stage
            down_idx = num_blocks  # e.g., 2 for 2 basicblocks
            # SparseConv3d weight
            src_key = f"{src_prefix}.{down_idx}.0.weight"
            dst_key = f"{dst_prefix}.{down_idx}.0.weight"
            mapping[src_key] = (dst_key, True)  # needs permute
            # BN
            for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
                src_key = f"{src_prefix}.{down_idx}.1.{param}"
                dst_key = f"{dst_prefix}.{down_idx}.1.{param}"
                mapping[src_key] = (dst_key, False)

    # conv_out: SparseConv3d + BN + ReLU
    mapping[f"{prefix}conv_out.0.weight"] = ("conv_out.0.weight", True)
    mapping[f"{prefix}conv_out.1.weight"] = ("conv_out.1.weight", False)
    mapping[f"{prefix}conv_out.1.bias"] = ("conv_out.1.bias", False)
    mapping[f"{prefix}conv_out.1.running_mean"] = ("conv_out.1.running_mean", False)
    mapping[f"{prefix}conv_out.1.running_var"] = ("conv_out.1.running_var", False)
    mapping[f"{prefix}conv_out.1.num_batches_tracked"] = ("conv_out.1.num_batches_tracked", False)

    return mapping


def permute_spconv_weight(w):
    """Convert spconv 2.1 weight [k,k,k,in,out] to spconv 2.3 weight [out,k,k,k,in]."""
    return w.permute(4, 0, 1, 2, 3).contiguous()


def load_weights(model, ckpt_path):
    """Load weights from BEVFusion checkpoint into SparseEncoder23."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    mapping = build_weight_mapping()

    new_state = {}
    for src_key, (dst_key, needs_permute) in mapping.items():
        if src_key not in state_dict:
            print(f"  WARNING: {src_key} not found in checkpoint")
            continue
        w = state_dict[src_key]
        if needs_permute:
            w = permute_spconv_weight(w)
        new_state[dst_key] = w

    # Load into model
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print(f"  Loaded {len(new_state)} parameters")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="BEVFusion checkpoint path")
    parser.add_argument("--verify-dir", default="lidar_verify_tensors", help="Directory with verification tensors")
    parser.add_argument("--output", default="lidar_spconv23_output.pt", help="Output tensor path")
    args = parser.parse_args()

    print("Phase 4: LiDAR SparseEncoder via spconv 2.3")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Verify dir: {args.verify_dir}")

    # Build model
    model = SparseEncoder23()
    print(f"  Model built: {sum(p.numel() for p in model.parameters())} parameters")

    # Load weights
    print("Loading weights...")
    load_weights(model, args.ckpt)
    model.eval().cuda().half()

    # Load verification data
    voxels_path = os.path.join(args.verify_dir, "voxel_features.pt")
    coors_path = os.path.join(args.verify_dir, "coors.pt")
    ref_output_path = os.path.join(args.verify_dir, "pytorch_output.pt")

    voxels = torch.load(voxels_path, map_location="cpu").cuda()  # [N, 5] fp16
    coors = torch.load(coors_path, map_location="cpu").cuda()    # [N, 4] int32
    print(f"  Input: voxels={voxels.shape} ({voxels.dtype}), coors={coors.shape}")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(voxels, coors, batch_size=1)
    print(f"  Output: {output.shape} ({output.dtype})")

    # Save output
    torch.save(output.cpu().float(), args.output)
    print(f"  Saved: {args.output}")

    # Compare with reference
    if os.path.exists(ref_output_path):
        ref = torch.load(ref_output_path, map_location="cpu").float()
        out_f = output.cpu().float()
        cos = torch.nn.functional.cosine_similarity(
            out_f.flatten().unsqueeze(0),
            ref.flatten().unsqueeze(0)
        ).item()
        mae = (out_f - ref).abs().max().item()
        print(f"\n  Verification vs PyTorch (spconv 2.1):")
        print(f"    cosine_sim  : {cos:.6f}")
        print(f"    max_abs_err : {mae:.6f}")
        if cos > 0.999:
            print(f"    PASSED")
        else:
            print(f"    FAILED (threshold: 0.999)")
    else:
        print(f"  No reference output found at {ref_output_path}")


if __name__ == "__main__":
    main()
