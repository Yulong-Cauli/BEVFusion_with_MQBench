"""
Extract LiDAR backbone weights from PTQ checkpoint to .npy format.

This enables zero-PyTorch deployment by avoiding torch.load() at runtime.
"""

import os
import sys
import argparse
import numpy as np
import torch


def extract_lidar_weights(ptq_ckpt_path, output_dir, dtype="fp16", int8_storage=False):
    """
    Extract LiDAR backbone weights from PTQ checkpoint.

    Args:
        ptq_ckpt_path: Path to ptq_minmax_model.pth
        output_dir: Directory to save .npy files
        dtype: "fp16" or "fp32" for weight storage
        int8_storage: If True, store weights as INT8 (with separate scale files)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {ptq_ckpt_path}")
    ckpt = torch.load(ptq_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    prefix = "encoders.lidar.backbone."

    # Layer definitions matching TVSparseEncoder
    conv_layers = [
        ("conv_input", "conv_input.0", 3, 1, 1, True, None),
        ("encoder_layer1.0", "encoder_layer1.0.conv1", 3, 1, 1, True, "subm1"),
        ("encoder_layer1.0", "encoder_layer1.0.conv2", 3, 1, 1, True, "subm1"),
        ("encoder_layer1.1", "encoder_layer1.1.conv1", 3, 1, 1, True, "subm1"),
        ("encoder_layer1.1", "encoder_layer1.1.conv2", 3, 1, 1, True, "subm1"),
        ("encoder_layer1.2", "encoder_layer1.2.0", 3, 2, 1, False, "spconv2"),
        ("encoder_layer2.0", "encoder_layer2.0.conv1", 3, 1, 1, True, "subm2"),
        ("encoder_layer2.0", "encoder_layer2.0.conv2", 3, 1, 1, True, "subm2"),
        ("encoder_layer2.1", "encoder_layer2.1.conv1", 3, 1, 1, True, "subm2"),
        ("encoder_layer2.1", "encoder_layer2.1.conv2", 3, 1, 1, True, "subm2"),
        ("encoder_layer2.2", "encoder_layer2.2.0", 3, 2, 1, False, "spconv3"),
        ("encoder_layer3.0", "encoder_layer3.0.conv1", 3, 1, 1, True, "subm3"),
        ("encoder_layer3.0", "encoder_layer3.0.conv2", 3, 1, 1, True, "subm3"),
        ("encoder_layer3.1", "encoder_layer3.1.conv1", 3, 1, 1, True, "subm3"),
        ("encoder_layer3.1", "encoder_layer3.1.conv2", 3, 1, 1, True, "subm3"),
        ("encoder_layer3.2", "encoder_layer3.2.0", 3, 2, (0, 1, 1), False, "spconv4"),
        ("encoder_layer4.0", "encoder_layer4.0.conv1", 3, 1, 1, True, "subm4"),
        ("encoder_layer4.0", "encoder_layer4.0.conv2", 3, 1, 1, True, "subm4"),
        ("encoder_layer4.1", "encoder_layer4.1.conv1", 3, 1, 1, True, "subm4"),
        ("encoder_layer4.1", "encoder_layer4.1.conv2", 3, 1, 1, True, "subm4"),
        ("conv_out", "conv_out.0", (1, 1, 3), (1, 1, 2), 0, False, "spconv_down2"),
    ]

    extracted = []
    total_params = 0

    for layer_name, conv_name, ksize, stride, padding, subm, ikey in conv_layers:
        if conv_name.startswith("encoder_layer"):
            ckpt_conv_name = f"encoder_layers.{conv_name}"
        else:
            ckpt_conv_name = conv_name

        src_key = f"{prefix}{ckpt_conv_name}.conv.weight"
        if src_key not in state_dict:
            src_key = f"{prefix}{ckpt_conv_name}.weight"
        if src_key not in state_dict:
            print(f"  Warning: {src_key} not found, skipping")
            continue

        w = state_dict[src_key].numpy()
        # spconv 2.3 layout: [out, k, k, k, in] -> no transpose needed for numpy
        # But TVSparseEncoder expects [k,k,k,in,out], so we transpose
        w = np.ascontiguousarray(np.transpose(w, (4, 0, 1, 2, 3)))

        # Weight fake quantization scale
        wfq_scale = state_dict.get(f"{prefix}{ckpt_conv_name}.weight_fake_quant.scale")

        if int8_storage and wfq_scale is not None:
            # Store as INT8 with separate scale
            s = wfq_scale.numpy().astype(np.float32).reshape(-1, 1, 1, 1, 1)
            w_int8 = np.clip(np.round(w.astype(np.float32) / s), -127, 127).astype(np.int8)
            weight_file = f"{conv_name.replace('.', '_')}_weight_int8.npy"
            scale_file = f"{conv_name.replace('.', '_')}_scale.npy"
            np.save(os.path.join(output_dir, weight_file), w_int8)
            np.save(os.path.join(output_dir, scale_file), wfq_scale.numpy().astype(np.float32))
            w_to_save = w_int8
            total_params += w_int8.size
        else:
            # Store as FP16/FP32 (already fake-quantized)
            if wfq_scale is not None:
                s = wfq_scale.numpy().astype(np.float32).reshape(-1, 1, 1, 1, 1)
                w = np.clip(np.round(w.astype(np.float32) / s), -127, 127) * s

            if dtype == "fp16":
                w = w.astype(np.float16)

            weight_file = f"{conv_name.replace('.', '_')}_weight_{dtype}.npy"
            np.save(os.path.join(output_dir, weight_file), w)
            w_to_save = w
            total_params += w.size

        # BN params
        if ckpt_conv_name.endswith(".0") and (".2.0" in ckpt_conv_name
                                         or "conv_input" in ckpt_conv_name or "conv_out" in ckpt_conv_name):
            bn_name = ckpt_conv_name[:-1] + "1"
        else:
            bn_name = ckpt_conv_name.replace("conv", "bn")
        bn_src = f"{prefix}{bn_name}"
        bn_keys = ["weight", "bias", "running_mean", "running_var"]
        has_bn = all(f"{bn_src}.{k}" in state_dict for k in bn_keys)

        bn_params = {}
        if has_bn:
            gamma = state_dict[f"{bn_src}.weight"].numpy().astype(np.float32)
            beta = state_dict[f"{bn_src}.bias"].numpy().astype(np.float32)
            running_mean = state_dict[f"{bn_src}.running_mean"].numpy().astype(np.float32)
            running_var = state_dict[f"{bn_src}.running_var"].numpy().astype(np.float32)

            eps = 1e-3
            scale_np = gamma / np.sqrt(running_var + eps)
            shift_np = beta - running_mean * scale_np

            bn_params = {
                "scale": scale_np.astype(np.float32),
                "shift": shift_np.astype(np.float32),
            }

            bn_scale_file = f"{conv_name.replace('.', '_')}_bn_scale.npy"
            bn_shift_file = f"{conv_name.replace('.', '_')}_bn_shift.npy"
            np.save(os.path.join(output_dir, bn_scale_file), bn_params["scale"])
            np.save(os.path.join(output_dir, bn_shift_file), bn_params["shift"])

        # Log2 activation quantization base
        log2_base = state_dict.get(f"{prefix}{ckpt_conv_name}.act_fake_quant.log2_base")
        log2_base_val = log2_base.item() if log2_base is not None else None

        layer_info = {
            "name": conv_name,
            "weight_file": weight_file,
            "has_bn": has_bn,
            "has_log2": log2_base_val is not None,
            "log2_base": log2_base_val,
        }
        extracted.append(layer_info)

        print(f"  {conv_name}: shape={w_to_save.shape}, dtype={w_to_save.dtype}")

    # Save metadata
    metadata = {
        "dtype": dtype,
        "int8_storage": int8_storage,
        "layers": extracted,
        "total_params": total_params,
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)

    # Calculate sizes
    total_size_mb = total_params * (1 if int8_storage else (2 if dtype == "fp16" else 4)) / 1024 / 1024
    print(f"\nExtraction complete:")
    print(f"  Total params: {total_params:,}")
    print(f"  Total size: ~{total_size_mb:.2f} MB")
    print(f"  Output dir: {output_dir}")
    print(f"  Files: {len(extracted)} layers")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract LiDAR weights to .npy format")
    parser.add_argument("--ptq-ckpt", default="pretrained/ptq_minmax_model.pth",
                       help="Path to PTQ checkpoint")
    parser.add_argument("--output-dir", default="pretrained/lidar_npy",
                       help="Output directory for .npy files")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                       help="Weight dtype (ignored if --int8-storage)")
    parser.add_argument("--int8-storage", action="store_true",
                       help="Store weights as INT8 with separate scale files")

    args = parser.parse_args()

    extract_lidar_weights(
        args.ptq_ckpt,
        args.output_dir,
        args.dtype,
        args.int8_storage
    )


if __name__ == "__main__":
    main()
