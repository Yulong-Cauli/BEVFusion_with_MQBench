"""
Phase 4: 验证 LiDAR SparseEncoder 导出的 ONNX 精度。

对比 PyTorch FP32 输出 vs libspconv.so FP16 推理输出。
需要先编译 pyscn Python 绑定。

用法：
    # 1. 保存真实 lidar 输入数据
    python tools/export_utils/verify_lidar.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --save-input --tensor-dir lidar_verify_tensors

    # 2. 验证 libspconv 推理精度（需要 pyscn）
    python tools/export_utils/verify_lidar.py \
        --onnx lidar_backbone_fp16.onnx \
        --tensor-dir lidar_verify_tensors \
        --threshold 0.999
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

import torch
import numpy as np


def save_real_input(args):
    """Save real lidar voxel features and coordinates from dataset."""
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import get_root_logger, recursive_eval
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataloader, build_dataset

    logger = get_root_logger(log_level=logging.INFO)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Build model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    # Filter out quant keys
    fp32_state = {k: v for k, v in state_dict.items()
                  if "fake_quant" not in k and "activation_post_process" not in k}
    model.load_state_dict(fp32_state, strict=False)
    model.eval().cuda()

    # Build dataset
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    # Get one sample
    for data in dataloader:
        break

    logger.info("Running PyTorch inference for verification...")
    backbone = model.encoders.lidar.backbone

    with torch.no_grad():
        # Handle DataContainer
        from mmcv.parallel import DataContainer
        points_data = data["points"]
        if isinstance(points_data, DataContainer):
            points_data = points_data.data
        if isinstance(points_data, list):
            points_data = points_data[0]
        if isinstance(points_data, list):
            points_data = points_data[0]

        # Use model's voxelize method
        feats, coords, sizes = model.voxelize([points_data.cuda()], "lidar")
        batch_size = coords[-1, 0].item() + 1

        logger.info(f"  voxel_features: {feats.shape} ({feats.dtype})")
        logger.info(f"  coords: {coords.shape} ({coords.dtype})")
        logger.info(f"  batch_size: {batch_size}")

        # Run backbone with sizes (for voxelize_reduce)
        output = backbone(feats, coords, batch_size, sizes=sizes)
        logger.info(f"  output: {output.shape} ({output.dtype})")

    # Save tensors
    os.makedirs(args.tensor_dir, exist_ok=True)

    # Save as FP16 for libspconv (it expects FP16 input)
    voxel_fp16 = feats.cpu().half()
    coors_int32 = coords.cpu().int()
    output_fp32 = output.cpu().float()

    torch.save(voxel_fp16, os.path.join(args.tensor_dir, "voxel_features.pt"))
    torch.save(coors_int32, os.path.join(args.tensor_dir, "coors.pt"))
    torch.save(output_fp32, os.path.join(args.tensor_dir, "pytorch_output.pt"))

    # Also save as numpy for pyscn
    np.save(os.path.join(args.tensor_dir, "voxel_features.npy"),
            voxel_fp16.numpy())
    np.save(os.path.join(args.tensor_dir, "coors.npy"),
            coors_int32.numpy())

    logger.info(f"Saved verification tensors to {args.tensor_dir}/")
    logger.info(f"  voxel_features: {voxel_fp16.shape}")
    logger.info(f"  coors: {coors_int32.shape}")
    logger.info(f"  pytorch_output: {output_fp32.shape}")

    # Print sparse_shape for reference
    logger.info(f"  sparse_shape: {backbone.sparse_shape}")


def verify_with_pyscn(args):
    """Verify ONNX output against PyTorch using pyscn."""
    logger = logging.getLogger("verify_lidar")
    logging.basicConfig(level=logging.INFO)

    # Load saved tensors
    pytorch_output = torch.load(os.path.join(args.tensor_dir, "pytorch_output.pt"))
    voxel_features = np.load(os.path.join(args.tensor_dir, "voxel_features.npy"))
    coors = np.load(os.path.join(args.tensor_dir, "coors.npy"))

    logger.info(f"Loaded tensors from {args.tensor_dir}/")
    logger.info(f"  voxel_features: {voxel_features.shape}")
    logger.info(f"  coors: {coors.shape}")
    logger.info(f"  pytorch_output: {pytorch_output.shape}")

    # Load pyscn
    pyscn_path = os.path.join(
        "temp/Lidar_AI_Solution/libraries/3DSparseConvolution/tool"
    )
    sys.path.insert(0, pyscn_path)
    try:
        import pyscn
    except ImportError:
        logger.error("pyscn not found. Please compile it first:")
        logger.error("  cd temp/Lidar_AI_Solution/libraries/3DSparseConvolution")
        logger.error("  CUDA_HOME=/usr/local/cuda SPCONV_CUDA_VERSION=11.4 make pyscn -j")
        return False

    # Run inference
    precision = "int8" if "int8" in args.onnx else "fp16"
    logger.info(f"Loading ONNX: {args.onnx} (precision={precision})")

    pyscn.set_verbose(True)
    model = pyscn.SCNModel(args.onnx, precision)

    # grid_size matches sparse_shape (H, W, D) = (1440, 1440, 41)
    # But libspconv expects [Z, Y, X] or [X, Y, Z] depending on format
    grid_size = [1440, 1440, 41]  # sparse_shape from config

    features_out, indices_out = model.forward(
        voxel_features, coors, grid_size, 0
    )

    logger.info(f"  libspconv output: {features_out.shape}")

    # Compare
    spconv_output = torch.from_numpy(features_out).float()
    pytorch_flat = pytorch_output.flatten()
    spconv_flat = spconv_output.flatten()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        pytorch_flat.unsqueeze(0), spconv_flat.unsqueeze(0)
    ).item()

    # Other metrics
    abs_diff = (pytorch_flat - spconv_flat).abs()
    max_abs_err = abs_diff.max().item()
    rmse = abs_diff.pow(2).mean().sqrt().item()

    logger.info(f"\n{'='*60}")
    logger.info(f"Precision Verification Results")
    logger.info(f"  cosine_sim:  {cos_sim:.6f}")
    logger.info(f"  max_abs_err: {max_abs_err:.6f}")
    logger.info(f"  RMSE:        {rmse:.6f}")
    logger.info(f"  threshold:   {args.threshold}")

    if cos_sim >= args.threshold:
        logger.info(f"  Result: PASS")
    else:
        logger.info(f"  Result: FAIL")
    logger.info(f"{'='*60}")

    del model
    return cos_sim >= args.threshold


def main():
    parser = argparse.ArgumentParser(description="Verify LiDAR SparseEncoder ONNX precision")
    parser.add_argument("--config", help="Model config yaml (for --save-input)")
    parser.add_argument("--ckpt", help="Checkpoint path (for --save-input)")
    parser.add_argument("--save-input", action="store_true", help="Save real lidar input tensors")
    parser.add_argument("--onnx", help="ONNX path (for verification)")
    parser.add_argument("--tensor-dir", default="lidar_verify_tensors", help="Tensor directory")
    parser.add_argument("--threshold", type=float, default=0.999, help="Cosine similarity threshold")
    args = parser.parse_args()

    if args.save_input:
        if not args.config or not args.ckpt:
            parser.error("--save-input requires --config and --ckpt")
        save_real_input(args)
    elif args.onnx:
        ok = verify_with_pyscn(args)
        exit(0 if ok else 1)
    else:
        parser.error("Specify --save-input or --onnx")


if __name__ == "__main__":
    main()
