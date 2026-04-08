"""
Phase 3: 导出 vtransform 的 depthnet 部分为 ONNX + 完整 bev_pool_v2 验证。

vtransform 拆分为两部分:
1. depthnet 引擎: dtransform + depthnet + outer_product + flatten (可导出 ONNX/TRT)
2. bev_pool_v2 Plugin: interval-sum 核心计算 (TRT Plugin)

中间的索引预计算在 PyTorch 端完成（验证时每帧动态，部署时离线一次性）。

用法:
    # 导出 depthnet ONNX
    python tools/export_utils/export_vtransform.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt path/to/ptq_minmax_model.pth \
        --output vtransform_depthnet.onnx

    # 验证 bev_pool_v2 等价性（对比原始 bev_pool）
    python tools/export_utils/export_vtransform.py \
        --config ... --ckpt ... --verify-only
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

# ── Q/DQ symbolic 注册 ──────────────────────────────────────────────────────
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper


def _learnable_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
    q = g.op("QuantizeLinear", x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq


def _fixed_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max):
    q = g.op("QuantizeLinear", x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq


def _learnable_per_channel_qdq(g, x, scale, zero_point, axis, quant_min, quant_max, grad_factor):
    axis_i = symbolic_helper._get_const(axis, 'i', 'axis')
    q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
    return dq


def _fixed_per_channel_qdq(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
    axis_i = symbolic_helper._get_const(ch_axis, 'i', 'ch_axis')
    q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
    return dq


register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _learnable_per_tensor_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_tensor_affine', _fixed_per_tensor_qdq, 13)
register_custom_op_symbolic('::_fake_quantize_learnable_per_channel_affine', _learnable_per_channel_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_channel_affine', _fixed_per_channel_qdq, 13)

import mqbench.custom_symbolic_opset  # noqa

try:
    from mqbench.fake_quantize.lsq import FakeQuantizeLearnablePerchannelAffine

    def _perchannel_symbolic(g, x, scale, zero_point, axis, quant_min, quant_max, grad_factor):
        axis_i = symbolic_helper._get_const(axis, 'i', 'axis')
        q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
        dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
        return dq

    FakeQuantizeLearnablePerchannelAffine.symbolic = staticmethod(_perchannel_symbolic)
except Exception:
    pass

# ── 主逻辑 ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import onnx
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval
from mqbench.utils.state import enable_quantization
from tools.quant_ptq_minmax import build_ptq_model


class DepthNetWrapper(nn.Module):
    """Wraps the depthnet portion of vtransform for ONNX export.

    Covers: dtransform + depthnet + softmax + outer_product + flatten
    Output: [N_total, C] features ready for bev_pool_v2

    This is the traceable part of vtransform. The bev_pool part
    (index computation + interval-sum) is handled separately.
    """

    def __init__(self, vtransform):
        super().__init__()
        self.vtransform = vtransform
        self.C = vtransform.C
        self.D = vtransform.D

    def forward(self, img, depth_input):
        """
        Args:
            img: [B, N, C_in, fH, fW] image features from camera backbone
            depth_input: [B, N, 1, H_img, W_img] depth map (for DepthLSSTransform)
                         or unused (for LSSTransform)
        Returns:
            x: [B*N*D*fH*fW, C] flattened features for bev_pool
        """
        # get_cam_feats returns [B, N, D, fH, fW, C]
        # DepthLSSTransform.get_cam_feats(img, depth, mats_dict=None)
        # LSSTransform.get_cam_feats(img)
        if hasattr(self.vtransform, 'dtransform'):
            # DepthLSSTransform path
            x = self.vtransform.get_cam_feats(img, depth_input)
        else:
            # LSSTransform path
            x = self.vtransform.get_cam_feats(img)

        # [B, N, D, fH, fW, C] -> [B*N*D*fH*fW, C]
        B, N, D, fH, fW, C = x.shape
        x = x.reshape(B * N * D * fH * fW, C)
        return x


def verify_bev_pool_equivalence(model, logger):
    """Verify bev_pool_v2 produces same output as original bev_pool."""
    vtransform = model.encoders.camera.vtransform

    # Create dummy geometry (need real camera params for meaningful test)
    B, N, D = 1, 6, vtransform.D
    fH, fW = vtransform.feature_size
    C = vtransform.C

    device = vtransform.bx.device

    # Dummy camera params (identity transforms)
    camera2lidar_rots = torch.eye(3, device=device).expand(B, N, 3, 3)
    camera2lidar_trans = torch.zeros(B, N, 3, device=device)
    intrins = torch.eye(3, device=device).expand(B, N, 3, 3).clone()
    intrins[:, :, 0, 0] = 1266.0
    intrins[:, :, 1, 1] = 1266.0
    intrins[:, :, 0, 2] = 352.0
    intrins[:, :, 1, 2] = 128.0
    post_rots = torch.eye(3, device=device).expand(B, N, 3, 3)
    post_trans = torch.zeros(B, N, 3, device=device)

    with torch.no_grad():
        geom = vtransform.get_geometry(
            camera2lidar_rots, camera2lidar_trans,
            intrins, post_rots, post_trans,
        )

        # Random features
        x = torch.randn(B, N, D, fH, fW, C, device=device)

        # Original bev_pool
        out_orig = vtransform.bev_pool(geom, x)

        # New bev_pool_v2 path
        indices = vtransform.precompute_bev_indices(geom, B)
        out_v2 = vtransform.bev_pool_with_indices(x, indices)

    # Compare
    out_orig_flat = out_orig.float().cpu().numpy().flatten()
    out_v2_flat = out_v2.float().cpu().numpy().flatten()

    cos_sim = float(np.dot(out_orig_flat, out_v2_flat) /
                    (np.linalg.norm(out_orig_flat) * np.linalg.norm(out_v2_flat) + 1e-8))
    max_err = float(np.abs(out_orig_flat - out_v2_flat).max())

    logger.info(f"bev_pool vs bev_pool_v2:")
    logger.info(f"  cosine_sim: {cos_sim:.6f} (threshold > 0.999)")
    logger.info(f"  max_abs_err: {max_err:.6f}")
    logger.info(f"  output shape: orig={out_orig.shape}, v2={out_v2.shape}")

    if cos_sim > 0.999:
        logger.info("  PASS")
    else:
        logger.error("  FAIL - outputs differ significantly!")
    return cos_sim


def export_depthnet(vtransform, output_path, logger):
    """Export depthnet portion as ONNX."""
    wrapper = DepthNetWrapper(vtransform)
    wrapper.eval()

    B, N = 1, 6
    C_in = vtransform.in_channels
    fH, fW = vtransform.feature_size
    H_img, W_img = vtransform.image_size

    dummy_img = torch.randn(B, N, C_in, fH, fW)
    dummy_depth = torch.randn(B, N, 1, H_img, W_img)

    # Move to same device as model
    device = next(vtransform.parameters()).device
    wrapper = wrapper.to(device)
    dummy_img = dummy_img.to(device)
    dummy_depth = dummy_depth.to(device)

    with torch.no_grad():
        out = wrapper(dummy_img, dummy_depth)
        logger.info(f"DepthNet output shape: {out.shape}")

    logger.info(f"Exporting depthnet to: {output_path}")
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_depth),
        output_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image_features", "depth_input"],
        output_names=["pooling_features"],
        dynamic_axes=None,
    )

    # Validate ONNX
    m = onnx.load(output_path)
    qdq = [n for n in m.graph.node
           if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    fakeq = [n for n in m.graph.node
             if any(x in n.op_type for x in ["FakeQuant", "Affine", "Learnable"])]

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"  File size: {size_mb:.1f} MB")
    logger.info(f"  Q/DQ nodes: {len(qdq)}")
    logger.info(f"  Unconverted FakeQuant: {len(fakeq)}")
    logger.info(f"  Total nodes: {len(m.graph.node)}")

    if len(fakeq) > 0:
        logger.warning("Unconverted FakeQuant nodes found!")
        for n in fakeq[:5]:
            logger.warning(f"  {n.op_type}")


def main():
    parser = argparse.ArgumentParser(description="导出 vtransform depthnet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="vtransform_depthnet.onnx")
    parser.add_argument("--verify-only", action="store_true",
                        help="只验证 bev_pool_v2 等价性，不导出 ONNX")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)

    # 加载配置和模型
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.verify_only:
        # 验证等价性不需要量化模型，用 FP32 即可
        from mmdet3d.models import build_model
        model = build_model(cfg.model).cuda().eval()
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        verify_bev_pool_equivalence(model, logger)
        return

    # 导出模式：构建量化模型，加载 PTQ 权重（含 KL 校准的 scale/zero_point），
    # 导出带 Q/DQ 节点的 INT8 ONNX。
    model, _, _ = build_ptq_model(cfg, logger)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # 修复 shape 不匹配：FakeQuant 的 scale/zero_point 在 build_ptq_model 中
    # 初始化为 [1]，但 PTQ checkpoint 中是 per-channel [out_channels]。
    # 需要先 resize 模型参数再加载。
    model_sd = model.state_dict()
    params_to_resize = {}
    for k, v in state_dict.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            if v.numel() == model_sd[k].numel():
                # 同 numel 不同 shape（如 [] vs [1]）：直接 reshape
                state_dict[k] = v.reshape(model_sd[k].shape)
            else:
                # 不同 numel（如 scale [1] vs [8]）：需要 resize 模型参数
                params_to_resize[k] = v

    # Resize model parameters to match checkpoint
    for k, ckpt_val in params_to_resize.items():
        parts = k.split('.')
        obj = model
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            elif p.isdigit() and hasattr(obj, '__getitem__'):
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        param_name = parts[-1]
        old = getattr(obj, param_name)
        if isinstance(old, torch.nn.Parameter):
            setattr(obj, param_name, torch.nn.Parameter(
                ckpt_val.clone(), requires_grad=old.requires_grad))
        else:
            setattr(obj, param_name, ckpt_val.clone())

    model.load_state_dict(state_dict, strict=False)
    enable_quantization(model)
    model.eval()

    # 只把 vtransform 移到 GPU，节省显存
    vtransform = model.encoders.camera.vtransform.cuda().eval()
    # 释放其余模块
    for name in list(model.encoders._modules.keys()):
        if name != 'camera':
            delattr(model.encoders, name)
    for name in ['fuser', 'decoder', 'heads']:
        if hasattr(model, name):
            delattr(model, name)
    torch.cuda.empty_cache()

    # Export depthnet
    export_depthnet(vtransform, args.output, logger)


if __name__ == "__main__":
    main()
