"""
Phase 4b: 导出 Fuser + Decoder (SECOND + SECONDFPN) 为 ONNX → TRT 引擎。

这三个模块都是标准 Conv2d/ConvTranspose2d，可以直接 torch.onnx.export。
合并为一个子模型导出，减少引擎间数据传输。

输入: camera_bev [1, 80, 180, 180] + lidar_bev [1, 256, 180, 180]
输出: neck_features [1, 512, 180, 180]

支持两种量化模式:
  --int8: 从 PTQ checkpoint 加载量化参数，导出 Q/DQ 节点（全 INT8）
  --w8a16: 仅权重 Q/DQ，激活保持 FP16（W8A16 版本）
  默认: FP16（无量化）

用法:
    # FP16
    python tools/export_utils/export_fuser_decoder.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --output fuser_decoder_fp16.onnx

    # INT8 (全量化)
    python tools/export_utils/export_fuser_decoder.py \
        --config ... --ckpt pretrained/ptq_minmax_model.pth \
        --output fuser_decoder_int8.onnx --int8
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

# ── Q/DQ symbolic 注册（必须在 import mqbench 之前）──────────────────────────
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


# Step 1: ATen domain symbolics
register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _learnable_per_tensor_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_tensor_affine', _fixed_per_tensor_qdq, 13)
register_custom_op_symbolic('::_fake_quantize_learnable_per_channel_affine', _learnable_per_channel_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_channel_affine', _fixed_per_channel_qdq, 13)

# Step 2: Load MQBench custom symbolics
import mqbench.custom_symbolic_opset  # noqa

# Step 3: Override autograd Function symbolic for per-channel
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

# ── 导出逻辑 ────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import onnx
import numpy as np
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval


class FuserDecoderWrapper(nn.Module):
    """Wrapper that combines Fuser + SECOND + SECONDFPN into one exportable module."""

    def __init__(self, fuser, decoder_backbone, decoder_neck):
        super().__init__()
        self.fuser = fuser
        self.decoder_backbone = decoder_backbone
        self.decoder_neck = decoder_neck

    def forward(self, camera_bev, lidar_bev):
        # Fuser: cat + conv + bn + relu
        fused = self.fuser([camera_bev, lidar_bev])
        # SECOND: multi-scale conv blocks
        x = self.decoder_backbone(fused)
        # SECONDFPN: upsample + concat
        x = self.decoder_neck(x)
        # SECONDFPN returns a list, take first element
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", default="fuser_decoder_fp16.onnx")
    p.add_argument("--int8", action="store_true", help="Export with INT8 Q/DQ nodes")
    p.add_argument("--verify", action="store_true", help="Run PyTorch inference and save output for verification")
    args = p.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("Phase 4b: Fuser + Decoder ONNX Export")

    # Load config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    if args.int8:
        # INT8: build PTQ model with FakeQuant
        from mqbench.utils.state import enable_quantization
        from tools.quant_ptq_minmax import build_ptq_model
        model, _, _ = build_ptq_model(cfg, logger)

        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Fix shape mismatch: FakeQuant scale/zero_point initialized as [1]
        # but PTQ checkpoint has per-channel [out_channels].
        model_sd = model.state_dict()
        params_to_resize = {}
        for k, v in state_dict.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                if v.numel() == model_sd[k].numel():
                    state_dict[k] = v.reshape(model_sd[k].shape)
                else:
                    params_to_resize[k] = v

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
    else:
        # FP16/FP32: build normal model
        from mmdet3d.models import build_model
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval().cuda()

    # Extract submodules
    wrapper = FuserDecoderWrapper(
        model.fuser,
        model.decoder["backbone"],
        model.decoder["neck"],
    ).eval().cuda()

    # Dummy inputs
    camera_bev = torch.randn(1, 80, 180, 180).cuda()
    lidar_bev = torch.randn(1, 256, 180, 180).cuda()

    # Verify forward works
    with torch.no_grad():
        out = wrapper(camera_bev, lidar_bev)
    logger.info(f"Forward OK: output shape = {out.shape}")

    if args.verify:
        torch.save(out.cpu(), args.output.replace(".onnx", "_pytorch_output.pt"))
        logger.info(f"Saved PyTorch output for verification")

    # Export ONNX
    logger.info(f"Exporting to {args.output}...")
    torch.onnx.export(
        wrapper,
        (camera_bev, lidar_bev),
        args.output,
        opset_version=13,
        input_names=["camera_bev", "lidar_bev"],
        output_names=["neck_features"],
        dynamic_axes=None,
        do_constant_folding=True,
        verbose=False,
    )

    # Summary
    m = onnx.load(args.output)
    qdq = [n for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
    fakeq = [n for n in m.graph.node if "FakeQuant" in n.op_type]
    all_ops = sorted(set(n.op_type for n in m.graph.node))
    file_size = os.path.getsize(args.output) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  ONNX: {args.output} ({file_size:.1f} MB)")
    print(f"  Total nodes: {len(m.graph.node)}")
    print(f"  Q/DQ nodes: {len(qdq)}")
    print(f"  FakeQuant residual: {len(fakeq)}")
    print(f"  Op types: {all_ops}")
    print(f"  Output shape: {list(out.shape)}")
    print(f"{'='*60}")

    if len(fakeq) > 0:
        print("⚠️  FakeQuant nodes remain — check symbolic registration")
    if args.int8 and len(qdq) == 0:
        print("⚠️  No Q/DQ nodes in INT8 mode — check symbolic registration")


if __name__ == "__main__":
    main()
