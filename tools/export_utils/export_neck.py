"""
Phase 6 Step 1: 导出 Camera Neck (GeneralizedLSSFPN) 为 ONNX → TRT 引擎。

GeneralizedLSSFPN 是标准 Conv2d + BN + ReLU + Bilinear Upsample + Cat，
量化路径一（fx 自动插桩），PTQ checkpoint 中已有 Q/DQ 参数。

输入: 3 个多尺度特征 (SwinT 输出)
  - scale1: [6, 192, 32, 88]
  - scale2: [6, 384, 16, 44]
  - scale3: [6, 768, 8, 22]
输出: [6, 256, 32, 88]

用法:
    # FP16
    python tools/export_utils/export_neck.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --output camera_neck_fp16.onnx

    # INT8 (从 PTQ checkpoint)
    python tools/export_utils/export_neck.py \
        --config ... --ckpt pretrained/ptq_minmax_model.pth \
        --output camera_neck_int8.onnx --int8
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

# ── 导出逻辑 ────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import onnx
import numpy as np
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval


class NeckWrapper(nn.Module):
    """Wrapper for GeneralizedLSSFPN that takes 3 separate tensors as input."""

    def __init__(self, neck):
        super().__init__()
        self.neck = neck

    def forward(self, feat1, feat2, feat3):
        out = self.neck([feat1, feat2, feat3])
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", default="camera_neck_fp16.onnx")
    p.add_argument("--int8", action="store_true", help="Export with INT8 Q/DQ nodes from PTQ checkpoint")
    p.add_argument("--verify", action="store_true", help="Save PyTorch output for verification")
    args = p.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("Phase 6 Step 1: Camera Neck ONNX Export")

    # Load config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    if args.int8:
        from mqbench.utils.state import enable_quantization
        from tools.quant_ptq_minmax import build_ptq_model
        model, _, _ = build_ptq_model(cfg, logger)

        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Fix shape mismatch for FakeQuant scale/zero_point
        model_sd = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in model_sd and v.shape != model_sd[k].shape:
                if v.numel() == model_sd[k].numel():
                    state_dict[k] = v.reshape(model_sd[k].shape)
                else:
                    # Resize parameter in model to match checkpoint
                    parts = k.split('.')
                    obj = model
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif part.isdigit() and hasattr(obj, '__getitem__'):
                            obj = obj[int(part)]
                        else:
                            obj = getattr(obj, part)
                    param_name = parts[-1]
                    old = getattr(obj, param_name)
                    if isinstance(old, nn.Parameter):
                        setattr(obj, param_name, nn.Parameter(v.clone(), requires_grad=old.requires_grad))
                    else:
                        setattr(obj, param_name, v.clone())

        model.load_state_dict(state_dict, strict=False)
        enable_quantization(model)
    else:
        from mmdet3d.models import build_model
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval().cuda()

    # Extract neck
    neck = model.encoders["camera"]["neck"]
    wrapper = NeckWrapper(neck).eval().cuda()

    # Dummy inputs: SwinT multi-scale outputs for B*N=6 images
    # SwinT out_indices=[1,2,3] → channels [192, 384, 768]
    # Feature sizes at each scale for input [1,3,256,704]:
    #   scale1: [6, 192, 32, 88]
    #   scale2: [6, 384, 16, 44]
    #   scale3: [6, 768, 8, 22]
    feat1 = torch.randn(6, 192, 32, 88).cuda()
    feat2 = torch.randn(6, 384, 16, 44).cuda()
    feat3 = torch.randn(6, 768, 8, 22).cuda()

    # Verify forward works
    with torch.no_grad():
        out = wrapper(feat1, feat2, feat3)
    logger.info(f"Forward OK: output shape = {out.shape}")

    if args.verify:
        torch.save(out.cpu(), args.output.replace(".onnx", "_pytorch_output.pt"))
        logger.info("Saved PyTorch output for verification")

    # Count FakeQuant nodes in neck
    fq_count = sum(1 for m in neck.modules() if hasattr(m, 'fake_quant_enabled'))
    logger.info(f"FakeQuant nodes in neck: {fq_count}")

    # Export ONNX
    logger.info(f"Exporting to {args.output}...")
    torch.onnx.export(
        wrapper,
        (feat1, feat2, feat3),
        args.output,
        opset_version=13,
        input_names=["feat_scale1", "feat_scale2", "feat_scale3"],
        output_names=["neck_output"],
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
        print("WARNING: FakeQuant nodes remain — check symbolic registration")
    if args.int8 and len(qdq) == 0:
        print("WARNING: No Q/DQ nodes in INT8 mode — check symbolic registration")


if __name__ == "__main__":
    main()
