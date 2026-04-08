"""
Phase 6 Step 2: 导出 TransFusionHead 为 ONNX → TRT 引擎。

TransFusionHead 量化路径二（手动 FakeQuant），PTQ checkpoint 中已有 Conv2d/Linear 的 Q/DQ 参数。

需要修复的 ONNX 导出问题：
1. argsort(descending=True)[..., :num_proposals] → topk (ONNX 不支持 argsort)
2. heatmap[:, 8] (3D) → heatmap[:, 8:9] (4D) for MaxPool2d
3. test_cfg["dataset"] 分支 → 静态化为 nuScenes

输入: [1, 512, 180, 180] (fuser+decoder 输出)
输出: center, height, dim, rot, vel, heatmap, query_heatmap_score, dense_heatmap

用法:
    # FP16
    python tools/export_utils/export_head.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --output transfusion_head_fp16.onnx

    # INT8 (从 PTQ checkpoint)
    python tools/export_utils/export_head.py \
        --config ... --ckpt pretrained/ptq_minmax_model.pth \
        --output transfusion_head_int8.onnx --int8
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

# ── 导出逻辑 ────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import numpy as np
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval


class ExportableTransFusionHead(nn.Module):
    """Wrapper that makes TransFusionHead ONNX-exportable.

    Fixes:
    1. argsort → topk (ONNX opset 13 supports topk but not argsort)
    2. heatmap[:, 8] → heatmap[:, 8:9] (keep 4D for MaxPool2d)
    3. Static dataset branch (nuScenes)
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, inputs):
        head = self.head
        batch_size = inputs.shape[0]
        lidar_feat = head.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        bev_pos = head.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # Heatmap
        dense_heatmap = head.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = head.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=head.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

        # FIX 2: heatmap[:, 8] → heatmap[:, 8:9] to keep 4D for MaxPool2d
        local_max[:, 8:9, :, :] = F.max_pool2d(
            heatmap[:, 8:9, :, :], kernel_size=1, stride=1, padding=0
        )
        local_max[:, 9:10, :, :] = F.max_pool2d(
            heatmap[:, 9:10, :, :], kernel_size=1, stride=1, padding=0
        )

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # FIX 1: argsort → topk
        # Original: top_proposals = heatmap.view(B, -1).argsort(dim=-1, descending=True)[..., :num_proposals]
        heatmap_flat = heatmap.view(batch_size, -1)
        _, top_proposals = torch.topk(heatmap_flat, head.num_proposals, dim=-1)

        top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='trunc')
        top_proposals_index = top_proposals % heatmap.shape[-1]

        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )

        one_hot = F.one_hot(top_proposals_class, num_classes=head.num_classes).permute(0, 2, 1)
        query_cat_encoding = head.class_encoding(one_hot.float())
        query_feat = query_feat + query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        # Transformer decoder
        ret_dicts = []
        for i in range(head.num_decoder_layers):
            query_feat = head.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res_layer = head.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, head.num_classes, -1),
            dim=-1,
        )
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        # Concatenate all decoder layers (auxiliary=True)
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]

        return (
            new_res["center"],
            new_res["height"],
            new_res["dim"],
            new_res["rot"],
            new_res["vel"],
            new_res["heatmap"],
            new_res["query_heatmap_score"],
            new_res["dense_heatmap"],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", default="transfusion_head_fp16.onnx")
    p.add_argument("--int8", action="store_true", help="Export with INT8 Q/DQ nodes from PTQ checkpoint")
    p.add_argument("--verify", action="store_true", help="Save PyTorch output for verification")
    args = p.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("Phase 6 Step 2: TransFusionHead ONNX Export")

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

        model_sd = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in model_sd and v.shape != model_sd[k].shape:
                if v.numel() == model_sd[k].numel():
                    state_dict[k] = v.reshape(model_sd[k].shape)
                else:
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

    # Extract head
    head = model.heads["object"]
    wrapper = ExportableTransFusionHead(head).eval().cuda()

    # Dummy input: fuser+decoder output
    neck_features = torch.randn(1, 512, 180, 180).cuda()

    # Verify forward
    with torch.no_grad():
        outputs = wrapper(neck_features)
    logger.info(f"Forward OK: {len(outputs)} outputs")
    for i, o in enumerate(outputs):
        logger.info(f"  output[{i}]: shape={o.shape}")

    output_names = ["center", "height", "dim", "rot", "vel",
                    "heatmap", "query_heatmap_score", "dense_heatmap"]

    if args.verify:
        save_dict = {name: o.cpu() for name, o in zip(output_names, outputs)}
        torch.save(save_dict, args.output.replace(".onnx", "_pytorch_output.pt"))
        logger.info("Saved PyTorch output for verification")

    fq_count = sum(1 for m in head.modules() if hasattr(m, 'fake_quant_enabled'))
    logger.info(f"FakeQuant nodes in head: {fq_count}")

    # Export ONNX
    logger.info(f"Exporting to {args.output}...")
    torch.onnx.export(
        wrapper,
        (neck_features,),
        args.output,
        opset_version=13,
        input_names=["neck_features"],
        output_names=output_names,
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
    print(f"  Outputs:")
    for o in m.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}: {dims}")
    print(f"{'='*60}")

    if len(fakeq) > 0:
        print("WARNING: FakeQuant nodes remain — check symbolic registration")
    if args.int8 and len(qdq) == 0:
        print("WARNING: No Q/DQ nodes in INT8 mode — check symbolic registration")


if __name__ == "__main__":
    main()
