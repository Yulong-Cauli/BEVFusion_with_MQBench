"""
Phase 4: 导出 LiDAR SparseEncoder 为 libspconv.so 可推理的自定义 ONNX。

技术路线：
  基于 NVIDIA CUDA-BEVFusion 的 monkey-patch trace hook 方式，
  绕过 torch.fx / torch.onnx.export 对 spconv 的限制。
  导出的 ONNX 包含自定义算子节点（SparseConvolution / ScatterDense / Reshape 等），
  由 libspconv.so 的 ONNX parser 解析并构建推理引擎。

支持模式：
  - FP16（默认）：所有层 FP16 精度，精度损失极小
  - INT8（可选）：从 MQBench PTQ checkpoint 提取线性量化参数，
    转换为 NVIDIA dynamic_range 格式。注意：Log2 量化层会退回 FP16。

用法：
    # FP16 导出
    python tools/export_utils/export_lidar.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --output lidar_backbone.onnx

    # INT8 导出（从 PTQ checkpoint 提取量化参数）
    python tools/export_utils/export_lidar.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/ptq_minmax_model.pth \
        --output lidar_backbone_int8.onnx \
        --int8

    # 保存验证用的 tensor（PyTorch 推理结果）
    python tools/export_utils/export_lidar.py \
        --config ... --ckpt ... --output ... --save-tensors
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnx.helper as helper

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval
from mmdet3d.ops import spconv as spconv

# ============================================================================
# Trace infrastructure (adapted from NVIDIA exptool.py)
# ============================================================================

avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False

# Per-layer quantization info (populated during trace if --int8)
layer_quant_info = {}


def register_node(fn):
    """Monkey-patch a method to record ONNX nodes during forward pass."""
    fnnames = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)

    def make_hook(bind_fn):
        ilayer = 0

        def internal_forward(self, *args):
            global enable_trace
            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            enable_trace_saved = enable_trace
            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = enable_trace_saved

            avoid_reuse_container.extend(list(args) + [y])
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)
    return make_hook


@register_node("spconv.conv.SparseConvolution.forward")
def symbolic_sparse_convolution(self, ilayer, y, x):
    register_tensor(y)
    subm_str = 'subm' if self.subm else 'conv'
    # Check if next operation is ReLU (for fusion)
    act_type = getattr(self, "_fused_activation", "None")
    print(f"   --> SparseConvolution{ilayer}[{subm_str}] "
          f"in={self.in_channels} out={self.out_channels} "
          f"k={self.kernel_size} s={self.stride} act={act_type} "
          f"key={self.indice_key} "
          f"-> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")

    inputs = [
        get_tensor_id(x),
        append_initializer(
            self.weight.data.permute(4, 0, 1, 2, 3),
            f"spconv{ilayer}.weight"
        ),
    ]
    # Parser always reads input(2) as bias — must provide one
    if self.bias is not None:
        inputs.append(append_initializer(self.bias.data, f"spconv{ilayer}.bias"))
    else:
        # Create zero bias
        zero_bias = torch.zeros(self.out_channels, device=self.weight.device, dtype=self.weight.dtype)
        inputs.append(append_initializer(zero_bias, f"spconv{ilayer}.bias"))

    output_bound = getattr(self, "output_bound", 200000)

    # Quantization attributes
    quant_attrs = {}
    conv_name = f"conv{ilayer}"
    if conv_name in layer_quant_info:
        qi = layer_quant_info[conv_name]
        quant_attrs["precision"] = qi.get("precision", "fp16")
        quant_attrs["output_precision"] = qi.get("output_precision", "fp16")
        if "input_dynamic_range" in qi:
            quant_attrs["input_dynamic_range"] = qi["input_dynamic_range"]
            quant_attrs["weight_dynamic_ranges"] = qi["weight_dynamic_ranges"]
    else:
        quant_attrs["precision"] = getattr(self, "precision", "fp16")
        quant_attrs["output_precision"] = getattr(self, "output_precision", "fp16")

    nodes.append(
        helper.make_node(
            "SparseConvolution", inputs, [get_tensor_id(y)], conv_name,
            ndim=self.ndim,
            input_spatial_shape=x.spatial_shape,
            output_spatial_shape=y.spatial_shape,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_bound=output_bound,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            transposed=self.transposed,
            inverse=self.inverse,
            output_padding=self.output_padding,
            groups=self.groups,
            subm=self.subm,
            rulebook=self.indice_key if self.indice_key else "",
            activation=act_type,
            input_shape=x.features.shape,
            output_shape=y.features.shape,
            **quant_attrs,
        )
    )


@register_node("torch.nn.ReLU.forward")
def symbolic_relu(self, ilayer, y, x):
    # Check if this ReLU was already fused into the preceding SparseConvolution
    if getattr(self, "_fused_into_conv", False):
        # Skip — just alias the output tensor ID to the input
        idd_x = __obj_to_id(x)
        idd_y = __obj_to_id(y)
        if idd_x in obj_to_tensor_id:
            obj_to_tensor_id[idd_y] = obj_to_tensor_id[idd_x]
            print(f"   --> ReLU{ilayer} [FUSED, skip]")
        else:
            # x is not tracked (e.g., came from BN on features tensor)
            # Just register y with a new ID
            register_tensor(y)
            print(f"   --> ReLU{ilayer} [FUSED, skip, new id for output]")
        return

    register_tensor(y)
    print(f"   --> ReLU{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Relu", [get_tensor_id(x)], [get_tensor_id(y)], f"relu{ilayer}"
        )
    )


@register_node("spconv.structure.SparseConvTensor.dense")
def node_sparse_conv_tensor_dense(self, ilayer, y):
    register_tensor(y)
    print(f"   --> ToDense{ilayer}[{self.spatial_shape}][{list(y.size())}] "
          f"-> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "ScatterDense", [get_tensor_id(self)], [get_tensor_id(y)],
            f"scatter{ilayer}",
            input_spatial_shape=self.spatial_shape,
            format="xyz",
            output_shape=list(y.size()),
        )
    )


@register_node("torch.Tensor.permute")
def node_permute(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Permute{ilayer}[{dims}][{list(y.shape)}] "
          f"-> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Transpose", [get_tensor_id(self)], [get_tensor_id(y)],
            f"transpose{ilayer}",
            dims=dims,
        )
    )


@register_node("torch.Tensor.reshape")
def node_reshape(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Reshape{ilayer}[{dims}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    nodes.append(
        helper.make_node(
            "Reshape", [get_tensor_id(self)], [get_tensor_id(y)],
            f"reshape{ilayer}",
            dims=dims,
        )
    )


# ── Tensor ID management ────────────────────────────────────────────────────

def __obj_to_id(obj):
    idd = id(obj)
    if isinstance(obj, spconv.SparseConvTensor):
        idd = id(obj.features)
    return idd


def register_tensor(obj):
    global obj_to_tensor_id
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))


def get_tensor_id(obj):
    idd = __obj_to_id(obj)
    assert idd in obj_to_tensor_id, (
        f"Cannot find tensor id for {type(obj)}. "
        "Some operator is not being traced."
    )
    return obj_to_tensor_id[idd]


def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT16,
            dims=list(value.shape),
            vals=value.cpu().data.numpy().astype(np.float16).tobytes(),
            raw=True,
        )
    )
    return name


# ============================================================================
# SparseEncoder forward hook (replaces model.forward during trace)
# ============================================================================

def make_model_forward_hook(model):
    """Create a clean forward function for SparseEncoder tracing.

    For basicblock architecture, we manually expand SparseBasicBlock.forward
    to ensure Add nodes are properly traced (the trace system only hooks
    SparseConvolution.forward, not Tensor.__iadd__).
    """
    from mmdet3d.ops.sparse_block import SparseBasicBlock
    from mmdet3d.ops.spconv.conv import SparseConvolution

    add_counter = [0]
    relu_counter = [0]

    def _run_basic_block(block, x):
        """Manually run SparseBasicBlock with explicit Add/Relu trace nodes."""
        global nodes, enable_trace

        identity_features = x.features  # save for residual

        # conv1 (with fused ReLU from prepare_model_for_export)
        out = block.conv1(x)
        # BN is already fused into conv1 weights, skip norm1
        # ReLU is fused into conv1 activation attribute

        # conv2 (no fused ReLU)
        out = block.conv2(out)
        # BN is already fused into conv2 weights, skip norm2

        # Residual Add: out.features += identity
        # We need to emit an Add node manually
        enable_trace_saved = enable_trace
        enable_trace = False

        # Save pre-add features tensor id
        out_id = get_tensor_id(out)
        identity_id = get_tensor_id(x)  # x is the SparseConvTensor with identity features

        # Perform the actual add
        out.features = out.features + identity_features

        # Register the result as a new tensor
        register_tensor(out)
        add_out_id = get_tensor_id(out)

        add_name = f"add{add_counter[0]}"
        add_counter[0] += 1
        print(f"   --> Add[{add_name}] -> Input {out_id}, {identity_id}, Output {add_out_id}")
        nodes.append(
            helper.make_node(
                "Add", [out_id, identity_id], [add_out_id], add_name
            )
        )

        # Relu after Add
        pre_relu = out.features.clone()
        out.features = torch.relu(out.features)
        register_tensor(out)
        relu_out_id = get_tensor_id(out)

        relu_name = f"relu{relu_counter[0]}"
        relu_counter[0] += 1
        print(f"   --> Relu[{relu_name}] -> Input {add_out_id}, Output {relu_out_id}")
        nodes.append(
            helper.make_node(
                "Relu", [add_out_id], [relu_out_id], relu_name
            )
        )

        enable_trace = enable_trace_saved
        return out

    def impl(voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, model.sparse_shape, batch_size
        )
        x = model.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in model.encoder_layers:
            # Iterate through children of each encoder stage
            for child in encoder_layer.children():
                if isinstance(child, SparseBasicBlock):
                    x = _run_basic_block(child, x)
                else:
                    # SparseConvModule (downsampling conv) — normal trace
                    x = child(x)
            encode_features.append(x)

        out = model.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3)
        spatial_features = spatial_features.reshape(N, C * D, H, W)
        return spatial_features
    return impl


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b, bn_eps):
    """Fuse Conv + BatchNorm weights into a single Conv (weight, bias).

    spconv weight shape: [k, k, k, in_ch, out_ch]  (out_ch is last dim)
    BN params shape: [out_ch]

    Standard formula:
        W_fused = W_conv * (gamma / sqrt(var + eps))
        B_fused = gamma * (B_conv - mean) / sqrt(var + eps) + beta
    """
    factor = bn_w / torch.sqrt(bn_rv + bn_eps)
    # spconv weight: [k, k, k, in, out] — factor applies to last dim (out_ch)
    fused_w = conv_w * factor.reshape(1, 1, 1, 1, -1)
    if conv_b is not None:
        fused_b = (conv_b - bn_rm) * factor + bn_b
    else:
        fused_b = -bn_rm * factor + bn_b
    return fused_w, fused_b


def prepare_model_for_export(backbone):
    """Pre-process backbone for export: fuse Conv+BN, fix indice_keys.

    For basicblock architecture:
    1. Fuse Conv+BN weights (merge BN params into Conv weight/bias)
    2. Mark activation fusion (ReLU into SparseConvolution node)
    3. Fix indice_keys (same stage subm layers share rulebook)
    """
    from mmdet3d.ops.spconv.conv import SparseConvolution
    from mmdet3d.ops.sparse_block import SparseBasicBlock

    # Fix indice_keys for encoder layers (same as NVIDIA's exptool.py)
    for i, layers in enumerate(backbone.encoder_layers):
        for name, module in layers.named_modules():
            if isinstance(module, SparseConvolution):
                if module.subm:
                    module.indice_key = f"subm{i+1}"

    # Fuse Conv+BN for SparseBasicBlock
    for name, module in backbone.named_modules():
        if isinstance(module, SparseBasicBlock):
            # conv1 + norm1 fusion
            w1, b1 = fuse_conv_bn_weights(
                module.conv1.weight.data, module.conv1.bias.data if module.conv1.bias is not None else None,
                module.norm1.running_mean, module.norm1.running_var,
                module.norm1.weight, module.norm1.bias, module.norm1.eps)
            module.conv1.weight.data = w1
            # spconv SparseConvolution stores bias differently — set it as attribute
            module.conv1.bias = torch.nn.Parameter(b1)
            # Mark conv1 as having fused ReLU (conv1 is followed by relu in basicblock)
            module.conv1._fused_activation = "ReLU"

            # conv2 + norm2 fusion
            w2, b2 = fuse_conv_bn_weights(
                module.conv2.weight.data, module.conv2.bias.data if module.conv2.bias is not None else None,
                module.norm2.running_mean, module.norm2.running_var,
                module.norm2.weight, module.norm2.bias, module.norm2.eps)
            module.conv2.weight.data = w2
            module.conv2.bias = torch.nn.Parameter(b2)
            # conv2 has NO fused ReLU (relu comes after Add in basicblock)
            module.conv2._fused_activation = "None"

    # Fuse Conv+BN+ReLU for SparseConvModule (conv_input, spconv downsampling, conv_out)
    def _fuse_convmodule(module):
        children = list(module.children())
        for i, child in enumerate(children):
            if isinstance(child, SparseConvolution):
                # Look for BN after conv
                bn = None
                for j in range(i+1, min(i+3, len(children))):
                    if isinstance(children[j], (nn.BatchNorm1d, nn.BatchNorm2d)):
                        bn = children[j]
                        break
                if bn is not None:
                    w, b = fuse_conv_bn_weights(
                        child.weight.data, child.bias.data if child.bias is not None else None,
                        bn.running_mean, bn.running_var,
                        bn.weight, bn.bias, bn.eps)
                    child.weight.data = w
                    child.bias = torch.nn.Parameter(b)

                # Check for ReLU fusion
                has_relu = False
                for j in range(i+1, min(i+3, len(children))):
                    if isinstance(children[j], nn.ReLU):
                        has_relu = True
                        child._fused_activation = "ReLU"
                        children[j]._fused_into_conv = True
                        break
                if not has_relu:
                    child._fused_activation = "None"
            elif hasattr(child, 'children') and not isinstance(child, SparseBasicBlock):
                _fuse_convmodule(child)

    _fuse_convmodule(backbone.conv_input)
    for layer in backbone.encoder_layers:
        # Only fuse the SparseConvModule children, not SparseBasicBlock (already done above)
        for child in layer.children():
            if not isinstance(child, SparseBasicBlock):
                _fuse_convmodule(child)
    _fuse_convmodule(backbone.conv_out)


# ============================================================================
# Quantization parameter extraction
# ============================================================================

def extract_quant_params_from_state_dict(state_dict, logger):
    """Extract quantization params directly from checkpoint state_dict.

    This avoids the need to build a PTQ model with matching observer types.
    Detects Log2 vs linear quantization by checking for 'log2_base' keys.

    Returns dict mapping conv index -> quant info for NVIDIA format.
    """
    # Find all lidar backbone quantized conv layers
    # Pattern: encoders.lidar.backbone.{path}.{weight_fake_quant|act_fake_quant}.{param}
    prefix = "encoders.lidar.backbone."

    # Collect unique conv layer paths (ordered by appearance)
    conv_paths = []
    seen = set()
    for key in sorted(state_dict.keys()):
        if not key.startswith(prefix):
            continue
        if ".weight_fake_quant." not in key:
            continue
        # Extract the conv path (everything before .weight_fake_quant)
        conv_path = key.split(".weight_fake_quant.")[0][len(prefix):]
        if conv_path not in seen:
            seen.add(conv_path)
            conv_paths.append(conv_path)

    logger.info(f"  Found {len(conv_paths)} quantized sparse conv layers")

    quant_info = {}
    for idx, conv_path in enumerate(conv_paths):
        conv_name = f"conv{idx}"
        full_prefix = f"{prefix}{conv_path}"

        # Check if activation uses Log2 quantization
        log2_key = f"{full_prefix}.act_fake_quant.log2_base"
        is_log2 = log2_key in state_dict

        if is_log2:
            log2_base = state_dict[log2_key].item()
            logger.info(f"  {conv_path}: Log2 (base={log2_base:.4f}) -> FP16 "
                        f"(libspconv 不支持 Log2)")
            quant_info[conv_name] = {
                "precision": "fp16",
                "output_precision": "fp16",
            }
            continue

        # Linear quantization: extract scale -> dynamic_range
        w_scale_key = f"{full_prefix}.weight_fake_quant.scale"
        a_scale_key = f"{full_prefix}.act_fake_quant.scale"

        if w_scale_key not in state_dict:
            logger.warning(f"  {conv_path}: no weight scale found -> FP16")
            quant_info[conv_name] = {"precision": "fp16", "output_precision": "fp16"}
            continue

        w_scale = state_dict[w_scale_key].detach().cpu().float()
        w_dynamic_ranges = (w_scale * 127.0).tolist()

        if a_scale_key in state_dict:
            a_scale = state_dict[a_scale_key].detach().cpu().float()
            a_dynamic_range = (a_scale.max().item() * 127.0)
        else:
            logger.info(f"  {conv_path}: no act scale -> FP16")
            quant_info[conv_name] = {"precision": "fp16", "output_precision": "fp16"}
            continue

        logger.info(f"  {conv_path}: INT8 (act_dr={a_dynamic_range:.4f}, "
                     f"w_dr=[{min(w_dynamic_ranges):.4f}, {max(w_dynamic_ranges):.4f}])")
        quant_info[conv_name] = {
            "input_dynamic_range": a_dynamic_range,
            "weight_dynamic_ranges": w_dynamic_ranges,
            "precision": "int8",
            "output_precision": "int8",
        }

    return quant_info


# ============================================================================
# Export ONNX
# ============================================================================

def export_onnx(model, voxels, coors, batch_size, save_path):
    """Trace SparseEncoder and export custom ONNX for libspconv.so."""
    global avoid_reuse_container, obj_to_tensor_id, nodes, initializers, enable_trace

    avoid_reuse_container = []
    obj_to_tensor_id = {}
    nodes = []
    initializers = []

    model.forward = make_model_forward_hook(model)

    print("Tracing model inference...")
    with torch.no_grad():
        register_tensor(voxels)
        enable_trace = True
        y = model(voxels, coors, batch_size)
        enable_trace = False

    print(f"Tracing done! {len(nodes)} nodes recorded.")

    inputs = [
        helper.make_value_info(
            name="0",
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=voxels.size(),
            ),
        )
    ]

    outputs = [
        helper.make_value_info(
            name=get_tensor_id(y),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=y.size(),
            ),
        )
    ]

    graph = helper.make_graph(
        name="lidar_backbone",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializers,
    )

    opset = [helper.make_operatorsetid("ai.onnx", 11)]
    onnx_model = helper.make_model(
        graph, opset_imports=opset,
        producer_name="bevfusion_mqbench",
        producer_version="1.0",
    )
    onnx.save_model(onnx_model, save_path)
    print(f"ONNX saved: {save_path}")

    # Cleanup
    avoid_reuse_container = []
    obj_to_tensor_id = {}
    nodes = []
    initializers = []

    return y


# ============================================================================
# Save tensors for verification
# ============================================================================

def save_verification_tensors(voxels, coors, output, save_dir):
    """Save input/output tensors for C++ verification."""
    os.makedirs(save_dir, exist_ok=True)

    voxels_path = os.path.join(save_dir, "lidar_voxels.pt")
    coors_path = os.path.join(save_dir, "lidar_coors.pt")
    output_path = os.path.join(save_dir, "lidar_output.pt")

    torch.save(voxels.cpu(), voxels_path)
    torch.save(coors.cpu(), coors_path)
    torch.save(output.cpu(), output_path)

    print(f"Saved verification tensors to {save_dir}/")
    print(f"  voxels: {voxels.shape} ({voxels.dtype})")
    print(f"  coors:  {coors.shape} ({coors.dtype})")
    print(f"  output: {output.shape} ({output.dtype})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export LiDAR SparseEncoder to custom ONNX for libspconv.so")
    parser.add_argument("--config", required=True, help="Model config yaml")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (FP32 or PTQ)")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--int8", action="store_true", help="Export with INT8 quantization params")
    parser.add_argument("--save-tensors", action="store_true", help="Save input/output tensors for verification")
    parser.add_argument("--tensor-dir", default="lidar_verify_tensors", help="Directory for verification tensors")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("Phase 4: LiDAR SparseEncoder ONNX Export")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Checkpoint: {args.ckpt}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  INT8: {args.int8}")

    # ── Load model ──────────────────────────────────────────────────────────
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.int8:
        # INT8 mode: load FP32 model, extract quant params from state_dict
        from mmdet3d.models import build_model

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Extract quant params before loading (need original keys)
        logger.info("Extracting quantization parameters from checkpoint...")
        global layer_quant_info
        layer_quant_info = extract_quant_params_from_state_dict(state_dict, logger)
        logger.info(f"  Total layers with quant info: {len(layer_quant_info)}")

        # Count INT8 vs FP16 layers
        n_int8 = sum(1 for v in layer_quant_info.values() if v.get("precision") == "int8")
        n_fp16 = sum(1 for v in layer_quant_info.values() if v.get("precision") == "fp16")
        logger.info(f"  INT8 layers: {n_int8}, FP16 layers (Log2 fallback): {n_fp16}")

        # Load FP32 weights (skip quant-related keys)
        fp32_state = {}
        for k, v in state_dict.items():
            if "fake_quant" in k or "activation_post_process" in k:
                continue
            fp32_state[k] = v
        model.load_state_dict(fp32_state, strict=False)
        model.eval()

        backbone = model.encoders.lidar.backbone
    else:
        # Load FP32 model directly
        from mmdet3d.models import build_model

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        backbone = model.encoders.lidar.backbone

    backbone.eval().cuda().half()

    # ── Prepare model: fix indice_keys ───────────────────────────────────────
    prepare_model_for_export(backbone)

    # ── Load real input data for tracing (realistic shapes needed) ──────────
    verify_dir = args.tensor_dir if args.tensor_dir else "lidar_verify_tensors"
    voxels_path = os.path.join(verify_dir, "voxel_features.pt")
    coors_path = os.path.join(verify_dir, "coors.pt")

    if os.path.exists(voxels_path) and os.path.exists(coors_path):
        voxels = torch.load(voxels_path).cuda()  # [N, 5] fp16
        coors = torch.load(coors_path).cuda()    # [N, 4] int32
        batch_size = 1
        logger.info(f"Using real input data: voxels={voxels.shape}, coors={coors.shape}")
    else:
        # Fallback to dummy input
        logger.warning("No real input data found, using dummy input (shapes may be wrong)")
        voxels = torch.zeros(1, backbone.in_channels).cuda().half()
        coors = torch.zeros(1, 4).int().cuda()
        batch_size = 1

    # ── Export ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    output_tensor = export_onnx(backbone, voxels, coors, batch_size, args.output)

    # ── Save verification tensors ───────────────────────────────────────────
    if args.save_tensors:
        save_verification_tensors(voxels, coors, output_tensor, args.tensor_dir)

    # ── Summary ─────────────────────────────────────────────────────────────
    onnx_model = onnx.load(args.output)
    n_nodes = len(onnx_model.graph.node)
    n_init = len(onnx_model.graph.initializer)
    file_size = os.path.getsize(args.output) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  ONNX: {args.output} ({file_size:.1f} MB)")
    print(f"  Nodes: {n_nodes}, Initializers: {n_init}")
    print(f"  Output shape: {list(output_tensor.shape)}")
    print(f"  Precision: {'INT8 (with FP16 fallback for Log2 layers)' if args.int8 else 'FP16'}")
    print(f"{'='*60}")

    # Print node types
    from collections import Counter
    op_counts = Counter(n.op_type for n in onnx_model.graph.node)
    print("Node types:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")


if __name__ == "__main__":
    main()
