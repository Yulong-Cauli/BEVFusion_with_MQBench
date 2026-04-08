"""
Phase 7: Standalone TRT inference pipeline for BEVFusion.

Runs in spconv23_deploy environment (Python 3.9, PyTorch 2.0, spconv 2.3.8, TRT 8.6.1).
No mmcv/mmdet3d dependency for core inference — only for dataset loading and NDS evaluation.

Usage:
    conda run --prefix /media/yellowstone/data2/CYL/spconv23_deploy \
        python tools/trt_infer_standalone.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --swin-engine swin_int8_sm86.engine \
        --depthnet-engine vtransform_depthnet_int8_sm86.engine \
        --fuser-engine fuser_decoder_int8_sm86.engine \
        --neck-engine camera_neck_int8_sm86.engine \
        --head-engine transfusion_head_int8_sm86.engine \
        --test-single
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Patch mmdet3d.ops to skip broken CUDA extensions (cpython-38 .so on py39)
# ============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build_sp39"))

# Enable standalone mode to skip broken CUDA extensions in mmdet3d.ops
os.environ["BEVFUSION_STANDALONE"] = "1"

# Pre-load our cpython-39 CUDA extensions before mmdet3d tries the cpython-38 ones
import bev_pool_ext as _bev_pool_ext
import voxel_layer as _voxel_layer
import iou3d_cuda as _iou3d_cuda
import roiaware_pool3d_ext as _roiaware_pool3d_ext

# Register our cpython-39 extensions under the mmdet3d paths
sys.modules["mmdet3d.ops.bev_pool.bev_pool_ext"] = _bev_pool_ext
sys.modules["mmdet3d.ops.voxel.voxel_layer"] = _voxel_layer
sys.modules["mmdet3d.ops.iou3d.iou3d_cuda"] = _iou3d_cuda
sys.modules["mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext"] = _roiaware_pool3d_ext
sys.modules["mmdet3d.ops.bev_pool.bev_pool_ext"] = _bev_pool_ext
sys.modules["mmdet3d.ops.voxel.voxel_layer"] = _voxel_layer
sys.modules["mmdet3d.ops.iou3d.iou3d_cuda"] = _iou3d_cuda

import tensorrt as trt
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.utils import recursive_eval

import spconv.pytorch as spconv


# ============================================================================
# TRT Engine Runner
# ============================================================================

class TRTRunner:
    """Runs a TRT engine using torch CUDA tensors."""

    def __init__(self, engine_path, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(self.engine.get_tensor_shape(name))
                dtype_trt = self.engine.get_tensor_dtype(name)
                if dtype_trt == trt.float16:
                    self.output_dtypes[name] = torch.float16
                else:
                    self.output_dtypes[name] = torch.float32

        self.logger.info(
            f"TRT engine loaded: {engine_path} "
            f"(inputs={self.input_names}, outputs={self.output_names})"
        )

    def __call__(self, *inputs):
        assert len(inputs) == len(self.input_names)
        for name, tensor in zip(self.input_names, inputs):
            t = tensor.contiguous()
            if t.dtype == torch.float64:
                t = t.float()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.output_dtypes[name]
            t = torch.zeros(shape, dtype=dtype, device="cuda").contiguous()
            self.context.set_tensor_address(name, t.data_ptr())
            outputs[name] = t

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()
        return [outputs[name] for name in self.output_names]


# ============================================================================
# LiDAR SparseEncoder (spconv 2.3)
# ============================================================================

class SparseBasicBlock23(spconv.SparseModule):
    """SparseBasicBlock rebuilt with spconv 2.3 API."""
    def __init__(self, channels, norm_cfg_eps=1e-3, norm_cfg_momentum=0.01):
        super().__init__()
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

    Matches BEVFusion config: in_channels=5, sparse_shape=[1440,1440,41],
    output_channels=128, block_type=basicblock.
    """
    def __init__(self):
        super().__init__()
        norm_eps = 1e-3
        norm_mom = 0.01

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(5, 16, 3, padding=1, bias=False, indice_key="subm1"),
            nn.BatchNorm1d(16, eps=norm_eps, momentum=norm_mom),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer1 = spconv.SparseSequential(
            SparseBasicBlock23(16, norm_eps, norm_mom),
            SparseBasicBlock23(16, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False, indice_key="spconv1"),
                nn.BatchNorm1d(32, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )
        self.encoder_layer2 = spconv.SparseSequential(
            SparseBasicBlock23(32, norm_eps, norm_mom),
            SparseBasicBlock23(32, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False, indice_key="spconv2"),
                nn.BatchNorm1d(64, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )
        self.encoder_layer3 = spconv.SparseSequential(
            SparseBasicBlock23(64, norm_eps, norm_mom),
            SparseBasicBlock23(64, norm_eps, norm_mom),
            spconv.SparseSequential(
                spconv.SparseConv3d(64, 128, 3, stride=2, padding=[1, 1, 0], bias=False, indice_key="spconv3"),
                nn.BatchNorm1d(128, eps=norm_eps, momentum=norm_mom),
                nn.ReLU(inplace=True),
            ),
        )
        self.encoder_layer4 = spconv.SparseSequential(
            SparseBasicBlock23(128, norm_eps, norm_mom),
            SparseBasicBlock23(128, norm_eps, norm_mom),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, (1, 1, 3), stride=(1, 1, 2), padding=0, bias=False, indice_key="spconv_down2"),
            nn.BatchNorm1d(128, eps=norm_eps, momentum=norm_mom),
            nn.ReLU(inplace=True),
        )
        self.sparse_shape = [1440, 1440, 41]

    def forward(self, voxel_features, coors, batch_size, **kwargs):
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
# Weight mapping: spconv 2.1 -> spconv 2.3
# ============================================================================

def permute_spconv_weight(w):
    """Convert spconv 2.1 weight [k,k,k,in,out] to spconv 2.3 weight [out,k,k,k,in]."""
    return w.permute(4, 0, 1, 2, 3).contiguous()


def build_weight_mapping():
    """Build mapping from BEVFusion checkpoint keys to SparseEncoder23 keys."""
    mapping = {}
    prefix = "encoders.lidar.backbone."

    # conv_input
    mapping[f"{prefix}conv_input.0.weight"] = ("conv_input.0.weight", True)
    for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        mapping[f"{prefix}conv_input.1.{param}"] = (f"conv_input.1.{param}", False)

    layer_configs = [
        (1, 2, 16, 32, True),
        (2, 2, 32, 64, True),
        (3, 2, 64, 128, True),
        (4, 2, 128, None, False),
    ]
    for layer_idx, num_blocks, in_ch, out_ch, has_down in layer_configs:
        src_prefix = f"{prefix}encoder_layers.encoder_layer{layer_idx}"
        dst_prefix = f"encoder_layer{layer_idx}"
        for block_idx in range(num_blocks):
            for conv_name in ["conv1", "conv2"]:
                mapping[f"{src_prefix}.{block_idx}.{conv_name}.weight"] = (
                    f"{dst_prefix}.{block_idx}.{conv_name}.weight", True)
            for bn_name in ["bn1", "bn2"]:
                for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
                    mapping[f"{src_prefix}.{block_idx}.{bn_name}.{param}"] = (
                        f"{dst_prefix}.{block_idx}.{bn_name}.{param}", False)
        if has_down:
            down_idx = num_blocks
            mapping[f"{src_prefix}.{down_idx}.0.weight"] = (f"{dst_prefix}.{down_idx}.0.weight", True)
            for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
                mapping[f"{src_prefix}.{down_idx}.1.{param}"] = (f"{dst_prefix}.{down_idx}.1.{param}", False)

    # conv_out
    mapping[f"{prefix}conv_out.0.weight"] = ("conv_out.0.weight", True)
    for param in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        mapping[f"{prefix}conv_out.1.{param}"] = (f"conv_out.1.{param}", False)

    return mapping


def load_lidar_weights(model, ckpt_path, logger):
    """Load weights from BEVFusion checkpoint into SparseEncoder23."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    mapping = build_weight_mapping()

    new_state = {}
    for src_key, (dst_key, needs_permute) in mapping.items():
        if src_key not in state_dict:
            logger.warning(f"  Missing: {src_key}")
            continue
        w = state_dict[src_key]
        if needs_permute:
            w = permute_spconv_weight(w)
        new_state[dst_key] = w

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing}")
    logger.info(f"  Loaded {len(new_state)} parameters into SparseEncoder23")


# ============================================================================
# Quantized LiDAR backbone (Log2 + per-channel INT8 weight)
# ============================================================================

class SparseLog2FakeQuantize(nn.Module):
    """Log2 activation quantization for sparse features (standalone, no MQBench)."""

    def __init__(self, n_bits=8, per_channel=False, ch_axis=1, eps=1e-6):
        super().__init__()
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.eps = eps
        self.qmin = -(2 ** (n_bits - 1) - 1)
        self.qmax = (2 ** (n_bits - 1) - 1)
        self.register_buffer('log2_base', torch.tensor(-8.0))
        self.register_buffer('_fake_quant_enabled', torch.tensor(1, dtype=torch.uint8))
        self.register_buffer('fake_quant_enabled', torch.tensor(1, dtype=torch.uint8))

    def forward(self, x):
        if not self._fake_quant_enabled.item():
            return x
        orig_dtype = x.dtype
        x_f = x.float()
        zero_mask = x_f.abs() < self.eps
        sign = x_f.sign()
        if self.per_channel and self.log2_base.ndim == 1:
            base = self.log2_base.to(x_f.device).unsqueeze(0)
        else:
            base = self.log2_base.to(x_f.device)
        log2_x = torch.log2(x_f.abs().clamp(min=1e-30)) - base
        q_int = torch.round(log2_x).clamp(self.qmin, self.qmax)
        x_dq = sign * torch.pow(2.0, q_int + base)
        x_dq = torch.where(zero_mask, torch.zeros_like(x_f), x_dq)
        out = x_f + (x_dq - x_f).detach()
        return out.to(orig_dtype)


class WeightFakeQuantize(nn.Module):
    """Per-channel symmetric INT8 weight fake quantization (standalone, no MQBench)."""

    def __init__(self, out_channels=1):
        super().__init__()
        self.register_buffer('scale', torch.ones(out_channels))
        self.register_buffer('zero_point', torch.zeros(out_channels, dtype=torch.long))
        self.register_buffer('fake_quant_enabled', torch.ones(1, dtype=torch.uint8))

    def forward(self, x):
        if not self.fake_quant_enabled.item():
            return x
        orig_dtype = x.dtype
        x = x.float()
        # Per-channel: scale shape [out_channels], weight shape [out, k, k, k, in] (spconv 2.3)
        # or [k, k, k, in, out] (spconv 2.1)
        s = self.scale.to(x.device).float()
        zp = self.zero_point.to(x.device).float()
        # Reshape scale for broadcasting
        if s.ndim == 1 and x.ndim == 5:
            # spconv 2.3: [out, k, k, k, in] -> scale on dim 0
            s = s.view(-1, 1, 1, 1, 1)
            zp = zp.view(-1, 1, 1, 1, 1)
        x_q = torch.clamp(torch.round(x / s) + zp, -127, 127)
        x_dq = (x_q - zp) * s
        return (x.detach() + (x_dq - x.detach()).detach()).to(orig_dtype)


class QuantizedSparseConv23(spconv.SparseModule):
    """Wrapper: spconv 2.3 SparseConv + FakeQuant (weight + activation)."""

    def __init__(self, conv, weight_fq, act_fq=None):
        super().__init__()
        self.conv = conv
        self.weight_fake_quant = weight_fq
        self.act_fake_quant = act_fq

    def forward(self, input):
        if self.act_fake_quant is not None:
            quant_feats = self.act_fake_quant(input.features)
            input = input.replace_feature(quant_feats)
        saved_weight = self.conv.weight.data.clone()
        try:
            self.conv.weight.data = self.weight_fake_quant(saved_weight)
            output = self.conv(input)
        finally:
            self.conv.weight.data = saved_weight
        return output


def quantize_sparse_encoder(model, logger):
    """Wrap all SparseConv3d/SubMConv3d in SparseEncoder23 with FakeQuant."""
    replacements = []
    for name, child in list(model.named_modules()):
        if isinstance(child, (spconv.SubMConv3d, spconv.SparseConv3d)):
            out_channels = child.out_channels
            wfq = WeightFakeQuantize(out_channels)
            afq = SparseLog2FakeQuantize()
            wrapped = QuantizedSparseConv23(child, wfq, afq)
            replacements.append((name, wrapped))

    for name, replacement in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        # SparseSequential doesn't support __setitem__, use _modules dict
        if parts[-1].isdigit():
            parent._modules[parts[-1]] = replacement
        else:
            setattr(parent, parts[-1], replacement)

    logger.info(f"  Quantized {len(replacements)} sparse conv layers")
    return model


def build_ptq_weight_mapping():
    """Build mapping from PTQ checkpoint keys to QuantizedSparseEncoder23 keys.

    PTQ checkpoint structure (spconv 2.1):
        encoders.lidar.backbone.conv_input.0.conv.weight          -> conv_input.0.conv.weight (permute)
        encoders.lidar.backbone.conv_input.0.weight_fake_quant.scale -> conv_input.0.weight_fake_quant.scale
        encoders.lidar.backbone.conv_input.0.act_fake_quant.log2_base -> conv_input.0.act_fake_quant.log2_base
        encoders.lidar.backbone.conv_input.1.weight               -> conv_input.1.weight (BN)
    """
    mapping = {}
    prefix = "encoders.lidar.backbone."

    def _add_conv_mapping(src_conv, dst_conv):
        # conv weight (needs permute)
        mapping[f"{src_conv}.conv.weight"] = (f"{dst_conv}.conv.weight", True)
        # weight_fake_quant params
        for p in ["scale", "zero_point", "fake_quant_enabled", "eps",
                   "observer_enabled",
                   "activation_post_process.min_val",
                   "activation_post_process.max_val",
                   "activation_post_process.eps"]:
            mapping[f"{src_conv}.weight_fake_quant.{p}"] = (f"{dst_conv}.weight_fake_quant.{p}", False)
        # act_fake_quant params
        for p in ["log2_base", "_fake_quant_enabled", "_observer_enabled",
                   "_initialized", "fake_quant_enabled"]:
            mapping[f"{src_conv}.act_fake_quant.{p}"] = (f"{dst_conv}.act_fake_quant.{p}", False)

    def _add_bn_mapping(src_bn, dst_bn):
        for p in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            mapping[f"{src_bn}.{p}"] = (f"{dst_bn}.{p}", False)

    # conv_input: [0]=QuantizedSparseConv, [1]=BN, [2]=ReLU
    _add_conv_mapping(f"{prefix}conv_input.0", "conv_input.0")
    _add_bn_mapping(f"{prefix}conv_input.1", "conv_input.1")

    layer_configs = [
        (1, 2, True),
        (2, 2, True),
        (3, 2, True),
        (4, 2, False),
    ]
    for layer_idx, num_blocks, has_down in layer_configs:
        src_prefix_l = f"{prefix}encoder_layers.encoder_layer{layer_idx}"
        dst_prefix_l = f"encoder_layer{layer_idx}"
        for block_idx in range(num_blocks):
            # BasicBlock: conv1, bn1, conv2, bn2
            _add_conv_mapping(f"{src_prefix_l}.{block_idx}.conv1", f"{dst_prefix_l}.{block_idx}.conv1")
            _add_bn_mapping(f"{src_prefix_l}.{block_idx}.bn1", f"{dst_prefix_l}.{block_idx}.bn1")
            _add_conv_mapping(f"{src_prefix_l}.{block_idx}.conv2", f"{dst_prefix_l}.{block_idx}.conv2")
            _add_bn_mapping(f"{src_prefix_l}.{block_idx}.bn2", f"{dst_prefix_l}.{block_idx}.bn2")
        if has_down:
            down_idx = num_blocks
            _add_conv_mapping(f"{src_prefix_l}.{down_idx}.0", f"{dst_prefix_l}.{down_idx}.0")
            _add_bn_mapping(f"{src_prefix_l}.{down_idx}.1", f"{dst_prefix_l}.{down_idx}.1")

    # conv_out
    _add_conv_mapping(f"{prefix}conv_out.0", "conv_out.0")
    _add_bn_mapping(f"{prefix}conv_out.1", "conv_out.1")

    return mapping


def load_ptq_lidar_weights(model, ptq_ckpt_path, logger):
    """Load PTQ checkpoint into quantized SparseEncoder23."""
    ckpt = torch.load(ptq_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    mapping = build_ptq_weight_mapping()

    new_state = {}
    skipped = 0
    for src_key, (dst_key, needs_permute) in mapping.items():
        if src_key not in state_dict:
            skipped += 1
            continue
        w = state_dict[src_key]
        if needs_permute:
            w = permute_spconv_weight(w)
        # Handle shape mismatch for weight_fake_quant params
        new_state[dst_key] = w

    # Load with strict=False to handle missing activation_post_process keys
    model_sd = model.state_dict()
    filtered_state = {}
    for k, v in new_state.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered_state[k] = v
            elif v.numel() == model_sd[k].numel():
                filtered_state[k] = v.reshape(model_sd[k].shape)
            else:
                logger.debug(f"  Shape mismatch: {k} ckpt={v.shape} model={model_sd[k].shape}")
        else:
            logger.debug(f"  Key not in model: {k}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logger.info(f"  Loaded {len(filtered_state)} PTQ params (skipped {skipped} missing from ckpt)")


# ============================================================================
# Voxelization (standalone, using compiled voxel_layer)
# ============================================================================

class Voxelization(nn.Module):
    """Hard voxelization using compiled voxel_layer CUDA extension."""

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels if isinstance(max_voxels, (list, tuple)) else (max_voxels, max_voxels)

    def forward(self, points):
        max_voxels = self.max_voxels[1]  # test mode
        voxels = points.new_zeros(size=(max_voxels, self.max_num_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
        voxel_num = _voxel_layer.hard_voxelize(
            points, voxels, coors, num_points_per_voxel,
            self.voxel_size, self.point_cloud_range,
            self.max_num_points, max_voxels, 3, True,
        )
        return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]


# ============================================================================
# BEV Pool v2 (standalone, using compiled bev_pool_ext)
# ============================================================================

def bev_pool_v2(x, geom_feats, interval_starts, interval_lengths, B, D, H, W):
    """BEV pooling with pre-computed indices."""
    geom_feats = geom_feats.int()
    interval_starts = interval_starts.int()
    interval_lengths = interval_lengths.int()
    out = _bev_pool_ext.bev_pool_forward(
        x, geom_feats, interval_lengths, interval_starts, B, D, H, W,
    )
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out


# ============================================================================
# VTransform geometry (standalone, no mmcv dependency)
# ============================================================================

class VTransformGeometry:
    """Computes vtransform geometry, depth map, and BEV pool indices."""

    def __init__(self, image_size, feature_size, xbound, ybound, zbound, dbound):
        self.image_size = image_size
        self.feature_size = feature_size

        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])
        self.dx = dx
        self.bx = bx
        self.nx = nx

        iH, iW = image_size
        fH, fW = feature_size
        ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        self.frustum = torch.stack((xs, ys, ds), -1)
        self.D = D
        self.C = None  # set from depthnet output

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins,
                     post_rots, post_trans, extra_rots=None, extra_trans=None):
        B, N, _ = camera2lidar_trans.shape
        frustum = self.frustum.to(camera2lidar_trans.device)
        points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]), 5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        if extra_rots is not None:
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1)).squeeze(-1)
            )
        if extra_trans is not None:
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        return points

    def precompute_bev_indices(self, geom, B):
        N_per_batch = geom.shape[1] * geom.shape[2] * geom.shape[3] * geom.shape[4]
        Nprime = B * N_per_batch
        dx = self.dx.to(geom.device)
        bx = self.bx.to(geom.device)
        nx = self.nx.to(geom.device)

        geom_feats = ((geom - (bx - dx / 2.0)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([
            torch.full([N_per_batch, 1], ix, device=geom.device, dtype=torch.long)
            for ix in range(B)
        ])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) &
            (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) &
            (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        )
        geom_feats = geom_feats[kept]
        D_val, H_val, W_val = int(nx[2]), int(nx[0]), int(nx[1])
        ranks = (
            geom_feats[:, 0] * (W_val * D_val * B)
            + geom_feats[:, 1] * (D_val * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sort_indices = ranks.argsort()
        geom_feats = geom_feats[sort_indices]
        ranks = ranks[sort_indices]

        kept_intervals = torch.ones(ranks.shape[0], device=ranks.device, dtype=torch.bool)
        kept_intervals[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept_intervals)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks.shape[0] - interval_starts[-1]

        return {
            "kept": kept, "sort_indices": sort_indices,
            "geom_feats": geom_feats.int(),
            "interval_starts": interval_starts,
            "interval_lengths": interval_lengths,
            "B": B, "D": D_val, "H": H_val, "W": W_val,
        }

    def compute_depth_map(self, points, img_aug_matrix, lidar_aug_matrix,
                          lidar2image, B, N):
        depth = torch.zeros(B, N, 1, *self.image_size, device=points[0].device)
        for b in range(B):
            cur_coords = points[b][:, :3]
            cur_img_aug = img_aug_matrix[b]
            cur_lidar_aug = lidar_aug_matrix[b]
            cur_l2i = lidar2image[b]

            cur_coords = cur_coords - cur_lidar_aug[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug[:3, :3]).matmul(cur_coords.transpose(1, 0))
            cur_coords = cur_l2i[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_l2i[:, :3, 3].reshape(-1, 3, 1)
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            cur_coords = cur_img_aug[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) &
                (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        return depth


# ============================================================================
# TransFusionHead post-processing (standalone, no mmdet3d)
# ============================================================================

class TransFusionBBoxCoder:
    """Standalone bbox decoder for TransFusionHead."""

    def __init__(self, pc_range, out_size_factor, voxel_size,
                 post_center_range=None, score_threshold=None, code_size=10):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]

            if filter and self.post_center_range is not None:
                post_center_range = torch.tensor(
                    self.post_center_range, device=heatmap.device)
                mask = (boxes3d[:, :3] >= post_center_range[:3]).all(1)
                mask &= (boxes3d[:, :3] <= post_center_range[3:]).all(1)
                if self.score_threshold is not None:
                    mask &= scores > self.score_threshold
                boxes3d = boxes3d[mask]
                scores = scores[mask]
                labels = labels[mask]

            predictions_dicts.append({
                'bboxes': boxes3d, 'scores': scores, 'labels': labels
            })
        return predictions_dicts


def circle_nms(dets, thresh):
    """Circular NMS (pure numpy)."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2
            if dist <= thresh ** 2:
                suppressed[j] = 1
    return keep


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Rotated NMS using compiled iou3d_cuda."""
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = _iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


class SimpleLiDARBox:
    """Minimal LiDARInstance3DBoxes replacement."""

    def __init__(self, tensor, box_dim=9):
        self.tensor = tensor
        self.box_dim = box_dim

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def gravity_center(self):
        """torch.Tensor: center of each box [x, y, z + h/2]."""
        bottom_center = self.tensor[:, :3]
        gc = torch.zeros_like(bottom_center)
        gc[:, :2] = bottom_center[:, :2]
        gc[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gc

    @property
    def dims(self):
        """torch.Tensor: [w, l, h] of each box."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: yaw angle of each box."""
        return self.tensor[:, 6]

    @property
    def bev(self):
        """BEV representation: [x, y, w, l, yaw] -> [x1, y1, x2, y2, yaw]."""
        t = self.tensor
        # center_x, center_y, w, l, yaw
        cx, cy = t[:, 0], t[:, 1]
        w, l = t[:, 3], t[:, 4]
        yaw = t[:, 6]
        # Convert to xyxyr format for NMS
        bev_boxes = torch.stack([
            cx - w / 2, cy - l / 2,
            cx + w / 2, cy + l / 2,
            yaw
        ], dim=1)
        return bev_boxes

    def to(self, device):
        self.tensor = self.tensor.to(device)
        return self


# ============================================================================
# Hybrid BEVFusion Model (TRT + spconv 2.3)
# ============================================================================

class StandaloneBEVFusion(nn.Module):
    """BEVFusion with TRT engines + spconv 2.3 LiDAR backbone."""

    def __init__(self, swin_trt, depthnet_trt, fuser_trt, neck_trt, head_trt,
                 lidar_backbone, voxelizer, vtransform_geom, bev_downsample,
                 bbox_coder, test_cfg, num_proposals, num_classes,
                 voxelize_reduce, logger, use_tv_lidar=False):
        super().__init__()
        self.swin_trt = swin_trt
        self.depthnet_trt = depthnet_trt
        self.fuser_trt = fuser_trt
        self.neck_trt = neck_trt
        self.head_trt = head_trt
        self.lidar_backbone = lidar_backbone
        self.voxelizer = voxelizer
        self.vtransform_geom = vtransform_geom
        self.bev_downsample = bev_downsample
        self.bbox_coder = bbox_coder
        self.test_cfg = test_cfg
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.voxelize_reduce = voxelize_reduce
        self.logger = logger
        self.use_tv_lidar = use_tv_lidar

    @torch.no_grad()
    def forward_single(self, img, points, camera2ego, lidar2ego, lidar2camera,
                       lidar2image, camera_intrinsics, camera2lidar,
                       img_aug_matrix, lidar_aug_matrix, metas, **kwargs):
        B, N, C, H, W = img.shape

        # Step 1: SwinT backbone (TRT)
        img_flat = img.view(B * N, C, H, W).float()
        swin_outputs = []
        for i in range(B * N):
            outs = self.swin_trt(img_flat[i:i+1])
            swin_outputs.append([o.float() for o in outs])
        num_scales = len(swin_outputs[0])
        multi_scale_feats = []
        for s in range(num_scales):
            feat = torch.cat([swin_outputs[i][s] for i in range(B * N)], dim=0)
            multi_scale_feats.append(feat)

        # Step 2: Camera neck (TRT)
        neck_out = self.neck_trt(
            multi_scale_feats[0].float(),
            multi_scale_feats[1].float(),
            multi_scale_feats[2].float())
        x_cam = neck_out[0].float()

        # Step 3: vtransform depthnet (TRT) + bev_pool_v2
        BN, C_neck, fH, fW = x_cam.shape
        x_cam_5d = x_cam.view(B, N, C_neck, fH, fW)

        depth_map = self.vtransform_geom.compute_depth_map(
            points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N)

        depthnet_out = self.depthnet_trt(x_cam_5d.float(), depth_map.float())
        cam_feats_flat = depthnet_out[0].float()

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.vtransform_geom.get_geometry(
            camera2lidar_rots, camera2lidar_trans,
            intrins, post_rots, post_trans,
            extra_rots=extra_rots, extra_trans=extra_trans)

        D = self.vtransform_geom.D
        C_bev = cam_feats_flat.shape[1] // (D * fH * fW) if cam_feats_flat.ndim == 2 else cam_feats_flat.shape[-1]
        # depthnet output: [B*N*D*fH*fW, C]
        Nprime_bev = B * N * D * fH * fW
        if cam_feats_flat.shape[0] == Nprime_bev:
            C_bev = cam_feats_flat.shape[1]
        cam_feats_6d = cam_feats_flat.view(B, N, D, fH, fW, C_bev)

        indices = self.vtransform_geom.precompute_bev_indices(geom, B)
        Nprime = B * N * D * fH * fW
        x_flat = cam_feats_6d.reshape(Nprime, C_bev)
        x_flat = x_flat[indices["kept"]]
        x_flat = x_flat[indices["sort_indices"]]

        out = bev_pool_v2(
            x_flat, indices["geom_feats"],
            indices["interval_starts"], indices["interval_lengths"],
            indices["B"], indices["D"], indices["H"], indices["W"])
        camera_bev = torch.cat(out.unbind(dim=2), 1)

        # Apply BEV downsample (Conv2d stride=2)
        if self.bev_downsample is not None:
            camera_bev = self.bev_downsample(camera_bev)

        # Step 4: LiDAR backbone (spconv 2.3)
        feats, coords, sizes = self._voxelize(points)

        if self.use_tv_lidar:
            # TV mode: convert torch → tv.Tensor, run TV backbone, convert back
            # Features must be PyTorch-backed (torch.empty + tv.from_blob)
            # for TRT execution context compatibility.
            from cumm import tensorview as tv
            feats_fp16 = feats.half().cuda()  # keep as torch.Tensor on GPU
            feats_tv = tv.from_blob(feats_fp16.data_ptr(),
                                    list(feats_fp16.shape), tv.float16, 0)
            coords_np = coords.cpu().numpy().astype(np.int32)
            coords_tv = tv.from_numpy(coords_np).cuda()
            batch_size = int(coords[-1, 0].item()) + 1
            lidar_bev_np = self.lidar_backbone.forward(
                feats_tv, coords_tv, batch_size, feature_ref=feats_fp16)
            lidar_bev = torch.from_numpy(lidar_bev_np).cuda()
        else:
            # PyTorch mode
            backbone_dtype = next(self.lidar_backbone.parameters()).dtype
            feats = feats.to(backbone_dtype)
            batch_size = coords[-1, 0] + 1
            lidar_bev = self.lidar_backbone(feats, coords, batch_size, sizes=sizes)

        # Step 5: Fuser + Decoder (TRT)
        fuser_out = self.fuser_trt(camera_bev.float(), lidar_bev.float())
        neck_features = fuser_out[0].float()

        # Step 6: TransFusionHead (TRT) + post-processing
        batch_size_int = img.shape[0]
        outputs = [{} for _ in range(batch_size_int)]

        head_outs = self.head_trt(neck_features.float())
        center = head_outs[0].float()
        height = head_outs[1].float()
        dim = head_outs[2].float()
        rot = head_outs[3].float()
        vel = head_outs[4].float()
        heatmap = head_outs[5].float()
        query_heatmap_score = head_outs[6].float()

        bboxes = self._decode_and_nms(
            center, height, dim, rot, vel, heatmap, query_heatmap_score, metas)
        for k, (boxes, scores, labels) in enumerate(bboxes):
            outputs[k].update({
                "boxes_3d": boxes.to("cpu"),
                "scores_3d": scores.cpu(),
                "labels_3d": labels.cpu(),
            })
        return outputs

    def _decode_and_nms(self, center, height, dim, rot, vel, heatmap,
                        query_heatmap_score, metas):
        num_proposals = self.num_proposals
        num_classes = self.num_classes
        test_cfg = self.test_cfg
        bbox_coder = self.bbox_coder

        batch_score = heatmap[..., -num_proposals:].sigmoid()
        query_labels = query_heatmap_score.max(1).indices
        one_hot = torch.nn.functional.one_hot(query_labels, num_classes=num_classes).permute(0, 2, 1)
        batch_score = batch_score * query_heatmap_score * one_hot.float()

        batch_center = center[..., -num_proposals:]
        batch_height = height[..., -num_proposals:]
        batch_dim = dim[..., -num_proposals:]
        batch_rot = rot[..., -num_proposals:]
        batch_vel = vel[..., -num_proposals:]

        temp = bbox_coder.decode(
            batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel,
            filter=True)

        tasks = [
            dict(num_class=8, class_names=[], indices=[0,1,2,3,4,5,6,7], radius=-1),
            dict(num_class=1, class_names=["pedestrian"], indices=[8], radius=0.175),
            dict(num_class=1, class_names=["traffic_cone"], indices=[9], radius=0.175),
        ]

        ret_layer = []
        for i in range(heatmap.shape[0]):
            boxes3d = temp[i]["bboxes"]
            scores = temp[i]["scores"]
            labels = temp[i]["labels"]

            if test_cfg.get("nms_type") is not None:
                keep_mask = torch.zeros_like(scores)
                for task in tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    if task["radius"] > 0:
                        if test_cfg["nms_type"] == "circle":
                            boxes_for_nms = torch.cat(
                                [boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                            task_keep_indices = torch.tensor(
                                circle_nms(boxes_for_nms.detach().cpu().numpy(), task["radius"]))
                        else:
                            boxes_for_nms = SimpleLiDARBox(boxes3d[task_mask][:, :7], 7).bev
                            top_scores = scores[task_mask]
                            task_keep_indices = nms_gpu(
                                boxes_for_nms, top_scores,
                                thresh=task["radius"],
                                pre_maxsize=test_cfg.get("pre_maxsize"),
                                post_max_size=test_cfg.get("post_maxsize"))
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())
                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                        keep_mask[keep_indices] = 1
                keep_mask = keep_mask.bool()
                ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
            else:
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
            ret_layer.append(ret)

        res = [[
            SimpleLiDARBox(ret_layer[0]["bboxes"], box_dim=ret_layer[0]["bboxes"].shape[-1]),
            ret_layer[0]["scores"],
            ret_layer[0]["labels"].int(),
        ]]
        return res

    @torch.no_grad()
    def _voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            f, c, n = self.voxelizer(res)
            feats.append(f)
            coords.append(torch.nn.functional.pad(c, (1, 0), mode="constant", value=k))
            sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        sizes = torch.cat(sizes, dim=0)
        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()
        return feats, coords, sizes


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(model, data_loader, logger):
    model.eval()
    results = []
    dataset = data_loader.dataset

    logger.info(f"Running evaluation on {len(dataset)} samples...")
    t_start = time.time()

    for i, data in enumerate(data_loader):
        img = data["img"].data[0].cuda()
        points = [p.cuda() for p in data["points"].data[0]]
        metas = data["metas"].data[0]
        camera2ego = data["camera2ego"].data[0].cuda()
        lidar2ego = data["lidar2ego"].data[0].cuda()
        lidar2camera = data["lidar2camera"].data[0].cuda()
        lidar2image = data["lidar2image"].data[0].cuda()
        camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
        camera2lidar = data["camera2lidar"].data[0].cuda()
        img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
        lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

        with torch.no_grad():
            outputs = model.forward_single(
                img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas)
        results.extend(outputs)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            logger.info(f"  [{i+1}/{len(data_loader)}] {fps:.1f} samples/s")

    elapsed = time.time() - t_start
    logger.info(f"Inference done: {len(results)} samples in {elapsed:.1f}s "
                f"({len(results)/elapsed:.1f} fps)")

    logger.info("Computing NDS metrics...")
    eval_results = dataset.evaluate(results)
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v}")
    return eval_results


def run_single_test(model, data_loader, logger):
    model.eval()
    data = next(iter(data_loader))

    img = data["img"].data[0].cuda()
    points = [p.cuda() for p in data["points"].data[0]]
    metas = data["metas"].data[0]
    camera2ego = data["camera2ego"].data[0].cuda()
    lidar2ego = data["lidar2ego"].data[0].cuda()
    lidar2camera = data["lidar2camera"].data[0].cuda()
    lidar2image = data["lidar2image"].data[0].cuda()
    camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
    camera2lidar = data["camera2lidar"].data[0].cuda()
    img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
    lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

    logger.info(f"Input shapes: img={img.shape}, points={points[0].shape}")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.forward_single(
            img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
            camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, metas)
    t1 = time.time()

    logger.info(f"Inference time: {(t1-t0)*1000:.1f} ms")
    for k, v in outputs[0].items():
        if hasattr(v, 'shape'):
            logger.info(f"  {k}: shape={v.shape}")
        elif hasattr(v, 'tensor'):
            logger.info(f"  {k}: shape={v.tensor.shape}")
        else:
            logger.info(f"  {k}: {type(v)}")

    n_boxes = outputs[0]["scores_3d"].shape[0]
    high_conf = (outputs[0]["scores_3d"] > 0.3).sum().item()
    logger.info(f"Detections: {n_boxes} total, {high_conf} with score > 0.3")
    return outputs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEVFusion Standalone TRT Inference (spconv 2.3)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--swin-engine", required=True)
    parser.add_argument("--depthnet-engine", required=True)
    parser.add_argument("--fuser-engine", required=True)
    parser.add_argument("--neck-engine", required=True)
    parser.add_argument("--head-engine", required=True)
    parser.add_argument("--lidar-quant", choices=["none", "w8a16", "int8"], default="none",
                        help="LiDAR backbone quantization mode")
    parser.add_argument("--ptq-ckpt", default="pretrained/ptq_minmax_model.pth",
                        help="PTQ checkpoint for LiDAR quantization")
    parser.add_argument("--test-single", action="store_true")
    parser.add_argument("--no-torch-lidar", action="store_true",
                        help="Use tv.Tensor LiDAR backbone (no PyTorch in backbone)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("standalone")
    logger.info("BEVFusion Standalone TRT Pipeline (spconv 2.3)")

    # Load config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Build LiDAR backbone
    logger.info("Building LiDAR backbone (spconv 2.3)...")

    if args.no_torch_lidar:
        from tools.tv_sparse_encoder import TVSparseEncoder, get_cuda_arch
        arch = get_cuda_arch(0)
        logger.info(f"  TV mode (no PyTorch), arch={arch}")
        lidar_backbone = TVSparseEncoder(arch=arch, stream=0)
        if args.lidar_quant != "none":
            logger.info(f"  TV Quantization mode: {args.lidar_quant}")
            lidar_backbone.load_ptq_weights(args.ptq_ckpt)
        else:
            lidar_backbone.load_weights(args.ckpt)
    else:
        lidar_backbone = SparseEncoder23()
        if args.lidar_quant != "none":
            logger.info(f"  Quantization mode: {args.lidar_quant}")
            lidar_backbone = quantize_sparse_encoder(lidar_backbone, logger)
            load_ptq_lidar_weights(lidar_backbone, args.ptq_ckpt, logger)
        else:
            load_lidar_weights(lidar_backbone, args.ckpt, logger)
        lidar_backbone.eval().cuda().half()

    # Build voxelizer
    voxel_cfg = cfg.model.encoders.lidar.voxelize
    voxelizer = Voxelization(
        voxel_size=voxel_cfg.voxel_size,
        point_cloud_range=voxel_cfg.point_cloud_range,
        max_num_points=voxel_cfg.max_num_points,
        max_voxels=voxel_cfg.max_voxels,
    )

    # Build vtransform geometry
    vt_cfg = cfg.model.encoders.camera.vtransform
    vtransform_geom = VTransformGeometry(
        image_size=vt_cfg.image_size,
        feature_size=vt_cfg.feature_size,
        xbound=vt_cfg.xbound,
        ybound=vt_cfg.ybound,
        zbound=vt_cfg.zbound,
        dbound=vt_cfg.dbound,
    )

    # Build BEV downsample (vtransform downsample=2)
    vt_downsample = vt_cfg.get("downsample", 1)
    bev_downsample = None
    if vt_downsample > 1:
        C_bev = vt_cfg.out_channels  # 80
        bev_downsample = nn.Sequential(
            nn.Conv2d(C_bev, C_bev, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_bev),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_bev, C_bev, 3, stride=vt_downsample, padding=1, bias=False),
            nn.BatchNorm2d(C_bev),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_bev, C_bev, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_bev),
            nn.ReLU(inplace=True),
        )
        # Load weights from checkpoint
        ckpt_sd = torch.load(args.ckpt, map_location="cpu")
        ckpt_sd = ckpt_sd.get("state_dict", ckpt_sd)
        ds_prefix = "encoders.camera.vtransform.downsample."
        ds_state = {}
        for k, v in ckpt_sd.items():
            if k.startswith(ds_prefix):
                ds_state[k[len(ds_prefix):]] = v
        bev_downsample.load_state_dict(ds_state)
        bev_downsample.eval().cuda()
        logger.info(f"  BEV downsample loaded ({len(ds_state)} params, stride={vt_downsample})")

    # Build bbox coder
    head_cfg = cfg.model.heads.object
    bbox_coder = TransFusionBBoxCoder(
        pc_range=voxel_cfg.point_cloud_range,
        out_size_factor=head_cfg.train_cfg.out_size_factor,
        voxel_size=voxel_cfg.voxel_size,
        post_center_range=head_cfg.test_cfg.get("post_center_range",
                                                  [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]),
        score_threshold=head_cfg.test_cfg.get("score_threshold", None),
        code_size=head_cfg.common_heads.get("vel", [2, 2])[0] + 8 if "vel" in head_cfg.common_heads else 8,
    )

    # Load TRT engines
    logger.info("Loading TRT engines...")
    swin_trt = TRTRunner(args.swin_engine, logger)
    depthnet_trt = TRTRunner(args.depthnet_engine, logger)
    fuser_trt = TRTRunner(args.fuser_engine, logger)
    neck_trt = TRTRunner(args.neck_engine, logger)
    head_trt = TRTRunner(args.head_engine, logger)

    # Build hybrid model
    hybrid = StandaloneBEVFusion(
        swin_trt=swin_trt,
        depthnet_trt=depthnet_trt,
        fuser_trt=fuser_trt,
        neck_trt=neck_trt,
        head_trt=head_trt,
        lidar_backbone=lidar_backbone,
        voxelizer=voxelizer,
        vtransform_geom=vtransform_geom,
        bev_downsample=bev_downsample,
        bbox_coder=bbox_coder,
        test_cfg=head_cfg.test_cfg,
        num_proposals=head_cfg.num_proposals,
        num_classes=head_cfg.num_classes,
        voxelize_reduce=cfg.model.get("voxelize_reduce", True),
        logger=logger,
        use_tv_lidar=args.no_torch_lidar,
    )
    hybrid.eval().cuda()

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False)

    if args.test_single:
        run_single_test(hybrid, data_loader, logger)
    else:
        eval_results = run_evaluation(hybrid, data_loader, logger)
        import json
        out_path = f"trt_standalone_eval.json"
        with open(out_path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                       for k, v in eval_results.items()}, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
