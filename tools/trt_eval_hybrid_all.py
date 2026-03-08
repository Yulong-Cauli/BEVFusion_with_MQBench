"""
Multi-module TRT Hybrid Evaluation.

Exports quantizable modules to TensorRT engines, replaces them in the
BEVFusion pipeline, and runs full NDS evaluation on nuScenes.

Modules exported (auto-detected):
  Always:
    1. fuser        (ConvFuser)           — cam_bev+lidar_bev → fused_bev
    2. dec_backbone  (SECOND)             — fused_bev → multi-scale features
    3. dec_neck      (SECONDFPN)          — multi-scale → concatenated feature
    4. camera_neck   (GeneralizedLSSFPN)  — multi-scale image → fused image features
  If camera backbone is ResNet (not SwinTransformer):
    5. camera_backbone (ResNet)           — images → multi-scale features

Usage:
  python tools/trt_eval_hybrid_all.py \\
      configs/.../convfuser.yaml pretrained/bevfusion-det.pth \\
      --precision int8 --calib-samples 50
"""
import argparse
import copy
import os
import sys
import time

sys.path.append(os.getcwd())

import numpy as np
try:
    np.long = int; np.int = int; np.float = float; np.bool = bool
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-load shared libraries on Linux to prevent ImportError when mmcv
# CUDA extensions try to dlopen libtorch_cuda_cu.so / libnvinfer.so.
if sys.platform.startswith('linux'):
    import ctypes
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    for _f in sorted(os.listdir(_torch_lib)):
        if 'cuda' in _f and _f.endswith('.so'):
            try:
                ctypes.CDLL(os.path.join(_torch_lib, _f), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        import importlib as _il
        _trt_lib_dir = _il.import_module('tensorrt_libs').__path__[0]
        for _f in sorted(os.listdir(_trt_lib_dir)):
            if _f.endswith('.so'):
                try:
                    ctypes.CDLL(os.path.join(_trt_lib_dir, _f),
                                mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except ImportError:
        pass

from torchpack.utils.config import configs
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet3d.utils import recursive_eval


# =====================================================================
#  Export Wrappers — deepcopy modules for safe ONNX export
# =====================================================================

class FuserForExport(nn.Module):
    """ConvFuser: two separate inputs → fused output."""
    def __init__(self, fuser):
        super().__init__()
        fc = copy.deepcopy(fuser)
        self.conv = fc[0]
        self.bn = fc[1]
        self.relu = fc[2]

    def forward(self, camera_bev, lidar_bev):
        x = torch.cat([camera_bev, lidar_bev], dim=1)
        return self.relu(self.bn(self.conv(x)))


class SECONDForExport(nn.Module):
    """SECOND backbone: single BEV tensor → tuple of multi-scale features."""
    def __init__(self, backbone):
        super().__init__()
        bb = copy.deepcopy(backbone)
        self.blocks = bb.blocks

    def forward(self, x):
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDFPNForExport(nn.Module):
    """SECONDFPN neck: two feature inputs → single concatenated output."""
    def __init__(self, neck):
        super().__init__()
        nc = copy.deepcopy(neck)
        self.deblocks = nc.deblocks

    def forward(self, feat0, feat1):
        up0 = self.deblocks[0](feat0)
        up1 = self.deblocks[1](feat1)
        return torch.cat([up0, up1], dim=1)


class CameraNeckForExport(nn.Module):
    """GeneralizedLSSFPN: three multi-scale inputs → two fused outputs."""
    def __init__(self, neck):
        super().__init__()
        nc = copy.deepcopy(neck)
        self.lateral_convs = nc.lateral_convs
        self.fpn_convs = nc.fpn_convs
        self.upsample_cfg = nc.upsample_cfg

    def forward(self, cam0, cam1, cam2):
        laterals = [cam0, cam1, cam2]
        # top-down path: i=1 then i=0
        for i in range(1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])
        return laterals[0], laterals[1]


class ResNetForExport(nn.Module):
    """ResNet backbone: image tensor → multi-scale features.
    Explicitly unrolls the forward pass for clean ONNX export.
    Assumes out_indices=[1,2,3] (3 outputs)."""
    def __init__(self, backbone):
        super().__init__()
        bb = copy.deepcopy(backbone)
        self.stem = nn.Sequential(bb.conv1, bb.norm1, bb.relu, bb.maxpool)
        self.layer1 = getattr(bb, bb.res_layers[0])
        self.layer2 = getattr(bb, bb.res_layers[1])
        self.layer3 = getattr(bb, bb.res_layers[2])
        self.layer4 = getattr(bb, bb.res_layers[3])

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# =====================================================================
#  TRT Inference Wrappers — drop-in replacements for original modules
# =====================================================================

class TRTBase(nn.Module):
    """Base class for TensorRT inference modules."""
    def __init__(self, engine_path):
        super().__init__()
        import tensorrt as trt
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self._trt_logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._bufs = {}

    def _buf(self, name, shape, device):
        key = (name, shape)
        if key not in self._bufs:
            self._bufs[key] = torch.empty(
                shape, device=device, dtype=torch.float32
            ).contiguous()
        return self._bufs[key]

    def _exec(self, tensors):
        """Set addresses and execute. tensors: dict name→tensor."""
        for name, t in tensors.items():
            self.context.set_tensor_address(name, int(t.data_ptr()))
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()


class TRTFuser(TRTBase):
    """Drop-in for ConvFuser: list[cam_bev, lidar_bev] → tensor."""
    def __init__(self, engine_path, out_shape):
        super().__init__(engine_path)
        self.out_shape = out_shape

    def forward(self, inputs):
        cam = inputs[0].contiguous().float()
        lid = inputs[1].contiguous().float()
        out = self._buf('fused_bev', self.out_shape, cam.device)
        self._exec({'camera_bev': cam, 'lidar_bev': lid, 'fused_bev': out})
        return out.clone()


class TRTDecBackbone(TRTBase):
    """Drop-in for SECOND: tensor → tuple(feat0, feat1)."""
    def __init__(self, engine_path, out_shapes):
        super().__init__(engine_path)
        self.out_shapes = out_shapes  # [shape0, shape1]

    def forward(self, x):
        x = x.contiguous().float()
        o0 = self._buf('feat0', self.out_shapes[0], x.device)
        o1 = self._buf('feat1', self.out_shapes[1], x.device)
        self._exec({'bev_feat': x, 'feat0': o0, 'feat1': o1})
        return (o0.clone(), o1.clone())


class TRTDecNeck(TRTBase):
    """Drop-in for SECONDFPN: tuple(feat0, feat1) → [tensor]."""
    def __init__(self, engine_path, out_shape):
        super().__init__(engine_path)
        self.out_shape = out_shape

    def forward(self, x):
        f0 = x[0].contiguous().float()
        f1 = x[1].contiguous().float()
        out = self._buf('neck_out', self.out_shape, f0.device)
        self._exec({'feat0': f0, 'feat1': f1, 'neck_out': out})
        return [out.clone()]


class TRTCameraNeck(TRTBase):
    """Drop-in for GeneralizedLSSFPN: tuple(3 tensors) → tuple(2 tensors)."""
    def __init__(self, engine_path, out_shapes):
        super().__init__(engine_path)
        self.out_shapes = out_shapes  # [shape0, shape1]

    def forward(self, inputs):
        c0 = inputs[0].contiguous().float()
        c1 = inputs[1].contiguous().float()
        c2 = inputs[2].contiguous().float()
        o0 = self._buf('out0', self.out_shapes[0], c0.device)
        o1 = self._buf('out1', self.out_shapes[1], c0.device)
        self._exec({
            'cam0': c0, 'cam1': c1, 'cam2': c2, 'out0': o0, 'out1': o1,
        })
        return (o0.clone(), o1.clone())


class TRTCameraBackbone(TRTBase):
    """Drop-in for ResNet backbone: tensor → list of 3 feature tensors."""
    def __init__(self, engine_path, out_shapes):
        super().__init__(engine_path)
        self.out_shapes = out_shapes  # [shape0, shape1, shape2]

    def forward(self, x):
        x = x.contiguous().float()
        o0 = self._buf('feat0', self.out_shapes[0], x.device)
        o1 = self._buf('feat1', self.out_shapes[1], x.device)
        o2 = self._buf('feat2', self.out_shapes[2], x.device)
        self._exec({'image': x, 'feat0': o0, 'feat1': o1, 'feat2': o2})
        return [o0.clone(), o1.clone(), o2.clone()]


# =====================================================================
#  INT8 Calibrator
# =====================================================================

def make_calibrator(data_list, input_names, cache_path):
    """Generic INT8 calibrator for any module."""
    import tensorrt as trt

    class Calibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self):
            super().__init__()
            self.data = data_list
            self.idx = 0
            self.cache = cache_path
            self.dev = []
            if data_list:
                for t in data_list[0]:
                    self.dev.append(
                        torch.empty(t.shape, device='cuda', dtype=torch.float32
                    ).contiguous())

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.data):
                return None
            sample = self.data[self.idx]
            ptrs = []
            for i, t in enumerate(sample):
                self.dev[i].copy_(t.float())
                ptrs.append(int(self.dev[i].data_ptr()))
            self.idx += 1
            return ptrs

        def read_calibration_cache(self):
            if os.path.exists(self.cache):
                with open(self.cache, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache, 'wb') as f:
                f.write(cache)

    return Calibrator()


# =====================================================================
#  TRT Engine Building
# =====================================================================

def build_engine(onnx_path, precision, calib=None):
    """Build TRT engine from ONNX. Returns engine file path or None."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'    ONNX parse error: {parser.get_error(i)}')
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if calib:
            config.int8_calibrator = calib

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None

    engine_path = onnx_path.replace('.onnx', f'_{precision}.engine')
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    size_kb = os.path.getsize(engine_path) / 1024
    print(f'    {os.path.basename(engine_path)}: {size_kb:.0f} KB')
    return engine_path


# =====================================================================
#  Main
# =====================================================================

BASE_MODULE_NAMES = ['camera_neck', 'fuser', 'dec_backbone', 'dec_neck']


def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-module TRT Hybrid NDS evaluation')
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file')
    p.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    p.add_argument('--precision', choices=['fp32', 'fp16', 'int8'],
                   default='int8')
    p.add_argument('--calib-samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-baseline', action='store_true')
    p.add_argument('--out-dir', type=str, default='runs/trt_hybrid_all',
                   help='output directory for ONNX and TRT engines')
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed, deterministic=True)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ==================== 1. Load model ====================
    print('=' * 60)
    print('  Multi-Module Hybrid TRT Evaluation')
    print('=' * 60)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = ckpt.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.eval()
    print(f'\n[1/7] Model loaded: {args.checkpoint}')

    # Detect camera backbone type — ResNet is TRT-exportable
    cam_bb_type = cfg.model.encoders.camera.backbone.type
    include_cam_backbone = cam_bb_type in ('ResNet',)
    MODULE_NAMES = list(BASE_MODULE_NAMES)
    if include_cam_backbone:
        MODULE_NAMES.insert(0, 'camera_backbone')
    n_modules = len(MODULE_NAMES)
    print(f'  Camera backbone: {cam_bb_type}'
          f' → {"included" if include_cam_backbone else "skipped"}'
          f' ({n_modules} modules total)')

    # ==================== 2. Export ONNX ====================
    print(f'\n[2/7] Exporting ONNX for {n_modules} modules...')

    # We need actual shapes — run one sample to collect them via hooks
    shapes = {}

    def shape_hook(name):
        def fn(module, inputs, output):
            if name in shapes:
                return
            inp = inputs[0]
            if isinstance(inp, (list, tuple)):
                in_s = [tuple(t.shape) for t in inp]
            else:
                in_s = [tuple(inp.shape)]
            if isinstance(output, (list, tuple)):
                out_s = [tuple(t.shape) for t in output]
            else:
                out_s = [tuple(output.shape)]
            shapes[name] = {'in': in_s, 'out': out_s}
        return fn

    hooks = [
        model.encoders["camera"]["neck"].register_forward_hook(
            shape_hook('camera_neck')),
        model.fuser.register_forward_hook(shape_hook('fuser')),
        model.decoder["backbone"].register_forward_hook(
            shape_hook('dec_backbone')),
        model.decoder["neck"].register_forward_hook(shape_hook('dec_neck')),
    ]
    if include_cam_backbone:
        hooks.append(model.encoders["camera"]["backbone"].register_forward_hook(
            shape_hook('camera_backbone')))

    model_tmp = MMDataParallel(model, device_ids=[0])
    with torch.no_grad():
        for data in data_loader:
            model_tmp(**data, return_loss=False)
            break
    for h in hooks:
        h.remove()
    del model_tmp
    torch.cuda.empty_cache()

    for name in MODULE_NAMES:
        s = shapes[name]
        in_str = ' + '.join(str(list(x)) for x in s['in'])
        out_str = ' + '.join(str(list(x)) for x in s['out'])
        print(f'  {name}: {in_str} → {out_str}')

    # Export each module
    onnx_paths = {}

    # Fuser
    w = FuserForExport(model.fuser).eval().cpu()
    p = os.path.join(out_dir, 'fuser.onnx')
    torch.onnx.export(
        w,
        (torch.randn(*shapes['fuser']['in'][0]),
         torch.randn(*shapes['fuser']['in'][1])),
        p, input_names=['camera_bev', 'lidar_bev'],
        output_names=['fused_bev'], opset_version=13,
        do_constant_folding=True)
    onnx_paths['fuser'] = p
    del w

    # Decoder backbone
    w = SECONDForExport(model.decoder["backbone"]).eval().cpu()
    p = os.path.join(out_dir, 'dec_backbone.onnx')
    torch.onnx.export(
        w,
        (torch.randn(*shapes['dec_backbone']['in'][0]),),
        p, input_names=['bev_feat'],
        output_names=['feat0', 'feat1'], opset_version=13,
        do_constant_folding=True)
    onnx_paths['dec_backbone'] = p
    del w

    # Decoder neck
    w = SECONDFPNForExport(model.decoder["neck"]).eval().cpu()
    p = os.path.join(out_dir, 'dec_neck.onnx')
    torch.onnx.export(
        w,
        (torch.randn(*shapes['dec_neck']['in'][0]),
         torch.randn(*shapes['dec_neck']['in'][1])),
        p, input_names=['feat0', 'feat1'],
        output_names=['neck_out'], opset_version=13,
        do_constant_folding=True)
    onnx_paths['dec_neck'] = p
    del w

    # Camera neck
    w = CameraNeckForExport(model.encoders["camera"]["neck"]).eval().cpu()
    p = os.path.join(out_dir, 'camera_neck.onnx')
    torch.onnx.export(
        w,
        (torch.randn(*shapes['camera_neck']['in'][0]),
         torch.randn(*shapes['camera_neck']['in'][1]),
         torch.randn(*shapes['camera_neck']['in'][2])),
        p, input_names=['cam0', 'cam1', 'cam2'],
        output_names=['out0', 'out1'], opset_version=13,
        do_constant_folding=True)
    onnx_paths['camera_neck'] = p
    del w

    # Camera backbone (ResNet only)
    if include_cam_backbone:
        w = ResNetForExport(model.encoders["camera"]["backbone"]).eval().cpu()
        p = os.path.join(out_dir, 'camera_backbone.onnx')
        torch.onnx.export(
            w,
            (torch.randn(*shapes['camera_backbone']['in'][0]),),
            p, input_names=['image'],
            output_names=['feat0', 'feat1', 'feat2'], opset_version=13,
            do_constant_folding=True)
        onnx_paths['camera_backbone'] = p
        del w

    print('  ONNX export complete.')

    # Verify model params still on CUDA
    dev = next(model.fuser.parameters()).device
    print(f'  model.fuser device after export: {dev}')

    # ==================== 3. Collect calibration data ====================
    calib_data = {n: [] for n in MODULE_NAMES}

    if args.precision == 'int8':
        print(f'\n[3/7] Collecting {args.calib_samples} calibration samples...')

        def calib_hook(name):
            def fn(module, inputs, output):
                inp = inputs[0]
                if isinstance(inp, (list, tuple)):
                    calib_data[name].append(
                        tuple(t.detach().cpu().float() for t in inp))
                else:
                    calib_data[name].append(
                        (inp.detach().cpu().float(),))
            return fn

        hooks = [
            model.encoders["camera"]["neck"].register_forward_hook(
                calib_hook('camera_neck')),
            model.fuser.register_forward_hook(calib_hook('fuser')),
            model.decoder["backbone"].register_forward_hook(
                calib_hook('dec_backbone')),
            model.decoder["neck"].register_forward_hook(
                calib_hook('dec_neck')),
        ]
        if include_cam_backbone:
            hooks.append(model.encoders["camera"]["backbone"].register_forward_hook(
                calib_hook('camera_backbone')))

        model_calib = MMDataParallel(model, device_ids=[0])
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= args.calib_samples:
                    break
                model_calib(**data, return_loss=False)
                if (i + 1) % 10 == 0:
                    print(f'    {i+1}/{args.calib_samples}')

        for h in hooks:
            h.remove()
        del model_calib
        torch.cuda.empty_cache()

        for name in MODULE_NAMES:
            print(f'    {name}: {len(calib_data[name])} samples')
    else:
        print(f'\n[3/7] Skipping calibration (precision={args.precision})')

    # ==================== 4. FP32 baseline ====================
    if not args.skip_baseline:
        print(f'\n[4/7] Running PyTorch FP32 baseline...')
        model_dp = MMDataParallel(copy.deepcopy(model), device_ids=[0])
        outputs_fp32 = single_gpu_test(model_dp, data_loader)
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                     'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs['metric'] = args.eval
        print('\n  --- FP32 Baseline ---')
        result_fp32 = dataset.evaluate(outputs_fp32, **eval_kwargs)
        nds_fp32 = result_fp32.get('pts_bbox_NuScenes/NDS',
                                    result_fp32.get('object/nds', 'N/A'))
        map_fp32 = result_fp32.get('pts_bbox_NuScenes/mAP',
                                    result_fp32.get('object/map', 'N/A'))
        print(f'  NDS={nds_fp32}, mAP={map_fp32}')
        del model_dp
        torch.cuda.empty_cache()
    else:
        print(f'\n[4/7] Skipping FP32 baseline')

    # ==================== 5. Build TRT engines ====================
    print(f'\n[5/7] Building TRT {args.precision.upper()} engines...')

    engine_paths = {}
    for name in MODULE_NAMES:
        calib = None
        if args.precision == 'int8' and calib_data[name]:
            cache = os.path.join(out_dir, f'{name}_calib.cache')
            if os.path.exists(cache):
                os.remove(cache)
            in_names = {
                'fuser': ['camera_bev', 'lidar_bev'],
                'dec_backbone': ['bev_feat'],
                'dec_neck': ['feat0', 'feat1'],
                'camera_neck': ['cam0', 'cam1', 'cam2'],
                'camera_backbone': ['image'],
            }[name]
            calib = make_calibrator(calib_data[name], in_names, cache)

        ep = build_engine(onnx_paths[name], args.precision, calib)
        if ep is None:
            print(f'    ERROR: Failed to build {name} engine!')
            return
        engine_paths[name] = ep

    # Free calibration data
    del calib_data
    torch.cuda.empty_cache()

    # ==================== 6. Sanity check ====================
    print(f'\n[6/7] Sanity check (PyTorch vs TRT, first sample)...')
    model.cuda()

    # Quick forward to get reference outputs from each module
    ref_data = {}

    def ref_hook(name):
        def fn(module, inputs, output):
            inp = inputs[0]
            if isinstance(inp, (list, tuple)):
                ref_data[name] = {
                    'in': [t.detach() for t in inp],
                    'out': [t.detach() for t in output]
                        if isinstance(output, (list, tuple))
                        else [output.detach()],
                }
            else:
                ref_data[name] = {
                    'in': [inp.detach()],
                    'out': [t.detach() for t in output]
                        if isinstance(output, (list, tuple))
                        else [output.detach()],
                }
        return fn

    hooks = [
        model.encoders["camera"]["neck"].register_forward_hook(
            ref_hook('camera_neck')),
        model.fuser.register_forward_hook(ref_hook('fuser')),
        model.decoder["backbone"].register_forward_hook(
            ref_hook('dec_backbone')),
        model.decoder["neck"].register_forward_hook(ref_hook('dec_neck')),
    ]
    if include_cam_backbone:
        hooks.append(model.encoders["camera"]["backbone"].register_forward_hook(
            ref_hook('camera_backbone')))

    model_ref = MMDataParallel(model, device_ids=[0])
    with torch.no_grad():
        for data in data_loader:
            model_ref(**data, return_loss=False)
            break
    for h in hooks:
        h.remove()
    del model_ref
    torch.cuda.empty_cache()

    # Test each TRT engine against reference
    trt_wrappers_test = {
        'fuser': TRTFuser(engine_paths['fuser'],
                          tuple(shapes['fuser']['out'][0])),
        'dec_backbone': TRTDecBackbone(engine_paths['dec_backbone'],
                                        [tuple(s) for s in
                                         shapes['dec_backbone']['out']]),
        'dec_neck': TRTDecNeck(engine_paths['dec_neck'],
                               tuple(shapes['dec_neck']['out'][0])),
        'camera_neck': TRTCameraNeck(engine_paths['camera_neck'],
                                      [tuple(s) for s in
                                       shapes['camera_neck']['out']]),
    }
    if include_cam_backbone:
        trt_wrappers_test['camera_backbone'] = TRTCameraBackbone(
            engine_paths['camera_backbone'],
            [tuple(s) for s in shapes['camera_backbone']['out']])

    with torch.no_grad():
        for name in MODULE_NAMES:
            ref = ref_data[name]
            trt_mod = trt_wrappers_test[name]
            # dec_backbone and camera_backbone take a single tensor
            if name in ('dec_backbone', 'camera_backbone'):
                trt_out = trt_mod(ref['in'][0])
            else:
                trt_out = trt_mod(ref['in'])
            if not isinstance(trt_out, (list, tuple)):
                trt_out = [trt_out]
            for j, (ro, to) in enumerate(zip(ref['out'], trt_out)):
                cos = torch.nn.functional.cosine_similarity(
                    ro.flatten().unsqueeze(0),
                    to.flatten().unsqueeze(0)).item()
                maxerr = (ro - to).abs().max().item()
                print(f'  {name}[{j}]: cos={cos:.6f}, maxErr={maxerr:.4f}')

    del trt_wrappers_test, ref_data
    torch.cuda.empty_cache()

    # ==================== 7. Replace & evaluate ====================
    print(f'\n[7/7] Replacing {n_modules} modules with TRT {args.precision.upper()}'
          f' engines...')

    # Create TRT wrappers with correct output shapes
    if include_cam_backbone:
        model.encoders["camera"]["backbone"] = TRTCameraBackbone(
            engine_paths['camera_backbone'],
            [tuple(s) for s in shapes['camera_backbone']['out']])
    model.encoders["camera"]["neck"] = TRTCameraNeck(
        engine_paths['camera_neck'],
        [tuple(s) for s in shapes['camera_neck']['out']])
    model.fuser = TRTFuser(
        engine_paths['fuser'],
        tuple(shapes['fuser']['out'][0]))
    model.decoder["backbone"] = TRTDecBackbone(
        engine_paths['dec_backbone'],
        [tuple(s) for s in shapes['dec_backbone']['out']])
    model.decoder["neck"] = TRTDecNeck(
        engine_paths['dec_neck'],
        tuple(shapes['dec_neck']['out'][0]))

    model_dp = MMDataParallel(model, device_ids=[0])

    t0 = time.perf_counter()
    outputs_trt = single_gpu_test(model_dp, data_loader)
    eval_time = time.perf_counter() - t0

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs['metric'] = args.eval

    print(f'\n  --- TRT {args.precision.upper()} All-Module Hybrid ---')
    result_trt = dataset.evaluate(outputs_trt, **eval_kwargs)
    nds_trt = result_trt.get('pts_bbox_NuScenes/NDS',
                              result_trt.get('object/nds', 'N/A'))
    map_trt = result_trt.get('pts_bbox_NuScenes/mAP',
                              result_trt.get('object/map', 'N/A'))
    print(f'  Eval time: {eval_time:.1f}s')

    # Engine sizes
    print('\n  Engine sizes:')
    total_kb = 0
    for name in MODULE_NAMES:
        sz = os.path.getsize(engine_paths[name]) / 1024
        total_kb += sz
        print(f'    {name}: {sz:.0f} KB')
    print(f'    TOTAL: {total_kb:.0f} KB ({total_kb/1024:.1f} MB)')

    # ==================== Summary ====================
    print('\n' + '=' * 60)
    print('  Summary')
    print('=' * 60)
    if not args.skip_baseline:
        print(f'  FP32 Baseline:          NDS={nds_fp32}, mAP={map_fp32}')
    print(f'  TRT {args.precision.upper():5s} ({n_modules} modules): '
          f'NDS={nds_trt}, mAP={map_trt}')
    if not args.skip_baseline and isinstance(nds_fp32, float) \
            and isinstance(nds_trt, float):
        print(f'  NDS delta:              {nds_trt - nds_fp32:+.4f}')
        print(f'  mAP delta:              {map_trt - map_fp32:+.4f}')
    print(f'  Total engine size:      {total_kb:.0f} KB ({total_kb/1024:.1f} MB)')
    fp32_pth_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    print(f'  FP32 .pth size:         {fp32_pth_mb:.1f} MB')
    if total_kb > 0:
        print(f'  Compression ({n_modules} modules): {fp32_pth_mb / (total_kb/1024):.1f}x')
    print('=' * 60)


if __name__ == '__main__':
    main()
