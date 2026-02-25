"""
Hybrid TRT Evaluation: Replace ConvFuser with TRT INT8 engine,
run full NDS evaluation on nuScenes to measure end-to-end accuracy impact.

Usage:
  python tools/trt_eval_hybrid.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --eval bbox
"""
import argparse
import copy
import os
import sys
import time
import warnings

sys.path.append(os.getcwd())

import numpy as np
try:
    np.long = int; np.int = int; np.float = float; np.bool = bool
except:
    pass

import mmcv
import torch
import torch.nn as nn
from torchpack.utils.config import configs
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet3d.utils import recursive_eval

H, W = 180, 180  # BEV grid size


class FuserForExport(nn.Module):
    """ONNX export wrapper: list input -> two separate tensor args."""
    def __init__(self, fuser):
        super().__init__()
        self.conv = fuser[0]
        self.bn = fuser[1]
        self.relu = fuser[2]

    def forward(self, camera_bev, lidar_bev):
        x = torch.cat([camera_bev, lidar_bev], dim=1)
        return self.relu(self.bn(self.conv(x)))


class TRTFuser(nn.Module):
    """Drop-in replacement for ConvFuser using TRT engine."""
    def __init__(self, engine_path):
        super().__init__()
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._output_buf = None

    def forward(self, inputs):
        camera_bev = inputs[0].contiguous()
        lidar_bev = inputs[1].contiguous()
        B = camera_bev.shape[0]

        if self._output_buf is None or self._output_buf.shape[0] != B:
            self._output_buf = torch.empty(
                B, 256, H, W, device=camera_bev.device, dtype=torch.float32
            ).contiguous()

        # Ensure FP32 for TRT
        if camera_bev.dtype != torch.float32:
            camera_bev = camera_bev.float().contiguous()
        if lidar_bev.dtype != torch.float32:
            lidar_bev = lidar_bev.float().contiguous()

        self.context.set_tensor_address('camera_bev', int(camera_bev.data_ptr()))
        self.context.set_tensor_address('lidar_bev', int(lidar_bev.data_ptr()))
        self.context.set_tensor_address('fused_bev', int(self._output_buf.data_ptr()))
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return self._output_buf


def make_calibrator(calib_data, cache_file):
    """Build INT8 calibrator from collected real BEV features."""
    import tensorrt as trt

    class RealDataCalibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self, data_list, cache_path):
            super().__init__()
            self.data_list = data_list
            self.batch_idx = 0
            self.cache_path = cache_path
            self.dev_a = torch.empty(1, 80, H, W, device='cuda').contiguous()
            self.dev_b = torch.empty(1, 256, H, W, device='cuda').contiguous()

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.batch_idx >= len(self.data_list):
                return None
            a, b = self.data_list[self.batch_idx]
            self.dev_a.copy_(a[:1].float())
            self.dev_b.copy_(b[:1].float())
            self.batch_idx += 1
            return [int(self.dev_a.data_ptr()), int(self.dev_b.data_ptr())]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_path, 'wb') as f:
                f.write(cache)

    return RealDataCalibrator(calib_data, cache_file)


def build_trt_engine(onnx_path, precision, calib=None):
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
                print(f'  Parse error: {parser.get_error(i)}')
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
    print(f'  {precision.upper()} engine: {engine_path} ({size_kb:.0f} KB)')
    return engine_path


def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid TRT+PyTorch NDS evaluation')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'],
                        default='int8', help='TRT precision for fuser')
    parser.add_argument('--calib-samples', type=int, default=50,
                        help='Number of samples for INT8 calibration')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip PyTorch FP32 baseline evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed, deterministic=True)

    out_dir = 'runs/trt_hybrid_eval'
    os.makedirs(out_dir, exist_ok=True)

    # ========== 1. Load config & build model ==========
    print('=' * 60)
    print('  Hybrid TRT Evaluation')
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
    checkpoint_data = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint_data.get('meta', {}):
        model.CLASSES = checkpoint_data['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model.eval()
    print(f'\n[1/5] Model loaded: {args.checkpoint}')

    # ========== 2. FP32 Baseline (optional) ==========
    if not args.skip_baseline:
        print('\n[2/5] Running PyTorch FP32 baseline...')
        model_dp = MMDataParallel(copy.deepcopy(model), device_ids=[0])
        outputs_fp32 = single_gpu_test(model_dp, data_loader)
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval))
        print('\n--- FP32 Baseline Results ---')
        result_fp32 = dataset.evaluate(outputs_fp32, **eval_kwargs)
        print(result_fp32)
        del model_dp
        torch.cuda.empty_cache()
    else:
        print('\n[2/5] Skipping FP32 baseline (--skip-baseline)')

    # ========== 3. Collect calibration data ==========
    print(f'\n[3/5] Collecting {args.calib_samples} calibration samples...')
    calib_data = []
    fuser_stats = {'min_a': float('inf'), 'max_a': float('-inf'),
                   'min_b': float('inf'), 'max_b': float('-inf')}

    def hook_fn(module, inputs, output):
        feat_list = inputs[0]  # List[Tensor]: [camera_bev, lidar_bev]
        a = feat_list[0].detach().cpu().float()
        b = feat_list[1].detach().cpu().float()
        calib_data.append((a, b))
        fuser_stats['min_a'] = min(fuser_stats['min_a'], a.min().item())
        fuser_stats['max_a'] = max(fuser_stats['max_a'], a.max().item())
        fuser_stats['min_b'] = min(fuser_stats['min_b'], b.min().item())
        fuser_stats['max_b'] = max(fuser_stats['max_b'], b.max().item())

    hook = model.fuser.register_forward_hook(hook_fn)
    model_calib = MMDataParallel(model, device_ids=[0])

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= args.calib_samples:
                break
            model_calib(**data, return_loss=False)
            if (i + 1) % 10 == 0:
                print(f'  Collected {i+1}/{args.calib_samples}')

    hook.remove()
    del model_calib
    torch.cuda.empty_cache()

    print(f'  Camera BEV range: [{fuser_stats["min_a"]:.4f}, {fuser_stats["max_a"]:.4f}]')
    print(f'  Lidar BEV range:  [{fuser_stats["min_b"]:.4f}, {fuser_stats["max_b"]:.4f}]')
    print(f'  Calibration samples shape: camera={calib_data[0][0].shape}, lidar={calib_data[0][1].shape}')

    # ========== 4. Export ONNX + Build TRT engine ==========
    print(f'\n[4/5] Exporting ONNX + building TRT {args.precision.upper()} engine...')

    wrapper = FuserForExport(model.fuser)
    wrapper.eval().cpu()
    onnx_path = os.path.join(out_dir, 'fuser_hybrid.onnx')
    dummy_a = torch.randn(1, 80, H, W)
    dummy_b = torch.randn(1, 256, H, W)
    torch.onnx.export(
        wrapper, (dummy_a, dummy_b), onnx_path,
        input_names=['camera_bev', 'lidar_bev'],
        output_names=['fused_bev'],
        opset_version=13, do_constant_folding=True,
    )
    print(f'  ONNX exported: {onnx_path}')

    calib = None
    if args.precision == 'int8':
        cache_file = os.path.join(out_dir, 'calibration_real.cache')
        # Delete old cache to force re-calibration with real data
        if os.path.exists(cache_file):
            os.remove(cache_file)
        calib = make_calibrator(calib_data, cache_file)

    engine_path = build_trt_engine(onnx_path, args.precision, calib)
    if engine_path is None:
        print('ERROR: Failed to build TRT engine!')
        return

    # Sanity check: compare PyTorch vs TRT output on real calibration data
    print('\n  Sanity check (PyTorch vs TRT on calibration data):')
    trt_fuser_check = TRTFuser(engine_path)
    original_fuser = model.fuser.cuda()
    cos_sims = []
    with torch.no_grad():
        for i in range(min(5, len(calib_data))):
            a, b = calib_data[i]
            a_gpu, b_gpu = a.cuda().float(), b.cuda().float()
            pt_out = original_fuser([a_gpu, b_gpu])
            trt_out = trt_fuser_check([a_gpu, b_gpu])
            cos = torch.nn.functional.cosine_similarity(
                pt_out.flatten().unsqueeze(0), trt_out.flatten().unsqueeze(0)
            ).item()
            cos_sims.append(cos)
            diff = (pt_out - trt_out).abs()
            print(f'    Sample {i}: cos={cos:.6f}, maxErr={diff.max():.4f}, '
                  f'PT[{pt_out.min():.2f},{pt_out.max():.2f}] '
                  f'TRT[{trt_out.min():.2f},{trt_out.max():.2f}]')
    avg_cos = sum(cos_sims) / len(cos_sims)
    print(f'    Average cosine similarity: {avg_cos:.6f}')
    if avg_cos < 0.99:
        print('    WARNING: Low cosine similarity! TRT output may be inaccurate.')
    del trt_fuser_check
    torch.cuda.empty_cache()

    # ========== 5. Replace fuser + Evaluate ==========
    print(f'\n[5/5] Replacing fuser with TRT {args.precision.upper()} engine, running evaluation...')
    trt_fuser = TRTFuser(engine_path)

    # Replace fuser in model
    model.fuser = trt_fuser
    model_dp = MMDataParallel(model, device_ids=[0])

    # Time the evaluation
    start_time = time.perf_counter()
    outputs_trt = single_gpu_test(model_dp, data_loader)
    eval_time = time.perf_counter() - start_time

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval))

    print(f'\n--- TRT {args.precision.upper()} Hybrid Results ---')
    result_trt = dataset.evaluate(outputs_trt, **eval_kwargs)
    print(result_trt)
    print(f'  Evaluation time: {eval_time:.1f}s')
    print(f'  Engine: {engine_path} ({os.path.getsize(engine_path)/1024:.0f} KB)')

    # ========== Summary ==========
    print('\n' + '=' * 60)
    print('  Summary')
    print('=' * 60)
    if not args.skip_baseline:
        nds_fp32 = result_fp32.get('pts_bbox_NuScenes/NDS', result_fp32.get('object/nds', 'N/A'))
        map_fp32 = result_fp32.get('pts_bbox_NuScenes/mAP', result_fp32.get('object/map', 'N/A'))
        print(f'  FP32 Baseline:   NDS={nds_fp32}, mAP={map_fp32}')
    nds_trt = result_trt.get('pts_bbox_NuScenes/NDS', result_trt.get('object/nds', 'N/A'))
    map_trt = result_trt.get('pts_bbox_NuScenes/mAP', result_trt.get('object/map', 'N/A'))
    print(f'  TRT {args.precision.upper():5s}:        NDS={nds_trt}, mAP={map_trt}')
    if not args.skip_baseline and isinstance(nds_fp32, float) and isinstance(nds_trt, float):
        print(f'  NDS delta:       {nds_trt - nds_fp32:+.4f}')
        print(f'  mAP delta:       {map_trt - map_fp32:+.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
