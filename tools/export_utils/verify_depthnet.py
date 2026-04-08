"""
验证 depthnet TRT 引擎精度：对比 TRT 输出 vs PyTorch PTQ FakeQuant 输出。

用法:
    python tools/export_utils/verify_depthnet.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/ptq_minmax_model.pth \
        --engine vtransform_depthnet_int8.engine \
        --threshold 0.999
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import tensorrt as trt
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval
from mqbench.utils.state import enable_quantization
from tools.quant_ptq_minmax import build_ptq_model


def cosine_sim(a, b):
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_trt_engine(engine_path, inputs_dict):
    """Run TRT engine using torch CUDA tensors (avoids pycuda context conflicts)."""
    logger_trt = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger_trt)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Set input shapes and addresses using torch tensors
    device_tensors = {}
    for name, arr in inputs_dict.items():
        t = torch.from_numpy(arr).cuda().contiguous()
        context.set_input_shape(name, tuple(t.shape))
        context.set_tensor_address(name, t.data_ptr())
        device_tensors[name] = t

    # Allocate output tensors
    output_tensors = {}
    num_io = engine.num_io_tensors
    for i in range(num_io):
        name = engine.get_tensor_name(i)
        if name in inputs_dict:
            continue
        shape = tuple(context.get_tensor_shape(name))
        dtype_trt = engine.get_tensor_dtype(name)
        if dtype_trt == trt.float32:
            dtype_torch = torch.float32
        elif dtype_trt == trt.float16:
            dtype_torch = torch.float16
        else:
            dtype_torch = torch.float32
        t = torch.zeros(shape, dtype=dtype_torch, device="cuda").contiguous()
        context.set_tensor_address(name, t.data_ptr())
        output_tensors[name] = t

    # Execute on default stream
    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    results = {}
    for name, t in output_tensors.items():
        results[name] = t.cpu().numpy()
    return results


def _load_ptq_vtransform(cfg, ckpt_path, logger):
    """构建 PTQ 量化模型并加载 checkpoint，返回 vtransform。"""
    model, _, _ = build_ptq_model(cfg, logger)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # 修复 shape 不匹配（KL observer 额外 buffer 等）
    model_sd = model.state_dict()
    for k, v in list(state_dict.items()):
        if k in model_sd and v.shape != model_sd[k].shape:
            if v.numel() == model_sd[k].numel():
                state_dict[k] = v.reshape(model_sd[k].shape)
            else:
                # 不同 numel：resize 模型参数以匹配 checkpoint
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
                        v.clone(), requires_grad=old.requires_grad))
                else:
                    setattr(obj, param_name, v.clone())

    model.load_state_dict(state_dict, strict=False)
    enable_quantization(model)
    model.eval()

    vtransform = model.encoders.camera.vtransform.cuda().eval()
    del model
    torch.cuda.empty_cache()
    return vtransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    vtransform = _load_ptq_vtransform(cfg, args.ckpt, logger)

    # 构造 dummy 输入
    B, N = 1, 6
    C_in = vtransform.in_channels
    fH, fW = vtransform.feature_size
    H_img, W_img = vtransform.image_size

    torch.manual_seed(42)
    dummy_img = torch.randn(B, N, C_in, fH, fW).cuda()
    dummy_depth = torch.randn(B, N, 1, H_img, W_img).cuda()

    # PyTorch forward (depthnet only)
    with torch.no_grad():
        x = vtransform.get_cam_feats(dummy_img, dummy_depth)
        # [B, N, D, fH, fW, C] -> [B*N*D*fH*fW, C]
        pt_out = x.reshape(-1, vtransform.C).cpu().numpy()

    logger.info(f"PyTorch output shape: {pt_out.shape}")

    # TRT forward
    inputs_dict = {
        "image_features": dummy_img.cpu().numpy(),
        "depth_input": dummy_depth.cpu().numpy(),
    }
    trt_results = run_trt_engine(args.engine, inputs_dict)
    trt_out = list(trt_results.values())[0]

    logger.info(f"TRT output shape: {trt_out.shape}")

    # Compare
    cs = cosine_sim(pt_out, trt_out)
    mae = float(np.abs(pt_out.astype(np.float32) - trt_out.astype(np.float32)).max())
    rmse = float(np.sqrt(np.mean((pt_out.astype(np.float32) - trt_out.astype(np.float32)) ** 2)))

    logger.info(f"{'=' * 50}")
    logger.info(f"  cosine_sim  : {cs:.6f}  (threshold > {args.threshold})")
    logger.info(f"  max_abs_err : {mae:.6f}")
    logger.info(f"  rmse        : {rmse:.6f}")

    if cs > args.threshold:
        logger.info(f"  PASS")
    else:
        logger.error(f"  FAIL")
        exit(1)


if __name__ == "__main__":
    main()
