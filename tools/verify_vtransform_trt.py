"""
验证 vtransform TRT 引擎的数值正确性
对比 PyTorch 输出 vs TRT 输出

用法:
    python tools/verify_vtransform_trt.py \
        --engine vtransform_int8_full.engine \
        --indices bev_indices.pth \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/ptq_minmax_model.pth

输出:
    - L2 distance between PyTorch and TRT outputs
    - Cosine similarity
    - Max absolute error
"""
import argparse
import sys
import os
import logging
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import ctypes

# Load TensorRT
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available, only PyTorch inference will run")


def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays"""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return float(dot / (norm_a * norm_b + 1e-8))


def run_pytorch(vtransform, indices_dict, dummy_img, dummy_depth):
    """Run PyTorch inference"""
    from tools.export_utils.export_vtransform import VTransformFullWrapper
    
    model = VTransformFullWrapper(vtransform, indices_dict)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_img, dummy_depth)
    
    return output.numpy()


def run_trt(engine_path, plugin_path, dummy_img, dummy_depth, indices_dict):
    """Run TensorRT inference"""
    if not TRT_AVAILABLE:
        print("TensorRT not available")
        return None
    
    # Load plugin
    if plugin_path and os.path.exists(plugin_path):
        ctypes.CDLL(plugin_path)
        print(f"Loaded plugin: {plugin_path}")
    
    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Get binding info
    n_bindings = engine.num_bindings
    print(f"Engine bindings: {n_bindings}")
    for i in range(n_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        dtype = engine.get_binding_dtype(i)
        print(f"  {name}: {shape} {'input' if is_input else 'output'} {dtype}")
    
    # Prepare inputs
    # Note: This is a simplified version. Real implementation needs proper buffer management
    print("\nNote: TRT inference code is a template. Complete implementation needs:")
    print("  - Proper CUDA buffer allocation")
    print("  - Asynchronous execution")
    print("  - Output retrieval")
    
    return None  # Placeholder


def main():
    parser = argparse.ArgumentParser(description='验证 vtransform TRT 引擎')
    parser.add_argument('--engine', required=True, help='TRT 引擎文件路径')
    parser.add_argument('--indices', required=True, help='预计算索引文件路径')
    parser.add_argument('--config', required=True, help='模型配置文件路径')
    parser.add_argument('--ckpt', required=True, help='PTQ 检查点路径')
    parser.add_argument('--plugin', default='tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so',
                        help='Plugin .so 文件路径')
    args = parser.parse_args()
    
    logger = get_root_logger(log_level=logging.INFO)
    
    # Load configuration
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mqbench.utils.state import enable_quantization
    from tools.quant_ptq_minmax import build_ptq_model
    
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    
    # Build model
    model, _, _ = build_ptq_model(cfg, logger)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    enable_quantization(model)
    model.eval()
    
    # Get vtransform
    vtransform = model.encoders.camera.vtransform
    
    # Load indices
    print(f"Loading indices from: {args.indices}")
    indices_dict = torch.load(args.indices, map_location='cpu')
    
    # Create dummy inputs
    B, N = 1, 6
    C = vtransform.C
    H_feat, W_feat = 32, 88
    H_img, W_img = 256, 704
    
    torch.manual_seed(42)
    dummy_img = torch.randn(B, N, C, H_feat, W_feat)
    dummy_depth = torch.randn(B, N, 1, H_img, W_img)
    
    print(f"\nInput shapes:")
    print(f"  img: {dummy_img.shape}")
    print(f"  depth: {dummy_depth.shape}")
    
    # Run PyTorch
    print("\nRunning PyTorch inference...")
    pytorch_out = run_pytorch(vtransform, indices_dict, dummy_img, dummy_depth)
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"PyTorch output stats: min={pytorch_out.min():.4f}, max={pytorch_out.max():.4f}, mean={pytorch_out.mean():.4f}")
    
    # Run TRT
    print("\nRunning TensorRT inference...")
    trt_out = run_trt(args.engine, args.plugin, dummy_img, dummy_depth, indices_dict)
    
    if trt_out is not None:
        print(f"TRT output shape: {trt_out.shape}")
        print(f"TRT output stats: min={trt_out.min():.4f}, max={trt_out.max():.4f}, mean={trt_out.mean():.4f}")
        
        # Compare
        l2_dist = np.linalg.norm(pytorch_out - trt_out)
        cos_sim = cosine_similarity(pytorch_out, trt_out)
        max_err = np.abs(pytorch_out - trt_out).max()
        
        print(f"\nComparison:")
        print(f"  L2 distance: {l2_dist:.6f}")
        print(f"  Cosine similarity: {cos_sim:.6f} (target > 0.999)")
        print(f"  Max absolute error: {max_err:.6f}")
        
        if cos_sim > 0.999:
            print("\n✅ Verification PASSED")
        else:
            print("\n❌ Verification FAILED")
    else:
        print("\n⚠️ TRT inference not completed (template only)")


if __name__ == '__main__':
    main()
