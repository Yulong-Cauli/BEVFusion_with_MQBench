"""
Export ConvFuser to ONNX and build TensorRT FP32/FP16/INT8 engines.
Benchmark latency and model size comparison.
"""
import os, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.makedirs('runs/trt_export', exist_ok=True)

from mmdet3d.models.fusers.conv import ConvFuser

H, W = 180, 180


class FuserForExport(nn.Module):
    """Wrapper: converts list input to two separate args for clean ONNX."""
    def __init__(self, fuser):
        super().__init__()
        self.conv = fuser[0]
        self.bn = fuser[1]
        self.relu = fuser[2]

    def forward(self, feat_a, feat_b):
        x = torch.cat([feat_a, feat_b], dim=1)
        return self.relu(self.bn(self.conv(x)))


def export_onnx():
    fuser = ConvFuser(in_channels=[80, 256], out_channels=256)
    fuser.eval()
    wrapper = FuserForExport(fuser)
    wrapper.eval()

    dummy_a = torch.randn(1, 80, H, W)
    dummy_b = torch.randn(1, 256, H, W)

    onnx_path = 'runs/trt_export/fuser_fp32.onnx'
    torch.onnx.export(
        wrapper, (dummy_a, dummy_b), onnx_path,
        input_names=['camera_bev', 'lidar_bev'],
        output_names=['fused_bev'],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f'ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)')

    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print('ONNX model valid!')
    return onnx_path


def build_engine(onnx_path, precision, calib=None):
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
                print(f'  ONNX parse error: {parser.get_error(i)}')
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

    print(f'Building {precision.upper()} engine...')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print(f'  Engine build FAILED for {precision}!')
        return None

    engine_path = f'runs/trt_export/fuser_{precision}.engine'
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    size_kb = os.path.getsize(engine_path) / 1024
    print(f'  {precision.upper()} engine saved: {engine_path} ({size_kb:.0f} KB)')
    return engine_path, size_kb


class SimpleCalibrator:
    """INT8 entropy calibrator using random data."""
    def __init__(self, num_batches=50):
        import tensorrt as trt
        self._base = trt.IInt8EntropyCalibrator2.__init__
        self.num_batches = num_batches
        self.batch_idx = 0
        self.device_inputs = [
            torch.randn(1, 80, H, W, device='cuda').contiguous(),
            torch.randn(1, 256, H, W, device='cuda').contiguous(),
        ]
        self.cache_file = 'runs/trt_export/calibration.cache'

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= self.num_batches:
            return None
        self.batch_idx += 1
        self.device_inputs[0].normal_()
        self.device_inputs[1].normal_()
        return [int(t.data_ptr()) for t in self.device_inputs]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def make_calibrator():
    import tensorrt as trt

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, num_batches=50):
            super().__init__()
            self.num_batches = num_batches
            self.batch_idx = 0
            self.device_inputs = [
                torch.randn(1, 80, H, W, device='cuda').contiguous(),
                torch.randn(1, 256, H, W, device='cuda').contiguous(),
            ]
            self.cache_file = 'runs/trt_export/calibration.cache'

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.batch_idx >= self.num_batches:
                return None
            self.batch_idx += 1
            self.device_inputs[0].normal_()
            self.device_inputs[1].normal_()
            return [int(t.data_ptr()) for t in self.device_inputs]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    return Calibrator()


def benchmark_engine(engine_path, warmup=50, iters=200):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    inputs_gpu = [
        torch.randn(1, 80, H, W, device='cuda').contiguous(),
        torch.randn(1, 256, H, W, device='cuda').contiguous(),
    ]
    output_gpu = torch.empty(1, 256, H, W, device='cuda').contiguous()

    context.set_tensor_address('camera_bev', int(inputs_gpu[0].data_ptr()))
    context.set_tensor_address('lidar_bev', int(inputs_gpu[1].data_ptr()))
    context.set_tensor_address('fused_bev', int(output_gpu.data_ptr()))

    stream = torch.cuda.Stream()

    for _ in range(warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000


def benchmark_pytorch(warmup=50, iters=200):
    """Benchmark PyTorch FP32 ConvFuser for comparison."""
    fuser = ConvFuser(in_channels=[80, 256], out_channels=256).cuda().eval()
    a = torch.randn(1, 80, H, W, device='cuda')
    b = torch.randn(1, 256, H, W, device='cuda')

    with torch.no_grad():
        for _ in range(warmup):
            fuser([a, b])
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fuser([a, b])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000


def main():
    print('=' * 60)
    print('  ConvFuser TensorRT Export & Benchmark')
    print('=' * 60)

    # Step 1: Export ONNX
    print('\n[1/4] Exporting FP32 ONNX...')
    onnx_path = export_onnx()

    # Step 2: Build engines
    print('\n[2/4] Building TRT engines...')
    results = {}
    for prec in ['fp32', 'fp16', 'int8']:
        calib = make_calibrator() if prec == 'int8' else None
        ret = build_engine(onnx_path, prec, calib)
        if ret:
            results[prec] = {'path': ret[0], 'size_kb': ret[1]}

    # Step 3: PyTorch baseline
    print('\n[3/4] PyTorch FP32 baseline...')
    pytorch_ms = benchmark_pytorch()
    print(f'  PyTorch FP32: {pytorch_ms:.3f} ms')

    # Step 4: Benchmark TRT engines
    print('\n[4/4] Benchmarking TRT engines...')
    for prec in results:
        try:
            lat = benchmark_engine(results[prec]['path'])
            results[prec]['latency_ms'] = lat
        except Exception as e:
            print(f'  {prec.upper()}: benchmark failed: {e}')

    # Summary
    print('\n' + '=' * 60)
    print('  ConvFuser Performance Summary')
    print('=' * 60)
    print(f'  {"Method":<15} {"Latency":>10} {"Speedup":>10} {"Size":>10} {"Compress":>10}')
    print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
    print(f'  {"PyTorch FP32":<15} {pytorch_ms:>8.3f}ms {"1.00x":>10} {"N/A":>10} {"N/A":>10}')

    for prec in ['fp32', 'fp16', 'int8']:
        if prec in results and 'latency_ms' in results[prec]:
            lat = results[prec]['latency_ms']
            kb = results[prec]['size_kb']
            speedup = pytorch_ms / lat if lat > 0 else 0
            fp32_kb = results.get('fp32', {}).get('size_kb', kb)
            compress = fp32_kb / kb if kb > 0 else 0
            print(f'  {"TRT " + prec.upper():<15} {lat:>8.3f}ms {speedup:>8.2f}x  {kb:>8.0f}KB {compress:>8.2f}x')

    print('=' * 60)


if __name__ == '__main__':
    main()
