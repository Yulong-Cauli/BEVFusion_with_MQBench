"""
Numerical diff between two TRT engine versions for the same ONNX.
Saves/loads I/O tensors as NPZ files to avoid version conflicts in one process.

Usage (two-step):
    # Step 1: generate random inputs and run reference engine (TRT 10.15)
    python tools/validate_trt_engine_numdiff.py \
        --engine artifacts/camera_neck_int8_sm86.engine \
        --save-inputs /tmp/neck_inputs.npz \
        --save-outputs /tmp/neck_ref.npz

    # Step 2: run test engine (TRT 10.3) against same inputs
    export TRT103=/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/trt_10.3_env
    export PYTHONPATH=$TRT103:$PYTHONPATH
    export LD_LIBRARY_PATH=$TRT103/tensorrt_libs:$LD_LIBRARY_PATH
    python tools/validate_trt_engine_numdiff.py \
        --engine artifacts/camera_neck_int8_trt103.engine \
        --load-inputs /tmp/neck_inputs.npz \
        --save-outputs /tmp/neck_test.npz

    # Step 3: compare
    python tools/validate_trt_engine_numdiff.py \
        --compare /tmp/neck_ref.npz /tmp/neck_test.npz
"""
import argparse
import os
import sys
import numpy as np

# Import TRT (version determined by PYTHONPATH/LD_LIBRARY_PATH at runtime)
import tensorrt as trt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.trt_infer_zero_torch import ZeroTorchTRTRunner, make_cuda_buffer_from_array


def run_engine(engine_path, inputs_npz=None, outputs_npz=None, save_inputs_npz=None):
    runner = ZeroTorchTRTRunner(engine_path)

    if inputs_npz is not None:
        data = np.load(inputs_npz)
        input_buffers = {name: data[name] for name in runner.input_names}
    else:
        input_buffers = {}
        for name in runner.input_names:
            shape = tuple(runner.engine.get_tensor_shape(name))
            dtype = runner.engine.get_tensor_dtype(name)
            np_dtype = np.float16 if dtype == trt.float16 else np.float32
            arr = np.random.randn(*shape).astype(np_dtype)
            input_buffers[name] = arr

    # Upload to GPU buffers and run inference
    input_buffers_gpu = {k: make_cuda_buffer_from_array(v) for k, v in input_buffers.items()}
    results = runner(input_buffers_gpu)

    if save_inputs_npz is not None:
        np.savez(save_inputs_npz, **input_buffers)
        print(f"Saved inputs to {save_inputs_npz}")

    if outputs_npz is not None:
        outs = {name: arr for name, arr in zip(runner.output_names, results)}
        np.savez(outputs_npz, **outs)
        print(f"Saved outputs to {outputs_npz}")
        for name, arr in outs.items():
            print(f"  {name}: {arr.shape} {arr.dtype}  mean={arr.mean():.6f}  std={arr.std():.6f}")


def compare_npz(ref_path, test_path):
    ref = np.load(ref_path)
    test = np.load(test_path)
    print(f"Comparing {ref_path} vs {test_path}")
    all_pass = True
    for name in ref.files:
        a = ref[name]
        b = test[name]
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            all_pass = False
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_diff = diff.max()
        mean_diff = diff.mean()
        norm_a = np.linalg.norm(a.flatten())
        norm_b = np.linalg.norm(b.flatten())
        cos_sim = float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b + 1e-12))
        status = "PASS" if max_diff < 1e-3 else "WARN" if max_diff < 1e-2 else "FAIL"
        print(f"  {name}: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} cos_sim={cos_sim:.8f} [{status}]")
        if status == "FAIL":
            all_pass = False
    if all_pass:
        print("✅ All tensors PASS")
    else:
        print("❌ Some tensors FAIL")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Path to .engine")
    parser.add_argument("--load-inputs", type=str, help="Load inputs from .npz")
    parser.add_argument("--save-inputs", type=str, help="Save inputs to .npz")
    parser.add_argument("--save-outputs", type=str, help="Save outputs to .npz")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "TEST"), help="Compare two .npz files")
    args = parser.parse_args()

    if args.compare:
        ok = compare_npz(args.compare[0], args.compare[1])
        sys.exit(0 if ok else 1)

    if not args.engine:
        parser.error("--engine required unless --compare")

    run_engine(args.engine, args.load_inputs, args.save_outputs, args.save_inputs)


if __name__ == "__main__":
    main()
