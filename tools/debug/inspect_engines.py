"""
Inspect existing TRT engines for layer composition and compatibility risks.
Useful for pre-validating TRT 10.3 / Orin readiness.
"""
import argparse
import os
import sys

import tensorrt as trt


def inspect_engine(path):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"  FAILED to deserialize: {path}")
        return

    print(f"Engine: {os.path.basename(path)}")
    print(f"  TRT runtime version: {trt.__version__}")
    print(f"  IO tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        print(f"    {name}: {mode.name} {shape} {dtype}")

    # Attempt to get profile information (TRT 10+)
    try:
        nb_profiles = engine.num_optimization_profiles
        print(f"  Optimization profiles: {nb_profiles}")
    except Exception:
        pass

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", nargs="+", help="Path(s) to .engine file(s)")
    args = parser.parse_args()

    for p in args.engine:
        if not os.path.exists(p):
            print(f"Not found: {p}")
            continue
        inspect_engine(p)


if __name__ == "__main__":
    main()
