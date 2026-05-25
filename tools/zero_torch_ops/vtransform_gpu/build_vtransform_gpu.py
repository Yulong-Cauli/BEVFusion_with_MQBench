"""
Build libbevfusion_vtransform_gpu.so without torch dependency.
Usage:
    python build_vtransform_gpu.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(HERE, "libbevfusion_vtransform_gpu.so")


def get_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.isdir(cuda_home):
        raise RuntimeError(f"CUDA_HOME not found: {cuda_home}")
    return cuda_home


def build():
    cuda_home = get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(f"nvcc not found: {nvcc}")

    # Server RTX 3090 = sm_86; Orin = sm_87; A100 = sm_80
    arch_flags = [
        "-gencode", "arch=compute_80,code=sm_80",
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_87,code=sm_87",
    ]

    cmd = [
        nvcc,
        "-shared",
        "-O3",
        "-Xcompiler", "-fPIC",
        *arch_flags,
        os.path.join(HERE, "vtransform_cuda.cu"),
        "-lcudart",
        "-o", SO_PATH,
    ]

    print("Building:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Built: {SO_PATH}")


if __name__ == "__main__":
    build()
