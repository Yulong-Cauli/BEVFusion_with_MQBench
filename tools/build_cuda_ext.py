"""
Build CUDA extensions for spconv23_deploy environment (Python 3.9, PyTorch 2.0).

Compiles:
  1. bev_pool_ext  — BEV pooling CUDA kernel
  2. voxel_layer   — Hard voxelization CUDA kernel
  3. iou3d_cuda    — Rotated NMS CUDA kernel

Usage:
    conda run --prefix /media/yellowstone/data2/CYL/spconv23_deploy \
        python tools/build_cuda_ext.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build_sp39")
os.makedirs(BUILD_DIR, exist_ok=True)

from torch.utils.cpp_extension import load


def build_bev_pool():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/bev_pool/src")
    print("Building bev_pool_ext ...")
    mod = load(
        name="bev_pool_ext",
        sources=[
            os.path.join(src_dir, "bev_pool_cpu.cpp"),
            os.path.join(src_dir, "bev_pool_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        verbose=True,
    )
    print(f"  bev_pool_ext built: {mod}")
    return mod


def build_voxel_layer():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/voxel/src")
    print("Building voxel_layer ...")
    mod = load(
        name="voxel_layer",
        sources=[
            os.path.join(src_dir, "voxelization.cpp"),
            os.path.join(src_dir, "voxelization_cpu.cpp"),
            os.path.join(src_dir, "voxelization_cuda.cu"),
            os.path.join(src_dir, "scatter_points_cpu.cpp"),
            os.path.join(src_dir, "scatter_points_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cflags=["-w", "-DWITH_CUDA"],
        extra_cuda_cflags=["-w", "-DWITH_CUDA"],
        verbose=True,
    )
    print(f"  voxel_layer built: {mod}")
    return mod


def build_iou3d():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/iou3d/src")
    print("Building iou3d_cuda ...")
    mod = load(
        name="iou3d_cuda",
        sources=[
            os.path.join(src_dir, "iou3d.cpp"),
            os.path.join(src_dir, "iou3d_kernel.cu"),
        ],
        build_directory=BUILD_DIR,
        verbose=True,
    )
    print(f"  iou3d_cuda built: {mod}")
    return mod


def build_roiaware_pool3d():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/roiaware_pool3d/src")
    print("Building roiaware_pool3d_ext ...")
    mod = load(
        name="roiaware_pool3d_ext",
        sources=[
            os.path.join(src_dir, "roiaware_pool3d.cpp"),
            os.path.join(src_dir, "roiaware_pool3d_kernel.cu"),
            os.path.join(src_dir, "points_in_boxes_cpu.cpp"),
            os.path.join(src_dir, "points_in_boxes_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cflags=["-w"],
        extra_cuda_cflags=["-w"],
        verbose=True,
    )
    print(f"  roiaware_pool3d_ext built: {mod}")
    return mod


if __name__ == "__main__":
    print(f"Build directory: {BUILD_DIR}")
    print(f"Python: {sys.executable}")

    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    build_bev_pool()
    build_voxel_layer()
    build_iou3d()
    build_roiaware_pool3d()

    print("\nAll CUDA extensions built successfully.")
