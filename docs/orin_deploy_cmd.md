# Zero-Torch 在 Jetson Orin 的构建与部署命令

> 目标：在 **Orin(SM87)** 本机重建 TRT engine 并运行 zero-torch（严格模式，无 PyTorch fallback）。
>
> 说明：本仓库打包文件不包含 NuScenes 数据集；数据集通过挂载或软链接接入。
>
> 当前适配基线：**AGX Orin 64GB + L4T 36.4.4**（JetPack 6 系列，TRT 10.3.x）。

## 0. 版本标注（必须先做）

```bash
mkdir -p logs

# 系统版本（L4T / JetPack）
cat /etc/nv_tegra_release | tee logs/orin_l4t.txt

# 推荐：conda Python 与系统 Python 主版本一致（L4T 36.4.4 通常是 3.10）
conda create -n bevfusion_orin python=3.10 -y
conda activate bevfusion_orin

pip install -U pip setuptools wheel
pip install -e .

# 如果 conda 环境里 import tensorrt 失败，可桥接系统 Python 包路径
export PYTHONPATH=/usr/lib/python3.10/dist-packages:${PYTHONPATH}

# 冻结并留档（后续复现/交付就看这个文件）
python - <<'PY' | tee logs/orin_env_versions.txt
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e:
    print("torch: ERROR", e)
try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as e:
    print("tensorrt: ERROR", e)
try:
    import spconv
    print("spconv:", spconv.__version__)
except Exception as e:
    print("spconv: ERROR", e)
PY
```

> `orin_env_versions.txt` 中 `torch / tensorrt / spconv` 任一为 `ERROR` 时不要继续，先补齐环境再部署。

## 1. 解包与环境

```bash
mkdir -p ~/work && cd ~/work
tar -xzf orin_zero_torch_src_20260417.tar.gz
cd BEVFusion_with_MQBench

# 上一节已建好 conda 环境时，这里只需激活
conda activate bevfusion_orin
```

## 2. 数据集接入（不打包数据）

```bash
# 假设 NuScenes 在 /data/sets/nuscenes
mkdir -p data
ln -sfn /data/sets/nuscenes data/nuscenes
```

## 3. 编译本地 CUDA 扩展

```bash
# 3.1 重新编译 TV Log2 CUDA so（必须在 Orin 本机编译）
nvcc -shared -O3 -Xcompiler -fPIC \
  -gencode arch=compute_87,code=sm_87 \
  tools/tv_log2_quant.cu -lcudart -o tools/libtv_log2_quant.so

# 3.2 编译 mmdet3d CUDA 扩展
python tools/build_cuda_ext.py

# 3.3 编译 vtransform GPU zero-copy so
cd tools/zero_torch_ops/vtransform_gpu
python build_vtransform_gpu.py
cd ../../..
```

## 4. 在 Orin 上重建 TRT 引擎（SM87）

```bash
mkdir -p artifacts logs

python tools/export_utils/build_engine.py \
  --onnx artifacts/swin_int8.onnx \
  --engine artifacts/swin_int8_sm87.engine \
  --int8 --fp16 2>&1 | tee logs/build_swin_sm87.log

python tools/export_utils/build_engine.py \
  --onnx artifacts/vtransform_depthnet_int8.onnx \
  --engine artifacts/vtransform_depthnet_int8_sm87.engine \
  --int8 --fp16 2>&1 | tee logs/build_depthnet_sm87.log

python tools/export_utils/build_engine.py \
  --onnx artifacts/fuser_decoder_int8.onnx \
  --engine artifacts/fuser_decoder_int8_sm87.engine \
  --int8 --fp16 2>&1 | tee logs/build_fuser_sm87.log

python tools/export_utils/build_engine.py \
  --onnx artifacts/camera_neck_int8.onnx \
  --engine artifacts/camera_neck_int8_sm87.engine \
  --int8 --fp16 2>&1 | tee logs/build_neck_sm87.log

python tools/export_utils/build_engine.py \
  --onnx artifacts/transfusion_head_int8.onnx \
  --engine artifacts/transfusion_head_int8_sm87.engine \
  --int8 --fp16 2>&1 | tee logs/build_head_sm87.log

python tools/export_utils/build_engine.py \
  --onnx artifacts/bev_downsample_fp32_sm86.onnx \
  --engine artifacts/bev_downsample_fp32_sm87.engine \
  --fp16 2>&1 | tee logs/build_bev_downsample_sm87.log
```

## 5. 单样本一致性冒烟（严格 zero-torch）

```bash
python tools/validate_e2e_zero_torch.py \
  --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --ckpt pretrained/bevfusion-det.pth \
  --swin-engine artifacts/swin_int8_sm87.engine \
  --auto-build-swin-batch \
  --depthnet-engine artifacts/vtransform_depthnet_int8_sm87.engine \
  --fuser-engine artifacts/fuser_decoder_int8_sm87.engine \
  --neck-engine artifacts/camera_neck_int8_sm87.engine \
  --head-engine artifacts/transfusion_head_int8_sm87.engine \
  --bev-downsample-engine artifacts/bev_downsample_fp32_sm87.engine \
  --lidar-npy-dir pretrained/lidar_npy_fp16
```

## 6. 全量评估（6019 帧）

```bash
python tools/eval_zero_torch_full.py \
  --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --ckpt pretrained/bevfusion-det.pth \
  --swin-engine artifacts/swin_int8_sm87.engine \
  --auto-build-swin-batch \
  --depthnet-engine artifacts/vtransform_depthnet_int8_sm87.engine \
  --fuser-engine artifacts/fuser_decoder_int8_sm87.engine \
  --neck-engine artifacts/camera_neck_int8_sm87.engine \
  --head-engine artifacts/transfusion_head_int8_sm87.engine \
  --bev-downsample-engine artifacts/bev_downsample_fp32_sm87.engine \
  --lidar-npy-dir pretrained/lidar_npy_fp16 \
  --workers 2 \
  --out zero_torch_eval_orin.json
```

> 首次 `--auto-build-swin-batch` 会自动生成 `artifacts/swin_int8_b6_sm87.engine`（后续复用）。
