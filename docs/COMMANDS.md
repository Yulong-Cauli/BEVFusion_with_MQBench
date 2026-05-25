# BEVFusion 运行命令全集

> 本文档汇总了项目全部可运行命令，按场景分类：PTQ 量化、TRT 部署、Zero-Torch 验证、Orin 部署。
>
> 工作目录：`cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench`

---

## 0. 公共变量

```bash
CFG=configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
CKPT=pretrained/bevfusion-det.pth
mkdir -p logs runs
```

---

## 1. PTQ 量化命令（`bevfusion_mqbench` 环境）

```bash
conda activate bevfusion_mqbench
```

### 1.1 FP32 基线评估

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/test.py $CFG $CKPT --eval bbox \
2>&1 | tee logs/fp32_baseline.log
# 期望: NDS 0.7069, mAP 0.6728
```

### 1.2 PTQ 8/8 全量化（推荐：vtransform KL + lidar Log2）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --vtransform-observer kl_divergence \
  --act-observer log2 --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ptq_88_kl_log2 \
  2>&1 | tee logs/ptq_88_kl_log2.log
# 期望: NDS 0.6875, mAP 0.6429
```

### 1.3 PTQ 7/8 推荐配置（skip lidar，精度最高）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules lidar/backbone \
  --vtransform-observer kl_divergence \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ptq_78_vtkl \
  2>&1 | tee logs/ptq_78_vtkl.log
# 期望: NDS 0.7033, mAP 0.6657
```

### 1.4 PTQ 6/8 基线（skip vtransform + lidar）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform lidar/backbone \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ptq_68_baseline \
  2>&1 | tee logs/ptq_68_baseline.log
# 期望: NDS 0.7010, mAP 0.6614
```

### 1.5 BRECQ 8/8 全量化（4 卡 DDP）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --standalone \
  tools/quant_ptq_brecq.py $CFG --load-from $CKPT \
  --calib-batches 256 --cache-batches 64 \
  --recon-iters 2000 \
  --w-lr 4e-4 --a-lr 4e-5 \
  --drop-prob 0.5 --round-loss-weight 0.01 --warm-up 0.2 \
  --sparse-act-mode per_tensor \
  --run-dir runs/brecq_88 \
  2>&1 | tee logs/brecq_88.log
```

### 1.6 消融实验命令速查

| 组合 | 命令差异 |
|------|---------|
| vt KL + lidar MinMax | `--vtransform-observer kl_divergence --act-observer ema_minmax` |
| vt MinMax + lidar Log2 | `--vtransform-observer ema_minmax --act-observer log2 --log-base 2.0` |
| vt KL + lidar Log2 | `--vtransform-observer kl_divergence --act-observer log2` |
| Log2 底数 sweep | `--act-observer log2 --log-base <a>` |

详见 `docs/ablation.md` 获取完整实验矩阵与已有结果。

---

## 2. TRT 部署命令（`bevfusion_mqbench` 环境）

### 2.1 单样本冒烟测试（Hybrid pipeline）

```bash
conda activate bevfusion_mqbench

# Phase 5 混合方案
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config $CFG --ckpt $CKPT \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_fp16_sm86.engine \
    --version A --test-single \
    2>&1 | tee logs/smoke_test.log

# Phase 6 全 TRT（含 neck + head）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config $CFG --ckpt $CKPT \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_fp16_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --version A --test-single \
    2>&1 | tee logs/smoke_test_phase6.log
```

### 2.2 NDS 评估 — 4 卡并行

```bash
mkdir -p trt_shards logs

for SHARD in 0 1 2 3; do
  GPU_IDS=(0 1 3 4)
  GPU=${GPU_IDS[$SHARD]}
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  python -u tools/trt_infer.py \
      --config $CFG --ckpt $CKPT \
      --swin-engine swin_int8_sm86.engine \
      --depthnet-engine vtransform_depthnet_int8_sm86.engine \
      --fuser-engine fuser_decoder_int8_sm86.engine \
      --neck-engine camera_neck_int8_sm86.engine \
      --head-engine transfusion_head_int8_sm86.engine \
      --version B --num-shards 4 --shard-id $SHARD \
      --out-dir trt_shards \
      2>&1 | tee logs/trt_eval_B_shard${SHARD}.log &
done
wait

python -u tools/merge_eval.py \
    --config $CFG --pred-dir trt_shards --version B --num-shards 4 \
    2>&1 | tee logs/trt_eval_B_merged.log
```

### 2.3 重建 TRT 引擎

```bash
# SwinT INT8
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx --engine swin_int8_sm86.engine --int8 --fp16

# depthnet INT8
python tools/export_utils/build_engine.py \
    --onnx vtransform_depthnet_int8.onnx --engine vtransform_depthnet_int8_sm86.engine --int8 --fp16

# Fuser+Decoder INT8
python tools/export_utils/build_engine.py \
    --onnx fuser_decoder_int8.onnx --engine fuser_decoder_int8_sm86.engine --int8 --fp16

# Camera Neck INT8
python tools/export_utils/build_engine.py \
    --onnx camera_neck_int8.onnx --engine camera_neck_int8_sm86.engine --int8 --fp16

# TransFusionHead INT8
python tools/export_utils/build_engine.py \
    --onnx transfusion_head_int8.onnx --engine transfusion_head_int8_sm86.engine --int8 --fp16
```

---

## 3. Standalone 部署命令（`spconv23_deploy` 环境）

```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
```

### 3.1 环境准备（首次）

```bash
# 编译 CUDA 扩展
python tools/build_cuda_ext.py
```

### 3.2 单样本冒烟测试

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config $CFG --ckpt $CKPT \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --test-single \
    2>&1 | tee logs/standalone_smoke.log
```

### 3.3 完整 NDS 评估（TV LiDAR FP16，去 PyTorch）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config $CFG --ckpt $CKPT \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --no-torch-lidar \
    2>&1 | tee logs/standalone_eval_tv_fp16.log
# 期望 NDS 0.7039
```

### 3.4 完整 NDS 评估（TV LiDAR INT8 Log2）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config $CFG --ckpt $CKPT \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    --no-torch-lidar \
    2>&1 | tee logs/standalone_eval_tv_int8.log
# 期望 NDS 0.6893, mAP 0.6474
```

---

## 4. Zero-Torch 验证命令（`bevfusion_mqbench` 环境）

### 4.1 端到端单样本一致性验证

```bash
conda activate bevfusion_mqbench

python tools/validate_e2e_zero_torch.py \
  --config $CFG --ckpt $CKPT \
  --swin-engine artifacts/swin_int8_sm86.engine \
  --auto-build-swin-batch \
  --depthnet-engine artifacts/vtransform_depthnet_int8_sm86.engine \
  --fuser-engine artifacts/fuser_decoder_int8_sm86.engine \
  --neck-engine artifacts/camera_neck_int8_sm86.engine \
  --head-engine artifacts/transfusion_head_int8_sm86.engine \
  --bev-downsample-engine artifacts/bev_downsample_fp32_sm86.engine \
  --lidar-npy-dir pretrained/lidar_npy_fp16
```

### 4.2 全量评估（6019 帧，x86 基线）

```bash
python tools/eval_zero_torch_full.py \
  --config $CFG --ckpt $CKPT \
  --swin-engine artifacts/swin_int8_sm86.engine \
  --auto-build-swin-batch \
  --depthnet-engine artifacts/vtransform_depthnet_int8_sm86.engine \
  --fuser-engine artifacts/fuser_decoder_int8_sm86.engine \
  --neck-engine artifacts/camera_neck_int8_sm86.engine \
  --head-engine artifacts/transfusion_head_int8_sm86.engine \
  --bev-downsample-engine artifacts/bev_downsample_fp32_sm86.engine \
  --lidar-npy-dir pretrained/lidar_npy_fp16 \
  --workers 2 \
  --out zero_torch_eval.json
```

---

## 5. Orin 部署命令

> 目标平台：Jetson AGX Orin 64GB + L4T 36.4.4 + TRT 10.3.x

### 5.1 环境准备

```bash
conda create -n bo python=3.10 -y
conda activate bo
pip install -U pip setuptools wheel
pip install -e .

# 版本留档
python - <<'PY' | tee logs/orin_env_versions.txt
import sys
print("python:", sys.version)
try: import tensorrt as trt; print("tensorrt:", trt.__version__)
except Exception as e: print("tensorrt: ERROR", e)
PY
```

### 5.2 编译本地 CUDA 扩展

```bash
# TV Log2 CUDA so（Orin 本机编译，SM87）
nvcc -shared -O3 -Xcompiler -fPIC \
  -gencode arch=compute_87,code=sm_87 \
  tools/tv_log2_quant.cu -lcudart -o tools/libtv_log2_quant.so

# mmdet3d CUDA 扩展
python tools/build_cuda_ext.py

# vtransform GPU zero-copy so
cd tools/zero_torch_ops/vtransform_gpu
python build_vtransform_gpu.py
cd ../../..
```

### 5.3 在 Orin 上重建 TRT 引擎（SM87）

```bash
python tools/export_utils/build_engine.py \
  --onnx artifacts/swin_int8.onnx --engine artifacts/swin_int8_sm87.engine --int8 --fp16

python tools/export_utils/build_engine.py \
  --onnx artifacts/vtransform_depthnet_int8.onnx --engine artifacts/vtransform_depthnet_int8_sm87.engine --int8 --fp16

python tools/export_utils/build_engine.py \
  --onnx artifacts/fuser_decoder_int8.onnx --engine artifacts/fuser_decoder_int8_sm87.engine --int8 --fp16

python tools/export_utils/build_engine.py \
  --onnx artifacts/camera_neck_int8.onnx --engine artifacts/camera_neck_int8_sm87.engine --int8 --fp16

python tools/export_utils/build_engine.py \
  --onnx artifacts/transfusion_head_int8.onnx --engine artifacts/transfusion_head_int8_sm87.engine --int8 --fp16

python tools/export_utils/build_engine.py \
  --onnx artifacts/bev_downsample_fp32_sm86.onnx --engine artifacts/bev_downsample_fp32_sm87.engine --fp16
```

### 5.4 单样本冒烟测试

```bash
python tools/validate_e2e_zero_torch.py \
  --config $CFG --ckpt $CKPT \
  --swin-engine artifacts/swin_int8_sm87.engine \
  --auto-build-swin-batch \
  --depthnet-engine artifacts/vtransform_depthnet_int8_sm87.engine \
  --fuser-engine artifacts/fuser_decoder_int8_sm87.engine \
  --neck-engine artifacts/camera_neck_int8_sm87.engine \
  --head-engine artifacts/transfusion_head_int8_sm87.engine \
  --bev-downsample-engine artifacts/bev_downsample_fp32_sm87.engine \
  --lidar-npy-dir pretrained/lidar_npy_fp16
```

### 5.5 全量评估

```bash
python tools/eval_zero_torch_full.py \
  --config $CFG --ckpt $CKPT \
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

---

## 6. 注意事项

| 项目 | 说明 |
|------|------|
| GPU 映射 | 服务器上 GPU 0,1,3,4 = RTX 3090 (SM 8.6)，GPU 2 = A100 |
| 引擎架构绑定 | `_sm86` 引擎不能在 A100 (SM 8.0) 或 Orin (SM 8.7) 上运行 |
| 4 卡并行 | 每卡独立加载模型 + 引擎，显存约 14GB/卡 |
| 合并评估 | `merge_eval.py` 不需要 GPU，只做 NDS 计算 |
| 校准集 | 使用训练集 (`cfg.data.train`)，关闭数据增强 (`test_mode=True`) |
