# BEVFusion TensorRT 部署命令手册

> **Orin 新版命令请优先使用：`docs/orin_deploy_cmd.md`**（SM87 本机重建引擎，strict zero-torch）。
>
> 所有命令在项目根目录下运行：`cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench`
> 环境：`conda activate bevfusion_mqbench`
> GPU 映射：GPU 0,1,3,4 = RTX 3090 (SM 8.6)，GPU 2 = A100 (跳过)

---

## 环境准备

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
```

---

## 1. 单样本冒烟测试（单卡）

```bash
# Phase 5 混合方案（无 neck/head 引擎）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_fp16_sm86.engine \
    --version A --test-single \
    2>&1 | tee logs/smoke_test.log

# Phase 6 全 TRT（含 neck + head 引擎）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_fp16_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --version A --test-single \
    2>&1 | tee logs/smoke_test_phase6.log
```

---

## 2. NDS 评估 — Version A (W8A16, FP16 fuser) — 4卡并行

### Phase 5 混合方案（无 neck/head 引擎）

```bash
mkdir -p trt_shards logs

for SHARD in 0 1 2 3; do
  GPU_IDS=(0 1 3 4)
  GPU=${GPU_IDS[$SHARD]}
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  python -u tools/trt_infer.py \
      --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --ckpt pretrained/bevfusion-det.pth \
      --swin-engine swin_int8_sm86.engine \
      --depthnet-engine vtransform_depthnet_int8_sm86.engine \
      --fuser-engine fuser_decoder_fp16_sm86.engine \
      --version A --num-shards 4 --shard-id $SHARD \
      --out-dir trt_shards \
      2>&1 | tee logs/trt_eval_A_shard${SHARD}.log &
done
wait
echo "All shards done"

# 合并 + NDS 评估
python -u tools/merge_eval.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --pred-dir trt_shards --version A --num-shards 4 \
    2>&1 | tee logs/trt_eval_A_merged.log
```

### Phase 6 全 TRT（含 neck + head 引擎）— 单卡评估

```bash
mkdir -p logs

# Version A (W8A16): FP16 fuser, INT8 neck + head
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_fp16_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --version A \
    2>&1 | tee logs/trt_eval_phase6_A.log

# Version B (全 INT8): INT8 fuser + INT8 neck + INT8 head
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --version B \
    2>&1 | tee logs/trt_eval_phase6_B.log
```

---

## 3. NDS 评估 — Version B (INT8 fuser) — 4卡并行

```bash
mkdir -p trt_shards logs

for SHARD in 0 1 2 3; do
  GPU_IDS=(0 1 3 4)
  GPU=${GPU_IDS[$SHARD]}
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  python -u tools/trt_infer.py \
      --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --ckpt pretrained/bevfusion-det.pth \
      --swin-engine swin_int8_sm86.engine \
      --depthnet-engine vtransform_depthnet_int8_sm86.engine \
      --fuser-engine fuser_decoder_int8_sm86.engine \
      --version B --num-shards 4 --shard-id $SHARD \
      --out-dir trt_shards \
      2>&1 | tee logs/trt_eval_B_shard${SHARD}.log &
done
wait
echo "All shards done"

python -u tools/merge_eval.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --pred-dir trt_shards --version B --num-shards 4 \
    2>&1 | tee logs/trt_eval_B_merged.log
```

---

## 4. 重建 TRT 引擎（换 GPU 后需要重建）

```bash
# SwinT INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx --engine swin_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_swin.log

# depthnet INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx vtransform_depthnet_int8.onnx --engine vtransform_depthnet_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_depthnet.log

# Fuser+Decoder FP16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx fuser_decoder_fp16.onnx --engine fuser_decoder_fp16_sm86.engine --fp16 \
    2>&1 | tee logs/build_fuser_fp16.log

# Fuser+Decoder INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx fuser_decoder_int8.onnx --engine fuser_decoder_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_fuser_int8.log

# TransFusionHead FP16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx transfusion_head_fp16.onnx --engine transfusion_head_fp16_sm86.engine --fp16 \
    2>&1 | tee logs/build_head.log

# Camera Neck INT8 (Phase 6)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx camera_neck_int8.onnx --engine camera_neck_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_neck_int8.log

# Camera Neck FP16 (Phase 6)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx camera_neck_fp16.onnx --engine camera_neck_fp16_sm86.engine --fp16 \
    2>&1 | tee logs/build_neck_fp16.log

# TransFusionHead INT8 (Phase 6)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx transfusion_head_int8.onnx --engine transfusion_head_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_head_int8.log

# TransFusionHead FP16 v2 (Phase 6, 修复 argsort→topk)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx transfusion_head_fp16_v2.onnx --engine transfusion_head_fp16_v2_sm86.engine --fp16 \
    2>&1 | tee logs/build_head_fp16_v2.log
```

---

## 5. FP32 基线评估（对照用）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --eval bbox \
    2>&1 | tee logs/fp32_baseline.log
# 期望: NDS 0.7069, mAP 0.6728
```

---

## 已有结果（单卡，2026-03-29）

| 版本 | NDS | mAP | 速度 | 日志 |
|------|-----|-----|------|------|
| FP32 baseline | 0.7069 | 0.6728 | — | — |
| Version A (W8A16) | 0.7144 | 0.6851 | 1.1 fps (单卡) | logs/trt_eval_version_A.log |
| Version B (INT8) | 0.7102 | 0.6786 | 5.0 fps (单卡) | logs/trt_eval_version_B.log |

## Phase 6 新增引擎（2026-03-29）

| 引擎 | 大小 | 说明 |
|------|------|------|
| camera_neck_int8_sm86.engine | 1.8 MB | Camera Neck INT8, 输入 3 个多尺度特征, 输出 [6,256,32,88] |
| camera_neck_fp16_sm86.engine | 3.2 MB | Camera Neck FP16 |
| transfusion_head_int8_sm86.engine | 3.7 MB | TransFusionHead INT8, 输入 [1,512,180,180], 输出 8 个 tensor |
| transfusion_head_fp16_v2_sm86.engine | 4.4 MB | TransFusionHead FP16 (argsort→topk 修复) |

## Phase 6 NDS 评估结果（单卡 RTX 3090，2026-03-29）

| 版本 | NDS | mAP | 速度 | 说明 |
|------|-----|-----|------|------|
| FP32 baseline | 0.7069 | 0.6728 | — | 原始模型 |
| Phase 6 (7/8 量化) | 0.7040 | 0.6654 | 5.6 fps | LiDAR FP32, 其余全 INT8 |
| PTQ 7/8 +vt KL (参考) | 0.7033 | 0.6657 | — | REPORT.md, PyTorch 仿真 |
| PTQ 8/8 KL+Log2 (参考) | 0.6875 | 0.6429 | — | REPORT.md, PyTorch 仿真 |

> **当前状态**：
> - SwinT / Neck / Depthnet / Fuser+Decoder / TransFusionHead 全部 TRT INT8。
> - **LiDAR backbone 仍为 PyTorch spconv FP32**，Log2 量化未集成到部署 pipeline。
> - Phase 6 结果等价于 PTQ 7/8（skip lidar），NDS 0.7040 ≈ 0.7033 吻合。
> - Phase 7 standalone 脚本已迁移到 spconv23_deploy 环境（见下方第 6 节）。

---

## 6. Phase 7 — Standalone 推理（spconv23_deploy 环境）

> 环境：`conda activate /media/yellowstone/data2/CYL/spconv23_deploy`
> Python 3.9 + PyTorch 2.0 + spconv 2.3.8 + TRT 10.15
> 不依赖 bevfusion_mqbench 环境的 cpython-38 扩展

### 环境准备（首次）

```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 编译 CUDA 扩展（bev_pool_ext, voxel_layer, iou3d_cuda, roiaware_pool3d_ext）
python tools/build_cuda_ext.py
```

### 单样本冒烟测试

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --test-single \
    2>&1 | tee logs/standalone_smoke.log
```

### 完整 NDS 评估（LiDAR FP16，其余 INT8）

```bash
mkdir -p logs

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    2>&1 | tee logs/standalone_eval_fp16.log
# 预期 NDS ≈ 0.7040（与 Phase 6 LiDAR FP32 一致，spconv 2.3 已验证 cosine=0.999994）
```

### 完整 NDS 评估（LiDAR Log2 INT8 量化）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    2>&1 | tee logs/standalone_eval_int8.log
# 预期 NDS ≈ 0.6875（与 PTQ 8/8 KL+Log2 一致）
```

---

## Section 7: Phase 8 — TV Backbone（去 PyTorch LiDAR）

环境：`spconv23_deploy`

### 完整 NDS 评估（TV backbone FP16，LiDAR 无 PyTorch）

> **2026-04-03 更新**：BN 已 fuse 进 `implicit_gemm`（bias + ReLU），残差 add 已改为 cuBLAS `axpy` 在 GPU 上完成。之前的 ~0.6 fps 瓶颈（CPU numpy roundtrip）已修复，预期帧率应接近 PyTorch 路径的 ~5 fps。若 `logs/standalone_eval_tv_fp16.log` 为旧代码运行结果，其帧率不具参考意义，但 NDS 精度仍有效。
>
> **2026-04-05 修复**：根因为 `TVAllocator` 及输入 `tv.Tensor` 的底层 `torch.Tensor` 被 Python GC 提前释放，导致后续 layer 读取悬空 CUDA 指针（NaN / garbage）。修复内容：
> - `TVSparseConvTensor` 增加 `_allocators` 列表，钉住每一层的 `TVAllocator`；
> - `TVSparseEncoder.forward()` 增加 `feature_ref` / `coors_ref` 参数，用于外部 caller 钉住输入 tensor；
> - `trt_infer_standalone.py` TV mode caller 已同步传入 `feature_ref`。
> 经 `smoke_isolate_conv0.py` 验证，PT vs TV diff = **0.000000**；end-to-end dense max diff = **0.053**（FP16 rounding）。可直接用下方命令重跑 full eval。

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate /media/yellowstone/data2/CYL/spconv23_deploy

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --no-torch-lidar \
    2>&1 | tee logs/standalone_eval_tv_fp16_v2.log
# 预期 NDS ≈ 0.7039（与 Phase 7 FP16 一致）
# 预期帧率 ≈ 5 fps（与 PyTorch 路径相当）
```

### 冒烟测试（单样本）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --no-torch-lidar --test-single \
    2>&1 | tee logs/standalone_tv_smoke.log
```

### Phase 7 收尾：PyTorch spconv INT8 NDS 评估

> 已通过 PT 路径验证 INT8 NDS ≈ 0.6875。TV INT8 路径的评估见下方 Phase 9。

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate /media/yellowstone/data2/CYL/spconv23_deploy

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    2>&1 | tee logs/standalone_eval_int8.log
# 预期 NDS ≈ 0.6875（与 PTQ 8/8 KL+Log2 一致）
```

---

## Section 8: Phase 9 — TV Backbone INT8 Log2（2026-04-05 完成）

> TV backbone INT8 Log2 已集成到 `--no-torch-lidar` 路径。
> 关键修复：`_basic_block` 中 `identity = x.features.clone()`，防止 in-place Log2 量化破坏残差连接。
> 端到端 dense max diff = **2.89**（INT8 rounding 差异，可接受）。
> 冒烟测试已通过。

### 完整 NDS 评估（TV backbone INT8，全零 PyTorch LiDAR）

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate /media/yellowstone/data2/CYL/spconv23_deploy

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    --no-torch-lidar \
    2>&1 | tee logs/standalone_eval_tv_int8.log
# 实际结果（2026-04-05）：NDS = 0.6893，mAP = 0.6474
```

### 冒烟测试（单样本）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    --no-torch-lidar --test-single \
    2>&1 | tee logs/standalone_tv_int8_smoke.log
```

---

## 注意事项

- GPU 映射：`GPU_IDS=(0 1 3 4)` 对应 4 张 RTX 3090，GPU 2 是 A100 跳过
- TRT 引擎和 GPU 架构绑定，SM 8.0 (A100) 引擎不能在 SM 8.6 (RTX 3090) 上运行
- 4 卡并行时每卡独立加载模型 + 引擎，显存约 14GB/卡
- 合并评估 `merge_eval.py` 不需要 GPU，只做 NDS 计算
- `*_sm86.engine` 后缀表示在 RTX 3090 上构建的引擎

---

## Section 10: TRT 10.3 Orin 兼容性预验证（x86 本地）

> Jetson Orin 预装 TensorRT 10.3。RTX 3090 上构建的 TRT 10.15 + SM 8.6 引擎**无法**直接在 Orin 上运行。需在 x86 上提前用 TRT 10.3 验证 ONNX 可构建性。
>
> 为了不碰服务器公共资源，TRT 10.3 以 `pip --target` 方式安装到项目本地目录：`trt_10.3_env/`。

### 本地 TRT 10.3 环境使用方式

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 设置 Python/库路径（仅当前 shell 生效）
export TRT103=/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/trt_10.3_env
export PYTHONPATH=$TRT103:$PYTHONPATH
export LD_LIBRARY_PATH=$TRT103/tensorrt_libs:$LD_LIBRARY_PATH

# 验证版本
conda run -n bevfusion_mqbench python -c "import tensorrt as trt; print(trt.__version__)"
# 期望输出：10.3.0
```

### 用 TRT 10.3 构建引擎

```bash
export TRT103=/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/trt_10.3_env
export PYTHONPATH=$TRT103:$PYTHONPATH
export LD_LIBRARY_PATH=$TRT103/tensorrt_libs:$LD_LIBRARY_PATH

conda run -n bevfusion_mqbench python tools/build_trt_engine.py \
    --onnx artifacts/camera_neck_int8.onnx \
    --engine artifacts/camera_neck_int8_trt103.engine \
    --int8 --fp16 --workspace 4096 \
    --timing-cache artifacts/trt103_timing.cache
```

### 预验证结果（2026-04-13）

| ONNX | TRT 10.3 Build | 说明 |
|------|----------------|------|
| `swin_int8.onnx` | 32.36 MB | 成功构建并反序列化 |
| `camera_neck_int8.onnx` | 2.18 MB | 成功构建并反序列化 |
| `vtransform_depthnet_int8.onnx` | 2.14 MB | 成功构建并反序列化 |
| `fuser_decoder_int8.onnx` | 6.93 MB | 成功构建并反序列化 |
| `transfusion_head_int8.onnx` | 3.84 MB | 成功构建并反序列化（8 个输出） |
| `lidar_backbone_*.onnx` | 失败 | ONNX 含循环/重复输出名；TRT 10.15 下也失败（missing SparseConvolution plugin）。需重新导出或加载 spconv plugin，**非 TRT 10.3 兼容性问题** |

### 关键观察

1. **UINT8 zero_point 警告**：所有 INT8 ONNX 都会报 `QuantizeLinear/DequantizeLinear with UINT8 zero_point` 警告。TRT 10.3 自动 fallback 到 INT8，**不影响构建**。
2. **SM 绑定**：上述 engine 是在 SM 8.6 (RTX 3090) 上构建的，**仍不能直接在 Orin SM 8.7 上运行**。此步骤只为验证“ONNX → engine”在 TRT 10.3 的 parser 层无兼容性问题。真正上 Orin 需在 Orin 本机或 cross-build 重新构建。
3. **Hardware Compatibility**：TRT 10.3 Python 包中 `HARDWARE_COMPATIBLE_AMAX` 不可用；跨 SM 8.6→8.7 的引擎迁移仍需在 Orin 上重新 build。
```
