# BEVFusion 全命令速查表

> 版本验证用命令合集。先跑通这些命令确认当前代码版本一切正常，再进入后续优化。
> 工作目录：`cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench`

---

## 环境说明

| 环境 | 用途 | 激活命令 |
|------|------|---------|
| `bevfusion_mqbench` | PTQ 校准、FP32 基线评估 | `conda activate bevfusion_mqbench` |
| `spconv23_deploy` | TRT standalone 部署评估 | `conda activate /media/yellowstone/data2/CYL/spconv23_deploy` |

公共配置路径：
```
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
```

---

## 1. 算法侧 PTQ 评估命令（`bevfusion_mqbench`）

> 运行后 checkpoint 默认保存在 `run_dir/ptq_minmax_model.pth`，需要手动拷贝到 `pretrained/` 并改名才能在 TRT 侧引用。

### 1.1 FP32 基线评估
```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --eval bbox \
    2>&1 | tee logs/fp32_baseline.log
```
**预期**: NDS 0.7069, mAP 0.6728

### 1.2 PTQ 6/8 — 跳过 vtransform + lidar（旧最优Baseline）
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules camera/vtransform lidar/backbone \
    --calib-batches 128 \
    2>&1 | tee logs/ptq_6_8_minmax.log
```

### 1.3 PTQ 7/8 MinMax — 只跳过 lidar（默认 EMAMinMax）
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --calib-batches 128 \
    2>&1 | tee logs/ptq_7_8_minmax.log
```

### 1.4 PTQ 7/8 KL — 只跳过 lidar（vtransform 用 KL，推荐高精度配置）
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --vtransform-observer kl_divergence \
    --calib-batches 128 \
    2>&1 | tee logs/ptq_7_8_kl.log
```
**预期**: NDS ≈ 0.7033, mAP ≈ 0.6657

### 1.5 PTQ 8/8 W8A16 — lidar 只权重量化，激活保持 FP（控制实验）
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --no-lidar-act-quant \
    --calib-batches 128 \
    2>&1 | tee logs/ptq_8_8_w8a16.log
```
**说明**: 跑完后若想在 TRT 侧复现，需将该 checkpoint 重命名为 `pretrained/ptq_w8a16.pth`，并在 TRT 命令中通过 `--ptq-ckpt` 指定。

### 1.6 PTQ 8/8 Log2 — 最新最优全量化（vtransform KL + lidar Log2）
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --act-observer log2 \
    --calib-batches 128 \
    2>&1 | tee logs/ptq_8_8_log2.log
```
**预期**: NDS ≈ 0.6875, mAP ≈ 0.6429

---

## 2. TRT 引擎构建/导出命令

> 当前仅存 `tools/export_utils/build_engine.py`。以下命令**假设 `.onnx` 文件已存在于根目录**；若需重新导出 ONNX，目前无现成脚本，需从 git 历史或备份中恢复。

### 2.1 构建所有常用引擎（SM 8.6, RTX 3090）
```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
mkdir -p logs

# SwinT INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx --engine swin_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_swin.log

# DepthNet INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx vtransform_depthnet_int8.onnx --engine vtransform_depthnet_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_depthnet.log

# Camera Neck INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx camera_neck_int8.onnx --engine camera_neck_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_neck_int8.log

# Fuser+Decoder INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx fuser_decoder_int8.onnx --engine fuser_decoder_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_fuser_int8.log

# TransFusionHead INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/export_utils/build_engine.py \
    --onnx transfusion_head_int8.onnx --engine transfusion_head_int8_sm86.engine --int8 --fp16 \
    2>&1 | tee logs/build_head_int8.log
```

---

## 3. 部署侧 TRT 评估命令（`spconv23_deploy`）

> 以下命令使用 `trt_infer_standalone.py`（standalone 零依赖部署路径）。
> **PTQ 与 TRT 的对应关系**：
> - PTQ 6/8、PTQ 7/8 MinMax、PTQ 7/8 KL 都“跳过了 lidar 量化”，TRT 侧对应 **LiDAR FP32**（默认不加 `--lidar-quant`）。
> - PTQ 8/8 W8A16 在 TRT 侧没有独立的 TV 路径，需要用 **PyTorch backbone + W8A16 checkpoint**（即 `--lidar-quant int8 --ptq-ckpt <w8a16_ckpt>` 且不加 `--no-torch-lidar`）。
> - PTQ 8/8 Log2 在 TRT 侧对应 **TV backbone INT8 Log2**（`--no-torch-lidar`）。

### 3.1 TRT LiDAR FP32 — 对应 PTQ 6/8 与 7/8 系列（skip lidar）
```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    2>&1 | tee logs/trt_lidar_fp32.log
```
**预期**: NDS ≈ 0.7040（PyTorch spconv 路径）

### 3.2 TRT TV FP16 — 去 PyTorch LiDAR 基线（Phase 8）
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
    --no-torch-lidar \
    2>&1 | tee logs/trt_tv_fp16.log
```
**预期**: NDS ≈ 0.7039

### 3.3 TRT W8A16 — 对应 PTQ 8/8 W8A16（PyTorch backbone 路径加载 W8A16 checkpoint）
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
    --lidar-quant int8 \
    --ptq-ckpt pretrained/ptq_w8a16.pth \
    2>&1 | tee logs/trt_w8a16.log
```
**注意**: `pretrained/ptq_w8a16.pth` 需要由 PTQ 8/8 W8A16 评估命令生成后手动复制。

### 3.4 TRT TV INT8 Log2 — 对应 PTQ 8/8 Log2（最新最优部署，Phase 9A）
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
    --lidar-quant int8 \
    --ptq-ckpt pretrained/ptq_minmax_model.pth \
    --no-torch-lidar \
    2>&1 | tee logs/trt_tv_int8_log2.log
```
**预期**: NDS ≈ 0.6893, mAP ≈ 0.6474

---

## 4. 单样本冒烟测试（快速验证不跑完整 6019 帧）

如果你只想确认命令能跑通、不报错，给每个命令加上 `--test-single` 即可，例如：

```bash
# PTQ 8/8 Log2 冒烟测试
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --act-observer log2 \
    --calib-batches 128 \
    --no-eval --test-single    # <-- 注意：quant_ptq_minmax.py 本身没有 --test-single，这里仅示意
```

> **注意**：`quant_ptq_minmax.py` 本身不支持 `--test-single`。如果需要快速跑通 PTQ 流程，可以把 `--calib-batches` 调到 10 并加上 `--no-eval`，大概 5~10 分钟就能跑完校准和保存，仅用于验证脚本不报错。

对于 TRT standalone，可以直接用 `--test-single`：

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
    2>&1 | tee logs/trt_tv_int8_smoke.log
```

---

## 5. 快速验证 Checklist

建议按以下顺序跑一遍，确认版本稳定：

- [ ] **FP32 基线** → 确认 NDS 0.7069
- [ ] **PTQ 7/8 KL** → 确认 NDS ≈ 0.7033
- [ ] **PTQ 8/8 Log2** → 确认 NDS ≈ 0.6875
- [ ] **TRT TV FP16** → 确认 NDS ≈ 0.7039
- [ ] **TRT TV INT8 Log2** → 确认 NDS ≈ 0.6893

以上 5 条跑通，说明算法侧和部署侧的主链路都正常。其余命令（6/8、7/8 MinMax、W8A16）可按需补充验证。
