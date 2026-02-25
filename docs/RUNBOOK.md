# 可复现运行手册

> 本文档包含所有评测结果的**完整运行命令**，可直接复制粘贴到 PowerShell 执行。
> 每条命令对应 `docs/RESULTS_LOG.md` 中的一个数据来源。

---

## 前置设置（每次开新 PowerShell 窗口时执行一次）

```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
conda activate bevfusion
cd D:\Research\Replication\BEVFusion_with_MQBench
```

---

## 1. FP32 基准精度（NDS = 0.5801）

**数据来源**：RESULTS_LOG 中所有 FP32 基线数据

```powershell
$env:PYTHONUTF8="1"
python tools/test.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    pretrained/bevfusion-det.pth `
    --eval bbox 2>&1 | Tee-Object -FilePath "results_fp32.log"
```

**预期输出**：NDS ≈ 0.5800, mAP ≈ 0.5742

---

## 2. PTQ 4/6 完整 NDS 评估（NDS = 0.5810）

**数据来源**：RESULTS_LOG「PTQ（MinMax，4/6 模块量化）完整 NDS 评估」

此命令同时执行：校准（128 batch）→ 量化推理（81 样本）→ NDS/mAP 评估 → 保存 checkpoint

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 128 2>&1 | Tee-Object -FilePath "results_ptq_eval.log"
```

**预期输出**（日志末尾）：NDS ≈ 0.5810, mAP ≈ 0.5759

> ⚠️ 不要加 `--no-eval`，否则会跳过精度评估。

---

## 3. Benchmark 对比（FP32 vs PTQ 延迟 / 模型大小）

**数据来源**：RESULTS_LOG「速度」和「模型大小」章节

先跑一次 PTQ（上面的命令 2），拿到 checkpoint 路径后再跑 benchmark：

```powershell
$env:PYTHONUTF8="1"
# 自动找到最新的 PTQ checkpoint
$ptq_ckpt = (Get-ChildItem -Recurse -Filter "ptq_minmax_model.pth" | `
    Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "PTQ model: $ptq_ckpt"

python tools/quant_benchmark.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --checkpoint pretrained/bevfusion-det.pth `
    --quant-checkpoint $ptq_ckpt `
    --num-iters 30 2>&1 | Tee-Object -FilePath "results_benchmark.log"
```

**预期输出**：FP32 ~389ms, PTQ ~408ms（FakeQuant 仿真开销，非真实 INT8 加速）

---

## 4. ConvFuser TensorRT 导出 PoC（INT8 6.81x 加速）

**数据来源**：RESULTS_LOG「ConvFuser TensorRT 导出 Proof-of-Concept」

> 前提：已安装 TensorRT（`pip install tensorrt`）

```powershell
$env:PYTHONUTF8="1"
python tools/trt_export_fuser.py `
    --checkpoint pretrained/bevfusion-det.pth 2>&1 | Tee-Object -FilePath "results_trt_fuser.log"
```

**预期输出**：FP32/FP16/INT8 三种引擎的延迟、大小对比

---

## 5. ConvFuser TRT 精度验证（余弦相似度）

**数据来源**：RESULTS_LOG「精度验证（真实预训练权重）」

```powershell
$env:PYTHONUTF8="1"
python tools/trt_accuracy_test.py `
    --checkpoint pretrained/bevfusion-det.pth 2>&1 | Tee-Object -FilePath "results_trt_accuracy.log"
```

**预期输出**：FP32 cos≈1.0, FP16 cos≈0.999995, INT8 cos≈0.999674

---

## 6. 仅模型大小报告（无需数据集推理）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_benchmark.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --checkpoint pretrained/bevfusion-det.pth `
    --size-only 2>&1 | Tee-Object -FilePath "results_size.log"
```

---

## 7. TRT Hybrid 端到端 NDS 评估

**数据来源**：RESULTS_LOG「TensorRT Hybrid 端到端 NDS 评估」

> 前提：已安装 TensorRT（`pip install tensorrt`）

### FP32 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid.py `
    --precision fp32 `
    --calib-samples 128 2>&1 | Tee-Object -FilePath "results_trt_hybrid_fp32.log"
```

**预期输出**：NDS ≈ 0.5801, mAP ≈ 0.5746

### FP16 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid.py `
    --precision fp16 `
    --calib-samples 128 2>&1 | Tee-Object -FilePath "results_trt_hybrid_fp16.log"
```

**预期输出**：NDS ≈ 0.5799, mAP ≈ 0.5744

### INT8 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid.py `
    --precision int8 `
    --calib-samples 128 2>&1 | Tee-Object -FilePath "results_trt_hybrid_int8.log"
```

**预期输出**：NDS ≈ 0.5727, mAP ≈ 0.5616

### 调试模式（逐样本对比 PyTorch vs TRT）

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid.py `
    --precision int8 `
    --calib-samples 128 `
    --debug 2>&1 | Tee-Object -FilePath "results_trt_hybrid_debug.log"
```

---

## 日志文件说明

运行完成后，所有结果日志保存在项目根目录：

| 文件 | 内容 |
|------|------|
| `results_fp32.log` | FP32 基准 NDS/mAP |
| `results_ptq_eval.log` | PTQ 4/6 校准 + 完整 NDS 评估 |
| `results_benchmark.log` | FP32 vs PTQ 延迟 / 大小对比 |
| `results_trt_fuser.log` | TRT FP32/FP16/INT8 延迟 / 大小 |
| `results_trt_accuracy.log` | TRT 各精度余弦相似度 |
| `results_size.log` | 模型参数量 / 内存大小 |
| `results_trt_hybrid_fp32.log` | TRT Hybrid FP32 端到端 NDS |
| `results_trt_hybrid_fp16.log` | TRT Hybrid FP16 端到端 NDS |
| `results_trt_hybrid_int8.log` | TRT Hybrid INT8 端到端 NDS |
| `results_trt_hybrid_debug.log` | TRT Hybrid 调试模式输出 |
