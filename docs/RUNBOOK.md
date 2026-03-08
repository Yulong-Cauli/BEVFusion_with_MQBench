# 可复现运行手册

> 本文档包含所有评测结果的**完整运行命令**，可直接复制粘贴到 PowerShell（本地）或 Bash（服务器）执行。
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
    --eval bbox 2>&1 | Tee-Object -FilePath "logs/results_fp32.log"
```

**预期输出**：NDS ≈ 0.5800, mAP ≈ 0.5742

---

## 2. PTQ 全模型 INT8 评估（8/8 模块量化）

**三路径量化**：校准（32 batch）→ 量化推理（81 样本）→ NDS/mAP 评估 → 保存 checkpoint

### 2.1 全模型 INT8（8/8 模块，NDS ≈ 0.4276）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 32 2>&1 | Tee-Object -FilePath "logs/results_ptq_8of8.log"
```

### 2.2 推荐配置：6/8 模块 INT8（跳过精度敏感模块，NDS ≈ 0.5799）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 32 `
    --skip-modules camera/vtransform lidar/backbone 2>&1 | Tee-Object -FilePath "logs/results_ptq_6of8.log"
```

### 2.3 消融：7/8 模块（仅加 vtransform，NDS ≈ 0.5485）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 32 `
    --skip-modules lidar/backbone 2>&1 | Tee-Object -FilePath "logs/results_ptq_7of8_vtrans.log"
```

### 2.4 消融：7/8 模块（仅加 lidar/backbone，NDS ≈ 0.4803）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 32 `
    --skip-modules camera/vtransform 2>&1 | Tee-Object -FilePath "logs/results_ptq_7of8_lidar.log"
```

### 2.5 诊断模式：逐模块 INT8 余弦相似度

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 32 `
    --diagnose --diagnose-samples 5 2>&1 | Tee-Object -FilePath "logs/results_ptq_diagnose.log"
```

> ⚠️ PTQ checkpoint 含 FakeQuant 结构，不能用 `tools/test.py` 直接评估。

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
    --num-iters 30 2>&1 | Tee-Object -FilePath "logs/results_benchmark.log"
```

**预期输出**：FP32 ~389ms, PTQ ~408ms（FakeQuant 仿真开销，非真实 INT8 加速）

---

## 4. 仅模型大小报告（无需数据集推理）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_benchmark.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --checkpoint pretrained/bevfusion-det.pth `
    --size-only 2>&1 | Tee-Object -FilePath "logs/results_size.log"
```

---

## 5. TRT Hybrid 全模块端到端 NDS 评估（4 模块替换）

**数据来源**：RESULTS_LOG「TRT Hybrid 全模块端到端 NDS 评估」

> 前提：已安装 TensorRT（`pip install tensorrt`）
>
> 此脚本将 camera/neck、fuser、decoder/backbone、decoder/neck 四个模块全部导出为 TRT 引擎并替换。

### FP32 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    pretrained/bevfusion-det.pth `
    --precision fp32 2>&1 | Tee-Object -FilePath "logs/results_trt_all_fp32.log"
```

**预期输出**：NDS ≈ 0.5800, mAP ≈ 0.5744, 4 模块引擎 42.6 MB

### FP16 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    pretrained/bevfusion-det.pth `
    --precision fp16 2>&1 | Tee-Object -FilePath "logs/results_trt_all_fp16.log"
```

**预期输出**：NDS ≈ 0.5795, mAP ≈ 0.5743, 4 模块引擎 13.5 MB

### INT8 精度（需校准）

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    pretrained/bevfusion-det.pth `
    --precision int8 `
    --calib-samples 50 2>&1 | Tee-Object -FilePath "logs/results_trt_all_int8.log"
```

**预期输出**：NDS ≈ 0.5723, mAP ≈ 0.5652, 4 模块引擎 7.2 MB

---

## 6. ResNet-50 Backbone — 本地评估（mini 数据集）

### FP32 基准

```powershell
$env:PYTHONUTF8="1"
python tools/test.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml `
    runs/epoch_6.pth `
    --eval bbox 2>&1 | Tee-Object -FilePath "logs/results_resnet50_fp32.log"
```

**预期输出**：NDS ≈ 0.3982, mAP ≈ 0.4135（mini 数据集，81 样本，不可与完整数据集结果直接对比）

### PTQ 5/6 模块量化（camera/backbone 可量化！）

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml `
    --load_from runs/epoch_6.pth `
    --run-dir runs/resnet50_ptq `
    --calib-batches 64 2>&1 | Tee-Object -FilePath "logs/results_resnet50_ptq.log"
```

**预期输出**：NDS ≈ 0.4079, mAP ≈ 0.4189（mini 数据集）

> ✅ 与 SwinT 版本不同，ResNet-50 的 camera/backbone 是纯 CNN，torch.fx 可完整追踪，量化覆盖率 5/6（仅 heads/object 因 Proxy 迭代问题跳过）。

### TRT Hybrid 全模块 NDS 评估（5 模块替换，含 camera/backbone）

> 此脚本将 camera/backbone、camera/neck、fuser、decoder/backbone、decoder/neck 五个模块全部导出为 TRT 引擎。
> 自动检测 ResNet backbone 并启用 5 模块模式。

#### FP32 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml `
    runs/epoch_6.pth `
    --precision fp32 2>&1 | Tee-Object -FilePath "logs/results_resnet50_trt_fp32.log"
```

**预期输出**：NDS ≈ 0.4030, mAP ≈ 0.4172, 5 模块引擎 160.7 MB

#### FP16 精度

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml `
    runs/epoch_6.pth `
    --precision fp16 --skip-baseline 2>&1 | Tee-Object -FilePath "logs/results_resnet50_trt_fp16.log"
```

**预期输出**：NDS ≈ 0.3981, mAP ≈ 0.4136, 5 模块引擎 59.9 MB

#### INT8 精度（需校准）

```powershell
$env:PYTHONUTF8="1"
python tools/trt_eval_hybrid_all.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml `
    runs/epoch_6.pth `
    --precision int8 --calib-samples 50 --skip-baseline 2>&1 | Tee-Object -FilePath "logs/results_resnet50_trt_int8.log"
```

**预期输出**：NDS ≈ 0.4078, mAP ≈ 0.4187, 5 模块引擎 31.4 MB

---

## 7. ResNet-50 Backbone — 服务器评估（完整 nuScenes trainval 数据集）

> **服务器环境**：详见[附录：服务器环境](#附录服务器环境)
> **数据集**：`data/nuscenes/`（v1.0-trainval，6019 val 样本）
> **权重路径**：`runs/resnet50_fulldata/epoch_6.pth`

**GPU 选择策略**：
- **不设 `CUDA_DEVICE_ORDER`/`CUDA_VISIBLE_DEVICES`**：CUDA 默认 FASTEST_FIRST 排序，自动使用 A100（推荐用于评估）
- **指定 3090**：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`
- `test.py` 当前仅支持**单卡评估**（`distributed=False` 硬编码），多卡不可用

### 前置设置

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
```

### FP32 基准评估（默认使用 A100）

```bash
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    runs/resnet50_fulldata/epoch_6.pth \
    --eval bbox 2>&1 | tee logs/results_resnet50_fulldata_fp32.log
```

### PTQ 5/6 模块量化 + 评估

```bash
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    --load_from runs/resnet50_fulldata/epoch_6.pth \
    --run-dir runs/resnet50_fulldata_ptq \
    --calib-batches 128 2>&1 | tee logs/results_resnet50_fulldata_ptq.log
```

> ⚠️ **注意路径**：服务器上的权重路径是 `runs/resnet50_fulldata/epoch_6.pth`（不是 `runs/epoch_6.pth`）。

---

## 8. SwinT Backbone — 服务器评估（完整 nuScenes trainval 数据集）

> 如果服务器上有 SwinT 预训练权重（`pretrained/bevfusion-det.pth`），可以做对比。
> 同样默认使用 A100 推理（不设 GPU 环境变量即可）。

### FP32 基准评估

```bash
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --eval bbox 2>&1 | tee logs/results_swint_fulldata_fp32.log
```

### PTQ 8/8 全模型量化 + 评估

```bash
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/swint_ptq \
    --calib-batches 128 2>&1 | tee logs/results_swint_fulldata_ptq_8of8.log
```

### PTQ 6/8 推荐配置（跳过敏感模块）

```bash
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/swint_ptq_6of8 \
    --skip-modules camera/vtransform lidar/backbone \
    --calib-batches 128 2>&1 | tee logs/results_swint_fulldata_ptq_6of8.log
```

---

## 9. 服务器一键全量评估脚本（REPORT §9 全部实验）

> **环境确认**：服务器 CUDA 12.2，TensorRT 10.15.1.29（与本地一致，脚本无需修改）
> **数据集**：nuScenes v1.0-trainval 验证集（6019 帧）
> **预计耗时**：10-15 小时（建议 nohup 后台运行）
> **前提**：`pretrained/bevfusion-det.pth`（SwinT 权重）和 `runs/resnet50_fulldata/epoch_6.pth`（ResNet-50 权重）均存在

### 9.1 确认权重文件

```bash
ls -la pretrained/bevfusion-det.pth runs/resnet50_fulldata/epoch_6.pth
```

如果 `pretrained/bevfusion-det.pth` 不存在，跳过 SwinT 部分（实验 1-5），只跑 ResNet-50 部分（实验 6-10）。

### 9.2 实验清单（排除 PTQ — 服务器未安装 MQBench）

> ⚠️ PTQ 实验（quant_ptq_minmax.py）依赖 MQBench，服务器未安装，**跳过**。PTQ 结果使用本地 mini 数据集的已有结论（精度无损）即可。

共 8 个实验，分配到 5 张 GPU 并行执行：

| 批次 | GPU# (PCI_BUS_ID) | 实验 | tmux 窗口 |
|------|----------|------|-----------|
| 第一批 | GPU#2 (A100) | SwinT FP32 基线 | tmux-gpu2 |
| 第一批 | GPU#1 | ResNet-50 FP32 基线 | tmux-gpu1 |
| 第一批 | GPU#3 | SwinT TRT FP32 | tmux-gpu3 |
| 第一批 | GPU#4 | ResNet-50 TRT FP32 | tmux-gpu4 |
| 第二批 | GPU#0 | SwinT TRT FP16 | tmux-gpu0 |
| 第二批 | GPU#1 | ResNet-50 TRT FP16 | tmux-gpu1 |
| 第二批 | GPU#3 | SwinT TRT INT8 | tmux-gpu3 |
| 第二批 | GPU#4 | ResNet-50 TRT INT8 | tmux-gpu4 |

> 注意：SwinT FP32 基线改用 A100（GPU#2）以避免 bev_pool CUDA 异常（3090 GPU#0 上首次运行出现 illegal memory access）。GPU#2 不能训练但可以推理。
>
> TRT 实验的引擎是针对目标 GPU 编译的。SwinT 和 ResNet-50 的 TRT 引擎各自独立，分在不同 GPU 上也各自独立构建，互不冲突。每个实验有独立的输出目录（`--out-dir` 参数区分）。

### 9.3 操作步骤

#### 前置准备

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 确认权重存在
ls -la pretrained/bevfusion-det.pth runs/resnet50_fulldata/epoch_6.pth

# 确认 TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"

# 确认 GPU 编号 (PCI_BUS_ID 排序)
CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,name --format=csv,noheader
# 预期: 0=3090, 1=3090, 2=A100, 3=3090, 4=3090
```

> ⚠️ **重要：每个 tmux 窗口都必须执行以下环境初始化**（缺少任一步都可能导致 `libtorch_cuda_cu.so` 或 TRT builder nullptr 错误）：
>
> ```bash
> conda activate bevfusion_mqbench
> cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
> export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
> export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
> ```

#### 第一批：4 个 tmux 窗口

```bash
# --- 创建 tmux 窗口 ---
tmux new-session -d -s gpu2
tmux new-session -d -s gpu1
tmux new-session -d -s gpu3
tmux new-session -d -s gpu4
```

然后分别 attach 到每个 tmux 窗口执行对应命令：

**窗口 gpu2 — SwinT FP32 基线（A100）**：
```bash
tmux attach -t gpu2
# 在 tmux 内执行：
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --eval bbox 2>&1 | tee logs/results_server_swint_fp32.log
# Ctrl+B D 断开
```

**窗口 gpu1 — ResNet-50 FP32 基线**：
```bash
tmux attach -t gpu1
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    runs/resnet50_fulldata/epoch_6.pth \
    --eval bbox 2>&1 | tee logs/results_server_resnet50_fp32.log
# Ctrl+B D 断开
```

**窗口 gpu3 — SwinT TRT FP32**：
```bash
tmux attach -t gpu3
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --precision fp32 \
    --out-dir runs/trt_server_swint \
    2>&1 | tee logs/results_server_swint_trt_fp32.log
# Ctrl+B D 断开
```

**窗口 gpu4 — ResNet-50 TRT FP32**：
```bash
tmux attach -t gpu4
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    runs/resnet50_fulldata/epoch_6.pth \
    --precision fp32 \
    --out-dir runs/trt_server_resnet50 \
    2>&1 | tee logs/results_server_resnet50_trt_fp32.log
# Ctrl+B D 断开
```

#### 第二批：第一批全部完成后，复用同样的 4 个 tmux 窗口

> ⚠️ **即使复用 tmux 窗口，也必须重新执行 `conda activate` 和 `export LD_LIBRARY_PATH`**，否则会出现 `libtorch_cuda_cu.so` 或 TRT builder nullptr 错误。

**窗口 gpu0 — SwinT TRT FP16**：
```bash
tmux attach -t gpu0
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --precision fp16 --skip-baseline \
    --out-dir runs/trt_server_swint \
    2>&1 | tee logs/results_server_swint_trt_fp16.log
```

**窗口 gpu1 — ResNet-50 TRT FP16**：
```bash
tmux attach -t gpu1
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    runs/resnet50_fulldata/epoch_6.pth \
    --precision fp16 --skip-baseline \
    --out-dir runs/trt_server_resnet50 \
    2>&1 | tee logs/results_server_resnet50_trt_fp16.log
```

**窗口 gpu3 — SwinT TRT INT8**：
```bash
tmux attach -t gpu3
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth \
    --precision int8 --calib-samples 50 --skip-baseline \
    --out-dir runs/trt_server_swint \
    2>&1 | tee logs/results_server_swint_trt_int8.log
```

**窗口 gpu4 — ResNet-50 TRT INT8**：
```bash
tmux attach -t gpu4
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null):$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml \
    runs/resnet50_fulldata/epoch_6.pth \
    --precision int8 --calib-samples 50 --skip-baseline \
    --out-dir runs/trt_server_resnet50 \
    2>&1 | tee logs/results_server_resnet50_trt_int8.log
```

#### 检查进度

```bash
# 列出所有 tmux 窗口
tmux ls

# 查看某个窗口（不中断进程）
tmux attach -t gpu0
# Ctrl+B D 断开（不会杀进程）

# 查看已完成的日志
ls -lh logs/results_server_*.log
```

### 9.4 传回结果

全部完成后打包：

```bash
tar czf server_results.tar.gz logs/results_server_*.log
# 传回本地（scp / rsync / 手动拷贝）
```

### 9.5 各实验预计耗时（单张 3090，6019 帧）

| 实验 | 脚本 | 预计耗时 |
|------|------|---------|
| FP32 基线（test.py） | ~2.5 小时（0.7 task/s × 6019 帧） |
| TRT FP32 导出+评估 | ~1-2 小时（引擎构建 + 评估） |
| TRT FP16 导出+评估 | ~1-2 小时 |
| TRT INT8 导出+评估 | ~1-2 小时（含校准） |

4 卡并行后（第一批含 A100）：
- **第一批**（4 个实验并行）：~2.5 小时
- **第二批**（4 个实验并行）：~2 小时
- **总计**：~4-5 小时（vs 单卡串行 ~12-15 小时）

---

## 日志文件说明

运行完成后，所有结果日志保存在 `logs/` 目录：

| 文件 | 内容 |
|------|------|
| `logs/results_fp32.log` | FP32 基准 NDS/mAP |
| `logs/results_ptq_eval.log` | PTQ 4/6 校准 + 完整 NDS 评估 |
| `logs/results_benchmark.log` | FP32 vs PTQ 延迟 / 大小对比 |
| `logs/results_size.log` | 模型参数量 / 内存大小 |
| `logs/results_trt_all_fp32.log` | TRT 全模块 Hybrid FP32 端到端 NDS |
| `logs/results_trt_all_fp16.log` | TRT 全模块 Hybrid FP16 端到端 NDS |
| `logs/results_trt_all_int8.log` | TRT 全模块 Hybrid INT8 端到端 NDS |
| `logs/results_resnet50_fp32.log` | ResNet-50 FP32 本地 mini 评估 |
| `logs/results_resnet50_ptq.log` | ResNet-50 PTQ 5/6 本地 mini 评估 |
| `logs/results_resnet50_trt_fp32.log` | ResNet-50 TRT FP32 5 模块 Hybrid 评估（本地 mini） |
| `logs/results_resnet50_trt_fp16.log` | ResNet-50 TRT FP16 5 模块 Hybrid 评估（本地 mini） |
| `logs/results_resnet50_trt_int8.log` | ResNet-50 TRT INT8 5 模块 Hybrid 评估（本地 mini） |
| `logs/results_resnet50_fulldata_fp32.log` | ResNet-50 FP32 服务器完整数据集评估（默认 A100） |
| `logs/results_resnet50_fulldata_ptq.log` | ResNet-50 PTQ 5/6 服务器完整数据集评估 |
| `logs/results_swint_fulldata_fp32.log` | SwinT FP32 服务器完整数据集评估 |
| `logs/results_swint_fulldata_ptq.log` | SwinT PTQ 4/6 服务器完整数据集评估 |
| `logs/results_server_swint_trt_fp32.log` | SwinT TRT FP32 4 模块服务器评估 |
| `logs/results_server_swint_trt_fp16.log` | SwinT TRT FP16 4 模块服务器评估 |
| `logs/results_server_swint_trt_int8.log` | SwinT TRT INT8 4 模块服务器评估 |
| `logs/results_server_resnet50_trt_fp32.log` | ResNet-50 TRT FP32 5 模块服务器评估 |
| `logs/results_server_resnet50_trt_fp16.log` | ResNet-50 TRT FP16 5 模块服务器评估 |
| `logs/results_server_resnet50_trt_int8.log` | ResNet-50 TRT INT8 5 模块服务器评估 |

---

## 附录：服务器环境

| 项目 | 内容 |
|------|------|
| 工作目录 | `/media/yellowstone/data2/CYL/BEVFusion_with_MQBench` |
| Conda 环境 | `bevfusion_mqbench` |
| CUDA Driver | 12.2（向下兼容 cu113） |

### GPU 配置

| GPU# | 型号 | 显存 | 备注 |
|------|------|------|------|
| 0 | RTX 3090 | 24 GB | — |
| 1 | RTX 3090 | 24 GB | — |
| 2 | A100-SXM4 | 80 GB | 可推理，不可训练 |
| 3 | RTX 3090 | 24 GB | — |
| 4 | RTX 3090 | 24 GB | — |

### 已知问题
- `setuptools>=65` 与 PyTorch 1.10 冲突：`pip install "setuptools<65"` 解决
- Windows 上传的 .sh 脚本含 `\r`：`sed -i 's/\r//' script.sh` 修复
- 长时间任务必须在 tmux 中运行
