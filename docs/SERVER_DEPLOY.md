# 服务器部署与运行手册

### 本地 PowerShell 环境初始化
```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
conda activate bevfusion
cd D:\Research\Replication\BEVFusion_with_MQBench
```

### 服务器 Bash 环境初始化
```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

---

## ⚠️ 重要修正：校准集来源（2026-03-13）

**问题**：之前校准数据从验证集采样（`cfg.data.val`），这是方法论错误，导致过拟合到测试分布。
**修正**：现在校准数据从**训练集**采样（`cfg.data.train`），并关闭数据增强（`test_mode=True`）。
**影响**：
- 已在 Round 5 中重新测试。
- 之前的 Round 1-4 结果（基于错误的校准集）需要标记为"仅参考"

---

## ⚠️ 服务器命令规范

1、 **永远不要用 `nohup ... &`。** 用户在 tmux 里直接运行，tmux 本身已经保证断线不丢进程。

我希望能看到 tmux 里面的输出。

✅ **正确写法**（输出直接显示在 tmux，同时写 log 文件）：
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=X python tools/xxx.py ... 2>&1 | tee logs/xxx.log
```

2、上传服务器的时候如果涉及多个文件，建议打包成一个压缩包上传，服务器上解压后再运行。避免多次上传和多次 SSH。

---

## Step 0：打包上传（本地 PowerShell）

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

# 打包所有更新的代码和文档（排除数据集/权重/编译产物）
git archive HEAD --format=tar.gz -o code_update.tar.gz `
    tools/quant_ptq_minmax.py `
    tools/quant_benchmark.py `
    tools/train.py `
    tools/trt_eval_hybrid_all.py `
    tools/make_ppt.py `
    tools/test.py `
    tools/scripts/ `
    mmdet3d/datasets/nuscenes_dataset.py `
    mmdet3d/datasets/pipelines/formating.py `
    mmdet3d/datasets/pipelines/transforms_3d.py `
    docs/ `
    README.md

# 一次 SCP 上传（输入一次密码）
scp code_update.tar.gz yellowstone@10.129.51.101:/tmp/

# SSH 解压（再输一次密码）
ssh yellowstone@10.129.51.101 `
    "cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench && tar xzf /tmp/code_update.tar.gz && rm /tmp/code_update.tar.gz && echo 'Upload OK'"
```

---

## Step 1：SSH 进服务器，确认环境

```bash
ssh yellowstone@10.129.51.101
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 确认 MQBench 已安装（PTQ 必须）
python -c "import mqbench; print('MQBench OK:', mqbench.__version__)" \
    || pip install mqbench==0.0.6

# 确认数据集是 trainval 完整集（6019 帧）
python -c "
import pickle
d = pickle.load(open('data/nuscenes/nuscenes_infos_val.pkl', 'rb'))
print(f'Val frames: {len(d[\"infos\"])}')
# 应输出: Val frames: 6019
"

# 确认权重
ls -la pretrained/bevfusion-det.pth

# 确认 4×3090 GPU 编号（PCI_BUS_ID 排序）
CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 预期: 0=3090, 1=3090, 2=A100, 3=3090, 4=3090

# 修复 Windows 换行符（如有）
sed -i 's/\r//' tools/scripts/*.sh

# 创建 logs 目录
mkdir -p logs
```

---

## Step 2：创建 4 个 tmux 窗口

```bash
tmux new-session -d -s gpu0
tmux new-session -d -s gpu1
tmux new-session -d -s gpu3
tmux new-session -d -s gpu4
```

---

## Step 3：实验清单（4 张 3090 并行，预计 ~3h）

| GPU# | 实验 | 说明 | 预期 NDS |
|------|------|------|---------|
| GPU#0 | PTQ 8/8 全模型 INT8 | 全部 8 个模块量化 | ~0.6xx（待测） |
| GPU#1 | PTQ 6/8 推荐配置 | skip vtransform+lidar，近零精度损失 | ~0.70x（预测） |
| GPU#3 | PTQ 7/8 消融 +vtransform | 6/8 基础上加 vtransform | ~0.68x（预测） |
| GPU#4 | PTQ 7/8 消融 +lidar | 6/8 基础上加 lidar/backbone | ~0.65x（预测） |

> FP32 基线（NDS=0.7069）和 PTQ 4/6（NDS=0.7015）已测，无需重跑。

---

## Step 4：启动实验

**每个 tmux 窗口的环境初始化（必须每次执行）：**

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

---

### GPU#0 — PTQ 8/8 全模型 INT8

```bash
tmux attach -t gpu0
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8 \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_8of8.log
# Ctrl+B D 断开（不会杀进程）
```

---

### GPU#1 — PTQ 6/8 推荐配置（跳过 vtransform + lidar）

```bash
tmux attach -t gpu1
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_6of8 \
    --skip-modules camera/vtransform lidar/backbone \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_6of8.log
# Ctrl+B D 断开
```

---

### GPU#3 — PTQ 7/8 消融（+vtransform，skip lidar）

```bash
tmux attach -t gpu3
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_7of8_vtrans \
    --skip-modules lidar/backbone \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_7of8_vtrans.log
# Ctrl+B D 断开
```

---

### GPU#4 — PTQ 7/8 消融（+lidar，skip vtransform）

```bash
tmux attach -t gpu4
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_7of8_lidar \
    --skip-modules camera/vtransform \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_7of8_lidar.log
# Ctrl+B D 断开
```

---

## Step 5：监控进度

```bash
# 查看所有 tmux 窗口
tmux ls

# 查看某个窗口的实时输出（Ctrl+B D 断开，不杀进程）
tmux attach -t gpu0

# 快速检查各实验 NDS 结果（全部完成后）
grep -h "'object/nds'" logs/results_server_ptq_*.log | \
    sed "s/.*'object\/nds': \([0-9.]*\).*/\1/"

# 或者直接 grep NDS 行
grep -h "NDS:" logs/results_server_ptq_*.log
```

---

## Step 6：传回结果（在服务器执行）

```bash
# 打包所有新日志
tar czf server_ptq_results.tar.gz logs/results_server_ptq_*.log

# 查看大小确认
ls -lh server_ptq_results.tar.gz
```

**在本地 PowerShell 拉取：**

```powershell
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/server_ptq_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

# 解压查看
cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf server_ptq_results.tar.gz
```

---

## 附：已有的服务器结果（无需重测）

| 实验 | NDS | 来源 |
|------|-----|------|
| SwinT FP32 基线（6019帧） | 0.7069 | 已完成 |
| SwinT PTQ 4/6（6019帧） | 0.7015 | 已完成 |
| SwinT PTQ 8/8 MinMax（6019帧）| 0.5754 | 已完成 |
| SwinT TRT FP16（6019帧） | 0.7069 | 已完成 |
| SwinT TRT INT8（6019帧） | 0.7022 | 已完成 |

---

## Round 2：LWC + 激活 Observer 消融实验

> **目标**：验证 LWC（权重截断）和 MSEObserver（激活校准优化）对 lidar/backbone 量化精度的改善  
> **基线**：PTQ 8/8 MinMax only → NDS = 0.5754（最大精度瓶颈来自 lidar/backbone）  
> **新增功能**：`--lwc`（权重截断）+ `--act-observer {mse,ema_quantile}`（激活 Observer）

### Round 2 Step 0：打包上传

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench
git archive HEAD --format=tar.gz -o code_update.tar.gz -- tools/quant_ptq_minmax.py

scp code_update.tar.gz yellowstone@10.129.51.101:/tmp/

ssh yellowstone@10.129.51.101 `
    "cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench && tar xzf /tmp/code_update.tar.gz && rm /tmp/code_update.tar.gz && echo 'Upload OK'"
```

### Round 2 Step 1：实验清单（4 个实验并行）

| GPU# | 实验 | 命令关键参数 | 对比 |
|------|------|-------------|------|
| GPU#0 | LWC only | `--lwc` | 权重截断单独效果 |
| GPU#1 | MSE observer only | `--act-observer mse` | 激活 Observer 单独效果 |
| GPU#3 | LWC + MSE | `--lwc --act-observer mse` | 双管齐下（推荐） |
| GPU#4 | LWC + EMA Quantile | `--lwc --act-observer ema_quantile` | 百分位截断对比 |

> 基线 PTQ 8/8 MinMax（NDS=0.5754）已有，无需重测。

### Round 2 Step 2：启动实验

**每个 tmux 窗口先执行：**

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

---

#### GPU#0 — PTQ 8/8 + LWC only

```bash
tmux attach -t gpu0

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_lwc \
    --lwc \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_8of8_lwc.log
# Ctrl+B D 断开
```

#### GPU#1 — PTQ 8/8 + MSE observer

```bash
tmux attach -t gpu1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_mse \
    --act-observer mse \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_8of8_mse.log
# Ctrl+B D 断开
```

#### GPU#3 — PTQ 8/8 + LWC + MSE（推荐组合）

```bash
tmux attach -t gpu3

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_lwc_mse \
    --lwc --act-observer mse \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_8of8_lwc_mse.log
# Ctrl+B D 断开
```

#### GPU#4 — PTQ 8/8 + LWC + EMA Quantile

```bash
tmux attach -t gpu4

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_lwc_quantile \
    --lwc --act-observer ema_quantile \
    --calib-batches 128 2>&1 | tee logs/results_server_ptq_8of8_lwc_quantile.log
# Ctrl+B D 断开
```

### Round 2 Step 3：传回结果

```bash
# 在服务器执行
tar czf server_lwc_results.tar.gz \
    logs/results_server_ptq_8of8_lwc*.log \
    logs/results_server_ptq_8of8_mse.log
ls -lh server_lwc_results.tar.gz
```

**本地拉取：**

```powershell
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/server_lwc_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf server_lwc_results.tar.gz
```

---

## Round 3：校准集多样性实验（calib-shuffle + 512 batch）

> **目标**：验证 `shuffle=True + 512 batch` 的校准策略是否改善 8/8 和 6/8 的量化精度  
> **动机**：旧 `shuffle=False` 仅取前 128 帧，约等于前 3~4 个场景（~2% 场景覆盖率）；  
>   改为 `shuffle=True + 512 batch` 可均匀覆盖全部 ~150 个场景，激活值分布更具代表性  
> **对照**：用 6/8 作为控制变量——若 6/8 NDS 不变，说明改善完全来自 vtransform/lidar 校准

### Round 3 Step 0：打包上传

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench
git archive HEAD --format=tar.gz -o code_update.tar.gz -- tools/quant_ptq_minmax.py

scp code_update.tar.gz yellowstone@10.129.51.101:/tmp/

ssh yellowstone@10.129.51.101 `
    "cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench && tar xzf /tmp/code_update.tar.gz && rm /tmp/code_update.tar.gz && echo 'Upload OK'"
```

### Round 3 Step 1：实验清单（2 张 3090 并行）

| GPU# | 实验 | 关键参数 | 对照基线 |
|------|------|---------|---------|
| GPU#0 | PTQ 8/8 + shuffle + 512 | `--calib-batches 512 --calib-shuffle` | 旧 8/8 NDS=0.4562 |
| GPU#1 | PTQ 6/8 + shuffle + 512 | `--calib-batches 512 --calib-shuffle --skip-modules camera/vtransform lidar/backbone` | 旧 6/8 NDS=0.7010 |

> GPU#2 = A100（共享，不使用）；GPU#3/4 空闲可备用

### Round 3 Step 2：环境初始化（每个 tmux 窗口）

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

---

#### GPU#0 — PTQ 8/8 + shuffle + 512 batch

```bash
tmux attach -t gpu0
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_calib512s \
    --calib-batches 512 \
    --calib-shuffle \
    2>&1 | tee logs/results_server_ptq_8of8_calib512_shuffle.log
# Ctrl+B D 断开
```

---

#### GPU#1 — PTQ 6/8 + shuffle + 512 batch（对照组）

```bash
tmux attach -t gpu1
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_6of8_calib512s \
    --skip-modules camera/vtransform lidar/backbone \
    --calib-batches 512 \
    --calib-shuffle \
    2>&1 | tee logs/results_server_ptq_6of8_calib512_shuffle.log
# Ctrl+B D 断开
```

---

### Round 3 Step 3：传回结果

```bash
# 在服务器执行
tar czf server_calib512_results.tar.gz \
    logs/results_server_ptq_8of8_calib512_shuffle.log \
    logs/results_server_ptq_6of8_calib512_shuffle.log
ls -lh server_calib512_results.tar.gz
```

**本地拉取：**

```powershell
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/server_calib512_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf server_calib512_results.tar.gz
```

### Round 3 结果解读指引

- **若 6/8 不变（仍 ~0.7010）+ 8/8 有改善**：改善确实来自 vtransform/lidar 场景多样性
- **若 6/8 也变好**：说明 neck/fuser/decoder 的校准也受益（概率较低，因为这些模块原本已近无损）
- **若两者都不变**：校准样本量不是瓶颈，损失根本原因是 INT8 精度不足以表示深度概率分布


---

#### GPU#3 — PTQ 8/8 + shuffle + 512 batch + LWC（核心实验）

> LWC 用校准数据的 MSE 来优化截断比率，校准数据越多样，优化目标越准确。

```bash
tmux attach -t gpu3
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_lwc_calib512s \
    --calib-batches 512 \
    --calib-shuffle \
    --lwc \
    2>&1 | tee logs/results_server_ptq_8of8_lwc_calib512_shuffle.log
# Ctrl+B D 断开
```

---

#### GPU#4 — PTQ 8/8 + shuffle + 512 batch + LWC + MSEObserver（组合实验）

```bash
tmux attach -t gpu4
# ↑ 先执行上面的环境初始化 ↑

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --run-dir runs/server_ptq_8of8_lwc_mse_calib512s \
    --calib-batches 512 \
    --calib-shuffle \
    --lwc \
    --act-observer mse \
    2>&1 | tee logs/results_server_ptq_8of8_lwc_mse_calib512_shuffle.log
# Ctrl+B D 断开
```

### Round 3 Step 3：传回结果

```bash
# 在服务器执行（包含本 round 全部4个日志）
tar czf server_calib512_results.tar.gz \
    logs/results_server_ptq_8of8_calib512_shuffle.log \
    logs/results_server_ptq_6of8_calib512_shuffle.log \
    logs/results_server_ptq_8of8_lwc_calib512_shuffle.log \
    logs/results_server_ptq_8of8_lwc_mse_calib512_shuffle.log
ls -lh server_calib512_results.tar.gz
```

**本地拉取：**

```powershell
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/server_calib512_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf server_calib512_results.tar.gz
```

### Round 3 结果解读指引

| 对比 | 说明 |
|------|------|
| 8/8 shuffle512 vs 旧 8/8 (0.4562) | 校准多样性的独立效果 |
| 6/8 shuffle512 vs 旧 6/8 (0.7010) | 对照：6/8 本来已近无损，应基本不变 |
| 8/8 LWC+shuffle512 vs 旧 LWC (0.4545) | shuffle 对 LWC 优化目标的影响（核心） |
| 8/8 LWC+MSE+shuffle512 vs 旧 LWC+MSE (0.4526) | 组合策略+好校准集的上限 |

---

## Round 4：KL 散度 Observer 实验（2026-03-11）

### 背景

Round 3 确认 EMAMinMaxObserver 是 8/8 精度崩溃的主因之一：vtransform downsample 层 94~97% range waste。
实现了 `KLDivergenceObserver`（类似 TensorRT entropy calibrator），在 mini 数据集上取得显著改善：
- 8/8 KL(both): NDS 0.4285 → 0.5085（+13.9 pts）

### Step 0：上传代码到服务器

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

# 打包 KL Observer 分支代码
git archive exp/lss-kl-divergence-calibration --format=tar.gz -o code_update_kl.tar.gz `
    tools/quant_ptq_minmax.py `
    AGENTS.md `
    docs/RESULTS_LOG.md `
    docs/SERVER_DEPLOY.md

# 上传到服务器
scp code_update_kl.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/
```

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar xzf code_update_kl.tar.gz
```

### Step 1：运行 4 个 KL Observer 实验（4×GPU，各开一个 tmux pane）

```bash
# ===== tmux pane for GPU#0: 8/8 KL(both) — 核心实验 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --calib-batches 512 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_8of8_kl_both.log
```

```bash
# ===== tmux pane for GPU#1: 7/8 +vtransform KL =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --calib-batches 512 --calib-shuffle \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_7of8_vt_kl.log
```

```bash
# ===== tmux pane for GPU#3: 7/8 +lidar KL =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules camera/vtransform \
    --calib-batches 512 --calib-shuffle \
    --act-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_7of8_lidar_kl.log
```

```bash
# ===== tmux pane for GPU#4: 8/8 KL(vt only) =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --calib-batches 512 --calib-shuffle \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_8of8_kl_vt.log
```

### Round 4 结果解读指引

| 对比 | mini 结果 | 预期全量结果 | 意义 |
|------|-----------|-------------|------|
| 8/8 KL(both) vs 8/8 EMA (0.4562) | 0.5085 (+13.9 pts) | 预计 +5~10 pts | KL 综合效果 |
| 7/8+vt KL vs 7/8+vt EMA (0.6179) | 0.5720 vs 0.5474 | 预计 NDS ~0.66+ | vtransform KL 隔离效果 |
| 7/8+lidar KL vs 7/8+lidar EMA (0.5751) | 0.5173 vs 0.4734 | 预计 NDS ~0.60+ | lidar KL 隔离效果 |
| 8/8 KL(vt) vs 8/8 EMA (0.4562) | 0.4680 vs 0.4285 | 预计 +2~5 pts | 仅 vtransform 侧 KL 的贡献 |

---

## Round 5：KL Observer + 校准集修正（2026-03-13）

> **重要修正**：之前 Round 1-4 使用验证集（`cfg.data.val`）进行量化校准，这是方法论错误（导致过拟合到测试分布）。
> **Round 5 修正**：改用**训练集**（`cfg.data.train` + `test_mode=True`）进行校准，关闭数据增强。
> **核心实验**：
>   - GPU#0：7/8 +vt KL（新最优配置，skip lidar）
>   - GPU#1：8/8 KL(both)（全量化基线）
>   - GPU#3：7/8 +vt KL + shuffle（加强数据多样性）
>   - GPU#4：8/8 KL(both) + shuffle

### Round 5 Step 0：本地打包上传（PowerShell）

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

# 打包修正后的代码
git archive HEAD --format=tar.gz -o code_update_round5.tar.gz `
    tools/quant_ptq_minmax.py `
    docs/SERVER_DEPLOY.md `
    docs/RESULTS_LOG.md

# 上传
scp code_update_round5.tar.gz yellowstone@10.129.51.101:/tmp/

# 解压
ssh yellowstone@10.129.51.101 `
    "cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench && tar xzf /tmp/code_update_round5.tar.gz && rm /tmp/code_update_round5.tar.gz && echo 'Upload OK'"
```

### Round 5 Step 1：服务器环境确认

```bash
ssh yellowstone@10.129.51.101
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 确认训练集大小（应该显示 ~27000+ 帧）
python -c "import pickle; d = pickle.load(open('data/nuscenes/nuscenes_infos_train.pkl', 'rb')); print(f'Train frames: {len(d[\"infos\"])}')"

# 准备日志目录
mkdir -p logs
```

### Round 5 Step 2：创建 4 个 tmux 会话

```bash
tmux new-session -d -s round5_gpu0
tmux new-session -d -s round5_gpu1
tmux new-session -d -s round5_gpu3
tmux new-session -d -s round5_gpu4
```

### Round 5 Step 3：启动 4 个 KL 实验

#### GPU#0 — PTQ 7/8 +vt KL（skip lidar，128 batch 无 shuffle）

```bash
tmux attach -t round5_gpu0
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round5_ptq_7of8_vt_kl_calib128 \
    --skip-modules lidar/backbone \
    --calib-batches 128 \
    --act-observer kl_divergence \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/round5_ptq_7of8_vt_kl_calib128.log
# Ctrl+B D 断开
```

#### GPU#1 — PTQ 8/8 KL(both)（128 batch 无 shuffle）

```bash
tmux attach -t round5_gpu1
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round5_ptq_8of8_kl_both_calib128 \
    --calib-batches 128 \
    --act-observer kl_divergence \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/round5_ptq_8of8_kl_both_calib128.log
# Ctrl+B D 断开
```

#### GPU#3 — PTQ 7/8 +vt KL + shuffle

```bash
tmux attach -t round5_gpu3
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round5_ptq_7of8_vt_kl_calib128_shuffle \
    --skip-modules lidar/backbone \
    --calib-batches 128 \
    --calib-shuffle \
    --act-observer kl_divergence \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/round5_ptq_7of8_vt_kl_calib128_shuffle.log
# Ctrl+B D 断开
```

#### GPU#4 — PTQ 8/8 KL(both) + shuffle

```bash
tmux attach -t round5_gpu4
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round5_ptq_8of8_kl_both_calib128_shuffle \
    --calib-batches 128 \
    --calib-shuffle \
    --act-observer kl_divergence \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/round5_ptq_8of8_kl_both_calib128_shuffle.log
# Ctrl+B D 断开
```

### Round 5 Step 4：监控进度

```bash
# 查看所有 tmux 会话
tmux ls

# 监控某个实验（Ctrl+B D 断开不杀进程）
tmux attach -t round5_gpu0

# 查看日志末尾
tail -f logs/round5_ptq_7of8_vt_kl_calib128.log
```

### Round 5 Step 5：收集结果

#### 在服务器打包

```bash
tar czf round5_kl_calib128_results.tar.gz \
    logs/round5_ptq_7of8_vt_kl_calib128.log \
    logs/round5_ptq_8of8_kl_both_calib128.log \
    logs/round5_ptq_7of8_vt_kl_calib128_shuffle.log \
    logs/round5_ptq_8of8_kl_both_calib128_shuffle.log

ls -lh round5_kl_calib128_results.tar.gz
```

#### 本地拉取（PowerShell）

```powershell
cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts

scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/round5_kl_calib128_results.tar.gz .

# 解压并查看 NDS 结果
tar xzf round5_kl_calib128_results.tar.gz
grep -h "object/nds" round5_ptq_*.log | head -4
```

### 预期结果对比

| 实验 | 校准集 | Batch | 预期 NDS | 备注 |
|------|--------|-------|---------|------|
| 7/8 +vt KL | train | 128 | ~0.70x | 对标 Round4 的 0.7033（可能略微调整） |
| 8/8 KL(both) | train | 128 | ~0.57x | 对标 Round4 的 0.5750 |
| 7/8 +vt KL +shuffle | train | 128 | TBD | 加强数据多样性 |
| 8/8 KL(both) +shuffle | train | 128 | TBD | 全量化 + 多样性 |

**关键点**：
- 校准集已修正为 `cfg.data.train`（~27000 帧），关闭数据增强
- 128 batch 对训练集覆盖率约 0.5%，但包含充分的场景多样性
- 预期精度可能略有变化，但方法论更加正确

---
## Round 6：LiDAR/backbone 逐通道量化实验（2026-03-14）

> **目标**：验证 `lidar/backbone` 稀疏激活改为逐通道量化（`--sparse-act-mode per_channel`）后，是否显著缓解 8/8 的残余精度瓶颈。  
> **校准设置**：继续使用训练集（`cfg.data.train` + `test_mode=True`），`--calib-batches 128 --calib-shuffle`。  
> **兼容性提醒（Round 6 历史）**：当时 `--sparse-act-mode per_channel` 与 `--act-observer kl_divergence` 不兼容；该限制已在 Round 7 实现中解除。

### Round 6 Step 0：本地打包上传（PowerShell）

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

# 本地直接打包（包含未提交的本地修改，不依赖 git archive）
Remove-Item code_update_round6.tar.gz -ErrorAction SilentlyContinue
tar -czf code_update_round6.tar.gz `
    tools/quant_ptq_minmax.py `
    docs/SERVER_DEPLOY.md

# 上传到服务器项目目录
scp code_update_round6.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/
```

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar xzf code_update_round6.tar.gz
mkdir -p logs
```

### Round 6 Step 1：实验清单（4 个并行）

| GPU# | 实验名 | 量化配置 | 关键参数 |
|------|--------|----------|----------|
| 0 | R6-A | **PTQ6 + lidar(per-channel)**（=7/8，skip vtransform） | `--skip-modules camera/vtransform --act-observer ema_minmax --sparse-act-mode per_channel` |
| 1 | R6-B | PTQ6 + lidar(per-channel) + vt(KL)（=8/8） | `--vtransform-observer kl_divergence --act-observer ema_minmax --sparse-act-mode per_channel` |
| 3 | R6-C | PTQ6 + lidar(per-channel) + MSE（=7/8，skip vtransform） | `--skip-modules camera/vtransform --act-observer mse --sparse-act-mode per_channel` |
| 4 | R6-D | PTQ6 + lidar(per-channel) + vt(KL) + MSE（=8/8） | `--vtransform-observer kl_divergence --act-observer mse --sparse-act-mode per_channel` |

### Round 6 Step 2：启动实验（tmux，各 pane 一条）

```bash
# ===== tmux pane for GPU#0: R6-A (PTQ6 + lidar per-channel, EMA) =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round6_ptq6_plus_lidar_pc_ema_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --act-observer ema_minmax \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round6_ptq6_plus_lidar_pc_ema_calib128s.log
```

```bash
# ===== tmux pane for GPU#1: R6-B (PTQ6 + lidar per-channel + vt KL, EMA) =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round6_ptq6_plus_lidar_pc_vtkl_ema_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer ema_minmax \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round6_ptq6_plus_lidar_pc_vtkl_ema_calib128s.log
```

```bash
# ===== tmux pane for GPU#3: R6-C (PTQ6 + lidar per-channel, MSE) =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round6_ptq6_plus_lidar_pc_mse_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --act-observer mse \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round6_ptq6_plus_lidar_pc_mse_calib128s.log
```

```bash
# ===== tmux pane for GPU#4: R6-D (PTQ6 + lidar per-channel + vt KL, MSE) =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round6_ptq6_plus_lidar_pc_vtkl_mse_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer mse \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round6_ptq6_plus_lidar_pc_vtkl_mse_calib128s.log
```

### Round 6 Step 3：服务器打包结果并传回本地

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar czf round6_results.tar.gz \
    logs/round6_ptq6_plus_lidar_pc_ema_calib128s.log \
    logs/round6_ptq6_plus_lidar_pc_vtkl_ema_calib128s.log \
    logs/round6_ptq6_plus_lidar_pc_mse_calib128s.log \
    logs/round6_ptq6_plus_lidar_pc_vtkl_mse_calib128s.log \
    runs/round6_ptq6_plus_lidar_pc_ema_calib128s/ptq_minmax_model.pth \
    runs/round6_ptq6_plus_lidar_pc_vtkl_ema_calib128s/ptq_minmax_model.pth \
    runs/round6_ptq6_plus_lidar_pc_mse_calib128s/ptq_minmax_model.pth \
    runs/round6_ptq6_plus_lidar_pc_vtkl_mse_calib128s/ptq_minmax_model.pth
ls -lh round6_results.tar.gz
```

```powershell
# ===== 在本地 PowerShell 执行 =====
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/round6_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf round6_results.tar.gz

# 可选：仅拉日志（不拉模型）
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/logs/round6_*.log `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\logs\
```

### Round 6 结果解读指引

| 对比 | 目标 |
|------|------|
| R6-A vs 历史 7/8 +lidar(per-tensor) | 逐通道激活量化是否单独提升 lidar 分支 |
| R6-B vs 历史 8/8 KL(both) | 在 vt(KL) 已修复前提下，逐通道 lidar 是否抬升 8/8 上限 |
| R6-C vs R6-A | 在 per-channel 前提下，MSEObserver 是否优于 EMA |
| R6-D vs R6-B | vt(KL)+lidar(per-channel) 组合下，MSE 是否继续带来收益 |

---

## Round 7：LiDAR/backbone per-channel + KL Observer 实验（2026-03-14）

> **目标**：验证 `lidar/backbone` 稀疏激活在 **逐通道 + KL observer** 下，是否优于 Round 6 的 per-channel + EMA/MSE。  
> **校准设置**：训练集（`cfg.data.train` + `test_mode=True`），`--calib-batches 128 --calib-shuffle`。  
> **实现前提**：本轮代码已支持 `--act-observer kl_divergence --sparse-act-mode per_channel`。

### Round 7 Step 0：本地打包上传（PowerShell）

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

# 本地直接打包（包含未提交修改，不依赖 git archive）
Remove-Item code_update_round7.tar.gz -ErrorAction SilentlyContinue
tar -czf code_update_round7.tar.gz `
    tools/quant_ptq_minmax.py `
    docs/SERVER_DEPLOY.md

# 上传到服务器项目目录
scp code_update_round7.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/
```

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar xzf code_update_round7.tar.gz
mkdir -p logs
```

### Round 7 Step 1：实验清单（4 个并行）

| GPU# | 实验名 | 量化配置 | 关键参数 |
|------|--------|----------|----------|
| 0 | R7-A | 7/8（skip vtransform）+ lidar(per-channel KL) | `--skip-modules camera/vtransform --act-observer kl_divergence --sparse-act-mode per_channel` |
| 1 | R7-B | 8/8 + vt(KL) + lidar(per-channel KL) | `--vtransform-observer kl_divergence --act-observer kl_divergence --sparse-act-mode per_channel` |
| 3 | R7-C | 7/8 + lidar(per-channel KL) + LWC | `--skip-modules camera/vtransform --act-observer kl_divergence --sparse-act-mode per_channel --lwc` |
| 4 | R7-D | 8/8 + vt(KL) + lidar(per-channel KL) + LWC | `--vtransform-observer kl_divergence --act-observer kl_divergence --sparse-act-mode per_channel --lwc` |

### Round 7 Step 2：启动实验（tmux，各 pane 一条）

```bash
# ===== tmux pane for GPU#0: R7-A =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round7_ptq6_lidar_pc_kl_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --act-observer kl_divergence \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round7_ptq6_lidar_pc_kl_calib128s.log
```

```bash
# ===== tmux pane for GPU#1: R7-B =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round7_ptq8_vtkl_lidarpc_kl_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round7_ptq8_vtkl_lidarpc_kl_calib128s.log
```

```bash
# ===== tmux pane for GPU#3: R7-C =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round7_ptq6_lidar_pc_kl_lwc_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --act-observer kl_divergence \
    --sparse-act-mode per_channel \
    --lwc \
    2>&1 | tee logs/round7_ptq6_lidar_pc_kl_lwc_calib128s.log
```

```bash
# ===== tmux pane for GPU#4: R7-D =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round7_ptq8_vtkl_lidarpc_kl_lwc_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --sparse-act-mode per_channel \
    --lwc \
    2>&1 | tee logs/round7_ptq8_vtkl_lidarpc_kl_lwc_calib128s.log
```

### Round 7 Step 3：服务器打包结果并传回本地

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar czf round7_results.tar.gz \
    logs/round7_ptq6_lidar_pc_kl_calib128s.log \
    logs/round7_ptq8_vtkl_lidarpc_kl_calib128s.log \
    logs/round7_ptq6_lidar_pc_kl_lwc_calib128s.log \
    logs/round7_ptq8_vtkl_lidarpc_kl_lwc_calib128s.log \
    runs/round7_ptq6_lidar_pc_kl_calib128s/ptq_minmax_model.pth \
    runs/round7_ptq8_vtkl_lidarpc_kl_calib128s/ptq_minmax_model.pth \
    runs/round7_ptq6_lidar_pc_kl_lwc_calib128s/ptq_minmax_model.pth \
    runs/round7_ptq8_vtkl_lidarpc_kl_lwc_calib128s/ptq_minmax_model.pth
ls -lh round7_results.tar.gz
```

```powershell
# ===== 在本地 PowerShell 执行 =====
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/round7_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf round7_results.tar.gz

# 可选：仅拉日志（不拉模型）
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/logs/round7_*.log `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\logs\
```

### Round 7 结果解读指引

| 对比 | 目标 |
|------|------|
| R7-A vs R6-A | 在 7/8 场景下，KL(per-channel) 相比 EMA(per-channel) 是否提升 |
| R7-B vs R6-B | 在 8/8 + vt(KL) 下，KL(per-channel) 相比 EMA(per-channel) 是否提升 |
| R7-C vs R7-A | 在 KL(per-channel) 前提下，LWC 是否继续带来收益 |
| R7-D vs R7-B | 在 vt(KL)+KL(per-channel) 前提下，LWC 是否有额外增益 |


## Round 8：Sparse-Aware KL 校准 + W8A16 控制实验（2026-03）

> **目标**：验证 `KLDivergenceObserver` 的 `sparse_mode=True`（稀疏感知校准）修复能否
> 显著缓解 lidar/backbone 量化的 −18% NDS 损失。同时通过 W8A16 控制实验定量区分
> 激活量化 vs 权重量化各自的精度贡献。
>
> **修复内容（本轮新增）**：
>
> 1. `KLDivergenceObserver` 新增 `sparse_mode` 参数 —— 构建直方图时跳过 `|x| < 1e-6`
>    的零值元素，消除 ReLU 零值在 bin[0] 的巨大尖峰对 KL 搜索的偏差
> 2. `_QuantizedSparseConv.forward()` 改用 `_replace_feature()` 替代 in-place 写入，
>    权重还原用 `try-finally` 保证异常安全
> 3. 新增 `--no-lidar-act-quant` 参数（W8A16 模式）：lidar 只量化权重，激活保持 FP
>
> **校准设置**：训练集（`cfg.data.train` + `test_mode=True`），
> `--calib-batches 128 --calib-shuffle`（与 Round 5/7 保持一致）。

---

### Round 8 Step 0：本地打包上传（PowerShell）

```powershell
# ===== 在本地 PowerShell 执行 =====
cd D:\Research\Replication\BEVFusion_with_MQBench

Remove-Item code_update_round8.tar.gz -ErrorAction SilentlyContinue
tar -czf code_update_round8.tar.gz `
    tools/quant_ptq_minmax.py `
    docs/SERVER_DEPLOY.md

scp code_update_round8.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/
```

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar xzf code_update_round8.tar.gz
mkdir -p logs
```

---

### Round 8 Step 1：实验清单（4 个并行）

| GPU# | 实验名   | 量化配置                                                     | 关键新参数                                                   | 对比基准                                 |
| ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| 0    | **R8-A** | 7/8（skip vtransform）+ lidar **sparse-aware KL** per-tensor | `--act-observer kl_divergence` (自动 sparse_mode=True)       | R7-A（lidar per-channel KL，−18%）       |
| 1    | **R8-B** | 8/8 + vt(KL) + lidar **sparse-aware KL** per-tensor          | `--vtransform-observer kl_divergence --act-observer kl_divergence` | Round 5 8/8 KL(both，−18.7%）            |
| 3    | **R8-C** | 7/8（skip vtransform）+ lidar **W8A16**（仅权重量化）        | `--no-lidar-act-quant`                                       | 控制实验：测激活量化的单独贡献           |
| 4    | **R8-D** | 8/8 + vt(KL) + lidar **sparse-aware KL** per-channel         | `--vtransform-observer kl_divergence --act-observer kl_divergence --sparse-act-mode per_channel` | R7-B（per-channel KL，未含 sparse_mode） |

> **核心假设**：Round 6/7 中 per-channel KL 对 lidar 无效，是因为直方图被零值尖峰污染，
> KL 搜索返回了过大的截断阈值。`sparse_mode=True` 修复后，理论上效果应接近 vtransform KL
> 的改善幅度（从 −18% 降至 <5%）。

---

### Round 8 Step 2：启动实验（tmux，各 pane 一条）

**每个 tmux 窗口执行前的环境初始化（必须）：**

```bash
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

---

```bash
# ===== tmux pane for GPU#0: R8-A =====
# 7/8（skip vtransform）+ lidar sparse-aware KL per-tensor
# 核心验证：sparse_mode 修复是否能让 per-tensor KL 打破 −18% 瓶颈
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round8_ptq7_lidar_sparse_kl_pt_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --act-observer kl_divergence \
    --sparse-act-mode per_tensor \
    2>&1 | tee logs/round8_ptq7_lidar_sparse_kl_pt_calib128s.log
```

```bash
# ===== tmux pane for GPU#1: R8-B =====
# 8/8 + vt(KL) + lidar sparse-aware KL per-tensor
# 核心验证：8/8 全量化在 sparse_mode 修复后是否能越过 −18% 的上限
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round8_ptq8_vtkl_lidar_sparse_kl_pt_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --sparse-act-mode per_tensor \
    2>&1 | tee logs/round8_ptq8_vtkl_lidar_sparse_kl_pt_calib128s.log
```

```bash
# ===== tmux pane for GPU#3: R8-C =====
# 7/8（skip vtransform）+ lidar W8A16（仅权重量化，激活保持 FP）
# 控制实验：如果 NDS 损失接近 0，说明 −18% 100% 来自激活量化；
#           如果仍有较大损失，说明权重量化也有贡献
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round8_ptq7_lidar_w8a16_calib128s \
    --skip-modules camera/vtransform \
    --calib-batches 128 --calib-shuffle \
    --no-lidar-act-quant \
    2>&1 | tee logs/round8_ptq7_lidar_w8a16_calib128s.log
```

```bash
# ===== tmux pane for GPU#4: R8-D =====
# 8/8 + vt(KL) + lidar sparse-aware KL per-channel
# 在 sparse_mode 修复的基础上，验证 per-channel 是否进一步带来增益
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --run-dir runs/round8_ptq8_vtkl_lidar_sparse_kl_pc_calib128s \
    --calib-batches 128 --calib-shuffle \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --sparse-act-mode per_channel \
    2>&1 | tee logs/round8_ptq8_vtkl_lidar_sparse_kl_pc_calib128s.log
```

---

### Round 8 Step 3：监控进度

```bash
# 查看所有 tmux 窗口
tmux ls

# 查看特定实验输出（Ctrl+B D 断开，不杀进程）
tmux attach -t gpu0   # R8-A
tmux attach -t gpu1   # R8-B
tmux attach -t gpu3   # R8-C
tmux attach -t gpu4   # R8-D

# 实验完成后快速汇总 NDS
grep -h "object/nds" logs/round8_*.log | sed "s/.*'object\/nds': \([0-9.]*\).*/\1/"

# 或直接 grep NDS 行
grep -h "NDS\|nds" logs/round8_*.log | grep -v "^#"
```

---

### Round 8 Step 4：服务器打包结果并传回本地

```bash
# ===== 在服务器执行 =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
tar czf round8_results.tar.gz \
    logs/round8_ptq7_lidar_sparse_kl_pt_calib128s.log \
    logs/round8_ptq8_vtkl_lidar_sparse_kl_pt_calib128s.log \
    logs/round8_ptq7_lidar_w8a16_calib128s.log \
    logs/round8_ptq8_vtkl_lidar_sparse_kl_pc_calib128s.log \
    runs/round8_ptq7_lidar_sparse_kl_pt_calib128s/ptq_minmax_model.pth \
    runs/round8_ptq8_vtkl_lidar_sparse_kl_pt_calib128s/ptq_minmax_model.pth \
    runs/round8_ptq7_lidar_w8a16_calib128s/ptq_minmax_model.pth \
    runs/round8_ptq8_vtkl_lidar_sparse_kl_pc_calib128s/ptq_minmax_model.pth
ls -lh round8_results.tar.gz
```

```powershell
# ===== 在本地 PowerShell 执行 =====
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/round8_results.tar.gz `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\

cd D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts
tar xzf round8_results.tar.gz

# 可选：仅拉日志（不拉模型）
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/logs/round8_*.log `
    D:\Research\Replication\BEVFusion_with_MQBench\server_artifacts\logs\
```

---

### Round 8 结果解读指引

| 对比                                                         | 结论判断标准                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **R8-A vs R7-A**（sparse_mode KL/per-tensor vs per-channel KL/无sparse_mode） | NDS 大幅提升 → 零值污染是 Round 6/7 失败的根因；无提升 → 问题在别处 |
| **R8-B vs Round5 8/8 KL(both)**（含 sparse_mode vs 不含）    | 直接验证 sparse_mode 对 8/8 全量化上限的影响                 |
| **R8-C（W8A16）**                                            | NDS 损失 ≈ 0 → −18% 完全来自激活量化；损失 > 3% → 权重量化也有贡献 |
| **R8-D vs R8-B**（per-channel sparse_mode KL vs per-tensor sparse_mode KL） | 在 sparse_mode 已修复的前提下，per-channel 是否进一步带来增益 |

#### 结果预期表（供参考，实测以数据为准）

| 实验                              | 乐观预期 NDS          | 悲观预期 NDS      | 关键判断                                    |
| --------------------------------- | --------------------- | ----------------- | ------------------------------------------- |
| R8-A（7/8 sparse KL per-tensor）  | 0.68~0.70（损失 <1%） | 0.60（损失 ~14%） | 是否突破 Round 7 的 −18% 上限               |
| R8-B（8/8 sparse KL per-tensor）  | 0.67~0.70             | 0.57              | 全量化能否达到 7/8 水平                     |
| R8-C（7/8 W8A16 控制）            | ≥0.70（近零损失）     | 0.68（损失 ~4%）  | 精确拆解激活 vs 权重贡献                    |
| R8-D（8/8 sparse KL per-channel） | 0.68~0.70             | 0.60              | per-channel 是否在 sparse_mode 后有额外收益 |