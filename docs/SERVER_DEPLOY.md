# 服务器部署与运行手册（PTQ 消融实验）

> **目标**：在服务器（4×RTX 3090，完整 nuScenes trainval 6019帧）上运行全部 PTQ 消融实验  
> **前提**：FP32 baseline（NDS=0.7069）已完成，**无需重测**  
> **新实验**：PTQ 三路径量化消融（8/8、6/8、7/8×2），4 张 3090 并行，约 3 小时

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

> **注意**：GPU#0 的 8/8 KL(both) 512 batch 已经在跑了。
> 下面 GPU#1/3/4 在各自 tmux pane 里直接运行（不用 nohup &，输出直接可见）。

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

## Round 5：KL Observer + 128 calib batch（2026-03-11）

> **背景**：Round 3 已证明 512 batch shuffle 相比 128 batch 对 EMAMinMax 几乎无收益（8/8 仅 +2.6%）。
> 对 KL Observer 而言，因为 KL 本身通过直方图积累多个 batch，128 batch 应已足够（服务器验证集不像 EMAMinMax 那样需要多样性）。
> Round 5 用 128 calib batch 重跑核心实验，与 Round 4 的 512 batch 对比，验证校准量的影响。

### 运行命令（Round 4 全部完成后再跑）

```bash
# ===== tmux pane for GPU#0: 8/8 KL(both) 128 batch =====
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --calib-batches 128 \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_8of8_kl_both_128.log
```

```bash
# ===== tmux pane for GPU#1: 7/8 +vtransform KL 128 batch =====
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --calib-batches 128 \
    --vtransform-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_7of8_vt_kl_128.log
```

```bash
# ===== tmux pane for GPU#3: 7/8 +lidar KL 128 batch =====
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules camera/vtransform \
    --calib-batches 128 \
    --act-observer kl_divergence \
    2>&1 | tee logs/results_server_ptq_7of8_lidar_kl_128.log
```

### Round 5 结果解读指引

| 对比 | 意义 |
|------|------|
| 8/8 KL 128 vs 8/8 KL 512 | KL 对校准量是否敏感（预期 ±0.005 以内） |
| 7/8+vt KL 128 vs 7/8+vt KL 512 | vtransform KL 的校准量敏感性 |
| 7/8+lidar KL 128 vs 7/8+lidar KL 512 | lidar KL 的校准量敏感性 |

