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
| SwinT TRT FP16（6019帧） | 0.7069 | 已完成 |
| SwinT TRT INT8（6019帧） | 0.7022 | 已完成 |
