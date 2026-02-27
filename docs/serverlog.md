# Server Deployment Log

记录实验室服务器上的部署过程、已知问题和操作指南。

## 服务器信息

| 项目 | 内容 |
|------|------|
| SSH | `ssh yellowstone@10.129.51.101` |
| 密码 | wave12968 |
| 工作目录 | `/media/yellowstone/data2/CYL/BEVFusion_with_MQBench` |
| Conda 环境 | `bevfusion_mqbench` |
| CUDA Driver | 12.2（向下兼容 cu113） |
| CUDA Toolkit | `/usr/local/cuda-11.8` |

### GPU 配置

| GPU# | 型号 | 显存 | 可训练 |
|------|------|------|--------|
| 0 | RTX 3090 | 24 GB | ✅ |
| 1 | RTX 3090 | 24 GB | ✅ |
| 2 | A100-SXM4 | 80 GB | ❌（不能跑训练） |
| 3 | RTX 3090 | 24 GB | ✅ |
| 4 | RTX 3090 | 24 GB | ✅ |

---

## nuScenes 数据位置

```
/media/yellowstone/databig/data/nuscenes/nuscenes/
├── v1.0-trainval/     # 完整元数据
├── samples/
│   ├── CAM_FRONT/     # 34149 张（完整 trainval）✅
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   └── LIDAR_TOP/
├── sweeps/
│   └── LIDAR_TOP/     # 297737 个文件 ✅
└── maps/
```

这是师兄的数据，**只读，不能在里面写入新文件**。

### 项目内 data/nuscenes 的正确做法

在项目工作目录下创建空 `data/nuscenes/`，用 symlink 指向数据，
新生成的 pkl 文件写入自己目录：

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
mkdir -p data/nuscenes
ln -s /media/yellowstone/databig/data/nuscenes/nuscenes/v1.0-trainval data/nuscenes/
ln -s /media/yellowstone/databig/data/nuscenes/nuscenes/samples data/nuscenes/
ln -s /media/yellowstone/databig/data/nuscenes/nuscenes/sweeps data/nuscenes/
ln -s /media/yellowstone/databig/data/nuscenes/nuscenes/maps data/nuscenes/
```

---

## 已知问题与修复

### 1. setuptools 版本冲突（已修复）

**现象**: `ImportError: cannot import name 'packaging' from 'pkg_resources'`

**原因**: Conda 默认安装 `setuptools==70.3.0`，PyTorch 1.10 需要 `setuptools<65`。

**修复**（已写入 `lab_server_deploy.sh`）:
```bash
pip install "setuptools<65"
# 然后再安装 torch
```

**如果手动遇到此错误**:
```bash
conda activate bevfusion_mqbench
pip install "setuptools<65"
pip install -e .
```

### 2. pip install -e . 编译失败（连锁于问题1）

**现象**: `× Encountered error while generating package metadata`

**原因**: `setup.py` 第一行 `import torch`，torch 因 setuptools 问题无法导入。

**修复**: 先修复 setuptools 问题，然后重新 `pip install -e .`

### 3. \r 换行符问题（已解决）

脚本从 Windows 上传时可能含 `\r`，用以下命令修复：
```bash
sed -i 's/\r//' tools/lab_server_deploy.sh
```

---

## 操作流程

### ⚠️ 重要：所有长时间任务必须在 tmux 里运行

SSH 断开或切换网络会杀死直接运行的进程。

```bash
tmux new -s bevfusion        # 新建 session
tmux attach -t bevfusion     # 重新连接
# Ctrl+B, D 暂时退出（进程继续运行）
```

### 环境激活

```bash
source /home/yellowstone/anaconda3/etc/profile.d/conda.sh
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
```

### 生成 pkl 数据文件（约 150 分钟，CPU-bound）

```bash
tmux new -s bevfusion
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

CUDA_VISIBLE_DEVICES=0 python tools/prepare_nuscenes_data.py \
  --root data/nuscenes \
  --version v1.0-trainval
```

完成后验证：
```bash
ls -lh data/nuscenes/nuscenes_infos_temporal_*.pkl
# 应看到两个文件：nuscenes_infos_temporal_train.pkl 和 nuscenes_infos_temporal_val.pkl
```

### 启动训练

```bash
bash tools/lab_server_deploy.sh train
```

训练参数（在脚本顶部配置）：
- `GPU_ID=0`（RTX 3090）
- `BATCH_SIZE=4`（RTX 3090 24GB 安全值）
- `MAX_EPOCHS=6`
- `RUN_DIR=runs/resnet50_fulldata`

查看训练进度：
```bash
tail -f runs/resnet50_fulldata/train.log
watch -n 5 nvidia-smi
```

### 下载训练好的权重（在本地执行）

```powershell
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/runs/resnet50_fulldata/latest.pth D:\Research\Replication\BEVFusion_with_MQBench\pretrained\bevfusion-resnet50.pth
```

---

## 部署状态记录

| 日期 | 操作 | 状态 |
|------|------|------|
| 2026-02-27 | 上传代码包 `bevfusion_deploy.tar.gz`，解压 | ✅ |
| 2026-02-27 | 创建 conda 环境 `bevfusion_mqbench` | ✅ |
| 2026-02-27 | 安装 PyTorch 1.10.2+cu113、mmcv、spconv 等 | ✅ |
| 2026-02-27 | 发现 setuptools 版本冲突，脚本已修复 | ✅ |
| 2026-02-27 | 找到 nuScenes 数据，设置 symlink | ✅ |
| 2026-02-27 | 修复 \r 换行符问题 | ✅ |
| 2026-02-27 | pip install -e . 编译 CUDA 算子 | ⏳ 待确认 |
| 2026-02-27 | 生成 pkl 文件 | ⏳ 待完成 |
| — | 启动训练 | ⏳ 待完成 |
