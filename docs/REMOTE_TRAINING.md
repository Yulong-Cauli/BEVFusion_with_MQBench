# ResNet-50 BEVFusion 远程训练指南

## 概述

将 SwinTransformer 替换为 ResNet-50 作为 camera backbone，需要在完整 nuScenes 数据集上从头训练。本指南适用于 AutoDL 或实验室服务器。

## 硬件需求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| GPU | V100-16GB | V100-32GB / A100-40GB |
| 显存 | 16 GB | 32+ GB |
| 内存 | 32 GB | 64+ GB |
| 磁盘 | 100 GB | 200+ GB |
| CUDA | 11.3+ | 11.3 |

### 训练时间估算

| GPU | batch_size | 每 epoch 时间 | 6 epochs 总时间 |
|-----|-----------|-------------|---------------|
| V100-32GB | 8 | ~4.5h | ~27h |
| V100-32GB | 4 | ~7h | ~42h |
| A100-40GB | 8 | ~3h | ~18h |
| RTX 4090 | 4 | ~5h | ~30h |

### AutoDL 费用参考

| GPU | 单价 | 6 epochs 估算 |
|-----|------|-------------|
| V100-32GB | 2元/h | ~50-55元 |
| A100-40GB | 3.28元/h | ~50-60元 |

## 第一步：准备代码

```bash
# 在远程服务器上
git clone https://github.com/<your-repo>/BEVFusion_with_MQBench.git
cd BEVFusion_with_MQBench
```

## 第二步：搭建环境

```bash
bash tools/autodl_setup.sh
conda activate bevfusion
```

手动验证安装：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmdet; print(mmdet.__version__)"
python -c "import mmdet3d; print('mmdet3d OK')"
```

## 第三步：准备 nuScenes 数据

### 数据目录结构

```
data/nuscenes/
├── v1.0-trainval/          # 元数据 (从官网下载 v1.0-trainval_meta.tgz)
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── samples/                # 关键帧传感器数据 (v1.0-trainval01~10_blobs.tgz)
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   └── LIDAR_TOP/
├── sweeps/                 # 非关键帧数据 (验证时用，训练时不需要)
│   └── LIDAR_TOP/
└── maps/
```

### 下载方式

**方式 A：从 nuScenes 官网下载**
1. 注册 https://www.nuscenes.org/
2. 下载 v1.0-trainval 数据（分 10 个 blob 包 + 1 个 meta 包）
3. 全部解压到 `data/nuscenes/`

**方式 B：仅下载关键帧（节省空间）**
如果磁盘有限，只需要：
- `v1.0-trainval_meta.tgz` (~300 MB) — 必须
- `v1.0-trainval01_blobs.tgz` 到 `v1.0-trainval10_blobs.tgz` — 包含 `samples/` 目录
- 不需要 `sweeps/` 目录的数据（训练时 sweeps_num=0）

> 注意：如果没有 sweeps 数据，验证时会缺少多帧 LiDAR 输入，NDS 略低。
> 可以用 `--no-val` 参数跳过训练中的验证，训练完再单独评估。

**方式 C：从网盘传输**
如果你有 50 GB 的关键帧数据包，直接传到 `data/nuscenes/` 并确保目录结构正确。

### 生成数据 info 文件

```bash
conda activate bevfusion
python tools/prepare_nuscenes_data.py --root data/nuscenes --version v1.0-trainval
```

这会生成：
- `data/nuscenes/nuscenes_infos_temporal_train.pkl` (训练集 info)
- `data/nuscenes/nuscenes_infos_temporal_val.pkl` (验证集 info)

## 第四步：开始训练

### 基本用法

```bash
# 从头训练 (batch_size=4, workers=4)
bash tools/train_resnet50_server.sh

# V100-32GB 推荐设置 (batch_size=8)
bash tools/train_resnet50_server.sh --batch 8 --workers 8

# 无 sweeps 数据时跳过验证
bash tools/train_resnet50_server.sh --batch 8 --workers 8 --no-val
```

### 断点续训

如果训练中断（AutoDL 到期、网络断开等）：
```bash
bash tools/train_resnet50_server.sh --resume --batch 8 --workers 8
```

### 使用 tmux/screen 后台训练

```bash
# 使用 tmux 防止 SSH 断开导致训练中断
tmux new -s train
bash tools/train_resnet50_server.sh --batch 8 --workers 8
# Ctrl+B 然后 D 分离，重新连接: tmux attach -t train
```

### 使用 nohup 后台训练

```bash
nohup bash tools/train_resnet50_server.sh --batch 8 --workers 8 > train.log 2>&1 &
tail -f train.log  # 查看进度
```

## 第五步：获取训练结果

训练完成后，模型权重在 `runs/resnet50_fulldata/` 目录下：
```bash
ls -lh runs/resnet50_fulldata/*.pth
```

下载最佳权重到本地：
```bash
# 在本地机器执行 (替换 <server> 为你的服务器地址)
scp <server>:~/BEVFusion_with_MQBench/runs/resnet50_fulldata/latest.pth ./pretrained/resnet50_bevfusion.pth
```

## 第六步：本地评估

将训练好的权重放到 `pretrained/` 目录，在本地 mini 数据集上快速验证：
```bash
$env:PYTHONUTF8="1"
python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml --checkpoint pretrained/resnet50_bevfusion.pth --eval bbox
```

## 配置说明

### ResNet-50 vs SwinTransformer

| 对比项 | SwinTransformer | ResNet-50 |
|--------|----------------|-----------|
| 参数量 (camera backbone) | 27.5M | 23.5M |
| FPN 输入通道 | [192, 384, 768] | [512, 1024, 2048] |
| fx 追踪兼容 | ❌ (动态控制流) | ✅ |
| 量化友好 | ❌ | ✅ |
| 预期 NDS | ~0.580 | ~0.540-0.560 |

### 关键配置文件

- 基础配置: `configs/nuscenes/default.yaml`
- 检测配置: `configs/nuscenes/det/default.yaml`
- ResNet-50: `configs/.../resnet50_v0p075/default.yaml`
- Fuser: `configs/.../resnet50_v0p075/convfuser.yaml`

### 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=2e-4, weight_decay=0.01) |
| LR 策略 | CosineAnnealing + linear warmup (500 iters) |
| 总 epochs | 6 |
| FP16 | 开启 (init_scale=512) |
| 梯度裁剪 | max_norm=35 |

## 常见问题

### Q: 训练到一半 AutoDL 到期了怎么办？
A: 每个 epoch 结束都会自动保存 checkpoint。续费后用 `--resume` 继续。

### Q: 没有 sweeps 目录，验证报错？
A: 使用 `--no-val` 跳过验证。训练完后再用完整数据评估。

### Q: 磁盘不够放 nuScenes？
A: 关键帧数据约 65 GB。AutoDL 可扩容数据盘。或用符号链接指向其他路径。

### Q: batch_size 开到 8 显存不够？
A: 降到 4。V100-16GB 用 batch_size=2-4，V100-32GB 用 4-8，A100 用 8-16。
