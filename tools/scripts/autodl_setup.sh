#!/bin/bash
# ============================================================
# AutoDL / Remote Server 环境搭建脚本
# BEVFusion + MQBench (ResNet-50 训练)
# ============================================================
# 用法: bash tools/autodl_setup.sh
# 
# 前提:
#   - Ubuntu 18.04/20.04
#   - CUDA 11.3+
#   - 已安装 conda (AutoDL 自带)
#   - nuScenes 数据已放在 data/nuscenes/
# ============================================================

set -e  # 遇错即停

echo "=========================================="
echo "  BEVFusion ResNet-50 环境搭建"
echo "=========================================="

# ---------- 1. 创建 conda 环境 ----------
echo "[1/7] 创建 conda 环境 bevfusion (Python 3.8) ..."
if conda info --envs | grep -q bevfusion; then
    echo "  环境已存在，跳过创建"
else
    conda create -n bevfusion python=3.8 -y
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate bevfusion

# ---------- 2. 安装 PyTorch ----------
echo "[2/7] 安装 PyTorch 1.10.2+cu113 ..."
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --quiet

# ---------- 3. 安装 mmcv-full ----------
echo "[3/7] 安装 mmcv-full 1.4.0 ..."
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html --quiet

# ---------- 4. 安装 mmdet ----------
echo "[4/7] 安装 mmdet 2.20.0 ..."
pip install mmdet==2.20.0 --quiet

# ---------- 5. 安装其他依赖 ----------
echo "[5/7] 安装其他依赖 ..."
pip install \
    torchpack==0.3.1 \
    nuscenes-devkit==1.1.9 \
    pyquaternion \
    pyyaml \
    numba \
    numpy==1.23.5 \
    pillow \
    shapely \
    tensorboard \
    --quiet

# ---------- 6. 安装 spconv (稀疏卷积) ----------
echo "[6/7] 安装 spconv-cu113 ..."
pip install spconv-cu113==2.1.25 --quiet 2>/dev/null || {
    echo "  spconv-cu113 安装失败，尝试 cumm-cu113 ..."
    pip install cumm-cu113 spconv-cu113 --quiet
}

# ---------- 7. 安装本项目 (mmdet3d) ----------
echo "[7/7] 安装 mmdet3d (本项目) ..."
cd "$(dirname "$0")/.."
pip install -e . --quiet

echo ""
echo "=========================================="
echo "  环境搭建完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 确保 nuScenes 数据在 data/nuscenes/ 下"
echo "  2. 生成数据 pkl: bash tools/prepare_nuscenes_data.sh"
echo "  3. 开始训练: bash tools/train_resnet50_server.sh"
