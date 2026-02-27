#!/bin/bash
# ============================================================
# ResNet-50 BEVFusion 训练脚本 (远程服务器)
# ============================================================
# 用法:
#   bash tools/train_resnet50_server.sh                  # 从头训练
#   bash tools/train_resnet50_server.sh --resume         # 断点续训
#   bash tools/train_resnet50_server.sh --batch 4        # 指定 batch_size
#   bash tools/train_resnet50_server.sh --workers 4      # 指定 workers
#   bash tools/train_resnet50_server.sh --no-val          # 跳过验证(无 sweeps 数据时)
# ============================================================

set -e

# ---------- 默认参数 ----------
BATCH_SIZE=4
WORKERS=4
RESUME=""
CONFIG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml"
RUN_DIR="runs/resnet50_fulldata"
NO_VALIDATE=""
EXTRA_OPTS=""

# ---------- 解析命令行参数 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="--resume"
            shift
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --no-val)
            NO_VALIDATE="--no-validate"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_OPTS="$EXTRA_OPTS $1"
            shift
            ;;
    esac
done

# ---------- 环境检查 ----------
echo "=========================================="
echo "  BEVFusion ResNet-50 训练"
echo "=========================================="
echo "Config:     $CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Workers:    $WORKERS"
echo "Run dir:    $RUN_DIR"
echo "Resume:     ${RESUME:-no}"
echo ""

# 检查 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

# 检查数据
if [ ! -f "data/nuscenes/nuscenes_infos_temporal_train.pkl" ]; then
    echo "[ERROR] 找不到 data/nuscenes/nuscenes_infos_temporal_train.pkl"
    echo "请先运行: python tools/prepare_nuscenes_data.py --root data/nuscenes --version v1.0-trainval"
    exit 1
fi

# ---------- 构建训练命令 ----------
RESUME_FLAG=""
if [ -n "$RESUME" ] && [ -f "$RUN_DIR/latest.pth" ]; then
    RESUME_FLAG="resume_from=$RUN_DIR/latest.pth"
    echo "[INFO] 从 $RUN_DIR/latest.pth 断点续训"
fi

echo ""
echo "[INFO] 开始训练..."

# 单GPU训练 (不使用 torchpack 分布式)
python tools/train.py \
    "$CONFIG" \
    --run-dir "$RUN_DIR" \
    $NO_VALIDATE \
    data.samples_per_gpu="$BATCH_SIZE" \
    data.workers_per_gpu="$WORKERS" \
    $RESUME_FLAG \
    $EXTRA_OPTS

echo ""
echo "=========================================="
echo "  训练完成！"
echo "=========================================="
echo "模型权重: $RUN_DIR/"
ls -lh "$RUN_DIR"/*.pth 2>/dev/null || echo "  (无 checkpoint 文件)"
