#!/bin/bash
# ============================================================
# 实验室服务器一键部署+训练脚本
# 服务器: yellowstone@10.129.51.101 (密码: wave12968)
# GPU: RTX 3090 24GB × 4 (A100 仅限验证不能训练)
# 工作目录: /media/yellowstone/data2/CYL
# ============================================================
#
# 【部署步骤】
#
# 在本地 PowerShell 执行:
#   scp bevfusion_deploy.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/
#
# SSH 登录服务器:
#   ssh yellowstone@10.129.51.101
#   cd /media/yellowstone/data2/CYL
#   tar xzf bevfusion_deploy.tar.gz
#   cd BEVFusion_with_MQBench
#   bash tools/lab_server_deploy.sh
#
# 【分阶段运行】
#   bash tools/lab_server_deploy.sh find-data   # 仅查找 nuScenes
#   bash tools/lab_server_deploy.sh setup-env   # 仅搭建环境
#   bash tools/lab_server_deploy.sh prepare     # 仅生成 pkl
#   bash tools/lab_server_deploy.sh train       # 仅启动训练
#   bash tools/lab_server_deploy.sh all         # 全部 (默认)
# ============================================================

set -e

# ---- 配置 ----
WORK_DIR="/media/yellowstone/data2/CYL/BEVFusion_with_MQBench"
CONDA_ENV="bevfusion_mqbench"
GPU_ID=0  # RTX 3090 24GB (GPU#2 A100 不能训练)
NUM_GPUS=1  # 单GPU训练。设 2/3/4 可多卡并行(需 torchpack)
BATCH_SIZE=4
WORKERS=8
MAX_EPOCHS=6
RUN_DIR="runs/resnet50_fulldata"

# ---- 颜色 ----
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# 查找 nuScenes 数据
# ============================================================
find_nuscenes_data() {
    log_info "========== 查找 nuScenes 数据 =========="
    NUSCENES_PATH=""

    for dir in /media/yellowstone/databig /media/yellowstone/databig2 \
               /media/yellowstone/databig3 /media/yellowstone/data2 \
               /media/yellowstone/Dataset /home/yellowstone; do
        [ ! -d "$dir" ] && continue
        log_info "搜索 $dir ..."

        while IFS= read -r f; do
            parent=$(dirname "$f")
            log_info "  候选: $parent"
            [ -d "$parent/samples/CAM_FRONT" ] && {
                cam=$(ls "$parent/samples/CAM_FRONT" 2>/dev/null | wc -l)
                log_info "    CAM_FRONT: $cam 张"; }
            [ -d "$parent/samples/LIDAR_TOP" ] && {
                lid=$(ls "$parent/samples/LIDAR_TOP" 2>/dev/null | wc -l)
                log_info "    LIDAR_TOP: $lid 个"; }
            [ -d "$parent/samples" ] && NUSCENES_PATH="$parent"
        done < <(find "$dir" -maxdepth 5 -type d -name "v1.0-trainval" 2>/dev/null)

        while IFS= read -r f; do
            [ -d "$f/v1.0-trainval" ] || [ -d "$f/samples" ] && {
                log_info "  候选: $f"; NUSCENES_PATH="$f"; }
        done < <(find "$dir" -maxdepth 3 -type d -iname "nuscenes" 2>/dev/null)
    done

    if [ -z "$NUSCENES_PATH" ]; then
        log_error "未找到 nuScenes！手动查找:"
        log_info "  find /media -name 'v1.0-trainval' -type d 2>/dev/null"
        log_info "  find /media -iname 'nuscenes' -type d 2>/dev/null"
        log_info "如果服务器没有数据，需要下载 (~65GB keyframe):"
        log_info "  https://www.nuscenes.org/download"
        return 1
    fi

    log_info "✅ nuScenes: $NUSCENES_PATH"
    echo "$NUSCENES_PATH"
}

# ============================================================
# 搭建环境
# ============================================================
setup_environment() {
    log_info "========== 搭建 Conda 环境 =========="
    cd "$WORK_DIR"

    # conda 初始化
    eval "$(conda shell.bash hook)" 2>/dev/null || \
        source /home/yellowstone/anaconda3/etc/profile.d/conda.sh 2>/dev/null

    if conda info --envs | grep -q "$CONDA_ENV"; then
        log_info "环境已存在，激活"
        conda activate "$CONDA_ENV"
    else
        log_info "创建 $CONDA_ENV (Python 3.8)..."
        conda create -n "$CONDA_ENV" python=3.8 -y
        conda activate "$CONDA_ENV"

        log_info "安装 PyTorch 1.10.2+cu113..."
        pip install "setuptools<65"  # PyTorch 1.10 需要旧版 setuptools (pkg_resources.packaging)
        pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
            -f https://download.pytorch.org/whl/cu113/torch_stable.html

        log_info "安装 mmcv-full 1.4.0..."
        pip install mmcv-full==1.4.0 \
            -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

        log_info "安装其他依赖..."
        pip install mmdet==2.20.0 torchpack==0.3.1 nuscenes-devkit==1.1.9 \
            pyquaternion pyyaml numba "numpy<1.24" pillow shapely tensorboard

        log_info "安装 spconv..."
        pip install spconv-cu113==2.1.25 2>/dev/null || pip install cumm-cu113 spconv-cu113

        pip install mqbench==0.0.6 2>/dev/null || log_warn "MQBench 跳过 (训练不需要)"
    fi

    # CUDA toolkit 处理 (编译自定义 CUDA 算子需要 nvcc)
    # 服务器 CUDA driver 12.2 向下兼容 cu113，但编译需要匹配的 toolkit
    log_info "检查 CUDA toolkit..."
    if [ -d "/usr/local/cuda-11.3" ]; then
        export CUDA_HOME=/usr/local/cuda-11.3
    elif [ -d "/usr/local/cuda-11.6" ]; then
        export CUDA_HOME=/usr/local/cuda-11.6
    elif [ -d "/usr/local/cuda-11.8" ]; then
        export CUDA_HOME=/usr/local/cuda-11.8
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    fi
    if [ -n "$CUDA_HOME" ]; then
        log_info "CUDA_HOME=$CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
    fi

    log_info "编译安装 mmdet3d (含 CUDA 算子)..."
    pip install -e . 2>&1 | tail -5
    
    log_info "验证..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA OK: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB')
import mmcv, mmdet, mmdet3d; print('mmcv/mmdet/mmdet3d OK')
"
    log_info "✅ 环境完成！"
}

# ============================================================
# 数据准备
# ============================================================
prepare_data() {
    log_info "========== 准备数据 =========="
    cd "$WORK_DIR"

    eval "$(conda shell.bash hook)" 2>/dev/null || source /home/yellowstone/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate "$CONDA_ENV"

    NUSCENES_PATH=$(find_nuscenes_data 2>/dev/null | tail -1)
    [ -z "$NUSCENES_PATH" ] && { log_error "找不到数据"; return 1; }

    mkdir -p data
    if [ ! -L "data/nuscenes" ] && [ ! -d "data/nuscenes" ]; then
        ln -s "$NUSCENES_PATH" data/nuscenes
        log_info "链接: data/nuscenes -> $NUSCENES_PATH"
    fi

    if [ -f "data/nuscenes/nuscenes_infos_temporal_train.pkl" ]; then
        log_info "✅ pkl 已存在"; return 0
    fi

    log_info "生成 pkl (约 5-10 分钟)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python tools/prepare_nuscenes_data.py \
        --root data/nuscenes --version v1.0-trainval

    [ -f "data/nuscenes/nuscenes_infos_temporal_train.pkl" ] && log_info "✅ 数据完成！" || {
        log_error "pkl 生成失败"; return 1; }
}

# ============================================================
# 开始训练
# ============================================================
start_training() {
    log_info "========== 开始训练 =========="
    cd "$WORK_DIR"

    eval "$(conda shell.bash hook)" 2>/dev/null || source /home/yellowstone/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate "$CONDA_ENV"

    CONFIG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml"

    [ ! -f "data/nuscenes/nuscenes_infos_temporal_train.pkl" ] && {
        log_error "缺少 pkl，先: bash $0 prepare"; return 1; }

    RESUME_FLAG=""
    [ -f "$RUN_DIR/latest.pth" ] && {
        RESUME_FLAG="resume_from=$RUN_DIR/latest.pth"
        log_info "断点续训: $RUN_DIR/latest.pth"; }

    NO_VAL=""
    [ ! -d "data/nuscenes/sweeps/LIDAR_TOP" ] && {
        NO_VAL="--no-validate"
        log_warn "无 sweeps，跳过验证"; }

    log_info "GPU: $NUM_GPUS x RTX 3090, batch=$BATCH_SIZE/gpu, epochs=$MAX_EPOCHS"
    mkdir -p "$RUN_DIR"

    # 设置 CUDA_HOME (编译算子时已设置，训练时也保持)
    for cuda_dir in /usr/local/cuda-11.3 /usr/local/cuda-11.6 /usr/local/cuda-11.8 /usr/local/cuda; do
        [ -d "$cuda_dir" ] && { export CUDA_HOME="$cuda_dir"; break; }
    done

    if [ "$NUM_GPUS" -gt 1 ]; then
        # 多GPU: 使用 torchpack 分布式训练
        # GPU 0,1,3,4 是 RTX 3090 (跳过 GPU#2 A100)
        GPU_LIST=$(seq 0 $((NUM_GPUS > 4 ? 3 : NUM_GPUS - 1)) | grep -v 2 | head -$NUM_GPUS | tr '\n' ',' | sed 's/,$//')
        log_info "多GPU模式: CUDA_VISIBLE_DEVICES=$GPU_LIST, np=$NUM_GPUS"
        CUDA_VISIBLE_DEVICES=$GPU_LIST nohup torchpack dist-run -np $NUM_GPUS \
            python tools/train.py \
            "$CONFIG" --run-dir "$RUN_DIR" $NO_VAL \
            data.samples_per_gpu=$BATCH_SIZE data.workers_per_gpu=$WORKERS \
            max_epochs=$MAX_EPOCHS $RESUME_FLAG \
            > "$RUN_DIR/train.log" 2>&1 &
    else
        # 单GPU
        CUDA_VISIBLE_DEVICES=$GPU_ID nohup python tools/train.py \
            "$CONFIG" --run-dir "$RUN_DIR" $NO_VAL \
            data.samples_per_gpu=$BATCH_SIZE data.workers_per_gpu=$WORKERS \
            max_epochs=$MAX_EPOCHS $RESUME_FLAG \
            > "$RUN_DIR/train.log" 2>&1 &
    fi

    PID=$!; echo "$PID" > "$RUN_DIR/train.pid"

    log_info "=========================================="
    log_info "  训练已后台启动! PID: $PID"
    log_info "  可安全断开 SSH"
    log_info "=========================================="
    log_info "看日志: tail -f $RUN_DIR/train.log"
    log_info "看GPU:  nvidia-smi"
    log_info "停训练: kill $PID"
    log_info ""
    log_info "完成后本地下载:"
    log_info "  scp yellowstone@10.129.51.101:$WORK_DIR/$RUN_DIR/latest.pth D:\\Research\\Replication\\BEVFusion_with_MQBench\\pretrained\\"

    sleep 5
    kill -0 "$PID" 2>/dev/null && {
        log_info "✅ 运行正常"; tail -3 "$RUN_DIR/train.log" 2>/dev/null; } || {
        log_error "进程退出! 日志:"; tail -20 "$RUN_DIR/train.log" 2>/dev/null; }
}

# ============================================================
STAGE="${1:-all}"
case "$STAGE" in
    find-data)  find_nuscenes_data ;;
    setup-env)  setup_environment ;;
    prepare)    prepare_data ;;
    train)      start_training ;;
    all)        setup_environment; prepare_data; start_training ;;
    *)          echo "用法: bash $0 [find-data|setup-env|prepare|train|all]"; exit 1 ;;
esac
