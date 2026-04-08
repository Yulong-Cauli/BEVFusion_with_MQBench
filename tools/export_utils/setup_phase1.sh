#!/bin/bash
# Phase 1 环境设置和快速验证脚本
# 在服务器上执行此脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Phase 1: BEVFusion TensorRT 部署环境设置"
echo "=========================================="

# 1. 检查 Conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ 请先激活 Conda 环境:"
    echo "   conda activate bevfusion_mqbench"
    exit 1
fi

echo "✅ Conda 环境: $CONDA_DEFAULT_ENV"

# 2. 设置环境变量
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
echo "✅ LD_LIBRARY_PATH 已设置"

# 3. 切换到项目目录
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
echo "✅ 工作目录: $(pwd)"

# 4. 验证 Python 导入
echo ""
echo "验证 Python 环境..."

python - <<'EOF'
import sys
import torch
import tensorrt as trt

print(f"✅ Python: {sys.version.split()[0]}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ TensorRT: {trt.__version__}")

# 检查关键模块
sys.path.insert(0, ".")
from tools.quant_ptq_minmax import SparseLog2FakeQuantize
print("✅ SparseLog2FakeQuantize 可导入")

import tools.export_utils.mqbench_onnx_symbolic
print("✅ MQBench ONNX Symbolic 已注册")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Python 环境验证失败"
    exit 1
fi

# 5. 验证关键文件存在
echo ""
echo "验证关键文件..."

check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
    else
        echo "❌ $1 不存在"
        exit 1
    fi
}

check_file "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
# swin.py 在 conda env 的 mmdet 包中
python -c "import mmdet.models.backbones.swin; print('✅ mmdet/models/backbones/swin.py 存在')"
check_file "tools/quant_ptq_minmax.py"

# 6. 验证工具脚本
echo ""
echo "验证工具脚本..."

check_file "tools/export_utils/build_engine.py"
check_file "tools/export_utils/get_static_pad_values.py"
check_file "tools/export_utils/mqbench_onnx_symbolic.py"
check_file "tools/export_utils/phase1_swin_export.py"

# 7. 检查量化权重
echo ""
echo "检查量化权重..."

if [ -f "pretrained/ptq_minmax_model.pth" ]; then
    echo "✅ 量化权重存在: pretrained/ptq_minmax_model.pth"
    python -c "import torch; sd = torch.load('pretrained/ptq_minmax_model.pth', map_location='cpu'); print(f'   模型参数数量: {len(sd)}')"
else
    echo "⚠️  量化权重不存在: pretrained/ptq_minmax_model.pth"
    echo "   请先运行量化流程:"
    echo "   python tools/quant_ptq_minmax.py \\"
    echo "       configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\"
    echo "       --load-from pretrained/bevfusion-det.pth"
fi

# 8. 环境信息汇总
echo ""
echo "=========================================="
echo "环境信息汇总"
echo "=========================================="
echo "硬件          : NVIDIA RTX 3090 (Ampere, SM 8.6)"
echo "CUDA (nvcc)   : 11.8"
echo "CUDA (PyTorch): $(torch.version.cuda)"
echo "TRT Python API: $(python -c 'import tensorrt as trt; print(trt.__version__)')  (conda env: $CONDA_DEFAULT_ENV)"
echo "TRT C++ SDK   : 8.6.1  (/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6)"
echo "工作目录      : $(pwd)"
echo "Conda 环境    : $CONDA_DEFAULT_ENV"

echo ""
echo "=========================================="
echo "✅ Phase 1 环境设置完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. 运行 Phase 1 完整流程:"
echo "   python tools/export_utils/phase1_swin_export.py"
echo ""
echo "2. 或者分步执行:"
echo "   # Step 1.1: 获取静态 pad 值"
echo "   python tools/export_utils/get_static_pad_values.py"
echo ""
echo "   # Step 1.3: 导出 ONNX"
echo "   python tools/export_utils/phase1_swin_export.py --skip-pad-detection"
echo ""
echo "3. 构建 TensorRT 引擎:"
echo "   python tools/export_utils/build_engine.py \\"
echo "       --onnx swin_int8.onnx --engine swin_int8.engine \\"
echo "       --int8 --fp16"
echo ""
echo "详细文档: tools/export_utils/README.md"