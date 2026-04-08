# BEVFusion TensorRT 部署工具集

## 目录结构

```
tools/export_utils/
├── README.md                      # 本文件
├── build_engine.py                # TensorRT 引擎构建工具
├── get_static_pad_values.py       # 获取 SwinTransformer 静态 pad 值
├── mqbench_onnx_symbolic.py       # MQBench ONNX Symbolic 注册
├── phase1_swin_export.py          # Phase 1 完整执行脚本
└── setup_phase1.sh                # Phase 1 环境设置脚本
```

## Phase 1: SwinTransformer 静态化与 Q/DQ ONNX 导出

### 快速开始

在服务器上执行：

```bash
# 1. 激活环境
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 2. 一键执行 Phase 1
python tools/export_utils/phase1_swin_export.py

# 3. 验证输出
ls -lh swin_int8.onnx
```

### 分步执行

#### Step 1.1: 获取静态 pad 值

```bash
python tools/export_utils/get_static_pad_values.py
```

#### Step 1.2: 静态化 swin.py

根据 Step 1.1 的输出，修改 `mmdet3d/models/backbones/swin.py`：

1. **AdaptivePadding 静态化**: 将动态 pad 计算改为常量
2. **attn_mask 静态化**: 将动态 mask 生成改为 buffer

#### Step 1.3: 导出 ONNX

```bash
python tools/export_utils/phase1_swin_export.py --skip-pad-detection
```

### 工具说明

#### build_engine.py

替代 `trtexec` 的 TensorRT 引擎构建工具。

**用法:**
```bash
# 基础 INT8+FP16 引擎
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx \
    --engine swin_int8.engine \
    --int8 --fp16

# 带 Plugin 的引擎
python tools/export_utils/build_engine.py \
    --onnx lidar_backbone.onnx \
    --engine lidar_backbone.engine \
    --fp16 \
    --plugins tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
```

#### get_static_pad_values.py

获取 SwinTransformer 中所有 AdaptivePadding 层的静态 pad 值。

**输出示例:**
```
AdaptivePadding | input=(256,704) | pad=(1,3)
AdaptivePadding | input=(128,352) | pad=(0,1)
...
```

#### mqbench_onnx_symbolic.py

注册 MQBench 自定义量化算子的 ONNX Symbolic：

- `LearnableFakeQuantize` → `QuantizeLinear` + `DequantizeLinear`
- `SparseLog2FakeQuantize` → `custom::SparseLog2Quant`

⚠️ **必须在任何 `torch.onnx.export` 调用之前导入**

#### phase1_swin_export.py

Phase 1 完整执行脚本，包含：

1. 自动获取静态 pad 值（可选）
2. 导出 SwinTransformer ONNX 模型
3. 验证 ONNX 模型结构

**用法:**
```bash
# 完整流程（含 pad 值获取）
python tools/export_utils/phase1_swin_export.py

# 仅导出（跳过 pad 值获取）
python tools/export_utils/phase1_swin_export.py --skip-pad-detection

# 自定义输出路径
python tools/export_utils/phase1_swin_export.py --output swin_fp32.onnx
```

## 环境要求

- Python >= 3.7
- PyTorch 1.10.2+cu113
- TensorRT Python API 10.15.1.29
- ONNX + ONNX Runtime (可选，用于验证)
- MQBench 0.0.6

## 常见问题

### 1. ONNX 导出失败

**问题**: `torch.onnx.export` 报错

**解决**:
- 确保已导入 `tools.export_utils.mqbench_onnx_symbolic`
- 检查模型中是否有不支持的动态控制流
- 查看 `mqbench_onnx_symbolic.py` 的注册日志

### 2. TensorRT 引擎构建失败

**问题**: `build_engine.py` 报错

**解决**:
- 检查 ONNX 文件是否有效
- 确认 TensorRT 版本匹配
- 对于自定义算子，确保 Plugin .so 文件存在且正确加载

### 3. Plugin 加载失败

**问题**: `ctypes.CDLL` 无法加载 Plugin

**解决**:
- 检查 Plugin 编译时的 CUDA/TensorRT 版本
- 确认 `LD_LIBRARY_PATH` 设置正确
- 使用 `ldd` 检查 Plugin 依赖

## 下一步

Phase 1 完成后，继续：

1. **Phase 2**: 导出其他模块（camera neck, vtransform, fusion, decoder）
2. **Phase 3**: LiDAR backbone SparseConv 量化导出
3. **Phase 4**: 端到端推理验证
4. **Phase 5**: 性能基准测试

详见 `docs/NEXT_PLAN.md`