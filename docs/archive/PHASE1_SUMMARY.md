# Phase 1 部署准备总结

## 🎯 Phase 1 目标

SwinTransformer 静态化与 Q/DQ ONNX 导出

## ✅ 已完成的工作

### 1. 核心工具开发

#### 1.1 TensorRT 引擎构建工具
- **文件**: `tools/export_utils/build_engine.py`
- **功能**: 替代 `trtexec`，适配 TRT Python API 10.15
- **特性**:
  - 支持 FP16/INT8 精度
  - 自定义 Plugin 加载
  - 可配置显存限制
  - 详细日志输出

#### 1.2 静态 Pad 值获取工具
- **文件**: `tools/export_utils/get_static_pad_values.py`
- **功能**: 自动获取 SwinTransformer 所有 AdaptivePadding 的静态 pad 值
- **输出**: 格式化的 pad 值列表，用于静态化修改

#### 1.3 MQBench ONNX Symbolic 注册
- **文件**: `tools/export_utils/mqbench_onnx_symbolic.py`
- **功能**: 注册自定义量化算子的 ONNX 导出规则
- **支持算子**:
  - `LearnableFakeQuantize` → `QuantizeLinear` + `DequantizeLinear`
  - `SparseLog2FakeQuantize` → `custom::SparseLog2Quant`

#### 1.4 Phase 1 完整执行脚本
- **文件**: `tools/export_utils/phase1_swin_export.py`
- **功能**: 一键执行 Phase 1 所有步骤
- **包含**: Pad 值检测 → ONNX 导出 → 模型验证

#### 1.5 环境设置脚本
- **文件**: `tools/export_utils/setup_phase1.sh`
- **功能**: 自动化环境验证和设置
- **检查项**:
  - Conda 环境
  - 依赖包版本
  - 关键文件存在性
  - 量化权重状态

### 2. 文档和指南

#### 2.1 工具使用文档
- **文件**: `tools/export_utils/README.md`
- **内容**:
  - 目录结构说明
  - 快速开始指南
  - 分步执行说明
  - 工具详细用法
  - 常见问题解答

#### 2.2 Phase 1 总结文档
- **文件**: `docs/PHASE1_SUMMARY.md` (本文件)
- **内容**: 完整的 Phase 1 工作总结和下一步指南

## 📁 创建的文件结构

```
tools/export_utils/
├── README.md                      # 工具使用文档
├── build_engine.py                # TensorRT 引擎构建工具
├── get_static_pad_values.py       # 静态 pad 值获取
├── mqbench_onnx_symbolic.py       # ONNX Symbolic 注册
├── phase1_swin_export.py          # Phase 1 完整脚本
└── setup_phase1.sh                # 环境设置验证

docs/
└── PHASE1_SUMMARY.md              # Phase 1 总结文档
```

## 🚀 服务器部署步骤

### 快速启动 (推荐)

```bash
# 1. 环境设置
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 2. 验证环境
bash tools/export_utils/setup_phase1.sh

# 3. 一键执行 Phase 1
python tools/export_utils/phase1_swin_export.py
```

### 分步执行

#### Step 1.1: 获取静态 Pad 值

```bash
python tools/export_utils/get_static_pad_values.py
```

**预期输出**:
```
AdaptivePadding | input=(256,704) | pad=(1,3)
AdaptivePadding | input=(128,352) | pad=(0,1)
...
```

#### Step 1.2: 静态化 swin.py (可选)

基于 Step 1.1 的输出，修改 `mmdet3d/models/backbones/swin.py`:

1. **AdaptivePadding 静态化**
   - 将动态 pad 计算替换为常量
   - 消除 `if pad_h > 0 or pad_w > 0` 的动态分支

2. **attn_mask 静态化**
   - 将动态 mask 生成改为预计算的 buffer
   - 避免运行时动态计算

#### Step 1.3: 导出 ONNX 模型

```bash
# 完整流程
python tools/export_utils/phase1_swin_export.py

# 或者仅导出 (跳过 pad 检测)
python tools/export_utils/phase1_swin_export.py --skip-pad-detection
```

**输出**: `swin_int8.onnx`

### 构建 TensorRT 引擎

```bash
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx \
    --engine swin_int8.engine \
    --int8 --fp16
```

## 🔧 关键技术要点

### 1. SwinTransformer 静态化挑战

**动态控制流问题**:
- `AdaptivePadding` 根据输入尺寸动态计算 pad
- `attn_mask` 根据窗口位置动态生成

**解决方案**:
- 基于固定输入尺寸 (256, 704) 预计算静态值
- 将动态分支改为常量路径
- 预计算 attn_mask 并注册为 buffer

### 2. MQBench 量化算子导出

**LearnableFakeQuantize**:
- Per-tensor: `QuantizeLinear` + `DequantizeLinear`
- Per-channel: 添加 `axis` 参数

**SparseLog2FakeQuantize**:
- 自定义 ONNX 算子: `custom::SparseLog2Quant`
- 需要 TensorRT Plugin 支持

### 3. ONNX 导出最佳实践

- **Opset Version**: 13 (兼容 TensorRT 8.6)
- **静态输入**: 固定 (1, 3, 256, 704)
- **Symbolic 注册**: 必须在 `torch.onnx.export` 之前导入
- **验证流程**: ONNX 结构检查 → ONNX Runtime 推理测试

## 📊 预期结果

### 成功标志

1. **静态 Pad 值获取**: 成功打印所有 AdaptivePadding 的 pad 值
2. **ONNX 导出**: 生成 `swin_int8.onnx` 文件 (~50-100 MB)
3. **ONNX 验证**: 结构检查通过，ONNX Runtime 推理成功
4. **TensorRT 引擎**: 成功构建 `.engine` 文件

### 失败排查

#### ONNX 导出失败
- 检查 `mqbench_onnx_symbolic.py` 是否正确导入
- 确认量化权重存在且正确加载
- 查看详细的 ONNX 导出日志

#### TensorRT 引擎构建失败
- 检查 ONNX 文件有效性
- 确认 TensorRT 版本匹配
- 对于自定义算子，验证 Plugin 加载

## 🎯 下一步工作

### Phase 2: 其他模块 ONNX 导出

1. **Camera Neck**: FPN neck 量化导出
2. **VTransform**: LSS bev_pool + KL Observer 量化
3. **Fusion Module**: ConvFuser 多模态融合
4. **Decoder**: SECOND backbone + neck
5. **Detection Head**: TransFusionHead

### Phase 3: LiDAR Backbone

1. **SparseConv 量化**: Log2 域量化支持
2. **Custom Plugin**: SparseConv TensorRT Plugin
3. **端到端测试**: LiDAR path 完整推理

### Phase 4: 完整模型部署

1. **多模块组合**: 所有模块串联推理
2. **精度验证**: 与 PyTorch 结果对比
3. **性能基准**: 延迟、吞吐量测试

### Phase 5: 优化和生产

1. **性能调优**: Kernel 优化、内存优化
2. **部署封装**: C++/Python 推理接口
3. **文档完善**: 部署指南、最佳实践

## 📝 开发笔记

### 关键文件映射

- **SwinTransformer**: `mmdet3d/models/backbones/swin.py`
- **量化配置**: `tools/quant_ptq_minmax.py`
- **模型配置**: `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`

### 重要参数

- **输入尺寸**: (1, 3, 256, 704) - 固定尺寸
- **Opset Version**: 13 - ONNX 导出版本
- **TensorRT**: 8.6.1 C++ SDK / 10.15.1 Python API
- **CUDA**: 11.3 (PyTorch) / 11.8 (nvcc)

### 依赖版本

```bash
PyTorch: 1.10.2+cu113
TensorRT Python API: 10.15.1.29
TensorRT C++ SDK: 8.6.1.6
MQBench: 0.0.6
```

## 🔗 相关资源

- **完整计划**: `docs/NEXT_PLAN.md`
- **项目报告**: `docs/REPORT.md`
- **技术问答**: `docs/QandA.md`
- **部署手册**: `docs/SERVER_DEPLOY.md`

---

**Phase 1 状态**: ✅ 准备就绪，可以开始服务器部署

**最后更新**: 2026-03-18 (Phase 1 工具完成)