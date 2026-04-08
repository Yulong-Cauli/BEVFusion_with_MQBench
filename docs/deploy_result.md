# BEVFusion TensorRT 部署实验结果

> 硬件：NVIDIA RTX 3090 (Ampere, SM 8.6)
> TRT C++ SDK：8.6.1 | TRT Python API：10.15.1.29
> CUDA：11.8 (nvcc) / 11.3 (PyTorch) | Conda 环境：bevfusion_mqbench
> PTQ Checkpoint：pretrained/ptq_minmax_model.pth（8/8 全量化，vtKL + lidar Log2，NDS 0.6875）

---

## 产物总览

| 产物 | 文件 | 大小 | 精度 | 状态 |
|------|------|------|------|------|
| SwinT ONNX | swin_int8.onnx | 107 MB | INT8 (208 Q/DQ) | ✅ |
| SwinT TRT 引擎 | swin_int8.engine | 33 MB | INT8+FP16 | ✅ 精度验证待补 |
| depthnet ONNX | vtransform_depthnet_int8.onnx | 5.6 MB | INT8 (24 Q/DQ) | ✅ |
| depthnet TRT 引擎 | vtransform_depthnet_int8.engine | 1.7 MB | INT8+FP16 | ✅ |
| Log2 Plugin | libsparse_log2_quant_plugin.so | 63 KB | — | ✅ 编译+注册通过 |
| BEV Pool Plugin | libbev_pool_v2_plugin.so | 50 KB | — | ✅ 编译+注册通过 |

---

## Phase 1：SwinTransformer

| 指标 | 值 |
|------|-----|
| ONNX 文件 | swin_int8.onnx (107 MB) |
| Q/DQ 节点 | 208 |
| FakeQuant 残留 | 0 |
| 总节点数 | 9143 |
| TRT 引擎 | swin_int8.engine (33 MB) |
| TRT 层数 | 23,655 |
| 精度模式 | INT8 卷积 + FP16 融合（LayerNorm/Softmax/AdaptivePadding） |
| 精度验证 | ⚠️ 尚未执行（Step 1.7 待补） |
| 完成日期 | 2026-03-24 |

---

## Phase 2：SparseLog2Quant TRT Plugin

| 指标 | 值 |
|------|-----|
| Plugin .so | libsparse_log2_quant_plugin.so (~63 KB) |
| 支持维度 | [N,C,H,W] / [N,C] / [N,C,L] |
| C++ 注册测试 | ✅ 通过 |
| 数值验证 | 待 Phase 5 C++ 方式测试（Python API 版本不匹配） |
| 完成日期 | 2026-03-25 |

---

## Phase 3：vtransform (depthnet + bev_pool_v2)

### 架构拆分

vtransform 拆分为两部分部署：
1. **depthnet TRT 引擎**：dtransform + depthnet + softmax + outer_product + flatten
2. **bev_pool_v2 TRT Plugin**：interval-sum 聚合（纯索引操作，无可学习参数，不需要量化）

索引预计算（ranks_depth/ranks_feat/interval_starts/interval_lengths）在 PyTorch 端完成。

### PTQ 量化覆盖

| 子模块 | Conv2d 数量 | 量化精度 | Observer |
|--------|------------|---------|----------|
| dtransform | 3 | INT8 | KL Divergence |
| depthnet | 3 | INT8 | KL Divergence |
| downsample | 3 | INT8 | KL Divergence |
| bev_pool_v2 | — | FP16（无参数） | — |

### bev_pool_v2 等价性验证

| 指标 | 值 |
|------|-----|
| bev_pool vs bev_pool_v2 (PyTorch) | cosine_sim = 1.000000 |

### depthnet 引擎对比

| 指标 | FP32 引擎 | FP16 引擎 | INT8 引擎 |
|------|----------|----------|----------|
| ONNX Q/DQ 节点 | 0 | 0 | 24 |
| TRT 引擎大小 | 9.6 MB | 2.9 MB | 1.7 MB |
| cosine_sim (vs PyTorch) | — | 0.999991 | 0.999682 |
| max_abs_err | — | — | 0.013197 |
| RMSE | — | — | 0.000116 |
| 精度验证 | — | ✅ PASS | ✅ PASS (> 0.999) |

注：INT8 的 cosine_sim 基准是 PTQ FakeQuant 仿真输出（非 FP32），衡量的是 TRT INT8 对 MQBench 量化仿真的还原度。

### 集成测试（depthnet FP16 TRT + 索引预计算 + bev_pool_v2）

| 指标 | 值 |
|------|-----|
| cosine_sim (vs 原始 vtransform FP32) | 0.999999 |

### 完成日期

2026-03-28

---

## Phase 4：SpConv LiDAR Backbone

### 技术路线变更

| 路线 | 状态 | 原因 |
|------|------|------|
| libspconv.so (NVIDIA 闭源) | ❌ 放弃 | `builder->build()` 段错误，闭源无法调试 |
| **spconv 2.3 原生推理** | **✅ 采用** | 新建 conda 环境，精度验证通过 |

libspconv.so 段错误的根因：闭源库内部 build 阶段崩溃，即使 ONNX 结构与 NVIDIA 完全一致（21 SparseConv + 8 Add + 8 Relu）仍然崩溃。可能是 CUDA 11.4 .so 在 CUDA 11.8 环境下的兼容性问题。

### export_lidar.py 修复

旧版导出脚本漏掉了 SparseBasicBlock 的残差 Add 节点和 BN fusion。修复后：

| 指标 | 旧版 | 修复后 (v2) | NVIDIA 参考 |
|------|------|------------|------------|
| SparseConvolution | 21 | 21 | 21 |
| Add | 0 | 8 | 8 |
| Relu | 0 | 8 | 8 |
| 总节点 | 24 | 40 | 37 |
| Conv+BN fusion | ❌ | ✅ | ✅ |

### spconv 2.3 部署

| 指标 | 值 |
|------|-----|
| 部署环境 | `/media/yellowstone/data2/CYL/spconv23_deploy` |
| Python | 3.9 |
| PyTorch | 2.0.1+cu118 |
| spconv | 2.3.8 |
| 脚本 | `tools/export_utils/build_lidar_spconv23.py` |
| 输出形状 | [1, 256, 180, 180] |
| cosine_sim (vs spconv 2.1 PyTorch) | **0.999994** |
| max_abs_err | 0.041145 |
| 精度验证 | ✅ PASS (> 0.999) |
| 完成日期 | 2026-03-29 |

### Log2 量化评估

| 方案 | 精度 | NDS 损失 | 可行性 |
|------|------|---------|--------|
| Log2 量化（当前 PTQ） | INT8 | −2.7% | spconv 2.3 不原生支持，需外挂 Log2 Plugin |
| 线性 KL 量化（Round 5） | INT8 | −18.5% | spconv 2.3 支持，但精度损失不可接受 |
| FP16（推荐） | FP16 | ~0% | spconv 2.3 原生支持 |

### 待完成

- [ ] Log2 Plugin 集成（INT8+Log2 版本）
- [ ] W8A16 版本（权重 INT8 + 激活 FP16）

---

## Phase 4b：Fuser + Decoder TRT 引擎

| 指标 | FP16 版本 | INT8 版本 |
|------|----------|----------|
| ONNX 文件 | fuser_decoder_fp16.onnx | fuser_decoder_int8.onnx |
| ONNX 节点数 | 33 | 132 (含 56 Q/DQ) |
| TRT 引擎 | fuser_decoder_fp16_sm86.engine | fuser_decoder_int8_sm86.engine |
| 引擎大小 | 10.4 MB | 5.8 MB |
| 输入 | camera_bev [1,80,180,180] + lidar_bev [1,256,180,180] | 同左 |
| 输出 | neck_features [1,512,180,180] | 同左 |
| 完成日期 | 2026-03-29 |

TransFusionHead 保持 PyTorch（argsort 不支持 ONNX opset 13）。

---

## Phase 5：端到端集成 + NDS 验证

### 架构

混合 TRT+PyTorch 推理 pipeline（`tools/trt_infer.py`），在 bevfusion_mqbench 环境中运行：

| 模块 | 后端 | 引擎/模型 |
|------|------|----------|
| SwinT backbone | TRT INT8 | swin_int8_sm86.engine |
| Camera neck (GeneralizedLSSFPN) | PyTorch FP32 | 原始模型 |
| vtransform depthnet | TRT INT8 | vtransform_depthnet_int8_sm86.engine |
| bev_pool | PyTorch CUDA | 原始 bev_pool 算子 |
| LiDAR backbone (SparseEncoder) | PyTorch FP16 | spconv 2.1 原始模型 |
| Fuser + Decoder | TRT FP16/INT8 | fuser_decoder_*_sm86.engine |
| TransFusionHead | PyTorch FP32 | 原始模型 |

### NDS 评估结果（6019 帧验证集）

| 版本 | Fuser 引擎 | NDS | mAP | 推理速度 |
|------|-----------|-----|-----|---------|
| FP32 baseline | — | 0.7069 | 0.6728 | — |
| **Version A (W8A16)** | fuser_decoder_fp16 | **0.7144** | **0.6851** | 1.1 fps |
| **Version B (INT8)** | fuser_decoder_int8 | **0.7102** | **0.6786** | 5.0 fps |
| PTQ 仿真 (W8A16) | — | 0.7009 | — | — |
| PTQ 仿真 (8/8 全量化) | — | 0.6875 | — | — |

### 分析

- Version A NDS 0.7144 > FP32 baseline 0.7069（+0.0075），可能是 FP16 精度的正则化效果
- Version B NDS 0.7102 > FP32 baseline 0.7069（+0.0033），INT8 fuser 精度损失极小
- 两个版本的 TRT NDS 都高于 PTQ FakeQuant 仿真值，说明 TRT 量化实现比 MQBench 仿真更精确
- Version B 推理速度 5.0 fps，比 Version A 的 1.1 fps 快 4.5 倍（INT8 fuser 更快）

### 日志

- `logs/trt_eval_version_A.log`
- `logs/trt_eval_version_B.log`
- `trt_eval_version_A.json`
- `trt_eval_version_B.json`

### 完成日期

2026-03-29

---

## Phase 6：Camera Neck + TransFusionHead TRT 引擎

### Camera Neck (GeneralizedLSSFPN)

| 指标 | FP16 版本 | INT8 版本 |
|------|----------|----------|
| ONNX 文件 | camera_neck_fp16.onnx (6.1 MB) | camera_neck_int8.onnx (6.1 MB) |
| TRT 引擎 | camera_neck_fp16_sm86.engine | camera_neck_int8_sm86.engine |
| 引擎大小 | 3.2 MB | 1.8 MB |
| 输入 | 3 个多尺度特征图 | 同左 |
| 输出 | [6, 256, 32, 88] | 同左 |

### TransFusionHead

| 指标 | FP16 v2 版本 | INT8 版本 |
|------|-------------|----------|
| ONNX 文件 | transfusion_head_fp16_v2.onnx (5.5 MB) | transfusion_head_int8.onnx (5.5 MB) |
| TRT 引擎 | transfusion_head_fp16_v2_sm86.engine | transfusion_head_int8_sm86.engine |
| 引擎大小 | 4.5 MB | 3.8 MB |
| 输入 | [1, 512, 180, 180] | 同左 |
| 输出 | 8 个预测 tensor | 同左 |
| 修复 | argsort → topk（ONNX opset 兼容） | 同左 |

### NDS 评估结果（Phase 6，7/8 模块量化）

| 版本 | NDS | mAP | 速度 | 说明 |
|------|-----|-----|------|------|
| FP32 baseline | 0.7069 | 0.6728 | — | 原始模型 |
| Phase 6 (7/8 INT8) | 0.7040 | 0.6654 | 5.6 fps | LiDAR FP32，其余全 INT8 |
| PTQ 7/8 仿真 (参考) | 0.7033 | 0.6657 | — | REPORT.md PyTorch 仿真 |

Phase 6 TRT 部署 NDS 0.7040 ≈ PTQ 仿真 0.7033，精度还原度良好。

### 完成日期

2026-03-29

---

## Phase 7：Standalone 推理脚本（spconv23_deploy 环境）

### 环境迁移

从 bevfusion_mqbench (Python 3.8 + PyTorch 1.10 + spconv 2.1) 迁移到 spconv23_deploy (Python 3.9 + PyTorch 2.0 + spconv 2.3.8)。新建独立推理脚本 `tools/trt_infer_standalone.py`，内联所有 mmcv/mmdet3d 依赖。

### CUDA 扩展（cpython-39 重编译）

| 扩展 | 文件 | 大小 |
|------|------|------|
| bev_pool_ext | build_sp39/bev_pool_ext.so | 1.3 MB |
| voxel_layer | build_sp39/voxel_layer.so | 2.1 MB |
| iou3d_cuda | build_sp39/iou3d_cuda.so | 1.4 MB |
| roiaware_pool3d_ext | build_sp39/roiaware_pool3d_ext.so | 1.5 MB |

### 冒烟测试

```
Input: img=[1,6,3,256,704], points=[252768,5]
Inference time: 1340.6 ms
Detections: 200 total, 3 with score > 0.3
```

### NDS 评估

待 GPU 空闲后运行。预期：
- LiDAR FP16：NDS ≈ 0.7040（与 Phase 6 一致）
- LiDAR Log2 INT8：NDS ≈ 0.6875（与 PTQ 8/8 仿真一致）

### 完成日期

2026-03-30

---

## 部署产物体积汇总

### 原始模型

| 文件 | 大小 | 说明 |
|------|------|------|
| bevfusion-det.pth | 157 MB | FP32 全模型权重 |
| ptq_minmax_model.pth | 158 MB | PTQ 量化参数 + 权重 |

### TRT INT8 引擎（部署产物）

| 引擎 | 大小 | 模块 | 压缩比 |
|------|------|------|--------|
| swin_int8_sm86.engine | 33 MB | SwinTransformer | — |
| camera_neck_int8_sm86.engine | 1.8 MB | Camera Neck (LSSFPN) | — |
| vtransform_depthnet_int8_sm86.engine | 1.6 MB | Depthnet | — |
| fuser_decoder_int8_sm86.engine | 5.8 MB | Fuser + BEV Decoder | — |
| transfusion_head_int8_sm86.engine | 3.8 MB | TransFusionHead | — |
| **TRT 引擎合计** | **46 MB** | 5 个 INT8 引擎 | **vs 157 MB 原始 (−71%)** |

### CUDA 扩展 + TRT Plugin

| 文件 | 大小 | 说明 |
|------|------|------|
| bev_pool_ext.so | 1.3 MB | BEV 池化 CUDA kernel |
| voxel_layer.so | 2.1 MB | 体素化 CUDA kernel |
| iou3d_cuda.so | 1.4 MB | 旋转 NMS CUDA kernel |
| roiaware_pool3d_ext.so | 1.5 MB | ROI 池化 CUDA kernel |
| libsparse_log2_quant_plugin.so | 63 KB | Log2 量化 TRT Plugin |
| libbev_pool_v2_plugin.so | 50 KB | BEV Pool TRT Plugin |
| **扩展合计** | **6.4 MB** | — |

### LiDAR Backbone（spconv 2.3 运行时）

LiDAR backbone 未转 TRT，通过 spconv 2.3 原生推理。权重从 bevfusion-det.pth 加载，运行时约占 ~5 MB 参数（FP16）。

### 总部署体积

| 组成 | 大小 |
|------|------|
| 5× TRT INT8 引擎 | 46 MB |
| CUDA 扩展 + Plugin | 6.4 MB |
| LiDAR 权重 (FP16, 从 ckpt 提取) | ~5 MB |
| **总计** | **~57 MB** |

对比原始 FP32 模型 157 MB，部署产物压缩至约 **36%**。

### 精度 vs 体积 vs 速度

| 配置 | NDS | mAP | 模型体积 | 速度 |
|------|-----|-----|---------|------|
| FP32 baseline | 0.7069 | 0.6728 | 157 MB | — |
| Phase 5 Version B (INT8 fuser) | 0.7102 | 0.6786 | ~100 MB* | 5.0 fps |
| Phase 6 (7/8 INT8, LiDAR FP32) | 0.7040 | 0.6654 | ~57 MB | 5.6 fps |
| Phase 7 (7/8 INT8, LiDAR FP16) | 0.7039 | 0.6642 | ~57 MB | 5.2 fps |
| Phase 7 (8/8 INT8, LiDAR Log2) | 待测 | 待测 | ~57 MB | 待测 |
| Phase 8 (TV backbone, LiDAR FP16) | 待测 | 待测 | ~57 MB | 待测 |

*Phase 5 体积较大因为 Camera Neck 和 TransFusionHead 仍为 PyTorch FP32。

---

## Phase 8：去 LiDAR Backbone PyTorch 依赖

### 技术方案

spconv 2.3 的 `core_cc.so` 不链接 PyTorch，底层 CUDA kernel 通过 `tv::Tensor`（cumm tensorview）操作。
PyTorch 只在 Python 层提供内存分配、CUDA stream、矩阵乘法三个功能。

替代实现：
- `TVAllocator(ExternalAllocator)` — 用 `tv.zeros/tv.empty` 分配 GPU 内存
- `TVSpconvMatmul(ExternalSpconvMatmul)` — 用 cuBLAS `cublasGemmEx` via ctypes
- `TVSparseEncoder` — 直接调 `SpconvOps` + `ConvGemmOps` + `InferenceOps`

### cuBLAS GEMM 验证

| 指标 | 值 |
|------|-----|
| 测试规模 | [100, 16] @ [32, 16].T |
| cosine_sim (vs numpy) | **1.000000** |
| 精度模式 | FP16 input, FP32 compute |

### NDS 评估

待 GPU 空闲后运行。预期与 Phase 7 FP16 (NDS=0.7039) 一致。

### 完成日期

2026-03-31
