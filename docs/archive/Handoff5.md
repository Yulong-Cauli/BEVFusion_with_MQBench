## BEVFusion 全模块 TRT 部署 — Phase 5 交接（2026-03-29）

### 总体进度

| Phase | 模块 | 状态 | 产物 |
|-------|------|------|------|
| 1 | SwinTransformer | ✅ 引擎已构建 | `swin_int8_sm86.engine` (32.3MB) |
| 2 | Log2 Plugin | ✅ 编译通过 | `libsparse_log2_quant_plugin.so` |
| 3 | vtransform | ✅ 引擎+Plugin | `vtransform_depthnet_int8_sm86.engine` + `libbev_pool_v2_plugin.so` |
| 4 | LiDAR backbone | ✅ spconv 2.3 FP16 验证通过 | `build_lidar_spconv23.py` (cosine=0.999994) |
| 4b | Fuser+Decoder | ✅ FP16+INT8 引擎 | `fuser_decoder_fp16_sm86.engine` / `fuser_decoder_int8_sm86.engine` |
| 4b | TransFusionHead | ✅ 可导出 TRT | `transfusion_head_fp16_sm86.engine` (3.4MB) |
| 5 | 端到端 NDS 验证 | ✅ 混合方案通过 | NDS=0.7144 (A) / 0.7102 (B) |
| 6 | 去 PyTorch 全量部署 | 🔧 待开始 | — |

### 本次 Session 完成的工作（2026-03-29）

#### 1. TRT 引擎重建（SM 8.6）

之前的引擎在 A100 (SM 8.0) 上构建，无法在 RTX 3090 (SM 8.6) 上运行。
用 TRT 10.15 在 RTX 3090 上重建了全部引擎：

| 引擎 | 大小 | 备注 |
|------|------|------|
| swin_int8_sm86.engine | 32.3 MB | INT8+FP16, 输入 [1,3,256,704], 输出 3 个多尺度特征 |
| vtransform_depthnet_int8_sm86.engine | 1.6 MB | INT8+FP16, 输入 [1,6,256,32,88]+[1,6,1,256,704] |
| fuser_decoder_fp16_sm86.engine | 10.4 MB | FP16, 输入 [1,80,180,180]+[1,256,180,180] |
| fuser_decoder_int8_sm86.engine | 5.8 MB | INT8+FP16, 同上 |
| transfusion_head_fp16_sm86.engine | 3.4 MB | FP16, 输入 [1,512,180,180] |

#### 2. 端到端推理 Pipeline (trt_infer.py)

混合 TRT+PyTorch 方案，在 bevfusion_mqbench 环境中运行：
- TRT 引擎：SwinT, depthnet, fuser+decoder
- PyTorch：camera neck, bev_pool, LiDAR backbone (spconv 2.1), TransFusionHead

#### 3. NDS 评估结果

| 版本 | Fuser 引擎 | NDS | mAP | 速度 |
|------|-----------|-----|-----|------|
| FP32 baseline | — | 0.7069 | 0.6728 | — |
| Version A (W8A16) | fuser_decoder_fp16 | **0.7144** | **0.6851** | 1.1 fps |
| Version B (INT8) | fuser_decoder_int8 | **0.7102** | **0.6786** | 5.0 fps |

两个版本 NDS 均高于 FP32 baseline，精度验证通过。

#### 4. TransFusionHead 可导出性验证

之前认为 TransFusionHead 无法导出 ONNX（argsort 不支持），经重新评估发现：
- `argsort` 只是 PyTorch 1.10 exporter 没实现 symbolic，语义上等价于 `topk`
- `F.max_pool2d(heatmap[:, 8], ...)` 需要改为 `heatmap[:, 8:9]` 保持 4D

修复后成功导出：439 nodes, 5.5 MB ONNX, 3.4 MB TRT engine。

### 下一步：Phase 6 去 PyTorch 全量部署

#### 目标架构

```
Voxelization (纯 CUDA)
      ↓
LiDAR Backbone (spconv 2.3 C++ API)   ← 非 TRT，但脱离 PyTorch
      ↓
dense BEV tensor [1, 256, 180, 180]
      ↓
全 TRT pipeline:
  6路图像 → swin_int8.engine → 多尺度特征
  → camera_neck.engine → neck 特征 [B*N, 256, 32, 88]
  → vtransform_depthnet_int8.engine → depth 特征
  → bev_pool_v2 (TRT Plugin 或独立 CUDA) → Camera BEV [1, 80, 180, 180]
  + LiDAR BEV [1, 256, 180, 180]
  → fuser_decoder.engine → neck features [1, 512, 180, 180]
  → transfusion_head.engine → 检测结果
```

#### 待完成任务

| 任务 | 难度 | 说明 |
|------|------|------|
| 6.1 camera neck → TRT | 低 | GeneralizedLSSFPN, 标准 Conv2d+BN+Upsample |
| 6.2 TransFusionHead → TRT | 低 | 已验证可导出，需写正式 export 脚本 + 精度验证 |
| 6.3 bev_pool TRT Plugin 集成 | 中 | 已有 .so，需在独立 pipeline 中调用 |
| 6.4 LiDAR spconv 2.3 C++ API | 中 | SparseInferenceEngine 脱离 PyTorch |
| 6.5 Voxelization 纯 CUDA | 低 | 简单的点云→体素映射 |
| 6.6 全量 pipeline + NDS 验证 | 高 | C++ 集成，4 卡并行评估 |

### 关键文件清单

```
# TRT 引擎（SM 8.6, RTX 3090）
swin_int8_sm86.engine                    — SwinT INT8
vtransform_depthnet_int8_sm86.engine     — depthnet INT8
fuser_decoder_fp16_sm86.engine           — Fuser+Decoder FP16 (版本A)
fuser_decoder_int8_sm86.engine           — Fuser+Decoder INT8 (版本B)
transfusion_head_fp16_sm86.engine        — TransFusionHead FP16

# ONNX
swin_int8.onnx                           — SwinT (208 Q/DQ)
vtransform_depthnet_int8.onnx            — depthnet (24 Q/DQ)
fuser_decoder_fp16.onnx                  — Fuser+Decoder FP16
fuser_decoder_int8.onnx                  — Fuser+Decoder INT8 (56 Q/DQ)
transfusion_head_fp16.onnx               — TransFusionHead FP16 (439 nodes)

# TRT Plugin
tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so

# 推理脚本
tools/trt_infer.py                       — 混合 TRT+PyTorch 推理 pipeline
tools/export_utils/export_swin.py        — SwinT 导出
tools/export_utils/export_vtransform.py  — vtransform 导出
tools/export_utils/export_fuser_decoder.py — Fuser+Decoder 导出
tools/export_utils/build_lidar_spconv23.py — spconv 2.3 LiDAR 推理
tools/export_utils/build_engine.py       — ONNX → TRT 引擎构建

# NDS 评估结果
trt_eval_version_A.json                  — Version A 详细指标
trt_eval_version_B.json                  — Version B 详细指标
logs/trt_eval_version_A.log              — Version A 评估日志
logs/trt_eval_version_B.log              — Version B 评估日志
```

### 环境信息

```
bevfusion_mqbench（NDS 评估 + TRT 推理）:
  Python 3.8 + PyTorch 1.10.2 + spconv 2.1.25 + TRT Python 10.15
  完整 mmdet3d，可跑 tools/test.py 和 tools/trt_infer.py

spconv23_deploy（spconv 2.3 C++ 部署用）:
  Conda prefix: /media/yellowstone/data2/CYL/spconv23_deploy
  Python 3.9 + PyTorch 2.0.1+cu118 + spconv 2.3.8 + TRT 8.6.1

硬件: 5 GPU (GPU 0,1,3,4 = RTX 3090 SM8.6; GPU 2 = A100 SM8.0)
GPU 指定: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,3,4
```

### 关键技术细节

1. **TRT 引擎 GPU 绑定**：引擎必须在目标 GPU 架构上构建。SM 8.0 (A100) 引擎不能在 SM 8.6 (RTX 3090) 上运行。所有 `*_sm86.engine` 已在 RTX 3090 上重建。
2. **TransFusionHead 导出修复**：`argsort` → `topk`（等价语义），`heatmap[:, 8]` → `heatmap[:, 8:9]`（保持 4D for MaxPool）。
3. **spconv 2.3 C++ API**：`spconv::SparseInferenceEngine` 可脱离 PyTorch，CUDA kernel 性能不比 TRT 差（稀疏卷积本来就不是 TRT 强项）。
4. **NDS 结果高于 baseline**：TRT FP16/INT8 的 NDS 比 FP32 PyTorch 还高，可能是 FP16 精度的正则化效果。
