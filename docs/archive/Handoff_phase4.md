## BEVFusion 全模块 TRT 部署 — 工作交接（2026-03-29）

### 总体进度

| Phase | 模块 | 状态 | 产物 |
|-------|------|------|------|
| 1 | SwinTransformer | ✅ 引擎已构建 | `swin_int8.engine` (33MB) |
| 2 | Log2 Plugin | ✅ 编译通过 | `libsparse_log2_quant_plugin.so` |
| 3 | vtransform | ✅ 引擎+Plugin | `vtransform_depthnet_int8.engine` + `libbev_pool_v2_plugin.so` |
| 4 | LiDAR backbone | ✅ spconv 2.3 FP16 验证通过 | `build_lidar_spconv23.py` (cosine=0.999994) |
| 4b | Fuser+Decoder | ✅ FP16+INT8 引擎 | `fuser_decoder_fp16.engine` / `fuser_decoder_int8.engine` |
| 4b | TransFusionHead | ❌ 无法导出 ONNX | 保持 PyTorch（argsort 不支持 ONNX） |
| 5 | 端到端集成 | 🔧 进行中 | `trt_infer.py` 待写 |

### 本次 Session 完成的工作（2026-03-29）

#### 1. libspconv.so 路线彻底排除

- 发现模型实际是 `basicblock`（有残差 Add），不是之前以为的 `conv_module`
- 修复了 `export_lidar.py`：添加 Conv+BN fusion + Add 节点 trace
- 修复后的 ONNX 结构与 NVIDIA 完全一致（21 SparseConv + 8 Add + 8 Relu）
- 但 libspconv.so 仍然段错误（闭源 `builder->build()` 崩溃）
- **结论：libspconv.so 路线放弃**

#### 2. spconv 2.3 部署环境

- 新建 conda 环境：`/media/yellowstone/data2/CYL/spconv23_deploy`
- Python 3.9 + PyTorch 2.0.1+cu118 + spconv 2.3.8 + TensorRT 8.6.1
- `build_lidar_spconv23.py`：用纯 spconv 2.3 API 重建 SparseEncoder
- 权重加载：自动处理 spconv 2.1→2.3 weight shape 转换（`[k,k,k,in,out]` → `[out,k,k,k,in]`）
- 精度验证：cosine_sim = 0.999994

#### 3. Fuser + Decoder TRT 引擎

- `export_fuser_decoder.py`：ConvFuser + SECOND + SECONDFPN 合并导出
- FP16 版本：33 nodes → `fuser_decoder_fp16.engine` (10.4 MB)
- INT8 版本：132 nodes, 56 Q/DQ → `fuser_decoder_int8.engine` (5.8 MB)
- 包含 MQBench FakeQuant → Q/DQ symbolic 转换 + shape resize workaround

#### 4. TransFusionHead

- `argsort` 不支持 ONNX opset 13，无法导出
- 保持 PyTorch 运行（仅占 2.5% 参数）

### 下一步：Phase 5 端到端集成

需要写 `tools/trt_infer.py`，在 **spconv23_deploy 环境**中运行完整推理 pipeline：

```
① 6路图像 → swin_int8.engine → SwinT features
② SwinT features → camera/neck (PyTorch) → neck features
③ neck features + 坐标 → vtransform_depthnet_int8.engine → depth features
④ depth features + 索引 → bev_pool_v2 Plugin → Camera BEV [1, 80, 180, 180]
⑤ 点云 → Voxelization (PyTorch) → voxels + coors
⑥ voxels + coors → spconv 2.3 SparseEncoder → LiDAR BEV [1, 256, 180, 180]
⑦ Camera BEV + LiDAR BEV → fuser_decoder_int8.engine → neck features [1, 512, 180, 180]
⑧ neck features → TransFusionHead (PyTorch) → 3D Bbox
```

注意事项：
- camera/neck (GeneralizedLSSFPN) 没有单独导出 TRT，需要在 PyTorch 中运行或者补导出
- Voxelization 是预处理，保持 PyTorch
- TransFusionHead 保持 PyTorch
- 两个版本差异仅在 LiDAR backbone（FP16 vs INT8+Log2）和 fuser_decoder（FP16 vs INT8）

### 两个交付版本

| 组件 | 版本 A (W8A16) | 版本 B (INT8+Log2) |
|------|---------------|-------------------|
| SwinT | swin_int8.engine (INT8) | 同左 |
| camera/neck | PyTorch FP16 | 同左 |
| vtransform depthnet | vtransform_depthnet_int8.engine | 同左 |
| bev_pool_v2 | Plugin FP16 | 同左 |
| LiDAR backbone | spconv 2.3 FP16 | spconv 2.3 FP16 + Log2 Plugin |
| fuser+decoder | fuser_decoder_fp16.engine | fuser_decoder_int8.engine |
| TransFusionHead | PyTorch | 同左 |
| 目标 NDS | ≈ 0.7009 | ≈ 0.6875 |

### 环境信息

```
统一部署环境（推荐用这个跑端到端）:
  Conda prefix: /media/yellowstone/data2/CYL/spconv23_deploy
  Python 3.9 + PyTorch 2.0.1+cu118 + spconv 2.3.8 + TensorRT 8.6.1
  激活: conda run --prefix /media/yellowstone/data2/CYL/spconv23_deploy --cwd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
  TRT lib: export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

研究环境（PTQ 量化、NDS 评估）:
  Conda: bevfusion_mqbench
  Python 3.8 + PyTorch 1.10.2 + spconv 2.1.25 + MQBench + TRT Python 10.15

通用:
  硬件: NVIDIA RTX 3090 (Ampere, SM 8.6)
  CUDA (nvcc): 11.8
  GPU 指定: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
```

### 关键文件清单

```
# TRT 引擎
swin_int8.engine                              — SwinTransformer INT8
vtransform_depthnet_int8.engine               — depthnet INT8
fuser_decoder_fp16.engine                     — Fuser+Decoder FP16 (版本A)
fuser_decoder_int8.engine                     — Fuser+Decoder INT8 (版本B)

# TRT Plugin
tools/trt_plugins/sparse_log2_quant/build/libsparse_log2_quant_plugin.so
tools/trt_plugins/bev_pool_v2/build/libbev_pool_v2_plugin.so

# ONNX
swin_int8.onnx                               — SwinT (208 Q/DQ)
vtransform_depthnet_int8.onnx                 — depthnet (24 Q/DQ)
fuser_decoder_fp16.onnx                       — Fuser+Decoder FP16
fuser_decoder_int8.onnx                       — Fuser+Decoder INT8 (56 Q/DQ)
lidar_backbone_fp16_v2.onnx                   — LiDAR ONNX (修复后，40 nodes)

# 导出脚本
tools/export_utils/export_swin.py             — SwinT 导出
tools/export_utils/export_vtransform.py       — vtransform 导出
tools/export_utils/export_lidar.py            — LiDAR ONNX 导出（已修复 Add+BN fusion）
tools/export_utils/export_fuser_decoder.py    — Fuser+Decoder 导出
tools/export_utils/build_lidar_spconv23.py    — spconv 2.3 LiDAR 推理
tools/export_utils/build_engine.py            — ONNX → TRT 引擎构建

# 验证数据
lidar_verify_tensors/                         — LiDAR 输入+输出 tensor
lidar_spconv23_output.pt                      — spconv 2.3 推理输出

# Checkpoint
pretrained/bevfusion-det.pth                  — FP32 原始权重
pretrained/ptq_minmax_model.pth               — PTQ 8/8 全量化权重
```

### 关键技术细节

1. **模型是 basicblock**：SparseEncoder 有残差 Add 连接，checkpoint 里 BN key 是 `bn1`/`bn2`
2. **spconv weight shape 转换**：2.1 是 `[k,k,k,in,out]`，2.3 是 `[out,k,k,k,in]`，需要 `permute(4,0,1,2,3)`
3. **MQBench FakeQuant → Q/DQ**：需要三步 symbolic 注册（ATen 域 + mqbench custom + PerChannel override）
4. **PTQ checkpoint 加载**：FakeQuant scale/zero_point shape 不匹配，需要 resize workaround（见 export_vtransform.py 第 276-308 行）
5. **TRT Python API 版本**：bevfusion_mqbench 有 10.15，spconv23_deploy 有 8.6.1（从本地 wheel 安装）
