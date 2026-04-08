## Phase 7：Standalone 推理脚本（去 bevfusion_mqbench 环境）— 工作交接（2026-03-30）

### 总体进度

| 模块 | 状态 | 引擎/方式 | 精度 | 环境 |
|------|------|----------|------|------|
| SwinTransformer | ✅ TRT | swin_int8_sm86.engine | INT8 | — |
| Camera Neck | ✅ TRT | camera_neck_int8_sm86.engine | INT8 | — |
| Depthnet | ✅ TRT | vtransform_depthnet_int8_sm86.engine | INT8 | — |
| bev_pool_v2 | ✅ CUDA kernel | bev_pool_ext.cpython-39.so | FP32 | spconv23_deploy |
| BEV downsample | ✅ PyTorch | Conv2d+BN (从 ckpt 加载) | FP32 | spconv23_deploy |
| LiDAR backbone | ✅ spconv 2.3 | SparseEncoder23 | FP16 | spconv23_deploy |
| LiDAR backbone (量化) | ✅ spconv 2.3 + FakeQuant | QuantizedSparseEncoder23 | Log2 INT8 | spconv23_deploy |
| Fuser+Decoder | ✅ TRT | fuser_decoder_int8_sm86.engine | INT8 | — |
| TransFusionHead | ✅ TRT | transfusion_head_int8_sm86.engine | INT8 | — |
| Voxelization | ✅ CUDA kernel | voxel_layer.cpython-39.so | FP32 | spconv23_deploy |
| NMS (iou3d) | ✅ CUDA kernel | iou3d_cuda.cpython-39.so | — | spconv23_deploy |
| roiaware_pool3d | ✅ CUDA kernel | roiaware_pool3d_ext.cpython-39.so | — | spconv23_deploy |

### 本次 Session 完成的工作

#### 1. spconv23_deploy 环境搭建

在已有的 spconv23_deploy conda 环境（Python 3.9 + PyTorch 2.0 + spconv 2.3.8）上补装：
- `nuscenes-devkit`, `torchpack`, `numba` — 数据加载和评估
- `mmcv-full==1.7.2` — Config/Runner 等基础设施
- `mmdet==2.20.0` — 数据集 builder（patch 了版本检查 `mmcv_maximum_version` → 1.8.0）
- `tensorrt-cu12==10.15.1.29` — 替换原有 TRT 8.6.1，与引擎版本匹配
- `numpy<2` — 降级解决 PyTorch 2.0 兼容性

#### 2. CUDA 扩展重新编译（cpython-39）

新建 `tools/build_cuda_ext.py`，用 `torch.utils.cpp_extension.load()` JIT 编译 4 个 CUDA 扩展到 `build_sp39/`：

| 扩展 | 源码 | 关键点 |
|------|------|--------|
| bev_pool_ext | mmdet3d/ops/bev_pool/src/ | 直接编译 |
| voxel_layer | mmdet3d/ops/voxel/src/ | 需要 `-DWITH_CUDA` 编译标志 |
| iou3d_cuda | mmdet3d/ops/iou3d/src/ | 直接编译 |
| roiaware_pool3d_ext | mmdet3d/ops/roiaware_pool3d/src/ | mmdet3d.core.bbox 依赖 |

#### 3. mmdet3d.ops 兼容性 patch

修改 `mmdet3d/ops/__init__.py`，增加 `BEVFUSION_STANDALONE` 环境变量开关：
- `BEVFUSION_STANDALONE=1` 时只导入 bev_pool、voxel、norm、roiaware_pool3d
- 跳过 ball_query、furthest_point_sample、interpolate、knn、paconv、sparse_block 等依赖 cpython-38 .so 的模块
- 原有 bevfusion_mqbench 环境不受影响（环境变量未设置时走原路径）

#### 4. 新建 `tools/trt_infer_standalone.py`（~1260 行）

完全独立的端到端推理脚本，在 spconv23_deploy 环境运行。内联了所有原本依赖 mmcv/mmdet3d 的组件：

| 组件 | 实现方式 |
|------|---------|
| TRTRunner | 复用，TRT 10.15 Python API |
| SparseEncoder23 | 从 build_lidar_spconv23.py 复制，spconv 2.3 API |
| 权重映射 | build_weight_mapping() + permute_spconv_weight() |
| Voxelization | 内联 nn.Module，调用编译好的 voxel_layer |
| bev_pool_v2 | 内联函数，调用编译好的 bev_pool_ext |
| VTransformGeometry | 内联类：get_geometry() + precompute_bev_indices() + compute_depth_map() |
| BEV downsample | 从 checkpoint 加载 Conv2d+BN 权重 |
| TransFusionBBoxCoder | 内联 decode()（~40 行纯 torch） |
| circle_nms | 复制纯 numpy 实现 |
| nms_gpu | 调用编译好的 iou3d_cuda |
| SimpleLiDARBox | 简化版，只需 .tensor 和 .bev 属性 |
| SparseLog2FakeQuantize | 内联（~40 行纯 torch，不依赖 MQBench） |
| WeightFakeQuantize | 内联 per-channel symmetric INT8 |
| QuantizedSparseConv23 | spconv 2.3 SparseConv + FakeQuant 包装 |
| 数据加载 | 通过 mmdet3d.datasets（STANDALONE 模式） |
| NDS 评估 | dataset.evaluate() |

#### 5. 冒烟测试通过

```
Input shapes: img=[1, 6, 3, 256, 704], points=[252768, 5]
Inference time: 1340.6 ms
Detections: 200 total, 3 with score > 0.3
```

5 个 TRT INT8 引擎 + spconv 2.3 LiDAR backbone (FP16) 端到端跑通。

### NDS 评估结果

| 配置 | NDS | mAP | 环境 | 说明 |
|------|-----|-----|------|------|
| FP32 baseline | 0.7069 | 0.6728 | bevfusion_mqbench | 原始模型 |
| Phase 6 (LiDAR FP32) | 0.7040 | 0.6654 | bevfusion_mqbench | trt_infer.py |
| Phase 7 standalone (LiDAR FP16) | **待跑** | **待跑** | spconv23_deploy | 预期 ≈ 0.7040 |
| Phase 7 standalone (LiDAR Log2 INT8) | **待跑** | **待跑** | spconv23_deploy | 预期 ≈ 0.6875 |

### 待完成

#### 1. 完整 NDS 评估（spconv23_deploy 环境）

命令见 `docs/deploy_cmd.md` 第 6 节。需要 GPU 空闲时跑：

```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# LiDAR FP16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    2>&1 | tee logs/standalone_eval_fp16.log

# LiDAR Log2 INT8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    2>&1 | tee logs/standalone_eval_int8.log
```

#### 2. 真正的零 PyTorch 部署（后续）

当前 standalone 脚本仍依赖 PyTorch 2.0（spconv 2.3 的运行时依赖）。要在 Jetson Orin 上实现零 PyTorch：

- **方案 A**：spconv 2.3 C++ inference API — spconv 提供了 `spconv::SparseConvTensor` C++ 类，可以不依赖 PyTorch 运行。需要写 C++ 推理程序。

建议优先级：A > B > C。方案 A 最直接，spconv 2.3 的 C++ API 已经验证过（Phase 4 cosine=0.999994）。

#### 3. BEV downsample 也可以做成 TRT engine

当前 BEV downsample 是 PyTorch Conv2d+BN，可以导出 ONNX → TRT engine。这是一个简单的 3 层 Conv2d 网络，导出很容易。但优先级低，因为它的计算量很小。

### 关键文件

```
# Phase 7 新建
tools/trt_infer_standalone.py              — 独立推理脚本（spconv23_deploy 环境）
tools/build_cuda_ext.py                    — CUDA 扩展编译脚本
build_sp39/                                — 编译好的 cpython-39 CUDA 扩展
    bev_pool_ext.so
    voxel_layer.so
    iou3d_cuda.so
    roiaware_pool3d_ext.so

# Phase 7 修改
mmdet3d/ops/__init__.py                    — 增加 BEVFUSION_STANDALONE 开关
mmdet (site-packages)                      — patch mmcv_maximum_version → 1.8.0
docs/deploy_cmd.md                         — 新增第 6 节 standalone 命令

# 环境
/media/yellowstone/data2/CYL/spconv23_deploy/  — conda 环境
    Python 3.9 + PyTorch 2.0 + spconv 2.3.8 + TRT 10.15
    mmcv-full 1.7.2 + mmdet 2.20.0
    nuscenes-devkit + torchpack + numba
```

### 两个环境对比

| | bevfusion_mqbench | spconv23_deploy |
|---|---|---|
| Python | 3.8 | 3.9 |
| PyTorch | 1.10.2 | 2.0.1 |
| spconv | 2.1 (mmdet3d 内置) | 2.3.8 |
| TRT Python | 10.15.1.29 | 10.15.1.29 |
| mmcv | 1.4.0 | 1.7.2 |
| mmdet | 2.20.0 | 2.20.0 |
| MQBench | 0.0.6 | 无（FakeQuant 内联实现） |
| 推理脚本 | tools/trt_infer.py | tools/trt_infer_standalone.py |
| CUDA ext | cpython-38 (setup.py) | cpython-39 (JIT, build_sp39/) |

### 量化方案说明

与 Phase 6 完全一致，standalone 脚本内联了所有量化逻辑：
- 5 个 TRT 引擎的 INT8 量化参数来自 `pretrained/ptq_minmax_model.pth`
- LiDAR backbone 量化通过 `--lidar-quant int8` 启用：
  - 权重：per-channel symmetric INT8（WeightFakeQuantize，从 PTQ ckpt 读 scale/zero_point）
  - 激活：Log2 对数域量化（SparseLog2FakeQuantize，从 PTQ ckpt 读 log2_base）
  - 不依赖 MQBench，纯 torch 实现
