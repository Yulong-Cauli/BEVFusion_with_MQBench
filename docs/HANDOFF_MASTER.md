# BEVFusion 全模块 INT8 量化部署 — 总交接文档（2026-04-05）

> 本文档整合了 Phase 1 ~ Phase 9 Part A 的全部 Handoff 与总结文档，是后续工作的**单一入口**。
> 若你是新接手的 Agent，**请先完整阅读本文档 1~6 节**，再动手改代码。

---

## 1. 项目定位与核心成果

### 1.1 这是什么项目？
基于 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) 的后训练量化（PTQ）研究 + TensorRT 部署项目。

- **研究目标**：在 nuScenes `val` 完整验证集（6019 帧）上，实现 BEVFusion **8/8 全模块 INT8 量化**，精度损失控制在可接受范围。
- **部署目标**：逐步去掉对 PyTorch 的依赖，最终能在 Jetson Orin 等边缘设备上运行。

### 1.2 核心算法突破

| 技术 | 解决的问题 | 效果 |
|------|-----------|------|
| **KL Observer** | vtransform `bev_pool` 长尾分布导致 98.3% range waste | −12.6% → −0.5%（+12.1 pts） |
| **Log2 对数域量化** | lidar 稀疏二模态分布，线性量化零点附近浪费 | −18.5% → −3.1%（+15.4 pts） |

**最终结果**：全模块 INT8 量化精度损失仅 **−2.7%**（NDS 0.6875 vs FP32 0.7069）。

### 1.3 部署成果（截至 2026-04-05）

| 路径 | LiDAR backbone | 量化 | NDS | mAP | 状态 |
|------|---------------|------|-----|-----|------|
| PyTorch FP16 | `SparseEncoder23` (spconv 2.3) | 无 | 0.7040 | 0.6654 | ✅ 已验证 |
| **TV FP16** | `TVSparseEncoder` (去 PyTorch) | 无 | **0.7039** | — | ✅ Phase 8 |
| PyTorch INT8 | `SparseEncoder23` (spconv 2.3) | Log2 | **0.6893** | 0.6478 | ✅ Phase 7 |
| **TV INT8** | `TVSparseEncoder` (去 PyTorch) | Log2 | **0.6893** | 0.6474 | ✅ Phase 9 Part A |

**关键结论**：
- TV FP16 与 PyTorch FP16 NDS 几乎一致（0.7039 vs 0.7040）
- TV INT8 与 PyTorch INT8 NDS **完全一致**（0.6893），mAP 差异仅 0.0004，属于正常 rounding 波动
- **Log2 去 PyTorch 部署已被严格验证可行**

---

## 2. 端到端 pipeline 架构

输入 → [Camera 分支] / [LiDAR 分支] → Fuser → Decoder → Head → 3D Bboxes

```
Camera 分支：
  6路图像 [1,6,3,256,704]
    → swin_int8_sm86.engine (TRT INT8) → 多尺度特征
    → camera_neck_int8_sm86.engine (TRT INT8) → neck features [B*N,256,32,88]
    → vtransform_depthnet_int8_sm86.engine (TRT INT8) → depth features
    → bev_pool_v2 (CUDA kernel / Plugin) → Camera BEV [1,80,180,180]

LiDAR 分支：
  点云 [N,5]
    → Voxelization (CUDA ext) → voxels + coors
    → TVSparseEncoder / SparseEncoder23 → LiDAR BEV [1,256,180,180]
      (FP16: NDS 0.7039 | INT8 Log2: NDS 0.6893)

Fusion + Decoder：
  Camera BEV + LiDAR BEV
    → fuser_decoder_int8_sm86.engine (TRT INT8) → neck features [1,512,180,180]

Head：
  → transfusion_head_int8_sm86.engine (TRT INT8) → 最终检测结果
```

### 2.1 各模块技术栈

| 模块 | 运行时 | 精度 | 备注 |
|------|--------|------|------|
| SwinTransformer | TRT Engine | INT8 | ONNX 导出 + Q/DQ |
| Camera Neck | TRT Engine | INT8 | GeneralizedLSSFPN 导出 |
| Depthnet | TRT Engine | INT8 | 含 `bev_pool_v2` Plugin 接口 |
| bev_pool_v2 | CUDA kernel / Plugin | FP16 | 预计算 rank/interval 索引 |
| Voxelization | CUDA ext (`voxel_layer`) | FP32 | `build_sp39/` 编译产物 |
| LiDAR backbone | spconv 2.3 (PyTorch/TV) | FP16/INT8 Log2 | TV 路径 = 去 PyTorch |
| Fuser+Decoder | TRT Engine | INT8/FP16 | `fuser_decoder_int8/fp16` |
| TransFusionHead | TRT Engine | INT8 | `argsort` → `topk` 修复后导出 |

---

## 3. 双环境体系（Agent 必须牢记）

### 3.1 环境 1：bevfusion_mqbench（研究/校准/导出）

```
Python 3.8 + PyTorch 1.10.2 + spconv 2.1.25 + mmcv 1.4.0 + MQBench 0.0.6
TensorRT Python API: 10.15.1.29
CUDA: 11.3 (PyTorch) / 11.8 (nvcc)
用途：PTQ 量化校准、ONNX 导出、最初的 NDS 评估
脚本：tools/trt_infer.py（混合 TRT+PyTorch， Phase 5/6 产物）
```

### 3.2 环境 2：spconv23_deploy（独立推理/部署）

```
路径：/media/yellowstone/data2/CYL/spconv23_deploy
Python 3.9 + PyTorch 2.0.1 + spconv 2.3.8 + mmcv 1.7.2 + mmdet 2.20.0
TensorRT Python API: 10.15.1.29
CUDA: 11.8
用途：Standalone 端到端推理 + NDS 评估（不依赖 cpython-38 扩展）
脚本：tools/trt_infer_standalone.py（Phase 7/8/9 核心产物）
```

### 3.3 为什么有两个环境？

- `bevfusion_mqbench` 有 MQBench（只支持 PyTorch 1.10 / Python 3.8），用于量化研究和 ONNX 导出。
- `spconv23_deploy` 是独立部署环境，Python 3.9 + spconv 2.3，兼容 TRT 10.15。
- **所有后续部署工作都在 `spconv23_deploy` 进行。** 但 ONNX 导出和 PTQ 校准若需修改 MQBench 逻辑，仍需回 `bevfusion_mqbench`。

---

## 4. 关键文件索引（按功能分类）

### 4.1 独立推理脚本（最常改）

| 文件 | 作用 | 注意 |
|------|------|------|
| `tools/trt_infer_standalone.py` | **端到端 standalone 推理**（~1300 行） | `--no-torch-lidar` 切换 TV backbone；`--lidar-quant int8` 启用 INT8 |
| `tools/tv_sparse_encoder.py` | **TV 去 PyTorch LiDAR backbone** | 核心：`TVSparseEncoder`、`sparse_conv_forward`、`load_ptq_weights` |
| `tools/tv_allocator.py` | `TVAllocator` + `TVSpconvMatmul` + cuBLAS `axpy` | 底层内存分配和 GEMM wrapper |
| `tools/tv_log2_quant.cu` | Log2 / BN 自定义 CUDA kernel | 编译产物 `libtv_log2_quant.so` |

### 4.2 导出与引擎构建工具

| 文件 | 作用 |
|------|------|
| `tools/export_utils/build_engine.py` | ONNX → TRT Engine（替代 trtexec） |
| `tools/export_utils/export_swin.py` | SwinT ONNX 导出 |
| `tools/export_utils/export_vtransform.py` | Depthnet + vtransform ONNX 导出 |
| `tools/export_utils/export_fuser_decoder.py` | Fuser+Decoder ONNX 导出 |
| `tools/export_utils/mqbench_onnx_symbolic.py` | MQBench FakeQuant → ONNX Q/DQ / custom op 注册 |

### 4.3 量化和评估脚本

| 文件 | 作用 |
|------|------|
| `tools/quant_ptq_minmax.py` | **核心 PTQ 工具**（MinMax / KL / Log2） |
| `tools/test.py` | FP32 基线评估 |
| `tools/build_cuda_ext.py` | 在 `spconv23_deploy` 中编译 `build_sp39/` 下的 CUDA 扩展 |

### 4.4 配置文件与预训练权重

| 文件 | 作用 |
|------|------|
| `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml` | 模型配置 |
| `pretrained/bevfusion-det.pth` | FP32 原始权重 |
| `pretrained/ptq_minmax_model.pth` | PTQ 8/8 全量化权重（含 `log2_base`、per-channel scale） |

### 4.5 文档索引

| 文件 | 作用 |
|------|------|
| `docs/HANDOFF_MASTER.md` | **本文档** — 总入口 |
| `docs/REPORT.md` | 完整技术研究报告（含 Round 1~9 实验结果） |
| `docs/RESULTS_LOG.md` | 实验结果时间线（按 Round 记录） |
| `docs/deploy_cmd.md` | **部署命令手册** — 所有冒烟测试和 NDS 评估命令都在这里 |
| `docs/NEXT_PLAN.md` | 长期执行计划（Phase 9 Part B 及以后） |

---

## 5. 历史里程碑（供定位问题用）

### Phase 1（2026-03-18）
SwinTransformer 静态化与 ONNX 导出。核心产出：
- `mqbench_onnx_symbolic.py`：FakeQuant → Q/DQ
- `swin_int8.onnx` / `swin_int8_sm86.engine`
- 关键技术：AdaptivePadding / attn_mask 静态化

### Phase 2（2026-03-25）
`SparseLog2Quant` TRT Plugin。手写 CUDA kernel + Plugin V2DynamicExt。

### Phase 3（2026-03-25）
`bev_pool_v2` TRT Plugin + vtransform 替换。将原始动态 `bev_pool` 替换为可 trace 的 `bev_pool_v2`（interval-sum）。

### Phase 4 / 4b（2026-03-29）
- 排除 `libspconv.so` 路线（闭源 builder 段错误）
- 确定 **spconv 2.3** 为 LiDAR backbone 部署路线
- `fuser_decoder_fp16/int8` 引擎构建
- `transfusion_head_fp16` 引擎构建（修复 `argsort` → `topk`）

### Phase 5（2026-03-29）
端到端混合 pipeline（`tools/trt_infer.py`，`bevfusion_mqbench` 环境）：
- Version A (W8A16): NDS 0.7144
- Version B (INT8): NDS 0.7102

### Phase 6（2026-03-29）
Camera Neck + TransFusionHead 也转为 TRT，全 TRT pipeline 打通。结果：
- Phase 6 (7/8 量化，LiDAR FP32): NDS 0.7040

### Phase 7（2026-03-30 ~ 04-05）
- 迁移到 `spconv23_deploy` 环境
- 新建 `trt_infer_standalone.py`
- JIT 编译 cpython-39 CUDA 扩展到 `build_sp39/`
- 修复 `WeightFakeQuantize` scale shape `[1] → [out_channels]`
- PyTorch INT8 NDS = **0.6893**（控制组验证通过）

### Phase 8（2026-04-01 ~ 04-05）
**去 PyTorch LiDAR backbone**：
- `TVSparseEncoder` + `TVAllocator` + `core_cc` API
- 修复 `implicit_gemm` CUDA error 700（根因：cumm allocator 与 TRT execution context 不兼容 → 改为 PyTorch CUDA allocator 后端）
- 修复 GC 导致的 CUDA tensor 悬空指针（`_allocators` pinning + `feature_ref`）
- 消除 CPU numpy roundtrip（fuse BN into conv + cuBLAS `axpy`）
- TV FP16 NDS = **0.7039**

### Phase 9 Part A（2026-04-05）
**TV backbone INT8 Log2**：
- 手写 CUDA kernel `tv_log2_quant.cu`（Log2 fake quant + BN in-place）
- `load_ptq_weights`：加载 per-channel INT8 weight + `log2_base`
- **关键修复**：`identity = x.features.clone()`，修复 in-place Log2 量化破坏残差连接
- TV INT8 NDS = **0.6893**

---

## 6. 血的教训：未来 Agent 必须知道的坑

### 6.1 TVAllocator 必须用 PyTorch CUDA allocator 后端

**现象**：`ConvGemmOps.implicit_gemm` CUDA error 700（illegal memory access）。

**根因**：cumm 自带的 allocator（`tv.empty/tv.zeros`）分配的 GPU 内存，在有 TRT execution context 的情况下与 spconv kernel 不兼容。

**解决方案**：`TVAllocator` 内部使用 `torch.empty()` + `tv.from_blob()`，对外仍然是 `tv.Tensor` 接口。见 `tools/tv_allocator.py`。

**警告**：未来若有人在 `tv_sparse_encoder.py` 里把任何 GPU tensor 分配改回 `tv.empty()`，**100% 复现 CUDA 700**。

### 6.2 GC 会导致 CUDA tensor 悬空指针

**现象**：TV backbone NDS 仅 0.027，第一层 conv 即输出 NaN / 极大垃圾值（`max=998`）。

**根因**：`sparse_conv_forward` 里局部创建的 `TVAllocator` 在函数返回后被 Python GC，底层 `torch.Tensor` 释放，导致 `pair_fwd` / `out_features` 悬空。

**解决方案**：
- `TVSparseConvTensor._allocators` 列表显式钉住每一层的 allocator
- `TVSparseEncoder.forward()` 接收 `feature_ref` / `coors_ref`，钉住输入 tensor

**警告**：若新增 TV layer 且忘记把 allocator append 到 `out._allocators`，会在 random sample 上偶发 NaN。

### 6.3 残差连接被 in-place 量化破坏

**现象**：TV INT8 end-to-end dense max diff = 16.8，逐层发散。

**根因**：`_basic_block` 里 `identity = x.features` 只是引用。`_conv_bn_relu` 在 conv1 前对 `x.features` **原地执行 Log2 量化**，导致 `identity`（用于残差 add）被篡改。

**修复**：`identity = x.features.clone()`

**位置**：`tools/tv_sparse_encoder.py:580`

### 6.4 spconv 2.1 → 2.3 weight shape 转换

spconv 2.1 checkpoint weight shape：`[k,k,k,in,out]`  
spconv 2.3 weight shape：`[out,k,k,k,in]`

**必须 permute**：`w.permute(4,0,1,2,3)`。  
见 `trt_infer_standalone.py` 中的 `permute_spconv_weight()`。

### 6.5 SparseSequential 不支持 `__setitem__`

**现象**：`pt_encoder.encoder_layer1[0] = new_module` 报错。

**解决方案**：通过 `parent._modules[parts[-1]] = replacement` 赋值。  
这影响到 `quantize_sparse_encoder` 中对 `SparseEncoder23` 子模块的替换。

---

## 7. 命令速查（部署用）

所有完整命令请参考 `docs/deploy_cmd.md`。下面是几条最常用的：

### TV INT8 冒烟测试（单样本）
```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
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
    --no-torch-lidar --test-single
```

### TV INT8 完整 NDS 评估
```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
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
    --no-torch-lidar \
    2>&1 | tee logs/standalone_eval_tv_int8.log
```

### PyTorch INT8 控制组（不加 `--no-torch-lidar`）
用于快速验证 PTQ checkpoint 本身是否正常：
```bash
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py ... --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth
```

---

## 8. 当前状态与下一步

### 已完成 ✅
- 8/8 全模块 INT8 量化算法（NDS 0.6875）
- spconv23_deploy 独立推理环境
- TV backbone FP16（NDS 0.7039，去 PyTorch LiDAR）
- TV backbone INT8 Log2（NDS 0.6893，去 PyTorch LiDAR）
- PyTorch spconv 2.3 INT8 控制组（NDS 0.6893）

### 进行中 / 待推进 ⏳
- **Phase 9 Part B：完全零 PyTorch（Jetson Orin）**
  - 剩余 PyTorch 依赖：TRT I/O（`torch.Tensor ↔` binding）、voxelization / bev_pool CUDA ext、DataLoader、VTransform geometry、TransFusionHead 后处理
  - **需要 Jetson Orin 设备到位后才能决定最终方案**（Python 慢拆 vs C++ 重写）

### 文档整理状态
- ✅ `docs/HANDOFF_MASTER.md` — 本文档（新建，单一入口）
- ✅ `docs/archive/` — 原始 Handoff 1~9、PHASE1_SUMMARY 等历史文档已归档至此
- ✅ `docs/REPORT.md` — 研究报告（已更新 TV INT8 结果）
- ✅ `docs/RESULTS_LOG.md` — 结果时间线（已更新）
- ✅ `docs/deploy_cmd.md` — 部署命令手册（已更新）
- ✅ `docs/NEXT_PLAN.md` — 长期计划（已更新指向本文档）

---

**最后更新**：2026-04-05  
**维护者**：按 Session 交接，以本文件为最终参考。
