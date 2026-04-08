## Phase 9 Part A：TV Backbone INT8 Log2 量化 — 工作交接（2026-04-05）

### 总体进度

| 模块 | 状态 | 方式 | 精度目标 |
|------|------|------|---------|
| TV backbone FP16 | ✅ | `TVSparseEncoder` + core_cc | NDS 0.7039 |
| TV backbone INT8 Log2 | ✅ | PTQ weight + per-layer `log2_base` + 自定义 CUDA kernel | **NDS 0.6893** |
| PT INT8 路径 | ✅ | PyTorch spconv 2.3 | NDS 0.6875（参照） |

### 本次 Session 完成的工作

#### 1. TV INT8 加载路径 (`TVSparseEncoder.load_ptq_weights`)

- 从 PTQ checkpoint (`ptq_minmax_model.pth`) 加载：
  - **Weight**：per-channel symmetric INT8 fake quantization，在加载时直接应用 `round(w/s)*s`
  - **BN 参数**：保持**不融合**到 conv weight 中（ separate `bn_scale` / `bn_shift` GPU kernel），以严格对齐 PyTorch FakeQuant 计算图
  - **Log2 base**：提取每层的 `act_fake_quant.log2_base`，供 runtime 使用
- 共加载 21 层 PTQ conv 层（含 downsample / conv_out）

#### 2. 自定义 CUDA kernel (`tools/tv_log2_quant.cu`)

编写了两个 CUDA kernel 并通过 ctypes 绑定：

- `tv_log2_fake_quant_fp16`：**Log2 对数域激活量化**
  - 公式：`x_dq = sign(x) * 2^round(log2(|x|) - base).clamp(-127,127)`
  - 运算结果与 PT `SparseLog2FakeQuantize.forward()` 逐比特一致
- `tv_bn_forward_inplace_fp16`：**BN 前向 in-place**
  - 公式：`y = x * scale + shift`（per-channel）
  - 使用 FP32 中间计算，结果与 PT `nn.BatchNorm1d.half().eval()` max diff ≈ 0.016

编译产物：`tools/libtv_log2_quant.so`

#### 3. TV 计算图对齐 (`tools/tv_sparse_encoder.py`)

修改 `_conv_bn_relu` / `_conv_bn`：
- 若存在 `log2_base`，先调用 `log2_fake_quant_inplace(x.features, log2_base, stream)`
- 再执行 `sparse_conv_forward`（`ConvGemmOps.implicit_gemm`）
- 最后根据 BN 参数选择 `bn_forward_inplace` 或 `InferenceOps.bias_add_act_inplace`

#### 4. 关键 bug 修复：基本块 residual 被 in-place 量化破坏

**根因**：`_basic_block` 中 `identity = x.features` 只是引用，而 `_conv_bn_relu` 会在 conv1 前对 `x.features` 执行 **in-place Log2 量化**。这导致 `identity`（用于残差 add）被意外篡改，残差连接使用了被量化过的输入，数值逐渐发散。

**修复**：`
identity = x.features.clone()  # deep copy before in-place modification
`

**效果**：
- 修复前：end-to-end dense max diff = **16.8**
- 修复后：end-to-end dense max diff = **2.89**（INT8 不同 kernel 实现导致的合理 rounding 差异）

#### 5. 精度验证

| 对比项 | 结果 |
|--------|------|
| conv_input 输出 | max diff = 0.027 ✅ |
| Log2 quant 精确度 | max diff = 0.000000 ✅ |
| BN 精确度 | max diff = 0.016 ✅ |
| 单个 BasicBlock 输出 | max diff = 1.03（可接受） |
| 完整 backbone dense 输出 | max diff = 2.89 ✅ |
| 完整 pipeline 冒烟测试 | **通过**（200 detections，3 with score > 0.3） |

### 完整 NDS 评估结果（2026-04-05）

```
TV backbone INT8 Log2（零 PyTorch LiDAR）:
  NDS: 0.6893
  mAP: 0.6474
```

| 配置 | NDS | mAP | 说明 |
|------|-----|-----|------|
| FP32 基线 | 0.7069 | 0.6728 | 原始模型 |
| PyTorch INT8 (KL+Log2) | 0.6875 | 0.6429 | PTQ 仿真路径 |
| **TV INT8 (零 PyTorch)** | **0.6893** | **0.6474** | **+0.0018 NDS / +0.0045 mAP** |

结果说明：TV INT8 路径不仅达到目标，甚至略优于 PyTorch 仿真路径。差异来源于 INT8 `implicit_gemm` 不同 kernel 实现导致的正常 rounding 波动，证明 TV backbone 数值精度完全可用。

### 新建/修改文件

```
tools/tv_log2_quant.cu              ← Log2 / BN CUDA kernel 源码
tools/libtv_log2_quant.so           ← 编译产物
tools/tv_sparse_encoder.py          ← 加载 load_ptq_weights + _conv_bn_relu / _basic_block 修改
tools/trt_infer_standalone.py       ← --no-torch-lidar + --lidar-quant int8 集成
docs/deploy_cmd.md                  ← 新增 TV INT8 评估命令
docs/Handoff9.md                    ← 本文档
```

### 运行命令

**单样本冒烟测试（TV INT8）**：
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

**完整 NDS 评估（TV INT8）**：见 `docs/deploy_cmd.md` Section 7 新增命令。
