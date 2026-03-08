# BEVFusion + MQBench 全模型量化工具集

> 本项目在 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion)（基于 mmdetection3d）的基础上，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具，通过**三路径量化策略**实现了 BEVFusion 全部 **8/8 子模块的 INT8 训练后量化（PTQ）**，覆盖 100% 可学习参数。

---

## 目录

- [项目简介](#项目简介)
- [核心成果](#核心成果)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [预训练模型](#预训练模型)
- [PTQ：训练后量化](#ptq训练后量化)
- [TensorRT 部署（Legacy）](#tensorrt-部署legacy)
- [Benchmark 工具](#benchmark-工具)
- [文档索引](#文档索引)
- [致谢](#致谢)

---

## 项目简介

本项目以 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) 为基础，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，实现了：

1. **全模型 INT8 PTQ（8/8 模块）**：通过三路径量化策略，对 BEVFusion 全部 8 个子模块进行 MinMax 训练后量化，覆盖 100% 可学习参数（40.9M）
2. **三路径量化策略**：针对 torch.fx 不兼容的模块，提出手动 FakeQuant 包装方案（Conv2d/Linear 包装 + SparseConv 包装），与 torch.fx 自动插桩互补
3. **逐模块消融分析**：`--skip-modules` / `--diagnose` CLI 支持，可定位各模块对量化精度的影响
4. **TRT Hybrid 部署**（Legacy）：4 个 fx 兼容模块可导出 TRT 引擎做混合推理

| 脚本 | 功能 |
|------|------|
| `tools/test.py` | **FP32 评估** — 基准精度测试 |
| `tools/train.py` | **训练** — 分布式训练（torchrun） |
| `tools/quant_ptq_minmax.py` | **PTQ** — 三路径 MinMax 校准 + 精度评估 + 消融分析 |
| `tools/quant_benchmark.py` | **Benchmark** — 模型大小与推理延迟测量 |
| `tools/trt_eval_hybrid_all.py` | **TRT Hybrid 评估** — 4 模块 TRT 导出 + Hybrid 端到端 NDS 评估 |

目标后端：**NVIDIA TensorRT INT8**（MQBench FakeQuant 仿真结果即为量化精度）

---

## 核心成果

### 全模型 INT8 PTQ 消融实验（MQBench FakeQuant 仿真，nuScenes mini 81 帧）

| 量化配置 | NDS | mAP | ΔNDS | 参数覆盖率 |
|----------|-----|-----|------|-----------|
| FP32 基线 | 0.5801 | 0.5747 | — | 0% |
| INT8 6/8（fx auto + SwinT 手动 + Head 手动） | 0.5799 | 0.5766 | −0.0002 | 87% |
| INT8 7/8（+vtransform 手动） | 0.5485 | — | −0.0316 | 93.5% |
| INT8 7/8（+lidar/backbone 手动） | 0.4803 | — | −0.0998 | 93.6% |
| **INT8 8/8（全模型）** | **0.4276** | **0.3667** | **−0.1525** | **100%** |

> 消融分析表明：camera/vtransform 和 lidar/backbone 是主要精度敏感模块。6/8 量化配置可实现近乎零损失的 INT8 量化。

---

## 环境安装

### 已验证环境

| 组件 | 版本 |
|------|------|
| Python | 3.8.20 |
| PyTorch | 1.10.2+cu113 |
| CUDA | 11.3 |
| mmcv-full | 1.4.0 |
| mmdet | 2.20.0 |
| mmdet3d | 0.0.0（本地安装） |
| MQBench | 0.0.6 |
| torchpack | 0.3.1 |
| nuscenes-devkit | 1.1.11 |
| TensorRT | 10.15.1.29|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

### 安装步骤

```bash
# 1. 安装 PyTorch（CUDA 11.3）
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 2. 安装 mmcv 和 mmdet
pip install mmcv-full==1.4.0
pip install mmdet==2.20.0

# 3. 安装其他依赖
pip install nuscenes-devkit torchpack numba

# 4. 安装本项目（含 CUDA 扩展编译）
python setup.py develop

# 5. 安装 MQBench
pip install mqbench

# 6. 安装 TensorRT
pip install tensorrt
```

### Windows 注意事项

- **编码问题**：所有 Python 命令必须设置 `$env:PYTHONUTF8="1"`，否则读取 YAML 配置时会报 GBK codec 错误
- **CUDA 编译**：如需重新编译 CUDA 扩展，需先激活 Visual Studio 编译器环境（`vcvars64.bat`）

```powershell
# Windows PowerShell 标准前置设置
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
conda activate bevfusion
```

---

## 数据准备

本项目使用 [nuScenes](https://www.nuscenes.org/) 数据集。

预处理完成后，目录结构如下：

```
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/          # 或 v1.0-mini
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
└── nuscenes_dbinfos_train.pkl
```

生成 info 文件：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes --extra-tag nuscenes
```

---

## 预训练模型

下载 BEVFusion 预训练权重到 `pretrained/` 目录：

| 文件 | 说明 |
|------|------|
| `bevfusion-det.pth` | BEVFusion 检测模型（Camera+LiDAR） |
| `swint-nuimages-pretrained.pth` | Swin Transformer Backbone 预训练权重 |

---

## PTQ：训练后量化

**脚本**：`tools/quant_ptq_minmax.py`

### 核心原理：三路径量化策略

BEVFusion 包含稀疏卷积、自定义 CUDA 算子和动态控制流，无法对全模型直接使用 `torch.fx` 追踪。本项目提出**三路径量化策略**，根据各模块的特性选择不同的 FakeQuant 插入方式，实现 8/8 模块全覆盖：

#### 路径 ①：torch.fx 自动插桩（`prepare_by_platform`）

适用于纯标准密集卷积、与符号追踪兼容的模块。MQBench 的 `prepare_by_platform` 通过 `torch.fx.symbolic_trace` 自动在计算图中插入 FakeQuantize 节点。

```
子模块 → torch.fx.symbolic_trace → FakeQuant 自动插入 → 量化仿真模型
```

**适用模块**：camera/neck、fuser、decoder/backbone、decoder/neck

#### 路径 ②：手动 FakeQuant 包装 Conv2d/Linear（`manual_quantize_nontraceable`）

适用于 torch.fx 追踪失败（动态控制流、自定义 CUDA 算子等），但内部仍为标准 `nn.Conv2d` / `nn.Linear` 层的模块。遍历模块树，将每个 Conv2d 替换为 `_QuantizedConv2d`、每个 Linear 替换为 `_QuantizedLinear`，各自包含权重 FakeQuant 和激活 FakeQuant。

```
子模块 → 遍历所有 Conv2d/Linear → 替换为 _QuantizedConv2d/_QuantizedLinear → 量化仿真模型
```

**适用模块**：camera/backbone（SwinTransformer）、camera/vtransform（DepthLSSTransform）、heads/object（TransFusionHead）

#### 路径 ③：手动 FakeQuant 包装 SparseConvolution（`manual_quantize_sparse`）

适用于 LiDAR backbone 的稀疏卷积层（spconv v1.x）。将每个 `SparseConvolution` 替换为 `_QuantizedSparseConv`（继承 `SparseModule`），对 features 张量 `[N,C]` 直接做激活量化，对 weight `[K,K,K,C_in,C_out]` 做 per-channel 量化（`ch_axis=4`）。

```
子模块 → 遍历所有 SparseConvolution → 替换为 _QuantizedSparseConv → 量化仿真模型
```

**适用模块**：lidar/backbone（SparseEncoder）

### 全模型量化模块一览（8/8）

| 模块 | 参数量 | 占比 | 量化路径 | torch.fx 失败原因 |
|------|--------|------|---------|------------------|
| camera/backbone（SwinT） | 27.6M | 67.5% | ② 手动 Conv2d/Linear | AdaptivePadding 动态控制流 |
| camera/neck（LSSFPN） | 1.6M | 3.9% | ① torch.fx 自动 | — |
| camera/vtransform（DepthLSSTransform） | 2.6M | 6.4% | ② 手动 Conv2d | bev_pool CUDA kernel |
| lidar/backbone（SparseEncoder） | 2.7M | 6.6% | ③ 手动 SparseConv | spconv 非 fx 兼容 |
| fuser（ConvFuser） | 0.8M | 1.9% | ① torch.fx 自动 | — |
| decoder/backbone（SECOND） | 4.3M | 10.5% | ① torch.fx 自动 | — |
| decoder/neck（SECONDFPN） | 0.3M | 0.7% | ① torch.fx 自动 | — |
| heads/object（TransFusionHead） | 1.0M | 2.5% | ② 手动 Conv2d/Linear | ModuleList 中 Proxy 迭代 |
| **合计** | **40.9M** | **100%** | | |

### 使用方法

```bash
# 全模型 INT8 PTQ（8/8 模块，推荐）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth

# 自定义校准 batch 数量
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --calib-batches 256

# 消融实验：跳过指定模块
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --skip-modules camera/vtransform lidar/backbone

# 诊断模式：逐模块余弦相似度分析
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --diagnose

# 仅校准并保存（跳过精度评估）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --no-eval
```

> ⚠️ PTQ checkpoint 的 `state_dict` 键名经 `torch.fx` 改造（路径 ① 的模块），**不能**用 `tools/test.py` 直接评估。精度评估请通过本脚本（不加 `--no-eval`）完成。

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calib-batches` | 128 | 校准 batch 数量，32~512 通常足够 |
| `--no-eval` | False | 是否跳过量化后的精度评估 |
| `--skip-modules` | 无 | 消融实验：跳过指定模块的量化（空格分隔模块名） |
| `--diagnose` | False | 诊断模式：输出逐模块 INT8 余弦相似度分析 |
| `--run-dir` | 自动生成 | 输出目录 |

### 输出

量化模型保存至 `runs/<run_dir>/ptq_minmax_model.pth`，包含量化后的权重与 scale/zero_point 参数。

### 消融实验结果

| 量化配置 | NDS | mAP | ΔNDS | 参数覆盖率 |
|----------|-----|-----|------|-----------|
| FP32 基线 | 0.5801 | 0.5747 | — | 0% |
| INT8 6/8（fx auto + SwinT + Head） | 0.5799 | 0.5766 | −0.0002 | 87% |
| INT8 7/8（+vtransform） | 0.5485 | — | −0.0316 | 93.5% |
| INT8 7/8（+lidar/backbone） | 0.4803 | — | −0.0998 | 93.6% |
| INT8 8/8（全模型） | 0.4276 | 0.3667 | −0.1525 | 100% |

> ✅ 6/8 模块量化（覆盖 87% 参数）可实现近乎零损失的 INT8 量化（ΔNDS = −0.0002）。
> ⚠️ camera/vtransform 和 lidar/backbone 为主要精度敏感模块，全模型量化时精度下降显著。

<details>
<summary>逐类 AP 详情（FP32 vs INT8 6/8）</summary>

| 类别 | FP32 | INT8 6/8 | 变化 |
|------|------|----------|------|
| car | 0.916 | 0.918 | +0.002 |
| truck | 0.833 | 0.840 | +0.007 |
| bus | 0.995 | 0.995 | 0.000 |
| pedestrian | 0.919 | 0.922 | +0.003 |
| motorcycle | 0.705 | 0.699 | −0.006 |
| bicycle | 0.517 | 0.518 | +0.001 |
| traffic_cone | 0.848 | 0.866 | +0.018 |

</details>

详细结果见 [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md)。

---

## Benchmark 工具

**脚本**：`tools/quant_benchmark.py`

用于报告模型大小（参数量、FP32 内存、估算 INT8 大小）及测量 GPU 推理延迟，支持 FP32 与量化模型的横向对比。

```bash
# 仅报告模型大小（无需数据集）
python tools/quant_benchmark.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint pretrained/bevfusion-det.pth \
    --size-only

# 对比 FP32 与 PTQ 模型（延迟 + 大小）
python tools/quant_benchmark.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --checkpoint pretrained/bevfusion-det.pth \
    --quant-checkpoint <ptq_checkpoint_path> \
    --num-iters 30
```

> FakeQuant 仿真在 FP32 上执行额外的 clamp/round 操作，本身有开销。真实 INT8 加速需要 TensorRT 引擎部署。

---

## TensorRT 部署（Legacy）

> 以下 TRT Hybrid 部署方案仅覆盖 4 个 torch.fx 兼容模块（路径 ① 的模块）。手动量化的模块（路径 ②③）的 TRT 部署尚未实现。

### 部署方案

MQBench 的 `convert_deploy` 和 `torch.onnx.export` 均无法直接导出 FakeQuant 模型（PyTorch 1.10 限制）。本项目采用的方案：

```
FP32 PyTorch 子模块 → torch.onnx.export → FP32 ONNX → TRT INT8/FP16 原生校准 → TRT 引擎
```

### Hybrid 推理架构

将 4 个已量化模块导出为 TRT 引擎，其余保持 PyTorch 执行：

| 组件 | 运行方式 |
|------|---------|
| camera/neck (GeneralizedLSSFPN) | → **TRT FP16/INT8 引擎** |
| fuser (ConvFuser) | → **TRT FP16/INT8 引擎** |
| decoder/backbone (SECOND) | → **TRT FP16/INT8 引擎** |
| decoder/neck (SECONDFPN) | → **TRT FP16/INT8 引擎** |
| camera/backbone (SwinTransformer) | → PyTorch FP32 |
| camera/vtransform, lidar/*, heads/* | → PyTorch FP32 |

### 全模块 TRT Hybrid 端到端结果

| 精度 | NDS | mAP | NDS 变化 | 4 模块引擎大小 | 模块压缩比 |
|------|-----|-----|---------|-------------|-----------|
| FP32 基线 | 0.5800 | 0.5744 | — | — | — |
| TRT FP32 | **0.5800** | **0.5744** | +0.0000 | 42.6 MB | — |
| TRT FP16 | **0.5795** | **0.5743** | −0.0005 | 13.5 MB | **1.96x** |
| TRT INT8 | **0.5723** | **0.5652** | −0.0077 | 7.2 MB | **3.68x** |

> 压缩比相对于 4 模块 FP32 权重（26.5 MB）计算。未量化模块仍需 129.4 MB。

各模块引擎大小（INT8）：

| 模块 | FP32 | FP16 | INT8 |
|------|------|------|------|
| camera_neck | 8,157 KB | 3,183 KB | 1,690 KB |
| fuser | 5,401 KB | 1,543 KB | 833 KB |
| dec_backbone | 28,905 KB | 8,442 KB | 4,307 KB |
| dec_neck | 1,207 KB | 692 KB | 585 KB |

### 可用脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `tools/trt_eval_hybrid_all.py` | fx 兼容模块 TRT 导出 + Hybrid 端到端 NDS 评估 | ✅ SwinT 4 模块已验证 |

### 使用方法

```bash
# 全模块 TRT INT8 端到端评估（推荐）
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --precision int8 --calib-samples 50

# 全模块 TRT FP16（精度优先，推荐部署方案）
python tools/trt_eval_hybrid_all.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --precision fp16
```

### 依赖

```bash
pip install tensorrt onnx onnxruntime
```

---

## 文档索引

| 文件 | 内容 |
|------|------|
| [docs/REPORT.md](docs/REPORT.md) | 完整技术报告（量化原理、实现细节、实验结果） |
| [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md) | 所有评测结果记录（FP32 / PTQ / TRT 精度、速度、大小） |
| [docs/PTQ_BENCHMARK_NOTES.md](docs/PTQ_BENCHMARK_NOTES.md) | 量化覆盖问题分析、TRT 导出方案 |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 可复现运行手册（所有命令 + 服务器部署） |

---

## 致谢

- **BEVFusion**：[论文](https://arxiv.org/abs/2205.13542) | [代码](https://github.com/mit-han-lab/bevfusion)
- **MQBench**：[代码](https://github.com/ModelTC/MQBench) | [文档](https://mqbench.readthedocs.io/)
- **mmdetection3d**：[代码](https://github.com/open-mmlab/mmdetection3d)
