# BEVFusion + MQBench 量化工具集

> 本项目在 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion)（基于 mmdetection3d）的基础上，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具，实现了面向 TensorRT INT8 后端的**训练后量化（PTQ）**。

---

## 目录

- [项目简介](#项目简介)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [预训练模型](#预训练模型)
- [PTQ：训练后量化](#ptq训练后量化)
- [量化精度验证结果](#量化精度验证结果)
- [Benchmark 工具](#benchmark-工具)
- [TensorRT 部署](#tensorrt-部署)
- [文档索引](#文档索引)
- [致谢](#致谢)

---

## 项目简介

本项目以 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) 为基础，集成了 [MQBench](https://github.com/ModelTC/MQBench) 量化工具库，新增以下量化脚本：

| 脚本 | 功能 |
|------|------|
| `tools/quant_ptq_minmax.py` | **PTQ** — MinMax 校准 + 精度评估 |
| `tools/quant_benchmark.py` | **Benchmark** — 模型大小与推理延迟测量 |
| `tools/trt_export_fuser.py` | **TRT 导出** — ConvFuser ONNX → TensorRT 引擎 |

目标后端：**NVIDIA TensorRT INT8**

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
| TensorRT | 10.15.1.29（可选，仅 TRT 导出需要） |
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

# 6.（可选）安装 TensorRT
pip install tensorrt
```

### Windows 注意事项

- **编码问题**：所有 Python 命令必须设置 `$env:PYTHONUTF8="1"`，否则读取 YAML 配置时会报 GBK codec 错误
- **CUDA 编译**：如需重新编译 CUDA 扩展，需先激活 Visual Studio 编译器环境（`vcvars64.bat`）
- 详细的环境修复记录见 [CLIlog.md](CLIlog.md)

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

### 原理

PTQ 无需重新训练，仅需少量校准数据即可确定量化参数。本项目采用 **MinMax 校准**策略：

1. 对可量化子模块逐一调用 `prepare_by_platform`，插入 FakeQuantize 节点
2. `enable_calibration`：在校准数据上前向推理，记录各层激活值的 min/max
3. `enable_quantization`：冻结 Observer，激活 FakeQuant，进入量化推理模式

### 选择性量化策略

BEVFusion 包含自定义 CUDA 算子和动态控制流，不能对全模型直接量化，因此采用**选择性量化**：

**✅ 已成功量化（4/6）**：

| 子模块 | 类型 | 说明 |
|--------|------|------|
| `decoder.backbone` | SECOND | 纯静态 Conv2d，fx 直接可追踪 |
| `decoder.neck` | SECONDFPN | 已修复 `len()` 断言 + mmcv 包装层 |
| `encoders.camera.neck` | GeneralizedLSSFPN | 已修复 `len()` 调用 + mmcv 包装层 |
| `fuser` | ConvFuser | 已修复 `torch.cat(Proxy)` 问题 |

**❌ 暂不支持（2/6）**：

| 子模块 | 类型 | 失败原因 |
|--------|------|----------|
| `encoders.camera.backbone` | SwinTransformer | 含 `if tensor_value:` 动态控制流 |
| `heads.object` | TransFusionHead | Proxy 对象被 for 循环迭代 |

**⊘ 设计跳过**（非标准卷积，不适合 PTQ）：

| 子模块 | 跳过原因 |
|--------|----------|
| `encoders.camera.vtransform` | 内含 `bev_pool`（QuickCumsumCuda）自定义 CUDA 算子 |
| `encoders.lidar.voxelize` | 点云体素化预处理，非神经网络层 |
| `encoders.lidar.backbone` | 稀疏卷积（SparseEncoder），FakeQuant 节点无法插入 |

### 使用方法

```bash
# 校准 + 精度评估（推荐，约 3 分钟）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth

# 自定义校准 batch 数量
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --calib-batches 256

# 仅校准并保存（跳过精度评估）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load_from pretrained/bevfusion-det.pth \
    --no-eval
```

> ⚠️ PTQ checkpoint 的 `state_dict` 键名经 `torch.fx` 改造，**不能**用 `tools/test.py` 直接评估。精度评估请通过本脚本（不加 `--no-eval`）完成。

### 输出

量化模型保存至 `runs/<run_dir>/ptq_minmax_model.pth`，包含量化后的权重与 scale/zero_point 参数。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calib-batches` | 128 | 校准 batch 数量，32~512 通常足够 |
| `--no-eval` | False | 是否跳过量化后的精度评估 |
| `--run-dir` | 自动生成 | 输出目录 |

---

## 量化精度验证结果

在 nuScenes v1.0-mini 验证集（81 样本）上，使用 128 batch 校准的完整 NDS 评估：

| 指标 | FP32 基线 | PTQ 4/6（MinMax） | 变化 |
|------|----------|------------------|------|
| **NDS** | 0.5801 | **0.5810** | **+0.0009**（无损） |
| **mAP** | 0.5742 | **0.5759** | **+0.0017**（无损） |

> ✅ 最朴素的 MinMax PTQ 在 4/6 模块量化后实现了**零精度损失**。

<details>
<summary>逐类 AP 详情</summary>

| 类别 | FP32 | PTQ 4/6 | 变化 |
|------|------|---------|------|
| car | 0.916 | 0.918 | +0.002 |
| truck | 0.833 | 0.840 | +0.007 |
| bus | 0.995 | 0.995 | 0.000 |
| pedestrian | 0.919 | 0.922 | +0.003 |
| motorcycle | 0.705 | 0.699 | −0.006 |
| bicycle | 0.517 | 0.518 | +0.001 |
| traffic_cone | 0.848 | 0.866 | +0.018 |

</details>

### ConvFuser TensorRT 导出 PoC

ConvFuser 单模块 TRT 导出验证（RTX 4060 Laptop，TensorRT 10.15）：

| 精度 | 延迟 | 加速比 | 引擎大小 | 压缩比 |
|------|------|--------|---------|--------|
| PyTorch FP32 | 5.08 ms | 1.00x | — | — |
| TRT FP32 | 4.02 ms | 1.27x | 5385 KB | 1.00x |
| TRT FP16 | 1.44 ms | **3.54x** | 1543 KB | 3.49x |
| TRT INT8 | 0.75 ms | **6.81x** | 832 KB | **6.48x** |

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

实测结果（RTX 4060 Laptop，nuScenes v1.0-mini）：

| 指标 | FP32 | PTQ（FakeQuant 仿真） |
|------|------|----------------------|
| 参数量 | 39.80 M | 39.81 M |
| .pth 文件大小 | 156.13 MB | 156.31 MB |
| 均值延迟 | 389 ms | 408 ms |
| 理论 INT8 部署大小 | — | ~39 MB（需 TRT 导出） |

> FakeQuant 仿真在 FP32 上执行额外的 clamp/round 操作，本身有开销。真实 INT8 加速需要 TensorRT 引擎部署。

---

## TensorRT 部署

### 已验证方案

MQBench 的 `convert_deploy` 和 `torch.onnx.export` 均无法直接导出 FakeQuant 模型（PyTorch 1.10 缺少自定义 op 的 ONNX symbolic）。实际可行的方案：

```
FP32 PyTorch 模型 → torch.onnx.export → FP32 ONNX → TRT IInt8MinMaxCalibrator → INT8 引擎
```

参考脚本：`tools/trt_export_fuser.py`（ConvFuser PoC，验证 6.81x INT8 加速）。

### 依赖

```bash
pip install tensorrt onnx onnxruntime
```

### 端到端 Hybrid 推理（开发中）

计划将 4 个已量化子模块分别导出为 TRT INT8 引擎，其余保持 PyTorch 执行：

| 组件 | 运行方式 |
|------|---------|
| camera/neck, fuser, decoder/backbone, decoder/neck | → TRT INT8 引擎 |
| camera/backbone, camera/vtransform, lidar/*, heads/* | → PyTorch FP32 |

---

## 文档索引

| 文件 | 内容 |
|------|------|
| [docs/RESULTS_LOG.md](docs/RESULTS_LOG.md) | 所有评测结果记录（FP32 / PTQ 精度、速度、大小） |
| [docs/PTQ_BENCHMARK_NOTES.md](docs/PTQ_BENCHMARK_NOTES.md) | 量化覆盖问题分析、TRT 导出方案、开放问题 |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 可复现运行手册（所有命令） |
| [CLIlog.md](CLIlog.md) | 完整历史修复记录（环境配置、bug 修复） |
| [AGENTS.md](AGENTS.md) | Agent 工作说明（环境约束、已知问题） |

---

## 致谢

- **BEVFusion**：[论文](https://arxiv.org/abs/2205.13542) | [代码](https://github.com/mit-han-lab/bevfusion)
- **MQBench**：[代码](https://github.com/ModelTC/MQBench) | [文档](https://mqbench.readthedocs.io/)
- **mmdetection3d**：[代码](https://github.com/open-mmlab/mmdetection3d)
