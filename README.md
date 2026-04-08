# BEVFusion 全模型 INT8 量化研究

> 基于 [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) 的后量化（PTQ）研究项目，实现了 BEVFusion 全部 **8/8 模块的 INT8 量化**，精度损失仅 **−2.7%**。

---

## 核心成果

### 量化精度（完整验证集 6019 帧）

| 配置 | NDS | mAP | ΔNDS | 量化模块 | 说明 |
|------|-----|-----|------|---------|------|
| **FP32 基线** | **0.7069** | **0.6728** | **0%** | 0/8 | 原始模型 |
| **8/8 全量化（最新最优）** | **0.6875** | **0.6429** | **−2.7%** | **8/8** | vtransform KL + lidar Log2 |
| 7/8 +vt KL | 0.7033 | 0.6657 | −0.5% | 7/8 | skip lidar |
| PTQ 6/8（旧最优） | 0.7010 | 0.6614 | −0.83% | 6/8 | skip vt+lidar |
| PTQ 8/8 MinMax 基线 | 0.4562 | 0.3536 | −35.5% | 8/8 | 传统 MinMax |

### 部署精度（nuScenes val 完整验证集）

| 路径 | LiDAR backbone | 精度 | NDS | mAP | 状态 |
|------|---------------|------|-----|-----|------|
| PyTorch FP16 | `SparseEncoder23` | 无 | 0.7040 | 0.6654 | ✅ 已验证 |
| **TV FP16** | `TVSparseEncoder`（去 PyTorch） | 无 | **0.7039** | — | ✅ Phase 8 |
| PyTorch INT8 | `SparseEncoder23` | Log2 | **0.6893** | 0.6478 | ✅ Phase 7 |
| **TV INT8** | `TVSparseEncoder`（去 PyTorch） | Log2 | **0.6893** | 0.6474 | ✅ Phase 9A |

**关键突破**：

- ✅ **Log2 对数域量化**解决 lidar 稀疏激活瓶颈（−18.5% → −3.1%）
- ✅ **KL Observer**解决 vtransform 量化瓶颈（−12.6% → −0.5%）
- ✅ 精度损失从 −35.5%（MinMax）降至 −2.7%（KL+Log2）

---

## 🔬 研究亮点

### 1. Log2 对数域量化（Round 9）

**问题**：lidar 稀疏激活呈二模态分布（大多数为 0，少数有大值），传统线性量化在零点附近浪费 90%+ 的 INT8 级别。

**解决方案**：对数域量化 $y = \text{sign}(x) \cdot \log_2(|x| + 1)$

**效果**：lidar 量化从 −18.5% 改善至 −3.1%（**+15.4 pts**）

### 2. KL Observer（Round 5）

**问题**：vtransform 的 bev_pool 输出呈长尾分布，EMAMinMax 导致 98.3% range waste。

**解决方案**：KL 散度最优截断校准器，动态寻找最优截断阈值。

**效果**：vtransform 量化从 −12.6% 改善至 −0.5%（**+12.1 pts**）

---

## 核心工具

### 主要脚本

| 脚本 | 功能 | 使用频率 |
|------|------|---------|
| `tools/quant_ptq_minmax.py` | **核心 PTQ 工具**（MinMax/KL Observer/Log2） | ⭐⭐⭐ |
| `tools/test.py` | FP32 基线评估 | ⭐⭐ |

### 快速开始

```bash
# 1. FP32 基线评估
python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    pretrained/bevfusion-det.pth --eval bbox

# 2. PTQ 8/8 全量化（最新最优：vtransform KL + lidar Log2）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --calib-batches 128

# 3. PTQ 7/8 推荐配置（skip lidar，精度最高）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --vtransform-observer kl_divergence \
    --calib-batches 128
```

---

## 📚 完整文档

### 入口文档（必读）
- **[docs/HANDOFF_MASTER.md](docs/HANDOFF_MASTER.md)** — **总交接文档，单一入口**。包含完整 pipeline 架构、双环境说明、关键文件索引、血的教训、命令速查。若你是新接手的 Agent/开发者，**请先读此文档**。 

### 技术文档
- **[docs/REPORT.md](docs/REPORT.md)** — 完整技术报告（量化原理、实现细节、实验结果）
- **[docs/RESULTS_LOG.md](docs/RESULTS_LOG.md)** — 实验结果时间线记录
- **[docs/SERVER_DEPLOY.md](docs/SERVER_DEPLOY.md)** — 服务器部署手册（含所有 Round 命令）
- **[docs/deploy_cmd.md](docs/deploy_cmd.md)** — 部署命令手册（所有冒烟测试和 NDS 评估命令）
- **[docs/NEXT_PLAN.md](docs/NEXT_PLAN.md)** — 长期执行计划（Phase 9 Part B 及以后）

### 归档文档
- **[docs/MINI_DATASET_EXPERIMENTS_ARCHIVE.md](docs/MINI_DATASET_EXPERIMENTS_ARCHIVE.md)** — Mini 数据集实验归档
- **[archive/resnet50_experiments/README.md](archive/resnet50_experiments/README.md)** — ResNet-50 替换实验归档

### 清理记录
- **[docs/CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md)** — Mini 数据集清理总结
- **[docs/RESNET50_CLEANUP_SUMMARY.md](docs/RESNET50_CLEANUP_SUMMARY.md)** — ResNet-50 实验整理总结
- **[docs/TOOLS_CLEANUP_SUMMARY.md](docs/TOOLS_CLEANUP_SUMMARY.md)** — 工具清理总结
- **[docs/DOCS_MERGE_SUMMARY.md](docs/DOCS_MERGE_SUMMARY.md)** — 文档合并总结

---

## 🏗️ 项目结构

```
BEVFusion_with_MQBench/
├── tools/                        # 核心工具脚本
│   ├── quant_ptq_minmax.py      # ⭐ PTQ 量化工具
│   ├── trt_infer_standalone.py  # 端到端 standalone 推理（部署环境）
│   ├── tv_sparse_encoder.py     # TVSparseEncoder（去 PyTorch LiDAR backbone）
│   ├── tv_allocator.py          # TVAllocator + cuBLAS GEMM wrapper
│   ├── tv_log2_quant.cu         # Log2 量化 CUDA kernel
│   ├── test.py                  # FP32 评估
│   └── export_utils/            # ONNX 导出 / TRT engine 构建工具
├── configs/                     # 模型配置
│   └── nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/
├── docs/                        # 文档
│   ├── HANDOFF_MASTER.md        # ⭐ 总交接文档（单一入口）
│   ├── REPORT.md                 # 技术报告
│   ├── RESULTS_LOG.md           # 实验结果
│   ├── deploy_cmd.md            # 部署命令手册
│   ├── NEXT_PLAN.md             # 长期计划
│   ├── SERVER_DEPLOY.md          # 部署手册
│   └── archive/                  # 历史 Handoff / PHASE1 文档归档
├── pretrained/                  # 预训练权重
│   ├── bevfusion-det.pth        # FP32 原始权重
│   └── ptq_minmax_model.pth     # PTQ 8/8 全量化权重
├── artifacts/                   # 历史 ONNX / engine / 参考输出归档
├── logs/                        # 核心评估日志
├── build/                       # mmdet3d 编译产物（Python 3.8）
├── build_sp39/                  # CUDA 扩展编译产物（Python 3.9 部署环境）
├── temp/                        # 第三方源码：spconv 2.3 + NVIDIA Lidar_AI_Solution
└── archive/                     # 实验归档
    ├── resnet50_experiments/     # ResNet-50 实验
    └── ...
```

**部署 engine 文件**（位于根目录，供 `trt_infer_standalone.py` 直接使用）：
- `swin_int8_sm86.engine`
- `vtransform_depthnet_int8_sm86.engine`
- `camera_neck_int8_sm86.engine`
- `fuser_decoder_fp16_sm86.engine` / `fuser_decoder_int8_sm86.engine`
- `transfusion_head_int8_sm86.engine`

---

## 🚀 快速复现

### 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.8+ |
| PyTorch | 1.10.2+cu113 |
| CUDA | 11.3 |
| MQBench | 0.0.6 |

### 安装

```bash
# 1. 安装 PyTorch
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 2. 安装依赖
pip install mmcv-full==1.4.0 mmdet==2.200 mqbench

# 3. 安装本项目
python setup.py develop
```

### 数据集

- [nuScenes v1.0-trainval](https://www.nuscenes.org/download)（6019 验证帧）
- 预训练权重：[BEVFusion 官方权重](https://github.com/mit-han-lab/bevfusion)

### 运行示例

```bash
# 最新最优：8/8 全量化（NDS 0.6875，−2.7%）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --calib-batches 128
```

---

## 🎓 研究历程

### 实验时间线

- **Round 1-3**：早期 PTQ 基础实验
- **Round 4**：KL Observer 引入（解决 vtransform 瓶颈）
- **Round 5**：校准集修正（Train vs Val）
- **Round 6**：Per-channel 量化探索
- **Round 7**：Per-channel KL Observer
- **Round 8**：W8A16 控制实验（确认激活量化是瓶颈）
- **Round 9**：**Log2 对数域量化突破**（解决 lidar 瓶颈）

### 关键发现

1. **vtransform 量化**：KL Observer 完全解决（−12.6% → −0.5%）
2. **lidar 量化**：Log2 对数域量化有效（−18.5% → −3.1%）
3. **校准集选择**：训练集校准更可靠（避免过拟合）
4. **128 batch > 512 batch**：KL Observer 在小校准量下表现更好

---

## 🤝 致谢

- **BEVFusion**：[论文](https://arxiv.org/abs/2205.13542) | [代码](https://github.com/mit-han-lab/bevfusion)
- **MQBench**：[代码](https://github.com/ModelTC/MQBench) | [文档](https://mqbench.readthedocs.io/)
- **mmdetection3d**：[代码](https://github.com/open-mmlab/mmdetection3d)

---

## 📄 许可证

本项目遵循 [Apache 2.0 许可证](LICENSE)。

---

**项目状态**：
- ✅ **研究阶段**：8/8 全量化算法完成，精度损失仅 −2.7%（NDS 0.6875）
- ✅ **部署阶段**：TV backbone 去 PyTorch + INT8 Log2 完成（NDS 0.6893）
- ⏳ **后续**：Phase 9 Part B — 完全零 PyTorch（Jetson Orin），等待硬件到位

**最后更新**：2026-04-08
**研究方向**：后量化研究 + TensorRT 边缘部署
