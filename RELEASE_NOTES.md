# BEVFusion + MQBench v1.0 发布说明

## 🎉 版本亮点

**首个实现 BEVFusion 8/8 全模块 INT8 量化的开源项目**

- ✅ **精度损失仅 −2.7%**（NDS 0.6875 vs FP32 0.7069）
- ✅ **100% 参数覆盖率**（全部 40.9M 参数）
- ✅ **创新算法**：Log2 对数域量化 + KL Observer
- ✅ **完整验证集评估**（6019 帧，nuScenes v1.0-trainval）

---

## 📊 核心成果

### 量化精度对比

| 配置 | NDS | mAP | ΔNDS | 量化模块 | 说明 |
|------|-----|-----|------|---------|------|
| FP32 基线 | 0.7069 | 0.6728 | 0% | 0/8 | 原始模型 |
| **8/8 全量化（v1.0）** | **0.6875** | **0.6429** | **−2.7%** | **8/8** | vtransform KL + lidar Log2 |
| 7/8 +vt KL | 0.7033 | 0.6657 | −0.5% | 7/8 | skip lidar |
| PTQ 8/8 MinMax 基线 | 0.4562 | 0.3536 | −35.5% | 8/8 | 传统 MinMax |

### 算法创新

#### 1. Log2 对数域量化（Round 9）
- **解决问题**：lidar 稀疏激活二模态分布（大多数为 0，少数有大值）
- **改进效果**：lidar 量化从 −18.5% → −3.1%（**+15.4 pts**）
- **技术原理**：对数域变换 $y = \text{sign}(x) \cdot \log_2(|x| + 1)$

#### 2. KL Observer（Round 5）
- **解决问题**：vtransform bev_pool 长尾分布导致的 98.3% range waste
- **改进效果**：vtransform 量化从 −12.6% → −0.5%（**+12.1 pts**）
- **技术原理**：KL 散度最优截断校准器，动态寻找最优截断阈值

#### 3. 校准集修正（Round 5）
- **问题**：之前使用验证集校准导致过拟合
- **修正**：改用训练集校准（`cfg.data.train` + `test_mode=True`）
- **影响**：方法论更正确，结论更可靠

---

## 🔧 工具特性

### 核心脚本

| 脚本 | 功能 | 关键参数 |
|------|------|---------|
| `tools/quant_ptq_minmax.py` | **核心 PTQ 工具** | `--vtransform-observer kl_divergence`, `--calib-batches 128` |
| `tools/test.py` | FP32 基线评估 | 标准评估流程 |
| `tools/quant_benchmark.py` | 性能基准测试 | 模型大小、推理延迟 |
| `tools/diag_lidar_distribution.py` | lidar 分布诊断 | 稀疏激活分析 |

### 快速开始

```bash
# 8/8 全量化（最新最优）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --act-observer kl_divergence \
    --calib-batches 128

# 7/8 推荐配置（精度最高）
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --skip-modules lidar/backbone \
    --vtransform-observer kl_divergence \
    --calib-batches 128
```

---

## 📚 完整文档

### 技术文档
- **[docs/REPORT.md](docs/REPORT.md)** — 完整技术报告（量化原理、实现细节、实验结果）
- **[docs/RESULTS_LOG.md](docs/RESULTS_LOG.md)** — 实验结果时间线（Round 1-9）
- **[docs/SERVER_DEPLOY.md](docs/SERVER_DEPLOY.md)** — 服务器部署手册（含完整命令）

### 归档文档
- **[docs/MINI_DATASET_EXPERIMENTS_ARCHIVE.md](docs/MINI_DATASET_EXPERIMENTS_ARCHIVE.md)** — Mini 数据集实验归档
- **[archive/resnet50_experiments/README.md](archive/resnet50_experiments/README.md)** — ResNet-50 替换实验归档

### 清理记录
- **[docs/CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md)** — Mini 数据集清理总结
- **[docs/RESNET50_CLEANUP_SUMMARY.md](docs/RESNET50_CLEANUP_SUMMARY.md)** — ResNet-50 实验整理总结
- **[docs/TOOLS_CLEANUP_SUMMARY.md](docs/TOOLS_CLEANUP_SUMMARY.md)** — 工具清理总结

---

## 🏷️ 版本信息

- **版本号**：v1.0
- **发布日期**：2026-03-17
- **Git Commit**：40d64ce
- **Git Tag**：v1.0
- **分支**：exp/lss-kl-divergence-calibration

---

## 🎯 核心价值

1. **首个 8/8 全量化 BEVFusion**：开源界首个完整量化的 BEVFusion 实现
2. **可复现的研究结果**：完整实验记录、服务器部署命令、精度基准
3. **创新的量化方法**：Log2 对数域量化 + KL Observer，可迁移到其他稀疏模型
4. **严谨的研究方法**：校准集修正、完整验证集评估、消融分析

---

## 🚀 快速开始

### 环境安装

```bash
# 1. 安装 PyTorch
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 2. 安装依赖
pip install mmcv-full==1.4.0 mmdet==2.2000 mqbench

# 3. 安装本项目
python setup.py develop
```

### 数据集准备

- 下载 [nuScenes v1.0-trainval](https://www.nuscenes.org/nuscenes)（6019 验证帧）
- 下载 [BEVFusion 预训练权重](https://github.com/mit-han-lab/bevfusion)

### 运行量化

```bash
python tools/quant_ptq_minmax.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --load-from pretrained/bevfusion-det.pth \
    --vtransform-observer kl_divergence \
    --calib-batches 128
```

预期结果：NDS ≈ 0.6875，mAP ≈ 0.6429

---

## 🤝 引用

如果您使用了本项目的代码或方法，请引用：

```bibtex
@software{bevfusion_mqbench_v1,
  title={{BEVFusion}+{MQBench}: Full Model INT8 Post-Training Quantization},
  author={Research Group},
  year={2026},
  version={1.0},
  url={https://github.com/Yulong-Cauli/BEVFusion_with_MQBench},
  note={8/8 modules quantized, NDS 0.6875 (-2.7\% vs FP32)}
}
```

---

## 📧 联系方式

- **GitHub**：https://github.com/Yulong-Cauli/BEVFusion_with_MQBench
- **Issues**：https://github.com/Yulong-Cauli/BEVFusion_with_MQBench/issues

---

**v1.0 发布总结**：

✅ **研究目标完成**：实现 BEVFusion 8/8 全模块 INT8 量化
✅ **精度目标达成**：精度损失仅 −2.7%，远超 MinMax 基线（−35.5%）
✅ **创新算法验证**：Log2 + KL Observer 方法有效且可复现
✅ **完整文档交付**：技术报告、实验记录、部署手册齐全
✅ **版本管理规范**：v1.0 标签，代码已提交到 GitHub

🎉 **这是 BEVFusion 量化研究的重要里程碑！**
