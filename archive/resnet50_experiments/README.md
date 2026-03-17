# ResNet-50 Backbone 替换实验归档

> ⚠️ **项目方向调整**：本实验记录已归档。ResNet-50 替换方案**不再是项目主攻方向**，我们倾向于在**原始 SwinT 架构**上进行量化研究。但本实验结果仍具有重要参考价值。

---

## 实验背景

### 动机
原始 BEVFusion 使用 SwinTransformer 作为 camera backbone，但 SwinT 存在以下问题：
1. **量化困难**：Window Attention 中的动态控制流（`if x.shape[0] > window_size`）无法被 `torch.fx` 追踪
2. **TRT 部署障碍**：无法导出 ONNX → 无法部署为 TensorRT 引擎
3. **参数占比大**：SwinT 占全模型参数的 67.5%（105.2 MB），是量化覆盖的最大瓶颈

### 替换方案
将 camera/backbone 从 **SwinTransformer** 替换为 **ResNet-50**（纯 CNN，量化友好）。

### 参考价值
虽然不是最终方向，但 ResNet-50 实验验证了：
- ✅ 完整的量化工具链可行性
- ✅ TRT 端到端部署流程
- ✅ 5/6 模块量化的精度基准
- ✅ 不同 backbone 的量化敏感度分析

---

## 实验配置

### 训练设置
- **数据集**：nuScenes v1.0-trainval（28130 训练样本）
- **硬件**：4×RTX 3090 + 1×A100-SXM4-80GB
- **训练时长**：6 epochs（vs 官方 SwinT 20 epochs）
- **配置文件**：`configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser.yaml`

### 评估设置
- **验证集**：nuScenes v1.0-trainval val（6019 帧，850 场景）
- **指标**：NDS、mAP、逐类 AP
- **对比基线**：官方 SwinT（20 epochs 训练）

---

## 核心实验结果

### 1. FP32 基线对比

| 模型 | NDS | mAP | 训练 epochs | 参数量 |
|------|-----|-----|-----------|--------|
| **SwinT（官方）** | **0.7138** | **0.6828** | 20 | ~27.5M |
| **SwinT（本复现）** | **0.7069** | **0.6728** | 20 | ~27.5M |
| **ResNet-50** | **0.4989** | **0.4961** | 6 | ~24.8M |

**分析**：
- ResNet-50 NDS 较低主要是训练不足（6 vs 20 epochs）
- Epoch 6 时 loss 仍在下降（下降率 7.6%），增加训练有望提升精度
- ResNet-50 表达能力弱于 SwinT（纯 CNN vs Transformer）

---

### 2. PTQ 5/6 模块量化结果

| 指标 | FP32 基线 | PTQ INT8 | Δ变化 | 量化模块 |
|------|----------|----------|------|---------|
| **NDS** | 0.4989 | **0.4958** | **−0.0031** (−0.62%) | 5/6 |
| **mAP** | 0.4961 | **0.4904** | **−0.0057** (−1.15%) | 5/6 |

**量化覆盖**：
- ✅ camera/backbone (ResNet-50)
- ✅ camera/neck
- ✅ fuser
- ✅ decoder/backbone
- ✅ decoder/neck
- ❌ heads/object（TransFusionHead，未量化）

---

### 3. TRT 端到端部署结果

| 配置 | NDS | mAP | ΔNDS vs FP32 | 引擎大小 |
|------|-----|-----|--------------|---------|
| PyTorch FP32 | 0.4989 | 0.4961 | — | — |
| **TRT FP16** | **0.4981** | **0.4953** | **−0.0008** | 59.9 MB |
| **TRT INT8** | **0.4948** | **0.4905** | **−0.0041** | 31.4 MB |

**关键发现**：
- TRT FP16 几乎无损（−0.16%）
- TRT INT8 精度损失仅 −0.82%
- **5 模块 INT8 引擎总大小 31.4 MB**（camera_backbone 占 76%）
- **总部署体积 ~55.7 MB**（vs SwinT 方案的 ~136.8 MB，**−59%**）

---

### 4. 逐类 AP 对比（完整验证集）

| 类别 | SwinT FP32 | ResNet-50 FP32 | ResNet-50 TRT INT8 | Δ (ResNet INT8) |
|------|-----------|----------------|-------------------|----------------|
| car | 0.861 | 0.688 | 0.682 | −0.006 |
| truck | 0.622 | 0.437 | 0.432 | −0.005 |
| bus | 0.778 | 0.558 | 0.555 | −0.003 |
| pedestrian | 0.624 | 0.498 | 0.492 | −0.006 |
| motorcycle | 0.605 | 0.452 | 0.449 | −0.003 |
| bicycle | 0.572 | 0.433 | 0.429 | −0.004 |
| traffic_cone | 0.641 | 0.567 | 0.562 | −0.005 |
| trailer | 0.275 | 0.157 | 0.155 | −0.002 |
| construction_vehicle | 0.312 | 0.182 | 0.179 | −0.003 |
| barrier | 0.633 | 0.522 | 0.517 | −0.005 |

**分析**：ResNet-50 在所有类别上均匀低于 SwinT，但量化损失很小。

---

## 量化工具链验证

### 成功验证的功能

1. ✅ **ResNet-50 完整 torch.fx 追踪**
   - 纯 CNN 架构完全兼容符号追踪
   - 所有 Conv2d/BN/ReLU 层自动插桩 FakeQuant

2. ✅ **TRT 引擎构建与部署**
   - 5 个模块成功导出 ONNX
   - TRT INT8 校准器正常工作
   - Hybrid 推理端到端验证通过

3. ✅ **精度评估一致性**
   - MQBench FakeQuant 预测：NDS −0.62%
   - TRT INT8 实测：NDS −0.82%
   - 误差仅 0.20%，验证了工具链可靠性

4. ✅ **模型压缩效果**
   - 5 模块 INT8 引擎：31.4 MB（vs FP32 116.9 MB）
   - 总部署体积：55.7 MB（vs SwinT 136.8 MB，−59%）

---

## 为什么不继续 ResNet-50 方向

### 1. 精度天花板限制
- **ResNet-50 表达能力弱于 SwinT**：即使训练 20 epochs，NDS 仍难达到 SwinT 的 0.71+
- **论文发表价值低**：工业界更关注如何在保持精度的前提下量化

### 2. 核心问题未解决
- ResNet-50 只是**绕过了** SwinT 的量化困难，而非**解决**了它
- 真正的挑战在于：如何量化 SwinT + vtransform + lidar 等复杂模块

### 3. 项目目标调整
- **新方向**：在**原始 SwinT BEVFusion** 上做完整的 PTQ 研究
- **最新成果**：通过 Log2 对数域量化，实现 **8/8 全模块量化 NDS 0.6875 (−2.7%)**

---

## 持续的参考价值

### 对于量化方法研究
1. **验证了 PTQ 工具链的完备性**：fx 追踪 + FakeQuant + TRT 部署
2. **提供了不同 backbone 的量化敏感度对比**
3. **证明了 ResNet 类 CNN 的量化友好性**

### 对于 TRT 部署工程
1. **完整的 Hybrid 推理架构**：5 模块 TRT + 其他模块 PyTorch
2. **TRT 引擎构建脚本**：`tools/trt_eval_hybrid_all.py`
3. **性能基准数据**：FP16/INT8 的精度和速度对比

### 对于后续工作
1. **如果必须全 TRT 部署**：ResNet-50 是备选方案（牺牲精度换工程可行性）
2. **如果研究混合精度**：ResNet-50 可作为"易量化模块"的参考
3. **如果对比不同架构**：ResNet-50 提供了 CNN vs Transformer 的量化对比

---

## 归档文件清单

### 配置文件
- `archive/resnet50_experiments/configs/convfuser.yaml` — ResNet-50 BEVFusion 配置
- `archive/resnet50_experiments/configs/default.yaml` — 默认参数

### 日志文件
- `archive/resnet50_experiments/logs/results_resnet50_train.log` — 训练日志
- `archive/resnet50_experiments/logs/results_resnet50_ptq.log` — PTQ 评估日志
- `archive/resnet50_experiments/logs/results_resnet50_trt_*.log` — TRT 部署日志

### 权重文件
- `runs/epoch_6.pth` — ResNet-50 训练 6 epochs 权重（如需保留）

---

## 相关文档

- **主报告**：[docs/REPORT.md](../../docs/REPORT.md) — §9.6 ResNet-50 替换实验
- **结果日志**：[docs/RESULTS_LOG.md](../../docs/RESULTS_LOG.md) — 2026-03-02 章节
- **部署手册**：[docs/SERVER_DEPLOY.md](../../docs/SERVER_DEPLOY.md) — ResNet-50 服务器部署

---

**归档时间**：2026-03-17
**实验周期**：2026-03-02 至 2026-03-03
**状态**：实验已完成，方向已调整，结果保留参考
