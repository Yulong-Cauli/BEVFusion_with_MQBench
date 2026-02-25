# 学习计划与 Jetson 部署指南

---

## 一、Jetson 平台选型

### Jetson Nano 能跑 BEVFusion 吗？

**完全不行。** 差距非常大：

| 资源 | Jetson Nano | BEVFusion 最低需求 |
|------|------------|-------------------|
| GPU | Maxwell 128核, CC 5.3 | 至少 CC 7.2 (Volta) |
| 显存 | 4 GB（共享） | ≥8 GB |
| CUDA | 最高 10.2 | 11.3+ |
| TensorRT | 最高 8.0 | 8.5+（CUDA-BEVFusion），10.15（本项目） |
| 算力 | 0.47 TFLOPS FP16 | NVIDIA 在 Orin 上才跑到 25 FPS |

### Jetson 系列选型表

| 平台 | CC | GPU 算力(FP16) | 内存 | BEVFusion 可行性 |
|------|-----|---------------|------|-----------------|
| **Nano** | 5.3 | 0.47 TFLOPS | 4 GB | ❌ 完全不行 |
| **TX2** | 6.2 | 1.3 TFLOPS | 8 GB | ❌ 算力和内存都不够 |
| **Xavier NX** | 7.2 | 21 TOPS INT8 | 8/16 GB | 🟡 极度勉强，FPS < 2 |
| **AGX Xavier** | 7.2 | 32 TOPS INT8 | 32 GB | 🟡 可跑但慢，FPS ≈ 3-5 |
| **Orin NX** | 8.7 | 100 TOPS INT8 | 8/16 GB | 🟢 可行，需 INT8 全量化 |
| **AGX Orin 64GB** | 8.7 | 275 TOPS INT8 | 64 GB | ✅ **NVIDIA 官方验证**，25 FPS |

**结论：最低 Orin NX（16GB 版），推荐 AGX Orin。**

NVIDIA 自己的 CUDA-BEVFusion 在 AGX Orin 上验证：FP16 18 FPS，INT8 25 FPS。

---

## 二、仿真方案（买到 Jetson 之前）

**不存在真正的 Jetson GPU 仿真器**，GPU 计算无法在 x86 上模拟。但有替代思路：

### 方案 A：用桌面 GPU 估算性能（最实用）

RTX 4060 Laptop 可以提供**精度验证**（NDS/mAP 数字是硬件无关的），然后用算力比换算延迟：

```
RTX 4060 Laptop FP16: ~178 TFLOPS
AGX Orin FP16:        ~67 TFLOPS  → 估算延迟 ≈ 桌面延迟 × 2.7
AGX Orin INT8:        ~275 TOPS   → INT8 进一步加速
```

已有的 ConvFuser TRT 延迟数据（FP32/FP16/INT8）乘以换算系数即为 Orin 上的估算值。**精度结果（NDS=0.5727 INT8）可以直接用——精度不随硬件变化。**

### 方案 B：NVIDIA LaunchPad / DLI 云端 Jetson

NVIDIA 曾提供过云端 Jetson 开发环境（[developer.nvidia.com/embedded/learn](https://developer.nvidia.com/embedded/learn)），可以远程使用真实 Jetson 硬件。可用性需确认。

### 方案 C：Jetson 容器（仅验证软件兼容性）

```bash
docker pull nvcr.io/nvidia/l4t-tensorrt:r35.4.1-runtime
```

只能验证代码能否在 ARM 上编译通过，**不能模拟 GPU 性能**。

### 游说老师的材料组合

1. **精度数据**（已有）：NDS=0.5810 PTQ 无损，TRT INT8 NDS=0.5727（−1.3%）
2. **ConvFuser 单模块实测加速**（已有）：6.81x INT8 加速
3. **引用 NVIDIA 官方数据**：CUDA-BEVFusion 在 Orin 上 FP16 18FPS → INT8 25FPS → 38% 加速
4. **成本论证**：AGX Orin 开发套件 ~$2000，Orin NX ~$600

---

## 三、学习顺序

### 第一阶段：理解全貌（~1 小时）

1. **`README.md`** — 了解项目做了什么、怎么用
2. **`docs/REPORT.md`** — 核心报告，重点看：
   - **第 2 节（架构分析）**：BEVFusion 各子模块的计算特性
   - **第 3 节（设计思路）**：为什么只能量化 4/6 模块、Hybrid 推理架构

### 第二阶段：理解量化原理（~2 小时）

3. **`docs/PTQ_BENCHMARK_NOTES.md`** — 第四节"扩大量化覆盖的可行方案"，理解每个模块失败的具体原因
4. **`tools/quant_ptq_minmax.py`** — 核心量化脚本，重点看三个函数：
   - `patch_mmcv_for_fx()`（~第 72 行）— mmcv 和 torch.fx 的兼容性问题
   - `apply_selective_ptq()`（~第 161 行）— 选择性量化的遍历逻辑
   - `run_calibration()`（~第 235 行）— MinMax 校准流程：enable_calibration → enable_quantization

### 第三阶段：理解模型修改（~1 小时）

5. 对照读三个被修改的模型文件，理解 `torch.fx` 追踪需要的改动：
   - `mmdet3d/models/fusers/conv.py` — 最简单，1 行改动（torch.cat 显式索引）
   - `mmdet3d/models/necks/second.py` — 删除 1 个 len() 断言
   - `mmdet3d/models/necks/generalized_lss.py` — `len(inputs)` → `self.num_ins`

### 第四阶段：理解 TRT 部署（~2 小时）

6. **`tools/trt_export_fuser.py`** — 最简单的 TRT 脚本，理解标准流程：ONNX 导出 → TRT 引擎构建 → 延迟测试
7. **`tools/trt_eval_hybrid.py`** — Hybrid 端到端评估，重点理解：
   - `FuserForExport`（为什么需要 deepcopy 隔离参数）
   - `TRTFuser`（如何在 PyTorch 管线中嵌入 TRT 推理）
   - 整个 5 步管线的数据流

### 第五阶段：对比 NVIDIA 方案（~1 小时）

8. **浏览 CUDA-BEVFusion 的 `qat/lean/quantize.py`** — 对比 `pytorch_quantization` vs MQBench 的量化方式
9. **浏览 `qat/lean/exptool.py`** — 理解 NVIDIA 如何用手工 ONNX 节点解决稀疏卷积导出

### 第六阶段：动手验证

10. **`docs/RUNBOOK.md`** — 按顺序自己跑一遍所有命令，验证每个数字
11. **`docs/RESULTS_LOG.md`** — 对照运行结果与记录是否一致

---

## 四、关键概念速查

| 概念 | 一句话解释 | 相关文件 |
|------|-----------|---------|
| PTQ（训练后量化） | 不需要重训练，用少量数据统计 min/max 确定量化范围 | `quant_ptq_minmax.py` |
| FakeQuant | 在 FP32 上模拟 INT8 的量化/反量化过程，验证精度 | MQBench 自动插入 |
| torch.fx | PyTorch 的符号追踪工具，将 forward 转为静态计算图 | `prepare_by_platform()` 内部调用 |
| Hybrid 推理 | 部分模块用 TRT 引擎，部分保持 PyTorch 执行 | `trt_eval_hybrid.py` |
| MinMax 校准 | 记录每层激活的全局 min/max，据此计算 scale/zero_point | `run_calibration()` |
| NDS | nuScenes Detection Score，综合 mAP + 5 个误差指标 | nuScenes 评估标准 |
| BEV | Bird's Eye View，将多模态特征投影到 180×180 鸟瞰网格 | BEVFusion 核心思想 |
