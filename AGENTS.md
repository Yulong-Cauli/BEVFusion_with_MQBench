# BEVFusion + MQBench 项目说明（Agent Instructions）

## 环境

- **操作系统**：Windows，PowerShell
- **Conda 环境**：`bevfusion`（路径：`D:\aconda\envs\bevfusion`）
- **关键依赖**：PyTorch 1.10.2+cu113，mmdet3d 0.0.0（本地安装），MQBench 0.0.6
- **数据集**：`data/nuscenes`（v1.0-mini，Junction 符号链接）
- **预训练权重**：`pretrained/bevfusion-det.pth`
- **所有 Python 命令**必须加 `$env:PYTHONUTF8="1"`，否则 Windows 会报 GBK codec 错误

运行任何脚本前的标准前置设置：
```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

## 项目文档

- `CLIlog.md`：完整历史修复记录，包含环境信息和所有已知 bug 修复
- `docs/PTQ_BENCHMARK_NOTES.md`：量化覆盖问题分析、Benchmark 命令、TensorRT 导出方案、当前开放问题
- `docs/RESULTS_LOG.md`：评测结果记录（FP32 vs PTQ 精度 / 速度 / 大小）

**开始任务前请先阅读以上三个文档。**

## 当前已验证可工作的脚本

| 脚本 | 状态 | 结果 |
|------|------|------|
| `tools/test.py` | ✅ | NDS = 0.5800 |
| `tools/quant_ptq_minmax.py` | ✅ | NDS = 0.5774（PTQ 4/6 模块量化，精度损失 0.27%） |
| `tools/quant_benchmark.py` | ✅ | 可运行 |
| `tools/trt_export_fuser.py` | ✅ | ConvFuser TRT PoC：INT8 6.81x 加速 |
| `tools/train.py` | ⚠️ | NaN 修复已应用但未验证 |
| `tools/quant_train.py` | ⚠️ | 脚本已修复但未测试 |

## 关键约束（务必遵守）

### 1. 不能破坏已有工作状态
- 每次修改代码后，必须跑 `tools/test.py` 确认 NDS 仍在 0.578 以上
- PTQ 脚本的已有量化逻辑（`decoder/backbone`）不能被破坏，扩大覆盖必须是向后兼容的**追加**

### 2. torch.fx 追踪兼容性
- **不要用 `torch.fx.wrap('len')`**：它全局拦截所有 `len()` 调用（包括对普通列表的），导致 `range(Proxy)` 等连锁失败。应改用 `__init__` 中预计算的常量（如 `self.num_ins`）替代 `len(input)` 调用。
- **mmcv 层包装器**：mmcv 的 `Conv2d`/`ConvTranspose2d` 等包含 `if x.numel() == 0` 兼容性检查，在 fx 追踪时会触发 `TraceError`。已通过 `patch_mmcv_for_fx()` 上下文管理器（在 `quant_ptq_minmax.py` 中）解决。

### 3. PTQ checkpoint 不能用 test.py 直接评估
`quant_ptq_minmax.py` 生成的 checkpoint 包含 torch.fx 改造过的结构（key 名变化），用 `test.py` 加载时 `strict=False` 会静默跳过量化模块的权重，导致那部分用随机权重推理，评估结果崩溃。PTQ 精度评估只能通过 `quant_ptq_minmax.py` 内部流程（不加 `--no-eval`）进行。

### 4. quant_benchmark.py 有独立的量化逻辑
`tools/quant_benchmark.py` 里的 `build_quant_model` 函数有自己的 `apply_selective_ptq` 调用（从 `quant_ptq_minmax.py` import）。扩大量化覆盖后，确认 benchmark 也能正确加载新的量化结构。

### 5. MMDataParallel 包装
在 PTQ 校准或推理时，模型必须包在 `MMDataParallel(model, device_ids=[0])` 里，否则 dataloader 返回的 `DataContainer` 对象无法被模型正确解包。

## 待完成任务（见 docs/PTQ_BENCHMARK_NOTES.md 第五节）

### 优先：扩大量化覆盖（方案 A，见 PTQ_BENCHMARK_NOTES.md 第四节）

按难度排序：

1. ~~**简单**：`decoder/neck`（SECONDFPN）和 `camera/neck`（GeneralizedLSSFPN）~~ ✅ 已完成
   - 修复：移除 Proxy 上的 `len()` 调用，改用 `self.num_ins` 等常量；新增 `patch_mmcv_for_fx()` 绕过 mmcv 包装层

2. ~~**中等**：`fuser`（ConvFuser）~~ ✅ 已完成
   - 修复：将 `torch.cat(inputs, dim=1)` 改为 `torch.cat([inputs[i] for i in range(len(self.in_channels))], dim=1)`，让 fx 看到独立的 Proxy 对象而非代表列表的单个 Proxy

3. **困难**：`camera/backbone`（SwinTransformer）、`heads/object`（TransFusionHead）
   - 见 PTQ_BENCHMARK_NOTES.md 的详细分析

### 次要：验证训练和 QAT
- 跑 `tools/train.py` 约 100 步确认 `grad_norm` 不再出现 NaN（已在 `configs/default.yaml` 加 `init_scale: 512`）
- 跑 `tools/quant_train.py` 验证 QAT 能启动不崩溃

### 下一步：TensorRT INT8 导出（见 PTQ_BENCHMARK_NOTES.md 第五节问题 2）

✅ **ConvFuser PoC 已完成**：FP32 ONNX → TRT INT8 引擎，6.81x 加速、6.48x 压缩。

**已验证的导出方案**：
- MQBench `convert_deploy` / `torch.onnx.export` 均无法导出 FakeQuant 模型（PyTorch 1.10 限制）
- 可行方案：导出 FP32 ONNX → TRT `IInt8EntropyCalibrator2` 原生 INT8 校准
- 参考脚本：`tools/trt_export_fuser.py`

**待推广到其余模块**：
1. decoder/backbone（SECOND）→ 确定输入 shape，编写 wrapper
2. decoder/neck（SECONDFPN）→ 含 ConvTranspose2d，编写 wrapper
3. camera/neck（GeneralizedLSSFPN）→ 多尺度输入，编写 wrapper
4. 编写 Hybrid Runner 整合 TRT 引擎 + PyTorch 推理
