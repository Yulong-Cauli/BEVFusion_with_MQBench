# PTQ Benchmark 说明与量化覆盖问题

---

## 一、当前量化覆盖情况

使用 MQBench `prepare_by_platform`（基于 `torch.fx` 符号追踪）对 BEVFusion 各子模块进行选择性量化，结果如下：

| 子模块 | 类型 | 量化结果 | 说明 |
|--------|------|---------|---------|
| `camera/backbone` | SwinTransformer | ❌ 失败 | 含动态控制流（`if tensor_value:` 分支） |
| `camera/neck` | GeneralizedLSSFPN | ✅ 成功 | 已修复：移除 Proxy 上的 `len()` 调用 + `patch_mmcv_for_fx()` |
| `fuser` | ConvFuser | ❌ 失败 | `torch.cat(Proxy, dim=int)` 参数冲突 |
| `decoder/backbone` | SECOND (Conv2d BEV) | ✅ 成功 | 纯静态卷积结构，fx 可追踪 |
| `decoder/neck` | SECONDFPN | ✅ 成功 | 已修复：移除 Proxy 上的 `len()` 断言 + `patch_mmcv_for_fx()` |
| `heads/object` | TransFusionHead | ❌ 失败 | Proxy 对象被 for 循环迭代 |

**结论**：`decoder/backbone`、`decoder/neck`、`camera/neck` 共 3 个模块成功量化，量化覆盖率 **3/6**。PTQ NDS = 0.5799（FP32 基线 0.5801，无精度损失）。

---

## 二、关于模型大小的说明

MQBench 的量化是"仿真量化"（Fake Quantization / QAT/PTQ 模拟），其本质是：

- 权重仍以 **FP32** 存储，额外附加 `scale` / `zero_point` 参数
- `.pth` 文件大小与 FP32 原始模型**几乎相同**（甚至略大）
- 真正的 INT8 压缩（≈ FP32 / 4）需要将模型**导出到 TensorRT 引擎**后才能实现

| 指标 | 数值 |
|------|------|
| FP32 模型内存占用 | 155.91 MB |
| FP32 `.pth` 文件大小 | 156.13 MB |
| 估算 INT8 部署大小（理论） | **38.98 MB**（× 0.25，仅供参考） |
| PTQ `.pth` 文件大小（实际） | ≈ 156+ MB（FakeQuant 参数略增） |

---

## 三、完整 Benchmark 命令

所有输出同时显示在屏幕并写入日志文件，跑完将日志发送即可对比结果。

### 前置设置（每次开新 PowerShell 窗口时执行一次）

```powershell
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

```powershell
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 128 2>&1 | Tee-Object -FilePath "results_ptq.log"
```

### Step 2：Benchmark 对比（模型大小 + 推理延迟）

```powershell
$env:PYTHONUTF8="1"
$ptq_ckpt = (Get-ChildItem -Recurse -Filter "ptq_minmax_model.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "PTQ model: $ptq_ckpt"
python tools/quant_benchmark.py `
   configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --checkpoint pretrained/bevfusion-det.pth `
    --quant-checkpoint $ptq_ckpt `
    --num-iters 30 2>&1 | Tee-Object -FilePath "results_benchmark.log"
```

输出内容包括：FP32 vs PTQ 参数量、内存大小、均值/P95/P99 延迟、估算加速比。

### Step 3：FP32 基准精度评估（NDS）

```powershell
$env:PYTHONUTF8="1"
python tools/test.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    pretrained/bevfusion-det.pth --eval bbox 2>&1 | Tee-Object -FilePath "results_fp32.log"
```

### Step 4：PTQ 模型精度评估（NDS）

> ⚠️ **注意**：PTQ checkpoint 通过 `torch.fx` 改造了 `decoder/backbone` 的模型结构（key 名变更），**不能直接用 `test.py` 加载评估**，否则该模块权重实际未被加载（`strict=False` 静默跳过），导致输出退化为空预测，评估崩溃。
>
> 正确做法：直接使用 Step 1 的 PTQ 脚本，**去掉 `--no-eval`**，脚本会在校准完成后自动重建 FakeQuant 结构并完整输出 NDS / mAP。

```powershell
# Step 1 已包含评估，直接运行 PTQ 脚本即可获得 NDS（不要加 --no-eval）
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
python tools/quant_ptq_minmax.py `
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml `
    --load_from pretrained/bevfusion-det.pth `
    --calib-batches 128 2>&1 | Tee-Object -FilePath "results_ptq.log"
```

FP32 基准：**NDS = 0.5800，mAP = 0.5742**。PTQ 结果在 `results_ptq.log` 末尾输出。

---

## 四、扩大量化覆盖的可行方案

以下方案中 A.1 已执行完成，其余**仅供参考，未执行**，难度从低到高排列。

---

### 方案 A：手动修复模型代码使其对 fx 可追踪（中等难度）

`torch.fx` 符号追踪要求所有控制流基于 Python 常量，不能依赖 Tensor 的运行时值。针对各失败原因：

**1. `len()` 调用 + mmcv 包装层（`camera/neck`、`decoder/neck`）** ✅ 已完成

> ⚠️ 原文档建议的 `torch.fx.wrap('len')` 方案**不可行**：它全局拦截所有 `len()` 调用（包括对普通 Python 列表的），导致 `range(Proxy)` 等连锁 `TypeError`。

实际修复方案：
- 移除 `forward` 中对 Proxy 输入调用 `len()` 的断言
- 将 `range(len(inputs))` 替换为 `range(self.num_ins)`（`__init__` 中预计算的常量）
- 新增 `patch_mmcv_for_fx()` 上下文管理器（在 `quant_ptq_minmax.py`），在 fx 追踪期间临时将 mmcv 的 `Conv2d`/`ConvTranspose2d`/`MaxPool2d`/`Linear` 包装层的 `forward` 替换为 PyTorch 原生父类版本，绕过 `if x.numel() == 0` 兼容性检查

**2. `fuser`（ConvFuser 中的 `torch.cat(Proxy, ...)`）**

问题在于 `torch.cat` 接收的是一个包含 Proxy 的列表，fx 无法解析。  
可以将 `torch.cat([a, b], dim=1)` 改写为显式拼接以避免 Proxy 列表问题，或在 `ConvFuser` 的 `forward` 里用 `@torch.fx.wrap` 包装 cat 操作。

**3. `camera/backbone`（SwinTransformer 动态控制流）**

SwinTransformer 内部存在形如 `if x.shape[0] > window_size:` 的分支，fx 默认无法追踪。  
可以给 `prepare_by_platform` 传入 `concrete_args`（固定输入尺寸），让 fx 在追踪时把这些分支常量化。

**4. `heads/object`（TransFusionHead Proxy 迭代）**

TransFusionHead 有 `for layer in self.decoder_layers:` 等动态迭代。可以将动态 ModuleList 迭代改写为静态展开，或使用 `torch.fx.wrap` 包装相关函数。

---

### 方案 B：改用逐层手动插入 Observer（低侵入，较繁琐）

不依赖 fx 的自动追踪，直接对目标模块的每个 `nn.Conv2d` / `nn.Linear` 手动替换为 `QuantizedConv2d` / `QuantizedLinear`（MQBench 提供了这些类），或用 `torch.quantization.prepare` 的手动 qconfig 注入方式。

优点：不受 fx 限制，可覆盖任意模块。  
缺点：需要手动枚举每个待量化子层，代码量较大。

---

### 方案 C：换用 PyTorch 原生 PTQ（`torch.quantization`）

PyTorch 官方的 `torch.quantization.prepare` → `calibrate` → `torch.quantization.convert` 流程不依赖 fx，基于 Module Hook，兼容性更好。

步骤：
1. 对模型的 `nn.Conv2d` / `nn.Linear` 层设置 `qconfig`
2. `torch.quantization.prepare(model, inplace=True)` 插入 Observer
3. 用验证集跑前向收集统计量
4. `torch.quantization.convert(model, inplace=True)` 将权重转为 INT8

缺点：与 MQBench 的 QAT 流程分离，如果后续要做 QAT 微调需要额外适配。

---

### 方案 D：直接导出 TensorRT（跳过 FakeQuant，直接得到部署结果）

跳过 MQBench，直接将 FP32 模型通过 ONNX 导出到 TensorRT，使用 TensorRT 自带的 PTQ（`IInt8EntropyCalibrator2`）校准。

优点：直接得到真实 INT8 推理速度和大小，不受 fx 限制，TensorRT 对逐层自动量化支持更好。  
缺点：BEVFusion 含稀疏卷积（SpConv）和自定义 CUDA 算子，ONNX 导出较困难，需要额外适配。

---

### 方案 E：仅量化 decoder 部分，其余保持 FP32（当前方案的最优化）

当前方案已经量化了 `decoder/backbone`（SECOND）、`decoder/neck`（SECONDFPN）和 `camera/neck`（GeneralizedLSSFPN）。可以在此基础上：
- 量化 `fuser`（ConvFuser，用方案 A.2 可修复）

这两个修复工作量最小，收益相对明显（fuser 和 decoder/neck 合计参数量不小），是性价比最高的扩展路径。

---

## 五、当前开放问题汇总

### 🔴 功能性问题（影响结果质量）

**1. 量化覆盖率待进一步提升**

当前 3/6 模块已量化（`decoder/backbone`、`decoder/neck`、`camera/neck`）。剩余 3 个模块按修复难度：

| 难度 | 模块 | 原因 | 修复思路 |
|------|------|------|---------|
| 中等 | `fuser`（ConvFuser） | `Proxy + cat()` | 改写 cat 调用方式 |
| 困难 | `camera/backbone`（SwinTransformer） | 动态控制流 | 传 `concrete_args` 固定输入尺寸 |
| 困难 | `heads/object`（TransFusionHead） | Proxy 被迭代 | 展开 ModuleList 或包装函数 |

**2. TensorRT 导出未做**

当前能看到的大小/速度都是 FakeQuant 仿真值，无实际意义。需要在 Windows GPU 机器上安装 TensorRT 才能导出，且覆盖率扩大后导出价值更高。

> 注意：TensorRT 引擎构建必须在有 NVIDIA GPU 的机器上进行，且引擎是硬件绑定的（在哪块 GPU 上构建就只能在那块 GPU 上运行）。无 GPU 的 Linux 机器无法参与此流程。

### 🟡 验证缺失（脚本改了但没跑过）

**3. `train.py` NaN 修复未验证**

`configs/default.yaml` 加了 `init_scale: 512`，但没有实际跑训练确认 `grad_norm` 不再出现 NaN。

**4. `quant_train.py`（QAT）完全未测试**

脚本修复已完成（dist、pretty_text、distributed flag），但从未端到端跑过一次。

### 优先级建议

- 目标是**看到真实量化效果** → 先做问题 1（扩大覆盖），再做问题 2（TRT 导出）。当前只量化 SECOND backbone，TRT 导出后加速比也有限，意义不大。
- 目标是**跑通完整流程** → 先做问题 3（验证训练）+ 问题 4（跑一次 QAT）。
