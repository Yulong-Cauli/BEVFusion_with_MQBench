# PTQ Benchmark 说明与量化覆盖问题

---

## 一、当前量化覆盖情况

使用 MQBench `prepare_by_platform`（基于 `torch.fx` 符号追踪）对 BEVFusion 各子模块进行选择性量化，结果如下：

| 子模块 | 类型 | 量化结果 | 说明 |
|--------|------|---------|---------|
| `camera/backbone` | SwinTransformer | ❌ 失败 | 含动态控制流（`if tensor_value:` 分支） |
| `camera/neck` | GeneralizedLSSFPN | ✅ 成功 | 已修复：移除 Proxy 上的 `len()` 调用 + `patch_mmcv_for_fx()` |
| `fuser` | ConvFuser | ✅ 成功 | 已修复：显式索引替代 Proxy 列表传递给 `torch.cat` |
| `decoder/backbone` | SECOND (Conv2d BEV) | ✅ 成功 | 纯静态卷积结构，fx 可追踪 |
| `decoder/neck` | SECONDFPN | ✅ 成功 | 已修复：移除 Proxy 上的 `len()` 断言 + `patch_mmcv_for_fx()` |
| `heads/object` | TransFusionHead | ❌ 失败 | Proxy 对象被 for 循环迭代 |

**结论**：`decoder/backbone`、`decoder/neck`、`camera/neck`、`fuser` 共 4 个模块成功量化，量化覆盖率 **4/6**。PTQ NDS = **0.5810**（FP32 基线 0.5801，精度无损，+0.0009）。

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
conda activate bevfusion
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

### 方案 ：手动修复模型代码使其对 fx 可追踪（中等难度）

`torch.fx` 符号追踪要求所有控制流基于 Python 常量，不能依赖 Tensor 的运行时值。针对各失败原因：

**1. `len()` 调用 + mmcv 包装层（`camera/neck`、`decoder/neck`）** ✅ 已完成

> ⚠️ 原文档建议的 `torch.fx.wrap('len')` 方案**不可行**：它全局拦截所有 `len()` 调用（包括对普通 Python 列表的），导致 `range(Proxy)` 等连锁 `TypeError`。

实际修复方案：
- 移除 `forward` 中对 Proxy 输入调用 `len()` 的断言
- 将 `range(len(inputs))` 替换为 `range(self.num_ins)`（`__init__` 中预计算的常量）
- 新增 `patch_mmcv_for_fx()` 上下文管理器（在 `quant_ptq_minmax.py`），在 fx 追踪期间临时将 mmcv 的 `Conv2d`/`ConvTranspose2d`/`MaxPool2d`/`Linear` 包装层的 `forward` 替换为 PyTorch 原生父类版本，绕过 `if x.numel() == 0` 兼容性检查

**2. `fuser`（ConvFuser 中的 `torch.cat(Proxy, ...)`）** ✅ 已完成

问题：`torch.cat(inputs, dim=1)` 中 `inputs` 是一个 Proxy 对象（代表列表），fx 无法将其展开。  
修复：将 `torch.cat(inputs, dim=1)` 改为 `torch.cat([inputs[i] for i in range(len(self.in_channels))], dim=1)`，通过 `__getitem__` 索引让 fx 看到独立的 Proxy 对象。

**3. `camera/backbone`（SwinTransformer 动态控制流）**

SwinTransformer 内部存在形如 `if x.shape[0] > window_size:` 的分支，fx 默认无法追踪。  
可以给 `prepare_by_platform` 传入 `concrete_args`（固定输入尺寸），让 fx 在追踪时把这些分支常量化。

**4. `heads/object`（TransFusionHead Proxy 迭代）**

TransFusionHead 有 `for layer in self.decoder_layers:` 等动态迭代。可以将动态 ModuleList 迭代改写为静态展开，或使用 `torch.fx.wrap` 包装相关函数。

---

## 五、当前开放问题汇总

### 🔴 功能性问题（影响结果质量）

**1. 量化覆盖率待进一步提升**

当前 4/6 模块已量化（`decoder/backbone`、`decoder/neck`、`camera/neck`、`fuser`）。剩余 2 个模块：

| 难度 | 模块 | 原因 | 修复思路 |
|------|------|------|---------|
| 困难 | `camera/backbone`（SwinTransformer） | 动态控制流 | 传 `concrete_args` 固定输入尺寸 |
| 困难 | `heads/object`（TransFusionHead） | Proxy 被迭代 | 展开 ModuleList 或包装函数 |

**2. TensorRT INT8 导出（将 FakeQuant 转为真实 INT8 部署）**

~~当前 benchmark 结果均为 FakeQuant 仿真（权重仍 FP32，GPU 上额外执行 quantize/dequantize），速度反而略慢。真实的 INT8 加速（2–4×）和体积压缩（4×）需要导出为 TensorRT 引擎。~~

✅ **ConvFuser PoC 已完成**：通过 FP32 ONNX → TRT 原生 INT8 校准方案验证了 6.81x 加速、6.48x 压缩。

**已验证方案（绕过 MQBench 导出限制）：**

MQBench `convert_deploy` 和 `torch.onnx.export` 都无法导出 FakeQuant 模型（PyTorch 1.10 缺少自定义 op 的 ONNX symbolic）。
实际可行方案：导出 FP32 ONNX → TRT `IInt8EntropyCalibrator2` 做 INT8 校准 → 构建引擎。

**ConvFuser PoC 结果（`tools/trt_export_fuser.py`）：**

| 方法 | 延迟 | 加速比 | 引擎大小 | 压缩比 |
|------|------|--------|---------|--------|
| PyTorch FP32 | 5.083 ms | 1.00x | — | — |
| TRT FP32 | 4.017 ms | 1.27x | 5385 KB | 1.00x |
| TRT FP16 | 1.437 ms | 3.54x | 1543 KB | 3.49x |
| TRT INT8 | 0.746 ms | **6.81x** | 832 KB | **6.48x** |

**下一步：推广到其余 3 个已量化模块**

| 模块 | ONNX 导出难度 | 备注 |
|------|-------------|------|
| fuser（ConvFuser） | ✅ 已完成 | 需 wrapper 将 list input → 两个独立参数 |
| decoder/backbone（SECOND） | 中等 | 纯 Conv2d 堆叠，需确定输入 shape |
| decoder/neck（SECONDFPN） | 中等 | 含 ConvTranspose2d，需 wrapper |
| camera/neck（GeneralizedLSSFPN） | 中等 | 含多尺度输入，需 wrapper |

**BEVFusion TRT 导出的核心难点：**

| 组件 | ONNX 兼容性 | 说明 |
|------|------------|------|
| camera/neck, fuser, decoder/backbone, decoder/neck | ✅ 纯标准算子 | Conv2d / ConvTranspose2d / BN / ReLU，可直接导出 |
| camera/backbone（SwinTransformer） | 🟡 需验证 | `roll` / `window_partition` 等操作 ONNX 支持参差不齐 |
| camera/vtransform（bev_pool） | 🔴 不支持 | `QuickCumsumCuda` 自定义 CUDA autograd Function，无 ONNX 等价 |
| lidar/backbone（SpConv） | 🔴 不支持 | 稀疏卷积，无标准 ONNX 算子 |
| heads/object（TransFusionHead） | 🟡 需 opset ≥ 11 | 含 `topk` / `scatter` / `nonzero` 等动态 shape 操作 |

**推荐方案：分段导出（Hybrid 推理）**

只将已量化且 ONNX 友好的 4 个子模块导出为 TRT INT8 引擎，其余保持 PyTorch 执行：

```
TRT INT8 引擎（已量化，标准算子）：
├── camera/neck (GeneralizedLSSFPN)  → ONNX → TRT INT8
├── fuser (ConvFuser)                → ONNX → TRT INT8
├── decoder/backbone (SECOND)        → ONNX → TRT INT8
└── decoder/neck (SECONDFPN)         → ONNX → TRT INT8

PyTorch 执行（未量化或 ONNX 不兼容）：
├── camera/backbone (SwinTransformer)
├── camera/vtransform (bev_pool CUDA 算子)
├── lidar/* (SpConv 稀疏卷积)
└── heads/object (TransFusionHead)
```

**实施步骤：**

1. **安装 TensorRT**：CUDA 11.3 对应 TRT 8.5.x 或 8.6.x（Windows pip 安装）
   ```powershell
   pip install tensorrt==8.5.3.1
   # 或从 NVIDIA 官网下载 Windows zip 包安装
   ```
2. **逐模块 ONNX 导出**：对每个已量化子模块调用 `torch.onnx.export`（或 MQBench `convert_deploy`）
3. **构建 INT8 引擎**：用 `trtexec` 或 Python API，传入量化参数（scale/zero_point）
4. **Hybrid Runner**：编写推理脚本，TRT 引擎处理已量化子模块，PyTorch 处理其余部分
5. **验证一致性**：对比 TRT INT8 输出 vs FakeQuant 输出，确认数值误差在可接受范围

**备选参考：NVIDIA 官方 BEVFusion TRT 部署**

NVIDIA 有 [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion) 项目，提供了完整的 TensorRT 适配（含 SpConv plugin、bev_pool plugin），但它是独立的 C++ 工程，与 MQBench 量化流程不直接兼容，仅供架构参考。

> 注意：TensorRT 引擎是硬件绑定的（在哪块 GPU 上构建就只能在那块 GPU 上运行）。引擎构建必须在目标部署 GPU 上进行。

### 🟡 验证缺失（脚本改了但没跑过）

**3. `train.py` NaN 修复未验证**

`configs/default.yaml` 加了 `init_scale: 512`，但没有实际跑训练确认 `grad_norm` 不再出现 NaN。

**4. `quant_train.py`（QAT）完全未测试**

脚本修复已完成（dist、pretty_text、distributed flag），但从未端到端跑过一次。

### 优先级建议

- 目标是**看到真实 INT8 部署效果** → ✅ ConvFuser PoC 已完成（6.81x 加速）。下一步推广到 decoder/backbone、decoder/neck、camera/neck，编写 Hybrid Runner 整合端到端推理
- 目标是**进一步扩大量化覆盖** → 尝试 `camera/backbone`（SwinTransformer）和 `heads/object`（TransFusionHead），均为困难级别
- 目标是**跑通完整训练流程** → 验证 `train.py`（问题 3）+ 跑一次 QAT（问题 4）
