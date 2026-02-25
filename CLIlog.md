# 注意事项

**cl 没有放在系统环境变量里面。** 如果涉及 cl 的操作，记得先运行：

```powershell
cmd /c "call `"D:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`" && powershell"
```

如果修改了 C++ / CUDA 代码，或者需要重新运行 `setup.py develop`，需要确保编译器环境可见：

```powershell
$env:DISTUTILS_USE_SDK=1
$env:MSSdk=1
```

**Windows 编码问题**：所有 Python 脚本必须加 `$env:PYTHONUTF8="1"`，否则读取 YAML 时会报 GBK codec 错误。

```powershell
# 1 测试原始 FP32 模型（基准分）
$env:PYTHONUTF8="1"
python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox

# 2 PTQ 校准（128 batch，校准完成后自动保存量化模型）
$env:PYTHONUTF8="1"
python tools/quant_ptq_minmax.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --load_from pretrained/bevfusion-det.pth --calib-batches 128

# 3 评估 PTQ 量化模型跑分（先找到保存的文件，再评估）
$env:PYTHONUTF8="1"
$ptq_ckpt = (Get-ChildItem -Recurse -Filter "ptq_minmax_model.pth" | Sort-Object LastWriteTime -Descending |
Select-Object -First 1).FullName
Write-Host "PTQ model: $ptq_ckpt"
python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml $ptq_ckpt --eval bbox
```

   说明：2的 PTQ 脚本默认也会在最后跑一次 eval（除非加了
  --no-eval），所以如果2跑完了直接出结果的话，3就不需要单独跑了。3是单独评估用的备用命令。

---

# 环境信息

| 项目 | 值 |
|------|-----|
| Conda 环境 | `bevfusion` (`D:\aconda\envs\bevfusion`) |
| PyTorch | 1.10.2+cu113 |
| mmdet3d | 0.0.0（本地 `pip install -e .` 安装） |
| MQBench | 0.0.6 |
| mmdet | 2.20.0 |
| Python | 3.8（Windows） |
| CUDA | 11.3 |
| mpi4py | **未安装**（torchpack 分布式需要，单 GPU 不需要） |

**数据集**：`data/nuscenes`（Junction 符号链接 → `D:\Pytorchlib\data\nuscenes`），v1.0-mini。  
**预训练权重**：`pretrained/bevfusion-det.pth`，`pretrained/swint-nuimages-pretrained.pth`。

---

# 运行命令（所有脚本均需设置 PYTHONUTF8）

```powershell
$env:PYTHONUTF8="1"

# 测试（推理评估）
python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox

# PTQ（离线量化校准）
python tools/quant_ptq_minmax.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --load_from pretrained/bevfusion-det.pth --calib-batches 128

# 训练
python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml

# QAT（量化感知训练）
python tools/quant_train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --load_from pretrained/bevfusion-det.pth

# Benchmark（模型大小 / 推理速度）
python tools/quant_benchmark.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --checkpoint pretrained/bevfusion-det.pth --size-only
```

---

# CLI 操作日志

---

## 2026-02-17 · 初始环境配置

**数据集与预训练模型**

- 数据集路径：`D:\Pytorchlib\data\nuscenes`，创建 Junction 符号链接映射到 `data/nuscenes`。
- 下载预训练模型到 `pretrained/`：`bevfusion-det.pth`、`swint-nuimages-pretrained.pth`。

**编译与 Import 修复**

- 调用 `vcvars64.bat` 环境编译 CUDA 扩展（`setup.py develop`）。
- 修复 `ImportError: cannot import name 'feature_decorator_ext'`：在 `mmdet3d/ops/__init__.py` 注释掉 `feature_decorator` 引用。
- 在 `mmdet3d/models/backbones/__init__.py` 注释掉 `radar_encoder` 引用。

**Pipeline / 配置修复**

- 修复 `KeyError: 'radar'`：在 `configs/nuscenes/det/default.yaml` 中移除 `LoadRadarPointsMultiSweeps` 管道步骤及 `Collect3D` 中的 `radar` 键。
- 修复因注释产生的 `None` 列表项导致的 `TypeError`。
- 修复 `RuntimeError`（维度不匹配）：`mmdet3d/models/vtransforms/depth_lss.py` 强制设置 `add_depth_features=False`（`__init__` 中已写死，见下方 2026-02-25 补充）。

---

## 2026-02-20 · Windows 环境训练与测试修复

**训练脚本（`tools/train.py` / `mmdet3d/apis/train.py`）**

- 增加单 GPU（非分布式）检测与支持，修复 `num_gpus` 为 `None` 导致的 `TypeError`。
- 非分布式模式下使用 `MMDataParallel` 替代 `MMDistributedDataParallel`。

**配置调整**

- `samples_per_gpu` 降为 1（解决 8GB 显存 OOM）。
- 禁用 `TensorboardLoggerHook`（规避 `distutils` 兼容性错误）。

**Tensor 类型修复**

- `mmdet3d/core/bbox/assigners/hungarian_assigner.py`：强制 `gt_labels` 转 `long`，修复 `IndexError`。
- `mmdet3d/models/heads/bbox/transfusion.py`：`gt_labels_3d` 转 `long`，修复 `RuntimeError: Index put requires ...`。

**验证结果**

- `train.py` 成功运行 450+ 迭代，loss 正常下降。
- `test.py` 全量评估：**NDS = 0.5803**。

---

## 2026-02-21 · 路径变更后环境修复 & VS Code 配置

**问题**：项目文件夹重命名导致编译产物路径失效，`ModuleNotFoundError: No module named 'mmcv'`。

**修复**：删除 `build/`、`mmdet3d.egg-info/`、`dist/`，在新路径下重新 `pip install -e .`。

**VS Code 调试配置**：创建 `.vscode/launch.json`，新增 `Python: Test BEVFusion` 配置，`env` 中加入 `"PYTHONUTF8": "1"`。

---

## 2026-02-25 · 重新配置环境 + MQBench PTQ 修复

**背景**：Conda 环境重建（新机器/新路径），重新 `pip install -e .`。此次对所有脚本进行了系统性修复，使 `test.py` 和 `quant_ptq_minmax.py` 可在单 GPU / 无 MPI 环境下正常运行。

### Numpy 兼容性修复（重装环境后发现）

| 文件 | 修复内容 |
|------|---------|
| `mmdet3d/core/utils/visualize.py` | `np.bool` → `np.bool_` |
| `mmdet3d/datasets/pipelines/loading.py` | `np.bool` → `bool` |
| `mmdet3d/datasets/pipelines/transforms_3d.py` | `np.bool` → `bool` |

### 训练 NaN 修复（`configs/default.yaml`）

- **根本原因**：`fp16.loss_scale` 中只配置了 `growth_interval: 2000`，未设置 `init_scale`，PyTorch `GradScaler` 默认 `init_scale=65536` → mini 数据集早期 FP16 溢出 → `grad_norm: nan`。
- **修复**：在 `configs/default.yaml` 的 `fp16.loss_scale` 下添加 `init_scale: 512`。

```yaml
# configs/default.yaml
fp16:
  loss_scale:
    init_scale: 512        # ← 新增，防止早期 FP16 溢出
    growth_interval: 2000
```

### quant_ptq_minmax.py 修复

| # | 问题 | 修复 |
|---|------|------|
| 1 | `dist.init()` 无条件调用，需要 `mpi4py`（未安装） | 仿照 `train.py`：仅在 `RANK`/`WORLD_SIZE` 环境变量存在时调用 |
| 2 | `cfg.pretty_text` → `yapf.FormatCode()` 参数不兼容 | 改为 `f"{cfg}"` |
| 3 | 校准时模型未包在 `MMDataParallel` 中，`DataContainer` 未解包 | 校准前 `model = MMDataParallel(model, device_ids=[0])` |
| 4 | 保存块中 `dist.is_master()` 在非分布式模式下报错 | 移除条件，无条件保存 |
| 5 | 校准数据集使用训练集（含 `GTDepth`），与 `return_loss=False` 推理模式冲突 | 改用验证集（`cfg.data.val`，`test_mode=True`） |

### quant_train.py 修复

| # | 问题 | 修复 |
|---|------|------|
| 1 | `dist.init()` 无条件调用 | 同上，条件化 |
| 2 | `cfg.pretty_text` 不兼容 | 改为 `f"{cfg}"` |
| 3 | `torch.cuda.set_device(dist.local_rank())` 在非分布式下报错 | 条件化，单 GPU 时 `set_device(0)` |
| 4 | `cfg.gpu_ids` 未设置导致单 GPU 模式异常 | 非分布式时补充 `cfg.gpu_ids = [0]` |
| 5 | `train_qat_model(distributed=True)` 硬编码 | 改为传入 `distributed` 变量 |

### depth_lss.py 修复

- **问题**：`BaseDepthTransform.forward()` 调用 `self.get_cam_feats(img, depth, mats_dict)`（3 个位置参数），但 `DepthLSSTransform.get_cam_feats(self, x, d)` 只接受 2 个 → `TypeError: takes 3 positional arguments but 4 were given`。
- **修复**：签名改为 `def get_cam_feats(self, x, d, mats_dict=None)`（`mats_dict` 为可选，`DepthLSSTransform` 不使用它）。

### 验证结果

| 脚本 | 状态 | 备注 |
|------|------|------|
| `test.py` | ✅ 通过 | NDS = 0.5803，与历史基线一致 |
| `quant_ptq_minmax.py` | ✅ 通过 | 16 batch 校准全部成功，模型保存至 `runs/.../ptq_minmax_model.pth` |
| `train.py` | ⚠️ 修复已应用，未重跑验证 | `init_scale: 512` 应解决 NaN 问题 |
| `quant_train.py` | ⚠️ 未测试 | 脚本修复已完成 |
| `quant_benchmark.py` | ⚠️ 未测试 | — |

### PTQ 量化覆盖情况（已知限制）

以下子模块因 `torch.fx` 符号追踪限制无法量化，已自动跳过：

- `camera/backbone`（SwinTransformer：含控制流）
- `camera/neck`（GeneralizedLSSFPN：含 `len()`）
- `fuser`（ConvFuser：`Proxy + cat()` 冲突）
- `decoder/neck`（SECONDFPN：含 `len()`）
- `heads/object`（TransFusionHead：含 Proxy 迭代）

**成功量化**：`decoder/backbone`（SECOND 稀疏卷积 backbone）。






# CLI 操作日志

**日期:** 2026-02-17-11:30

**数据集配置**

- 用户提供了 nuScenes 数据集路径：`D:\Pytorchlib\data\nuscenes`。
- 创建了符号链接（Junction），将外部数据集映射到项目内的 `data/nuscenes`。

**预训练模型**

- 下载了以下模型文件到 `pretrained/` 目录：
  - `bevfusion-det.pth` (检测模型 checkpoint)
  - `swint-nuimages-pretrained.pth` (Backbone 预训练权重)

**编译与代码修复**

- **编译扩展**: 尝试调用 Visual Studio 2019 环境 (`vcvars64.bat`) 重新编译 CUDA 扩展 (`setup.py develop`)。
- **Import 错误修复**: 
  - 遇到 `ImportError: cannot import name 'feature_decorator_ext'`。
  - 由于该模块在当前配置中未被使用，已在 `mmdet3d/ops/__init__.py` 中注释掉 `feature_decorator` 的引用以绕过错误。
  - 同样在 `mmdet3d/models/backbones/__init__.py` 中注释掉了 `radar_encoder` 的引用。

---

**日期:** 2026-02-17-14:20

**成功运行测试脚本**

- **问题修复**:
  
  - **Radar配置**: 修复 KeyError: 'radar'。在 configs/nuscenes/det/default.yaml 中移除了 LoadRadarPointsMultiSweeps 管道步骤及 Collect3D 中的 
  adar 键值。
  - **Pipeline格式**: 修复 TypeError。清理了配置文件中因注释产生的 None 列表项。
  - **维度不匹配**: 修复 RuntimeError。修改 mmdet3d/models/vtransforms/depth_lss.py，强制设置 dd_depth_features=False 以解决输入通道数（期望1，实际6）不匹配的问题。
- **运行验证**:
  - 使用命令成功启动：
    ```bash
    python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
    ```
  
  - 脚本成功开始处理数据批次，流程跑通。

---

**日期:** 2026-02-20-21:30

**Windows环境训练与测试修复**

- **训练脚本修复 (`tools/train.py` & `mmdet3d/apis/train.py`)**:
  - 增加了对非分布式（单GPU）环境的检测与支持。
  - 修复了 `num_gpus` 为 `None` 导致的 `TypeError`。
  - 在非分布式模式下正确使用 `MMDataParallel` 替代 `MMDistributedDataParallel`。

- **配置调整 (`configs/nuscenes/default.yaml` & `configs/default.yaml`)**:
  - 将 `samples_per_gpu` 从 4 降低到 1，解决 8GB 显存下的 CUDA OOM 问题。
  - 禁用了 `TensorboardLoggerHook`，规避因 Python 环境版本导致的 `distutils` 兼容性错误。

- **代码逻辑修复 (Tensor类型匹配)**:
  - **`mmdet3d/core/bbox/assigners/hungarian_assigner.py`**: 强制将 `gt_labels` 转换为 `long` 类型，修复 `IndexError`。
  - **`mmdet3d/models/heads/bbox/transfusion.py`**: 在计算损失时将 `gt_labels_3d` 转换为 `long` 类型，修复 `RuntimeError: Index put requires ...` 错误。

- **验证结果**:
  - **训练**: `train.py` 成功运行超过 450 个迭代，Loss 正常下降。
  - **测试**: `test.py` 成功加载预训练模型并完成全量评估，NDS 指标为 0.5803。


---

**日期:** 2026-02-21-16:20

**项目路径变更后的环境修复与VS Code调试配置**

- **问题描述**: 
  - 项目文件夹重命名导致之前的编译产物路径失效，运行 `test.py` 报错 `ModuleNotFoundError: No module named 'mmcv'` 以及大量乱码垃圾文件。
  - Windows 环境下运行 python 脚本出现 `UnicodeDecodeError: 'gbk' codec can't decode` 编码错误。

- **环境修复**:
  - **清理旧产物**: 删除了 `build/`, `mmdet3d.egg-info/`, `dist/` 文件夹，清除包含旧路径的编译缓存。
  - **重新编译**: 在新路径下运行 `pip install -e .` 重新编译安装 `mmdet3d` 的 CUDA 扩展。

- **VS Code 调试配置**:
  - 创建了 `.vscode/launch.json` 配置文件。
  - 新增配置 **"Python: Test BEVFusion"**，支持一键运行测试脚本。
  - **编码修复**: 在 launch 配置的 `env` 中添加 `"PYTHONUTF8": "1"`，强制使用 UTF-8 编码，彻底解决 Windows 下读取配置文件时的 GBK 解码错误。

- **验证结果**:
  - 通过 VS Code 成功启动测试脚本：
    ```json
    "args": [
        "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
        "pretrained/bevfusion-det.pth",
        "--eval", "bbox"
    ]
    ```
  - 脚本成功运行并输出评估结果，NDS 指标稳定在 ~0.58。



---

**日期:** 2026-02-21-16:46

**成功解决 Windows 编码与环境问题**

- **问题描述**: Windows 默认使用 GBK 编码，导致读取 UTF-8 格式的 YAML 配置文件时报错 \UnicodeDecodeError: 'gbk' codec can't decode\.
- **解决方案**: 设置环境变量 \PYTHONUTF8=1\ 强制 Python 使用 UTF-8 编码。
- **运行指令**:
  
  ```powershell
  $env:PYTHONUTF8 = "1" #1. 设置环境变量，强制 Python 使用 UTF-8
  
  python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox
  ```
- **验证结果**: 测试脚本成功运行，NDS 指标为 0.5800。

**VS Code 调试配置说明**

- 已在 \.vscode/launch.json\ 中添加配置 \Python: Test BEVFusion\。
- 该配置自动包含 \nv: { "PYTHONUTF8": "1" }\，无需手动设置环境变量。

