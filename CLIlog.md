# 注意事项

**cl 没有放在 系统环境变量里面。**如果涉及cl的操作记得运行类似的指令：

cmd /c "call `"D:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`" && powershell"

如果修改了 C++ / CUDA 代码，或者需要重新运行 `setup.py develop`，您需要确保编译器环境可见。

```powershell
# 设置构建变量 (仅编译时需要)
$env:DISTUTILS_USE_SDK=1
$env:MSSdk=1
```





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

