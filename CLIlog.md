# 注意事项

**cl 没有放在 系统环境变量里面。**如果涉及cl的操作记得运行类似的指令：

cmd /c "call `"D:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`" && powershell"





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

