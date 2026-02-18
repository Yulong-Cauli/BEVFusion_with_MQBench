# 环境部署总结

## 状态: 已完成

BEVFusion 环境已在 Windows 上成功设置并编译。

## 已完成步骤

1.  **创建 Conda 环境**: evfusion (Python 3.8).
2.  **安装 PyTorch**: PyTorch 1.10.1 with CUDA 11.3 (via conda).
3.  **安装依赖项**: mmcv-full, mmdet, 
uscenes-devkit 等.
4.  **安装 Visual Studio 构建工具**: 用户已安装 VS 2022 Build Tools.
5.  **编译 BEVFusion**:
    - 解决了 setup.py 中的 MSVC/CUDA 版本不匹配问题 (-allow-unsupported-compiler).
    - 修复了 MSVC STL 版本检查问题 (_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH).
    - 修复了 PyTorch 的 NumericLimits.cuh 文件以解决 Windows 上的 constexpr double inf 错误.
    - 成功运行 setup.py develop.
6.  **验证**: import mmdet3d 运行成功.

## 未来注意事项

- **PyTorch 修改**: 文件 D:\aconda\envs\bevfusion\lib\site-packages\torch\include\ATen\cuda\NumericLimits.cuh 已被修改为使用 std::numeric_limits<double>::infinity()。如果您重新安装 PyTorch，可能需要再次应用此修复。
- **Setup.py**: setup.py 包含针对此 Windows/CUDA 环境的特定标志。
