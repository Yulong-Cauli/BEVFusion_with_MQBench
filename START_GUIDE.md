cl在这里：没有放到 系统环境变量 里面，而是作为一个临时变量打开窗口的
```powershell
cmd /c "call `"D:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat`" && powershell"

```





# BEVFusion 启动指南 (重启后)

## 1. 正常使用 (运行代码)

如果您只是运行代码（训练、测试），**不需要**再次设置复杂的环境变量或编译。编译好的文件已经保存在环境里了。

只需打开终端（PowerShell 或 CMD）并运行：

```powershell
conda activate bevfusion
# 然后就可以运行您的 python 脚本了
python tools/test.py ...
```

## 2. 如果需要重新编译

如果您修改了 C++ / CUDA 代码，或者需要重新运行 `setup.py develop`，您需要确保编译器环境可见。

推荐创建一个启动脚本 `start_compile_env.ps1`：

```powershell
# 1. 激活 Conda 环境
conda activate bevfusion

# 2. 设置构建变量 (仅编译时需要)
$env:DISTUTILS_USE_SDK=1
$env:MSSdk=1

# 3. 提示
Write-Host "编译环境已就绪。可以运行 python setup.py develop"
```

## 常见问题

- **ImportError: DLL load failed**: 如果遇到这个错误，通常是因为 Windows 找不到 CUDA 或 PyTorch 的 DLL。请确保系统 PATH 中包含 CUDA 路径（通常安装时会自动添加），或者重新安装 PyTorch。
