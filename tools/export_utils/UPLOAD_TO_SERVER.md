# Phase 1 服务器部署命令

## 📋 本地准备和上传 (PowerShell)

```powershell
# ===== 在本地 PowerShell 执行 =====

# 1. 环境初始化
$env:PYTHONUTF8="1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
conda activate bevfusion
cd D:\Research\Replication\BEVFusion_with_MQBench

# 2. 打包 Phase 1 部署工具
# 创建临时目录结构
mkdir -p temp_upload
mkdir -p temp_upload/tools/export_utils
mkdir -p temp_upload/docs

# 复制核心文件
Copy-Item tools/export_utils/build_engine.py temp_upload/tools/export_utils/
Copy-Item tools/export_utils/get_static_pad_values.py temp_upload/tools/export_utils/
Copy-Item tools/export_utils/mqbench_onnx_symbolic.py temp_upload/tools/export_utils/
Copy-Item tools/export_utils/phase1_swin_export.py temp_upload/tools/export_utils/
Copy-Item tools/export_utils/setup_phase1.sh temp_upload/tools/export_utils/
Copy-Item tools/export_utils/README.md temp_upload/tools/export_utils/

# 复制文档
Copy-Item docs/archive/PHASE1_SUMMARY.md temp_upload/docs/

# 创建压缩包
tar -czf phase1_deployment.tar.gz -C temp_upload .
Remove-Item -Recurse -Force temp_upload

# 3. 上传到服务器
scp phase1_deployment.tar.gz yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/
```

## 🚀 服务器部署 (SSH)

```bash
# ===== 在服务器上执行 =====

# 1. 连接到服务器
ssh yellowstone@10.129.51.101
# 密码: wave12968

# 2. 环境初始化
conda activate bevfusion_mqbench
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# 3. 解压部署工具
tar -xzf phase1_deployment.tar.gz
rm phase1_deployment.tar.gz

# 4. 验证文件结构
ls -la tools/export_utils/
ls -la docs/archive/PHASE1_SUMMARY.md

# 5. 运行环境验证
bash tools/export_utils/setup_phase1.sh
```

## ⚡ 快速执行 Phase 1

### 方式一：完整一键执行（推荐）

```bash
# 在服务器上执行（在 tmux 中）
tmux new-session -d -s bevfusion_phase1
tmux send-keys -t bevfusion_phase1 "conda activate bevfusion_mqbench" C-m
tmux send-keys -t bevfusion_phase1 "cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench" C-m
tmux send-keys -t bevfusion_phase1 "export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:\$LD_LIBRARY_PATH" C-m
tmux send-keys -t bevfusion_phase1 "python tools/export_utils/phase1_swin_export.py 2>&1 | tee logs/phase1_swin_export.log" C-m

# 查看进度
tmux attach -t bevfusion_phase1

# 分离会话（保持后台运行）
# 按 Ctrl+B 然后按 D
```

### 方式二：分步执行

```bash
# Step 1.1: 获取静态 pad 值
python tools/export_utils/get_static_pad_values.py 2>&1 | tee logs/phase1_step1.1_pad_values.log

# Step 1.3: 导出 ONNX 模型
python tools/export_utils/phase1_swin_export.py --skip-pad-detection 2>&1 | tee logs/phase1_step1.3_onnx_export.log

# Step 1.4: 构建 TensorRT 引擎
python tools/export_utils/build_engine.py \
    --onnx swin_int8.onnx \
    --engine swin_int8.engine \
    --int8 --fp16 2>&1 | tee logs/phase1_step1.4_build_engine.log
```

## 📊 验证结果

```bash
# 检查生成的文件
ls -lh swin_int8.onnx
ls -lh swin_int8.engine

# 查看日志
tail -100 logs/phase1_swin_export.log
tail -100 logs/phase1_step1.4_build_engine.log
```

## 🔍 故障排查

### 如果环境验证失败

```bash
# 检查 Conda 环境
conda info --envs

# 检查 Python 依赖
python -c "import torch; print(torch.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"

# 检查项目文件
ls -la configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
ls -la tools/quant_ptq_minmax.py
```

### 如果 ONNX 导出失败

```bash
# 检查量化权重
ls -la pretrained/ptq_minmax_model.pth

# 手动测试导入
python -c "import tools.export_utils.mqbench_onnx_symbolic"
```

### 如果 TensorRT 引擎构建失败

```bash
# 检查 ONNX 文件有效性
python -c "import onnx; onnx.checker.check_model('swin_int8.onnx')"

# 检查 TensorRT 版本
python -c "import tensorrt as trt; print(trt.__version__)"

# 查看详细日志
# 在 build_engine.py 中已经设置为 VERBOSE，会输出详细信息
```

## 📝 下载结果到本地

```powershell
# ===== 在本地 PowerShell 执行 =====

# 下载 ONNX 模型
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/swin_int8.onnx D:\Research\Replication\BEVFusion_with_MQBench\

# 下载 TensorRT 引擎
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/swin_int8.engine D:\Research\Replication\BEVFusion_with_MQBench\

# 下载日志
scp yellowstone@10.129.51.101:/media/yellowstone/data2/CYL/BEVFusion_with_MQBench/logs/phase1_*.log D:\Research\Replication\BEVFusion_with_MQBench\logs\
```

## 🎯 预期执行时间

- **环境验证**: ~1 分钟
- **Pad 值获取**: ~1 分钟
- **ONNX 导出**: ~5-10 分钟
- **引擎构建**: ~10-20 分钟（首次）
- **总计**: ~20-35 分钟

## ⚠️ 重要注意事项

1. **永远不要用 `nohup ... &`**：使用 tmux 直接运行，可以看到实时输出
2. **打包上传**：将多个文件打包后一次性上传，避免多次 SSH 连接
3. **日志记录**：所有命令都使用 `2>&1 | tee` 同时输出到屏幕和日志文件
4. **GPU 使用**：如果需要指定 GPU，使用 `CUDA_VISIBLE_DEVICES=X` 环境变量
5. **断线处理**：tmux 会话保持运行，断线后重新 `tmux attach` 即可

## 🔗 相关文档

- **完整计划**: `docs/NEXT_PLAN.md`
- **Phase 1 总结**: `docs/archive/PHASE1_SUMMARY.md`（已归档，总入口请见 `docs/HANDOFF_MASTER.md`）
- **工具使用**: `tools/export_utils/README.md`
- **部署手册**: `docs/SERVER_DEPLOY.md`

---

**准备就绪！** 现在你可以按照上述命令开始 Phase 1 的服务器部署了。