# 工具清理总结

## 已删除的工具

### 1. TRT 混合评估工具（已不用）
- `tools/trt_eval_hybrid_all.py` ❌ 已删除
- **原因**：Round 7-9 的 Log2 量化实验不再使用混合 TRT 评估
- **影响**：需要在文档中移除相关引用

### 2. 可视化工具（从未使用）
- `tools/visualize.py` ❌ 已删除
- **原因**：检查所有文档，未被任何实验使用
- **影响**：无，可以直接删除

### 3. 自动化脚本（过时）
- `tools/scripts/download_pretrained.sh` ❌ 已删除
- `tools/scripts/train_resnet50_server.sh` ❌ 已删除
- `tools/scripts/lab_server_deploy.sh` ❌ 已删除
- `tools/scripts/autodl_setup.sh` ❌ 已删除
- **原因**：ResNet-50 不是主攻方向，脚本已过时
- **影响**：需要在文档中移除相关引用

---

## 保留的核心工具

### 主要工具（必需）
- ✅ `tools/quant_ptq_minmax.py` — **核心 PTQ 工具**（2000+ 行）
- ✅ `tools/test.py` — **FP32 基线评估**（必需）
- ✅ `tools/train.py` — 模型训练（备用功能）

### 辅助工具
- ✅ `tools/quant_benchmark.py` — 性能基准测试
- ✅ `tools/diag_lidar_distribution.py` — lidar 分布诊断
- ✅ `tools/vis_channel_distribution.py` — 通道分布可视化
- ✅ `tools/prepare_nuscenes_data.py` — 数据准备
- ✅ `tools/create_data.py` — 数据创建
- ✅ `tools/data_converter/` — 数据转换工具集

---

## 文档更新建议

### 需要从 REPORT.md 移除的内容

**§5.3 `tools/trt_eval_hybrid_all.py` 章节**（第 355 行）
- 可以完全删除此节
- 或者简化为："TRT Hybrid 评估已废弃，聚焦量化仿真研究"

**§5.5 `tools/trt_eval_hybrid.py` 章节**（第 409 行）
- 删除，因为不再使用单模块 TRT 评估

**相关 TRT 引用**：
- 第 518 行：`trt_eval_hybrid_all.py` 不支持逐模块精度
- 第 619 行：`trt_eval_hybrid_all.py` 余弦相似度验证
- 第 1186 行：添加计时代码建议
- 第 1189 行：逐模块精度设置建议

### 需要从 SERVER_DEPLOY.md 移除的内容

**Step 0 打包脚本**（第 57、60 行）
```bash
# 删除这些行：
tools/trt_eval_hybrid_all.py \
tools/scripts/ \
```

**Step 1 脚本修复**（第 104 行）
```bash
# 删除：
sed -i 's/\r//' tools/scripts/*.sh
```

---

## 清理后的工具状态

### 当前 tools/ 目录（核心功能）
```
tools/
├── quant_ptq_minmax.py      # 核心量化工具 ⭐
├── quant_benchmark.py         # 性能测试
├── test.py                    # FP32 评估 ⭐
├── train.py                   # 训练（备用）
├── diag_lidar_distribution.py # lidar 诊断
├── vis_channel_distribution.py # 可视化
├── prepare_nuscenes_data.py   # 数据准备
├── create_data.py             # 数据创建
└── data_converter/            # 数据转换
```

### 聚焦当前研究方向
- ✅ **量化仿真研究**：`quant_ptq_minmax.py` + `quant_benchmark.py`
- ✅ **FP32 基线**：`test.py`
- ✅ **诊断分析**：`diag_lidar_distribution.py` + `vis_channel_distribution.py`
- ❌ **不再使用**：TRT 混合评估、可视化脚本、自动化脚本

---

**清理时间**：2026-03-17
**删除文件**：4 个（1 个 Python + 3 个脚本）
**保留文件**：9 个（聚焦核心量化研究）
**文档影响**：需更新 REPORT.md 和 SERVER_DEPLOY.md 中的相关引用
