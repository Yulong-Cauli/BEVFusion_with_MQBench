# Phase 3: BEV Pooling TRT Plugin - 进行中交接文档

**日期**: 2026-03-25  
**交接人**: Phase 3 Agent

---

## 0. 核心约束（Agent 必读，不得违背）

**必须全 TRT 部署，禁止 PyTorch 混合方案。**

在边缘设备上保留 PyTorch 运行 bev_pool，量化部署毫无意义。
任何"depthnet 导出 TRT + bev_pool 留在 PyTorch 端"的方案均不可接受，
必须找到让 bev_pool 完整进入 TRT 引擎的路径。

---

## 1. 实际完成状态

| 任务 | 状态 | 说明 |
|------|------|------|
| BEVPoolV2 Plugin 源码 | ✅ | 完整实现，编译通过，C++ 测试通过 |
| depthnet ONNX 导出 | ⚠️ | 5.5 MB，24 Q/DQ，引擎 1.6 MB（仅 depthnet，不可单独使用） |
| **bev_pool → bev_pool_v2 模型替换** | 🔄 | **进行中** - 正在修改 vtransform 代码 |
| **完整 vtransform 导出（含 bev_pool）** | ❌ | **未完成** - 依赖 bev_pool_v2 替换完成 |
| **BEVPoolV2 Plugin 接口适配** | 🔄 | **进行中** - Plugin 已有，接口需与项目代码对齐 |

---

## 2. 核心问题分析

### 问题 1: 原始 bev_pool 不可量化也不可 trace

`BaseTransform.bev_pool` 包含动态控制流：
- `geom_feats.long()` - 离散化
- `x = x[kept]` - 布尔索引（动态 shape）
- `torch.cat(x.unbind(dim=2), 1)` - Z 轴 collapse

这些操作无法 trace，也无法插入 Q/DQ 节点。这是选择替换为 `bev_pool_v2` 的根本原因。

### 问题 2: BEVPoolV2 Plugin 接口与项目当前 bev_pool 不匹配

| 项目 | 项目当前使用 (mmdet3d) | Plugin 实现 (BEVDet) |
|------|-------------------|---------------------|
| 函数 | `bev_pool(x, geom_feats, B, D, H, W)` | `bev_pool_v2(depth, feat, ranks_*, intervals_*)` |
| 输入 | 展平特征 [N,C] + 坐标 [N,4] | 7 个张量（含预计算索引） |
| 算法 | scatter-based（不可量化） | interval sum-based（可量化） |

**结论**：不能直接用 Plugin 替换。需要先修改模型代码，切换为 bev_pool_v2 的调用接口，
然后 Plugin 才能对上。

### 问题 3: depthnet ONNX 不是完整 vtransform

目前已导出的 `vtransform_int8.onnx` 只包含：
```
vtransform.get_cam_feats(x, d)
├── dtransform (深度图下采样)
├── depthnet (深度预测 + 特征提取)
└── 外积 (depth × features) → [B,N,C,D,H,W]
```

缺失的 `bev_pool`（在 `BaseTransform.forward` 中调用）才是核心投影步骤。
**这个 ONNX 不能单独用于推理**，必须等完整导出完成。

---

## 3. 当前工作：bev_pool → bev_pool_v2 替换

### 替换目标
将 `mmdet3d/models/vtransforms/` 中对 `bev_pool` 的调用，替换为 `bev_pool_v2` 接口。
`bev_pool_v2` 使用预计算的 rank/interval 索引做 interval-sum，可以静态 trace，
且可以插入量化节点。

### 需要修改的文件
- `mmdet3d/models/vtransforms/lsstransform.py`（或对应的 BaseTransform）
- 预计算逻辑：在 `__init__` 或 `create_frustum` 时预先生成 `ranks_depth`、`ranks_feat`、
  `ranks_bev`、`interval_starts`、`interval_lengths`
- 前向传播：将 `self.bev_pool(x, geom_feats, ...)` 改为 `bev_pool_v2(depth, feat, ranks_*, ...)`

### 验证节点
1. 替换后 PyTorch 前向输出与替换前 cosine_sim > 0.999（确认等价性）
2. 替换后可成功 `torch.onnx.export`，ONNX 中出现 `custom::BEVPoolV2` 节点
3. TRT 引擎构建成功
4. TRT 输出与 PyTorch 输出 cosine_sim > 0.999

---

## 4. 交付物清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `tools/trt_plugins/bev_pool_v2/` | Plugin 源码 | ✅ 编译通过，接口待适配 |
| `tools/export_utils/export_vtransform.py` | 导出脚本 | ⚠️ 当前只导出 depthnet，待更新 |
| `vtransform_int8.onnx` | depthnet-only ONNX | ⚠️ 不完整，仅供参考 |
| `vtransform_int8.engine` | depthnet-only 引擎 | ⚠️ 不完整，仅供参考 |

---

## 5. Phase 3 接收 Agent 待办

1. 完成 vtransform 代码中 `bev_pool` → `bev_pool_v2` 的替换
2. 验证替换后 PyTorch 精度（cosine_sim > 0.999）
3. 在 `lss.py` 或对应文件中注册 `bev_pool_v2` ONNX Symbolic（`custom::BEVPoolV2`）
4. 适配 BEVPoolV2 Plugin 接口，确保与步骤 1 的调用方式一致
5. 导出完整 `vtransform_int8.onnx`（含 `custom::BEVPoolV2` 节点）
6. 构建 `vtransform_int8.engine`
7. **精度验证**：TRT 输出 vs PyTorch 输出，cosine_sim > 0.999（必须做，见 NEXT_PLAN.md）

---

## 6. 工作流说明

**所有开发和运行均在服务器上直接进行**，不需要本地开发后上传。

```bash
# 每次开始前：
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
```

服务器：`yellowstone@10.129.51.101`  
项目路径：`/media/yellowstone/data2/CYL/BEVFusion_with_MQBench`  
Conda 环境：`bevfusion_mqbench`