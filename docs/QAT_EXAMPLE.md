# MQBench QAT Integration Example Code Snippets
# MQBench QAT 集成示例代码片段

## 核心代码示例 (Core Code Examples)

本文档提供了将 MQBench 集成到 BEVFusion 的核心代码片段。

---

## 1. 加载浮点 BEVFusion 模型 (Load FP32 BEVFusion Model)

```python
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs

# 加载配置文件
configs.load('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml', recursive=True)
cfg = Config(recursive_eval(configs))

# 构建模型
model = build_model(cfg.model)
model.init_weights()

# 可选：加载预训练权重
if pretrained_path:
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
```

---

## 2. 定义 TensorRT 后端配置 (Define TensorRT Backend Config)

```python
from mqbench.prepare_by_platform import BackendType

def get_backend_config_for_tensorrt():
    """
    为 TensorRT Int8 后端配置 MQBench
    
    TensorRT 量化策略:
    - Per-channel 对称量化（权重）
    - Per-tensor 量化（激活）
    - INT8 推理优化
    """
    return BackendType.Tensorrt
```

---

## 3. 获取 mmdetection3d 自定义算子的 Leaf Module 列表

```python
def get_leaf_modules_for_mmdet3d():
    """
    返回需要作为 leaf modules 的 mmdetection3d 自定义算子列表
    
    这些模块包含 CUDA 扩展，torch.fx 无法追踪
    """
    leaf_modules = []
    
    # 1. BEV Pooling - 核心算子
    from mmdet3d.ops.bev_pool.bev_pool import QuickCumsum, QuickCumsumCuda
    leaf_modules.extend([QuickCumsum, QuickCumsumCuda])
    
    # 2. Sparse Convolution - 3D 稀疏卷积
    from mmdet3d.ops.spconv import SparseModule, SparseConvolution, SparseMaxPool
    from mmdet3d.ops.spconv import SparseSequential, ToDense
    from mmdet3d.ops.sparse_block import SparseBasicBlock, SparseBottleneck
    leaf_modules.extend([
        SparseModule, SparseConvolution, SparseMaxPool,
        SparseSequential, ToDense, SparseBasicBlock, SparseBottleneck
    ])
    
    # 3. Voxelization - 点云体素化
    from mmdet3d.ops.voxel import Voxelization
    from mmdet3d.ops.voxel.scatter_points import DynamicScatter
    leaf_modules.extend([Voxelization, DynamicScatter])
    
    # 4. ROI Aware Pool3D
    from mmdet3d.ops.roiaware_pool3d import RoIAwarePool3d
    leaf_modules.append(RoIAwarePool3d)
    
    # 5. Point Cloud Sampling
    from mmdet3d.ops.furthest_point_sample import (
        Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler
    )
    leaf_modules.extend([Points_Sampler, DFPS_Sampler, FFPS_Sampler, FS_Sampler])
    
    # 6. PAConv
    from mmdet3d.ops.paconv import PAConv, ScoreNet
    leaf_modules.extend([PAConv, ScoreNet])
    
    # 7. Group Points
    from mmdet3d.ops.group_points import QueryAndGroup, GroupAll
    leaf_modules.extend([QueryAndGroup, GroupAll])
    
    # 8. View Transformers
    from mmdet3d.models.vtransforms import BaseTransform, LSSTransform
    leaf_modules.extend([BaseTransform, LSSTransform])
    
    return leaf_modules
```

---

## 4. 使用 prepare_by_platform 应用量化 (Apply Quantization with prepare_by_platform)

```python
from mqbench.prepare_by_platform import prepare_by_platform

def prepare_model_for_qat(model, backend_type, leaf_modules):
    """
    使用 MQBench 准备模型进行 QAT
    
    Args:
        model: 原始浮点模型
        backend_type: 后端类型 (e.g., BackendType.Tensorrt)
        leaf_modules: 需要作为 leaf 处理的模块列表
    
    Returns:
        准备好进行 QAT 的模型
    """
    # 配置额外的量化参数
    extra_quantizer_dict = {
        'additional_module_type': tuple(leaf_modules) if leaf_modules else (),
    }
    
    # 准备模型进行 QAT
    model = prepare_by_platform(
        model,
        backend_type,
        extra_quantizer_dict=extra_quantizer_dict,
    )
    
    return model

# 使用示例
backend_type = get_backend_config_for_tensorrt()
leaf_modules = get_leaf_modules_for_mmdet3d()
model = prepare_model_for_qat(model, backend_type, leaf_modules)
```

---

## 5. 设置训练循环进行微调 (Setup Training Loop for Fine-tuning)

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def setup_qat_training(model, train_dataset, cfg):
    """
    设置 QAT 训练
    
    Args:
        model: 准备好进行 QAT 的模型
        train_dataset: 训练数据集
        cfg: 配置对象
    """
    # 1. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=True,
        num_workers=cfg.data.workers_per_gpu,
    )
    
    # 2. 配置优化器（使用较小的学习率）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  # QAT 通常使用较小的学习率
        weight_decay=0.01
    )
    
    # 3. 配置学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs,
        eta_min=1e-6
    )
    
    return train_loader, optimizer, scheduler

# 训练循环示例
def train_qat_epoch(model, train_loader, optimizer, device):
    """单个 epoch 的 QAT 训练"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 将数据移到 GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 前向传播
        losses = model(**batch)
        loss = sum(losses.values())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

---

## 6. 完整的 QAT 训练流程 (Complete QAT Training Pipeline)

```python
import torch
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization

def full_qat_pipeline(config_path, pretrained_path=None):
    """
    完整的 QAT 训练流程
    
    步骤:
    1. 加载浮点模型
    2. 插入量化节点
    3. (可选) Calibration
    4. QAT 微调
    5. 保存量化模型
    """
    # === 步骤 1: 加载配置和模型 ===
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    # === 步骤 2: 准备 QAT ===
    backend_type = BackendType.Tensorrt
    leaf_modules = get_leaf_modules_for_mmdet3d()
    
    extra_quantizer_dict = {
        'additional_module_type': tuple(leaf_modules),
    }
    
    model = prepare_by_platform(
        model,
        backend_type,
        extra_quantizer_dict=extra_quantizer_dict,
    )
    
    # === 步骤 3: 可选 - Calibration ===
    # 如果需要，可以先进行 calibration 收集统计信息
    # enable_calibration(model)
    # ... 运行若干 batch 的数据 ...
    # enable_quantization(model)
    
    # === 步骤 4: QAT 训练 ===
    model = model.cuda()
    train_dataset = build_dataset(cfg.data.train)
    train_loader, optimizer, scheduler = setup_qat_training(model, train_dataset, cfg)
    
    for epoch in range(cfg.max_epochs):
        avg_loss = train_qat_epoch(model, train_loader, optimizer, 'cuda')
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{cfg.max_epochs}, Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % cfg.checkpoint_config.interval == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoints/qat_epoch_{epoch+1}.pth')
    
    # === 步骤 5: 导出量化模型 ===
    # from mqbench.convert_deploy import convert_deploy
    # convert_deploy(model, BackendType.Tensorrt, dummy_input, 'quantized_model.onnx')
    
    return model

# 使用示例
if __name__ == '__main__':
    model = full_qat_pipeline(
        'configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml',
        pretrained_path='pretrained/bevfusion-det.pth'
    )
```

---

## 7. mmdetection3d 模型中通常需要设置为 Leaf 的模块列表总结

### 核心模块 (Core Modules)

1. **BEV Pooling** (必须)
   - `mmdet3d.ops.bev_pool.bev_pool.QuickCumsum`
   - `mmdet3d.ops.bev_pool.bev_pool.QuickCumsumCuda`

2. **Sparse Convolution** (LiDAR 处理必须)
   - `mmdet3d.ops.spconv.SparseModule`
   - `mmdet3d.ops.spconv.SparseConvolution`
   - `mmdet3d.ops.spconv.SparseMaxPool`
   - `mmdet3d.ops.spconv.SparseSequential`
   - `mmdet3d.ops.spconv.ToDense`

3. **Voxelization** (点云处理必须)
   - `mmdet3d.ops.voxel.Voxelization`
   - `mmdet3d.ops.voxel.scatter_points.DynamicScatter`

4. **View Transformers** (相机分支必须)
   - `mmdet3d.models.vtransforms.BaseTransform`
   - `mmdet3d.models.vtransforms.LSSTransform`

### 可选模块 (Optional Modules)

5. **ROI Operations**
   - `mmdet3d.ops.roiaware_pool3d.RoIAwarePool3d`

6. **Point Cloud Sampling**
   - `mmdet3d.ops.furthest_point_sample.Points_Sampler`
   - `mmdet3d.ops.furthest_point_sample.DFPS_Sampler`
   - `mmdet3d.ops.furthest_point_sample.FFPS_Sampler`
   - `mmdet3d.ops.furthest_point_sample.FS_Sampler`

7. **Point Adaptive Convolution**
   - `mmdet3d.ops.paconv.PAConv`
   - `mmdet3d.ops.paconv.ScoreNet`

8. **Point Grouping**
   - `mmdet3d.ops.group_points.QueryAndGroup`
   - `mmdet3d.ops.group_points.GroupAll`

---

## 关键注意事项 (Key Notes)

1. **Leaf Modules 的重要性**: 
   - 这些模块包含 CUDA 扩展，torch.fx 无法追踪
   - 必须显式标记为 leaf，否则会导致追踪错误

2. **后端配置**:
   - TensorRT: `BackendType.Tensorrt`
   - ONNX Runtime: `BackendType.ONNX_QNN`
   - Qualcomm SNPE: `BackendType.SNPE`

3. **训练建议**:
   - 学习率: 1e-4 到 2e-4 (是预训练的 1/10)
   - Epoch: 10-20 个 epoch
   - 精度下降: 应控制在 1-2% 以内

4. **调试技巧**:
   - 如果遇到追踪错误，检查是否有新的自定义算子
   - 使用小数据集先测试流程
   - 监控量化模型和浮点模型的输出差异

---

**完整实现**: 参见 `tools/quant_train.py`
