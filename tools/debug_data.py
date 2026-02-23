# ========== 数据调试脚本 ==========
# 用于单步跟踪数据流、检查字段和形状
# 使用方法：python tools/debug_data.py <config_file> --debug-samples 2

import argparse
import os
import sys
sys.path.append(os.getcwd())
import random
import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Debug data pipeline")
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--debug-samples", type=int, default=1, 
                       help="number of samples to debug (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def print_tensor_info(name, tensor, indent=0):
    """打印张量的详细信息"""
    prefix = "  " * indent
    if isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}:")
        print(f"{prefix}  shape: {tensor.shape}")
        print(f"{prefix}  dtype: {tensor.dtype}")
        print(f"{prefix}  device: {tensor.device}")
        print(f"{prefix}  min: {tensor.min():.4f}, max: {tensor.max():.4f}")
    elif isinstance(tensor, np.ndarray):
        print(f"{prefix}{name}:")
        print(f"{prefix}  shape: {tensor.shape}")
        print(f"{prefix}  dtype: {tensor.dtype}")
        print(f"{prefix}  min: {tensor.min():.4f}, max: {tensor.max():.4f}")
    elif isinstance(tensor, dict):
        print(f"{prefix}{name}: dict")
        for k, v in tensor.items():
            print_tensor_info(k, v, indent+1)
    elif isinstance(tensor, list):
        print(f"{prefix}{name}: list[{len(tensor)}]")
        for i, v in enumerate(tensor[:3]):  # 仅显示前3个
            print_tensor_info(f"[{i}]", v, indent+1)
    else:
        print(f"{prefix}{name}: {type(tensor).__name__}")


def main():
    """主调试函数"""
    args = parse_args()
    
    # ========== 设置随机种子 ==========
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ========== 加载配置 ==========
    print("=" * 80)
    print("【步骤1】加载配置文件")
    print("=" * 80)
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(f"✓ 配置加载成功")
    print(f"  数据集类型: {cfg.data.train.type}")
    print(f"  数据集路径: {cfg.data.train.dataset_root}")
    print(f"  目标类别数: {len(cfg.object_classes)}")
    print(f"  点云范围: {cfg.point_cloud_range}")
    print(f"  体素大小: {cfg.voxel_size}")
    
    # ========== 【断点1】构建数据集 ==========
    print("\n" + "=" * 80)
    print("【步骤2】构建训练数据集")
    print("=" * 80)
    dataset = build_dataset(cfg.data.train)
    print(f"✓ 数据集构建成功")
    print(f"  数据集大小: {len(dataset)} samples")
    print(f"  类别列表: {dataset.CLASSES}")
    print(f"  地图类别: {dataset.map_classes}")
    
    # ========== 【断点2】单个样本检查 ==========
    print("\n" + "=" * 80)
    print("【步骤3】检查单个样本数据结构")
    print("=" * 80)
    sample_idx = np.random.randint(0, len(dataset))
    print(f"随机选择样本索引: {sample_idx}")
    
    sample = dataset[sample_idx]
    print(f"\n样本中包含的字段:")
    for key in sorted(sample.keys()):
        print(f"  - {key}")
    
    print(f"\n【样本字段详情】")
    print_tensor_info("sample_data", sample, indent=0)
    
    # ========== 【断点3】批量样本检查（通过DataLoader） ==========
    print("\n" + "=" * 80)
    print("【步骤4】检查批量样本（DataLoader）")
    print("=" * 80)
    
    # 构建数据加载器
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2,  # 小批量用于调试
        workers_per_gpu=0,  # 单线程加载
        dist=False,
        shuffle=False,
    )
    print(f"✓ 数据加载器构建成功")
    print(f"  批次大小: 2")
    print(f"  总批次数: {len(data_loader)}")
    
    # 获取第一个批次
    print(f"\n获取第一个批次...")
    batch_data = next(iter(data_loader))
    
    print(f"\n【批次中的顶层键】")
    for key in sorted(batch_data.keys()):
        print(f"  - {key}")
    
    print(f"\n【批次数据详情】")
    for key in batch_data.keys():
        data = batch_data[key]
        if isinstance(data, torch.Tensor):
            print(f"{key}:")
            print(f"  shape: {data.shape}")
            print(f"  dtype: {data.dtype}")
            print(f"  device: {data.device}")
        elif isinstance(data, dict):
            print(f"{key}: dict with keys {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(data, list):
            print(f"{key}: list[{len(data)}]")
            if len(data) > 0 and isinstance(data[0], torch.Tensor):
                print(f"  第一个元素shape: {data[0].shape}")
    
    # ========== 【断点4】模型前向检查 ==========
    print("\n" + "=" * 80)
    print("【步骤5】模型前向检查")
    print("=" * 80)
    
    print(f"构建模型...")
    model = build_model(cfg.model)
    model.init_weights()
    model.eval()  # 评估模式
    print(f"✓ 模型构建成功")
    print(f"  模型类型: {model.__class__.__name__}")
    
    # 将batch数据移到CPU（便于调试）
    print(f"\n检查模型输入...")
    print(f"  img shape: {batch_data['img'].shape}")
    print(f"  points shape: {batch_data['points'].shape}")
    print(f"  gt_bboxes_3d 类型: {type(batch_data['gt_bboxes_3d'])}")
    
    # ========== 关键字段展示 ==========
    print("\n" + "=" * 80)
    print("【关键字段说明】")
    print("=" * 80)
    
    print("\n【img】- 相机图像")
    print(f"  格式: (B, C, H, W) = {batch_data['img'].shape}")
    print(f"  B: 批次大小，C: 通道数(6=3相机×2)，H: 高，W: 宽")
    print(f"  取值范围: [{batch_data['img'].min():.3f}, {batch_data['img'].max():.3f}]")
    
    print("\n【points】- 激光雷达点云")
    print(f"  格式: (N, 5) = {batch_data['points'].shape}")
    print(f"  N: 点数，5: [x, y, z, intensity, frame_id]")
    print(f"  坐标范围 X: [{batch_data['points'][:, 0].min():.2f}, {batch_data['points'][:, 0].max():.2f}]")
    print(f"  坐标范围 Y: [{batch_data['points'][:, 1].min():.2f}, {batch_data['points'][:, 1].max():.2f}]")
    print(f"  坐标范围 Z: [{batch_data['points'][:, 2].min():.2f}, {batch_data['points'][:, 2].max():.2f}]")
    
    print("\n【gt_bboxes_3d】- 3D标注框")
    print(f"  类型: {type(batch_data['gt_bboxes_3d']).__name__}")
    print(f"  数量: {len(batch_data['gt_bboxes_3d'])}")
    
    print("\n【gt_labels_3d】- 3D标签")
    print(f"  形状: {batch_data['gt_labels_3d'].shape}")
    print(f"  类别ID: {sorted(set(batch_data['gt_labels_3d'].tolist()))}")
    
    print("\n【meta】- 元信息（坐标变换矩阵）")
    meta = batch_data['img_metas'][0]  # 第一个样本的元信息
    print(f"  包含的变换矩阵:")
    for key in ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 
                'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix']:
        if key in meta:
            if isinstance(meta[key], list):
                print(f"    {key}: list")
            elif isinstance(meta[key], torch.Tensor):
                print(f"    {key}: shape={meta[key].shape}")
            else:
                print(f"    {key}: {type(meta[key])}")
    
    # ========== 调试建议 ==========
    print("\n" + "=" * 80)
    print("【调试建议】")
    print("=" * 80)
    print("""
1. 在IDE中设置断点的位置：
   - Line: build_dataset() 之后
   - Line: build_dataloader() 之后  
   - Line: batch_data 获取后
   
2. 调试时查看的关键信息：
   - batch_data.keys() - 所有字段
   - batch_data['img'].shape - 图像形状
   - batch_data['points'].shape - 点云形状
   - batch_data['img_metas'][0].keys() - 元信息
   - batch_data['gt_bboxes_3d'] - 3D框
   
3. 在IDE Python控制台中可以执行：
   - len(batch_data['gt_labels_3d']) - 检查有多少个目标
   - batch_data['points'][:5] - 查看前5个点
   - batch_data['img_metas'][0]['camera_intrinsics'] - 相机内参
    """)
    
    print("=" * 80)
    print("✓ 调试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
