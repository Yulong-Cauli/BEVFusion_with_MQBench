"""
离线预计算 BEV pooling 索引。

对于固定相机参数的部署场景（如 nuScenes），bev_pool 的索引
（geom_feats, interval_starts, interval_lengths）只依赖相机参数，
不依赖图像内容，可以一次性预计算后复用。

用法:
    python tools/prepare_bev_indices.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --checkpoint pretrained/bevfusion-det.pth \
        --output bev_indices.pth

输出:
    bev_indices.pth 包含:
        - geom_feats: [N_kept, 4] 排序后的体素坐标 (int)
        - interval_starts: [M] 每个 interval 的起始索引 (int)
        - interval_lengths: [M] 每个 interval 的长度 (int)
        - kept: [Nprime] 布尔掩码
        - sort_indices: [N_kept] 排序索引
        - B, D, H, W: 输出维度 (batch, Z, X, Y)
        - nx: [3] BEV 网格尺寸 [X, Y, Z]
        - C: 通道数
"""
import argparse
import sys
import os
import logging

sys.path.insert(0, os.getcwd())

import torch
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import get_root_logger, recursive_eval
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset


def main():
    parser = argparse.ArgumentParser(description="预计算 BEV pooling 索引")
    parser.add_argument("--config", required=True, help="模型配置文件")
    parser.add_argument("--checkpoint", required=True, help="预训练权重")
    parser.add_argument("--output", default="bev_indices.pth", help="输出文件路径")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="用于提取相机参数的样本索引")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)

    # 加载配置
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # 构建模型
    model = build_model(cfg.model).cuda().eval()

    # 加载权重
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # 构建数据集，取一个样本
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, shuffle=False, dist=False
    )

    # 取指定样本
    sample = None
    for i, data in enumerate(dataloader):
        if i == args.sample_idx:
            sample = data
            break
    assert sample is not None, f"Sample index {args.sample_idx} not found"

    vtransform = model.encoders.camera.vtransform
    vtransform.eval()

    # 提取相机参数并计算 geometry
    # mmdet3d DataContainer 需要 .data 取出 tensor
    def _to_cuda(val):
        if hasattr(val, 'data'):
            val = val.data
        if isinstance(val, list):
            val = val[0]  # batch dim wrapped in list
        return val.cuda()

    with torch.no_grad():
        camera2lidar = _to_cuda(sample["camera2lidar"])
        img_aug_matrix = _to_cuda(sample["img_aug_matrix"])
        lidar_aug_matrix = _to_cuda(sample["lidar_aug_matrix"])

        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # DepthLSSTransform 用 cam_intrinsic, LSSTransform 用 camera_intrinsics
        if "cam_intrinsic" in sample:
            intrins = _to_cuda(sample["cam_intrinsic"])[..., :3, :3]
        else:
            intrins = _to_cuda(sample["camera_intrinsics"])[..., :3, :3]

        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = vtransform.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        B = camera2lidar.shape[0]
        indices = vtransform.precompute_bev_indices(geom, B)

    # 保存
    save_dict = {
        "geom_feats": indices["geom_feats"].cpu(),
        "interval_starts": indices["interval_starts"].cpu(),
        "interval_lengths": indices["interval_lengths"].cpu(),
        "kept": indices["kept"].cpu(),
        "sort_indices": indices["sort_indices"].cpu(),
        "B": indices["B"],
        "D": indices["D"],
        "H": indices["H"],
        "W": indices["W"],
        "nx": vtransform.nx.cpu(),
        "C": vtransform.C,
    }

    torch.save(save_dict, args.output)

    logger.info(f"Saved BEV indices to: {args.output}")
    logger.info(f"  geom_feats: {save_dict['geom_feats'].shape}")
    logger.info(f"  intervals: {save_dict['interval_starts'].shape[0]}")
    logger.info(f"  kept: {save_dict['kept'].sum()}/{save_dict['kept'].shape[0]} "
                f"({save_dict['kept'].float().mean():.2%})")
    logger.info(f"  B={save_dict['B']}, D(Z)={save_dict['D']}, "
                f"H(X)={save_dict['H']}, W(Y)={save_dict['W']}")


if __name__ == "__main__":
    main()
