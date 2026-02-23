from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    """
    BEVFusion 多模态融合3D目标检测模型。
    
    融合摄像头、激光雷达、毫米波雷达三种传感器的数据，在BEV(鸟瞰图)空间进行
    特征融合，实现端到端的3D目标检测和BEV语义分割。
    
    Args:
        encoders (Dict[str, Any]): 各传感器编码器配置字典，包含camera/lidar/radar
        fuser (Dict[str, Any]): 特征融合模块配置
        decoder (Dict[str, Any]): 解码器配置，包含backbone和neck
        heads (Dict[str, Any]): 检测头配置，通常包含object和map两个head
        **kwargs: 其他参数，如loss_scale等
    """
    
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        # ==================== 摄像头编码器初始化 ====================
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            # 摄像头分支：backbone提取特征 -> neck融合多尺度 -> vtransform投影到BEV
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        
        # ==================== 激光雷达编码器初始化 ====================
        if encoders.get("lidar") is not None:
            # 根据voxelize方式选择：硬体素化(Voxelization)或动态散点(DynamicScatter)
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,  # 点云体素化
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),  # 3D卷积主干
                }
            )
            # 是否对体素特征进行平均池化降低维度
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # ==================== 毫米波雷达编码器初始化 ====================
        if encoders.get("radar") is not None:
            # 与激光雷达编码器结构相同
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        # ==================== 融合模块初始化 ====================
        # 融合来自多个传感器的BEV特征
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        # ==================== 解码器初始化 ====================
        # 在融合特征基础上进行上采样和进一步的特征提取
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),  # BEV backbone
                "neck": build_neck(decoder["neck"]),  # BEV neck
            }
        )
        
        # ==================== 检测头初始化 ====================
        # 可包含object(3D目标检测)和map(BEV分割)头
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        # ==================== 损失权重配置 ====================
        # 为不同的head分配损失权重
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # ==================== 深度损失配置 ====================
        # 判断是否使用BEVDepth等需要深度监督的vtransform方法
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in [
            'BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth'
        ]


        self.init_weights()

    def init_weights(self) -> None:
        """初始化模型权重，主要初始化摄像头backbone"""
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        """
        提取摄像头特征并投影到BEV空间。
        
        这个方法的核心流程：
        1. 图像CNN特征提取 (backbone + neck)
        2. 图像特征投影到BEV空间 (vtransform)
        3. 返回BEV特征或(BEV特征 + 深度损失)
        
        Args:
            x: 输入图像张量
               - 形状: [B, N, C, H, W]
               - B: batch size (通常为1在推理时)
               - N: 摄像头数量 (nuscenes中为6个)
               - C: 图像通道数 (通常为3)
               - H, W: 图像高宽 (1600x900等)
            
            points: 激光雷达点云，形状[batch_size, num_points, 4]
               - 用于建立图像坐标系到BEV坐标系的映射关系
               - 包含x, y, z, intensity四个信息
            
            radar_points: 毫米波雷达点云
               - 类似points，用于坐标变换参考
            
            camera2ego: 摄像头到自车坐标系的刚体变换矩阵
               - 形状: [B, N, 4, 4] 或 [B, 4, 4]
               - 用于将摄像头坐标系的点转换到自车坐标系
               
            lidar2ego: 激光雷达到自车坐标系的变换矩阵
               - 形状: [B, 4, 4]
               - 标准的激光雷达到自车的变换
               
            lidar2camera: 激光雷达到摄像头坐标系的变换矩阵
               - 形状: [B, N, 4, 4] 或 [B, 4, 4]
               - 反向变换：从LiDAR坐标系到相机坐标系
               
            lidar2image: 激光雷达到图像平面的投影矩阵
               - 形状: [B, N, 4, 4]
               - 将3D点投影到2D图像坐标
               
            camera_intrinsics: 摄像头内参矩阵
               - 形状: [B, N, 4, 4] 或 [B, N, 3, 3]
               - 包含焦距、主点等相机内参
               
            camera2lidar: 摄像头到激光雷达坐标系的变换矩阵
               - 形状: [B, N, 4, 4] 或 [B, 4, 4]
               - 反向变换：从相机坐标系到LiDAR/BEV坐标系
               
            img_aug_matrix: 图像增强变换矩阵（如旋转、裁剪）
               - 形状: [B, 4, 4]
               - 应用于图像平面的几何变换
               
            lidar_aug_matrix: 激光雷达增强变换矩阵
               - 形状: [B, 4, 4]
               - 应用于点云的几何变换
               
            img_metas: 图像元数据列表
               - 包含图像文件名、原始分辨率、pad信息等
               - 在vtransform中用于处理图像的裁剪/pad信息
               
            gt_depths: 地面真值深度图（可选）
               - 形状: [B, N, H, W]
               - 仅在use_depth_loss=True时需要
               - 用于BEVDepth等方法的深度监督
            
        Returns:
            - 如果use_depth_loss=False: torch.Tensor
              - BEV特征张量，形状[B, C_bev, H_bev, W_bev]
              - 高度：256像素，宽度：256像素（默认）
              - 通道数：通常64-256
              
            - 如果use_depth_loss=True: tuple
              - (BEV特征张量, 深度损失)
              - 深度损失是标量，用于反向传播
        """
        # ============ 步骤1：提取图像多尺度特征 ============
        B, N, C, H, W = x.size()
        # x 形状变化：[B, N, C, H, W]
        # 例如: [1, 6, 3, 1600, 900]
        # B=1, N=6个摄像头, C=3, H=1600, W=900
        
        # 将N个摄像头的图像合并到batch维度，便于并行处理
        x = x.view(B * N, C, H, W)
        # x 形状变化：[B*N, C, H, W]
        # 例如: [6, 3, 1600, 900]
        # 6张图像一起处理

        # ============ backbone特征提取 ============
        # backbone通常是ResNet系列，输出多尺度特征
        # 例如 FPN 会输出 [1/4, 1/8, 1/16, 1/32] 分辨率的特征
        x = self.encoders["camera"]["backbone"](x)
        # 输出：多尺度特征列表或张量
        # 例如: [feat_1/4, feat_1/8, feat_1/16, ...]
        # 每个特征的形状: [6, C, H/scale, W/scale]

        # ============ neck融合多尺度特征 ============
        # neck的作用：融合多尺度特征为统一尺度，增强特征表示力
        # 常见的neck：FPN, PAFPN等
        x = self.encoders["camera"]["neck"](x)
        # 输出：单一尺度或少数几个尺度的特征
        # 例如: [6, 256, 200, 112]
        # 即 [B*N, C_neck, H_neck, W_neck]

        # ============ 处理neck可能的list输出 ============
        # 某些neck实现会输出list，需要取第一个元素
        if not isinstance(x, torch.Tensor):
            x = x[0]
        # 确保x是张量: [B*N, C, H, W]
        # 例如: [6, 256, 200, 112]

        # ============ 步骤2：拆分batch和摄像头维度 ============
        BN, C, H, W = x.size()
        # 将特征拆分回原始的batch和摄像头维度
        x = x.view(B, int(BN / B), C, H, W)
        # x 形状变化：[B*N, C, H, W]
        # → [B, N, C, H, W]
        # 例如: [6, 256, 200, 112]
        # → [1, 6, 256, 200, 112]
        # 这样保持多摄像头的信息结构

        # ============ 步骤3：视图变换 (Image to BEV) ============
        # 这是整个方法的核心！将图像特征投影到BEV空间
        # vtransform实现了从image坐标系到BEV坐标系的转换
        x = self.encoders["camera"]["vtransform"](
            x,  # [B, N, C, H, W] 多摄像头特征
            points,  # 用于建立坐标系关联
            radar_points,  # 用于建立坐标系关联
            camera2ego,  # 相机→自车坐标变换
            lidar2ego,   # 激光雷达→自车坐标变换
            lidar2camera,  # 激光雷达→相机坐标变换（反向投影用）
            lidar2image,   # 激光雷达→图像平面投影
            camera_intrinsics,  # 相机内参
            camera2lidar,  # 相机→激光雷达坐标变换（关键！）
            img_aug_matrix,  # 图像增强矩阵
            lidar_aug_matrix,  # 点云增强矩阵
            img_metas,  # 图像元数据（如pad信息）
            depth_loss=self.use_depth_loss,  # 是否计算深度损失
            gt_depths=gt_depths,  # 地面真值深度
        )
        # ┌────────────────────────────────────────────┐
        # │ vtransform的内部工作流程：                   │
        # │                                             │
        # │ 1. 生成BEV网格点：                          │
        # │    grid = [H_bev, W_bev, 2]                │
        # │    范围通常是 [-51.2, 51.2] (nuscenes)     │
        # │    网格密度 0.2m/像素                       │
        # │                                             │
        # │ 2. 反向投影到相机图像：                     │
        # │    对每个BEV网格点，通过变换矩阵投影到各摄像头
        # │    grid_cam = grid @ lidar2camera          │
        # │    grid_img = grid_cam @ camera_intrinsics │
        # │                                             │
        # │ 3. 双线性插值采样：                         │
        # │    从多摄像头图像特征中采样对应位置的特征    │
        # │    feat_bev = bilinear_sample(feat_img, uv) │
        # │                                             │
        # │ 4. 多摄像头融合：                           │
        # │    对超出图像边界的点进行mask               │
        # │    多个摄像头覆盖的区域取加权平均或最大值    │
        # │                                             │
        # │ 5. 深度损失计算（可选）：                   │
        # │    如果use_depth_loss=True:                │
        # │    计算预测深度与gt_depths的L1损失          │
        # └────────────────────────────────────────────┘
        
        # 返回值处理
        # 输出形状：[B, C_bev, H_bev, W_bev]
        # 例如: [1, 80, 256, 256]
        # 如果use_depth_loss=True:
        # 输出: (feat_bev, depth_loss)
        # depth_loss是标量张量
        
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        """
        提取激光雷达/毫米波雷达特征。
        
        Args:
            x: 点云列表，每个元素为该帧的点云
            sensor: 传感器类型，"lidar" 或 "radar"
            
        Returns:
            编码后的特征张量
        """
        # 将点云体素化为体素特征、坐标和大小
        feats, coords, sizes = self.voxelize(x, sensor)
        # 从坐标推断batch size
        batch_size = coords[-1, 0] + 1
        # 通过3D backbone网络处理体素特征
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # extract_lidar_features 的流程说明：
    # 1) voxelize 将点云转换为体素特征(feats)、体素坐标(coords)和体素内点数(sizes)
    # 2) 通过 coords 推断 batch_size
    # 3) 使用 lidar backbone 将稀疏体素特征编码为BEV/稀疏特征
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        """
        将点云体素化为稀疏张量表示。
        
        Args:
            points: 批量点云列表，列表长度为batch size
            sensor: 传感器类型 "lidar" 或 "radar"
            
        Returns:
            feats: 体素特征，形状[num_voxels, C]
            coords: 体素坐标，形状[num_voxels, 4]，包含batch idx和3D坐标
            sizes: 每个体素内的点数，形状[num_voxels]
        """
        feats, coords, sizes = [], [], []
        # 逐帧处理点云
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # 硬体素化返回三元组：特征、坐标、大小
                f, c, n = ret
            else:
                assert len(ret) == 2
                # 动态散点返回二元组
                f, c = ret
                n = None
            feats.append(f)
            # 在坐标前添加batch索引
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        # 拼接所有batch的数据
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            # 如果启用voxelize_reduce，对每个体素内的特征求平均
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    def show_results(self, data, result, out_dir, show=False, **kwargs):
        """
        可视化检测结果的BEV视图。
        
        Args:
            data: 输入数据字典
            result: 模型输出的检测结果列表
            out_dir: 输出目录路径
            show: 是否实时显示
        """
        import mmcv
        from os import path as osp
        import numpy as np
        
        # We need to adapt the visualization based on mmdet3d utils
        # from mmdet3d.core import show_result
        from mmdet3d.core.bbox import get_box_type

        def show_result(
            points,
            gt_bboxes,
            pred_result,
            out_dir,
            filename,
            show=False,
            snapshot=False,
            pred_bboxes=None, 
        ):
            """
            绘制单个样本的BEV检测结果。
            """
            import cv2
            import matplotlib.pyplot as plt

            # 检查并解析预测结果格式
            if isinstance(pred_result, dict):
                # 从dict格式提取3D框、置信度和类别
                bboxes = pred_result.get('boxes_3d', None)
                scores = pred_result.get('scores_3d', None)
                labels = pred_result.get('labels_3d', None)
                
                if bboxes is not None:
                    # 转换为CPU numpy数组
                    bboxes = bboxes.tensor.cpu().numpy()
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
            else:
                return

            # 创建BEV可视化图
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_aspect('equal')
            
            # 绘制激光雷达点云（BEV投影）
            if points is not None:
                plt.scatter(points[:, 0], points[:, 1], s=0.5, c='gray', alpha=0.5)
            
            # 绘制检测框
            if bboxes is not None:
                for i in range(len(bboxes)):
                    # 过滤低置信度检测
                    if scores[i] < 0.3:
                        continue
                    
                    # 提取框的中心坐标
                    xc, yc = bboxes[i, 0], bboxes[i, 1]
                    
                    # 绘制框中心和标签
                    plt.plot(xc, yc, 'r+', markersize=10)
                    plt.text(xc, yc, f"{int(labels[i])}: {scores[i]:.2f}", 
                            color='red', fontsize=8)
            
            # 设置坐标轴范围（根据nuscenes数据集）
            plt.xlim(-54, 54)
            plt.ylim(-54, 54)
            
            # 保存可视化结果
            if out_dir:
                out_path = osp.join(out_dir, f"{filename}_bev.png")
                print(f"保存可视化结果到 {out_path}")
                plt.savefig(out_path)
                plt.close(fig)
        
        if out_dir:
            mmcv.mkdir_or_exist(out_dir)

        # ==================== 处理批量样本 ====================
        batch_size = len(result)
        for i in range(batch_size):
            # 提取该样本的元数据
            if 'metas' in data:
                img_metas = data['metas'].data[0][i]
            elif 'img_metas' in data:
                img_metas = data['img_metas'].data[0][i]
            else:
                continue
                
            # 提取点云数据
            if 'points' in data:
                points = data['points'].data[0][i].numpy()
            else:
                points = None

            # 获取该样本的预测结果
            pred_result = result[i]
            
            # 构造输出文件名
            if 'pts_filename' in img_metas:
                file_name = osp.split(img_metas['pts_filename'])[-1].split('.')[0]
            else:
                file_name = f"sample_{i}"
                
            out_file = osp.join(out_dir, file_name) if out_dir else None

            # 调用可视化函数
            show_result(
                points,
                None,  # gt_bboxes
                pred_result,
                out_dir,
                file_name,
                show=show,
                snapshot=True
            )

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        前向传播入口（支持多batch扩展）。
        
        @auto_fp16装饰器：将img和points转换为float16进行计算加速
        """
        if isinstance(img, list):
            raise NotImplementedError
        else:
            # 调用单batch前向传播
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        单batch样本的核心前向推理流程。
        
        处理流程：
        1. 分别提取多模态特征 -> 2. 融合特征 -> 3. 解码 -> 4. 检测和分割
        """
        features = []
        auxiliary_losses = {}
        
        # ==================== 多模态特征提取 ====================
        # 训练时按顺序(camera->lidar->radar)，推理时反序(radar->lidar->camera)以减少显存占用
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # 摄像头特征提取和BEV投影
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                # 如果使用深度监督，分离特征和损失
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                # 激光雷达特征提取
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                # 毫米波雷达特征提取
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"不支持的传感器: {sensor}")

            features.append(feature)

        # ==================== 特征融合 ====================
        if not self.training:
            # 推理时恢复原始顺序
            features = features[::-1]

        if self.fuser is not None:
            # 融合多模态特征
            x = self.fuser(features)
        else:
            # 仅使用单个传感器的特征
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        # ==================== BEV解码 ====================
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        # ==================== 训练阶段：计算损失 ====================
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    # 3D目标检测
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    # BEV语义分割
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"不支持的head类型: {type}")
                
                # 累加加权损失
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            
            # 添加深度监督损失
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('启用了深度损失但未找到深度损失')
            return outputs
        
        # ==================== 推理阶段：生成预测结果 ====================
        else:
            # 为每个样本创建结果字典
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    # 3D目标检测推理
                    pred_dict = head(x, metas)
                    # 从网络输出提取最终的3D框
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    # BEV语义分割推理
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None,
                            }
                        )
                else:
                    raise ValueError(f"不支持的head类型: {type}")
            return outputs

