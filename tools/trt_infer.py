"""
Phase 5/6: End-to-end TRT inference pipeline for BEVFusion.

Phase 5 (hybrid): TRT engines for SwinT, depthnet, fuser+decoder;
    PyTorch for camera neck, bev_pool, voxelization, LiDAR backbone, TransFusionHead.

Phase 6 (full TRT): additionally replaces camera neck and TransFusionHead with TRT engines,
    and uses bev_pool_v2 CUDA kernel directly (no vtransform module dependency).

Runs in bevfusion_mqbench environment (Python 3.8, PyTorch 1.10.2, TRT 10.15, mmdet3d).

Usage:
    # Phase 6 — full TRT (neck + head engines)
    python tools/trt_infer.py \
        --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --ckpt pretrained/bevfusion-det.pth \
        --swin-engine swin_int8_sm86.engine \
        --depthnet-engine vtransform_depthnet_int8_sm86.engine \
        --fuser-engine fuser_decoder_fp16_sm86.engine \
        --neck-engine camera_neck_int8_sm86.engine \
        --head-engine transfusion_head_int8_sm86.engine \
        --version A --test-single
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

import tensorrt as trt
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval


# ============================================================================
# TRT Engine Runner
# ============================================================================

class TRTRunner:
    """Runs a TRT engine using torch CUDA tensors."""

    def __init__(self, engine_path, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Parse I/O tensor info
        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(self.engine.get_tensor_shape(name))
                dtype_trt = self.engine.get_tensor_dtype(name)
                if dtype_trt == trt.float16:
                    self.output_dtypes[name] = torch.float16
                else:
                    self.output_dtypes[name] = torch.float32

        self.logger.info(
            f"TRT engine loaded: {engine_path} "
            f"(inputs={self.input_names}, outputs={self.output_names})"
        )

    def __call__(self, *inputs):
        """Run engine with positional torch.Tensor inputs.

        Returns list of output tensors (on GPU).
        """
        assert len(inputs) == len(self.input_names)

        # Set input tensors
        for name, tensor in zip(self.input_names, inputs):
            t = tensor.contiguous()
            if t.dtype == torch.float64:
                t = t.float()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # Allocate output tensors
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.output_dtypes[name]
            t = torch.zeros(shape, dtype=dtype, device="cuda").contiguous()
            self.context.set_tensor_address(name, t.data_ptr())
            outputs[name] = t

        # Execute
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        return [outputs[name] for name in self.output_names]


# ============================================================================
# Hybrid BEVFusion Model (TRT + PyTorch)
# ============================================================================

class HybridBEVFusion(nn.Module):
    """BEVFusion with TRT engines replacing selected submodules.

    TRT engines:
        - SwinT backbone → swin_int8.engine
        - depthnet → vtransform_depthnet_int8.engine
        - fuser+decoder → fuser_decoder_*.engine

    PyTorch modules (from original model):
        - camera neck (GeneralizedLSSFPN)
        - vtransform geometry + bev_pool
        - voxelization
        - LiDAR backbone (SparseEncoder, spconv 2.1)
        - TransFusionHead
    """

    def __init__(self, model, swin_engine, depthnet_engine, fuser_engine, logger,
                 neck_engine=None, head_engine=None):
        super().__init__()
        self.logger = logger

        # TRT engines
        self.swin_trt = swin_engine
        self.depthnet_trt = depthnet_engine
        self.fuser_trt = fuser_engine
        self.neck_trt = neck_engine  # Phase 6: optional TRT neck
        self.head_trt = head_engine  # Phase 6: optional TRT head

        # PyTorch modules from original model
        self.camera_neck = model.encoders["camera"]["neck"] if neck_engine is None else None
        self.vtransform = model.encoders["camera"]["vtransform"]
        self.lidar_voxelize = model.encoders["lidar"]["voxelize"]
        self.lidar_backbone = model.encoders["lidar"]["backbone"]
        self.voxelize_reduce = model.voxelize_reduce
        self.heads = model.heads if head_engine is None else None

        # Phase 6: extract config for post-processing
        if head_engine is not None:
            head = model.heads["object"]
            self.head_num_proposals = head.num_proposals
            self.head_num_classes = head.num_classes
            self.head_test_cfg = head.test_cfg
            self.head_bbox_coder = head.bbox_coder
            # Store query_labels buffer (set during forward)
            self._query_labels = None

        self.fp16_enabled = False
        self._last_intermediates = {}

    @torch.no_grad()
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
        **kwargs,
    ):
        B, N, C, H, W = img.shape

        # ============================================================
        # Step 1: SwinT backbone (TRT)
        # ============================================================
        # SwinT engine expects [1, 3, 256, 704], process each image
        img_flat = img.view(B * N, C, H, W).float()
        swin_outputs = []
        for i in range(B * N):
            outs = self.swin_trt(img_flat[i : i + 1])
            swin_outputs.append([o.float() for o in outs])

        # Stack multi-scale features: list of [B*N, C_i, H_i, W_i]
        num_scales = len(swin_outputs[0])
        multi_scale_feats = []
        for s in range(num_scales):
            feat = torch.cat([swin_outputs[i][s] for i in range(B * N)], dim=0)
            multi_scale_feats.append(feat)

        # ============================================================
        # Step 2: Camera neck (TRT or PyTorch)
        # ============================================================
        if self.neck_trt is not None:
            # Phase 6: TRT neck expects 3 separate inputs
            neck_out = self.neck_trt(multi_scale_feats[0].float(),
                                      multi_scale_feats[1].float(),
                                      multi_scale_feats[2].float())
            x_cam = neck_out[0].float()
        else:
            neck_out = self.camera_neck(multi_scale_feats)
            if not isinstance(neck_out, torch.Tensor):
                x_cam = neck_out[0]
            else:
                x_cam = neck_out
        # x_cam: [B*N, 256, 32, 88]

        # ============================================================
        # Step 3: vtransform depthnet (TRT) + bev_pool_v2 (CUDA kernel)
        # ============================================================
        BN, C_neck, fH, fW = x_cam.shape
        x_cam_5d = x_cam.view(B, N, C_neck, fH, fW)

        # Compute depth map (same as BaseDepthTransform.forward)
        depth_map = self._compute_depth_map(
            points, img_aug_matrix, lidar_aug_matrix, lidar2image, B, N
        )
        # depth_map: [B, N, 1, 256, 704]

        # Depthnet TRT engine expects [1, 6, 256, 32, 88] + [1, 6, 1, 256, 704]
        # Output: [B*N*D*fH*fW, C]
        depthnet_out = self.depthnet_trt(x_cam_5d.float(), depth_map.float())
        cam_feats_flat = depthnet_out[0].float()

        # Compute geometry
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.vtransform.get_geometry(
            camera2lidar_rots, camera2lidar_trans,
            intrins, post_rots, post_trans,
            extra_rots=extra_rots, extra_trans=extra_trans,
        )

        # Reshape cam_feats to [B, N, D, fH, fW, C] for bev_pool
        D = self.vtransform.D
        C_bev = self.vtransform.C
        cam_feats_6d = cam_feats_flat.view(B, N, D, fH, fW, C_bev)

        # bev_pool_v2: precompute indices + CUDA kernel
        indices = self.vtransform.precompute_bev_indices(geom, B)
        # Call bev_pool_with_indices logic directly to avoid @force_fp32
        # casting bool/int indices to float
        Nprime = B * N * D * fH * fW
        x_flat = cam_feats_6d.reshape(Nprime, C_bev)
        x_flat = x_flat[indices["kept"]]
        x_flat = x_flat[indices["sort_indices"]]

        from mmdet3d.ops.bev_pool.bev_pool import bev_pool_v2
        out = bev_pool_v2(
            x_flat,
            indices["geom_feats"],
            indices["interval_starts"],
            indices["interval_lengths"],
            indices["B"],
            indices["D"],
            indices["H"],
            indices["W"],
        )
        camera_bev = torch.cat(out.unbind(dim=2), 1)
        # camera_bev: [1, 80, 180, 180]

        # Apply downsample if present
        if hasattr(self.vtransform, 'downsample') and not isinstance(
            self.vtransform.downsample, nn.Identity
        ):
            camera_bev = self.vtransform.downsample(camera_bev)
        self._last_intermediates['camera_bev'] = camera_bev.detach().cpu().numpy()

        # ============================================================
        # Step 4: LiDAR backbone (PyTorch)
        # ============================================================
        feats, coords, sizes = self._voxelize(points)
        batch_size = coords[-1, 0] + 1
        lidar_bev = self.lidar_backbone(feats, coords, batch_size, sizes=sizes)
        # lidar_bev: [1, 256, 180, 180]
        self._last_intermediates['lidar_bev'] = lidar_bev.detach().cpu().numpy()

        # ============================================================
        # Step 5: Fuser + Decoder (TRT)
        # ============================================================
        fuser_out = self.fuser_trt(camera_bev.float(), lidar_bev.float())
        neck_features = fuser_out[0].float()
        # neck_features: [1, 512, 180, 180]
        self._last_intermediates['neck_features'] = neck_features.detach().cpu().numpy()

        # ============================================================
        # Step 6: TransFusionHead (TRT or PyTorch)
        # ============================================================
        batch_size_int = img.shape[0]
        outputs = [{} for _ in range(batch_size_int)]

        if self.head_trt is not None:
            # Phase 6: TRT head outputs raw predictions
            head_outs = self.head_trt(neck_features.float())
            # head_outs: [center, height, dim, rot, vel, heatmap, query_heatmap_score, dense_heatmap]
            center = head_outs[0].float()
            height = head_outs[1].float()
            dim = head_outs[2].float()
            rot = head_outs[3].float()
            vel = head_outs[4].float()
            heatmap = head_outs[5].float()
            query_heatmap_score = head_outs[6].float()
            self._last_intermediates['center'] = center.detach().cpu().numpy()
            self._last_intermediates['height'] = height.detach().cpu().numpy()
            self._last_intermediates['dim'] = dim.detach().cpu().numpy()
            self._last_intermediates['rot'] = rot.detach().cpu().numpy()
            self._last_intermediates['vel'] = vel.detach().cpu().numpy()
            self._last_intermediates['heatmap'] = heatmap.detach().cpu().numpy()
            self._last_intermediates['query_heatmap_score'] = query_heatmap_score.detach().cpu().numpy()

            # Pure Python post-processing (decode + NMS)
            bboxes = self._decode_and_nms(
                center, height, dim, rot, vel, heatmap, query_heatmap_score, metas
            )
            for k, (boxes, scores, labels) in enumerate(bboxes):
                outputs[k].update({
                    "boxes_3d": boxes.to("cpu"),
                    "scores_3d": scores.cpu(),
                    "labels_3d": labels.cpu(),
                })
        else:
            for type_name, head in self.heads.items():
                if type_name == "object":
                    pred_dict = head(neck_features, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
        return outputs

    def _decode_and_nms(self, center, height, dim, rot, vel, heatmap, query_heatmap_score, metas):
        """Pure Python post-processing for TRT TransFusionHead output.

        Reimplements TransFusionHead.get_bboxes() without mmdet3d dependency.
        """
        from mmdet3d.core import circle_nms, xywhr2xyxyr
        from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu

        num_proposals = self.head_num_proposals
        num_classes = self.head_num_classes
        test_cfg = self.head_test_cfg
        bbox_coder = self.head_bbox_coder

        # Take last num_proposals (for auxiliary=True with single decoder layer, this is all)
        batch_score = heatmap[..., -num_proposals:].sigmoid()

        # Compute query_labels from heatmap (same as forward_single topk logic)
        # query_heatmap_score already has the right shape [1, num_classes, num_proposals]
        # We need query_labels: for each proposal, which class had highest heatmap score
        # This is embedded in the topk selection during forward — approximate from query_heatmap_score
        query_labels = query_heatmap_score.max(1).indices  # [B, num_proposals]

        one_hot = torch.nn.functional.one_hot(query_labels, num_classes=num_classes).permute(0, 2, 1)
        batch_score = batch_score * query_heatmap_score * one_hot.float()

        batch_center = center[..., -num_proposals:]
        batch_height = height[..., -num_proposals:]
        batch_dim = dim[..., -num_proposals:]
        batch_rot = rot[..., -num_proposals:]
        batch_vel = vel[..., -num_proposals:]

        temp = bbox_coder.decode(
            batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel,
            filter=True,
        )

        # NMS (same as TransFusionHead.get_bboxes)
        tasks = [
            dict(num_class=8, class_names=[], indices=[0,1,2,3,4,5,6,7], radius=-1),
            dict(num_class=1, class_names=["pedestrian"], indices=[8], radius=0.175),
            dict(num_class=1, class_names=["traffic_cone"], indices=[9], radius=0.175),
        ]

        ret_layer = []
        for i in range(heatmap.shape[0]):
            boxes3d = temp[i]["bboxes"]
            scores = temp[i]["scores"]
            labels = temp[i]["labels"]

            if test_cfg["nms_type"] is not None:
                keep_mask = torch.zeros_like(scores)
                for task in tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    if task["radius"] > 0:
                        if test_cfg["nms_type"] == "circle":
                            boxes_for_nms = torch.cat(
                                [boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1
                            )
                            task_keep_indices = torch.tensor(
                                circle_nms(boxes_for_nms.detach().cpu().numpy(), task["radius"])
                            )
                        else:
                            boxes_for_nms = xywhr2xyxyr(
                                metas[i]["box_type_3d"](boxes3d[task_mask][:, :7], 7).bev
                            )
                            top_scores = scores[task_mask]
                            task_keep_indices = nms_gpu(
                                boxes_for_nms, top_scores,
                                thresh=task["radius"],
                                pre_maxsize=test_cfg["pre_maxsize"],
                                post_max_size=test_cfg["post_maxsize"],
                            )
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())
                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                        keep_mask[keep_indices] = 1
                keep_mask = keep_mask.bool()
                ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
            else:
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
            ret_layer.append(ret)

        res = [
            [
                metas[0]["box_type_3d"](ret_layer[0]["bboxes"], box_dim=ret_layer[0]["bboxes"].shape[-1]),
                ret_layer[0]["scores"],
                ret_layer[0]["labels"].int(),
            ]
        ]
        return res

    def _compute_depth_map(self, points, img_aug_matrix, lidar_aug_matrix,
                           lidar2image, B, N):
        """Compute depth map from point cloud (same as BaseDepthTransform)."""
        image_size = self.vtransform.image_size
        depth = torch.zeros(
            B, N, 1, *image_size,
            device=points[0].device
        )

        for b in range(B):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords = cur_coords - cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        return depth

    @torch.no_grad()
    def _voxelize(self, points):
        """Voxelize point cloud (same as BEVFusion.voxelize)."""
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_voxelize(res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                f, c = ret
                n = None
            feats.append(f)
            coords.append(
                torch.nn.functional.pad(c, (1, 0), mode="constant", value=k)
            )
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()
        return feats, coords, sizes


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(model, data_loader, logger):
    """Run NDS evaluation on the full validation set."""
    from mmdet3d.core import LiDARInstance3DBoxes

    model.eval()
    results = []
    dataset = data_loader.dataset

    logger.info(f"Running evaluation on {len(dataset)} samples...")
    t_start = time.time()

    for i, data in enumerate(data_loader):
        # Move data to GPU
        img = data["img"].data[0].cuda()
        points = [p.cuda() for p in data["points"].data[0]]
        metas = data["metas"].data[0]

        camera2ego = data["camera2ego"].data[0].cuda()
        lidar2ego = data["lidar2ego"].data[0].cuda()
        lidar2camera = data["lidar2camera"].data[0].cuda()
        lidar2image = data["lidar2image"].data[0].cuda()
        camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
        camera2lidar = data["camera2lidar"].data[0].cuda()
        img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
        lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

        with torch.no_grad():
            outputs = model.forward_single(
                img, points,
                camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix,
                metas,
            )

        results.extend(outputs)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            logger.info(f"  [{i+1}/{len(data_loader)}] {fps:.1f} samples/s")

    elapsed = time.time() - t_start
    logger.info(f"Inference done: {len(results)} samples in {elapsed:.1f}s "
                f"({len(results)/elapsed:.1f} fps)")

    # Run NDS evaluation
    logger.info("Computing NDS metrics...")
    eval_results = dataset.evaluate(results)
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v}")

    return eval_results


def run_single_test(model, data_loader, logger):
    """Run a single sample for sanity check."""
    model.eval()
    data = next(iter(data_loader))

    img = data["img"].data[0].cuda()
    points = [p.cuda() for p in data["points"].data[0]]
    metas = data["metas"].data[0]

    camera2ego = data["camera2ego"].data[0].cuda()
    lidar2ego = data["lidar2ego"].data[0].cuda()
    lidar2camera = data["lidar2camera"].data[0].cuda()
    lidar2image = data["lidar2image"].data[0].cuda()
    camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
    camera2lidar = data["camera2lidar"].data[0].cuda()
    img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
    lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()

    logger.info(f"Input shapes: img={img.shape}, points={points[0].shape}")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.forward_single(
            img, points,
            camera2ego, lidar2ego, lidar2camera, lidar2image,
            camera_intrinsics, camera2lidar,
            img_aug_matrix, lidar_aug_matrix,
            metas,
        )
    t1 = time.time()

    logger.info(f"Inference time: {(t1-t0)*1000:.1f} ms")
    for k, v in outputs[0].items():
        if hasattr(v, 'shape'):
            logger.info(f"  {k}: shape={v.shape}")
        elif hasattr(v, 'tensor'):
            logger.info(f"  {k}: shape={v.tensor.shape}")
        else:
            logger.info(f"  {k}: {type(v)}")

    n_boxes = outputs[0]["scores_3d"].shape[0]
    high_conf = (outputs[0]["scores_3d"] > 0.3).sum().item()
    logger.info(f"Detections: {n_boxes} total, {high_conf} with score > 0.3")

    return outputs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEVFusion TRT end-to-end inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--swin-engine", required=True)
    parser.add_argument("--depthnet-engine", required=True)
    parser.add_argument("--fuser-engine", required=True)
    parser.add_argument("--version", choices=["A", "B"], default="A",
                        help="A=W8A16 (FP16 fuser), B=INT8+Log2 (INT8 fuser)")
    parser.add_argument("--neck-engine", default=None,
                        help="Phase 6: Camera neck TRT engine (optional, falls back to PyTorch)")
    parser.add_argument("--head-engine", default=None,
                        help="Phase 6: TransFusionHead TRT engine (optional, falls back to PyTorch)")
    parser.add_argument("--lidar-quant", choices=["none", "w8a16", "int8"], default="none",
                        help="LiDAR backbone quantization: none=FP32, w8a16=weight INT8 only, int8=Log2 full INT8")
    parser.add_argument("--ptq-ckpt", default="pretrained/ptq_minmax_model.pth",
                        help="PTQ checkpoint for LiDAR quantization (used when --lidar-quant != none)")
    parser.add_argument("--test-single", action="store_true",
                        help="Run single sample test only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for multi-GPU parallel eval (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (set to num GPUs)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for results")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info(f"BEVFusion TRT Pipeline — Version {args.version}")
    logger.info(f"  SwinT engine: {args.swin_engine}")
    logger.info(f"  Depthnet engine: {args.depthnet_engine}")
    logger.info(f"  Fuser engine: {args.fuser_engine}")

    # Load config
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # Build full model (for PyTorch modules)
    logger.info("Building BEVFusion model...")

    if args.lidar_quant != "none":
        # Build quantized model for LiDAR backbone
        from tools.quant_ptq_minmax import (
            build_ptq_model, SparseLog2FakeQuantize, KLDivergenceObserver
        )
        from mqbench.utils.state import enable_quantization

        no_lidar_act = (args.lidar_quant == "w8a16")
        # Log2 act quantizer for int8 mode, None for w8a16 (weight-only)
        act_obs = SparseLog2FakeQuantize if args.lidar_quant == "int8" else None

        logger.info(f"  LiDAR quantization: {args.lidar_quant}")
        logger.info(f"  PTQ checkpoint: {args.ptq_ckpt}")

        model, _, _ = build_ptq_model(
            cfg, logger,
            act_observer_cls=act_obs,
            no_lidar_act_quant=no_lidar_act,
        )

        # Load PTQ checkpoint
        ckpt = torch.load(args.ptq_ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # Fix shape mismatch for FakeQuant scale/zero_point
        model_sd = model.state_dict()
        for k, v in list(state_dict.items()):
            if k in model_sd and v.shape != model_sd[k].shape:
                if v.numel() == model_sd[k].numel():
                    state_dict[k] = v.reshape(model_sd[k].shape)
                else:
                    parts = k.split('.')
                    obj = model
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif part.isdigit() and hasattr(obj, '__getitem__'):
                            obj = obj[int(part)]
                        else:
                            obj = getattr(obj, part)
                    param_name = parts[-1]
                    old = getattr(obj, param_name)
                    if isinstance(old, nn.Parameter):
                        setattr(obj, param_name, nn.Parameter(v.clone(), requires_grad=old.requires_grad))
                    else:
                        setattr(obj, param_name, v.clone())

        model.load_state_dict(state_dict, strict=False)
        enable_quantization(model)

        # Sync Log2 quantizer state (MQBench doesn't know about SparseLog2FakeQuantize)
        if args.lidar_quant == "int8":
            from tools.quant_ptq_minmax import _sync_log2_quantizer_state
            _sync_log2_quantizer_state(model, observe=False)

        model.eval().cuda()
    else:
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval().cuda()

    # Load TRT engines
    logger.info("Loading TRT engines...")
    swin_trt = TRTRunner(args.swin_engine, logger)
    depthnet_trt = TRTRunner(args.depthnet_engine, logger)
    fuser_trt = TRTRunner(args.fuser_engine, logger)

    neck_trt = None
    if args.neck_engine:
        neck_trt = TRTRunner(args.neck_engine, logger)
        logger.info(f"  Neck engine: {args.neck_engine}")

    head_trt = None
    if args.head_engine:
        head_trt = TRTRunner(args.head_engine, logger)
        logger.info(f"  Head engine: {args.head_engine}")

    # Build hybrid model
    hybrid = HybridBEVFusion(model, swin_trt, depthnet_trt, fuser_trt, logger,
                              neck_engine=neck_trt, head_engine=head_trt)
    hybrid.eval().cuda()

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg.data.test)

    if args.num_shards > 1:
        # Shard the dataset for multi-GPU parallel eval
        total = len(dataset)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        indices = list(range(start, end))
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: samples [{start}, {end}) = {len(dataset)}")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    if args.test_single:
        run_single_test(hybrid, data_loader, logger)
    else:
        if args.num_shards > 1:
            # Multi-GPU shard mode: save raw predictions, merge later
            import pickle
            model_eval = hybrid
            model_eval.eval()
            results = []
            logger.info(f"Running shard {args.shard_id} inference on {len(dataset)} samples...")
            t_start = time.time()
            for i, data in enumerate(data_loader):
                img = data["img"].data[0].cuda()
                points = [p.cuda() for p in data["points"].data[0]]
                metas = data["metas"].data[0]
                camera2ego = data["camera2ego"].data[0].cuda()
                lidar2ego = data["lidar2ego"].data[0].cuda()
                lidar2camera = data["lidar2camera"].data[0].cuda()
                lidar2image = data["lidar2image"].data[0].cuda()
                camera_intrinsics = data["camera_intrinsics"].data[0].cuda()
                camera2lidar = data["camera2lidar"].data[0].cuda()
                img_aug_matrix = data["img_aug_matrix"].data[0].cuda()
                lidar_aug_matrix = data["lidar_aug_matrix"].data[0].cuda()
                with torch.no_grad():
                    outputs = model_eval.forward_single(
                        img, points, camera2ego, lidar2ego, lidar2camera,
                        lidar2image, camera_intrinsics, camera2lidar,
                        img_aug_matrix, lidar_aug_matrix, metas)
                results.extend(outputs)
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    logger.info(f"  [{i+1}/{len(data_loader)}] {(i+1)/elapsed:.1f} samples/s")

            elapsed = time.time() - t_start
            logger.info(f"Shard {args.shard_id} done: {len(results)} samples in {elapsed:.1f}s")

            os.makedirs(args.out_dir, exist_ok=True)
            pkl_path = os.path.join(args.out_dir, f"preds_version_{args.version}_shard{args.shard_id}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Predictions saved to {pkl_path}")
        else:
            eval_results = run_evaluation(hybrid, data_loader, logger)
            import json
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"trt_eval_version_{args.version}.json")
            with open(out_path, "w") as f:
                json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                           for k, v in eval_results.items()}, f, indent=2)
            logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
