"""
Full nuScenes val evaluation for ZeroTorchBEVFusion with GPU zero-copy vtransform.
Runs all 6019 validation samples and outputs NDS/mAP.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
_BUILD_SP39 = os.path.join(ROOT, "build_sp39")
if (
    os.environ.get("BEVFUSION_STANDALONE") == "1"
    and sys.version_info >= (3, 9)
    and os.path.isdir(_BUILD_SP39)
):
    # Optional standalone mode for py39; full-ops mode remains default.
    sys.path.insert(0, _BUILD_SP39)
    try:
        import bev_pool_ext as _bev_pool_ext
        import voxel_layer as _voxel_layer
        import iou3d_cuda as _iou3d_cuda
        import roiaware_pool3d_ext as _roiaware_pool3d_ext

        sys.modules["mmdet3d.ops.bev_pool.bev_pool_ext"] = _bev_pool_ext
        sys.modules["mmdet3d.ops.voxel.voxel_layer"] = _voxel_layer
        sys.modules["mmdet3d.ops.iou3d.iou3d_cuda"] = _iou3d_cuda
        sys.modules["mmdet3d.ops.roiaware_pool3d.roiaware_pool3d_ext"] = _roiaware_pool3d_ext
    except Exception:
        # Let downstream imports raise a clearer error if extensions are missing.
        pass

from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, recursive_eval

from tools.trt_infer import HybridBEVFusion, TRTRunner
from tools.engine_utils import current_sm_tag, load_runner_with_fallback
from tools.trt_infer_zero_torch import (
    ZeroTorchTRTRunner,
    ZeroTorchVoxelization,
    NumpyVTransformGeometry,
    NumpyTransFusionBBoxCoder,
    ZeroTorchBEVFusion,
    prepare_swin_batched_engine,
    prepare_bev_downsample_engine,
)


class EvalSimpleLiDARBox:
    """Minimal box wrapper compatible with nuScenes dataset evaluation."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def gravity_center(self):
        bottom_center = self.tensor[:, :3]
        gc = torch.zeros_like(bottom_center)
        gc[:, :2] = bottom_center[:, :2]
        gc[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gc

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]


def _to_eval_result(det):
    boxes = det["boxes_3d"]
    scores = det["scores_3d"]
    labels = det["labels_3d"]

    if isinstance(boxes, np.ndarray):
        boxes_t = torch.from_numpy(boxes.astype(np.float32, copy=False)).cpu()
        boxes = EvalSimpleLiDARBox(boxes_t)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores.astype(np.float32, copy=False)).cpu()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(np.int64, copy=False)).cpu()

    return {
        "boxes_3d": boxes,
        "scores_3d": scores,
        "labels_3d": labels,
    }


def run_evaluation(model, data_loader, logger):
    results = []
    dataset = data_loader.dataset
    logger.info(f"Running ZeroTorch evaluation on {len(dataset)} samples...")
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

        img_np = img.cpu().numpy()
        points_np = [p.cpu().numpy() for p in points]
        camera2ego_np = camera2ego.cpu().numpy()
        lidar2ego_np = lidar2ego.cpu().numpy()
        lidar2camera_np = lidar2camera.cpu().numpy()
        lidar2image_np = lidar2image.cpu().numpy()
        camera_intrinsics_np = camera_intrinsics.cpu().numpy()
        camera2lidar_np = camera2lidar.cpu().numpy()
        img_aug_matrix_np = img_aug_matrix.cpu().numpy()
        lidar_aug_matrix_np = lidar_aug_matrix.cpu().numpy()

        outputs = model.forward(
            img_np, points_np,
            camera2ego_np, lidar2ego_np, lidar2camera_np, lidar2image_np,
            camera_intrinsics_np, camera2lidar_np,
            img_aug_matrix_np, lidar_aug_matrix_np, metas,
        )

        # Convert zero-torch outputs to dataset-eval compatible format
        for out in outputs:
            results.append(_to_eval_result(out))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            logger.info(f"  [{i+1}/{len(data_loader)}] {fps:.1f} samples/s")

    elapsed = time.time() - t_start
    logger.info(f"Inference done: {len(results)} samples in {elapsed:.1f}s "
                f"({len(results)/elapsed:.1f} fps)")

    logger.info("Computing NDS metrics...")
    eval_results = dataset.evaluate(results)
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v}")
    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--swin-engine", default="swin_int8_sm86.engine")
    parser.add_argument("--depthnet-engine", default="vtransform_depthnet_int8_sm86.engine")
    parser.add_argument("--fuser-engine", default="fuser_decoder_int8_sm86.engine")
    parser.add_argument("--neck-engine", default="camera_neck_int8_sm86.engine")
    parser.add_argument("--head-engine", default="transfusion_head_int8_sm86.engine")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="zero_torch_eval.json")
    parser.add_argument("--swin-batch-size", type=int, default=6)
    parser.add_argument("--auto-build-swin-batch", dest="auto_build_swin_batch", action="store_true")
    parser.add_argument("--no-auto-build-swin-batch", dest="auto_build_swin_batch", action="store_false")
    parser.set_defaults(auto_build_swin_batch=False)
    parser.add_argument("--bev-downsample-engine", default="bev_downsample_fp32_sm86.engine")
    parser.add_argument("--lidar-npy-dir", default="pretrained/lidar_npy_fp16")
    args = parser.parse_args()

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Full val eval: ZeroTorch BEVFusion (GPU zero-copy vtransform)")
    logger.info("=" * 60)
    logger.info(f"CUDA target sm{current_sm_tag()}")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    logger.info("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    logger.info("Loading TRT engines...")
    args.swin_engine, _ = load_runner_with_fallback(
        args.swin_engine, TRTRunner, logger, "eval.base.swin"
    )
    args.neck_engine, _ = load_runner_with_fallback(
        args.neck_engine, TRTRunner, logger, "eval.base.neck"
    )
    args.depthnet_engine, _ = load_runner_with_fallback(
        args.depthnet_engine, TRTRunner, logger, "eval.base.depthnet"
    )
    args.fuser_engine, _ = load_runner_with_fallback(
        args.fuser_engine, TRTRunner, logger, "eval.base.fuser"
    )
    args.head_engine, _ = load_runner_with_fallback(
        args.head_engine, TRTRunner, logger, "eval.base.head"
    )

    swin_batched_engine = prepare_swin_batched_engine(
        args.swin_engine,
        batch_size=args.swin_batch_size,
        logger=logger,
        auto_build=args.auto_build_swin_batch,
    )
    swin_engine_zt_path, _ = load_runner_with_fallback(
        swin_batched_engine, ZeroTorchTRTRunner, logger, "eval.zero.swin"
    )
    args.neck_engine, neck_zt = load_runner_with_fallback(
        args.neck_engine, ZeroTorchTRTRunner, logger, "eval.zero.neck"
    )
    args.depthnet_engine, depthnet_zt = load_runner_with_fallback(
        args.depthnet_engine, ZeroTorchTRTRunner, logger, "eval.zero.depthnet"
    )
    args.fuser_engine, fuser_zt = load_runner_with_fallback(
        args.fuser_engine, ZeroTorchTRTRunner, logger, "eval.zero.fuser"
    )
    args.head_engine, head_zt = load_runner_with_fallback(
        args.head_engine, ZeroTorchTRTRunner, logger, "eval.zero.head"
    )
    swin_zt = ZeroTorchTRTRunner(swin_engine_zt_path, logger)

    logger.info("Building original model (for LiDAR backbone weights)...")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()

    lidar_voxelize_cfg = cfg.model.encoders.lidar.voxelize
    max_voxels = lidar_voxelize_cfg.max_voxels
    if isinstance(max_voxels, (list, tuple)):
        max_voxels = max_voxels[1]
    voxelizer = ZeroTorchVoxelization(
        voxel_size=lidar_voxelize_cfg.voxel_size,
        point_cloud_range=lidar_voxelize_cfg.point_cloud_range,
        max_num_points=lidar_voxelize_cfg.max_num_points,
        max_voxels=max_voxels,
    )

    vtransform = model.encoders["camera"]["vtransform"]
    vtransform_geom = NumpyVTransformGeometry(
        image_size=tuple(vtransform.image_size),
        feature_size=tuple(vtransform.feature_size),
        xbound=vtransform.xbound,
        ybound=vtransform.ybound,
        zbound=vtransform.zbound,
        dbound=vtransform.dbound,
    )
    bev_downsample_in_shape = (
        1,
        int(vtransform.C) * int(vtransform_geom.nx[2]),
        int(vtransform_geom.nx[0]),
        int(vtransform_geom.nx[1]),
    )
    bev_downsample_engine = prepare_bev_downsample_engine(
        vtransform.downsample,
        args.bev_downsample_engine,
        bev_downsample_in_shape,
        logger=logger,
    )
    bev_downsample_zt = None
    if bev_downsample_engine is not None:
        try:
            args.bev_downsample_engine, _ = load_runner_with_fallback(
                bev_downsample_engine, ZeroTorchTRTRunner, logger, "eval.zero.bev_downsample"
            )
            bev_downsample_zt = ZeroTorchTRTRunner(args.bev_downsample_engine, logger)
        except Exception as exc:
            raise RuntimeError(
                "TRT bev_downsample is required in strict zero-torch mode; "
                f"failed to initialize from '{bev_downsample_engine}': {exc}"
            ) from exc

    try:
        from tools.tv_sparse_encoder import TVSparseEncoder, get_cuda_arch
        lidar_arch = get_cuda_arch(0)
        logger.info(f"Building TVSparseEncoder for zero-torch LiDAR, arch={lidar_arch}")
        lidar_backbone_tv = TVSparseEncoder(arch=lidar_arch, stream=0)
        lidar_backbone_tv.load_npy_weights(args.lidar_npy_dir)
        lidar_backbone_zt = lidar_backbone_tv
    except Exception as exc:
        raise RuntimeError(
            "TVSparseEncoder is required in strict zero-torch mode; "
            "fix TV/spconv environment and retry."
        ) from exc

    head_obj = model.heads["object"]
    bbox_coder = NumpyTransFusionBBoxCoder(
        pc_range=head_obj.bbox_coder.pc_range,
        out_size_factor=head_obj.bbox_coder.out_size_factor,
        voxel_size=head_obj.bbox_coder.voxel_size,
        post_center_range=head_obj.bbox_coder.post_center_range,
        score_threshold=0.0,
    )
    test_cfg = dict(head_obj.test_cfg)
    voxelize_reduce = cfg.model.get("voxelize_reduce", True)

    zero_model = ZeroTorchBEVFusion(
        swin_trt=swin_zt,
        depthnet_trt=depthnet_zt,
        fuser_trt=fuser_zt,
        neck_trt=neck_zt,
        head_trt=head_zt,
        lidar_backbone=lidar_backbone_zt,
        voxelizer=voxelizer,
        vtransform_geom=vtransform_geom,
        bev_downsample=bev_downsample_zt,
        bbox_coder=bbox_coder,
        test_cfg=test_cfg,
        num_proposals=head_obj.num_proposals,
        num_classes=head_obj.num_classes,
        voxelize_reduce=voxelize_reduce,
        logger=logger,
        use_tv_lidar=True,
        use_gpu_vtransform=True,
        capture_intermediates=False,
        enable_lidar_gpu_chain=True,
    )

    eval_results = run_evaluation(zero_model, data_loader, logger)

    with open(args.out, "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in eval_results.items()}, f, indent=2)
    logger.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
