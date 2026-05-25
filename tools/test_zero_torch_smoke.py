"""
End-to-end smoke test for zero-torch BEVFusion runner.
Uses random inputs and dummy LiDAR backbone to verify integration.
"""
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.trt_infer_zero_torch import (
    CudaBuffer,
    ZeroTorchTRTRunner,
    ZeroTorchVoxelization,
    NumpyVTransformGeometry,
    NumpyTransFusionBBoxCoder,
    ZeroTorchBEVFusion,
    np_bev_pool_v2,
)


class DummyLiDARBackbone:
    """Returns random BEV features; ignores tv.Tensor inputs."""

    def forward(self, feats_tv, coords_tv, batch_size):
        return np.random.randn(batch_size, 256, 180, 180).astype(np.float32)


def random_affine_matrices(B, N=None):
    if N is not None:
        mats = np.zeros((B, N, 4, 4), dtype=np.float32)
        for b in range(B):
            for n in range(N):
                m = np.eye(4, dtype=np.float32)
                angle = np.random.uniform(-0.1, 0.1)
                m[:3, :3] = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                m[:3, 3] = np.random.randn(3).astype(np.float32) * 0.5
                mats[b, n] = m
    else:
        mats = np.zeros((B, 4, 4), dtype=np.float32)
        for b in range(B):
            m = np.eye(4, dtype=np.float32)
            angle = np.random.uniform(-0.1, 0.1)
            m[:3, :3] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=np.float32)
            m[:3, 3] = np.random.randn(3).astype(np.float32) * 0.5
            mats[b] = m
    return mats


def run_smoke():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Use TRT 10.3 local env if available, otherwise fallback to system TRT
    trt103_dir = os.path.join(ROOT, "trt_10.3_env")
    if os.path.exists(trt103_dir):
        logger.info("Using local TRT 10.3 from %s", trt103_dir)
        os.environ["PYTHONPATH"] = f"{trt103_dir}:{os.environ.get('PYTHONPATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{trt103_dir}/tensorrt_libs:{os.environ.get('LD_LIBRARY_PATH', '')}"
    else:
        logger.info("Local TRT 10.3 not found; using system TensorRT")

    art = os.path.join(ROOT, "artifacts")
    engines = {
        "swin": os.path.join(art, "swin_int8_trt103.engine"),
        "neck": os.path.join(art, "camera_neck_int8_trt103.engine"),
        "depthnet": os.path.join(art, "vtransform_depthnet_int8_trt103.engine"),
        "fuser": os.path.join(art, "fuser_decoder_int8_trt103.engine"),
        "head": os.path.join(art, "transfusion_head_int8_trt103.engine"),
    }

    # Fallback to _sm86 engines if TRT 10.3 variants missing
    for k, v in list(engines.items()):
        if not os.path.exists(v):
            fallback = v.replace("_trt103", "_sm86")
            if not os.path.exists(fallback):
                fallback = v.replace("_trt103", "")
            if os.path.exists(fallback):
                logger.warning("%s missing; falling back to %s", v, fallback)
                engines[k] = fallback
            else:
                raise FileNotFoundError(f"Engine not found: {v}")

    logger.info("Loading TRT engines...")
    swin_trt = ZeroTorchTRTRunner(engines["swin"], logger)
    neck_trt = ZeroTorchTRTRunner(engines["neck"], logger)
    depthnet_trt = ZeroTorchTRTRunner(engines["depthnet"], logger)
    fuser_trt = ZeroTorchTRTRunner(engines["fuser"], logger)
    head_trt = ZeroTorchTRTRunner(engines["head"], logger)

    logger.info("Loading voxelizer...")
    voxelizer = ZeroTorchVoxelization(
        voxel_size=[0.075, 0.075, 0.2],
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        max_num_points=10,
        max_voxels=40000,
    )

    logger.info("Setting up geometry...")
    vtransform_geom = NumpyVTransformGeometry(
        image_size=(900, 1600),
        feature_size=(32, 88),
        xbound=[0.0, 51.2, 0.4],
        ybound=[-25.6, 25.6, 0.4],
        zbound=[-2.0, 4.4, 0.4],
        dbound=[1.0, 60.0, 0.5],
    )

    bbox_coder = NumpyTransFusionBBoxCoder(
        pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        out_size_factor=8,
        voxel_size=[0.075, 0.075, 0.2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        score_threshold=0.0,
    )

    test_cfg = {
        "use_rotate_nms": True,
        "nms": 0.2,
        "score_threshold": 0.1,
        "pre_max_size": 1000,
        "post_max_size": 83,
        "min_radius": [],
        "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    }

    model = ZeroTorchBEVFusion(
        swin_trt=swin_trt,
        depthnet_trt=depthnet_trt,
        fuser_trt=fuser_trt,
        neck_trt=neck_trt,
        head_trt=head_trt,
        lidar_backbone=DummyLiDARBackbone(),
        voxelizer=voxelizer,
        vtransform_geom=vtransform_geom,
        bev_downsample=None,
        bbox_coder=bbox_coder,
        test_cfg=test_cfg,
        num_proposals=200,
        num_classes=10,
        voxelize_reduce=True,
        logger=logger,
        use_tv_lidar=True,
    )

    B, N = 1, 6
    img = np.random.randn(B, N, 3, 256, 704).astype(np.float32)
    points = [np.random.randn(20000, 5).astype(np.float32) for _ in range(B)]
    camera2ego = random_affine_matrices(B, N)
    lidar2ego = random_affine_matrices(B)
    lidar2camera = random_affine_matrices(B, N)
    lidar2image = random_affine_matrices(B, N)
    camera_intrinsics = np.zeros((B, N, 4, 4), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            camera_intrinsics[b, n] = np.eye(4, dtype=np.float32)
            camera_intrinsics[b, n, 0, 0] = 1200.0
            camera_intrinsics[b, n, 1, 1] = 800.0
            camera_intrinsics[b, n, 0, 2] = 800.0
            camera_intrinsics[b, n, 1, 2] = 450.0
    camera2lidar = random_affine_matrices(B, N)
    img_aug_matrix = random_affine_matrices(B, N)
    lidar_aug_matrix = random_affine_matrices(B)
    metas = [{} for _ in range(B)]

    logger.info("Running forward pass...")
    outputs = model.forward(
        img, points, camera2ego, lidar2ego, lidar2camera,
        lidar2image, camera_intrinsics, camera2lidar,
        img_aug_matrix, lidar_aug_matrix, metas,
    )

    logger.info("Forward pass completed")
    for b, out in enumerate(outputs):
        logger.info("Batch %d: boxes_3d shape=%s, scores shape=%s, labels shape=%s",
                    b, out["boxes_3d"].shape, out["scores_3d"].shape, out["labels_3d"].shape)

    logger.info("Zero-torch smoke test PASSED")


if __name__ == "__main__":
    run_smoke()
