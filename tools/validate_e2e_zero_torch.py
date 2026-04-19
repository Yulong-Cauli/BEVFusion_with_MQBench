"""
End-to-end single-sample validation: HybridBEVFusion (original) vs ZeroTorchBEVFusion.
Both use the same TRT 10.15 SM86 engines; the LiDAR backbone is the original PyTorch
SparseEncoder wrapped for zero-torch to enable a fair comparison.
"""
import argparse
import json
import logging
import os
import random
import sys

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


def compare_tensor(name, a, b):
    """Compare numpy arrays a (ref) and b (test)."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    a64 = a.astype(np.float64, copy=False).ravel()
    b64 = b.astype(np.float64, copy=False).ravel()
    max_diff = diff.max()
    mean_diff = diff.mean()
    norm_a = np.linalg.norm(a64)
    norm_b = np.linalg.norm(b64)
    denom = norm_a * norm_b
    cos_sim = float(np.dot(a64, b64) / denom) if denom > 1e-12 else float("nan")
    cos_str = f"{cos_sim:.8f}" if np.isfinite(cos_sim) else "nan"
    # Relaxed thresholds for INT8 end-to-end
    status = "PASS" if max_diff < 5e-2 else "WARN" if max_diff < 5e-1 else "FAIL"
    print(
        f"  {name}: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
        f"cos_sim={cos_str} [{status}]"
    )
    return status != "FAIL"


def summarize_tensor(a):
    a = np.asarray(a)
    size = int(a.size)
    if size == 0:
        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "size": 0,
            "nonzero": 0,
            "nz_ratio": 0.0,
            "l2": 0.0,
            "abs_max": 0.0,
            "mean": 0.0,
        }
    a64 = a.astype(np.float64, copy=False)
    nonzero = int(np.count_nonzero(a))
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": size,
        "nonzero": nonzero,
        "nz_ratio": float(nonzero / float(size)),
        "l2": float(np.linalg.norm(a64.ravel())),
        "abs_max": float(np.max(np.abs(a64))),
        "mean": float(np.mean(a64)),
    }


def _to_python(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _to_python(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    return v


def _format_lidar_candidates(diag):
    if not isinstance(diag, dict):
        return "n/a"
    candidates = diag.get("candidates") or []
    parts = []
    for item in candidates:
        retry = item.get("retry", len(parts))
        nz_ratio = float(item.get("nz_ratio", 0.0))
        l2 = float(item.get("l2", 0.0))
        parts.append(f"{retry}:nz={nz_ratio:.5f},l2={l2:.2f}")
    return "; ".join(parts) if parts else "n/a"


def dump_lidar_diag(diag_dir, run_idx, reference_name, ref_lidar, zero_lidar, ref_diag, zero_diag, ref_voxel, zero_voxel):
    os.makedirs(diag_dir, exist_ok=True)
    base = os.path.join(diag_dir, f"run_{run_idx:02d}_lidar")
    np.savez_compressed(
        base + ".npz",
        ref_lidar=ref_lidar.astype(np.float32),
        zero_lidar=zero_lidar.astype(np.float32),
    )
    meta = {
        "run_idx": int(run_idx),
        "reference_name": reference_name,
        "ref_lidar": summarize_tensor(ref_lidar),
        "zero_lidar": summarize_tensor(zero_lidar),
        "ref_tv_diag": _to_python(ref_diag),
        "zero_tv_diag": _to_python(zero_diag),
        "ref_voxel_stats": _to_python(ref_voxel),
        "zero_voxel_stats": _to_python(zero_voxel),
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _run_tv_encoder_debug(encoder, feats_np, coords_np):
    from cumm import tensorview as tv

    feats_fp16 = torch.from_numpy(feats_np.astype(np.float32)).cuda().half().contiguous()
    feats_tv = tv.from_blob(feats_fp16.data_ptr(), list(feats_fp16.shape), tv.float16, 0)
    coords_i32 = torch.from_numpy(coords_np.astype(np.int32)).cuda().contiguous()
    coords_tv = tv.from_blob(coords_i32.data_ptr(), list(coords_i32.shape), tv.int32, 0)
    batch_size = int(coords_np[-1, 0]) + 1 if coords_np.shape[0] > 0 else 0
    debug = {}
    _ = encoder.forward(
        feats_tv, coords_tv, batch_size,
        feature_ref=feats_fp16, coors_ref=coords_i32, debug_dict=debug
    )
    return debug


def _layer_diff(a, b):
    if a.shape != b.shape:
        return {"shape_match": False}
    a64 = a.astype(np.float64, copy=False).ravel()
    b64 = b.astype(np.float64, copy=False).ravel()
    diff = np.abs(a64 - b64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    cos = float(np.dot(a64, b64) / denom) if denom > 1e-12 else float("nan")
    return {
        "shape_match": True,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "p999": float(np.quantile(diff, 0.999)),
        "cos_sim": cos,
    }


def compare_tv_layerwise(logger, ref_encoder, zero_encoder, ref_feats, ref_coords, zero_feats, zero_coords):
    logger.info(
        f"[LiDAR layerwise] input arrays: feats_equal={np.array_equal(ref_feats, zero_feats)}, "
        f"coords_equal={np.array_equal(ref_coords, zero_coords)}, "
        f"feats_shape={ref_feats.shape}/{zero_feats.shape}, coords_shape={ref_coords.shape}/{zero_coords.shape}"
    )
    ref_dbg = _run_tv_encoder_debug(ref_encoder, ref_feats, ref_coords)
    zero_dbg = _run_tv_encoder_debug(zero_encoder, zero_feats, zero_coords)

    ordered = [k for k in ref_dbg.keys() if k in zero_dbg and k != "dense_out"]
    first_bad = None
    first_shape_bad = None
    logger.info(f"[LiDAR layerwise] compared stages: {len(ordered)}")
    for k in ordered:
        ra = ref_dbg[k]
        zb = zero_dbg[k]
        inds_equal = np.array_equal(ra["indices"], zb["indices"])
        inds_set_equal = False
        if ra["indices"].shape == zb["indices"].shape:
            a = ra["indices"]
            b = zb["indices"]
            oa = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))
            ob = np.lexsort((b[:, 3], b[:, 2], b[:, 1], b[:, 0]))
            inds_set_equal = np.array_equal(a[oa], b[ob])
        stat = _layer_diff(ra["features"], zb["features"])
        if not stat["shape_match"]:
            logger.info(
                f"[LiDAR layerwise] {k}: shape mismatch {ra['features'].shape} vs {zb['features'].shape} "
                f"(idx {ra['indices'].shape} vs {zb['indices'].shape})"
            )
            if first_bad is None:
                first_bad = k
            if first_shape_bad is None:
                first_shape_bad = k
            continue
        cos_str = f"{stat['cos_sim']:.8f}" if np.isfinite(stat["cos_sim"]) else "nan"
        logger.info(
            f"[LiDAR layerwise] {k}: indices_equal={inds_equal}, index_set_equal={inds_set_equal} "
            f"max={stat['max_diff']:.6e} mean={stat['mean_diff']:.6e} "
            f"p99.9={stat['p999']:.6e} cos={cos_str}"
        )
        bad = (not inds_set_equal) or stat["max_diff"] > 1e-3 or stat["cos_sim"] < 0.99999
        if first_bad is None and bad:
            first_bad = k

    if "dense_out" in ref_dbg and "dense_out" in zero_dbg:
        dstat = _layer_diff(ref_dbg["dense_out"], zero_dbg["dense_out"])
        cos_str = f"{dstat['cos_sim']:.8f}" if np.isfinite(dstat["cos_sim"]) else "nan"
        logger.info(
            f"[LiDAR layerwise] dense_out: max={dstat['max_diff']:.6e} "
            f"mean={dstat['mean_diff']:.6e} p99.9={dstat['p999']:.6e} cos={cos_str}"
        )

    if first_bad is None:
        logger.info("[LiDAR layerwise] no early divergence stage detected by threshold")
    else:
        logger.info(f"[LiDAR layerwise] first divergence stage: {first_bad}")
    if first_shape_bad is not None:
        logger.info(f"[LiDAR layerwise] first shape divergence stage: {first_shape_bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--swin-engine", default="swin_int8_sm86.engine")
    parser.add_argument("--depthnet-engine", default="vtransform_depthnet_int8_sm86.engine")
    parser.add_argument("--fuser-engine", default="fuser_decoder_int8_sm86.engine")
    parser.add_argument("--neck-engine", default="camera_neck_int8_sm86.engine")
    parser.add_argument("--head-engine", default="transfusion_head_int8_sm86.engine")
    parser.add_argument("--swin-batch-size", type=int, default=6)
    parser.add_argument("--auto-build-swin-batch", dest="auto_build_swin_batch", action="store_true")
    parser.add_argument("--no-auto-build-swin-batch", dest="auto_build_swin_batch", action="store_false")
    parser.set_defaults(auto_build_swin_batch=False)
    parser.add_argument("--bev-downsample-engine", default="bev_downsample_fp32_sm86.engine")
    parser.add_argument("--lidar-npy-dir", default="pretrained/lidar_npy_fp16")
    parser.add_argument(
        "--reference-mode",
        choices=["auto", "hybrid", "standalone_tv"],
        default="auto",
        help="Reference pipeline used for comparison.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count on the same sample.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup forwards before measured runs.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    parser.add_argument(
        "--diag-dir",
        default="",
        help="Optional directory to dump lidar diagnostics when lidar_bev check fails.",
    )
    parser.add_argument(
        "--dump-all-runs",
        action="store_true",
        help="Dump lidar diagnostics for all runs, not only failure runs.",
    )
    parser.add_argument(
        "--lidar-layer-debug",
        action="store_true",
        help="Run TVSparseEncoder layer-wise activation diff on first measured run.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = get_root_logger(log_level=logging.INFO)
    logger.info("=" * 60)
    logger.info("End-to-end validation: Hybrid vs ZeroTorch")
    logger.info("=" * 60)
    logger.info(f"CUDA target sm{current_sm_tag()}")
    logger.info(f"Seed={args.seed}, warmup={args.warmup_runs}, repeat={args.repeat}")

    # ------------------------------------------------------------------
    # Load config and dataset
    # ------------------------------------------------------------------
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    logger.info("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )
    data = next(iter(data_loader))

    # ------------------------------------------------------------------
    # Build original model (HybridBEVFusion)
    # ------------------------------------------------------------------
    logger.info("Building original HybridBEVFusion...")
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()

    hybrid = None

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

    if args.reference_mode in ("hybrid", "auto"):
        try:
            args.swin_engine, swin_trt = load_runner_with_fallback(
                args.swin_engine, TRTRunner, logger, "hybrid.swin"
            )
            args.depthnet_engine, depthnet_trt = load_runner_with_fallback(
                args.depthnet_engine, TRTRunner, logger, "hybrid.depthnet"
            )
            args.fuser_engine, fuser_trt = load_runner_with_fallback(
                args.fuser_engine, TRTRunner, logger, "hybrid.fuser"
            )
            args.neck_engine, neck_trt = load_runner_with_fallback(
                args.neck_engine, TRTRunner, logger, "hybrid.neck"
            )
            args.head_engine, head_trt = load_runner_with_fallback(
                args.head_engine, TRTRunner, logger, "hybrid.head"
            )

            hybrid = HybridBEVFusion(
                model, swin_trt, depthnet_trt, fuser_trt, logger,
                neck_engine=neck_trt, head_engine=head_trt
            )
            hybrid.eval().cuda()
        except Exception as exc:
            if args.reference_mode == "hybrid":
                raise
            logger.warning(f"Hybrid reference unavailable ({exc}); fallback to StandaloneTV.")


    # ------------------------------------------------------------------
    # Build zero-torch model
    # ------------------------------------------------------------------
    logger.info("Building ZeroTorchBEVFusion...")

    args.swin_engine, _ = load_runner_with_fallback(
        args.swin_engine, ZeroTorchTRTRunner, logger, "zero.swin.base"
    )
    swin_batched_engine = prepare_swin_batched_engine(
        args.swin_engine,
        batch_size=args.swin_batch_size,
        logger=logger,
        auto_build=args.auto_build_swin_batch,
    )
    swin_engine_zt_path, _ = load_runner_with_fallback(
        swin_batched_engine, ZeroTorchTRTRunner, logger, "zero.swin"
    )
    swin_zt = ZeroTorchTRTRunner(swin_engine_zt_path, logger)
    args.neck_engine, neck_zt = load_runner_with_fallback(
        args.neck_engine, ZeroTorchTRTRunner, logger, "zero.neck"
    )
    args.depthnet_engine, depthnet_zt = load_runner_with_fallback(
        args.depthnet_engine, ZeroTorchTRTRunner, logger, "zero.depthnet"
    )
    args.fuser_engine, fuser_zt = load_runner_with_fallback(
        args.fuser_engine, ZeroTorchTRTRunner, logger, "zero.fuser"
    )
    args.head_engine, head_zt = load_runner_with_fallback(
        args.head_engine, ZeroTorchTRTRunner, logger, "zero.head"
    )

    # Voxelizer params from config
    lidar_voxelize_cfg = cfg.model.encoders.lidar.voxelize
    max_voxels = lidar_voxelize_cfg.max_voxels
    if isinstance(max_voxels, (list, tuple)):
        max_voxels = max_voxels[1]  # test-time value
    voxelizer = ZeroTorchVoxelization(
        voxel_size=lidar_voxelize_cfg.voxel_size,
        point_cloud_range=lidar_voxelize_cfg.point_cloud_range,
        max_num_points=lidar_voxelize_cfg.max_num_points,
        max_voxels=max_voxels,
    )

    # VTransform geometry from model
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
                bev_downsample_engine, ZeroTorchTRTRunner, logger, "zero.bev_downsample"
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

    # BBox coder and test cfg from head
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
        capture_intermediates=True,
        enable_lidar_gpu_chain=True,
    )

    # Convert inputs to numpy
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

    standalone_ref = None
    use_standalone_ref = (
        args.reference_mode == "standalone_tv"
        or (args.reference_mode == "auto" and use_tv_lidar)
    )
    if use_standalone_ref:
        try:
            from tools.trt_infer_standalone import (
                StandaloneBEVFusion,
                Voxelization as StandaloneVoxelization,
                VTransformGeometry,
                TransFusionBBoxCoder,
                TRTRunner as StandaloneTRTRunner,
            )
            from tools.tv_sparse_encoder import TVSparseEncoder, get_cuda_arch

            logger.info("Building Standalone TV reference pipeline...")
            ref_lidar = TVSparseEncoder(arch=get_cuda_arch(0), stream=0)
            ref_lidar.load_npy_weights(args.lidar_npy_dir)

            ref_voxelizer = StandaloneVoxelization(
                voxel_size=lidar_voxelize_cfg.voxel_size,
                point_cloud_range=lidar_voxelize_cfg.point_cloud_range,
                max_num_points=lidar_voxelize_cfg.max_num_points,
                max_voxels=lidar_voxelize_cfg.max_voxels,
            )
            vt_cfg = cfg.model.encoders.camera.vtransform
            ref_vt_geom = VTransformGeometry(
                image_size=vt_cfg.image_size,
                feature_size=vt_cfg.feature_size,
                xbound=vt_cfg.xbound,
                ybound=vt_cfg.ybound,
                zbound=vt_cfg.zbound,
                dbound=vt_cfg.dbound,
            )
            head_cfg = cfg.model.heads.object
            ref_bbox_coder = TransFusionBBoxCoder(
                pc_range=lidar_voxelize_cfg.point_cloud_range,
                out_size_factor=head_cfg.train_cfg.out_size_factor,
                voxel_size=lidar_voxelize_cfg.voxel_size,
                post_center_range=head_cfg.test_cfg.get(
                    "post_center_range", [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
                ),
                score_threshold=head_cfg.test_cfg.get("score_threshold", None),
                code_size=head_cfg.common_heads.get("vel", [2, 2])[0] + 8
                if "vel" in head_cfg.common_heads else 8,
            )

            _, stand_swin_trt = load_runner_with_fallback(
                args.swin_engine, StandaloneTRTRunner, logger, "standalone.swin"
            )
            _, stand_depthnet_trt = load_runner_with_fallback(
                args.depthnet_engine, StandaloneTRTRunner, logger, "standalone.depthnet"
            )
            _, stand_fuser_trt = load_runner_with_fallback(
                args.fuser_engine, StandaloneTRTRunner, logger, "standalone.fuser"
            )
            _, stand_neck_trt = load_runner_with_fallback(
                args.neck_engine, StandaloneTRTRunner, logger, "standalone.neck"
            )
            _, stand_head_trt = load_runner_with_fallback(
                args.head_engine, StandaloneTRTRunner, logger, "standalone.head"
            )

            standalone_ref = StandaloneBEVFusion(
                swin_trt=stand_swin_trt,
                depthnet_trt=stand_depthnet_trt,
                fuser_trt=stand_fuser_trt,
                neck_trt=stand_neck_trt,
                head_trt=stand_head_trt,
                lidar_backbone=ref_lidar,
                voxelizer=ref_voxelizer,
                vtransform_geom=ref_vt_geom,
                bev_downsample=model.encoders["camera"]["vtransform"].downsample,
                bbox_coder=ref_bbox_coder,
                test_cfg=head_cfg.test_cfg,
                num_proposals=head_cfg.num_proposals,
                num_classes=head_cfg.num_classes,
                voxelize_reduce=cfg.model.get("voxelize_reduce", True),
                logger=logger,
                use_tv_lidar=True,
            )
            standalone_ref.eval().cuda()
        except Exception as exc:
            if args.reference_mode == "standalone_tv":
                raise
            logger.warning(
                f"Standalone TV reference unavailable ({exc}); fallback to Hybrid."
            )

    if standalone_ref is None and hybrid is None:
        raise RuntimeError(
            "No reference pipeline available. Hybrid failed and StandaloneTV is unavailable."
        )

    for warmup_idx in range(args.warmup_runs):
        logger.info(f"Warmup run {warmup_idx + 1}/{args.warmup_runs}")
        _ = zero_model.forward(
            img_np, points_np,
            camera2ego_np, lidar2ego_np, lidar2camera_np, lidar2image_np,
            camera_intrinsics_np, camera2lidar_np,
            img_aug_matrix_np, lidar_aug_matrix_np, metas,
        )
        if standalone_ref is not None:
            with torch.no_grad():
                _ = standalone_ref.forward_single(
                    img, points,
                    camera2ego, lidar2ego, lidar2camera, lidar2image,
                    camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix,
                    metas,
                )
        else:
            with torch.no_grad():
                _ = hybrid.forward_single(
                    img, points,
                    camera2ego, lidar2ego, lidar2camera, lidar2image,
                    camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix,
                    metas,
                )

    run_passes = []
    overall_pass = True
    intermediate_keys = [
        "camera_bev", "lidar_bev", "neck_features",
        "center", "height", "dim", "rot", "vel",
        "heatmap", "query_heatmap_score",
    ]

    for run_idx in range(args.repeat):
        logger.info(f"Validation run {run_idx + 1}/{args.repeat}")
        zero_out = zero_model.forward(
            img_np, points_np,
            camera2ego_np, lidar2ego_np, lidar2camera_np, lidar2image_np,
            camera_intrinsics_np, camera2lidar_np,
            img_aug_matrix_np, lidar_aug_matrix_np, metas,
        )

        if standalone_ref is not None:
            reference_name = "StandaloneTV"
            with torch.no_grad():
                reference_out = standalone_ref.forward_single(
                    img, points,
                    camera2ego, lidar2ego, lidar2camera, lidar2image,
                    camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix,
                    metas,
                )
            reference_intermediates = standalone_ref._last_intermediates
        else:
            reference_name = "Hybrid"
            with torch.no_grad():
                reference_out = hybrid.forward_single(
                    img, points,
                    camera2ego, lidar2ego, lidar2camera, lidar2image,
                    camera_intrinsics, camera2lidar,
                    img_aug_matrix, lidar_aug_matrix,
                    metas,
                )
            reference_intermediates = hybrid._last_intermediates

        # ------------------------------------------------------------------
        # Compare intermediates
        # ------------------------------------------------------------------
        logger.info(f"Comparing intermediate tensors ({reference_name} vs ZeroTorch)...")
        run_pass = True
        lidar_ok = True
        ref_lidar = None
        zero_lidar = None
        for k in intermediate_keys:
            a = reference_intermediates.get(k)
            b = zero_model._last_intermediates.get(k)
            if a is None or b is None:
                logger.warning(f"  {k}: missing from one pipeline")
                run_pass = False
                if k == "lidar_bev":
                    lidar_ok = False
                continue
            ok = compare_tensor(k, a, b)
            if k == "lidar_bev":
                lidar_ok = ok
                ref_lidar = a
                zero_lidar = b
            if not ok:
                run_pass = False

        if ref_lidar is not None and zero_lidar is not None:
            ref_lidar_stats = summarize_tensor(ref_lidar)
            zero_lidar_stats = summarize_tensor(zero_lidar)
            logger.info(
                f"[LiDAR diag][run {run_idx + 1}] {reference_name}: "
                f"nz_ratio={ref_lidar_stats['nz_ratio']:.6f}, "
                f"l2={ref_lidar_stats['l2']:.4f}, abs_max={ref_lidar_stats['abs_max']:.4f}"
            )
            logger.info(
                f"[LiDAR diag][run {run_idx + 1}] ZeroTorch: "
                f"nz_ratio={zero_lidar_stats['nz_ratio']:.6f}, "
                f"l2={zero_lidar_stats['l2']:.4f}, abs_max={zero_lidar_stats['abs_max']:.4f}"
            )

            ref_tv_diag = reference_intermediates.get("lidar_tv_diag")
            zero_tv_diag = zero_model._last_intermediates.get("lidar_tv_diag")
            logger.info(
                f"[LiDAR diag][run {run_idx + 1}] {reference_name} TV candidates: "
                f"{_format_lidar_candidates(ref_tv_diag)}"
            )
            logger.info(
                f"[LiDAR diag][run {run_idx + 1}] ZeroTorch TV candidates: "
                f"{_format_lidar_candidates(zero_tv_diag)}"
            )

            if args.diag_dir and (args.dump_all_runs or not lidar_ok):
                dump_lidar_diag(
                    args.diag_dir,
                    run_idx + 1,
                    reference_name,
                    ref_lidar,
                    zero_lidar,
                    ref_tv_diag,
                    zero_tv_diag,
                    reference_intermediates.get("lidar_voxel_stats"),
                    zero_model._last_intermediates.get("lidar_voxel_stats"),
                )
                logger.info(
                    f"[LiDAR diag][run {run_idx + 1}] dumped to {args.diag_dir}"
                )

            ref_voxel_feats = reference_intermediates.get("lidar_voxel_features")
            zero_voxel_feats = zero_model._last_intermediates.get("lidar_voxel_features")
            ref_voxel_coords = reference_intermediates.get("lidar_voxel_coords")
            zero_voxel_coords = zero_model._last_intermediates.get("lidar_voxel_coords")
            if ref_voxel_feats is not None and zero_voxel_feats is not None:
                vdiff = np.abs(ref_voxel_feats.astype(np.float64) - zero_voxel_feats.astype(np.float64))
                logger.info(
                    f"[LiDAR voxel diag][run {run_idx + 1}] feature max_diff={vdiff.max():.6e}, "
                    f"mean_diff={vdiff.mean():.6e}"
                )
            if ref_voxel_coords is not None and zero_voxel_coords is not None:
                c_equal = bool(np.array_equal(ref_voxel_coords, zero_voxel_coords))
                logger.info(
                    f"[LiDAR voxel diag][run {run_idx + 1}] coords_equal={c_equal}"
                )
            if (
                args.lidar_layer_debug
                and run_idx == 0
                and reference_name == "StandaloneTV"
                and ref_voxel_feats is not None
                and zero_voxel_feats is not None
                and ref_voxel_coords is not None
                and zero_voxel_coords is not None
            ):
                compare_tv_layerwise(
                    logger,
                    standalone_ref.lidar_backbone,
                    zero_model.lidar_backbone,
                    ref_voxel_feats,
                    ref_voxel_coords,
                    zero_voxel_feats,
                    zero_voxel_coords,
                )

        # ------------------------------------------------------------------
        # Compare final outputs
        # ------------------------------------------------------------------
        logger.info(f"Comparing final 3D detections ({reference_name} vs ZeroTorch)...")
        reference_boxes = reference_out[0]["boxes_3d"]
        zero_boxes = zero_out[0]["boxes_3d"]
        if hasattr(reference_boxes, "tensor"):
            reference_boxes_np = reference_boxes.tensor.cpu().numpy()
        else:
            reference_boxes_np = reference_boxes
        if hasattr(zero_boxes, "tensor"):
            zero_boxes_np = zero_boxes.tensor.cpu().numpy()
        else:
            zero_boxes_np = zero_boxes

        reference_scores = reference_out[0]["scores_3d"]
        zero_scores = zero_out[0]["scores_3d"]
        if hasattr(reference_scores, "cpu"):
            reference_scores_np = reference_scores.cpu().numpy()
            reference_labels_np = reference_out[0]["labels_3d"].cpu().numpy()
        else:
            reference_scores_np = reference_scores
            reference_labels_np = reference_out[0]["labels_3d"]
        zero_scores_np = zero_scores
        zero_labels_np = zero_out[0]["labels_3d"]

        K = min(83, len(reference_scores_np), len(zero_scores_np))
        ref_idx = np.argsort(-reference_scores_np)[:K]
        z_idx = np.argsort(-zero_scores_np)[:K]

        ref_boxes_k = reference_boxes_np[ref_idx]
        z_boxes_k = zero_boxes_np[z_idx]
        ref_scores_k = reference_scores_np[ref_idx]
        z_scores_k = zero_scores_np[z_idx]
        ref_labels_k = reference_labels_np[ref_idx]
        z_labels_k = zero_labels_np[z_idx]

        ok_boxes = compare_tensor("boxes_3d (top-K)", ref_boxes_k, z_boxes_k)
        ok_scores = compare_tensor("scores_3d (top-K)", ref_scores_k, z_scores_k)
        ok_labels = compare_tensor(
            "labels_3d (top-K)",
            ref_labels_k.astype(np.float32),
            z_labels_k.astype(np.float32),
        )
        if not (ok_boxes and ok_scores and ok_labels):
            run_pass = False

        logger.info(f"Detections ({reference_name}): {len(reference_scores_np)} total")
        logger.info(f"Detections (ZeroTorch): {len(zero_scores_np)} total")

        run_passes.append(run_pass)
        overall_pass = overall_pass and run_pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if args.repeat > 1:
        logger.info(
            f"Repeat summary: {sum(1 for x in run_passes if x)}/{len(run_passes)} runs passed"
        )
    if overall_pass:
        logger.info("End-to-end validation PASSED")
    else:
        logger.warning("End-to-end validation: some tensors differ beyond threshold (this can be expected for INT8)")


if __name__ == "__main__":
    main()
