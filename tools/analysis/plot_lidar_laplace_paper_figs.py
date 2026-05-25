#!/usr/bin/env python3
import argparse
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from matplotlib.patches import Polygon, Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return root / pp


def rot2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def bev_box_corners(cx: float, cy: float, w: float, l: float, yaw: float) -> np.ndarray:
    # Local frame: x-forward(length), y-left(width)
    local = np.array(
        [
            [l / 2, w / 2],
            [l / 2, -w / 2],
            [-l / 2, -w / 2],
            [-l / 2, w / 2],
        ],
        dtype=np.float32,
    )
    return local @ rot2d(yaw).T + np.array([cx, cy], dtype=np.float32)


def box3d_corners_lidar(cx: float, cy: float, cz: float, w: float, l: float, h: float, yaw: float) -> np.ndarray:
    # 8 corners, z uses box center convention from nuscenes infos
    x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float32)
    y_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float32)
    z_c = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dtype=np.float32)
    corners = np.stack([x_c, y_c], axis=1) @ rot2d(yaw).T
    out = np.zeros((8, 3), dtype=np.float32)
    out[:, 0] = corners[:, 0] + cx
    out[:, 1] = corners[:, 1] + cy
    out[:, 2] = z_c + cz
    return out


def lidar_to_cam(points_lidar: np.ndarray, sensor2lidar_rot: np.ndarray, sensor2lidar_trans: np.ndarray) -> np.ndarray:
    # x_lidar = R * x_cam + t  -> x_cam = (x_lidar - t) * R (row-vector form)
    return (points_lidar - sensor2lidar_trans[None, :]) @ sensor2lidar_rot


def project_cam(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-4
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    if valid.any():
        pc = points_cam[valid]
        uv_valid = (pc @ K.T)
        uv_valid = uv_valid[:, :2] / uv_valid[:, 2:3]
        uv[valid] = uv_valid
    return uv, valid


def pick_background_center(points_xy: np.ndarray, obj_center: np.ndarray) -> np.ndarray:
    xs = np.arange(-34.0, 35.0, 4.0)
    ys = np.arange(-34.0, 35.0, 4.0)
    best = None
    best_cnt = 10**9
    for x in xs:
        for y in ys:
            c = np.array([x, y], dtype=np.float32)
            if np.linalg.norm(c - obj_center) < 14.0:
                continue
            m = (
                (points_xy[:, 0] >= x - 2.0)
                & (points_xy[:, 0] <= x + 2.0)
                & (points_xy[:, 1] >= y - 2.0)
                & (points_xy[:, 1] <= y + 2.0)
            )
            cnt = int(m.sum())
            if cnt < best_cnt:
                best_cnt = cnt
                best = c
                if cnt == 0:
                    return best
    return best if best is not None else np.array([-30.0, 30.0], dtype=np.float32)


def draw_kernel_grid_bev(ax, center_xy: np.ndarray, cell: float, pattern: np.ndarray, title: str):
    h, w = pattern.shape
    sx = center_xy[0] - (w * cell) / 2
    sy = center_xy[1] - (h * cell) / 2
    for i in range(h):
        for j in range(w):
            v = pattern[i, j]
            color = "#1f77b4" if v > 0 else "#ffffff"
            alpha = 0.8 if v > 0 else 0.5
            rect = Rectangle(
                (sx + j * cell, sy + i * cell),
                cell,
                cell,
                linewidth=0.8,
                edgecolor="black",
                facecolor=color,
                alpha=alpha,
                zorder=7,
            )
            ax.add_patch(rect)
    ax.text(sx, sy - 0.9, title, fontsize=8, color="black", zorder=8)


def draw_kernel_grid_cam(img: np.ndarray, top_left: Tuple[int, int], cell: int, pattern: np.ndarray, title: str):
    x0, y0 = top_left
    h, w = pattern.shape
    for i in range(h):
        for j in range(w):
            v = pattern[i, j]
            x1, y1 = x0 + j * cell, y0 + i * cell
            x2, y2 = x1 + cell, y1 + cell
            color = (255, 120, 20) if v > 0 else (255, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
    cv2.putText(img, title, (x0, max(16, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_bev_figure(
    out_path: Path,
    points: np.ndarray,
    box_xywlh_yaw: np.ndarray,
    sample_idx: int,
):
    x = points[:, 0]
    y = points[:, 1]
    m = (x > -45) & (x < 45) & (y > -45) & (y < 45)
    x, y = x[m], y[m]

    cx, cy, cz, w, l, h, yaw = box_xywlh_yaw.tolist()
    a_poly = bev_box_corners(cx, cy, w, l, yaw)
    b_outer = bev_box_corners(cx, cy, w * 1.25, l * 1.25, yaw)
    c_center = pick_background_center(np.stack([x, y], axis=1), np.array([cx, cy], dtype=np.float32))

    fig = plt.figure(figsize=(8.2, 8.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=0.6, c="#4d4d4d", alpha=0.45, linewidths=0, zorder=1)

    ax.add_patch(Polygon(a_poly, closed=True, facecolor="#2ca02c", edgecolor="#1f7f1f", alpha=0.35, lw=1.8, zorder=4))
    ax.text(cx, cy, "A: Foreground interior", fontsize=8, color="#145a14", zorder=6)

    ax.add_patch(Polygon(b_outer, closed=True, fill=False, edgecolor="#ff7f0e", lw=2.2, ls="--", zorder=5))
    ax.add_patch(Polygon(a_poly, closed=True, fill=False, edgecolor="#ff7f0e", lw=1.2, ls="--", zorder=5))
    b_txt = b_outer.mean(axis=0)
    ax.text(b_txt[0] + 1.2, b_txt[1] + 1.2, "B: Object boundary", fontsize=8, color="#b35300", zorder=6)

    c_size = 4.0
    c_rect = Rectangle(
        (c_center[0] - c_size / 2, c_center[1] - c_size / 2),
        c_size,
        c_size,
        facecolor="#1f77b4",
        edgecolor="#104a79",
        alpha=0.30,
        lw=1.8,
        zorder=4,
    )
    ax.add_patch(c_rect)
    ax.text(c_center[0] - 3.5, c_center[1] + 3.0, "C: Background (strict 0)", fontsize=8, color="#0b3d66", zorder=6)

    pattern_a = np.ones((3, 3), dtype=np.int32)
    pattern_b = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.int32)
    draw_kernel_grid_bev(ax, np.array([cx, cy]), cell=0.8, pattern=pattern_a, title="3x3 in A (mostly valid)")
    edge_center = np.array([cx + math.cos(yaw) * (l * 0.65), cy + math.sin(yaw) * (l * 0.65)], dtype=np.float32)
    draw_kernel_grid_bev(ax, edge_center, cell=0.8, pattern=pattern_b, title="3x3 in B (partial empty)")

    explain = (
        "A/B: non-empty sparse voxels -> responses exist,\n"
        "then BatchNorm compresses amplitudes near 0.\n"
        "C: empty voxels -> sparse-to-dense fill = strict 0."
    )
    ax.text(
        -44.0,
        -43.0,
        explain,
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#bdbdbd", alpha=0.9, boxstyle="round,pad=0.3"),
        zorder=10,
    )

    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_xlabel("X (m, forward)")
    ax.set_ylabel("Y (m, left)")
    ax.set_title(f"Sample {sample_idx:02d} - LiDAR BEV with A/B/C regions", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def draw_camera_figure(
    out_path: Path,
    image_bgr: np.ndarray,
    proj_poly: np.ndarray,
    sample_idx: int,
):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    poly_i = np.round(proj_poly).astype(np.int32)

    if len(poly_i) >= 3:
        cv2.fillPoly(img, [poly_i], color=(60, 180, 75))
        img = cv2.addWeighted(img, 0.85, image_bgr, 0.15, 0)
        cv2.polylines(img, [poly_i], True, color=(20, 120, 20), thickness=2)
        cv2.polylines(img, [poly_i], True, color=(0, 140, 255), thickness=5)
        m = poly_i.mean(axis=0).astype(int)
        cv2.putText(img, "A: Foreground interior", (m[0] - 80, m[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 120, 20), 2)
        cv2.putText(img, "B: Object boundary", (m[0] - 70, m[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 220), 2)

    c_x1, c_y1 = 30, 30
    c_x2, c_y2 = min(260, w - 30), min(170, h - 30)
    cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (180, 80, 40), thickness=-1)
    img = cv2.addWeighted(img, 0.90, image_bgr, 0.10, 0)
    cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), (130, 60, 25), thickness=2)
    cv2.putText(img, "C: Background (strict 0 after dense fill)", (c_x1 + 6, c_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 1)

    pattern_a = np.ones((3, 3), dtype=np.int32)
    pattern_b = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=np.int32)
    draw_kernel_grid_cam(img, (w - 220, 38), 22, pattern_a, "3x3 in A (valid)")
    draw_kernel_grid_cam(img, (w - 220, 132), 22, pattern_b, "3x3 in B (partial)")

    explain_lines = [
        "A/B: sparse non-empty responses exist; BN compresses amplitudes near 0.",
        "C: empty voxels become strict 0 in sparse-to-dense conversion.",
    ]
    y0 = h - 42
    for i, t in enumerate(explain_lines):
        cv2.putText(img, t, (20, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (15, 15, 15), 1, cv2.LINE_AA)

    cv2.putText(img, f"Sample {sample_idx:02d} - CAM_FRONT with A/B/C regions", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)
    cv2.imwrite(str(out_path), img)


def choose_visible_box_in_cam(info: dict, cam_name: str, img_w: int, img_h: int) -> Optional[Tuple[int, np.ndarray]]:
    boxes = info["gt_boxes"]
    if len(boxes) == 0:
        return None

    cam = info["cams"][cam_name]
    R = np.array(cam["sensor2lidar_rotation"], dtype=np.float32)
    t = np.array(cam["sensor2lidar_translation"], dtype=np.float32)
    K = np.array(cam["cam_intrinsic"], dtype=np.float32)
    num_pts = np.array(info.get("num_lidar_pts", np.ones((len(boxes),), dtype=np.float32)))
    dists = np.linalg.norm(boxes[:, :2], axis=1)
    order = np.argsort(-(num_pts + 1e-3) / (dists + 1e-3))

    for idx in order.tolist():
        cx, cy, cz, w, l, h, yaw = boxes[idx][:7]
        corners = box3d_corners_lidar(cx, cy, cz, w, l, h, yaw)
        corners_cam = lidar_to_cam(corners, R, t)
        uv, valid = project_cam(corners_cam, K)
        if valid.sum() < 6:
            continue
        u = uv[valid, 0]
        v = uv[valid, 1]
        if u.max() < 0 or v.max() < 0 or u.min() > img_w - 1 or v.min() > img_h - 1:
            continue
        # convex hull polygon for camera overlay
        hull = cv2.convexHull(np.round(uv[valid]).astype(np.float32)).squeeze(1)
        if hull.ndim != 2 or hull.shape[0] < 3:
            continue
        return idx, hull
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate 20 separate paper figures for LiDAR Laplacian behavior (10 CAM + 10 BEV).")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--infos", type=str, default="data/nuscenes/nuscenes_infos_temporal_val.pkl")
    parser.add_argument("--out-dir", type=str, default="artifacts/paper_laplace_figs")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos_path = resolve_path(root, args.infos)

    with open(infos_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else data

    selected: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    # tuple: (global_idx, box_idx, box7, proj_poly, image_shape(h,w))
    for global_idx, info in enumerate(infos):
        if len(selected) >= args.num_samples:
            break
        if args.camera not in info.get("cams", {}):
            continue

        cam_path = resolve_path(root, info["cams"][args.camera]["data_path"])
        if not cam_path.exists():
            continue
        img = cv2.imread(str(cam_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        choice = choose_visible_box_in_cam(info, args.camera, w, h)
        if choice is None:
            continue
        box_idx, hull = choice
        box7 = np.array(info["gt_boxes"][box_idx][:7], dtype=np.float32)
        selected.append((global_idx, box_idx, box7, hull, np.array([h, w], dtype=np.int32)))

    if len(selected) < args.num_samples:
        raise RuntimeError(f"Only found {len(selected)} valid samples with visible boxes in {args.camera}, need {args.num_samples}.")

    # Spread chosen indices for diversity.
    if len(selected) > args.num_samples:
        idxs = np.linspace(0, len(selected) - 1, args.num_samples, dtype=int).tolist()
        selected = [selected[i] for i in idxs]

    for i, (global_idx, box_idx, box7, hull, _) in enumerate(selected, start=1):
        info = infos[global_idx]
        cam_path = resolve_path(root, info["cams"][args.camera]["data_path"])
        lidar_path = resolve_path(root, info["lidar_path"])

        img = cv2.imread(str(cam_path))
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)[:, :3]

        cam_out = out_dir / f"laplace_cam_{i:02d}.png"
        bev_out = out_dir / f"laplace_bev_{i:02d}.png"

        draw_camera_figure(cam_out, img, hull, i)
        draw_bev_figure(bev_out, points, box7, i)

        print(f"[{i:02d}/{args.num_samples}] saved: {cam_out.name}, {bev_out.name}")

    print(f"\nDone. Generated {args.num_samples * 2} separate figures in: {out_dir}")


if __name__ == "__main__":
    main()
