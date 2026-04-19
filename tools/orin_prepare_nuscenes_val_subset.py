#!/usr/bin/env python3
"""
Utilities for preparing a val-only nuScenes subset on storage-limited Orin.

Usage examples:
  # 1) Build keep list from metadata
  python tools/orin_prepare_nuscenes_val_subset.py build-keep \
    --root data/nuscenes --out val_keep_list.txt --max-prev-sweeps 9

  # 2) Extract only needed files from archives
  python tools/orin_prepare_nuscenes_val_subset.py extract \
    --root data/nuscenes --downloads downloads/nuscenes \
    --keep-list val_keep_list.txt --delete-archives

  # 3) Patch nuscenes_converter.py for val-only local files
  python tools/orin_prepare_nuscenes_val_subset.py patch-converter
"""
import argparse
import glob
import json
import os
import tarfile
from pathlib import Path


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_keep_list(root: str, out: str, max_prev_sweeps: int):
    from nuscenes.utils import splits

    meta = os.path.join(root, "v1.0-trainval")
    required = [
        os.path.join(meta, "scene.json"),
        os.path.join(meta, "sample.json"),
        os.path.join(meta, "sample_data.json"),
    ]
    miss = [p for p in required if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError(
            "Missing metadata files:\n  " + "\n  ".join(miss) +
            "\nPlease extract v1.0-trainval metadata under data/nuscenes first."
        )

    scene = _load_json(os.path.join(meta, "scene.json"))
    sample = _load_json(os.path.join(meta, "sample.json"))
    sample_data = _load_json(os.path.join(meta, "sample_data.json"))

    val_scene_names = set(splits.val)
    val_scene_tokens = {s["token"] for s in scene if s["name"] in val_scene_names}
    sd_map = {s["token"]: s for s in sample_data}

    def _infer_channel(sd):
        ch = sd.get("channel")
        if ch:
            return ch
        fn = sd.get("filename", "")
        parts = fn.split("/")
        if len(parts) >= 2 and parts[0] in ("samples", "sweeps"):
            return parts[1]
        return None

    # Build keyframe map from sample_data for metadata variants where sample["data"] is absent.
    keyframe_by_sample = {}
    for sd in sample_data:
        if not sd.get("is_key_frame", False):
            continue
        st = sd.get("sample_token")
        if not st:
            continue
        ch = _infer_channel(sd)
        if not ch:
            continue
        keyframe_by_sample.setdefault(st, {})[ch] = sd["token"]

    keep = set()
    for s in sample:
        if s["scene_token"] not in val_scene_tokens:
            continue

        sensor_map = s.get("data") or keyframe_by_sample.get(s["token"], {})
        lidar_tok = sensor_map.get("LIDAR_TOP")
        if lidar_tok is None:
            continue
        for cam in CAMERAS:
            cam_tok = sensor_map.get(cam)
            if cam_tok is not None and cam_tok in sd_map:
                keep.add(sd_map[cam_tok]["filename"])

        if lidar_tok not in sd_map:
            continue
        keep.add(sd_map[lidar_tok]["filename"])
        cur = sd_map[lidar_tok].get("prev")
        n = 0
        while cur and n < max_prev_sweeps:
            if cur not in sd_map:
                break
            keep.add(sd_map[cur]["filename"])
            cur = sd_map[cur].get("prev")
            n += 1

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in sorted(keep):
            f.write(p + "\n")

    print(f"[OK] keep list written: {out_path}")
    print(f"[OK] kept files: {len(keep)}")


def extract_subset(root: str, downloads: str, keep_list: str, delete_archives: bool):
    with open(keep_list, "r") as f:
        keep = set(x.strip() for x in f if x.strip())

    archives = sorted(glob.glob(os.path.join(downloads, "*.tgz")) + glob.glob(os.path.join(downloads, "*.tar")))
    if not archives:
        raise FileNotFoundError(f"No archives found under: {downloads}")

    os.makedirs(root, exist_ok=True)
    for ap in archives:
        print(f"[INFO] extracting: {ap}")
        with tarfile.open(ap) as tf:
            members = []
            for m in tf:
                n = m.name.lstrip("./")
                if n.startswith("v1.0-trainval/") or n in keep:
                    members.append(m)
            tf.extractall(root, members=members)

        if delete_archives:
            os.remove(ap)
            print(f"[INFO] deleted archive: {ap}")

    print("[OK] extraction finished")


def patch_converter(repo_root: str):
    p = Path(repo_root) / "tools" / "data_converter" / "nuscenes_converter.py"
    if not p.exists():
        raise FileNotFoundError(f"Converter not found: {p}")

    src = p.read_text()
    target = "        mmcv.check_file_exist(lidar_path)\n"
    replacement = (
        "        if not os.path.exists(lidar_path):\n"
        "            continue\n"
    )

    if replacement in src:
        print("[OK] converter already patched")
        return
    if target not in src:
        raise RuntimeError("Target line not found; converter format may have changed.")

    bak = p.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_text(src)
        print(f"[INFO] backup created: {bak}")
    p.write_text(src.replace(target, replacement, 1))
    print(f"[OK] patched: {p}")


def main():
    parser = argparse.ArgumentParser(description="Prepare nuScenes val-only subset for Orin")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_keep = sub.add_parser("build-keep")
    p_keep.add_argument("--root", default="data/nuscenes")
    p_keep.add_argument("--out", default="val_keep_list.txt")
    p_keep.add_argument("--max-prev-sweeps", type=int, default=9)

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--root", default="data/nuscenes")
    p_ext.add_argument("--downloads", default="downloads/nuscenes")
    p_ext.add_argument("--keep-list", default="val_keep_list.txt")
    p_ext.add_argument("--delete-archives", action="store_true")

    p_patch = sub.add_parser("patch-converter")
    p_patch.add_argument("--repo-root", default=".")

    args = parser.parse_args()
    if args.cmd == "build-keep":
        build_keep_list(args.root, args.out, args.max_prev_sweeps)
    elif args.cmd == "extract":
        extract_subset(args.root, args.downloads, args.keep_list, args.delete_archives)
    elif args.cmd == "patch-converter":
        patch_converter(args.repo_root)


if __name__ == "__main__":
    main()
