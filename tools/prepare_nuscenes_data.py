#!/usr/bin/env python
"""
生成 nuScenes 训练所需的 pkl info 文件。

用法:
    python tools/prepare_nuscenes_data.py --root data/nuscenes --version v1.0-trainval
    python tools/prepare_nuscenes_data.py --root data/nuscenes --version v1.0-mini

生成文件:
    data/nuscenes/nuscenes_infos_temporal_train.pkl
    data/nuscenes/nuscenes_infos_temporal_val.pkl
"""
import argparse
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_converter"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nuscenes_converter import create_nuscenes_infos


def main():
    parser = argparse.ArgumentParser(description="Prepare nuScenes data for BEVFusion")
    parser.add_argument("--root", type=str, default="data/nuscenes",
                        help="nuScenes dataset root directory")
    parser.add_argument("--version", type=str, default="v1.0-trainval",
                        choices=["v1.0-trainval", "v1.0-mini"],
                        help="Dataset version")
    parser.add_argument("--max-sweeps", type=int, default=10,
                        help="Max number of lidar sweeps")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"[INFO] nuScenes root: {root}")
    print(f"[INFO] Version: {args.version}")

    # 检查数据目录
    version_dir = os.path.join(root, args.version)
    if not os.path.isdir(version_dir):
        print(f"[ERROR] 找不到 {version_dir}，请确认 nuScenes 数据已正确放置")
        print(f"  需要的目录结构:")
        print(f"    {root}/")
        print(f"    ├── {args.version}/       (元数据 JSON)")
        print(f"    ├── samples/              (关键帧数据)")
        print(f"    │   ├── CAM_FRONT/")
        print(f"    │   ├── CAM_FRONT_LEFT/")
        print(f"    │   ├── ... (6 cameras)")
        print(f"    │   └── LIDAR_TOP/")
        print(f"    ├── sweeps/               (可选，验证时用)")
        print(f"    │   └── LIDAR_TOP/")
        print(f"    └── maps/")
        sys.exit(1)

    # 运行转换器
    print(f"[INFO] 开始生成 info 文件...")
    create_nuscenes_infos(
        root_path=root,
        info_prefix="nuscenes",
        version=args.version,
        max_sweeps=args.max_sweeps,
    )

    # 重命名为 temporal 格式（BEVFusion config 期望的文件名）
    # 转换器生成: nuscenes/nuscenes_infos_train_radar.pkl
    # 目标文件名: data/nuscenes/nuscenes_infos_temporal_train.pkl
    renames = {
        os.path.join("nuscenes", "nuscenes_infos_train_radar.pkl"):
            os.path.join(root, "nuscenes_infos_temporal_train.pkl"),
        os.path.join("nuscenes", "nuscenes_infos_val_radar.pkl"):
            os.path.join(root, "nuscenes_infos_temporal_val.pkl"),
    }

    for src, dst in renames.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"[INFO] {src} -> {dst}")
        elif os.path.exists(os.path.join(root, os.path.basename(src))):
            # 可能直接在 root 下
            alt_src = os.path.join(root, os.path.basename(src))
            shutil.move(alt_src, dst)
            print(f"[INFO] {alt_src} -> {dst}")
        else:
            # 检查是否已经存在目标文件
            if os.path.exists(dst):
                print(f"[INFO] {dst} 已存在，跳过")
            else:
                print(f"[WARN] 找不到源文件 {src}，请手动检查")

    print(f"\n[INFO] 数据准备完成！")
    print(f"  训练 pkl: {renames[list(renames.keys())[0]]}")
    print(f"  验证 pkl: {renames[list(renames.keys())[1]]}")


if __name__ == "__main__":
    main()
