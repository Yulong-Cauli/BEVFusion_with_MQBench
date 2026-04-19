# TEMP CMD（覆盖式）

> 根因：`tools/orin_prepare_nuscenes_val_subset.py` 在你那边已被多次热修，文件状态漂移，出现 `NameError: keyframe_by_sample`；同时你的 metadata 里 `sample["data"]` 不稳定。  
> 解决：**下面命令完全不依赖这个脚本**，直接可跑。

```bash
cd ~/work
conda activate bo
export PYTHONPATH=$PWD:/usr/lib/python3.10/dist-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1
mkdir -p downloads/nuscenes data/nuscenes
```

## 1) 生成 val 保留清单（鲁棒版）

```bash
cd ~/work
python - <<'PY'
import os, json
from nuscenes.utils import splits

root = "data/nuscenes"
meta = os.path.join(root, "v1.0-trainval")
cams = ["CAM_FRONT","CAM_FRONT_RIGHT","CAM_FRONT_LEFT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
max_prev = 9

scene = json.load(open(os.path.join(meta, "scene.json")))
sample = json.load(open(os.path.join(meta, "sample.json")))
sample_data = json.load(open(os.path.join(meta, "sample_data.json")))

val_scene_names = set(splits.val)
val_scene_tokens = {s["token"] for s in scene if s["name"] in val_scene_names}
sd_map = {s["token"]: s for s in sample_data}

def infer_channel(sd):
    ch = sd.get("channel")
    if ch:
        return ch
    fn = sd.get("filename", "")
    parts = fn.split("/")
    if len(parts) >= 2 and parts[0] in ("samples", "sweeps"):
        return parts[1]
    return None

keyframe_by_sample = {}
for sd in sample_data:
    if not sd.get("is_key_frame", False):
        continue
    st = sd.get("sample_token")
    if not st:
        continue
    ch = infer_channel(sd)
    if not ch:
        continue
    keyframe_by_sample.setdefault(st, {})[ch] = sd["token"]

keep = set()
for s in sample:
    if s["scene_token"] not in val_scene_tokens:
        continue
    sensor_map = s.get("data") or keyframe_by_sample.get(s["token"], {})
    lidar_tok = sensor_map.get("LIDAR_TOP")
    if lidar_tok is None or lidar_tok not in sd_map:
        continue

    for c in cams:
        t = sensor_map.get(c)
        if t is not None and t in sd_map:
            keep.add(sd_map[t]["filename"])

    keep.add(sd_map[lidar_tok]["filename"])
    cur = sd_map[lidar_tok].get("prev")
    n = 0
    while cur and n < max_prev:
        if cur not in sd_map:
            break
        keep.add(sd_map[cur]["filename"])
        cur = sd_map[cur].get("prev")
        n += 1

with open("val_keep_list.txt", "w") as f:
    for p in sorted(keep):
        f.write(p + "\n")
print("OK keep files:", len(keep))
PY
```

## 2) 自动生成 10 个 blobs 链接（不用手抄）

```bash
cd ~/work/downloads/nuscenes
for i in $(seq -w 1 10); do
  echo "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval${i}_blobs.tgz"
done > urls.txt
cat urls.txt
```

## 3) 低磁盘模式：逐个下载→抽取子集→删除压缩包

```bash
cd ~/work
while IFS= read -r url; do
  [ -z "$url" ] && continue
  echo "==> downloading: $url"
  aria2c -x 8 -s 8 -c -d downloads/nuscenes "$url" || exit 1

  python - <<'PY'
import glob, os, tarfile
root = "data/nuscenes"
downloads = "downloads/nuscenes"
keep = set(x.strip() for x in open("val_keep_list.txt") if x.strip())
archives = sorted(glob.glob(os.path.join(downloads, "*.tgz")) + glob.glob(os.path.join(downloads, "*.tar")))
for ap in archives:
    print("[INFO] extracting:", ap)
    with tarfile.open(ap) as tf:
        members = []
        for m in tf:
            n = m.name.lstrip("./")
            if n.startswith("v1.0-trainval/") or n in keep:
                members.append(m)
        tf.extractall(root, members=members)
    os.remove(ap)
    print("[INFO] deleted:", ap)
PY

  echo "==> disk status"
  df -h .
done < downloads/nuscenes/urls.txt
```

## 4) patch converter + 生成 pkl

```bash
cd ~/work
python - <<'PY'
from pathlib import Path
p = Path("tools/data_converter/nuscenes_converter.py")
src = p.read_text()
target = "        mmcv.check_file_exist(lidar_path)\n"
replacement = "        if not os.path.exists(lidar_path):\n            continue\n"
if replacement not in src:
    p.write_text(src.replace(target, replacement, 1))
    print("patched converter")
else:
    print("converter already patched")
PY

python tools/prepare_nuscenes_data.py --root data/nuscenes --version v1.0-trainval
```

## 5) 检查

```bash
cd ~/work
ls data/nuscenes/v1.0-trainval | head
ls data/nuscenes/samples | head
ls data/nuscenes/sweeps/LIDAR_TOP | head
ls data/nuscenes/nuscenes_infos_temporal_val.pkl
```
