# PTQ 消融实验整理（模块间 + 组件内）

> 本文档聚焦**实验设计与结果分析**。所有可运行命令的速查版见 [`docs/COMMANDS.md`](COMMANDS.md)。
> 
> 参考来源：`docs/RESULTS_LOG.md`、`runs/round7_* / round8_* / round9_*`、`runs/lloydmax_analysis_100/summary.csv`。

## 0. 统一环境与公共变量

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
mkdir -p logs runs

CFG=configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
CKPT=pretrained/bevfusion-det.pth
```

---

## 1. 模块间消融（你列出的主线组合）

| ID | 组合 | 状态 | 现有精度（NDS/mAP） | 备注 |
|---|---|---|---|---|
| A0 | PTQ 6/8（skip vt + lidar） | ✅ 已做 | 0.7010 / 0.6614 | `RESULTS_LOG` 2026-03-08 |
| A1 | PTQ 8/8：vt KL + lidar MinMax | ✅ 已做 | 0.5706 / 0.5221 | 对应 `8/8 KL(vt)` |
| A2 | PTQ 8/8：vt MinMax + lidar KL/sparseKL | ✅ 已做 | 0.4785 / 0.3791 | `runs/ablation_A2_vtminmax_lidar_kl/20260414_082340.log` |
| A3 | PTQ 8/8：vt KL + lidar Log2 | ✅ 已做 | 0.6875 / 0.6429 | Round 9 最终主线 |
| A4 | PTQ 8/8：vt MinMax + lidar Log2 | ⏳ 待跑 | — | 本节新增命令 |

> 说明：你写的 `PTQ6/8+(KL+lidar)MINMAX` 与 `PTQ6/8+vtKL+lidarMINMAX` 在这里合并为 A1。

### 模块间命令表（单卡）

> `quant_ptq_minmax.py` 当前单实验**不支持多卡并行**（脚本内固定 `single_gpu_test` + `MMDataParallel(device_ids=[0])`）。
> 你新增要跑的 A4 组合已按 **GPU 3 单卡**给出命令。

```bash
# A0: PTQ 6/8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --skip-modules camera/vtransform lidar/backbone \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_A0_ptq6_8 \
  2>&1 | tee logs/ablation_A0_ptq6_8.log

# A1: 8/8 = vt KL + lidar MinMax
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --vtransform-observer kl_divergence \
  --act-observer ema_minmax \
  --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_A1_vtkl_lidar_minmax \
  2>&1 | tee logs/ablation_A1_vtkl_lidar_minmax.log

# A2: 8/8 = vt MinMax + lidar KL/sparseKL（已完成）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --vtransform-observer ema_minmax \
  --act-observer kl_divergence \
  --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_A2_vtminmax_lidar_kl \
  2>&1 | tee logs/ablation_A2_vtminmax_lidar_kl.log

# A3: 8/8 = vt KL + lidar Log2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --vtransform-observer kl_divergence \
  --act-observer log2 \
  --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_A3_vtkl_lidar_log2 \
  2>&1 | tee logs/ablation_A3_vtkl_lidar_log2.log

# A4: 8/8 = vt MinMax + lidar Log2
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --vtransform-observer ema_minmax \
  --act-observer log2 --log-base 2.0 \
  --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_A4_vtminmax_lidar_log2 \
  2>&1 | tee logs/ablation_A4_vtminmax_lidar_log2.log
```

---

## 2. LiDAR 组件内对比（per-tensor / per-channel）

| ID | 设置（均为 skip vt） | 状态 | 现有精度（NDS/mAP） |
|---|---|---|---|
| L0 | MinMax + per_tensor | ✅ 已做 | 0.5751 / 0.5394 |
| L1 | MinMax + per_channel | ✅ 已做 | 0.5733 / 0.5357 |
| L2 | MSE + per_tensor | ⏳ 待补 | — |
| L3 | MSE + per_channel | ✅ 已做 | 0.5768 / 0.5359 |
| L4 | KL/sparseKL + per_tensor | ✅ 已做 | 0.5629 / 0.5054（Round 8） |
| L5 | KL/sparseKL + per_channel | ✅ 已做 | 0.5729 / 0.5366（`round8_ptq7_lidar_sparse_kl_pc_calib128s`） |
| L6 | KL/sparseKL + per_channel + LWC | ✅ 已做 | 0.5700 / 0.5353（`round7_ptq6_lidar_pc_kl_lwc_calib128s`） |
| L7 | Log2 + per_tensor | ✅ 已做 | 0.6849 / 0.6417 |
| L8 | Log2 + per_channel | ✅ 已做 | 0.6721 / 0.6179 |
| L9 | Log2 + per_channel + LWC | ✅ 已做 | 0.6878 / 0.6439 |

> 注意：当前代码里 `--act-observer kl_divergence` 会自动开启 `sparse_mode=True`，即按 sparse-aware KL 路径走。

### LiDAR 命令模板（单卡）

```bash
# L0 MinMax PT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer ema_minmax --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L0_lidar_minmax_pt \
  2>&1 | tee logs/ablation_L0_lidar_minmax_pt.log

# L1 MinMax PC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer ema_minmax --sparse-act-mode per_channel \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L1_lidar_minmax_pc \
  2>&1 | tee logs/ablation_L1_lidar_minmax_pc.log

# L2 MSE PT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer mse --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L2_lidar_mse_pt \
  2>&1 | tee logs/ablation_L2_lidar_mse_pt.log

# L3 MSE PC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer mse --sparse-act-mode per_channel \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L3_lidar_mse_pc \
  2>&1 | tee logs/ablation_L3_lidar_mse_pc.log

# L4 KL/sparseKL PT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer kl_divergence --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L4_lidar_sparsekl_pt \
  2>&1 | tee logs/ablation_L4_lidar_sparsekl_pt.log

# L5 KL/sparseKL PC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer kl_divergence --sparse-act-mode per_channel \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L5_lidar_sparsekl_pc \
  2>&1 | tee logs/ablation_L5_lidar_sparsekl_pc.log

# L6 KL/sparseKL PC + LWC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer kl_divergence --sparse-act-mode per_channel \
  --lwc --lwc-lr 0.01 --lwc-iters 500 \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L6_lidar_sparsekl_pc_lwc \
  2>&1 | tee logs/ablation_L6_lidar_sparsekl_pc_lwc.log

# L7 Log2 PT
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer log2 --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L7_lidar_log2_pt \
  2>&1 | tee logs/ablation_L7_lidar_log2_pt.log

# L8 Log2 PC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer log2 --sparse-act-mode per_channel \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L8_lidar_log2_pc \
  2>&1 | tee logs/ablation_L8_lidar_log2_pc.log

# L9 Log2 PC + LWC
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules camera/vtransform \
  --act-observer log2 --sparse-act-mode per_channel \
  --lwc --lwc-lr 0.01 --lwc-iters 500 \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_L9_lidar_log2_pc_lwc \
  2>&1 | tee logs/ablation_L9_lidar_log2_pc_lwc.log
```

---

## 3. vtransform 组件内对比（与 lidar 解耦）

| ID | 设置（均为 skip lidar） | 状态 | 现有精度（NDS/mAP） |
|---|---|---|---|
| V0 | vt MinMax | ✅ 已做 | 0.6179 / 0.5194 |
| V1 | vt MSEObserver | ✅ 已做 | 0.6421 / 0.5511（`runs/ablation_V1_vt_mse/20260414_082711.log`） |
| V2 | vt KL | ✅ 已做 | 0.7033 / 0.6657 |

> 说明：`sparseKL` 与 `LWC` 是 lidar/sparse conv 路径概念，vtransform 不适用。

```bash
# V0 vt MinMax
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules lidar/backbone \
  --vtransform-observer ema_minmax \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_V0_vt_minmax \
  2>&1 | tee logs/ablation_V0_vt_minmax.log

# V1 vt MSEObserver
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules lidar/backbone \
  --vtransform-observer mse \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_V1_vt_mse \
  2>&1 | tee logs/ablation_V1_vt_mse.log

# V2 vt KL
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
  --skip-modules lidar/backbone \
  --vtransform-observer kl_divergence \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/ablation_V2_vt_kl \
  2>&1 | tee logs/ablation_V2_vt_kl.log
```

---

## 4. Log2 vs LogA（已支持端到端精度）

你提到的 LogA 底数比较，之前主要在 `runs/lloydmax_analysis_100/summary.csv` 做 REL-MSE / MSE。  
现在 `tools/quant_ptq_minmax.py` 已支持 `--log-base`，可以直接做端到端 NDS/mAP。

### 当前批量扫底数进度（端到端）

| 底数 a | 状态 | NDS/mAP | 结果来源 |
|---|---|---|---|
| 1.25 | ✅ 完成 | 0.6997 / 0.6616 | `logs/ablation_LOGA_a1p25.log` |
| 1.41421356 | ✅ 完成 | 0.6941 / 0.6535 | `logs/ablation_LOGA_a1p41421356.log` |
| 1.5 | ✅ 完成 | 0.6948 / 0.6535 | `logs/ablation_LOGA_a1p5.log` |
| 2.0（Log2 基线） | ✅ 完成 | 0.6840 / 0.6410 | `logs/ablation_LOGA_a2p0.log` |
| 2.71828183 | ✅ 完成 | 0.6574 / 0.6031 | `logs/ablation_LOGA_a2p71828183.log` |
| 3.0 | ✅ 完成 | 0.6464 / 0.5877 | `logs/ablation_LOGA_a3p0.log` |
| 4.0 | ✅ 完成 | 0.5434 / 0.4439 | `logs/ablation_LOGA_a4p0.log` |
| 8.0 | ✅ 完成 | 0.2277 / 0.0377 | `logs/ablation_LOGA_a8p0.log` |
| 10.0 | ✅ 完成 | 0.0705 / 0.0019 | `logs/ablation_LOGA_a10p0.log` |
| 16.0 | ✅ 完成 | 0.0267 / 0.0000 | `logs/ablation_LOGA_a16p0.log` |

结论：当前 sweep 中 **a=1.25** 最优；随底数继续增大，精度明显下降。

```bash
# 本节命令已切换为 4 卡（0,1,2,3）；先更新命令，当前未执行。

# 本轮需要补跑：a=8 和 a=16（4卡并行，任务数<卡数时会空闲）
GPU_IDS=(0 1 2 3)
BASES=(8.0 16.0)

for idx in "${!BASES[@]}"; do
  A=${BASES[$idx]}
  GPU=${GPU_IDS[$((idx % 4))]}
  TAG=$(echo "$A" | sed 's/\./p/g')

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  python tools/quant_ptq_minmax.py $CFG --load-from $CKPT \
    --skip-modules camera/vtransform \
    --act-observer log2 --log-base $A \
    --sparse-act-mode per_tensor \
    --calib-batches 128 --calib-shuffle \
    --run-dir runs/ablation_LOGA_a${TAG} \
    2>&1 | tee logs/ablation_LOGA_a${TAG}.log &
done
wait

# 如需全量重扫（4卡并行）可用：
# BASES=(1.25 1.41421356 1.5 2.0 2.71828183 3.0 4.0 8.0 10.0 16.0)
```

产物：
- 每个 run 的 `*.log` 里会打印 `量化模型评估结果`（NDS/mAP）
- 每个 run 的 `ptq_minmax_model.pth` 的 `meta` 会记录 `sparse_log_base`

---

## 5. 全指标三方对比实验（FP32.pth vs PTQ 8/8 KL+Log2 vs TRT 8/8）

> 目标：输出 **nuScenes 数据集 evaluate 返回的全部指标**（不只 NDS/mAP），并统一对比三组结果。  
> 这里固定单卡 **GPU 3**。

### 5.1 跑三组评估

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
mkdir -p runs/compare_triplet logs

CFG=configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
CKPT=pretrained/bevfusion-det.pth

# (1) FP32 基线（全量 val）
conda activate bevfusion_mqbench
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/test.py $CFG $CKPT --eval bbox \
  2>&1 | tee logs/compare_fp32_all_metrics.log

# (2) PTQ 8/8 KL+Log2（全量 val）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python tools/quant_ptq_minmax.py $CFG \
  --load-from $CKPT \
  --vtransform-observer kl_divergence \
  --act-observer log2 --log-base 2.0 \
  --sparse-act-mode per_tensor \
  --calib-batches 128 --calib-shuffle \
  --run-dir runs/compare_triplet/ptq88_kl_log2 \
  2>&1 | tee logs/compare_ptq88_kl_log2_all_metrics.log

# (3) TRT 8/8（TV lidar INT8，全量 val）
conda activate /media/yellowstone/data2/CYL/spconv23_deploy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -u tools/trt_infer_standalone.py \
  --config $CFG \
  --ckpt $CKPT \
  --swin-engine swin_int8_sm86.engine \
  --depthnet-engine vtransform_depthnet_int8_sm86.engine \
  --fuser-engine fuser_decoder_int8_sm86.engine \
  --neck-engine camera_neck_int8_sm86.engine \
  --head-engine transfusion_head_int8_sm86.engine \
  --lidar-quant int8 \
  --ptq-ckpt runs/compare_triplet/ptq88_kl_log2/ptq_minmax_model.pth \
  --no-torch-lidar \
  2>&1 | tee logs/compare_trt88_all_metrics.log

cp -f trt_standalone_eval.json runs/compare_triplet/trt88_metrics_raw.json
```

### 5.2 从日志/JSON导出三份“全指标”文件

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
python - <<'PY'
import json
import math
import re
from pathlib import Path

ROOT = Path("runs/compare_triplet")
ROOT.mkdir(parents=True, exist_ok=True)

def _to_float_or_none(v):
    if isinstance(v, (int, float)):
        v = float(v)
        return None if math.isnan(v) else v
    return v

def normalize_metrics(d):
    return {k: _to_float_or_none(v) for k, v in d.items() if str(k).startswith("object/")}

def extract_metrics_from_log(path: Path):
    text = path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if "object/nds" in ln and "{" in ln and "}" in ln]
    if not lines:
        raise RuntimeError(f"未在日志中找到完整 metrics dict: {path}")
    raw = lines[-1]
    raw = raw[raw.find("{"):raw.rfind("}") + 1]
    # 受控日志输入，允许解析 nan
    data = eval(raw, {"__builtins__": {}}, {"nan": float("nan")})
    return normalize_metrics(data)

fp32 = extract_metrics_from_log(Path("logs/compare_fp32_all_metrics.log"))
ptq = extract_metrics_from_log(Path("logs/compare_ptq88_kl_log2_all_metrics.log"))
trt = normalize_metrics(json.loads(Path("runs/compare_triplet/trt88_metrics_raw.json").read_text()))

(ROOT / "fp32_metrics.json").write_text(json.dumps(fp32, ensure_ascii=False, indent=2))
(ROOT / "ptq88_kl_log2_metrics.json").write_text(json.dumps(ptq, ensure_ascii=False, indent=2))
(ROOT / "trt88_metrics.json").write_text(json.dumps(trt, ensure_ascii=False, indent=2))
print("saved:", ROOT / "fp32_metrics.json")
print("saved:", ROOT / "ptq88_kl_log2_metrics.json")
print("saved:", ROOT / "trt88_metrics.json")
PY
```

### 5.3 生成“所有指标”对比表（CSV + Markdown）

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
python - <<'PY'
import csv
import json
from pathlib import Path

ROOT = Path("runs/compare_triplet")
fp32 = json.loads((ROOT / "fp32_metrics.json").read_text())
ptq = json.loads((ROOT / "ptq88_kl_log2_metrics.json").read_text())
trt = json.loads((ROOT / "trt88_metrics.json").read_text())

keys = sorted(set(fp32) | set(ptq) | set(trt))

def fmt(v):
    return "" if v is None else f"{v:.10f}"

rows = []
for k in keys:
    f = fp32.get(k)
    p = ptq.get(k)
    t = trt.get(k)
    dp = (p - f) if isinstance(p, float) and isinstance(f, float) else None
    dt = (t - f) if isinstance(t, float) and isinstance(f, float) else None
    rows.append([k, f, p, t, dp, dt])

csv_path = ROOT / "all_metrics_compare.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "fp32", "ptq88_kl_log2", "trt88", "delta_ptq_vs_fp32", "delta_trt_vs_fp32"])
    for r in rows:
        w.writerow([r[0], fmt(r[1]), fmt(r[2]), fmt(r[3]), fmt(r[4]), fmt(r[5])])

md_path = ROOT / "all_metrics_compare.md"
with md_path.open("w") as f:
    f.write("| metric | fp32 | ptq88_kl_log2 | trt88 | delta_ptq_vs_fp32 | delta_trt_vs_fp32 |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r[0]} | {fmt(r[1])} | {fmt(r[2])} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} |\n")

print("saved:", csv_path)
print("saved:", md_path)
PY
```

产物：
- `runs/compare_triplet/fp32_metrics.json`
- `runs/compare_triplet/ptq88_kl_log2_metrics.json`
- `runs/compare_triplet/trt88_metrics.json`
- `runs/compare_triplet/all_metrics_compare.csv`
- `runs/compare_triplet/all_metrics_compare.md`

---

## 6. BRECQ vs KL+Log2 主对比（DDP 4 卡 A100，2026-04-27 新增）

> 目标：用 **完整 BRECQ（AdaRound 权重 + QDrop 激活 + 子模块级 reconstruction）** 跑全网 8/8 PTQ，与现有最佳 A3（vt KL + lidar Log2，NDS=0.6875）做端到端对比，证明 KL+Log2 路线在 BEVFusion 上仍有竞争力。

### 6.0 启动前提（不可跳过）

1. **必须 4 卡 A100 全空闲**：脚本 `tools/quant_ptq_brecq.py` 启动时会跑 `nvidia-smi`，任一可见卡 `mem.used ≥ 5GB` 或 `util ≥ 10%` 都会拒绝运行。
2. **必须 DDP 启动**：单进程会报错（与实验室"四卡共享"规定冲突）。

```bash
# 启动前手动确认（脚本会再查一次）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

> ⚠ **启动器选择**：当前环境 OpenMPI 为 5.x（`prterun`），`torchpack dist-run` 生成的 `-mca btl ^openib` 参数与 prterun 不兼容，会导致进程初始化时 `mca_btl_smcuda` double-free crash。请改用 **`torchrun`**（PyTorch 原生 DDP launcher）。

### 6.1 BRECQ 全量跑（4 卡 DDP via torchrun）

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
conda activate bevfusion_mqbench
mkdir -p runs/compare_triplet logs

CFG=configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
CKPT=pretrained/bevfusion-det.pth

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --standalone \
  tools/quant_ptq_brecq.py $CFG \
  --load-from $CKPT \
  --calib-batches 256 \
  --cache-batches 64 \
  --recon-iters 2000 \
  --w-lr 4e-4 --a-lr 4e-5 \
  --drop-prob 0.5 \
  --round-loss-weight 0.01 \
  --warm-up 0.2 \
  --sparse-act-mode per_tensor \
  --run-dir runs/compare_triplet/brecq88 \
  2>&1 | tee logs/compare_brecq88_all_metrics.log

# 产物：
#   runs/compare_triplet/brecq88/ptq_brecq_model.pth
#   runs/compare_triplet/brecq88/configs.yaml
#   runs/compare_triplet/brecq88/<timestamp>_rank{0..3}.log
```

**预算**：A100×4 DDP 下 ~1.5h（BRECQ 优化 ~65min + 全量 val 评估 ~25min）。

### 6.2 加入 §5 的全指标对比

把 §5.2 / §5.3 的 Python 脚本扩展加一列 `brecq88`：

```bash
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
python - <<'PY'
import json, math, csv
from pathlib import Path

ROOT = Path("runs/compare_triplet")

def _to_f(v):
    if isinstance(v, (int, float)):
        v = float(v); return None if math.isnan(v) else v
    return v

def normalize(d):
    return {k: _to_f(v) for k, v in d.items() if str(k).startswith("object/")}

def from_log(path):
    text = Path(path).read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if "object/nds" in ln and "{" in ln and "}" in ln]
    raw = lines[-1]
    raw = raw[raw.find("{"):raw.rfind("}") + 1]
    return normalize(eval(raw, {"__builtins__": {}}, {"nan": float("nan")}))

fp32  = from_log("logs/compare_fp32_all_metrics.log")
ptq   = from_log("logs/compare_ptq88_kl_log2_all_metrics.log")
brecq = from_log("logs/compare_brecq88_all_metrics.log")
trt   = normalize(json.loads((ROOT / "trt88_metrics_raw.json").read_text()))

(ROOT / "brecq88_metrics.json").write_text(json.dumps(brecq, ensure_ascii=False, indent=2))

keys = sorted(set(fp32) | set(ptq) | set(brecq) | set(trt))
def fmt(v): return "" if v is None else f"{v:.10f}"

with (ROOT / "all_metrics_compare_v2.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["metric", "fp32", "ptq_kl_log2", "brecq", "trt", "Δptq", "Δbrecq", "Δtrt"])
    for k in keys:
        f, p, b, t = fp32.get(k), ptq.get(k), brecq.get(k), trt.get(k)
        dp = (p - f) if isinstance(p, float) and isinstance(f, float) else None
        db = (b - f) if isinstance(b, float) and isinstance(f, float) else None
        dt = (t - f) if isinstance(t, float) and isinstance(f, float) else None
        w.writerow([k, fmt(f), fmt(p), fmt(b), fmt(t), fmt(dp), fmt(db), fmt(dt)])

with (ROOT / "all_metrics_compare_v2.md").open("w") as fh:
    fh.write("| metric | fp32 | ptq_kl_log2 | brecq | trt | Δptq | Δbrecq | Δtrt |\n")
    fh.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for k in keys:
        f, p, b, t = fp32.get(k), ptq.get(k), brecq.get(k), trt.get(k)
        dp = (p - f) if isinstance(p, float) and isinstance(f, float) else None
        db = (b - f) if isinstance(b, float) and isinstance(f, float) else None
        dt = (t - f) if isinstance(t, float) and isinstance(f, float) else None
        fh.write(f"| {k} | {fmt(f)} | {fmt(p)} | {fmt(b)} | {fmt(t)} | {fmt(dp)} | {fmt(db)} | {fmt(dt)} |\n")

print("saved: all_metrics_compare_v2.{csv,md}")
PY
```

### 6.3 BRECQ 关键超参说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--calib-batches` | 256 | 初始 MinMax 校准 batch 数；DDP 下 4 卡各跑 64 |
| `--cache-batches` | 64 | 每个子模块缓存的 (input, fp32_output) 对数；越多越精确，越多越占显存 |
| `--recon-iters` | 2000 | 每个子模块的 reconstruction 迭代数；BRECQ 论文用 20000，先取 2000 看趋势 |
| `--w-lr` | 4e-4 | AdaRound alpha 学习率（Adam） |
| `--a-lr` | 4e-5 | QDrop 激活 scale 学习率（Adam，cosine decay） |
| `--drop-prob` | 0.5 | QDrop 训练期激活随机丢弃概率（论文默认） |
| `--round-loss-weight` | 0.01 | round_loss / lp_loss 权衡系数 |
| `--warm-up` | 0.2 | beta linear-decay warmup 比例（先 b=20，warmup 后线性降到 b=2） |
| `--sparse-act-mode` | per_tensor | 与 A3 对齐 |

### 6.4 已通过的冒烟测试（GPU-free，2026-04-27）

- ✅ syntax / import / `--help` parse
- ✅ pre-flight GPU 检查（4 个用例：全空闲 / 单卡忙 / 可见过滤）
- ✅ AdaRound + QDrop 在 `_BRECQConv2d` / `_BRECQLinear` 上的完整生命周期：observer → fake_quant → init alpha → forward → backward 梯度可达 → snap
- ✅ **torchrun 4 卡 DDP smoke 测试**：`--smoke --bypass-gpu-check` 成功跑完全部 8 个子模块 BRECQ 插桩（camera/backbone 52 层、camera/neck 4 层、camera/vtransform 9 层、lidar/backbone 21 SparseConv、fuser fx、decoder/backbone fx、decoder/neck 1 层、heads/object 7 层）

**已定位并解决的阻塞**：
- ~~⏳ `torchpack dist-run` + OpenMPI 5.x (`prterun`) 的 `mca_btl_smcuda` double-free~~ → **已解决：改用 `torchrun`**

**剩余（GPU 空闲即可跑）**：
- ⏳ 真实 BRECQ reconstruction（calib → cache IO → submodule recon → eval）
- ⏳ `multi_gpu_test` 全量 val 评估（6019 帧）

GPU 已空闲，可直接跑 §6.1 的真实 BRECQ。

