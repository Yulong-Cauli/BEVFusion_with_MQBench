# High CPU Load Performance Regression Report

## 问题描述

2026-04-08 下午运行 `trt_tv_int8_log2.log` 时，发现 TV INT8 Log2 部署路径的帧率相比 2026-04-05 的 `standalone_eval_tv_int8.log` 出现显著下降。

---

## 现象对比

| 指标 | 04-05 (正常) | 04-08 (异常) | 差异 |
|------|-------------|-------------|------|
| 日志文件 | `logs/standalone_eval_tv_int8.log` | `logs/trt_tv_int8_log2.log` | — |
| 总推理时间 | **1131.0 s** | **1483.9 s** | +352.9 s (+31%) |
| FPS | **5.3 fps** | **4.1 fps** | -1.2 fps (-23%) |
| 纯 CPU Eval 时间 | **82.5 s** | **161.2 s** | +78.7 s (+95%) |
| 100 批次瞬时速度 | 4.8 ~ 5.0 samples/s | 3.6 ~ 3.7 samples/s | -23% |
| NDS / mAP | 0.6893 / 0.6474 | 0.6893 / 0.6472 | 无差异 |

**关键观察**：精度几乎无变化，说明模型输出正确；但纯 CPU 的 NDS 后处理也翻倍，这强烈暗示问题不在 GPU 代码，而在系统整体性能。

---

## 排查过程与证据

### 1. 代码文件时间戳排查

核心代码自 04-05 后**未被修改过**：

```bash
$ ls -l --time-style=+%Y-%m-%d-%H:%M:%S tools/tv_sparse_encoder.py tools/tv_allocator.py tools/trt_infer_standalone.py tools/quant_ptq_minmax.py
-rw-rw-r-- 1 yellowstone yellowstone 55409 2026-04-05-15:07:43 tools/trt_infer_standalone.py
-rw-rw-r-- 1 yellowstone yellowstone 14582 2026-04-03-08:39:29 tools/tv_allocator.py
-rw-rw-r-- 1 yellowstone yellowstone 29948 2026-04-05-15:59:04 tools/tv_sparse_encoder.py
-rw-rw-r-- 1 yellowstone yellowstone 88060 2026-04-08-14:06:55 tools/quant_ptq_minmax.py
```

结论：`tv_sparse_encoder.py`、`tv_allocator.py`、`trt_infer_standalone.py` 这三份核心部署代码在 04-05 18:00 之后零修改。`quant_ptq_minmax.py` 虽然是 04-08 改的，但它属于 PTQ 训练脚本，不影响 TRT standalone 推理。

### 2. Git 状态排查

```bash
$ git status
位于分支 main
未跟踪的文件:
  docs/cmd.md
  trt_standalone_eval.json
```

没有任何已跟踪文件被修改。

### 3. 编译产物排查

```bash
$ ls -ld --time-style=+%Y-%m-%d-%H:%M:%S build/ build_sp39/
drwxrwxr-x 4 yellowstone yellowstone 4096 2026-03-18-10:17:29 build/
drwxrwxr-x 2 yellowstone yellowstone 4096 2026-03-31-10:06:23 build_sp39/
```

`build/` 和 `build_sp39/` 无更新；`torch extension` 缓存也无更新。

### 4. GPU 状态排查

```bash
$ nvidia-smi
GPU 0: RTX 3090, 10MiB / 24576MiB, 0% 利用率, 无其他计算进程
GPU 1-4: 同样空闲
```

GPU 显存和利用率均正常，排除 GPU 被抢占。

### 5. 磁盘 I/O 排查

04-08 运行期间的 `iostat` 输出：
```
sda:  r/s=0.00, rkB/s=0.00, w/s=2.00, wkB/s=64.00, %util=1.20
```

磁盘利用率极低，排除 DataLoader 被 HDD I/O 瓶颈卡住。

### 6. 环境版本排查

```bash
$ conda run -p /media/yellowstone/data2/CYL/spconv23_deploy python -c "import torch, tensorrt, spconv; print(torch.__version__, tensorrt.__version__, spconv.__version__)"
torch 2.0.1+cu118
tensorrt 10.15.1.29
spconv 2.3.8
```

版本与历史运行完全一致。

### 7. 系统 CPU 负载排查（决定性证据）

```bash
$ cat /proc/loadavg
46.35 45.81 45.03 46/5005 3332874

$ uptime
16:51:07 up 73 days, 20:45, 10 users,  load average: 46.95, 45.82, 44.99

$ cat /proc/cpuinfo | grep "MHz" | head -n 20
cpu MHz: 1300.000   <-- 部分核心在低频
cpu MHz: 3900.000
cpu MHz: 3600.000
cpu MHz: 1400.000   <-- 部分核心在低频
```

**服务器当前 load average ≈ 47**，10 个用户在线。高负载 + CPU 降频直接导致：
- Python DataLoader 的预处理（图像 decode、LiDAR 体素化、几何变换）变慢
- GPU 被迫等待 CPU 喂数据，产生 GPU bubble
- 纯 CPU 的 NDS eval 计算时间直接翻倍

---

## 根因推断

**不是代码 bug，不是环境退化，而是服务器 CPU 资源竞争。**

04-05 跑 `standalone_eval_tv_int8.log` 时，服务器负载可能 <5，CPU 能全频跑预处理和后处理，因此能稳定达到 5.3 fps。

04-08 同一时刻，其他用户/任务占满 CPU，导致 BEVFusion 的 CPU-bound 步骤被显著拉长。由于 pipeline 中存在同步点（`cudaDeviceSynchronize`、`torch.cuda.synchronize` 等），CPU 慢会传导为整体 FPS 下降。

---

## 待验证实验

### 实验 1：对比高峰 vs 低谷的系统 CPU 性能

**步骤 A：现在（高负载）跑**
```bash
time /media/yellowstone/data2/CYL/spconv23_deploy/bin/python -c "
import numpy as np, time
a = np.random.rand(8000, 8000).astype('float64')
t0 = time.time()
np.dot(a, a.T)
print('CPU matmul time (high load):', time.time() - t0, 's')
"
```
记录耗时（预期 >10 s）。

**步骤 B：凌晨 1~3 点（低负载）跑**
```bash
# 先确认负载
uptime   # 期望 loadavg < 5

# 再跑同样的测试
time /media/yellowstone/data2/CYL/spconv23_deploy/bin/python -c "
import numpy as np, time
a = np.random.rand(8000, 8000).astype('float64')
t0 = time.time()
np.dot(a, a.T)
print('CPU matmul time (low load):', time.time() - t0, 's')
"
```
记录耗时（预期 <7 s）。

**判定标准**：如果低负载时的 matmul 耗时显著低于高负载（>30% 差距），则 100% 确认帧率下降是 CPU 竞争导致。

### 实验 2：凌晨重跑完整 TRT eval

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u tools/trt_infer_standalone.py \
    --config configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --ckpt pretrained/bevfusion-det.pth \
    --swin-engine swin_int8_sm86.engine \
    --depthnet-engine vtransform_depthnet_int8_sm86.engine \
    --fuser-engine fuser_decoder_int8_sm86.engine \
    --neck-engine camera_neck_int8_sm86.engine \
    --head-engine transfusion_head_int8_sm86.engine \
    --lidar-quant int8 --ptq-ckpt pretrained/ptq_minmax_model.pth \
    --no-torch-lidar \
    2>&1 | tee logs/trt_tv_int8_3am_verify.log
```

**预期结果**：若 loadavg < 5，FPS 应回到 **5.0 ~ 5.3** 区间，eval time 回到 **80 ~ 90 s**。

---

## 影响评估

- **算法精度**：无影响，NDS/mAP 完全一致。
- **部署代码**：无需为本次帧率下降做 hotfix。
- **性能上限**：当前代码中确实存在 `cudaDeviceSynchronize()` 和 `torch.cuda.synchronize()` 等同步点，这些是长期的性能优化项，但不是导致“04-05 vs 04-08 掉帧”的根因。
- **建议**：等 CPU 负载验证完成后，再决定是否修同步点优化代码。

---

## 相关文件

- `logs/standalone_eval_tv_int8.log` — 2026-04-05 正常基准
- `logs/trt_tv_int8_log2.log` — 2026-04-08 异常运行
- `logs/trt_tv_int8_smoke_check.log` — 2026-04-08 单样本冒烟测试 (1060 ms)
- `docs/cmd.md` — 所有复现命令

---

## 2026-04-09 更新：元凶定位 — 实验室同学的 CFD 液滴模拟任务

### 排查结果

通过 `ps aux` 和 `/proc` 排查，确认占满 CPU 的是另一位实验室同学（目录 `/home/yellowstone/Desktop/XY/CFD/`）的 **CFD 参数扫描任务**。

**任务详情：**
- 类型：液滴下落/碰撞数值模拟（自定义求解器 `drop_case`）
- 工作目录：`/home/yellowstone/Desktop/XY/CFD/`
- 启动方式：`xargs -I{} -P 40 bash -lc {}`（40 并发 workers）
- 任务规划：`run_cfd_cases_100x4_oneline.sh` 共 404 行，约 **400 个 case**
- 总 case 数（以 summary CSV 为准）：**400 个**
- 已完成 case（已有 `diag_case.csv`）：**145 个**
- 剩余 case：**约 255 个**
- 启动时间：2026-04-06 22:38
- 最新完成 case 时间：2026-04-09 08:56

### 剩余时间估算

- 已运行约 **2.4 天**，完成 **145/400** 个 case
- 完成速度：~60 cases/天
- 剩余 **255** 个，按当前速度还需 **4.3 天**
- 由于大直径 case（D_8pxxx）计算量更大，保守估计还需 **5 ~ 7 天**

### 对 BEVFusion 的直接干扰

40 个 `drop_case` 进程每个占满一个物理核心，总 CPU 占用约 **4000%+**。这导致：
- BEVFusion DataLoader 的预处理步骤（图像 decode、LiDAR 体素化）抢不到 CPU
- GPU 长时间等待数据，产生大量 GPU bubble
- 纯 CPU 的 NDS eval 后处理时间翻倍

### 自查命令（用于明天汇报）

```bash
# 1. 查看当前系统负载
uptime
cat /proc/loadavg

# 2. 查看 CPU 占用最高的进程（确认 drop_case 是否还在跑）
ps aux --sort=-%cpu | head -n 20

# 3. 统计 CFD 已完成 / 总 case 数
echo "已完成: $(find /home/yellowstone/Desktop/XY/CFD/cases_by_D -name diag_case.csv | wc -l)"
echo "总规划: $(wc -l /home/yellowstone/Desktop/XY/CFD/run_cfd_cases_100x4_summary.csv | awk '{print $1-1}')"

# 4. 查看 CFD 任务日志的最后输出
tail -n 20 /home/yellowstone/Desktop/XY/CFD/logs/cfd_run_100x4.out

# 5. 统计多少 drop_case 进程在运行
pgrep -c -f "drop_case"
```

### 建议

1. **短期**：这些 CFD 任务预计还要跑 5~7 天，想在这台服务器上跑出 5.3 fps 的基线，要么等任务结束，要么与 `XY` 同学协商调整并发数或暂停部分 case。
2. **中期**：当前代码里确实存在 `cudaDeviceSynchronize()` / `torch.cuda.synchronize()` 等同步点，等 CPU 竞争问题解决后，可以进一步修这些瓶颈来榨取更高上限。
3. **验证优先级**：即使现在不清 CFD 任务，凌晨 2~3 点如果 `drop_case` 完成了大部分长 case，也可以尝试跑一下 `实验 2` 看 fps 是否回升。

---

## 记录时间

- 初稿：2026-04-08 17:30
- 更新（定位 CFD 元凶与自查命令）：2026-04-09 09:05
