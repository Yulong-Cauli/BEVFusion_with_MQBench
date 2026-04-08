首先你需要了解这个项目，主要是 README 和 docs/REPORT.md，现在在做部署。然后看 docs/NEXT_PLAN.md 和 docs/Handoff8.md（重点看 Session 3）。

## 当前任务：修复 TV backbone FP16 帧率

TV backbone（`--no-torch-lidar`）帧率只有 ~0.6 fps，PyTorch 路径是 ~5.2 fps，慢了 8 倍。

### 根因

BN 和残差 add 走 CPU numpy roundtrip：每层 conv 后 `features.cpu().numpy()` → numpy 计算 → `torch.from_numpy().cuda()`，21 次 BN + 8 次残差 add = 29 次 GPU↔CPU 数据拷贝 + 同步。

### 需要修改的文件

- `tools/tv_sparse_encoder.py`：
  - `batch_norm_forward()`：当前走 CPU numpy，需改为 GPU 上计算
  - `_basic_block()` 中的残差 add：当前走 CPU numpy，需改为 GPU 上计算

### 优化方向

1. **BN fuse 进 implicit_gemm**（最优）：`ConvGemmOps.implicit_gemm` 支持 `bias` + `act_type` 参数，可以把 BN 的 scale/shift fuse 成 bias，同时 fuse ReLU。这样 conv+BN+ReLU 一次 kernel 完成，零额外开销。参考 `temp/spconv/example/libspconv/main.cu` 第 239-242 行和第 436-443 行的 bias + Activation::kReLU 用法。

2. **GPU 上做 BN**（次优）：用 `InferenceOps` 或写简单的逐元素 CUDA kernel（scale * x + shift），避免 CPU roundtrip。

3. **残差 add**：用 torch 在 GPU 上做加法（`torch.add`），或用 cuBLAS `axpy`，或用 `tv.Tensor` 的 GPU 操作。

### 约束

- 环境：`conda activate /media/yellowstone/data2/CYL/spconv23_deploy`
- GPU：RTX 3090（GPU 0/1/3/4），A100（GPU 2）跳过
- 测试命令见 `docs/deploy_cmd.md` Section 7
- 冒烟测试：加 `--test-single`，完整评估不加
- 帧率目标：接近 PyTorch 路径的 ~5 fps
- NDS 精度不能下降（预期 ≈ 0.7039）

### 当前正在跑的测试

`logs/standalone_eval_tv_fp16.log` 正在跑完整 6019 样本 NDS 评估（~0.6 fps，预计 ~2.8 小时），跑完后检查 NDS 结果。这个测试验证的是精度，帧率不具参考意义。

### 其他待办

- Phase 7 收尾：PyTorch spconv INT8 NDS 评估，命令见 `docs/deploy_cmd.md` Section 7 最后一段
- sparse→dense transpose bug 已修复（Session 3），如果 NDS 结果异常检查 `standalone_eval_tv_fp16.log` 是否用的旧代码
