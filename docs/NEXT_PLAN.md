我想保持现有组件（SwinT、BEVPooling、SpConv）和量化策略不变，从仿真走向 TensorRT 真实部署的需求，核心挑战在于**将 PyTorch 的自定义算子和 FakeQuant 节点正确映射为 TensorRT 可以理解的引擎，并处理底层硬件对非标准量化（如 Log2）的兼容性**。

------

### 给 Agent 的 BEVFusion 混合量化 TRT 部署执行计划

**任务目标**：将包含 FakeQuant 节点（含自定义 Log2 量化）的 BEVFusion PyTorch 模型导出为带有 Q/DQ（Quantize/Dequantize）节点的 ONNX，并结合自定义 TRT 插件（Plugins）编译为 TensorRT 引擎。不更换算法组件，维持 Hybrid 推理架构或走向全 TRT。

#### Phase 1: Swin-Transformer 静态化与标准 ONNX 导出

*目标：解决动态控制流导致 `torch.fx` 和 ONNX 导出失败的问题。*

1. **静态 shape 绑定**：由于 SwinT 的 Window Attention 包含 `if x.shape[0] > window_size:` 等逻辑，Agent 需修改 `mmdet3d/models/backbones/swin.py`，将涉及尺寸判断的地方替换为基于推理输入分辨率（例如 $256 \times 704$）硬编码的静态常量。
2. **Q/DQ 节点映射**：在导出 ONNX 前，Agent 需要遍历 `_QuantizedConv2d` 和 `_QuantizedLinear`，确保 `LearnableFakeQuantize` 能被 `torch.onnx.export` 正确识别为一对标准的 ONNX `QuantizeLinear` 和 `DequantizeLinear` 算子（通常需要设置 `opset_version=13` 或更高）。
3. **验证步骤**：导出 SwinT 独立的 `swin_int8.onnx`，使用 Netron 检查图中是否包含 Q/DQ 节点，并用 `trtexec` 验证是否能成功构建引擎。

#### Phase 2: BEV Pooling TRT Plugin 移植 (参考 BEVDet)

*目标：替换 PyTorch CUDA 算子为 TRT Plugin。*

1. **Plugin 提取与编译**：Agent 需要从 [BEVDet](https://github.com/HuangJunJie2017/BEVDet) 或 mmdet3d 的 deployment 分支中提取 `bev_pool_v2` 的 TensorRT C++ Plugin 源码（包含 `.cpp` 和 `.cu` 文件）。
2. **编写 CMakeList**：构建 CMake 脚本，链接 TensorRT 库，将源码编译为 `libbevpool_trt.so`。
3. **注册 ONNX Symbolic Function**：在 PyTorch 侧，Agent 需为 `LSSTransform` 中的 `bev_pool` 操作编写对应的 `symbolic` 导出函数，使其在 ONNX 图中生成一个 `TRT_Plugin` 类型的节点，并携带正确的 plugin name 和 version。
4. **验证步骤**：使用 Python `ctypes.CDLL("libbevpool_trt.so")` 加载插件后，通过 TensorRT Python API 构建包含 BEV Pooling 的引擎。

#### Phase 3: SpConv 稀疏卷积的 TRT 部署 (参考 spconv)

*目标：将 `SparseEncoder` 部署到 TRT。*可以参考这个仓库： https://github.com/traveller59/spconv

1. **框架升级/对接**：由于标准的 ONNX 不支持稀疏张量，Agent 需要查阅 `spconv` (推荐 v2.x 版本) 的官方 TRT 导出方案。SpConv 提供了一套将稀疏卷积转换为自定义 ONNX 节点的机制。
2. **提取 NVIDIA 插件（备选方案）**：如果原生 SpConv 导出受阻，Agent 需直接提取 NVIDIA `CUDA-BEVFusion` 仓库中的 `lidar-scn.cpp` 和对应的 ONNX 解析器，编译为 `libspconv_trt.so`。
3. **图修改**：确保 `Voxelization` 的输出能够被正确展平并作为 SpConv TRT Plugin 的输入格式（通常是 `features` 和 `indices` 分离）。

#### Phase 4: 自定义量化算子（Log2）的底层对齐（核心难点）

*目标：解决硬件 ALUs 不支持 Log2 INT8 矩阵乘法的问题。*

**背景说明**：你的 `SparseLog2FakeQuantize` 在仿真中非常成功，但 NVIDIA Ada Lovelace (RTX 4060) 的 Tensor Core 仅支持标准的均匀线性映射 INT8 矩阵乘法。如果强制导出 Log2 的 Q/DQ 节点，TRT 会将其回退到 FP16 或报错。

1. **制定硬件妥协策略**：Agent 需要基于 Cutlass 或纯 CUDA 从零编写一个支持对数反量化的 SpConv TRT Plugin。可以参考 https://github.com/zkkli/RepQ-ViT/tree/main/detection/tools/deployment 这个仓库
2. **执行指令**：Agent 需要修改 `quant_ptq_minmax.py` 的导出逻辑，将 Log2 FakeQuant 节点导出为显式的“对数运算+舍入” ONNX 节点，并确保它在流入 TRT 引擎的下一个卷积层前，转换回标准的浮点数据类型（FP16）。