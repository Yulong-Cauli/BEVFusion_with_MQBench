#!/usr/bin/env python3
"""
MQBench FakeQuant → ONNX Q/DQ 转换。

问题诊断：
1. MQBench 0.0.6 的 custom_symbolic_opset 注册了 aten::_fake_quantize_learnable_* 的 symbolic
2. 但这些 symbolic 输出的是 LearnablePerTensorAffine / FakeQuantizeLearnablePerchannelAffine
3. 我们无法覆�� aten 域的注册（domain already used）

解决方案：
在 MQBench 加载之前，先注册我们的 symbolic，这样 MQBench 就无法覆盖。

用法：在任何 import mqbench 之前 import 此模块！
"""
import sys, os
import torch
import torch.onnx
from torch.onnx import register_custom_op_symbolic
import torch._C._onnx as _C_onnx


# ─────────────────────────────────────────────────────────────────────────────────
# Q/DQ symbolic 函数
# ─────────────────────────────────────────────────────────────────────────────────

def _qdq_per_channel(g, x, scale, zero_point, axis, quant_min=-128, quant_max=127, grad_factor=1.0):
    """Per-Channel Q/DQ"""
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis)
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


def _qdq_per_tensor(g, x, scale, zero_point, quant_min=-128, quant_max=127, grad_factor=1.0):
    """Per-Tensor Q/DQ"""
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", x, scale, zero_point)
    return g.op("DequantizeLinear", quantized, scale, zero_point)


# ─────────────────────────────────────────────────────────────────────────────────
# 关键！在 MQBench 加载之前注册
# ─────────────────────────────────────────────────────────────────────────────────

# 先注册 aten 域的算子（在 MQBench 之前）
_ops_to_register = [
    ('aten::_fake_quantize_learnable_per_tensor_affine', _qdq_per_tensor),
    ('aten::_fake_quantize_learnable_per_channel_affine', _qdq_per_channel),
]

_registered = []
for _op_name, _handler in _ops_to_register:
    try:
        register_custom_op_symbolic(_op_name, _handler, 13)
        _registered.append(_op_name)
        print(f"[mqbench_onnx_symbolic] ✓ 预注册 {_op_name} → Q/DQ")
    except Exception as e:
        print(f"[mqbench_onnx_symbolic] ✗ 预注册 {_op_name} 失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────────
# 现在才加载 MQBench
# ─────────────────────────────────────────────────────────────────────────────────

import mqbench.custom_symbolic_opset  # noqa
import mqbench.fake_quantize  # noqa
print(f"[mqbench_onnx_symbolic] MQBench 版本: {mqbench.__version__ if hasattr(mqbench, '__version__') else 'unknown'}")


# ─────────────────────────────────────────────────────────────────────────────────
# 再次尝试注册（以防 MQBench 覆盖了我们的注册）
# ─────────────────────────────────────────────────────────────────────────────────

# 尝试使用 torch.onnx.symbolic_helper 的内部注册机制
try:
    from torch.onnx._internal import registration
    # 检查当前注册
    print("[mqbench_onnx_symbolic] 检查已注册的 symbolic:")
    for domain, ops in registration._registry._registry.items():
        for op_name in ops:
            if 'fake_quant' in op_name.lower():
                print(f"  {domain}::{op_name}")
except ImportError:
    print("[mqbench_onnx_symbolic] torch.onnx._internal.registration 不可用")
except Exception as e:
    print(f"[mqbench_onnx_symbolic] 检查注册表失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────────
# Log2 量化支持
# ─────────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.getcwd())
try:
    from tools.quant_ptq_minmax import SparseLog2FakeQuantize

    class _Log2QuantExportFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, log2_base, per_channel_flag):
            orig_dtype = x.dtype
            x_f = x.float()
            eps = 1e-6
            zero_mask = x_f.abs() < eps
            sign = x_f.sign()
            base = log2_base.to(x_f.device)
            if per_channel_flag and base.ndim == 1:
                base = base.unsqueeze(0)
            log2_x = torch.log2(x_f.abs().clamp(min=1e-30)) - base
            q_int = torch.round(log2_x).clamp(-127, 127)
            x_dq = sign * torch.pow(2.0, q_int + base)
            x_dq = torch.where(zero_mask, torch.zeros_like(x_f), x_dq)
            return x_dq.to(orig_dtype)

        @staticmethod
        def symbolic(g, x, log2_base, per_channel_flag):
            base_list = log2_base.detach().cpu().tolist()
            if not isinstance(base_list, list):
                base_list = [base_list]
            return g.op("custom::SparseLog2Quant", x, log2_base_f=base_list,
                        per_channel_i=int(per_channel_flag), plugin_version_s="1")

    _orig_log2_forward = SparseLog2FakeQuantize.forward

    def _patched_log2_forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return _Log2QuantExportFunc.apply(x, self.log2_base, int(self.per_channel))
        return _orig_log2_forward(self, x)

    SparseLog2FakeQuantize.forward = _patched_log2_forward
    print("[mqbench_onnx_symbolic] SparseLog2FakeQuantize → custom::SparseLog2Quant")
except ImportError:
    print("[mqbench_onnx_symbolic] SparseLog2FakeQuantize 未找到，跳过 Log2 支持")

print(f"[mqbench_onnx_symbolic] 注册完成（预注册 {len(_registered)} 个）")
