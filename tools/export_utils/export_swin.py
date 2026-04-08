"""
Phase 1: 导出 SwinTransformer 为 swin_int8.onnx

运行: python tools/export_utils/export_swin.py 2>&1 | tee logs/export_swin.log
"""
import sys, os, logging
sys.path.insert(0, os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────────
# Step 1: 先注册 ATen 域的 symbolic（在 import mqbench 之前）
# ─────────────────────────────────────────────────────────────────────────────────
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper

def _learnable_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
    q = g.op("QuantizeLinear", x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq

def _fixed_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max):
    q = g.op("QuantizeLinear", x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq

def _learnable_per_channel_qdq(g, x, scale, zero_point, axis, quant_min, quant_max, grad_factor):
    axis_i = symbolic_helper._get_const(axis, 'i', 'axis')
    q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
    return dq

def _fixed_per_channel_qdq(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
    axis_i = symbolic_helper._get_const(ch_axis, 'i', 'ch_axis')
    q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
    return dq

# 注册 ATen 域
register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _learnable_per_tensor_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_tensor_affine', _fixed_per_tensor_qdq, 13)
register_custom_op_symbolic('::_fake_quantize_learnable_per_channel_affine', _learnable_per_channel_qdq, 13)
register_custom_op_symbolic('::fake_quantize_per_channel_affine', _fixed_per_channel_qdq, 13)
print("[export_swin] ATen 域 Q/DQ symbolic 已注册")

# ─────────────────────────────────────────────────────────────────────────────────
# Step 2: 加载 MQBench（这会触发 custom_symbolic_opset 的加载）
# ─────────────────────────────────────────────────────────────────────────────────
import mqbench.custom_symbolic_opset  # noqa

# ─────────────────────────────────────────────────────────────────────────────────
# Step 3: 为 autograd Function 添加 symbolic（在 mqbench 加载后）
# ─────────────────────────────────────────────────────────────────────────────────
try:
    from mqbench.fake_quantize.lsq import FakeQuantizeLearnablePerchannelAffine

    def _perchannel_symbolic(g, x, scale, zero_point, axis, quant_min, quant_max, grad_factor):
        axis_i = symbolic_helper._get_const(axis, 'i', 'axis')
        q = g.op("QuantizeLinear", x, scale, zero_point, axis_i=axis_i)
        dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis_i)
        return dq

    # 直接覆盖，不管原来有没有
    FakeQuantizeLearnablePerchannelAffine.symbolic = staticmethod(_perchannel_symbolic)
    print("[export_swin] FakeQuantizeLearnablePerchannelAffine.symbolic 已覆盖")
except ImportError as e:
    print(f"[export_swin] 无法导入 FakeQuantizeLearnablePerchannelAffine: {e}")
except Exception as e:
    print(f"[export_swin] 添加 PerChannel symbolic 失败: {e}")

# PerTensor 版本（可能不存在）
try:
    from mqbench.fake_quantize.lsq import FakeQuantizeLearnablePertensorAffine

    def _pertensor_symbolic(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
        q = g.op("QuantizeLinear", x, scale, zero_point)
        dq = g.op("DequantizeLinear", q, scale, zero_point)
        return dq

    if not hasattr(FakeQuantizeLearnablePertensorAffine, 'symbolic'):
        FakeQuantizeLearnablePertensorAffine.symbolic = staticmethod(_pertensor_symbolic)
        print("[export_swin] FakeQuantizeLearnablePertensorAffine.symbolic 已添加")
except ImportError:
    pass  # 正常情况，这个类可能不存在

# ─────────────────────────────────────────────────────────────────────────────────
# Step 4: 正常的导出流程
# ─────────────────────────────────────────────────────────────────────────────────
import torch, onnx
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import get_root_logger, recursive_eval
from mqbench.utils.state import enable_quantization
from tools.quant_ptq_minmax import build_ptq_model

config_path = (
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/"
    "swint_v0p075/convfuser.yaml"
)
configs.load(config_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=config_path)

logger = get_root_logger(log_level=logging.WARNING)

model, _, _ = build_ptq_model(cfg, logger)

ckpt = torch.load("pretrained/ptq_minmax_model.pth", map_location="cpu")
state_dict = ckpt["state_dict"]

model_state = model.state_dict()
for name, param in list(state_dict.items()):
    if name in model_state:
        model_param = model_state[name]
        if param.shape != model_param.shape:
            if 'weight_fake_quant' in name and ('scale' in name or 'zero_point' in name):
                state_dict[name] = param
            elif 'activation_post_process' in name and ('min_val' in name or 'max_val' in name):
                state_dict[name] = param
            elif param.numel() == model_param.numel():
                state_dict[name] = param.view(model_param.shape)
            else:
                del state_dict[name]

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")

enable_quantization(model)
model.eval()

swint = model.encoders.camera.backbone
swint.eval()

dummy = torch.randn(1, 3, 256, 704)

fq_count = sum(1 for m in swint.modules()
               if 'FakeQuant' in type(m).__name__ or 'fake_quant' in type(m).__name__.lower())
print(f"FakeQuant 模块数: {fq_count}")

with torch.no_grad():
    out = swint(dummy)
if isinstance(out, (list, tuple)):
    print(f"前向推理成功，shapes: {[o.shape for o in out]}")
else:
    print(f"前向推理成功，shape: {out.shape}")

try:
    torch.onnx.export(
        swint, dummy, "swin_int8.onnx",
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes=None,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        enable_onnx_checker=False,
    )
except torch.onnx.utils.ONNXCheckerError as e:
    print(f"[注意] ONNX checker 报错（忽略）: {e}")
    if not os.path.exists("swin_int8.onnx"):
        raise

print(f"文件大小: {os.path.getsize('swin_int8.onnx')/1024/1024:.1f} MB")

m = onnx.load("swin_int8.onnx")
qdq = [n for n in m.graph.node
       if n.op_type in ("QuantizeLinear","DequantizeLinear")]
fq  = [n for n in m.graph.node
       if any(x in n.op_type for x in ["FakeQuant","Affine","Learnable"])]
print(f"Q/DQ 节点: {len(qdq)}")
print(f"未转换 FakeQuant: {len(fq)}")
