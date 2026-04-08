#!/usr/bin/env python3
"""
Phase 1: SwinTransformer ONNX 导出完整流程

【使用方法】
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

python tools/export_utils/phase1_swin_export.py 2>&1 | tee logs/phase1_swin_export.log
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, ".")


def get_static_pad_values():
    """Step 1.1: 获取静态 pad 值"""
    print("\n" + "=" * 80)
    print("Step 1.1: 获取 SwinTransformer 静态 pad 值")
    print("=" * 80)

    from torchpack.utils.config import configs
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from mmdet3d.models import build_model

    # 加载配置
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(config_file, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_file)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).eval()

    # 收集 pad 值
    pad_values = []

    from mmdet.models.utils.transformer import AdaptivePadding
    _orig = AdaptivePadding.forward

    def _patched(self, x):
        input_h, input_w = x.shape[-2:]
        kernel_h = self.kernel_size[0] + (self.kernel_size[0]-1)*(self.dilation[0]-1)
        kernel_w = self.kernel_size[1] + (self.kernel_size[1]-1)*(self.dilation[1]-1)
        stride_h, stride_w = self.stride
        out_h = (input_h + stride_h - 1) // stride_h
        out_w = (input_w + stride_w - 1) // stride_w
        pad_h = max((out_h-1)*stride_h + kernel_h - input_h, 0)
        pad_w = max((out_w-1)*stride_w + kernel_w - input_w, 0)
        pad_values.append((input_h, input_w, pad_h, pad_w))
        print(f"AdaptivePadding | input=({input_h},{input_w}) | pad=({pad_h},{pad_w})")
        return _orig(self, x)

    AdaptivePadding.forward = _patched

    swint = model.encoders.camera.backbone
    dummy = torch.randn(1, 3, 256, 704)

    print("\n开始前向传播...")
    with torch.no_grad():
        _ = swint(dummy)

    print(f"\n✅ 共收集 {len(pad_values)} 个 AdaptivePadding 的 pad 值")
    return pad_values


def verify_swin_static():
    """Step 1.2-1.3: 验证 SwinTransformer 静态化（无 If 节点）"""
    print("\n" + "=" * 80)
    print("Step 1.2-1.3: 验证 SwinTransformer 静态化")
    print("=" * 80)

    from torchpack.utils.config import configs
    from mmcv import Config
    from mmdet3d.utils import recursive_eval
    from mmdet3d.models import build_model
    import onnx

    # 加载配置
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(config_file, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_file)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).eval()

    swint = model.encoders.camera.backbone
    dummy = torch.randn(1, 3, 256, 704)

    # 导出 ONNX 进行验证
    onnx_path = "/tmp/swin_static_check.onnx"
    print(f"\n导出 ONNX 到 {onnx_path} 进行静态化验证...")

    torch.onnx.export(
        swint,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features"],
    )

    # 检查 If 节点
    model_onnx = onnx.load(onnx_path)
    if_nodes = [n for n in model_onnx.graph.node if n.op_type == "If"]
    all_ops = sorted(set(n.op_type for n in model_onnx.graph.node))

    print(f"\nIf 节点数: {len(if_nodes)} (期望 = 0)")
    print(f"全部 op 类型: {all_ops}")

    if len(if_nodes) > 0:
        print(f"\n❌ 仍有 {len(if_nodes)} 个动态分支，需要继续修改 swin.py")
        return False

    print("\n✅ 静态化验证通过！可以继续导出量化 ONNX")
    return True


# =============================================================================
# MQBench 量化工具（从 quant_ptq_minmax.py 复制必要的部分）
# =============================================================================

from mqbench.observer import EMAMinMaxObserver, MinMaxObserver
from mqbench.scheme import QuantizeScheme
from mqbench.fake_quantize import LearnableFakeQuantize


def _create_tensorrt_fakeq_pair(act_observer_cls=None):
    """创建一对 (weight_fq, act_fq)，匹配 MQBench TensorRT INT8 配置。"""
    if act_observer_cls is None:
        act_observer_cls = EMAMinMaxObserver
    w_params = QuantizeScheme(
        symmetry=True, per_channel=True, pot_scale=False, bit=8
    ).to_observer_params()
    w_params['quant_min'] = -127
    a_params = QuantizeScheme(
        symmetry=True, per_channel=False, pot_scale=False, bit=8
    ).to_observer_params()
    a_params['quant_min'] = -127
    weight_fq = LearnableFakeQuantize(observer=MinMaxObserver, **w_params)
    act_fq = LearnableFakeQuantize(observer=act_observer_cls, **a_params)
    return weight_fq, act_fq


class _QuantizedConv2d(nn.Module):
    """Conv2d + MQBench FakeQuantize（适用于无法 fx 追踪的模块）。"""

    def __init__(self, original, act_observer_cls=None):
        super().__init__()
        self.conv = original
        self.weight_fake_quant, self.act_fake_quant = _create_tensorrt_fakeq_pair(act_observer_cls)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = self.act_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        return F.conv2d(
            x, weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


def _set_nested_attr(obj, key: str, value):
    """设置嵌套属性。"""
    parts = key.rsplit(".", 1)
    if len(parts) == 1:
        setattr(obj, parts[0], value)
    else:
        parent_key, attr_name = parts
        parent = obj
        for part in parent_key.split("."):
            parent = getattr(parent, part)
        setattr(parent, attr_name, value)


def manual_quantize_swin(module, module_name="swin", act_observer_cls=None):
    """对 SwinTransformer 手动插入 FakeQuantize 节点。

    与 quant_ptq_minmax.py 中的 _QuantizedConv2d 逻辑完全一致，
    确保 checkpoint 加载后结构匹配。
    """
    replacements = []
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, _QuantizedConv2d):
            replacements.append((name, _QuantizedConv2d(child, act_observer_cls)))

    for name, replacement in replacements:
        _set_nested_attr(module, name, replacement)

    print(f"  ↪ 手动量化 SwinTransformer: 替换 {len(replacements)} 个 Conv2d")
    return module


def export_swin_onnx(output_path="swin_int8.onnx"):
    """Step 1.4: 导出 SwinTransformer Q/DQ ONNX 模型"""
    print("\n" + "=" * 80)
    print("Step 1.4: 导出 SwinTransformer Q/DQ ONNX 模型")
    print("=" * 80)

    # 必须先导入，注册 ONNX Symbolic
    import tools.export_utils.mqbench_onnx_symbolic

    from torchpack.utils.config import configs
    from mmcv import Config
    from mmdet3d.utils import recursive_eval

    # 加载配置
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(config_file, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_file)

    # 使用 build_model 构建模型（与 quant_ptq_minmax.py 相同）
    from mmdet3d.models import build_model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).eval()

    # 提取 SwinTransformer 并量化（添加 FakeQuant 包装）
    swint = model.encoders.camera.backbone
    print("\n量化 SwinTransformer backbone...")
    manual_quantize_swin(swint, "camera/backbone")

    # 加载量化权重
    ckpt_path = "pretrained/ptq_minmax_model.pth"
    print(f"加载量化模型: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # 过滤出 camera.backbone 相关的权重（去除前缀，并修正键名）
    # checkpoint 包含两种键名格式：
    #   - patch_embed.projection.conv.weight  (Conv2dWrapper)
    #   - stages.*.attn.w_msa.qkv.linear.weight (Linear 包装)
    # 模型期望：.weight（原始层名）
    # 映射规则：.conv.weight → .weight, .linear.weight → .weight
    backbone_keys = {}
    for k, v in state_dict.items():
        if k.startswith('encoders.camera.backbone.'):
            new_key = k.replace('encoders.camera.backbone.', '')
            # 修正键名
            new_key = new_key.replace('.conv.', '.')
            new_key = new_key.replace('.linear.', '.')
            backbone_keys[new_key] = v

    print(f"Checkpoint 中 camera.backbone 权重: {len(backbone_keys)} 个")

    print(f"Checkpoint 中 camera.backbone 权重: {len(backbone_keys)} 个")

    # 加载权重（strict=False 允许部分匹配）
    missing, unexpected = swint.load_state_dict(backbone_keys, strict=False)
    if missing:
        print(f"⚠️ 缺少键: {missing[:5]}...")
    if unexpected:
        print(f"⚠️ 未知键: {unexpected[:5]}...")

    print(f"✅ 已加载量化权重")

    dummy_input = torch.randn(1, 3, 256, 704)

    print(f"\n导出 ONNX 模型到: {output_path}")
    print("这可能需要几分钟...")

    # 导出 ONNX（使用 opset 13）
    torch.onnx.export(
        swint,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        enable_onnx_checker=False,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes=None,
    )

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ ONNX 导出成功: {output_path} ({file_size_mb:.1f} MB)")
    return True


def verify_onnx(onnx_path):
    """验证 ONNX 模型"""
    print("\n" + "=" * 80)
    print("验证 ONNX 模型")
    print("=" * 80)

    try:
        import onnx

        # 加载并检查 ONNX
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✅ ONNX 模型结构有效")

        # 检查节点类型
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        print("\nONNX 节点统计:")
        for op_type, count in sorted(node_types.items()):
            print(f"  {op_type}: {count}")

        # 检查 Q/DQ 节点
        q_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
        dq_nodes = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]
        print(f"\nQ/DQ 节点数: QuantizeLinear={len(q_nodes)}, DequantizeLinear={len(dq_nodes)}")

        # 检查 FakeQuant 节点（应该是 0）
        fq_nodes = [n for n in model.graph.node if "FakeQuant" in n.op_type or "Learnable" in n.op_type]
        print(f"FakeQuant 相关节点数: {len(fq_nodes)}")

        return True

    except Exception as e:
        print(f"❌ ONNX 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SwinTransformer ONNX 导出")
    parser.add_argument("--skip-pad-detection", action="store_true",
                        help="跳过 pad 值获取（Step 1.1）")
    parser.add_argument("--skip-static-check", action="store_true",
                        help="跳过静态化验证（Step 1.2-1.3）")
    parser.add_argument("--output", default="swin_int8.onnx",
                        help="输出 ONNX 文件名")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 ONNX 验证")

    args = parser.parse_args()

    print("🚀 开始 Phase 1: SwinTransformer ONNX 导出")
    print(f"工作目录: {os.getcwd()}")

    # Step 1.1: 获取静态 pad 值
    if not args.skip_pad_detection:
        pad_values = get_static_pad_values()
        print("\n📝 记录以上 pad 值，用于后续静态化 swin.py（Phase 1.2）")

    # Step 1.2-1.3: 验证静态化
    if not args.skip_static_check:
        static_ok = verify_swin_static()
        if not static_ok:
            print("\n⚠️ 静态化验证失败，尝试继续导出...")

    # Step 1.4: 导出 ONNX
    success = export_swin_onnx(args.output)
    if not success:
        print("\n❌ ONNX 导出失败")
        return 1

    # 验证 ONNX
    if not args.no_verify:
        verify_onnx(args.output)

    print("\n" + "=" * 80)
    print("🎉 Phase 1 完成！")
    print("=" * 80)
    print(f"✅ SwinTransformer ONNX 模型: {args.output}")
    print("\n后续步骤:")
    print("1. 使用 build_engine.py 构建 TensorRT 引擎")
    print("2. 继续 Phase 2: 导出其他模块")
    return 0


if __name__ == "__main__":
    exit(main())
