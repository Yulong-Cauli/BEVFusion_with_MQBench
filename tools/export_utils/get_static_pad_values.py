#!/usr/bin/env python3
"""
Phase 1: SwinTransformer 静态化与 Q/DQ ONNX 导出完整流程

【执行流程】
1. 自动获取静态 pad 值（Step 1.1）
2. 导出 SwinTransformer ONNX 模型（Step 1.3）
3. 验证 ONNX 模型结构

【使用方法】
conda activate bevfusion_mqbench
export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench

# 完整流程：
python tools/export_utils/phase1_swin_export.py

# 仅导出（跳过 pad 值获取）：
python tools/export_utils/phase1_swin_export.py --skip-pad-detection
"""

import argparse
import os
import sys
import torch
import torch.onnx
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

# 添加项目路径
sys.path.insert(0, ".")

# 必须最先导入：注册 ONNX Symbolic
import tools.export_utils.mqbench_onnx_symbolic


def get_static_pad_values():
    """Step 1.1: 获取静态 pad 值"""
    print("\n" + "=" * 80)
    print("Step 1.1: 获取 SwinTransformer 静态 pad 值")
    print("=" * 80)

    # 使用 torchpack 加载配置（与 quant_ptq_minmax.py 一致）
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(config_file, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_file)

    # 关键：必须设置这些，否则 build_model 会报错
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).eval()

    # 收集 pad 值
    pad_values = []

    # SwinTransformer 在 mmdet 包中
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


def export_swin_onnx(output_path="swin_int8.onnx"):
    """Step 1.3: 导出 SwinTransformer ONNX 模型"""
    print("\n" + "=" * 80)
    print("Step 1.3: 导出 SwinTransformer Q/DQ ONNX 模型")
    print("=" * 80)

    # 加载量化模型
    print("加载量化模型...")
    # 使用 torchpack 加载配置（与 quant_ptq_minmax.py 一致）
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(config_file, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_file)

    # 关键：必须设置这些，否则 build_model 会报错
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")).eval()

    # 加载量化权重
    ckpt_path = "pretrained/ptq_minmax_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ 量化权重不存在: {ckpt_path}")
        print("请先运行量化流程：python tools/quant_ptq_minmax.py ...")
        return False

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 已加载量化权重: {ckpt_path}")

    # 提取 SwinTransformer
    swint = model.encoders.camera.backbone
    dummy_input = torch.randn(1, 3, 256, 704)

    print(f"导出 ONNX 模型到: {output_path}")
    print("这可能需要几分钟...")

    # 导出 ONNX
    torch.onnx.export(
        swint,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        keep_initializers_as_inputs=False,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes=None,  # 静态输入
        verbose=False,
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
        import onnxruntime as ort

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

        # 尝试 ONNX Runtime 推理
        print("\n尝试 ONNX Runtime 推理...")
        sess = ort.InferenceSession(onnx_path)
        dummy = torch.randn(1, 3, 256, 704).numpy()
        outputs = sess.run(None, {"images": dummy})
        print(f"✅ ONNX Runtime 推理成功，输出形状: {outputs[0].shape}")

        return True

    except ImportError:
        print("⚠️  未安装 onnx 或 onnxruntime，跳过验证")
        return True
    except Exception as e:
        print(f"❌ ONNX 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SwinTransformer 静态化与 ONNX 导出")
    parser.add_argument("--skip-pad-detection", action="store_true",
                        help="跳过 pad 值获取（Step 1.1）")
    parser.add_argument("--output", default="swin_int8.onnx",
                        help="输出 ONNX 文件名")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 ONNX 验证")

    args = parser.parse_args()

    print("🚀 开始 Phase 1: SwinTransformer 静态化与 Q/DQ ONNX 导出")
    print(f"工作目录: {os.getcwd()}")

    # Step 1.1: 获取静态 pad 值
    if not args.skip_pad_detection:
        pad_values = get_static_pad_values()
        print("\n📝 记录以上 pad 值，用于后续静态化 swin.py（Phase 1.2）")

    # Step 1.3: 导出 ONNX
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
    print("1. 如果需要，基于 Step 1.1 的 pad 值静态化 swin.py")
    print("2. 使用 build_engine.py 构建 TensorRT 引擎")
    print("3. 继续 Phase 2: 导出其他模块")
    return 0


if __name__ == "__main__":
    exit(main())