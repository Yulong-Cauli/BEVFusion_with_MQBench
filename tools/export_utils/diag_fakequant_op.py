#!/usr/bin/env python3
"""
诊断 MQBench LearnableFakeQuantize 实际调用的 ATen 算子名称。

运行方法:
    python tools/export_utils/diag_fakequant_op.py

输出:
    1. LearnableFakeQuantize.per_channel forward 方法签名
    2. 实际调用的 torch._C._jit 算子名
    3. ONNX 导出时生成的 op_type
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.onnx
import onnx
import inspect


def main():
    print("=" * 60)
    print("Step 1: 检查 MQBench LearnableFakeQuantize 源码位置")
    print("=" * 60)

    try:
        from mqbench.fake_quantize.lsq import LearnableFakeQuantize
        import mqbench
        mqbench_path = os.path.dirname(mqbench.__file__)
        lsq_path = inspect.getsourcefile(LearnableFakeQuantize)
        print(f"MQBench 路径: {mqbench_path}")
        print(f"LSQ 源码路径: {lsq_path}")
    except ImportError as e:
        print(f"❌ 无法导入 MQBench: {e}")
        return

    print("\n" + "=" * 60)
    print("Step 2: 检查 LearnableFakeQuantize.per_channel 方法")
    print("=" * 60)

    # 检查 per_channel 方法
    if hasattr(LearnableFakeQuantize, 'per_channel'):
        method = getattr(LearnableFakeQuantize, 'per_channel')
        print(f"per_channel 方法: {method}")
        try:
            source = inspect.getsource(method)
            print("源码片段:")
            for i, line in enumerate(source.split('\n')[:30], 1):
                print(f"  {i:3}: {line}")
        except Exception as e:
            print(f"无法获取源码: {e}")

    print("\n" + "=" * 60)
    print("Step 3: 检查 forward 方法")
    print("=" * 60)

    try:
        source = inspect.getsource(LearnableFakeQuantize.forward)
        print("LearnableFakeQuantize.forward 源码:")
        for i, line in enumerate(source.split('\n')[:50], 1):
            print(f"  {i:3}: {line}")
    except Exception as e:
        print(f"无��获取源码: {e}")

    print("\n" + "=" * 60)
    print("Step 4: 追踪实际调用的 ATen 算子")
    print("=" * 60)

    # 创建一个简单的测试模块
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fq = LearnableFakeQuantize()

        def forward(self, x):
            return self.fq(x)

    # 初始化
    m = TestModule()
    m.fq.scale = torch.nn.Parameter(torch.tensor(0.1))
    m.fq.zero_point = torch.nn.Parameter(torch.tensor(0))
    m.fq._initialized.fill_(1)
    m.fq._observer_enabled.fill_(0)
    m.fq._fake_quant_enabled.fill_(1)
    m.eval()

    # 用 JIT trace 追踪
    dummy = torch.randn(1, 64, 32, 32)
    with torch.no_grad():
        traced = torch.jit.trace(m, dummy)

    print("JIT Trace 图中的节点:")
    for node in traced.graph.nodes():
        print(f"  {node.kind()}")

    print("\n" + "=" * 60)
    print("Step 5: ONNX 导出测试")
    print("=" * 60)

    # 导出到 ONNX 看实际 op_type
    onnx_path = "/tmp/test_fakequant.onnx"
    try:
        torch.onnx.export(
            m, dummy, onnx_path,
            opset_version=13,
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            enable_onnx_checker=False,
        )

        onnx_model = onnx.load(onnx_path)
        ops = sorted(set(n.op_type for n in onnx_model.graph.node))
        print(f"ONNX 中的所有 op 类型: {ops}")

        # 找 FakeQuant 相关的
        fq_ops = [n for n in onnx_model.graph.node
                  if any(x in n.op_type.lower() for x in ['fake', 'quant', 'learnable', 'affine'])]
        if fq_ops:
            print(f"\nFakeQuant 相关节点:")
            for n in fq_ops:
                print(f"  op_type: {n.op_type}")
                print(f"  name: {n.name}")
                print(f"  inputs: {[i for i in n.input]}")
                print(f"  outputs: {[o for o in n.output]}")
                print()
    except Exception as e:
        print(f"ONNX 导出失败: {e}")

    print("\n" + "=" * 60)
    print("Step 6: 检查 MQBench custom_symbolic_opset 注册")
    print("=" * 60)

    try:
        import mqbench.custom_symbolic_opset as cso
        # 检查这个模块里有什么
        print(f"custom_symbolic_opset 模块内容:")
        for name in dir(cso):
            if not name.startswith('_'):
                print(f"  {name}")
    except ImportError as e:
        print(f"无法导入 mqbench.custom_symbolic_opset: {e}")

    # 检查 torch.onnx 的注册表
    print("\n检查已注册的 custom symbolic ops:")
    try:
        from torch.onnx._internal import registration
        for domain, ops in registration._registry._registry.items():
            for op_name in ops:
                if 'quant' in op_name.lower() or 'fake' in op_name.lower():
                    print(f"  {domain}::{op_name}")
    except ImportError:
        print("torch.onnx._internal.registration 不可用")
    except Exception as e:
        print(f"检查注册表失败: {e}")


if __name__ == "__main__":
    main()
