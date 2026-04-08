"""
Phase 1 部署脚本 - 在服务器上运行

这个脚本会：
1. 找到 MQBench 的 custom_symbolic_opset.py 路径
2. 备份原始文件
3. 替换为新的 symbolic 实现
4. 运行 export_swin.py

使用方法：
    conda activate bevfusion_mqbench
    export LD_LIBRARY_PATH=/media/yellowstone/databig2/gzj/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
    cd /media/yellowstone/data2/CYL/BEVFusion_with_MQBench
    python tools/export_utils/deploy_phase1.py
"""

import os
import sys
import shutil

# 新的 symbolic 实现内容
NEW_SYMBOLIC_CONTENT = '''from torch.onnx import register_custom_op_symbolic

def _learnable_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
    q  = g.op("QuantizeLinear",   x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq

def _fixed_per_tensor_qdq(g, x, scale, zero_point, quant_min, quant_max):
    q  = g.op("QuantizeLinear",   x, scale, zero_point)
    dq = g.op("DequantizeLinear", q, scale, zero_point)
    return dq

def _learnable_per_channel_qdq(g, x, scale, zero_point, axis, quant_min, quant_max, grad_factor):
    q  = g.op("QuantizeLinear",   x, scale, zero_point, axis_i=axis)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=axis)
    return dq

def _fixed_per_channel_qdq(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
    q  = g.op("QuantizeLinear",   x, scale, zero_point, axis_i=ch_axis)
    dq = g.op("DequantizeLinear", q, scale, zero_point, axis_i=ch_axis)
    return dq

register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine',  _learnable_per_tensor_qdq,   13)
register_custom_op_symbolic('::fake_quantize_per_tensor_affine',             _fixed_per_tensor_qdq,        13)
register_custom_op_symbolic('::_fake_quantize_learnable_per_channel_affine', _learnable_per_channel_qdq,  13)
register_custom_op_symbolic('::fake_quantize_per_channel_affine',            _fixed_per_channel_qdq,       13)
'''


def main():
    print("=" * 60)
    print("Phase 1 部署脚本")
    print("=" * 60)

    # Step 1: 找到 MQBench 的 custom_symbolic_opset.py 路径
    print("\nStep 1: 查找 MQBench custom_symbolic_opset.py 路径")
    import mqbench
    mqbench_dir = os.path.dirname(mqbench.__file__)
    symbolic_path = os.path.join(mqbench_dir, 'custom_symbolic_opset.py')
    print(f"路径: {symbolic_path}")

    if not os.path.exists(symbolic_path):
        print(f"❌ 文件不存在: {symbolic_path}")
        sys.exit(1)

    # Step 2: 备份原始文件
    print("\nStep 2: 备份原始文件")
    backup_path = symbolic_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(symbolic_path, backup_path)
        print(f"✓ 已备份到: {backup_path}")
    else:
        print(f"✓ 备份已存在: {backup_path}")

    # Step 3: 替换文件内容
    print("\nStep 3: 替换为新的 symbolic 实现")
    with open(symbolic_path, 'w', encoding='utf-8') as f:
        f.write(NEW_SYMBOLIC_CONTENT)
    print("✓ 文件已替换")

    # Step 4: 验证替换结果
    print("\nStep 4: 验证替换结果")
    print("-" * 40)
    with open(symbolic_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    print("-" * 40)

    # Step 5: 运行 export_swin.py
    print("\nStep 5: 运行 export_swin.py")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "tools/export_utils/export_swin.py"],
        cwd=os.getcwd()
    )

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ Phase 1 部署完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"❌ export_swin.py 返回错误码: {result.returncode}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
