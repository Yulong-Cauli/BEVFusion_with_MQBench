"""
Phase 4: Fix LiDAR ONNX for libspconv.so compatibility.

Approach: Unfuse ReLU from SparseConvolution activation attribute into standalone Relu nodes,
and remove precision/output_precision extra attributes that NVIDIA's ONNX doesn't have.

NOTE: This is a quick-fix attempt. The fundamental issue may be that our SparseEncoder
uses conv_module (no residual Add) while NVIDIA's uses basicblock (with Add+Relu residual).
If libspconv.so still segfaults after this fix, we need spconv 2.3 approach.

Usage:
    python tools/export_utils/fix_lidar_onnx.py \
        --input lidar_backbone_fp16.onnx \
        --output lidar_backbone_fp16_fixed.onnx
"""
import argparse
import onnx
import onnx.helper as helper
from collections import Counter


def fix_onnx(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    new_nodes = []
    next_tensor_id = 0
    # Find max existing tensor id
    for n in graph.node:
        for o in n.output:
            try:
                tid = int(o)
                if tid >= next_tensor_id:
                    next_tensor_id = tid + 1
            except ValueError:
                pass

    for node in graph.node:
        if node.op_type != "SparseConvolution":
            new_nodes.append(node)
            continue

        # Check activation attribute
        act_value = "None"
        new_attrs = []
        for attr in node.attribute:
            if attr.name == "activation":
                act_value = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            elif attr.name in ("precision", "output_precision"):
                # Remove these extra attributes that NVIDIA's ONNX doesn't have
                continue
            else:
                new_attrs.append(attr)

        if act_value == "ReLU":
            # Unfuse: set activation to None, add standalone Relu node
            new_attrs.append(helper.make_attribute("activation", "None"))

            # Create intermediate tensor id
            orig_output = node.output[0]
            intermediate_id = str(next_tensor_id)
            next_tensor_id += 1

            # Modify SparseConvolution to output to intermediate
            new_conv = helper.make_node(
                "SparseConvolution",
                inputs=list(node.input),
                outputs=[intermediate_id],
                name=node.name,
            )
            for a in new_attrs:
                new_conv.attribute.append(a)
            new_nodes.append(new_conv)

            # Add standalone Relu node
            relu_name = f"relu_{node.name}"
            relu_node = helper.make_node(
                "Relu",
                inputs=[intermediate_id],
                outputs=[orig_output],
                name=relu_name,
            )
            new_nodes.append(relu_node)
        else:
            # No ReLU fusion, just remove precision attrs
            new_attrs.append(helper.make_attribute("activation", act_value))
            new_conv = helper.make_node(
                "SparseConvolution",
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
            )
            for a in new_attrs:
                new_conv.attribute.append(a)
            new_nodes.append(new_conv)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.save_model(model, output_path)

    # Summary
    op_counts = Counter(n.op_type for n in graph.node)
    print(f"Fixed ONNX saved: {output_path}")
    print(f"Node types: {dict(op_counts)}")
    print(f"Total nodes: {len(graph.node)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    fix_onnx(args.input, args.output)


if __name__ == "__main__":
    main()
