from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

_QUANTIZED_OPS = {
    "ConvInteger",
    "QLinearConv",
    "MatMulInteger",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearConcat",
    "QLinearSigmoid",
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
}
_INFRA_OPS = {
    "Constant",
    "Identity",
    "Cast",
    "Shape",
    "ConstantOfShape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Reshape",
    "Slice",
}
_INTEGER_ACTIVATION_TYPES = {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
_SHAPE_TYPES = {onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.BOOL}
_DEFAULT_HINT = (
    "not a supported operator for this quantization mode",
    "use the matching static / dynamic mode, or leave the node in FP32",
)
_UNQUANTIZED_HINTS = {
    "dynamic": {
        "Resize": (
            "dynamic IntegerOps quantize Conv/MatMul weights only; Resize is out of scope",
            "quantize Conv with static quantization; leave Resize in FP32",
        ),
        "Add": (
            "dynamic quantization does not quantize addition",
            "use static QDQ, or leave addition in FP32",
        ),
        "Mul": (
            "dynamic quantization does not quantize multiplication",
            "use static QDQ, or leave multiplication in FP32",
        ),
        "Sigmoid": (
            "dynamic quantization does not quantize activations",
            "use static QDQ, or leave the activation in FP32",
        ),
        "Concat": (
            "dynamic quantization does not quantize concatenation",
            "use static QDQ, or leave concatenation in FP32",
        ),
        "Relu": (
            "usually not quantized on its own; often fused into adjacent integer ops",
            "fuse Conv+BN+Relu in preprocessing, then apply static quantization",
        ),
        "AveragePool": (
            "out of scope for dynamic quantization; skipped when the kernel depends on input size",
            "use static quantization; fix the shape if the kernel depends on input size",
        ),
        "Slice": (
            "shape-manipulation op, not a quantization target",
            "leave in FP32; do not convert to INT8",
        ),
    },
    "static": {
        "Resize": (
            "Resize falls back to the FP32 path when its input stays FP32",
            "insert Q/DQ on the producer so Resize takes the Direct8Bit path",
        ),
        "Add": (
            "QDQ addition requires a DequantizeLinear input",
            "register Add with QDQOperatorBase and quantize the producer",
        ),
        "Mul": (
            "QDQ multiplication requires a DequantizeLinear input",
            "register Mul with QDQOperatorBase and quantize the producer",
        ),
        "Sigmoid": (
            "QDQ activation requires a DequantizeLinear input",
            "register Sigmoid with QDQOperatorBase and quantize the producer",
        ),
        "Concat": (
            "QDQ concatenation requires a DequantizeLinear input",
            "register Concat with QDQOperatorBase and quantize the producer",
        ),
        "AveragePool": (
            "often skipped when the kernel depends on input size",
            "fix the shape, or leave AveragePool in FP32",
        ),
        "Slice": (
            "shape-manipulation op, not a quantization target",
            "leave in FP32; do not convert to INT8",
        ),
        "Relu": (
            "Relu is removed when its input is quantized; leftover Relu means the input is FP32",
            "fuse Conv+BN+Relu in preprocessing",
        ),
        "BatchNormalization": (
            "unfused BN stays FP32 even under QDQ",
            "fuse Conv+BN with quant_pre_process",
        ),
    },
}


@dataclass(frozen=True)
class UnquantizedLayer:
    name: str
    op_type: str
    reason: str
    mitigation: str


@dataclass(frozen=True)
class QuantizeResult:
    path: Path
    unquantized: tuple[UnquantizedLayer, ...] = ()


def collect_unquantized_layers(
    model_path: str | Path, mode: str = "dynamic"
) -> tuple[UnquantizedLayer, ...]:
    model = onnx.load(str(model_path))
    producers = {output: node for node in model.graph.node for output in node.output}
    elem_types = _tensor_elem_types(model.graph)
    hints = _UNQUANTIZED_HINTS.get(mode, {})
    layers = []
    for node in model.graph.node:
        if _is_quantized_or_infra(node, producers, elem_types):
            continue
        reason, mitigation = hints.get(node.op_type, _DEFAULT_HINT)
        layers.append(
            UnquantizedLayer(node.name or node.output[0], node.op_type, reason, mitigation)
        )
    return tuple(layers)


def format_unquantized_table(layers: tuple[UnquantizedLayer, ...] | list[UnquantizedLayer]) -> str:
    header = ("layer", "op", "reason", "mitigation")
    rows = [header, *[(item.name, item.op_type, item.reason, item.mitigation) for item in layers]]
    if len(rows) == 1:
        rows.append(("-", "-", "none", "-"))
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def _is_quantized_or_infra(
    node: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    elem_types: dict[str, int],
) -> bool:
    if node.op_type in _QUANTIZED_OPS or node.op_type in _INFRA_OPS:
        return True
    if node.op_type.startswith("QLinear"):
        return True
    # QDQ keeps the original op name (Conv/Add); quantization is via DequantizeLinear inputs.
    if _fed_by_dequant(node, producers):
        return True
    # scale/bias helpers inserted by ConvInteger are not original layers.
    if _is_integer_conv_helper(node):
        return True
    # Concat/Slice over INT64 shape tensors (e.g. Resize sizes) are not compute layers.
    if _is_shape_only(node, elem_types):
        return True
    # QOperator Resize / MaxPool keep their op names but pass UINT8 tensors.
    return _node_uses_integer_activation(node, producers, elem_types)


def _is_integer_conv_helper(node: onnx.NodeProto) -> bool:
    # ORT often leaves the scale Mul unnamed and only tags the output
    # (`logitsquant_scaled_output`). node.name alone would leak that helper.
    name = node.name or (node.output[0] if node.output else "")
    return node.op_type in {"Add", "Mul"} and (
        "_quant_scales" in name
        or "_quant_output_scale" in name
        or "_bias_add" in name
        or "quant_scaled" in name
    )


def _fed_by_dequant(node: onnx.NodeProto, producers: dict[str, onnx.NodeProto]) -> bool:
    return any(
        (source := producers.get(name)) is not None and source.op_type == "DequantizeLinear"
        for name in node.input
        if name
    )


def _is_shape_only(node: onnx.NodeProto, elem_types: dict[str, int]) -> bool:
    types = [elem_types.get(name) for name in node.input if name]
    known = [item for item in types if item is not None]
    return bool(known) and all(item in _SHAPE_TYPES for item in known)


def _node_uses_integer_activation(
    node: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    elem_types: dict[str, int],
) -> bool:
    for name in node.input:
        if not name:
            continue
        if elem_types.get(name) in _INTEGER_ACTIVATION_TYPES:
            return True
        source = producers.get(name)
        # DequantizeLinear outputs FP32. Only QLinear* / QuantizeLinear stay in INT8.
        if source is None or source.op_type == "DequantizeLinear":
            continue
        if source.op_type in _QUANTIZED_OPS or source.op_type.startswith("QLinear"):
            return True
    return False


def _tensor_elem_types(graph: onnx.GraphProto) -> dict[str, int]:
    types: dict[str, int] = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        types[value.name] = value.type.tensor_type.elem_type
    for initializer in graph.initializer:
        types[initializer.name] = initializer.data_type
    return types
