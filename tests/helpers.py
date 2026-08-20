from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

DEFAULT_OPSET = 21
DEFAULT_IR_VERSION = 9


def _save(graph, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", DEFAULT_OPSET)])
    model.ir_version = DEFAULT_IR_VERSION
    save(model, str(path))
    return path


def _float_weight(name: str, shape: tuple[int, ...], rng: np.random.Generator):
    return numpy_helper.from_array(rng.standard_normal(shape).astype(np.float32), name)


def _bias(name: str, channels: int):
    return numpy_helper.from_array(np.zeros((channels,), dtype=np.float32), name)


def save_conv_onnx(
    path: Path, *, height: int = 8, width: int = 12, batch: int | str = 1
) -> Path:
    rng = np.random.default_rng(0)
    conv = helper.make_node("Conv", ["image", "W", "B"], ["logits"], kernel_shape=[1, 1])
    graph = helper.make_graph(
        [conv],
        "conv",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [batch, 3, height, width])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, 2, height, width])],
        [_float_weight("W", (2, 3, 1, 1), rng), _bias("B", 2)],
    )
    return _save(graph, path)


def save_elementwise_onnx(path: Path, *, height: int = 8, width: int = 12) -> Path:
    rng = np.random.default_rng(0)
    nodes = [
        helper.make_node("Conv", ["image", "W1", "B1"], ["stem"], kernel_shape=[1, 1]),
        helper.make_node("Conv", ["stem", "W2", "B2"], ["refine"], kernel_shape=[1, 1]),
        helper.make_node("Add", ["stem", "refine"], ["residual"]),
        helper.make_node("Conv", ["stem", "W3", "B3"], ["attn"], kernel_shape=[1, 1]),
        helper.make_node("Sigmoid", ["attn"], ["gate"]),
        helper.make_node("Mul", ["residual", "gate"], ["gated"]),
        helper.make_node("Concat", ["gated", "residual"], ["joined"], axis=1),
        helper.make_node("Conv", ["joined", "W4", "B4"], ["logits"], kernel_shape=[1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "elementwise",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, height, width])],
        [
            _float_weight("W1", (2, 3, 1, 1), rng),
            _bias("B1", 2),
            _float_weight("W2", (2, 2, 1, 1), rng),
            _bias("B2", 2),
            _float_weight("W3", (2, 2, 1, 1), rng),
            _bias("B3", 2),
            _float_weight("W4", (2, 4, 1, 1), rng),
            _bias("B4", 2),
        ],
    )
    return _save(graph, path)


def save_resize_onnx(path: Path, *, height: int = 8, width: int = 12) -> Path:
    rng = np.random.default_rng(0)
    conv = helper.make_node(
        "Conv",
        ["image", "W", "B"],
        ["down"],
        kernel_shape=[1, 1],
        strides=[2, 2],
    )
    resize = helper.make_node(
        "Resize",
        ["down", "roi", "scales", "sizes"],
        ["logits"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph(
        [conv, resize],
        "resize",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, height, width])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, height, width])],
        [
            _float_weight("W", (2, 3, 1, 1), rng),
            _bias("B", 2),
            numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
            numpy_helper.from_array(np.array([], dtype=np.float32), "scales"),
            numpy_helper.from_array(
                np.array([1, 2, height, width], dtype=np.int64), "sizes"
            ),
        ],
    )
    return _save(graph, path)
