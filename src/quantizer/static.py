from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.registry import QDQRegistry
from onnxruntime.quantization.shape_inference import quant_pre_process

from quantizer import QuantizeResult, collect_unquantized_layers, format_unquantized_table

# NPUs want Q/DQ + the original op, not QLinear*. Add ops missing from the ORT QDQ registry.
_QDQ_EXTRA_OPS = ("Add", "Mul", "Sigmoid", "Concat")
DEFAULT_OPSET = 21
DEFAULT_IR_VERSION = 9


class NumpyCalibrationReader(CalibrationDataReader):
    def __init__(self, batches: Sequence[np.ndarray], input_name: str = "image") -> None:
        self._batches = [{input_name: np.ascontiguousarray(batch)} for batch in batches]
        self._index = 0

    def get_next(self) -> dict | None:
        if self._index >= len(self._batches):
            return None
        batch = self._batches[self._index]
        self._index += 1
        return batch


def quantize_static_onnx(
    model_path: str | Path,
    output_path: str | Path,
    calibration: CalibrationDataReader,
) -> QuantizeResult:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = destination.with_name(f"{destination.stem}.preprocessed.onnx")
    # Without Conv+BN+Relu fusion, later Add/Mul stay out of the INT8 region.
    quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
    # per-channel QDQ DequantizeLinear needs axis; pin opset 21 / IR 9.
    _ensure_qdq_opset(prepared, DEFAULT_OPSET)
    try:
        op_types = _register_qdq_elementwise()
        quantize_static(
            str(prepared),
            str(destination),
            calibration,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=op_types,
            extra_options={"ForceQuantizeNoInputCheck": False},
        )
    finally:
        prepared.unlink(missing_ok=True)
    _set_ir_version(destination, DEFAULT_IR_VERSION)
    return QuantizeResult(destination, collect_unquantized_layers(destination, "static"))


def quantize_static_experiment(
    model_path: str | Path,
    output_path: str | Path,
    calib_dir: str | Path,
    num_calib: int | None = None,
) -> QuantizeResult:
    return quantize_static_onnx(
        model_path,
        output_path,
        calibration_reader_from_arrays(model_path, calib_dir, num_calib),
    )


def calibration_reader_from_arrays(
    model_path: str | Path,
    calib_dir: str | Path,
    num_calib: int | None = None,
) -> NumpyCalibrationReader:
    if num_calib is not None and num_calib < 1:
        raise ValueError(f"num_calib must be >= 1: {num_calib}")
    input_name, onnx_shape = _onnx_nchw_spec(model_path)
    paths = sorted(Path(calib_dir).glob("*.npy"))
    if num_calib is not None:
        paths = paths[:num_calib]
    if not paths:
        raise FileNotFoundError(f"no calibration arrays (*.npy) found: {calib_dir}")
    batches = [_load_calib_array(path, input_name, onnx_shape) for path in paths]
    return NumpyCalibrationReader(batches, input_name)


def _register_qdq_elementwise() -> list[str]:
    for op_type in _QDQ_EXTRA_OPS:
        QDQRegistry.setdefault(op_type, QDQOperatorBase)
    return sorted(QDQRegistry)


def _ensure_qdq_opset(model_path: Path, opset: int) -> None:
    model = onnx.load(str(model_path))
    current = next(
        (item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}),
        0,
    )
    if current == opset:
        return
    if current > opset:
        raise ValueError(f"ONNX opset is {current} (expected {opset})")
    onnx.save(version_converter.convert_version(model, opset), str(model_path))


def _set_ir_version(model_path: Path, ir_version: int) -> None:
    model = onnx.load(str(model_path))
    if model.ir_version == ir_version:
        return
    model.ir_version = ir_version
    onnx.save(model, str(model_path))


def _onnx_nchw_spec(model_path: str | Path) -> tuple[str, tuple[int, int, int, int]]:
    graph_input = onnx.load(str(model_path)).graph.input[0]
    dims = [axis.dim_value for axis in graph_input.type.tensor_type.shape.dim]
    if len(dims) != 4:
        raise ValueError(f"ONNX input must be rank-4 NCHW: {dims}")
    batch, channels, height, width = dims
    if channels < 1 or height < 1 or width < 1:
        raise ValueError(f"ONNX input C/H/W must be static and >= 1: {dims}")
    if batch not in {0, 1}:
        raise ValueError(f"ONNX input batch must be 0 or 1: {dims}")
    return graph_input.name, (batch, channels, height, width)


def _load_calib_array(
    path: Path, input_name: str, onnx_shape: tuple[int, int, int, int]
) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(
            f"{path.name} must be float32 (1, C, H, W): "
            f"dtype={array.dtype} shape={tuple(array.shape)}"
        )
    _, channels, height, width = onnx_shape
    if array.shape[1:] != (channels, height, width):
        raise ValueError(
            f"{path.name} shape {tuple(array.shape)} != ONNX input {input_name} {onnx_shape}"
        )
    return np.ascontiguousarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX static quantization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--num-calib", type=int, default=None)
    args = parser.parse_args()
    result = quantize_static_experiment(args.model, args.output, args.calib_dir, args.num_calib)
    print(f"wrote {result.path}")
    print(format_unquantized_table(result.unquantized))


if __name__ == "__main__":
    main()
