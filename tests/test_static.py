import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from quantizer import QuantizeResult
from quantizer.static import (
    NumpyCalibrationReader,
    calibration_reader_from_arrays,
    quantize_static_experiment,
    quantize_static_onnx,
)
from tests.helpers import save_conv_onnx, save_elementwise_onnx, save_resize_onnx


def _zeros(height: int = 8, width: int = 12) -> np.ndarray:
    return np.zeros((1, 3, height, width), dtype=np.float32)


def _write_npy(directory: Path, *batches: np.ndarray) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches):
        np.save(directory / f"{index:04d}.npy", batch)
    return directory


def _has_quantized_conv(path: Path) -> bool:
    return "Conv" in _qdq_wrapped_ops(path)


def _qdq_wrapped_ops(path: Path) -> set[str]:
    model = onnx.load(str(path))
    producers = {output: node for node in model.graph.node for output in node.output}
    wrapped = set()
    for node in model.graph.node:
        if any(
            (source := producers.get(name)) is not None and source.op_type == "DequantizeLinear"
            for name in node.input
            if name
        ):
            wrapped.add(node.op_type)
    return wrapped


def test_static_quantize_entrypoint_passes_paths(monkeypatch, tmp_path: Path) -> None:
    from quantizer import static

    received = []
    monkeypatch.setattr(
        static,
        "quantize_static_experiment",
        lambda model, output, calib_dir, num_calib=None: received.append(
            (model, output, calib_dir, num_calib)
        )
        or QuantizeResult(tmp_path / "q.onnx"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "static.py",
            "--model",
            "model.onnx",
            "--output",
            "static.onnx",
            "--calib-dir",
            "arrays",
        ],
    )

    static.main()

    assert received == [("model.onnx", "static.onnx", "arrays", None)]


def test_static_quantization_quantizes_conv(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx")
    destination = tmp_path / "static.onnx"
    reader = NumpyCalibrationReader([_zeros() for _ in range(2)])

    result = quantize_static_onnx(source, destination, reader)

    assert result.path == destination
    assert destination.is_file()
    assert _has_quantized_conv(destination)
    assert all(layer.op_type != "Conv" for layer in result.unquantized)
    model = onnx.load(str(destination))
    imported = {item.domain: item.version for item in model.opset_import}
    assert imported.get("") == 21
    assert model.ir_version == 9


def test_static_experiment_quantizes_conv_from_calib_arrays(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx")
    calib = _write_npy(tmp_path / "calib", _zeros())

    result = quantize_static_experiment(source, tmp_path / "static.onnx", calib, num_calib=1)

    assert _has_quantized_conv(result.path)


def test_calibration_reader_rejects_missing_npy(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx")
    tmp_path.joinpath("calib.png").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=r"\*\.npy"):
        calibration_reader_from_arrays(source, tmp_path)


def test_calibration_reader_rejects_shape_mismatch(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx")
    calib = _write_npy(tmp_path / "calib", _zeros(8, 8))

    with pytest.raises(ValueError, match=r"0000\.npy.*\(1, 3, 8, 8\).*image.*\(1, 3, 8, 12\)"):
        calibration_reader_from_arrays(source, calib)


def test_calibration_reader_accepts_dynamic_batch(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx", batch="N")
    calib = _write_npy(tmp_path / "calib", _zeros())

    reader = calibration_reader_from_arrays(source, calib)

    batch = reader.get_next()
    assert batch is not None
    assert batch["image"].shape == (1, 3, 8, 12)


def test_calibration_reader_ignores_nested_npy(tmp_path: Path) -> None:
    source = save_conv_onnx(tmp_path / "fp32.onnx")
    calib = _write_npy(tmp_path / "calib", _zeros())
    nested = tmp_path / "calib" / "nested"
    nested.mkdir()
    np.save(nested / "0001.npy", _zeros())

    reader = calibration_reader_from_arrays(source, calib)

    assert reader.get_next() is not None
    assert reader.get_next() is None


def test_static_quantization_quantizes_add_mul_sigmoid_concat(tmp_path: Path) -> None:
    source = save_elementwise_onnx(tmp_path / "fp32.onnx")
    reader = NumpyCalibrationReader([_zeros() for _ in range(2)])

    result = quantize_static_onnx(source, tmp_path / "static.onnx", reader)

    types = {node.op_type for node in onnx.load(str(result.path)).graph.node}
    wrapped = _qdq_wrapped_ops(result.path)
    assert "QuantizeLinear" in types
    assert "DequantizeLinear" in types
    assert not {op for op in types if op.startswith("QLinear")}
    assert {"Conv", "Add", "Mul", "Sigmoid", "Concat"} <= wrapped
    leftover = {layer.op_type for layer in result.unquantized}
    assert leftover.isdisjoint({"Add", "Mul", "Sigmoid", "Concat", "Conv"})


def test_static_quantization_quantizes_resize(tmp_path: Path) -> None:
    source = save_resize_onnx(tmp_path / "fp32.onnx")
    reader = NumpyCalibrationReader([_zeros() for _ in range(2)])

    result = quantize_static_onnx(source, tmp_path / "static.onnx", reader)

    model = onnx.load(str(result.path))
    producers = {output: node for node in model.graph.node for output in node.output}
    resize_nodes = [node for node in model.graph.node if node.op_type == "Resize"]

    assert resize_nodes
    assert all(
        (
            (source := producers.get(node.input[0])) is not None
            and source.op_type == "DequantizeLinear"
        )
        or node.input[0].endswith("_quantized")
        for node in resize_nodes
    )
    leftover = {layer.op_type for layer in result.unquantized}
    assert "Resize" not in leftover
    assert "Concat" not in leftover
    assert "Slice" not in leftover

    session = ort.InferenceSession(str(result.path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"image": _zeros()})[0]
    assert output.shape == (1, 2, 8, 12)
