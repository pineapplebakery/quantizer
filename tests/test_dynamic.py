import sys
from pathlib import Path

import onnx

from quantizer import QuantizeResult, format_unquantized_table
from quantizer.dynamic import quantize_dynamic_onnx
from tests.helpers import save_conv_onnx, save_resize_onnx


def _op_types(path: Path) -> set[str]:
    return {node.op_type for node in onnx.load(str(path)).graph.node}


def test_dynamic_quantize_entrypoint_passes_paths(monkeypatch, tmp_path: Path) -> None:
    from quantizer import dynamic

    received = []
    monkeypatch.setattr(
        dynamic,
        "quantize_dynamic_onnx",
        lambda model, output: received.append((model, output))
        or QuantizeResult(tmp_path / "q.onnx"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dynamic.py", "--model", "model.onnx", "--output", "dyn.onnx"],
    )

    dynamic.main()

    assert received == [("model.onnx", "dyn.onnx")]


def test_dynamic_quantization_quantizes_conv(tmp_path: Path) -> None:
    destination = tmp_path / "dynamic.onnx"

    result = quantize_dynamic_onnx(save_conv_onnx(tmp_path / "fp32.onnx"), destination)

    assert result.path == destination
    assert destination.is_file()
    assert "ConvInteger" in _op_types(destination)
    assert all("_quant_scales" not in layer.name for layer in result.unquantized)
    assert all(layer.op_type != "Mul" for layer in result.unquantized)


def test_dynamic_quantization_reports_unquantized_resize(tmp_path: Path) -> None:
    source = save_resize_onnx(tmp_path / "fp32.onnx")

    result = quantize_dynamic_onnx(source, tmp_path / "dynamic.onnx")

    by_op = {layer.op_type: layer for layer in result.unquantized}
    assert "Resize" in by_op
    assert "Conv" not in by_op
    assert by_op["Resize"].reason
    assert by_op["Resize"].mitigation
    table = format_unquantized_table(result.unquantized)
    assert "Resize" in table
    assert "reason" in table
    assert "mitigation" in table
