from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from quantizer import QuantizeResult, collect_unquantized_layers, format_unquantized_table


def quantize_dynamic_onnx(model_path: str | Path, output_path: str | Path) -> QuantizeResult:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # IntegerOps includes Conv by default; name it so weighted Conv is not dropped.
    quantize_dynamic(
        str(model_path),
        str(destination),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul"],
    )
    return QuantizeResult(destination, collect_unquantized_layers(destination, "dynamic"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX dynamic quantization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = quantize_dynamic_onnx(args.model, args.output)
    print(f"wrote {result.path}")
    print(format_unquantized_table(result.unquantized))


if __name__ == "__main__":
    main()
