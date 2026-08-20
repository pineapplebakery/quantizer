# quantizer

Dynamic and static quantization of arbitrary ONNX graphs. This package is
model-free: it does not depend on a host model's training, datasets,
architecture, or preprocessing.

Inputs are an ONNX graph and, for static quantization, already-preprocessed
float32 `.npy` arrays.

## Contract

- Dynamic quantization covers Conv / MatMul weights (IntegerOps). Activations
  stay unquantized.
- Static quantization is QDQ. Activations QUInt8, weights QInt8, per-channel.
  Output is opset 21 / IR 9.
- Static quantization reads `*.npy` in the calibration directory
  (non-recursive, name order). `np.load(..., allow_pickle=False)`. dtype
  float32, shape `(1, C, H, W)`.
- Compare C/H/W. npy N is 1. The ONNX leading input N is 0 (dynamic) or 1.
- Unquantized layers are printed as a table.

JPEG image directories are not accepted. The host must write the npy arrays.
`prepare_calib.sample.py` (repo root) is a host-side writer: class coverage,
leftover `*.npy` cleanup, and zero-padded `0000.npy` names. Replace
`host_samples()` with your eval-preprocessed training split. Ignore/void
label ids are host-defined; pass them to `classes_in_mask`, do not assume
255.

## Install

```bash
pip install -r requirements.txt
```

Put `src/` on `PYTHONPATH`. This repo is not an installable package.

## Commands

From this repository:

```bash
PYTHONPATH=src python -m quantizer.dynamic --model model.onnx --output dyn.onnx

PYTHONPATH=src python -m quantizer.static \
  --model model.onnx \
  --output static.onnx \
  --calib-dir /path/to/npy

python prepare_calib.sample.py --output-dir /path/to/npy --num-classes N
```

From a host that vendors this repo as a submodule, keep the host cwd:

```bash
PYTHONPATH=path/to/quantizer/src python -m quantizer.dynamic \
  --model outputs/model.onnx \
  --output outputs/model.dynamic.onnx

PYTHONPATH=path/to/quantizer/src python -m quantizer.static \
  --model outputs/model.onnx \
  --output outputs/model.static.onnx \
  --calib-dir outputs/calib
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

Tests use tiny generic ONNX graphs and hand-written npy. They do not need a
real-model ONNX, a large calibration set, or a network.
