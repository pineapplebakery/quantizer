# Agent instructions

This file holds **invariants that always apply**. Operating procedures live in `README.md`. Review it periodically; add a line here only after the same failure repeats.

## Conversation and done

- Be concise, direct, and frank. Separate verified facts from guesses.
- Pin a one-sentence done condition before starting. Ask only when a choice is ambiguous, destructive, or has external effects.
- For every change, ask whether dropping it would still prove done. Do not add abstractions, dependencies, or refactors that the proof does not need.
- Before claiming done, verify with observable results (tests, command output).
- Multi-step work: save and update a Markdown plan under `docs/plans/`. Filename is `YYYY-MM-DD-<slug>.md` (creation date first). Write what to do, in what order, what is done, and what remains, at a granularity a third party can resume without the chat.

## This repository

Dynamic and static quantization of arbitrary ONNX graphs. This package is model-free: it must not depend on a host model's training, datasets, architecture, or preprocessing.

- Inputs are an ONNX graph and, for static quantization, already-preprocessed float32 `.npy` arrays.

## Change principles

- Pick the simplest implementation that meets the current requirement. Do not abstract for hypothetical future compatibility.
- Keep diffs small and focused. Prefer editing existing files. Do not leave unused new files, dead code, or shims.
- Check mature existing libraries and current dependencies first. Get approval before adding a heavy dependency.
- If the same failure repeats, add it here as an **invariant**, not as a procedure.
- New comments explain why the code is needed. Do not restate the obvious.

## Repository boundaries

| Area | Rule |
|---|---|
| Project code | `src/` |
| Tests | `tests/`. Verify with tiny generic ONNX graphs and hand-written npy |
| Dependencies | `uv`. `pyproject.toml` and `uv.lock` are source of truth |
| Artifacts | Do not commit ONNX, npy calibration sets, or experiment outputs |
| Work plans | `docs/plans/YYYY-MM-DD-<slug>.md` |

## Quantization contract

- Dynamic quantization covers Conv / MatMul weights (IntegerOps). Activations stay unquantized.
- Static quantization is QDQ. Activations QUInt8, weights QInt8, per-channel. Output is opset 21 / IR 9.
- Static quantization reads `*.npy` in the calibration directory (non-recursive, name order). `np.load(..., allow_pickle=False)`. dtype float32, shape `(1, C, H, W)`.
- Compare C/H/W. npy N is 1. The ONNX leading input N is 0 (dynamic) or 1.
- Return unquantized layers as a table.

## Self-check (prefer a narrow command)

```bash
# Single test or file
uv run pytest tests/path/to/test_module.py::test_name -q
uv run pytest tests/path/to/test_module.py -q

# Broader check after a change
uv run pytest -q

# Sync deps (honor the lock)
uv sync --frozen
```

- Default verification uses small inputs and does not need a network. Do not assume a real-model ONNX, a large calibration set, or a long quantization run.
- Run tests that cover the edited area first. For new behavior, pin the expectation in a test before implementing it.

## Permission

| No need to ask | Ask first |
|---|---|
| Reads; focused edits under `src/` / `tests/` | Adding, removing, or major-updating a dependency |
| Targeted `pytest`, `uv sync --frozen` | Destructive git, force push, push to a shared remote |
| Small doc fixes | Long quantization of a real ONNX; deleting artifacts |
