from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Host hook. Return the training split with inference preprocessing (no train aug).
# Each image must already be the float32 (1, C, H, W) tensor the ONNX graph expects.
# Do not decode JPEGs or reimplement preprocessing here.


@dataclass(frozen=True)
class HostSample:
    source_name: str
    image: np.ndarray
    classes: frozenset[int]


@dataclass(frozen=True)
class CalibrationSet:
    source_names: tuple[str, ...]
    covered_classes: frozenset[int]
    missing_classes: frozenset[int]


def host_samples() -> list[HostSample]:
    raise NotImplementedError(
        "replace host_samples() with your eval-preprocessed training split"
    )


def select_calibration_indices(
    class_sets: list[frozenset[int]],
    count: int,
    seed: int,
    num_classes: int,
) -> list[int]:
    if count < 1:
        raise ValueError(f"count must be >= 1: {count}")
    if count > len(class_sets):
        raise ValueError(f"count={count} exceeds {len(class_sets)} candidates")
    uncovered = set(range(num_classes))
    remaining = set(range(len(class_sets)))
    selected: list[int] = []
    while uncovered and remaining and len(selected) < count:
        best = max(remaining, key=lambda index: (len(class_sets[index] & uncovered), -index))
        if not class_sets[best] & uncovered:
            break
        selected.append(best)
        uncovered -= class_sets[best]
        remaining.remove(best)
    fillers = [
        index
        for index in np.random.default_rng(seed).permutation(len(class_sets))
        if index in remaining
    ]
    selected.extend(fillers[: count - len(selected)])
    return selected


def prepare_calibration_set(
    samples: Sequence[HostSample],
    dest: Path,
    count: int,
    seed: int,
    num_classes: int,
    split: str = "train",
) -> CalibrationSet:
    class_sets = [sample.classes for sample in samples]
    indices = select_calibration_indices(class_sets, count, seed, num_classes)
    chosen = [samples[index] for index in indices]
    source_names = tuple(sample.source_name for sample in chosen)
    batches = [np.ascontiguousarray(sample.image) for sample in chosen]
    _write_calibration_arrays(batches, dest)
    dest.joinpath("manifest.txt").write_text(
        "\n".join(source_names) + "\n", encoding="utf-8"
    )
    covered = frozenset().union(*(class_sets[index] for index in indices))
    missing = frozenset(range(num_classes)) - covered
    shape = ",".join(str(dim) for dim in batches[0].shape)
    dest.joinpath("meta.txt").write_text(
        f"seed={seed}\ncount={count}\nsplit={split}\n"
        f"covered={sorted(covered)}\nmissing={sorted(missing)}\n"
        f"shape={shape}\n",
        encoding="utf-8",
    )
    return CalibrationSet(source_names, covered, missing)


def classes_in_mask(mask: np.ndarray, ignore_label: int | None = None) -> frozenset[int]:
    # Void/ignore ids are host-defined (Cityscapes/COCO-Stuff often use 255).
    # They are not classes to cover. Pass ignore_label; do not assume 255.
    present = {int(item) for item in np.unique(mask).tolist()}
    if ignore_label is not None:
        present.discard(ignore_label)
    return frozenset(present)


def _write_calibration_arrays(batches: Sequence[np.ndarray], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for index, batch in enumerate(batches):
        target = dest / f"{index:04d}.npy"
        np.save(target, np.ascontiguousarray(batch, dtype=np.float32))
        written.append(target)
    keep = {path.name for path in written}
    for leftover in dest.glob("*.npy"):
        if leftover.name not in keep:
            leftover.unlink()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static-quantization calibration arrays")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num", type=int, default=128)
    parser.add_argument("--seed", type=int, default=304)
    parser.add_argument("--num-classes", type=int, required=True)
    args = parser.parse_args()
    result = prepare_calibration_set(
        host_samples(),
        Path(args.output_dir),
        args.num,
        args.seed,
        args.num_classes,
    )
    print(
        f"wrote {len(result.source_names)} arrays to {args.output_dir} "
        f"(covered={sorted(result.covered_classes)} missing={sorted(result.missing_classes)})"
    )


if __name__ == "__main__":
    main()
