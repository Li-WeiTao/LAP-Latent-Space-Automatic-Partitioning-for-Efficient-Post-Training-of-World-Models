"""Load immutable scratch benchmark artifacts and merge split-run outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_scratch_results(scratch_dir: Path) -> dict[str, Any]:
    """Read persisted phase results from scratch subdirectories."""
    scratch_dir = scratch_dir.resolve()
    inference: list[dict[str, Any]] = []
    inference_dir = scratch_dir / "inference"
    if inference_dir.is_dir():
        for path in sorted(inference_dir.glob("inference_*.json")):
            payload = _read_json(path)
            if payload is not None:
                inference.append(payload)

    return {
        "joint_training": _read_json(scratch_dir / "training" / "joint_training.json"),
        "lap_regional_training": _read_json(
            scratch_dir / "training" / "lap_regional_training.json"
        ),
        "gate_partition": _read_json(scratch_dir / "gate_partition" / "gate_partition.json"),
        "inference": inference,
    }


def merge_phase_results(
    scratch_dir: Path,
    *,
    joint: dict[str, Any] | None = None,
    lap: dict[str, Any] | None = None,
    gate_partition: dict[str, Any] | None = None,
    inference: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Overlay newly measured phases onto scratch artifacts without dropping prior runs."""
    stored = load_scratch_results(scratch_dir)
    merged_inference = list(stored["inference"])
    if inference:
        by_key = {(item["task"], item["mode"]): item for item in merged_inference}
        for item in inference:
            by_key[(item["task"], item["mode"])] = item
        merged_inference = sorted(by_key.values(), key=lambda row: (row["task"], row["mode"]))

    return {
        "joint_training": joint if joint is not None else stored["joint_training"],
        "lap_regional_training": lap
        if lap is not None
        else stored["lap_regional_training"],
        "gate_partition": gate_partition
        if gate_partition is not None
        else stored["gate_partition"],
        "inference": merged_inference,
    }
