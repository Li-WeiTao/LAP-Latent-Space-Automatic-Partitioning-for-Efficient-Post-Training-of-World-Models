"""Gate and partition one-time cost measurements."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import TrainingAnchorConfig


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def measure_gate_and_partition(
    repo_root: Path,
    cfg: TrainingAnchorConfig,
    *,
    scratch_dir: Path,
    rerun: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    gate_manifest_path = cfg.gate_manifest.resolve(strict=True)
    committed = _read_json(gate_manifest_path)
    selected = committed.get("selected_method", committed.get("method"))

    if rerun:
        auto_out = scratch_dir / "auto_gate_rerun"
        auto_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            str(repo_root / "experiments/control_matrix/fit_partition.py"),
            "--method",
            "auto",
            "--dataset-name",
            cfg.task,
            "--data-file",
            str(cfg.data_file),
            "--latent-cache",
            str(cfg.latent_cache),
            "--frameskip",
            str(cfg.frameskip),
            "--num-clusters",
            "3",
            "--out-dir",
            str(auto_out),
            "--overwrite",
            "--device",
            "cuda",
        ]
        started = time.perf_counter()
        subprocess.run(cmd, check=True, cwd=repo_root)
        auto_elapsed = time.perf_counter() - started
        auto_manifest = _read_json(auto_out / "manifest.json")
        gate_meta = auto_manifest.get("method_metadata", {}).get("automatic_gate", {})
        gate_sec = float(gate_meta.get("elapsed_sec", auto_elapsed))
        total_auto_sec = float(auto_manifest.get("elapsed_sec", auto_elapsed))
        partition_sec = max(total_auto_sec - gate_sec, 0.0)
        partition_status = selected if selected != "global" else "N/A"
    else:
        gate_meta = committed.get("method_metadata", {}).get("automatic_gate", {})
        gate_sec = float(gate_meta.get("elapsed_sec", float("nan")))
        total_auto_sec = float(committed.get("elapsed_sec", float("nan")))
        partition_sec = (
            max(total_auto_sec - gate_sec, 0.0) if selected != "global" else float("nan")
        )
        partition_status = "N/A" if selected == "global" else selected

    partition_only_sec = float("nan")
    if selected != "global" and rerun:
        spectral_out = scratch_dir / "partition_only_rerun"
        spectral_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            str(repo_root / "experiments/control_matrix/fit_partition.py"),
            "--method",
            "spectral",
            "--dataset-name",
            cfg.task,
            "--data-file",
            str(cfg.data_file),
            "--latent-cache",
            str(cfg.latent_cache),
            "--frameskip",
            str(cfg.frameskip),
            "--num-clusters",
            "3",
            "--seed",
            str(committed.get("partition_seed", 0)),
            "--out-dir",
            str(spectral_out),
            "--overwrite",
            "--device",
            "cuda",
        ]
        started = time.perf_counter()
        subprocess.run(cmd, check=True, cwd=repo_root)
        partition_only_sec = time.perf_counter() - started

    result = {
        "task": cfg.task,
        "selected_branch": selected,
        "gate_wall_sec": gate_sec,
        "auto_total_wall_sec": total_auto_sec,
        "partition_wall_sec": partition_sec if selected != "global" else "N/A",
        "partition_only_rerun_sec": partition_only_sec if selected != "global" else "N/A",
        "gate_manifest": str(gate_manifest_path),
        "latent_cache": str(cfg.latent_cache),
    }
    if provenance is not None:
        result["provenance"] = provenance
    (scratch_dir / "gate_partition.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
