"""Environment and provenance metadata for efficiency benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_metadata(repo_root: Path, *, seed: int, device: str) -> dict[str, Any]:
    git_commit = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_status = _run(["git", "-C", str(repo_root), "status", "--short"])
    gpu_name = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpu_lines = [line.strip() for line in (gpu_name or "").splitlines() if line.strip()]
    return {
        "repo_root": str(repo_root.resolve()),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": device,
        "precision_training": "fp32",
        "amp_inference": False,
        "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cpu_threads_torch": torch.get_num_threads(),
        "seed": seed,
        "gpus": gpu_lines,
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _gpu_name_for_device(device: str) -> str | None:
    if not device.startswith("cuda"):
        return None
    index = device.split(":", 1)[1] if ":" in device else "0"
    line = _run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
    )
    return line.splitlines()[0].strip() if line else None


def phase_provenance(
    repo_root: Path,
    *,
    phase: str,
    device: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Self-contained provenance stored inside each scratch artifact."""
    payload: dict[str, Any] = {
        "phase": phase,
        "device": device,
        "seed": seed,
        "git_commit": _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_status_short": _run(["git", "-C", str(repo_root), "status", "--short"]),
        "gpu_name": _gpu_name_for_device(device),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload
