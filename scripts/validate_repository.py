#!/usr/bin/env python3
"""Fast repository validation that does not require TwoRoom data or a GPU."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_python_syntax() -> list[str]:
    failures: list[str] = []
    for path in ROOT.rglob("*.py"):
        if any(part in {"results", "_python_deps", ".venv"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    return failures


def validate_shell_syntax() -> list[str]:
    failures: list[str] = []
    for path in (ROOT / "experiments" / "tworoom" / "scripts").glob("*.sh"):
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode:
            failures.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")
    return failures


def validate_routing_artifacts() -> list[str]:
    from lap.partition.artifact import PartitionArtifact
    from lap.routing.voronoi import VoronoiRouter

    failures: list[str] = []
    root = ROOT / "experiments" / "tworoom" / "results" / "latent_landmark_spectral_k3"
    for directory in sorted(root.glob("spectral_*")):
        try:
            artifact = PartitionArtifact.load(directory)
            router = VoronoiRouter(artifact)
            labels = router.route(artifact.prototypes[: min(8, len(artifact.prototypes))])
            if labels.shape != (min(8, len(artifact.prototypes)),):
                raise ValueError("unexpected router output shape")
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{directory.relative_to(ROOT)}: {exc}")
    return failures


def validate_compact_results() -> list[str]:
    failures: list[str] = []
    result_root = ROOT / "experiments" / "tworoom" / "results"
    for path in result_root.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    return failures


def main() -> None:
    checks = {
        "python_syntax": validate_python_syntax(),
        "shell_syntax": validate_shell_syntax(),
        "routing_artifacts": validate_routing_artifacts(),
        "compact_json": validate_compact_results(),
    }
    print(json.dumps({name: len(errors) for name, errors in checks.items()}, indent=2))
    failures = [f"[{name}] {error}" for name, errors in checks.items() for error in errors]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
