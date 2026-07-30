#!/usr/bin/env python3
"""Fast repository validation that does not require TwoRoom data or a GPU."""

from __future__ import annotations

import ast
import hashlib
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
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
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


def validate_migration_manifest() -> list[str]:
    failures: list[str] = []
    manifest_path = ROOT / "MIGRATION_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in payload.get("files", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing manifest file: {entry['path']}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != int(entry["bytes"]):
            failures.append(
                f"{entry['path']}: bytes={len(data)}, manifest={entry['bytes']}"
            )
        if digest != entry["sha256"]:
            failures.append(
                f"{entry['path']}: sha256={digest}, manifest={entry['sha256']}"
            )
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        failures.append(f"git ls-files failed: {result.stderr.strip()}")
    else:
        repository_files = {
            path for path in result.stdout.splitlines() if path != "MIGRATION_MANIFEST.json"
        }
        manifest_files = {entry["path"] for entry in payload.get("files", [])}
        missing_entries = sorted(repository_files - manifest_files)
        stale_entries = sorted(manifest_files - repository_files)
        failures.extend(f"unmanifested repository file: {path}" for path in missing_entries)
        failures.extend(f"stale manifest entry: {path}" for path in stale_entries)
    return failures


def validate_main_result_audit() -> list[str]:
    script = ROOT / "experiments" / "tworoom" / "aggregate_tworoom_main.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check-existing"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        return [detail]
    return []


def main() -> None:
    checks = {
        "python_syntax": validate_python_syntax(),
        "shell_syntax": validate_shell_syntax(),
        "routing_artifacts": validate_routing_artifacts(),
        "compact_json": validate_compact_results(),
        "migration_manifest": validate_migration_manifest(),
        "main_result_audit": validate_main_result_audit(),
    }
    print(json.dumps({name: len(errors) for name, errors in checks.items()}, indent=2))
    failures = [f"[{name}] {error}" for name, errors in checks.items() for error in errors]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
