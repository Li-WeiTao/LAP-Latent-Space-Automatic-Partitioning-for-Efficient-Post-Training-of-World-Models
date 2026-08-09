#!/usr/bin/env python3
"""Mandatory pre-matrix checks before 50-epoch Sub-JEPA PushT training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
FORMAL_SCRIPTS = SCRIPT_DIR.parent.parent / "formal" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(FORMAL_SCRIPTS))

from experiments.control_matrix.region_risk_lib import atomic_write_json  # noqa: E402
import matrix_prelock  # noqa: E402
import pusht_formal_lib  # noqa: E402

ALLOWED_GATE_BRANCHES = pusht_formal_lib.ALLOWED_GATE_BRANCHES
sha256_file = pusht_formal_lib.sha256_file
verify_smoke_verified = pusht_formal_lib.verify_smoke_verified
load_passport = matrix_prelock.load_passport
gate_selected_branch = matrix_prelock.gate_selected_branch
build_pre_execution_lock = matrix_prelock.build_pre_execution_lock


def resolve_auto_lap_mapping(matrix_root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    branch = lock["gate_selected_branch"]
    auto = lock["auto_lap_source"]
    mapping: dict[str, Any] = {
        "gate_selected_branch": branch,
        "auto_lap_source": auto,
    }
    auto_partition = matrix_root / "auto" / "partition"
    if auto_partition.exists():
        mapping["auto_partition_link"] = str(auto_partition.resolve())
        if auto_partition.is_symlink():
            mapping["auto_partition_target"] = str(auto_partition.resolve())
    if branch == "spectral":
        mapping["expected_training_glob"] = (
            f"training/spectral/partition{auto['deployment_seed']}_train{{train_seed}}"
        )
    else:
        mapping["expected_training_glob"] = "training/global/train{train_seed}"
    return mapping


def run_preflight(
    *,
    formal_root: Path,
    matrix_root: Path,
    smoke_root: Path,
    dataset: Path,
    checkpoint: Path,
    task_spec: Path,
    canon_short: Path,
    canon_long: Path,
    expected_smoke_cache_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}

    try:
        smoke = verify_smoke_verified(
            smoke_root, expected_cache_sha256=expected_smoke_cache_sha256
        )
        checks["smoke_artifacts"] = {"passed": True, **smoke["artifact_bindings"]}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"smoke artifacts: {exc}")
        checks["smoke_artifacts"] = {"passed": False, "error": str(exc)}

    passport_path = formal_root / "manifests" / "material_passport.json"
    if not passport_path.is_file():
        errors.append("missing material_passport.json")
        passport = {}
    else:
        passport = load_passport(formal_root)
        checks["material_passport"] = {
            "path": str(passport_path),
            "verification_status": passport.get("verification_status"),
            "selected_branch": passport.get("selected_branch"),
        }
        if passport.get("verification_status") != "VERIFIED":
            errors.append(
                f"material_passport not VERIFIED: {passport.get('verification_status')}"
            )

    cache_path = formal_root / "preparation" / "embedding_cache.npz"
    if cache_path.is_file():
        cache_sha = sha256_file(cache_path)
        checks["full_cache_sha256"] = cache_sha
        if passport and cache_sha != passport.get("full_cache_sha256"):
            errors.append("full cache sha differs from passport")
    else:
        errors.append(f"missing full cache: {cache_path}")

    replay = passport.get("replay_audit", {}) if passport else {}
    replay_reports = replay.get("reports", {})
    checks["replay_audit"] = {
        "all_passed": replay.get("all_passed"),
        "replay_16": replay_reports.get("replay_16"),
        "replay_64": replay_reports.get("replay_64"),
    }
    if not replay.get("all_passed"):
        errors.append("formal replay audit not all_passed")

    if passport:
        try:
            branch = gate_selected_branch(passport)
            if branch not in ALLOWED_GATE_BRANCHES:
                errors.append(f"invalid selected_branch: {branch}")
            checks["gate_selected_branch"] = branch
            gate = passport.get("gate_task_summary", {})
            checks["gate_two_checks"] = {
                "safety_pass": gate.get("safety_pass"),
                "background_pass": gate.get("background_pass"),
                "selected_method": gate.get("selected_method"),
                "reason": gate.get("reason"),
            }
            if branch == "spectral":
                if not gate.get("safety_pass") or not gate.get("background_pass"):
                    errors.append(
                        "spectral branch requires gate safety and background checks to pass"
                    )
            elif branch == "global":
                if gate.get("selected_method") != "global":
                    errors.append(
                        "global branch requires gate selected_method=global"
                    )
            else:
                errors.append(f"invalid selected_branch: {branch}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gate branch: {exc}")

    lock_path = matrix_root / "manifests" / "pre_execution_lock.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks["pre_execution_lock"] = lock
        checks["auto_lap_mapping"] = resolve_auto_lap_mapping(matrix_root, lock)
        if lock.get("formal_producer_commit") == lock.get("matrix_runner_commit"):
            checks["commit_note"] = "formal producer and matrix runner on same commit"
        else:
            checks["commit_note"] = (
                "formal producer and matrix runner commits differ (allowed for "
                "downstream script fixes without cache re-encode)"
            )
            checks["formal_producer_commit"] = lock.get("formal_producer_commit")
            checks["matrix_runner_commit"] = lock.get("matrix_runner_commit")
    else:
        errors.append(f"missing pre_execution_lock: {lock_path}")
        lock = {}

    try:
        rebuilt = build_pre_execution_lock(
            formal_root=formal_root,
            matrix_root=matrix_root,
            smoke_root=smoke_root,
            repo_root=PROJECT_ROOT,
            dataset=dataset,
            checkpoint=checkpoint,
            task_spec=task_spec,
            canon_short=canon_short,
            canon_long=canon_long,
            expected_smoke_cache_sha256=expected_smoke_cache_sha256,
        )
        checks["prelock_validation"] = {"passed": True, "gate_selected_branch": rebuilt["gate_selected_branch"]}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"prelock validation: {exc}")
        checks["prelock_validation"] = {"passed": False, "error": str(exc)}

    report = {
        "schema_version": 1,
        "scope": "pusht_subjepa_matrix_preflight",
        "passed": not errors,
        "errors": errors,
        "checks": checks,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--canon-short", type=Path, required=True)
    parser.add_argument("--canon-long", type=Path, required=True)
    parser.add_argument("--expected-smoke-cache-sha256", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_preflight(
        formal_root=args.formal_root,
        matrix_root=args.matrix_root,
        smoke_root=args.smoke_root,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        task_spec=args.task_spec,
        canon_short=args.canon_short,
        canon_long=args.canon_long,
        expected_smoke_cache_sha256=args.expected_smoke_cache_sha256,
    )
    out = args.out or (args.matrix_root / "manifests" / "preflight_report.json")
    atomic_write_json(out, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
