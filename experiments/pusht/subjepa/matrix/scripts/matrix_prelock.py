#!/usr/bin/env python3
"""PushT Sub-JEPA matrix pre-lock: dynamic gate branch, asset validation, paired starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
FORMAL_SCRIPTS = Path(__file__).resolve().parents[1].parent / "formal" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FORMAL_SCRIPTS))

from experiments.control_matrix.region_risk_lib import atomic_write_json  # noqa: E402
import pusht_formal_lib  # noqa: E402

ALLOWED_GATE_BRANCHES = pusht_formal_lib.ALLOWED_GATE_BRANCHES
sha256_file = pusht_formal_lib.sha256_file
verify_smoke_verified = pusht_formal_lib.verify_smoke_verified

TRAIN_SEEDS = (0, 42, 625)
PARTITION_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2, 3, 4)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT
    ).strip()


def load_passport(formal_root: Path) -> dict[str, Any]:
    path = formal_root / "manifests" / "material_passport.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing material passport: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def gate_selected_branch(passport: dict[str, Any]) -> str:
    branch = passport.get("selected_branch")
    if branch not in ALLOWED_GATE_BRANCHES:
        raise RuntimeError(
            f"invalid gate selected_branch {branch!r}; "
            f"allowed={sorted(ALLOWED_GATE_BRANCHES)}"
        )
    return str(branch)


def _partition_complete(seed_dir: Path) -> bool:
    return (
        (seed_dir / "cluster_labels.npz").is_file()
        and (seed_dir / "partition").exists()
    )


def validate_forced_spectral(formal_root: Path) -> dict[str, Any]:
    spectral_root = formal_root / "partitions" / "spectral"
    seeds: dict[str, bool] = {}
    for seed in PARTITION_SEEDS:
        seed_dir = spectral_root / f"seed{seed}"
        seeds[str(seed)] = _partition_complete(seed_dir)
        if not seeds[str(seed)]:
            raise RuntimeError(f"incomplete forced-spectral partition: {seed_dir}")
    return {
        "formal_root": str(spectral_root.resolve()),
        "seeds": list(PARTITION_SEEDS),
        "complete": True,
    }


def validate_spectral_auto_lap(
    formal_root: Path, *, deployment_seed: int
) -> dict[str, Any]:
    gate_partition = formal_root / "gate" / "partition"
    router = formal_root / "router" / f"deployment_seed{deployment_seed}"
    for path in (gate_partition / "manifest.json", gate_partition / "partition"):
        if not path.exists():
            raise RuntimeError(f"missing spectral auto-lap gate artifact: {path}")
    if not router.is_dir():
        raise RuntimeError(f"missing deployment router: {router}")
    return {
        "type": "spectral",
        "deployment_seed": deployment_seed,
        "partition_path": str(gate_partition.resolve()),
        "router_path": str(router.resolve()),
        "training_template": f"spectral/partition{deployment_seed}_train{{train_seed}}",
        "eval_template": (
            f"spectral/partition{deployment_seed}_train{{train_seed}}/eval{{eval_seed}}"
        ),
    }


def validate_global_auto_lap(formal_root: Path) -> dict[str, Any]:
    gate_partition = formal_root / "gate" / "partition"
    if not (gate_partition / "manifest.json").is_file():
        raise RuntimeError(f"missing global gate partition manifest: {gate_partition}")
    return {
        "type": "global",
        "partition_path": str(gate_partition.resolve()),
        "training_template": "global/train{train_seed}",
        "eval_template": "global/train{train_seed}/eval{eval_seed}",
        "note": (
            "Auto-LAP symlinks Global-FT checkpoints; rollout results must match "
            "Global-FT exactly per train seed."
        ),
    }


def paired_start_hashes(
    *,
    canon_short: Path,
    canon_long: Path,
) -> dict[str, Any]:
    def hash_horizon(root: Path, label: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for seed in EVAL_SEEDS:
            path = root / f"eval{seed}" / "results.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing canonical {label} eval seed {seed}: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            starts = payload["eval_start_indices"]
            digest = hashlib.sha256(
                json.dumps([int(v) for v in starts], separators=(",", ":")).encode(
                    "ascii"
                )
            ).hexdigest()
            out[str(seed)] = digest
        return out

    return {
        "short": hash_horizon(canon_short, "short"),
        "long": hash_horizon(canon_long, "long"),
    }


def build_pre_execution_lock(
    *,
    repo_root: Path,
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
    verify_smoke_verified(smoke_root, expected_cache_sha256=expected_smoke_cache_sha256)
    passport = load_passport(formal_root)
    if passport.get("verification_status") != "VERIFIED":
        raise RuntimeError(
            f"gate passport not VERIFIED: {passport.get('verification_status')}"
        )
    if not passport.get("replay_audit", {}).get("all_passed"):
        raise RuntimeError("formal replay audit not all_passed")

    branch = gate_selected_branch(passport)
    gate = passport.get("gate_task_summary", {})
    deployment_seed = int(gate.get("deployment_seed", 0))

    cache_path = formal_root / "preparation" / "embedding_cache.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing formal full cache: {cache_path}")
    cache_sha = sha256_file(cache_path)
    if cache_sha != passport.get("full_cache_sha256"):
        raise RuntimeError(
            f"formal cache sha mismatch: {cache_sha} != {passport.get('full_cache_sha256')}"
        )

    forced_spectral = validate_forced_spectral(formal_root)
    if branch == "spectral":
        auto_lap_source = validate_spectral_auto_lap(
            formal_root, deployment_seed=deployment_seed
        )
        auto_partition_target = formal_root / "gate" / "partition"
    else:
        auto_lap_source = validate_global_auto_lap(formal_root)
        auto_partition_target = formal_root / "gate" / "partition"

    replay = passport.get("replay_audit", {}).get("reports", {})
    replay_16 = replay.get("replay_16", {})
    replay_64 = replay.get("replay_64", {})

    lock = {
        "schema_version": 2,
        "gate_selected_branch": branch,
        "auto_lap_source": auto_lap_source,
        "forced_spectral_partition": {
            **forced_spectral,
            "matrix_root": str((matrix_root / "partitions" / "spectral").resolve()),
        },
        "gate_status": passport.get("verification_status"),
        "gate_passport": str((formal_root / "manifests" / "material_passport.json").resolve()),
        "deployment_seed": deployment_seed,
        "formal_producer_commit": passport.get("formal_producer_commit")
        or passport.get("git_commit"),
        "matrix_runner_commit": git_head(),
        "sha256": {
            "dataset": sha256_file(dataset),
            "checkpoint": sha256_file(checkpoint),
            "full_cache": cache_sha,
            "task_spec": sha256_file(task_spec),
        },
        "task_spec_path": str(task_spec.resolve()),
        "eval_config_name": "pusht",
        "replay_audit": {
            "all_passed": passport.get("replay_audit", {}).get("all_passed"),
            "replay_16_passed": replay_16.get("passed"),
            "replay_16_max_abs_diff": replay_16.get("max_abs_diff"),
            "replay_64_passed": replay_64.get("passed"),
            "replay_64_max_abs_diff": replay_64.get("max_abs_diff"),
        },
        "gate_checks": {
            "safety_pass": gate.get("safety_pass"),
            "background_pass": gate.get("background_pass"),
            "selected_method": gate.get("selected_method"),
            "reason": gate.get("reason"),
        },
        "smoke_artifact_bindings": passport.get("smoke_artifact_bindings"),
        "paired_start_hashes": paired_start_hashes(
            canon_short=canon_short, canon_long=canon_long
        ),
        "paired_start_sources": {
            "short": str(canon_short.resolve()),
            "long": str(canon_long.resolve()),
        },
        "auto_partition_symlink_source": str(auto_partition_target.resolve()),
    }
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--canon-short", type=Path, required=True)
    parser.add_argument("--canon-long", type=Path, required=True)
    parser.add_argument("--expected-smoke-cache-sha256", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="pre_execution_lock.json path (default: matrix/manifests/)",
    )
    args = parser.parse_args()

    lock = build_pre_execution_lock(
        repo_root=args.repo_root,
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
    out = args.out or (args.matrix_root / "manifests" / "pre_execution_lock.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
