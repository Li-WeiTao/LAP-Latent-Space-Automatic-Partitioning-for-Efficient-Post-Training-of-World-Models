#!/usr/bin/env python3
"""PushT Sub-JEPA formal helpers: smoke protection, cache reuse, passport augmentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.region_risk_lib import atomic_write_json  # noqa: E402

SMOKE_MANIFESTS = (
    "verification_status.json",
    "smoke_cache-equivalence.json",
    "smoke_frozen-audit.json",
    "smoke_route-equivalence.json",
)

ALLOWED_GATE_BRANCHES = frozenset({"global", "spectral"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=PROJECT_ROOT
    ).strip()


def smoke_manifest_dir(smoke_root: Path) -> Path:
    return smoke_root / "manifests"


def collect_smoke_artifact_bindings(smoke_root: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    manifest_dir = smoke_manifest_dir(smoke_root)
    for name in SMOKE_MANIFESTS:
        path = manifest_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing smoke manifest: {path}")
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    cache_path = smoke_root / "preparation" / "embedding_cache.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"missing smoke cache: {cache_path}")
    bindings["embedding_cache.npz"] = {
        "path": str(cache_path.resolve()),
        "sha256": sha256_file(cache_path),
    }
    return bindings


def verify_smoke_verified(smoke_root: Path, *, expected_cache_sha256: str) -> dict[str, Any]:
    status_path = smoke_manifest_dir(smoke_root) / "verification_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "VERIFIED":
        raise RuntimeError(
            f"smoke status is not VERIFIED: {status.get('status')!r}"
        )
    preserved = status.get("preserved_cache", {})
    preserved_sha = preserved.get("sha256")
    if preserved_sha != expected_cache_sha256:
        raise RuntimeError(
            f"verification_status preserved_cache sha mismatch: "
            f"{preserved_sha} != {expected_cache_sha256}"
        )
    bindings = collect_smoke_artifact_bindings(smoke_root)
    cache_sha = bindings["embedding_cache.npz"]["sha256"]
    if cache_sha != expected_cache_sha256:
        raise RuntimeError(
            f"smoke cache sha changed: {cache_sha} != {expected_cache_sha256}"
        )
    return {
        "status": status,
        "artifact_bindings": bindings,
        "smoke_cache_sha256": cache_sha,
    }


def current_binding_hashes(
    *,
    dataset: Path,
    checkpoint: Path,
    task_spec: Path,
) -> dict[str, str]:
    return {
        "dataset": sha256_file(dataset),
        "checkpoint": sha256_file(checkpoint),
        "task_spec": sha256_file(task_spec),
    }


def formal_cache_reusable(
    *,
    formal_root: Path,
    dataset: Path,
    checkpoint: Path,
    task_spec: Path,
) -> tuple[bool, dict[str, Any]]:
    formal_root = formal_root.resolve()
    cache_path = formal_root / "preparation" / "embedding_cache.npz"
    manifest_path = formal_root / "manifests" / "formal_cache_manifest.json"
    report: dict[str, Any] = {
        "cache_path": str(cache_path),
        "manifest_path": str(manifest_path),
    }
    if not cache_path.is_file():
        report["reason"] = "missing_embedding_cache"
        return False, report

    current = current_binding_hashes(
        dataset=dataset, checkpoint=checkpoint, task_spec=task_spec
    )
    current["full_cache"] = sha256_file(cache_path)
    report["current_sha256"] = current

    if not manifest_path.is_file():
        report["reason"] = "missing_formal_cache_manifest"
        return False, report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["manifest_sha256"] = {
        "full_cache": manifest.get("full_cache_sha256"),
        "checkpoint": manifest.get("checkpoint_sha256"),
        "dataset": manifest.get("dataset_sha256"),
    }

    mismatches = []
    for key in ("dataset", "checkpoint"):
        manifest_key = f"{key}_sha256"
        if manifest.get(manifest_key) != current[key]:
            mismatches.append(key)
    if manifest.get("full_cache_sha256") != current["full_cache"]:
        mismatches.append("full_cache")
    if mismatches:
        report["reason"] = "hash_mismatch"
        report["mismatches"] = mismatches
        return False, report

    report["reason"] = "reusable"
    report["reusable"] = True
    return True, report


def augment_material_passport(
    *,
    formal_root: Path,
    smoke_root: Path,
    expected_cache_sha256: str,
    formal_producer_commit: str | None = None,
) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    passport_path = formal_root / "manifests" / "material_passport.json"
    if not passport_path.is_file():
        raise FileNotFoundError(f"missing material passport: {passport_path}")

    smoke_report = verify_smoke_verified(
        smoke_root, expected_cache_sha256=expected_cache_sha256
    )
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    producer = formal_producer_commit or passport.get("git_commit") or git_head()
    passport["formal_producer_commit"] = producer
    passport["smoke_root"] = str(smoke_root.resolve())
    passport["smoke_cache_sha256"] = smoke_report["smoke_cache_sha256"]
    passport["smoke_artifact_bindings"] = smoke_report["artifact_bindings"]
    passport["smoke_protection"] = {
        "verified_status": smoke_report["status"].get("status"),
        "note": "Formal writes only under formal/; smoke preparation/ is read-only.",
    }
    atomic_write_json(passport_path, passport)
    return passport


def write_cache_reuse_record(formal_root: Path, report: dict[str, Any]) -> None:
    out = formal_root / "manifests" / "cache_reuse_audit.json"
    payload = {
        "schema_version": 1,
        "scope": "pusht_subjepa_formal_cache_reuse",
        "git_commit": git_head(),
        **report,
    }
    atomic_write_json(out, payload)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--expected-smoke-cache-sha256", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "verify-smoke",
            "cache-reusable",
            "augment-passport",
        ),
        required=True,
    )
    parser.add_argument("--formal-producer-commit", default="")
    args = parser.parse_args()

    if args.phase == "verify-smoke":
        report = verify_smoke_verified(
            args.smoke_root, expected_cache_sha256=args.expected_smoke_cache_sha256
        )
    elif args.phase == "cache-reusable":
        if not all([args.dataset, args.checkpoint, args.task_spec]):
            raise ValueError("--dataset --checkpoint --task-spec required")
        reusable, report = formal_cache_reusable(
            formal_root=args.formal_root,
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            task_spec=args.task_spec,
        )
        report["reusable"] = reusable
        write_cache_reuse_record(args.formal_root, report)
    else:
        report = augment_material_passport(
            formal_root=args.formal_root,
            smoke_root=args.smoke_root,
            expected_cache_sha256=args.expected_smoke_cache_sha256,
            formal_producer_commit=args.formal_producer_commit or None,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
