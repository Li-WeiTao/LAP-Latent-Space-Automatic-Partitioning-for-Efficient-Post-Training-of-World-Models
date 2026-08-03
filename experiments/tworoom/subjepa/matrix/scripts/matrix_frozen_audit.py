#!/usr/bin/env python3
"""Frozen-parameter audit for every Sub-JEPA TwoRoom matrix training job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint
from experiments.control_matrix.jepa_checkpoint_probe import sha256_file


def audit_run(
    *,
    base_frozen: dict[str, torch.Tensor],
    checkpoint: Path,
    model_family: str,
) -> dict[str, object]:
    trained = load_jepa_object_checkpoint(
        checkpoint, model_family=model_family, map_location=torch.device("cpu")
    )
    violations: list[str] = []
    trained_params = dict(trained.named_parameters())
    for name, tensor in base_frozen.items():
        other = trained_params.get(name)
        if other is None:
            violations.append(f"missing:{name}")
            continue
        if not torch.equal(tensor.cpu(), other.cpu()):
            violations.append(name)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "passed": not violations,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model-family", default="subjepa")
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock["sha256"]["checkpoint"] != sha256_file(args.checkpoint):
        raise SystemExit("checkpoint sha256 does not match pre_execution_lock")

    base = load_jepa_object_checkpoint(
        args.checkpoint, model_family=args.model_family, map_location=torch.device("cpu")
    )
    base_frozen = {
        name: param.detach().clone()
        for name, param in base.named_parameters()
        if not name.startswith(("predictor.", "pred_proj."))
    }

    rows: list[dict[str, object]] = []
    for manifest_path in sorted(args.work_root.glob("training/**/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for region in manifest.get("regions", {}).values():
            ckpt_path = Path(region["checkpoint"])
            if not ckpt_path.is_file():
                ckpt_path = PROJECT_ROOT / region["checkpoint"]
            rows.append(
                {
                    "run_dir": str(manifest_path.parent.relative_to(args.work_root)),
                    **audit_run(
                        base_frozen=base_frozen,
                        checkpoint=ckpt_path,
                        model_family=args.model_family,
                    ),
                }
            )

    passed = all(bool(row["passed"]) for row in rows)
    report = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_matrix_frozen_audit",
        "git_commit": lock.get("git_commit"),
        "pre_execution_lock": str(args.lock),
        "num_runs": len(rows),
        "all_passed": passed,
        "runs": rows,
    }
    out = args.work_root / "manifests/frozen_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": passed, "num_runs": len(rows), "out": str(out)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
