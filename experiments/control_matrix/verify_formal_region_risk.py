#!/usr/bin/env python3
"""Verify formal region-risk run acceptance criteria."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.episode_split import load_split_manifest, split_manifest_is_unsubsampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expect-smoke", action="store_true")
    return parser.parse_args()


def assert_finite_csv(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value in ("", "nan", "inf", "-inf"):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise AssertionError(f"non-finite value in {path}: {row}")


def main() -> None:
    args = parse_args()
    root = args.work_root.resolve()
    audit = json.loads((root / "evaluation/audit.json").read_text(encoding="utf-8"))
    split = load_split_manifest(root / "split_manifest.json")

    if args.expect_smoke:
        assert audit.get("smoke_only") is True, audit
        assert audit.get("paper_eligible") is False, audit
        assert split.get("subsampled") is True, split
    else:
        assert audit.get("smoke_only") is False, audit
        assert audit.get("paper_eligible") is True, audit
        assert split.get("subsampled") is False, split
        assert split_manifest_is_unsubsampled(split), split

    assert audit.get("posttraining_train_only_valid") is True, audit
    assert audit.get("split_manifest_unsubsampled_valid") is (not args.expect_smoke), audit
    for key in (
        "auto_gate_train_only_valid",
        "forced_spectral_train_only_valid",
        "global_partition_train_only_valid",
        "cache_starts_exact_valid",
        "action_norm_starts_hash_match",
        "checkpoint_provenance_valid",
    ):
        assert audit.get(key) is True, key

    if args.expect_smoke:
        assert int(audit.get("common_h10_anchor_count", 0)) > 0, audit
    else:
        train_seed_values = np.unique(np.load(root / "evaluation/sample_metrics.npz")["train_seed"])
        assert len(train_seed_values) >= 3, train_seed_values

    for horizon in ("1", "5", "10"):
        count = int(audit.get("horizon_anchor_counts", {}).get(horizon, 0))
        assert count > 0, audit.get("horizon_anchor_counts")

    bootstrap = list(csv.DictReader((root / "evaluation/bootstrap_summary.csv").open()))
    loss_kinds = {row.get("loss_kind") for row in bootstrap}
    assert "mean_trajectory" in loss_kinds, loss_kinds
    assert "terminal" in loss_kinds, loss_kinds

    metrics = np.load(root / "evaluation/sample_metrics.npz")
    h1_main = (metrics["horizon"] == 1) & (metrics["anchor_support"] == "horizon_valid")
    np.testing.assert_allclose(
        metrics["terminal_global_loss"][h1_main],
        metrics["global_loss"][h1_main],
    )

    for csv_name in (
        "episode_metrics.csv",
        "region_summary.csv",
        "weighted_summary.csv",
        "bootstrap_summary.csv",
        "common_h10_support_summary.csv",
    ):
        assert_finite_csv(root / "evaluation" / csv_name)

    runtime = audit.get("runtime", {})
    if runtime or not args.expect_smoke:
        assert runtime.get("python_executable"), runtime
        assert runtime.get("torch_version"), runtime
        assert runtime.get("git_commit"), runtime

    print(f"[verify] formal acceptance ok -> {root}")


if __name__ == "__main__":
    main()
