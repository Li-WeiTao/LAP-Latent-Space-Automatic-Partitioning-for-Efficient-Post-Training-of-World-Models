#!/usr/bin/env python3
"""Paired block bootstrap on Sub-JEPA TwoRoom matrix success rates (LeWM-style)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_SEEDS = (0, 42, 625)
PARTITION_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2, 3, 4)


def load_rate(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["metrics"]["success_rate"])


def method_label(method: str) -> str:
    if method == "global":
        return "Global-FT50"
    if method == "kmeanspp":
        return "K-means++ K3-50"
    if method == "spectral":
        return "Spectral K3-50"
    raise ValueError(method)


def paired_success_bootstrap(
    blocks: list[tuple[float, float]],
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if not blocks:
        raise ValueError("empty bootstrap blocks")
    rng = np.random.default_rng(seed)
    reference = np.asarray([left for left, _ in blocks], dtype=np.float64)
    candidate = np.asarray([right for _, right in blocks], dtype=np.float64)
    count = len(blocks)
    deltas = np.empty(reps, dtype=np.float64)
    for draw in range(reps):
        chosen = rng.integers(0, count, size=count)
        deltas[draw] = candidate[chosen].mean() - reference[chosen].mean()
    return {
        "estimate_delta_percent": float(candidate.mean() - reference.mean()),
        "ci_low_percent": float(np.quantile(deltas, 0.025)),
        "ci_high_percent": float(np.quantile(deltas, 0.975)),
    }


def collect_blocks(
    eval_root: Path,
    *,
    method: str,
    reference: str,
) -> list[tuple[float, float]]:
    blocks: list[tuple[float, float]] = []
    if method == "global":
        for train_seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                cand = (
                    eval_root
                    / "global"
                    / f"train{train_seed}"
                    / f"eval{eval_seed}"
                    / "results.json"
                )
                ref = eval_root / "official" / f"eval{eval_seed}" / "results.json"
                blocks.append((load_rate(ref), load_rate(cand)))
        return blocks

    for train_seed in TRAIN_SEEDS:
        for eval_seed in EVAL_SEEDS:
            ref_values = [
                load_rate(
                    eval_root
                    / reference
                    / f"partition{pseed}_train{train_seed}"
                    / f"eval{eval_seed}"
                    / "results.json"
                )
                for pseed in PARTITION_SEEDS
            ]
            cand_values = [
                load_rate(
                    eval_root
                    / method
                    / f"partition{pseed}_train{train_seed}"
                    / f"eval{eval_seed}"
                    / "results.json"
                )
                for pseed in PARTITION_SEEDS
            ]
            blocks.append(
                (statistics.fmean(ref_values), statistics.fmean(cand_values))
            )
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    comparisons: list[dict[str, object]] = []
    for method, reference, label in (
        ("global", "official", "Global-FT50 vs Official baseline"),
        ("kmeanspp", "global", "K-means++ K3-50 vs Global-FT50"),
        ("spectral", "global", "Spectral K3-50 vs Global-FT50"),
    ):
        blocks = collect_blocks(args.eval_root, method=method, reference=reference)
        seed = args.seed + hash(method) % 10_000
        comparisons.append(
            {
                "comparison": label,
                "blocks": len(blocks),
                **paired_success_bootstrap(blocks, reps=args.reps, seed=seed),
            }
        )

    payload = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_matrix_paired_bootstrap",
        "eval_root": str(args.eval_root),
        "bootstrap_reps": args.reps,
        "bootstrap_seed": args.seed,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
