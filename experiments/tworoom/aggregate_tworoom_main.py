#!/usr/bin/env python3
"""Rebuild the TwoRoom main-table inputs from committed per-run results.

The aggregation unit is the predictor fine-tuning seed.  For partitioned
methods, each fine-tuning-seed value first averages the 3 partition seeds x 5
paired evaluation seeds.  The plotted error bar is then the sample standard
deviation across fine-tuning seeds 0, 42, and 625.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = THIS_DIR / "results"
TRAIN_SEEDS = (0, 42, 625)
PARTITION_SEEDS = (0, 1, 2)
EVAL_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Method:
    method_id: str
    label: str
    source_scope: str


METHODS = (
    Method("baseline", r"Official\nbaseline", "Original LeWM baseline; no post-training seed"),
    Method("joint3", r"Joint-Continue\n3ep", "Mean over eval seeds 0-4"),
    Method("globalft50", r"Global-FT\n50ep", "Mean over eval seeds 0-4"),
    Method("random", r"Random-Voronoi\nK3-50", "Mean over 3 partition seeds x 5 eval seeds"),
    Method("kmeans", r"K-means++\nK3-50", "Mean over 3 outer seeds x 5 eval seeds"),
    Method("spectral", r"Spectral\nK3-50", "Mean over 3 partition seeds x 5 eval seeds"),
    Method("rooms3", r"Human partition\nrooms3-50", "Mean over eval seeds 0-4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=("short", "long"), default="long")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Compare recomputed seed rows with --seed-csv without modifying files.",
    )
    parser.add_argument(
        "--spectral-summary",
        type=Path,
        default=None,
        help=(
            "Optional stability_summary.json from a fresh spectral run. Its "
            "artifact basenames replace the migrated canonical spectral tags."
        ),
    )
    args = parser.parse_args()
    asset_dir = THIS_DIR / "assets" / f"{args.horizon}_horizon_metrics"
    stem = f"tworoom_{args.horizon}_horizon"
    args.seed_csv = args.seed_csv or asset_dir / f"{stem}_method_seeds.csv"
    args.summary_csv = args.summary_csv or asset_dir / f"{stem}_method_summary_from_results.csv"
    args.audit_json = args.audit_json or asset_dir / f"{stem}_result_audit.json"
    return args


def path_for(
    results: Path,
    horizon: str,
    method: str,
    train_seed: int | None,
    eval_seed: int,
    partition_seed: int | None = None,
    spectral_tags: dict[int, str] | None = None,
) -> Path:
    if horizon not in {"short", "long"}:
        raise ValueError(horizon)
    if method == "baseline":
        name = (
            f"tworoom_success_rate_baseline_seed{eval_seed}"
            if horizon == "short"
            else f"tworoom_success_rate_baseline_exp6_seed{eval_seed}"
        )
    elif method == "joint3":
        name = (
            f"tworoom_success_rate_joint_continue_3ep_trainseed{train_seed}_"
            f"{horizon}_evalseed{eval_seed}"
        )
    elif method == "globalft50":
        if horizon == "long" and train_seed == 42:
            name = f"tworoom_success_rate_global_ft_50ep_exp6_seed{eval_seed}"
        elif horizon == "short":
            name = (
                f"tworoom_success_rate_global_ft_50ep_trainseed{train_seed}_"
                f"short_evalseed{eval_seed}"
            )
        else:
            name = (
                f"tworoom_success_rate_global_ft_50ep_trainseed{train_seed}_"
                f"exp6_seed{eval_seed}"
            )
    elif method == "random":
        name = (
            f"tworoom_success_rate_random_voronoi_k3_seed{partition_seed}_"
            f"trainseed{train_seed}_{horizon}_evalseed{eval_seed}"
        )
    elif method == "kmeans":
        prefix = (
            f"tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_"
            f"outer{partition_seed}"
        )
        if horizon == "short":
            name = f"{prefix}_trainseed{train_seed}_short_evalseed{eval_seed}"
        else:
            name = (
                f"{prefix}_seed{eval_seed}"
                if train_seed == 42
                else f"{prefix}_trainseed{train_seed}_evalseed{eval_seed}"
            )
    elif method == "spectral":
        tag = (
            spectral_tags[int(partition_seed)]
            if spectral_tags is not None
            else f"spectral_M20000_k30_P16_seed{partition_seed}"
        )
        name = (
            "tworoom_success_rate_latent_spectral_"
            f"{tag}_"
            f"trainseed{train_seed}_mpc_"
            f"{'short_' if horizon == 'short' else ''}evalseed{eval_seed}"
        )
    elif method == "rooms3":
        if horizon == "short" and train_seed == 42:
            name = f"tworoom_success_rate_rooms3_50ep_seed{eval_seed}"
        elif train_seed == 42:
            name = f"tworoom_success_rate_rooms3_exp6_50ep_seed{eval_seed}"
        else:
            name = (
                f"tworoom_success_rate_trainseed{train_seed}_rooms3_50ep_"
                f"{horizon}_evalseed{eval_seed}"
            )
    else:  # pragma: no cover - guarded by the fixed method table
        raise KeyError(method)
    return results / name / "results.json"


def starts_sha256(starts: Iterable[int]) -> str:
    payload = json.dumps([int(value) for value in starts], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_run(path: Path, eval_seed: int, horizon: str) -> tuple[float, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "seed": eval_seed,
        "num_eval": 50,
        "goal_offset_steps": 25 if horizon == "short" else 50,
        "eval_budget": 50,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{path}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    starts = payload.get("eval_start_indices")
    if not isinstance(starts, list) or len(starts) != 50:
        raise ValueError(f"{path}: expected 50 eval_start_indices")
    rate = float(payload["metrics"]["success_rate"])
    return rate, starts_sha256(starts)


def load_spectral_tags(summary_path: Path | None) -> dict[int, str] | None:
    if summary_path is None:
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    by_seed = payload.get("artifacts_by_seed", {})
    tags = {seed: Path(by_seed[str(seed)]).name for seed in PARTITION_SEEDS}
    if len(set(tags.values())) != len(PARTITION_SEEDS):
        raise ValueError(f"{summary_path}: spectral artifact tags are not unique")
    return tags


def collect_rows(
    results: Path,
    horizon: str,
    spectral_tags: dict[int, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    pairing: dict[int, set[str]] = {seed: set() for seed in EVAL_SEEDS}
    files_read = 0

    baseline_values = []
    for eval_seed in EVAL_SEEDS:
        rate, digest = load_run(
            path_for(results, horizon, "baseline", None, eval_seed),
            eval_seed,
            horizon,
        )
        baseline_values.append(rate)
        pairing[eval_seed].add(digest)
        files_read += 1
    baseline = METHODS[0]
    rows.append(
        {
            "method_id": baseline.method_id,
            "method_label": baseline.label,
            "seed_type": "official",
            "seed": "official",
            "value_percent": statistics.mean(baseline_values),
            "source_scope": baseline.source_scope,
        }
    )

    for method in METHODS[1:]:
        for train_seed in TRAIN_SEEDS:
            values = []
            partition_seeds: tuple[int | None, ...] = (
                PARTITION_SEEDS
                if method.method_id in {"random", "kmeans", "spectral"}
                else (None,)
            )
            for partition_seed in partition_seeds:
                for eval_seed in EVAL_SEEDS:
                    path = path_for(
                        results,
                        horizon,
                        method.method_id,
                        train_seed,
                        eval_seed,
                        partition_seed,
                        spectral_tags,
                    )
                    rate, digest = load_run(path, eval_seed, horizon)
                    values.append(rate)
                    pairing[eval_seed].add(digest)
                    files_read += 1
            rows.append(
                {
                    "method_id": method.method_id,
                    "method_label": method.label,
                    "seed_type": "fine_tuning",
                    "seed": train_seed,
                    "value_percent": statistics.mean(values),
                    "source_scope": method.source_scope,
                }
            )

    pairing_failures = {
        str(seed): sorted(digests)
        for seed, digests in pairing.items()
        if len(digests) != 1
    }
    if pairing_failures:
        raise ValueError(
            "Evaluation starts are not paired across methods: "
            + json.dumps(pairing_failures, indent=2)
        )
    audit = {
        "files_read": files_read,
        "expected_files": 185,
        "train_seeds": list(TRAIN_SEEDS),
        "partition_seeds": list(PARTITION_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "eval_start_indices_sha256_by_seed": {
            str(seed): next(iter(pairing[seed])) for seed in EVAL_SEEDS
        },
        "aggregation_unit": "fine_tuning_seed",
        "horizon": horizon,
        "goal_offset_steps": 25 if horizon == "short" else 50,
        "partition_method_inner_average": "3 partition seeds x 5 eval seeds",
        "spectral_artifact_tags": spectral_tags,
    }
    if files_read != audit["expected_files"]:
        raise AssertionError(f"Expected 185 result files, read {files_read}")
    return rows, audit


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method_id"] == method.method_id]
        values = [float(row["value_percent"]) for row in selected]
        output.append(
            {
                "method_id": method.method_id,
                "method_label": method.label,
                "seed_type": selected[0]["seed_type"],
                "n_method_seeds": len(values),
                "mean_percent": statistics.mean(values),
                "sd_percent": statistics.stdev(values) if len(values) > 1 else "",
            }
        )
    return output


def normalized_seed_key(row: dict[str, str]) -> tuple[str, str]:
    return row["method_id"], str(row["seed"])


def check_existing(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        existing = list(csv.DictReader(handle))
    expected_by_key = {
        (str(row["method_id"]), str(row["seed"])): row for row in rows
    }
    existing_by_key = {normalized_seed_key(row): row for row in existing}
    if set(expected_by_key) != set(existing_by_key):
        missing = sorted(set(expected_by_key) - set(existing_by_key))
        extra = sorted(set(existing_by_key) - set(expected_by_key))
        raise ValueError(f"{path}: key mismatch; missing={missing}, extra={extra}")
    for key, expected in expected_by_key.items():
        actual = existing_by_key[key]
        if not math.isclose(
            float(actual["value_percent"]),
            float(expected["value_percent"]),
            rel_tol=0.0,
            abs_tol=5e-9,
        ):
            raise ValueError(
                f"{path}: {key} value is {actual['value_percent']}, "
                f"recomputed {expected['value_percent']}"
            )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows, audit = collect_rows(
        args.results_root,
        args.horizon,
        spectral_tags=load_spectral_tags(args.spectral_summary),
    )
    summaries = summary_rows(rows)
    audit["summary"] = summaries
    if args.check_existing:
        check_existing(args.seed_csv, rows)
        print(
            f"PASS: {audit['files_read']} result JSONs reproduce {args.seed_csv}; "
            "all evaluation starts are paired."
        )
        return
    write_csv(args.seed_csv, rows)
    write_csv(args.summary_csv, summaries)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.seed_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.audit_json}")


if __name__ == "__main__":
    main()
