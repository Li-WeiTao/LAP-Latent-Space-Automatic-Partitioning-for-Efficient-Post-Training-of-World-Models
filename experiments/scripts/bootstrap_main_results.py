#!/usr/bin/env python3
"""Unified CPU bootstrap for main experiment success-rate tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Limit BLAS threading inside each worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_lib.loader import load_cell  # noqa: E402
from bootstrap_lib.resample import bootstrap_cell_with_contrasts  # noqa: E402
from bootstrap_lib.tex import render_tex_tables  # noqa: E402

DEFAULT_MODELS = ("lewm", "subjepa")
DEFAULT_TASKS = ("tworoom", "pusht", "reacher", "cube")
DEFAULT_HORIZONS = ("short", "long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "bootstrap_config.json")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/bootstrap_results"))
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", default="auto")
    parser.add_argument(
        "--resampling-unit",
        choices=("eval-block", "episode"),
        default="eval-block",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--horizons", nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-replicates", action="store_true")
    return parser.parse_args()


def git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_workers(spec: str) -> int:
    if spec == "auto":
        return max(1, min(16, os.cpu_count() or 1))
    return max(1, int(spec))


def _process_cell(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(payload["repo_root"])
    config = payload["config"]
    model = payload["model"]
    task = payload["task"]
    horizon = payload["horizon"]
    t0 = time.perf_counter()
    cell = load_cell(repo_root=repo_root, config=config, model=model, task=task, horizon=horizon)
    elapsed_load = time.perf_counter() - t0

    if cell.status == "pending":
        return {
            "model": model,
            "task": task,
            "horizon": horizon,
            "status": "pending",
            "summary_rows": [],
            "contrast_rows": [],
            "metadata": {
                "elapsed_sec": elapsed_load,
                "validation": cell.validation,
            },
        }

    t1 = time.perf_counter()
    results, contrasts = bootstrap_cell_with_contrasts(
        cell,
        n_bootstrap=payload["n_bootstrap"],
        seed=payload["seed"],
        batch_size=payload["batch_size"],
        resampling_unit=payload["resampling_unit"],
        save_draws=payload["save_replicates"],
    )
    elapsed_boot = time.perf_counter() - t1

    summary_rows: list[dict[str, Any]] = []
    for mid, res in results.items():
        method = cell.methods[mid]
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "horizon": horizon,
                "method": method.label,
                "method_id": mid,
                "point_estimate": res.point_estimate,
                "bootstrap_mean": res.bootstrap_mean,
                "bootstrap_std": res.bootstrap_std,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "n_train_seeds": res.n_train_seeds,
                "n_partition_seeds": res.n_partition_seeds,
                "n_eval_blocks": res.n_eval_blocks,
                "episodes_per_block": cell.episodes_per_block,
                "n_bootstrap": payload["n_bootstrap"],
                "resampling_unit": payload["resampling_unit"],
                "status": cell.status,
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    for c in contrasts:
        contrast_rows.append(
            {
                "model": model,
                "task": task,
                "horizon": horizon,
                "method_a": "Auto-LAP",
                "method_b": cell.methods[c.baseline_method].label,
                "baseline_method_id": c.baseline_method,
                "point_difference_pp": c.point_difference,
                "ci_low_pp": c.ci_low,
                "ci_high_pp": c.ci_high,
                "pr_delta_gt_zero": c.pr_gt_zero,
                "n_common_eval_blocks": c.n_common_blocks,
                "n_bootstrap": payload["n_bootstrap"],
                "resampling_unit": payload["resampling_unit"],
            }
        )

    files_read = sorted({f for m in cell.methods.values() for f in m.files_read})
    return {
        "model": model,
        "task": task,
        "horizon": horizon,
        "status": cell.status,
        "summary_rows": summary_rows,
        "contrast_rows": contrast_rows,
        "metadata": {
            "elapsed_sec": elapsed_load + elapsed_boot,
            "load_sec": elapsed_load,
            "bootstrap_sec": elapsed_boot,
            "gate_info": cell.gate_info,
            "validation": cell.validation,
            "reference_estimates": cell.reference_estimates,
            "files_read": files_read,
            "has_episode_data": cell.has_episode_data,
            "aggregation": {
                "train_seeds": list(cell.train_seeds),
                "partition_seeds": list(cell.partition_seeds),
                "eval_seeds": list(cell.eval_seeds),
                "goal_offset_steps": cell.goal_offset_steps,
                "partition_average_before_bootstrap": True,
                "autolap_deployment_only": True,
            },
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.smoke_test:
        args.n_bootstrap = min(args.n_bootstrap, 100)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cells = [
        (model, task, horizon)
        for model in args.models
        for task in args.tasks
        for horizon in args.horizons
    ]

    worker_payloads = [
        {
            "repo_root": str(repo_root),
            "config": config,
            "model": model,
            "task": task,
            "horizon": horizon,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "resampling_unit": args.resampling_unit,
            "save_replicates": args.save_replicates,
        }
        for model, task, horizon in cells
    ]

    t0 = time.perf_counter()
    n_workers = resolve_workers(args.workers)
    results: list[dict[str, Any]] = []

    if n_workers == 1 or len(cells) == 1:
        results = [_process_cell(p) for p in worker_payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_cell, p): p for p in worker_payloads}
            for fut in as_completed(futures):
                results.append(fut.result())

    total_sec = time.perf_counter() - t0
    results.sort(key=lambda r: (r["model"], r["task"], r["horizon"]))

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    cell_metadata: dict[str, Any] = {}
    pending: list[str] = []
    failures: list[str] = []

    for res in results:
        key = f"{res['model']}/{res['task']}/{res['horizon']}"
        cell_metadata[key] = res["metadata"]
        if res["status"] == "pending":
            pending.append(key)
            summary_rows.append(
                {
                    "model": res["model"],
                    "task": res["task"],
                    "horizon": res["horizon"],
                    "method": "Pending",
                    "point_estimate": "",
                    "bootstrap_mean": "",
                    "bootstrap_std": "",
                    "ci_low": "",
                    "ci_high": "",
                    "n_train_seeds": "",
                    "n_partition_seeds": "",
                    "n_eval_blocks": "",
                    "episodes_per_block": "",
                    "n_bootstrap": args.n_bootstrap,
                    "resampling_unit": args.resampling_unit,
                    "status": "pending",
                }
            )
            continue
        if res["status"] in {"failed", "incomplete"}:
            failures.append(key)
        summary_rows.extend(res["summary_rows"])
        contrast_rows.extend(res["contrast_rows"])

    summary_fields = [
        "model",
        "task",
        "horizon",
        "method",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_low",
        "ci_high",
        "n_train_seeds",
        "n_partition_seeds",
        "n_eval_blocks",
        "episodes_per_block",
        "n_bootstrap",
        "resampling_unit",
        "status",
    ]
    contrast_fields = [
        "model",
        "task",
        "horizon",
        "method_a",
        "method_b",
        "point_difference_pp",
        "ci_low_pp",
        "ci_high_pp",
        "pr_delta_gt_zero",
        "n_common_eval_blocks",
        "n_bootstrap",
        "resampling_unit",
    ]

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    write_csv(out_dir / "bootstrap_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "bootstrap_contrasts.csv", contrast_rows, contrast_fields)

    rows_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row["model"], row["task"], row["horizon"])
        rows_by_cell.setdefault(key, []).append(row)
    tex = render_tex_tables(rows_by_cell)
    (out_dir / "bootstrap_tables.tex").write_text(tex, encoding="utf-8")

    cmd = " ".join(sys.argv)
    metadata = {
        "git_commit": git_commit(repo_root),
        "command": cmd,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "batch_size": args.batch_size,
        "workers": n_workers,
        "resampling_unit": args.resampling_unit,
        "smoke_test": args.smoke_test,
        "total_elapsed_sec": total_sec,
        "cells": cell_metadata,
        "pending_cells": pending,
        "failed_or_incomplete_cells": failures,
        "episode_data_note": (
            "Block-level CSV/rates only; episode_successes used when present."
            if args.resampling_unit == "eval-block"
            else "Episode-level resampling enabled."
        ),
    }
    (out_dir / "bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(out_dir), "total_sec": total_sec, "pending": pending}, indent=2))

    if args.strict and (pending or failures):
        issues = pending + failures
        raise SystemExit(f"strict mode: unresolved cells: {issues}")


if __name__ == "__main__":
    main()
