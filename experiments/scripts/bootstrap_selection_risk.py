#!/usr/bin/env python3
"""Post-hoc uncertainty-aware branch-selection analysis (appendix diagnostics)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_lib.loader import load_cell  # noqa: E402
from bootstrap_lib.resample import bootstrap_cell_with_contrasts  # noqa: E402
from bootstrap_lib.selection_risk import (  # noqa: E402
    MARGIN_PP_DEFAULT,
    SelectionRiskSummary,
    aggregate_selection_risk_summary,
    render_selection_risk_report,
    selection_delta_draws,
    summarize_selection_risk,
)

DEFAULT_MODELS = ("lewm", "subjepa")
DEFAULT_TASKS = ("tworoom", "pusht", "reacher", "cube")
DEFAULT_SEED = 20260818
DEFAULT_N_BOOTSTRAP = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "bootstrap_config.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/bootstrap_results"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--resampling-unit",
        choices=("eval-block", "episode"),
        default="eval-block",
    )
    parser.add_argument("--margin-pp", type=float, default=MARGIN_PP_DEFAULT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--horizons", nargs="+", default=["long"])
    parser.add_argument("--save-draws", action="store_true", default=True)
    parser.add_argument("--no-save-draws", dest="save_draws", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_cell(
    *,
    repo_root: Path,
    config: dict,
    model: str,
    task: str,
    horizon: str,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
    resampling_unit: str,
    margin_pp: float,
) -> tuple[SelectionRiskSummary | None, np.ndarray | None, dict]:
    cell = load_cell(
        repo_root=repo_root,
        config=config,
        model=model,
        task=task,
        horizon=horizon,
    )
    meta = {
        "model": model,
        "task": task,
        "horizon": horizon,
        "status": cell.status,
    }
    if cell.status != "ok":
        return None, None, meta

    results, _contrasts = bootstrap_cell_with_contrasts(
        cell,
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=batch_size,
        resampling_unit=resampling_unit,
        save_draws=True,
    )
    raw_draws = {mid: res.draws for mid, res in results.items() if res.draws is not None}
    delta = selection_delta_draws(cell, raw_draws)
    summary = summarize_selection_risk(
        delta,
        margin_pp=margin_pp,
        model=model,
        task=task,
        horizon=horizon,
        gate_info=cell.gate_info,
        cell=cell,
        n_bootstrap=n_bootstrap,
        rng_seed=seed,
    )
    meta["gate_info"] = cell.gate_info
    return summary, delta, meta


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.smoke_test:
        args.n_bootstrap = min(args.n_bootstrap, 256)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SelectionRiskSummary] = []
    draw_payload: dict[str, np.ndarray] = {}
    draw_meta: dict[str, dict] = {}

    for model in args.models:
        for task in args.tasks:
            for horizon in args.horizons:
                summary, delta, meta = analyze_cell(
                    repo_root=repo_root,
                    config=config,
                    model=model,
                    task=task,
                    horizon=horizon,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    resampling_unit=args.resampling_unit,
                    margin_pp=args.margin_pp,
                )
                key = f"{model}_{task}_{horizon}"
                draw_meta[key] = meta
                if summary is None or delta is None:
                    continue
                summaries.append(summary)
                if args.save_draws:
                    draw_payload[key] = delta

    rows = [s.to_row() for s in summaries]
    fieldnames = [
        "model",
        "task",
        "horizon",
        "pilot_status",
        "selected_branch",
        "rejected_branch",
        "selected_aggregation",
        "rejected_aggregation",
        "point_delta_pp",
        "bootstrap_mean_delta_pp",
        "ci_low_pp",
        "ci_high_pp",
        "point_estimate_favors_selected",
        "p_harm",
        "eol_pp",
        "practical_eol_pp",
        "margin_pp",
        "classification",
        "n_bootstrap",
        "rng_seed",
        "gate_source",
    ]
    write_csv(out_dir / "selection_risk.csv", rows, fieldnames)

    agg = aggregate_selection_risk_summary(summaries, margin_pp=args.margin_pp)
    agg["command"] = " ".join(sys.argv)
    agg["n_bootstrap"] = args.n_bootstrap
    agg["rng_seed"] = args.seed
    agg["resampling_unit"] = args.resampling_unit
    (out_dir / "selection_risk_summary.json").write_text(
        json.dumps(agg, indent=2) + "\n",
        encoding="utf-8",
    )

    report = render_selection_risk_report(summaries, agg)
    (out_dir / "selection_risk_report.md").write_text(report, encoding="utf-8")

    if args.save_draws and draw_payload:
        np.savez_compressed(
            out_dir / "selection_risk_draws.npz",
            **draw_payload,
            metadata=json.dumps(
                {
                    "keys": sorted(draw_payload.keys()),
                    "cells": draw_meta,
                    "margin_pp": args.margin_pp,
                    "margin_status": agg["margin_status"],
                    "n_bootstrap": args.n_bootstrap,
                    "rng_seed": args.seed,
                    "resampling_unit": args.resampling_unit,
                }
            ),
        )

    print(
        json.dumps(
            {
                "output_dir": str(out_dir.relative_to(repo_root)),
                "rows": len(rows),
                "overall_agreement": (
                    f"{agg['overall_point_estimate_agreement_count']}/"
                    f"{agg['overall_point_estimate_total']}"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
