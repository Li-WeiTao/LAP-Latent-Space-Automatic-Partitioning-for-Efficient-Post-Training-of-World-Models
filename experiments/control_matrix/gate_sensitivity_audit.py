#!/usr/bin/env python3
"""Gate-only one-at-a-time sensitivity audit for LAP spectral auto-gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWOROOM_DIR = PROJECT_ROOT / "experiments" / "tworoom"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TWOROOM_DIR))

from experiments.control_matrix.fit_partition import (  # noqa: E402
    episode_ids,
    load_unique_latents,
)
from experiments.control_matrix.gate_audit_lib import (  # noqa: E402
    AUDIT_SAMPLE_SEED,
    AUDIT_SAMPLE_SIZE,
    B_SEED_PREFIXES,
    MAX_EIGENVALUES,
    OATScenario,
    PAIR_SPECS,
    SpectrumCache,
    atomic_write_json,
    atomic_write_text,
    NeighborDrawCache,
    build_oat_scenarios,
    choose_audit_rows,
    compare_to_manifest,
    config_from_manifest_dict,
    enumerate_seed_subsets,
    compute_landmark_spectrum,
    evaluate_config,
    fit_audit_labels,
    git_info,
    hungarian_align,
    margins_from_result,
    resolve_pair_inputs,
    result_row,
    sha256_file,
    spectrum_cache_key,
)
from lap.partition.gate import SpectralGateConfig  # noqa: E402
from lap.partition.landmark import _sample_landmarks, _zscore_l2  # noqa: E402
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402


def build_neighbor_cache(gpu_id: int, query_chunk: int) -> NeighborDrawCache:
    if gpu_id < 0:
        return NeighborDrawCache()
    search = lambda landmarks, max_k: exact_cosine_knn_torch(
        landmarks, max_k, gpu_id=gpu_id, query_chunk=query_chunk
    )
    return NeighborDrawCache(neighbor_search=search)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/control_matrix/assets/gate_sensitivity",
    )
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=-1, help="Use GPU exact kNN when >= 0.")
    parser.add_argument("--query-chunk", type=int, default=2048)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


OAT_FIELDS = [
    "model",
    "task",
    "varied_factor",
    "varied_value",
    "K",
    "rho",
    "c_pert",
    "c_bg",
    "num_landmarks",
    "B",
    "diagnostic_seeds",
    "nominal_knn",
    "perturb_knn",
    "candidate_gap_min",
    "perturbation_threshold_max",
    "S",
    "R_K",
    "T_bg",
    "safety_margin",
    "prominence_margin",
    "safety_pass",
    "background_pass",
    "decision",
    "reason",
    "agreement_with_baseline",
    "elapsed_sec",
    "cache_hit",
]


def decision_agreement_rows(
    oat_rows: list[dict],
    scenarios: list[OATScenario],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_by_key = {
        (row["model"], row["task"]): row["decision"]
        for row in oat_rows
        if row["varied_factor"] == "baseline"
    }
    scenario_lookup = {(s.varied_factor, s.varied_value): s for s in scenarios}
    by_pair_factor: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in oat_rows:
        by_pair_factor[(row["model"], row["task"], row["varied_factor"])].append(row)

    summary_rows: list[dict[str, object]] = []
    non_k_agree = 0
    non_k_total = 0
    for (model, task, factor), rows in sorted(by_pair_factor.items()):
        if factor == "baseline":
            continue
        baseline_decision = baseline_by_key[(model, task)]
        alts = [
            row
            for row in rows
            if (row["varied_factor"], row["varied_value"]) in scenario_lookup
            and not scenario_lookup[(row["varied_factor"], row["varied_value"])].is_baseline
        ]
        if not alts:
            continue
        matches = sum(1 for row in alts if row["decision"] == baseline_decision)
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "varied_factor": factor,
                "n_alternatives": len(alts),
                "n_agree": matches,
                "agreement_rate": matches / len(alts),
                "baseline_decision": baseline_decision,
            }
        )
        if factor != "K":
            non_k_agree += matches
            non_k_total += len(alts)

    overall = {
        "non_k_overall_agreement_rate": (
            None if non_k_total == 0 else non_k_agree / non_k_total
        ),
        "non_k_overall_agreement": f"{non_k_agree}/{non_k_total}",
    }
    return summary_rows, overall


def draw_pass_rows(
    *,
    model: str,
    task: str,
    spectra_bank: dict[int, dict[int, np.ndarray]],
    baseline: SpectralGateConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    universe = tuple(range(10))
    for b in (3, 5, 10):
        subsets = enumerate_seed_subsets(
            universe=universe, subset_size=b, required_seed=baseline.deployment_seed
        )
        passes = 0
        safety_margins: list[float] = []
        prominence_margins: list[float] = []
        for seeds in subsets:
            cfg = replace(baseline, diagnostic_seeds=seeds)
            result = evaluate_config(cfg, spectra_bank)
            margins = margins_from_result(result)
            if result.use_partition:
                passes += 1
            if margins["safety_margin"] is not None:
                safety_margins.append(float(margins["safety_margin"]))
            if margins["prominence_margin"] is not None:
                prominence_margins.append(float(margins["prominence_margin"]))
        rows.append(
            {
                "model": model,
                "task": task,
                "B": b,
                "n_subsets": len(subsets),
                "pass_frequency": passes / len(subsets),
                "min_safety_margin": min(safety_margins) if safety_margins else None,
                "median_safety_margin": float(np.median(safety_margins))
                if safety_margins
                else None,
                "min_prominence_margin": min(prominence_margins)
                if prominence_margins
                else None,
                "median_prominence_margin": float(np.median(prominence_margins))
                if prominence_margins
                else None,
                "note": "draw-subset sensitivity diagnostic; gate definition unchanged",
            }
        )
    return rows


def audit_pair(
    repo_root: Path,
    output_dir: Path,
    *,
    spec,
    smoke_test: bool,
    log_handle,
    gpu_id: int = -1,
    query_chunk: int = 2048,
) -> dict[str, object]:
    resolved, issues = resolve_pair_inputs(repo_root, spec)
    pair_log: dict[str, object] = {"pair": spec.key, "issues": issues}
    if resolved is None:
        pair_log["status"] = "missing_inputs"
        log_handle.write(json.dumps(pair_log) + "\n")
        return pair_log

    pair_out = output_dir / spec.key
    pair_out.mkdir(parents=True, exist_ok=True)
    cache = SpectrumCache(pair_out / "spectrum_cache")
    cache_hash = sha256_file(resolved.latent_cache)

    latents, sample_ids, cache_stats = load_unique_latents(resolved.latent_cache, frameskip=5)
    groups = episode_ids(resolved.data_file, sample_ids, "auto")
    transformed, _, _ = _zscore_l2(latents)

    baseline = resolved.baseline_config
    if smoke_test:
        baseline = replace(
            baseline,
            num_landmarks=min(baseline.num_landmarks, 2000),
            diagnostic_seeds=(0, 1, 2),
        )

    scenarios = build_oat_scenarios(baseline)
    if smoke_test:
        scenarios = [s for s in scenarios if s.varied_factor in {"baseline", "rho", "K", "B"}][:8]

    all_knns: set[int] = set()
    all_requests: set[tuple[int, int, int]] = set()
    for scenario in scenarios:
        all_knns.update(scenario.config.graph_knn_values)
        for seed in set(scenario.config.diagnostic_seeds) | set(range(10)):
            for knn in scenario.config.graph_knn_values:
                all_requests.add((scenario.config.num_landmarks, seed, knn))
    neighbor_cache = build_neighbor_cache(gpu_id, query_chunk)
    max_k = max(all_knns)

    master_bank: dict[int, dict[int, np.ndarray]] = {}
    for m, seed, knn in sorted(all_requests):
        key = spectrum_cache_key(
            cache_hash=cache_hash,
            model=spec.model,
            task=spec.task,
            num_landmarks=m,
            seed=seed,
            knn=knn,
        )
        values = cache.get(key)
        if values is None:
            values = compute_landmark_spectrum(
                transformed,
                group_ids=groups,
                num_landmarks=m,
                seed=seed,
                knn=knn,
                count=MAX_EIGENVALUES,
                eig_seed=seed * 1000 + knn,
                config=baseline,
                neighbor_cache=neighbor_cache,
                max_k=max_k,
            )
            cache.put(key, values)
        master_bank.setdefault(seed, {})[knn] = values

    spectra_bank = {
        seed: {knn: master_bank[seed][knn] for knn in baseline.graph_knn_values}
        for seed in set(baseline.diagnostic_seeds) | set(range(10))
        if seed in master_bank
    }

    baseline_result = evaluate_config(baseline, spectra_bank)
    repro_issues = compare_to_manifest(baseline_result, resolved.manifest_gate)
    pair_log["baseline_reproduction_issues"] = repro_issues
    if repro_issues and not smoke_test:
        pair_log["status"] = "baseline_reproduction_failed"
        atomic_write_json(pair_out / "baseline_reproduction_issues.json", {"issues": repro_issues})
        log_handle.write(json.dumps(pair_log) + "\n")
        return pair_log

    baseline_decision = baseline_result.selected_method
    baseline_margins = margins_from_result(baseline_result)
    baseline_margin_row = {
        "model": spec.model,
        "task": spec.task,
        "decision": baseline_decision,
        "reason": baseline_result.reason,
        **baseline_margins,
        "candidate_gap_min": baseline_result.candidate_gap_min,
        "perturbation_threshold_max": baseline_result.perturbation_threshold_max,
    }

    # Reference partition labels for ARI.
    ref_landmarks = _sample_landmarks(
        len(transformed), baseline.num_landmarks, baseline.deployment_seed, groups
    )
    audit_rows = choose_audit_rows(
        len(transformed), ref_landmarks, size=min(AUDIT_SAMPLE_SIZE, len(transformed) // 4), seed=AUDIT_SAMPLE_SEED
    )
    reference_labels = fit_audit_labels(
        transformed,
        audit_rows,
        num_clusters=baseline.num_regions,
        num_landmarks=baseline.num_landmarks,
        knn=baseline.nominal_knn,
        seed=baseline.deployment_seed,
        group_ids=groups,
        cpu_threads=baseline.cpu_threads,
    )

    oat_rows: list[dict] = []
    partition_rows: list[dict] = []
    for scenario in scenarios:
        t0 = time.perf_counter()
        cache_hit = True
        result = evaluate_config(scenario.config, master_bank)
        elapsed = time.perf_counter() - t0
        oat_rows.append(
            result_row(
                model=spec.model,
                task=spec.task,
                scenario=scenario,
                result=result,
                baseline_decision=baseline_decision,
                elapsed_sec=elapsed,
                cache_hit=cache_hit,
            )
        )

        if scenario.needs_partition_ari and not scenario.is_baseline:
            labels = fit_audit_labels(
                transformed,
                audit_rows,
                num_clusters=scenario.config.num_regions,
                num_landmarks=scenario.config.num_landmarks,
                knn=scenario.config.nominal_knn,
                seed=scenario.config.deployment_seed,
                group_ids=groups,
                cpu_threads=scenario.config.cpu_threads,
            )
            aligned = hungarian_align(reference_labels, labels, scenario.config.num_regions)
            ari = float(adjusted_rand_score(reference_labels, aligned))
            partition_rows.append(
                {
                    "model": spec.model,
                    "task": spec.task,
                    "varied_factor": scenario.varied_factor,
                    "varied_value": scenario.varied_value,
                    "K": scenario.config.num_regions,
                    "num_landmarks": scenario.config.num_landmarks,
                    "nominal_knn": scenario.config.nominal_knn,
                    "adjusted_rand_index": ari,
                    "deployed": bool(result.use_partition),
                    "note": "diagnosis-only candidate when deployed=false",
                }
            )

    draw_rows = draw_pass_rows(
        model=spec.model,
        task=spec.task,
        spectra_bank=spectra_bank,
        baseline=baseline,
    )

    decision_rows, overall_agreement = decision_agreement_rows(oat_rows, scenarios)

    pair_log.update(
        {
            "status": "ok",
            "cache_hash": cache_hash,
            "baseline_margin_row": baseline_margin_row,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "overall_agreement": overall_agreement,
        }
    )
    atomic_write_json(pair_out / "pair_summary.json", pair_log)
    return {
        **pair_log,
        "baseline_margin_row": baseline_margin_row,
        "oat_rows": oat_rows,
        "decision_rows": decision_rows,
        "draw_rows": draw_rows,
        "partition_rows": partition_rows,
        "overall_agreement": overall_agreement,
    }


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# Gate-only sensitivity audit",
        "",
        "This audit reuses frozen latent caches and recomputes kNN landmark graphs,",
        "spectra, and gate arithmetic only. **No predictors were trained** and **no",
        "planning / MPC evaluation** was run. Thresholds were not adjusted post hoc.",
        "",
        "## Baseline margins",
        "",
        "| Model | Task | Decision | Safety margin | Prominence margin |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary.get("baseline_rows", []):
        sm = row.get("safety_margin")
        pm = row.get("prominence_margin")
        lines.append(
            f"| {row['model']} | {row['task']} | {row['decision']} | "
            f"{'' if sm is None else f'{sm:.4f}'} | "
            f"{'' if pm is None else f'{pm:.4f}'} |"
        )

    lines.extend(["", "## Non-K OAT decision agreement", ""])
    lines.append(f"Overall agreement (excludes K sweep): **{summary.get('non_k_overall_agreement')}**")
    lines.append("")
    for row in summary.get("decision_rows", []):
        lines.append(
            f"- {row['model']}/{row['task']} {row['varied_factor']}: "
            f"{row['n_agree']}/{row['n_alternatives']} "
            f"({100 * float(row['agreement_rate']):.1f}%) vs baseline={row['baseline_decision']}"
        )

    lines.extend(["", "## Draw-subset pass frequency (seeds 0–9)", ""])
    for row in summary.get("draw_rows", []):
        lines.append(
            f"- {row['model']}/{row['task']} B={row['B']}: pass={row['pass_frequency']:.3f}, "
            f"min safety margin={row.get('min_safety_margin')}, "
            f"min prominence margin={row.get('min_prominence_margin')}"
        )

    lines.extend(["", "## Partition stability (graph/landmark-changing OAT only)", ""])
    if summary.get("partition_rows"):
        lines.append(
            f"Mean ARI={summary.get('partition_mean_ari'):.4f}, "
            f"min ARI={summary.get('partition_min_ari'):.4f} "
            f"({len(summary['partition_rows'])} comparisons)"
        )
    else:
        lines.append("_No graph-changing comparisons completed._")

    lines.extend(["", "## Boundary / abstention flips", ""])
    if summary.get("boundary_cases"):
        for case in summary["boundary_cases"]:
            lines.append(f"- {case}")
    else:
        lines.append("_No non-K OAT flips observed among completed pairs._")

    lines.extend(
        [
            "",
            "## Why no predictor training?",
            "",
            "The spectral gate consumes only frozen state latents. All sweeps re-evaluate",
            "cached spectra or rebuild kNN graphs; partition labels are diagnosis-only.",
            "",
            "## What this audit cannot prove",
            "",
            "- Planning performance under alternate K or flipped gate decisions.",
            "- Sub-JEPA pairs without latent caches (preflight lists exact missing paths).",
            "- Draw-subset diagnostics do not modify the predeclared gate definition.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paper_table(baseline_rows: list[dict]) -> str:
    lines = [
        "% Gate sensitivity audit (gate-only; no predictor training).",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Task & Safety margin & Prominence margin & Decision & Reason \\\\",
        "\\midrule",
    ]
    for row in baseline_rows:
        sm = row.get("safety_margin")
        pm = row.get("prominence_margin")
        lines.append(
            f"{row['model']} & {row['task']} & "
            f"{'' if sm is None else f'{sm:.3f}'} & "
            f"{'' if pm is None else f'{pm:.3f}'} & "
            f"{row['decision']} & {row['reason']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "audit.log"

    if args.pairs == "all":
        specs = PAIR_SPECS
    else:
        wanted = {part.strip() for part in args.pairs.split(",") if part.strip()}
        specs = tuple(s for s in PAIR_SPECS if s.key in wanted or s.task in wanted)

    preflight_rows: list[dict] = []
    for spec in PAIR_SPECS:
        resolved, issues = resolve_pair_inputs(repo_root, spec)
        row = {
            "model": spec.model,
            "task": spec.task,
            "ready": resolved is not None,
            "issues": issues,
            "manifest": spec.manifest_rel,
        }
        if resolved is None:
            row["run_command"] = (
                f"bash experiments/control_matrix/scripts/run_gate_sensitivity_audit.sh "
                f"--pairs {spec.key}"
            )
            row["note"] = (
                "Resolve missing latent cache / data_file paths listed in issues, then rerun."
            )
        preflight_rows.append(row)
    atomic_write_json(output_dir / "preflight_report.json", {"pairs": preflight_rows})

    all_baseline: list[dict] = []
    all_oat: list[dict] = []
    all_decision: list[dict] = []
    all_draw: list[dict] = []
    all_partition: list[dict] = []
    pair_status: list[dict] = []
    boundary_cases: list[str] = []
    non_k_agree = 0
    non_k_total = 0

    with log_path.open("a", encoding="utf-8") as log_handle:
        for spec in specs:
            result = audit_pair(
                repo_root,
                output_dir,
                spec=spec,
                smoke_test=args.smoke_test,
                log_handle=log_handle,
                gpu_id=args.gpu_id,
                query_chunk=args.query_chunk,
            )
            pair_status.append(result)
            if result.get("status") != "ok":
                continue
            all_baseline.append(result["baseline_margin_row"])
            all_oat.extend(result["oat_rows"])
            all_decision.extend(result["decision_rows"])
            all_draw.extend(result["draw_rows"])
            all_partition.extend(result["partition_rows"])
            overall = result.get("overall_agreement", {})
            if overall.get("non_k_overall_agreement"):
                agree, total = str(overall["non_k_overall_agreement"]).split("/")
                non_k_agree += int(agree)
                non_k_total += int(total)

            for row in result["oat_rows"]:
                if row["varied_factor"] != "baseline" and not row["agreement_with_baseline"]:
                    boundary_cases.append(
                        f"{row['model']}/{row['task']} {row['varied_factor']}={row['varied_value']} "
                        f"flipped to {row['decision']} (safety_margin={row['safety_margin']}, "
                        f"prominence_margin={row['prominence_margin']})"
                    )

    write_csv(
        output_dir / "baseline_margins.csv",
        all_baseline,
        [
            "model",
            "task",
            "decision",
            "reason",
            "S",
            "rho",
            "R_K",
            "T_bg",
            "safety_margin",
            "prominence_margin",
            "safety_pass",
            "background_pass",
            "candidate_gap_min",
            "perturbation_threshold_max",
        ],
    )
    write_csv(output_dir / "oat_results.csv", all_oat, OAT_FIELDS)
    write_csv(
        output_dir / "decision_agreement.csv",
        all_decision,
        ["model", "task", "varied_factor", "n_alternatives", "n_agree", "agreement_rate", "baseline_decision"],
    )
    write_csv(
        output_dir / "draw_pass_frequency.csv",
        all_draw,
        [
            "model",
            "task",
            "B",
            "n_subsets",
            "pass_frequency",
            "min_safety_margin",
            "median_safety_margin",
            "min_prominence_margin",
            "median_prominence_margin",
            "note",
        ],
    )
    write_csv(
        output_dir / "partition_stability.csv",
        all_partition,
        [
            "model",
            "task",
            "varied_factor",
            "varied_value",
            "K",
            "num_landmarks",
            "nominal_knn",
            "adjusted_rand_index",
            "deployed",
            "note",
        ],
    )

    summary = {
        "pairs_requested": [s.key for s in specs],
        "pair_status": pair_status,
        "non_k_overall_agreement": f"{non_k_agree}/{non_k_total}",
        "boundary_cases": boundary_cases,
        "baseline_rows": all_baseline,
        "decision_rows": all_decision,
        "draw_rows": all_draw,
        "partition_rows": all_partition,
    }
    if all_partition:
        aris = [float(r["adjusted_rand_index"]) for r in all_partition]
        summary["partition_mean_ari"] = float(np.mean(aris))
        summary["partition_min_ari"] = float(np.min(aris))

    atomic_write_json(output_dir / "gate_sensitivity_summary.json", summary)
    atomic_write_text(output_dir / "REPORT.md", render_report(summary))
    atomic_write_text(output_dir / "paper_table.tex", render_paper_table(all_baseline))
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "command": " ".join(sys.argv),
            "git": git_info(repo_root),
            "smoke_test": args.smoke_test,
            "output_dir": str(output_dir),
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "pairs_ok": len(all_baseline)}, indent=2))


if __name__ == "__main__":
    main()
