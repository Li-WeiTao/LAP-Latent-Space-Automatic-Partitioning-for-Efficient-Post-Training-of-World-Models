#!/usr/bin/env python3
"""Gate-only one-at-a-time sensitivity audit for LAP spectral auto-gate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
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
    CACHE_SCHEMA_VERSION,
    MAX_EIGENVALUES,
    OATScenario,
    PAIR_SPECS,
    VersionedSpectrumCache,
    atomic_write_json,
    atomic_write_text,
    audit_source_hashes,
    baseline_draw_bank,
    build_excluded_landmark_indices,
    build_oat_scenarios,
    build_spectra_by_m,
    choose_held_out_audit_rows,
    collect_minimal_spectrum_requests,
    collect_non_k_boundary_cases,
    compare_to_manifest,
    decision_agreement_rows,
    draw_pass_rows,
    fit_audit_labels,
    git_info,
    hungarian_align,
    k_behavior_rows,
    latex_escape,
    margins_from_result,
    partition_stability_summaries,
    resolve_pair_inputs,
    result_row,
    scenario_spectra_bank,
    sha256_file,
    evaluate_config,
    NeighborDrawCache,
)
from lap.partition.gate import SpectralGateConfig  # noqa: E402
from lap.partition.landmark import _zscore_l2  # noqa: E402
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
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Write outputs here first; atomically promote on success.",
    )
    parser.add_argument("--pairs", default="lewm_tworoom,lewm_pusht,lewm_reacher,lewm_cube")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=-1, help="Use GPU exact kNN when >= 0.")
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Atomically replace --output-dir with --staging-dir after success.",
    )
    parser.add_argument(
        "--refresh-provenance",
        action="store_true",
        help="Rewrite manifest/report provenance from existing outputs without rerunning pairs.",
    )
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


def scenario_cache_hit(
    scenario: OATScenario,
    *,
    spectra_by_m,
    cache: VersionedSpectrumCache,
    identity_base: dict[str, object],
    baseline_config: SpectralGateConfig,
) -> bool:
    if scenario.varied_factor in {"rho", "c_pert", "c_bg", "K"}:
        return True
    cfg = scenario.config
    seeds = set(cfg.diagnostic_seeds)
    if cfg.num_landmarks == identity_base["baseline_m"]:
        if scenario.varied_factor in {"baseline", "B"} or scenario.is_baseline:
            seeds.update(range(10))
    from experiments.control_matrix.gate_audit_lib import SpectrumCacheIdentity

    for seed in seeds:
        for knn in cfg.graph_knn_values:
            identity = SpectrumCacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                latent_cache_sha256=str(identity_base["latent_cache_sha256"]),
                group_ids_hash=str(identity_base["group_ids_hash"]),
                model=str(identity_base["model"]),
                task=str(identity_base["task"]),
                num_landmarks=cfg.num_landmarks,
                landmark_seed=seed,
                knn=knn,
                eigenvalue_count=MAX_EIGENVALUES,
                eig_tol=baseline_config.eig_tol,
                eig_maxiter=baseline_config.eig_maxiter,
                preprocessing_version="zscore_l2_v1",
                source_code_id="gate_audit_lib_v2",
            )
            if cache._read(identity) is None:
                return False
    return True


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
    cache = VersionedSpectrumCache(pair_out / "spectrum_cache")
    cache_hash = sha256_file(resolved.latent_cache)

    latents, sample_ids, cache_stats = load_unique_latents(resolved.latent_cache, frameskip=5)
    groups = episode_ids(resolved.data_file, sample_ids, "auto")
    transformed, _, _ = _zscore_l2(latents)
    group_hash = __import__(
        "experiments.control_matrix.gate_audit_lib", fromlist=["hash_group_ids"]
    ).hash_group_ids(groups)

    baseline = resolved.baseline_config
    if smoke_test:
        baseline = replace(
            baseline,
            num_landmarks=min(baseline.num_landmarks, 2000),
            diagnostic_seeds=(0, 1, 2),
        )

    scenarios = build_oat_scenarios(baseline)
    if smoke_test:
        scenarios = [s for s in scenarios if s.varied_factor in {"baseline", "rho", "K", "M"}][:10]

    requests = collect_minimal_spectrum_requests(scenarios, baseline)
    all_knns = {knn for _, _, knn in requests}
    neighbor_cache = build_neighbor_cache(gpu_id, query_chunk)
    identity_base = {
        "latent_cache_sha256": cache_hash,
        "group_ids_hash": group_hash,
        "model": spec.model,
        "task": spec.task,
        "baseline_m": baseline.num_landmarks,
    }
    spectra_by_m = build_spectra_by_m(
        cache,
        identity_base=identity_base,
        transformed=transformed,
        group_ids=groups,
        baseline_config=baseline,
        requests=requests,
        neighbor_cache=neighbor_cache,
        max_k=max(all_knns),
    )

    baseline_bank = scenario_spectra_bank(spectra_by_m, baseline)
    baseline_result = evaluate_config(baseline, baseline_bank)
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

    excluded = build_excluded_landmark_indices(
        len(transformed), baseline, scenarios, groups
    )
    audit_rows = choose_held_out_audit_rows(
        len(transformed),
        excluded_indices=excluded,
        size=min(AUDIT_SAMPLE_SIZE, len(transformed) // 4),
        seed=AUDIT_SAMPLE_SEED,
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
        scenario_bank = scenario_spectra_bank(spectra_by_m, scenario.config)
        result = evaluate_config(scenario.config, scenario_bank)
        elapsed = time.perf_counter() - t0
        cache_hit = scenario_cache_hit(
            scenario,
            spectra_by_m=spectra_by_m,
            cache=cache,
            identity_base=identity_base,
            baseline_config=baseline,
        )
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
                    "gate_would_select_partition": bool(result.use_partition),
                    "note": "diagnosis-only candidate partition; not deployed or trained",
                }
            )

    draw_rows = draw_pass_rows(
        model=spec.model,
        task=spec.task,
        draw_bank=baseline_draw_bank(spectra_by_m, baseline),
        baseline=baseline,
    )

    decision_rows, overall_agreement = decision_agreement_rows(oat_rows, scenarios)
    k_rows = k_behavior_rows(oat_rows)
    partition_summaries = partition_stability_summaries(partition_rows)

    pair_log.update(
        {
            "status": "ok",
            "cache_hash": cache_hash,
            "group_ids_hash": group_hash,
            "baseline_margin_row": baseline_margin_row,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "eigensolves": cache.eigensolves,
            "spectrum_requests": len(requests),
            "overall_agreement": overall_agreement,
        }
    )
    atomic_write_json(pair_out / "pair_summary.json", pair_log)
    return {
        **pair_log,
        "baseline_margin_row": baseline_margin_row,
        "oat_rows": oat_rows,
        "decision_rows": decision_rows,
        "k_rows": k_rows,
        "draw_rows": draw_rows,
        "partition_rows": partition_rows,
        "partition_summaries": partition_summaries,
        "overall_agreement": overall_agreement,
    }


def render_report(summary: dict[str, object]) -> str:
    provenance = summary.get("provenance", {})
    lines = [
        "# Gate-only sensitivity audit",
        "",
        "This audit reuses frozen latent caches and recomputes kNN landmark graphs,",
        "spectra, and gate arithmetic only. **No predictors were trained** and **no",
        "planning / MPC evaluation** was run. Thresholds were not adjusted post hoc.",
        "",
        f"Provenance: commit `{provenance.get('git_commit', 'unknown')}`, "
        f"dirty={provenance.get('git_dirty')}, "
        f"final={provenance.get('marked_final', False)}.",
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
    for row in summary.get("decision_rows", []):
        lines.append(
            f"- {row['model']}/{row['task']} {row['varied_factor']}: "
            f"{row['n_agree']}/{row['n_alternatives']} "
            f"({100 * float(row['agreement_rate']):.1f}%)"
        )

    lines.extend(["", "## Behavior across candidate region counts (K sweep)", ""])
    for row in summary.get("k_behavior_rows", []):
        lines.append(
            f"- {row['model']}/{row['task']} K={row['K']}: decision={row['decision']}, "
            f"prominence_margin={row.get('prominence_margin')}"
        )

    lines.extend(["", "## Draw-subset pass frequency (baseline M, seeds 0–9)", ""])
    for row in summary.get("draw_rows", []):
        lines.append(
            f"- {row['model']}/{row['task']} B={row['B']}: pass={row['pass_frequency']:.3f}, "
            f"min prominence={row.get('min_prominence_margin')}"
        )

    lines.extend(["", "## Partition stability by pair/factor", ""])
    for row in summary.get("partition_summaries", []):
        lines.append(
            f"- {row['model']}/{row['task']} {row['varied_factor']}: "
            f"mean ARI={row['mean_ari']:.4f}, min ARI={row['min_ari']:.4f}, "
            f"n={row['comparison_count']}"
        )

    lines.extend(["", "## Non-K boundary / abstention flips", ""])
    if summary.get("boundary_cases"):
        for case in summary["boundary_cases"]:
            lines.append(f"- {case}")
    else:
        lines.append("_No non-K OAT flips observed._")

    lines.extend(
        [
            "",
            "## What this audit cannot prove",
            "",
            "- Planning performance under alternate K or flipped gate decisions.",
            "- Draw-subset diagnostics do not modify the predeclared gate definition.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paper_table(baseline_rows: list[dict]) -> str:
    lines = [
        "% Gate sensitivity audit (gate-only; no predictor training).",
        "\\begin{tabular}{llrrll}",
        "\\toprule",
        "Model & Task & Safety margin & Prominence margin & Decision & Reason \\\\",
        "\\midrule",
    ]
    for row in baseline_rows:
        sm = row.get("safety_margin")
        pm = row.get("prominence_margin")
        lines.append(
            f"{latex_escape(row['model'])} & {latex_escape(row['task'])} & "
            f"{'' if sm is None else f'{sm:.3f}'} & "
            f"{'' if pm is None else f'{pm:.3f}'} & "
            f"{latex_escape(row['decision'])} & {latex_escape(row['reason'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def hash_output_files(output_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in (
        "baseline_margins.csv",
        "oat_results.csv",
        "decision_agreement.csv",
        "draw_pass_frequency.csv",
        "partition_stability.csv",
        "k_behavior.csv",
        "partition_stability_summary.csv",
        "gate_sensitivity_summary.json",
        "REPORT.md",
        "paper_table.tex",
    ):
        path = output_dir / name
        if path.is_file():
            out[name] = sha256_file(path)
    return out


def promote_staging(staging_dir: Path, final_dir: Path) -> None:
    backup = final_dir.with_name(final_dir.name + "_prev")
    if backup.exists():
        shutil.rmtree(backup)
    if final_dir.exists():
        final_dir.replace(backup)
    staging_dir.replace(final_dir)
    if backup.exists():
        shutil.rmtree(backup)


def refresh_provenance(repo_root: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    summary_path = output_dir / "gate_sensitivity_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    git = git_info(repo_root)
    source_hashes = audit_source_hashes(repo_root)
    marked_final = not git.get("dirty", True)
    summary["provenance"] = {
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "marked_final": marked_final,
        "source_hashes": source_hashes,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_text(output_dir / "REPORT.md", render_report(summary))
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"] = git
    manifest["marked_final"] = marked_final
    manifest["source_hashes"] = source_hashes
    manifest["output_hashes"] = hash_output_files(output_dir)
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "marked_final": marked_final,
                "git": git,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    final_dir = args.output_dir
    if not final_dir.is_absolute():
        final_dir = repo_root / final_dir
    if args.refresh_provenance:
        refresh_provenance(repo_root, final_dir)
        return

    git = git_info(repo_root)
    output_dir = args.staging_dir or final_dir
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

    source_hashes = audit_source_hashes(repo_root)

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
                "bash experiments/control_matrix/scripts/run_gate_sensitivity_audit.sh "
                f"--pairs {spec.key}"
            )
        preflight_rows.append(row)
    atomic_write_json(output_dir / "preflight_report.json", {"pairs": preflight_rows})

    all_baseline: list[dict] = []
    all_oat: list[dict] = []
    all_decision: list[dict] = []
    all_k: list[dict] = []
    all_draw: list[dict] = []
    all_partition: list[dict] = []
    all_partition_summaries: list[dict] = []
    pair_status: list[dict] = []
    boundary_cases: list[str] = []
    non_k_agree = 0
    non_k_total = 0
    total_eigensolves = 0
    total_cache_hits = 0
    total_cache_misses = 0

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
            all_k.extend(result["k_rows"])
            all_draw.extend(result["draw_rows"])
            all_partition.extend(result["partition_rows"])
            all_partition_summaries.extend(result["partition_summaries"])
            overall = result.get("overall_agreement", {})
            if overall.get("non_k_overall_agreement"):
                agree, total = str(overall["non_k_overall_agreement"]).split("/")
                non_k_agree += int(agree)
                non_k_total += int(total)
            total_eigensolves += int(result.get("eigensolves", 0))
            total_cache_hits += int(result.get("cache_hits", 0))
            total_cache_misses += int(result.get("cache_misses", 0))
            boundary_cases.extend(collect_non_k_boundary_cases(result["oat_rows"]))

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
        output_dir / "k_behavior.csv",
        all_k,
        ["model", "task", "K", "decision", "reason", "safety_margin", "prominence_margin", "agreement_with_baseline_K"],
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
            "gate_would_select_partition",
            "note",
        ],
    )
    write_csv(
        output_dir / "partition_stability_summary.csv",
        all_partition_summaries,
        ["model", "task", "varied_factor", "comparison_count", "mean_ari", "min_ari"],
    )

    marked_final = not git.get("dirty", True)
    summary = {
        "pairs_requested": [s.key for s in specs],
        "pair_status": pair_status,
        "non_k_overall_agreement": f"{non_k_agree}/{non_k_total}",
        "boundary_cases": boundary_cases,
        "baseline_rows": all_baseline,
        "decision_rows": all_decision,
        "k_behavior_rows": all_k,
        "draw_rows": all_draw,
        "partition_rows": all_partition,
        "partition_summaries": all_partition_summaries,
        "cache_stats": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "hits": total_cache_hits,
            "misses": total_cache_misses,
            "eigensolves": total_eigensolves,
        },
        "provenance": {
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            "marked_final": marked_final,
            "source_hashes": source_hashes,
        },
    }
    atomic_write_json(output_dir / "gate_sensitivity_summary.json", summary)
    atomic_write_text(output_dir / "REPORT.md", render_report(summary))
    atomic_write_text(output_dir / "paper_table.tex", render_paper_table(all_baseline))
    input_cache_hashes = {
        row["pair"]: row.get("cache_hash")
        for row in pair_status
        if row.get("cache_hash")
    }
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "command": " ".join(sys.argv),
            "git": git,
            "marked_final": marked_final,
            "smoke_test": args.smoke_test,
            "output_dir": str(output_dir),
            "spectrum_cache_schema_version": CACHE_SCHEMA_VERSION,
            "source_hashes": source_hashes,
            "input_cache_hashes": input_cache_hashes,
            "output_hashes": hash_output_files(output_dir),
            "cache_stats": summary["cache_stats"],
        },
    )
    if args.promote and args.staging_dir is not None:
        promote_staging(output_dir, final_dir)
        output_dir = final_dir

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pairs_ok": len(all_baseline),
                "marked_final": marked_final,
                "eigensolves": total_eigensolves,
                "cache_hits": total_cache_hits,
                "cache_misses": total_cache_misses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
