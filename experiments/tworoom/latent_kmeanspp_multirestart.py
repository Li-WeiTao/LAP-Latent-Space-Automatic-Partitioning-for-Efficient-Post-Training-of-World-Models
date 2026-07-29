#!/usr/bin/env python3
"""Z-score + spherical K-means++ multi-restart stability experiment.

Outer seeds 0..19; inner 50 K-means++ restarts per outer seed.
Evaluates restart budgets R in {1,5,10,20,50} from the same inner runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from latent_cluster_common import (  # noqa: E402
    default_embed_dir,
    load_all_train_dedup_latent_vectors,
    resolve_cluster_torch_device,
)
from latent_preprocess_stability import (  # noqa: E402
    K,
    consensus_stats,
    git_commit,
    pairwise_stability_for_preprocess,
    row_l2,
    sorted_cluster_fracs,
    summarize_pairwise,
    write_csv,
)
from latent_spherical_kmeans_lib import (  # noqa: E402
    assign_labels_from_centroids,
    cluster_torch_spherical_kmeans_converged,
)
from latent_preprocess_stability import apply_zscore_l2  # noqa: E402

OUTER_SEEDS = list(range(20))
INNER_RESTARTS = 50
R_BUDGETS = (1, 5, 10, 20, 50)
MAX_ITER = 1000
REL_TOL = 1e-7
PATIENCE = 10
EPS = 1e-6

# v2 zscore random-init baseline (converged, 20 outer seeds)
V2_RANDOM_BASELINE = {
    "init": "random_sample",
    "inner_restart": 1,
    "pairwise_ari_mean": 0.22608446483789002,
    "pairwise_ari_median": 0.20666239335018532,
    "consensus_mean_q": 0.6891440031745926,
    "objective_gap": None,
}


@dataclass
class InnerRunResult:
    outer_seed: int
    inner_restart: int
    inner_seed: int
    objective_final: float
    niter: int
    converged: bool
    cluster_frac_min: float
    cluster_frac_mid: float
    cluster_frac_max: float
    fit_sec: float
    centroids: torch.Tensor


def inner_seed_value(outer_seed: int, restart_id: int) -> int:
    return outer_seed * 100_000 + restart_id


def run_inner_restarts(
    data: torch.Tensor,
    outer_seed: int,
    *,
    max_iter: int,
    rel_tol: float,
    patience: int,
) -> list[InnerRunResult]:
    results: list[InnerRunResult] = []
    for restart_id in range(INNER_RESTARTS):
        seed = inner_seed_value(outer_seed, restart_id)
        centroids, _labels, info = cluster_torch_spherical_kmeans_converged(
            data,
            num_clusters=K,
            seed=seed,
            max_iter=max_iter,
            rel_tol=rel_tol,
            patience=patience,
            init_mode="kmeanspp",
        )
        labels_np = assign_labels_from_centroids(data, centroids).detach().cpu().numpy()
        fmin, fmid, fmax = sorted_cluster_fracs(labels_np)
        results.append(
            InnerRunResult(
                outer_seed=outer_seed,
                inner_restart=restart_id,
                inner_seed=seed,
                objective_final=info["objective_final"],
                niter=info["niter"],
                converged=info["converged"],
                cluster_frac_min=fmin,
                cluster_frac_mid=fmid,
                cluster_frac_max=fmax,
                fit_sec=info["fit_sec"],
                centroids=centroids.detach(),
            )
        )
    return results


def pick_best_for_budget(runs: list[InnerRunResult], budget: int) -> InnerRunResult:
    pool = runs[:budget]
    return max(pool, key=lambda r: r.objective_final)


def outer_runs_from_budget_metrics(
    by_outer: dict[int, list[dict]],
    budget: int,
    data: torch.Tensor,
    global_idx: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    runs: list[dict] = []
    for outer_seed in OUTER_SEEDS:
        pool = by_outer[outer_seed][:budget]
        best = max(pool, key=lambda r: r["objective_final"])
        centroids = rerun_centroids_for_selection(data, best)
        labels = assign_labels_from_centroids(data, centroids.to(data.device))
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        centroids_np = centroids.cpu().numpy().astype(np.float32)
        fmin, fmid, fmax = sorted_cluster_fracs(labels_np)
        run = {
            "preprocess": f"zscore_kmeanspp_R{budget}",
            "seed": outer_seed,
            "labels": labels_np,
            "centroids": centroids_np,
            "objective_final": best["objective_final"],
            "inner_restart_budget": budget,
            "selected_inner_restart": best["inner_restart"],
            "cluster_frac_min": fmin,
            "cluster_frac_mid": fmid,
            "cluster_frac_max": fmax,
        }
        runs.append(run)
        np.savez_compressed(
            out_dir / "labels" / f"kmeanspp_R{budget}_outer{outer_seed}.npz",
            global_idx=global_idx,
            labels=labels_np,
            centroids=centroids_np,
            objective_final=best["objective_final"],
            selected_inner_restart=best["inner_restart"],
        )
    return runs


def outer_runs_from_budget(
    all_inner: dict[int, list[InnerRunResult]],
    budget: int,
    data: torch.Tensor,
    global_idx: np.ndarray,
    out_dir: Path,
) -> list[dict]:
    runs: list[dict] = []
    for outer_seed in OUTER_SEEDS:
        best = pick_best_for_budget(all_inner[outer_seed], budget)
        labels = assign_labels_from_centroids(data, best.centroids.to(data.device))
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        centroids_np = best.centroids.cpu().numpy().astype(np.float32)
        fmin, fmid, fmax = sorted_cluster_fracs(labels_np)
        run = {
            "preprocess": f"zscore_kmeanspp_R{budget}",
            "seed": outer_seed,
            "labels": labels_np,
            "centroids": centroids_np,
            "objective_final": best.objective_final,
            "inner_restart_budget": budget,
            "selected_inner_restart": best.inner_restart,
            "cluster_frac_min": fmin,
            "cluster_frac_mid": fmid,
            "cluster_frac_max": fmax,
        }
        runs.append(run)
        np.savez_compressed(
            out_dir / "labels" / f"kmeanspp_R{budget}_outer{outer_seed}.npz",
            global_idx=global_idx,
            labels=labels_np,
            centroids=centroids_np,
            objective_final=best.objective_final,
            selected_inner_restart=best.inner_restart,
        )
    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed-dir", type=Path, default=default_embed_dir())
    p.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results" / "latent_kmeanspp_multirestart_k3",
    )
    p.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip inner clustering; rebuild R-budget summaries from inner_run_metrics.csv",
    )
    return p.parse_args()


def load_inner_metrics_csv(path: Path) -> dict[int, list[dict]]:
    by_outer: dict[int, list[dict]] = {s: [] for s in OUTER_SEEDS}
    with path.open() as f:
        for row in csv.DictReader(f):
            outer = int(row["outer_seed"])
            by_outer[outer].append(
                {
                    "outer_seed": outer,
                    "inner_restart": int(row["inner_restart"]),
                    "inner_seed": int(row["inner_seed"]),
                    "objective_final": float(row["objective_final"]),
                    "niter": int(row["niter"]),
                    "converged": bool(int(row["converged"])),
                    "cluster_frac_min": float(row["cluster_frac_min"]),
                    "cluster_frac_mid": float(row["cluster_frac_mid"]),
                    "cluster_frac_max": float(row["cluster_frac_max"]),
                    "fit_sec": float(row["fit_sec"]),
                }
            )
    for outer in OUTER_SEEDS:
        by_outer[outer].sort(key=lambda r: r["inner_restart"])
    return by_outer


def rerun_centroids_for_selection(data: torch.Tensor, metrics_row: dict) -> torch.Tensor:
    centroids, _labels, _info = cluster_torch_spherical_kmeans_converged(
        data,
        num_clusters=K,
        seed=metrics_row["inner_seed"],
        max_iter=MAX_ITER,
        rel_tol=REL_TOL,
        patience=PATIENCE,
        init_mode="kmeanspp",
    )
    return centroids.detach()


def build_all_inner_from_metrics(
    data: torch.Tensor,
    by_outer: dict[int, list[dict]],
) -> dict[int, list[InnerRunResult]]:
    all_inner: dict[int, list[InnerRunResult]] = {}
    for outer_seed in OUTER_SEEDS:
        runs: list[InnerRunResult] = []
        for row in by_outer[outer_seed]:
            centroids = rerun_centroids_for_selection(data, row)
            runs.append(
                InnerRunResult(
                    outer_seed=outer_seed,
                    inner_restart=row["inner_restart"],
                    inner_seed=row["inner_seed"],
                    objective_final=row["objective_final"],
                    niter=row["niter"],
                    converged=row["converged"],
                    cluster_frac_min=row["cluster_frac_min"],
                    cluster_frac_mid=row["cluster_frac_mid"],
                    cluster_frac_max=row["cluster_frac_max"],
                    fit_sec=row["fit_sec"],
                    centroids=centroids,
                )
            )
        all_inner[outer_seed] = runs
        print(f"  [summary-only] outer_seed={outer_seed} centroids ready", flush=True)
    return all_inner


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels").mkdir(exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)

    torch_device = resolve_cluster_torch_device(args.device, args.gpu_id)
    print(f"[device] {torch_device}", flush=True)

    t0 = time.perf_counter()
    Z, global_idx, data_stats = load_all_train_dedup_latent_vectors(
        args.embed_dir, frameskip=args.frameskip
    )
    mu = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = Z.std(axis=0, dtype=np.float64).astype(np.float32)
    X, _zparams = apply_zscore_l2(Z, mu, sigma)
    data = torch.from_numpy(row_l2(X)).to(torch_device)
    load_sec = time.perf_counter() - t0
    print(f"[load+preprocess] N={len(Z)} dim={Z.shape[1]} in {load_sec:.2f}s", flush=True)

    np.savez_compressed(
        args.out_dir / "zscore_params.npz",
        mu=mu,
        sigma=sigma,
        eps=np.float32(EPS),
    )

    inner_rows: list[dict] = []
    all_inner: dict[int, list[InnerRunResult]] = {}
    by_outer: dict[int, list[dict]] | None = None
    cluster_t0 = time.perf_counter()

    if args.summary_only:
        metrics_path = args.out_dir / "inner_run_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing {metrics_path} for --summary-only")
        by_outer = load_inner_metrics_csv(metrics_path)
        inner_rows = [row for rows in by_outer.values() for row in rows]
        cluster_sec = 0.0
        print(f"[summary-only] loaded {len(inner_rows)} inner runs from CSV", flush=True)
    else:
        for outer_seed in OUTER_SEEDS:
            print(f"\n==== outer_seed={outer_seed} ({INNER_RESTARTS} kmeans++ restarts) ====", flush=True)
            runs = run_inner_restarts(
                data,
                outer_seed,
                max_iter=MAX_ITER,
                rel_tol=REL_TOL,
                patience=PATIENCE,
            )
            all_inner[outer_seed] = runs
            for r in runs:
                inner_rows.append(
                    {
                        "outer_seed": r.outer_seed,
                        "inner_restart": r.inner_restart,
                        "inner_seed": r.inner_seed,
                        "objective_final": r.objective_final,
                        "niter": r.niter,
                        "converged": int(r.converged),
                        "cluster_frac_min": r.cluster_frac_min,
                        "cluster_frac_mid": r.cluster_frac_mid,
                        "cluster_frac_max": r.cluster_frac_max,
                        "fit_sec": r.fit_sec,
                    }
                )
            objs = [r.objective_final for r in runs]
            print(
                f"  objectives: min={min(objs):.6f} max={max(objs):.6f} "
                f"mean={np.mean(objs):.6f}",
                flush=True,
            )
        cluster_sec = time.perf_counter() - cluster_t0
        write_csv(
            args.out_dir / "inner_run_metrics.csv",
            inner_rows,
            list(inner_rows[0].keys()),
        )

    global_objective_max = max(float(r["objective_final"]) for r in inner_rows)

    budget_rows: list[dict] = []
    pairwise_all: list[dict] = []

    for budget in R_BUDGETS:
        if by_outer is not None:
            outer_runs = outer_runs_from_budget_metrics(
                by_outer, budget, data, global_idx, args.out_dir
            )
        else:
            outer_runs = outer_runs_from_budget(all_inner, budget, data, global_idx, args.out_dir)
        pw_rows, ari_mat, _ = pairwise_stability_for_preprocess(outer_runs)
        for row in pw_rows:
            row["restart_budget"] = budget
        pairwise_all.extend(pw_rows)
        consensus = consensus_stats(outer_runs)
        summary = summarize_pairwise(pw_rows)
        objs = [r["objective_final"] for r in outer_runs]
        objective_mean = float(np.mean(objs))
        objective_std = float(np.std(objs))
        fracs_min = [r["cluster_frac_min"] for r in outer_runs]
        fracs_mid = [r["cluster_frac_mid"] for r in outer_runs]
        fracs_max = [r["cluster_frac_max"] for r in outer_runs]
        budget_rows.append(
            {
                "init": "kmeanspp",
                "inner_restart": budget,
                "pairwise_ari_mean": summary["pairwise_ari_mean"],
                "pairwise_ari_std": summary["pairwise_ari_std"],
                "pairwise_ari_median": summary["pairwise_ari_median"],
                "pairwise_nmi_mean": summary["pairwise_nmi_mean"],
                "pairwise_label_agreement_mean": summary["pairwise_label_agreement_mean"],
                "pairwise_centroid_cosine_mean": summary["pairwise_centroid_cosine_mean"],
                "consensus_mean_q": consensus["consensus_mean_q"],
                "consensus_q_eq_1_frac": consensus["consensus_q_eq_1_frac"],
                "consensus_q_ge_0p9_frac": consensus["consensus_q_ge_0p9_frac"],
                "consensus_q_lt_0p7_frac": consensus["consensus_q_lt_0p7_frac"],
                "objective_mean": objective_mean,
                "objective_std": objective_std,
                "objective_gap_to_global_max": float(global_objective_max - objective_mean),
                "global_objective_max": global_objective_max,
                "cluster_frac_min_mean": float(np.mean(fracs_min)),
                "cluster_frac_mid_mean": float(np.mean(fracs_mid)),
                "cluster_frac_max_mean": float(np.mean(fracs_max)),
                "total_cluster_sec": cluster_sec,
                "medoid_outer_seed": consensus["medoid_seed"],
            }
        )
        print(
            f"\n[R={budget}] ARI={summary['pairwise_ari_mean']:.4f}±{summary['pairwise_ari_std']:.4f}  "
            f"consensus_q={consensus['consensus_mean_q']:.3f}  "
            f"obj_mean={objective_mean:.6f} gap={global_objective_max - objective_mean:.6f}",
            flush=True,
        )

    # prepend v2 random baseline row
    budget_rows.insert(
        0,
        {
            "init": V2_RANDOM_BASELINE["init"],
            "inner_restart": V2_RANDOM_BASELINE["inner_restart"],
            "pairwise_ari_mean": V2_RANDOM_BASELINE["pairwise_ari_mean"],
            "pairwise_ari_std": 0.10707960919648102,
            "pairwise_ari_median": V2_RANDOM_BASELINE["pairwise_ari_median"],
            "pairwise_nmi_mean": 0.2122428996107451,
            "pairwise_label_agreement_mean": 0.5995812432668198,
            "pairwise_centroid_cosine_mean": 0.5239129228484736,
            "consensus_mean_q": V2_RANDOM_BASELINE["consensus_mean_q"],
            "consensus_q_eq_1_frac": 0.031108370350095577,
            "consensus_q_ge_0p9_frac": 0.12810712711451727,
            "consensus_q_lt_0p7_frac": 0.4694055223403168,
            "objective_mean": None,
            "objective_std": None,
            "objective_gap_to_global_max": None,
            "global_objective_max": global_objective_max,
            "cluster_frac_min_mean": None,
            "cluster_frac_mid_mean": None,
            "cluster_frac_max_mean": None,
            "total_cluster_sec": None,
            "medoid_outer_seed": 9,
            "note": "from latent_preprocess_convergence_v1 zscore_l2 random init",
        },
    )

    kpp_fields = list(budget_rows[1].keys())
    for row in budget_rows:
        row.setdefault("note", "")
    write_csv(args.out_dir / "restart_budget_summary.csv", budget_rows, kpp_fields + ["note"])
    write_csv(
        args.out_dir / "pairwise_stability.csv",
        pairwise_all,
        list(pairwise_all[0].keys()),
    )

    # figure: ARI vs restart budget
    fig, ax = plt.subplots(figsize=(8, 5))
    kpp = [r for r in budget_rows if r["init"] == "kmeanspp"]
    xs = [r["inner_restart"] for r in kpp]
    ys = [r["pairwise_ari_mean"] for r in kpp]
    yerr = [r["pairwise_ari_std"] for r in kpp]
    ax.errorbar(xs, ys, yerr=yerr, marker="o", label="kmeans++")
    ax.axhline(V2_RANDOM_BASELINE["pairwise_ari_mean"], ls="--", color="gray", label="random (v2)")
    ax.axhline(0.9, ls=":", color="green", alpha=0.5, label="ARI=0.9")
    ax.axhline(0.4, ls=":", color="orange", alpha=0.5, label="ARI=0.4")
    ax.set_xlabel("inner restart budget R")
    ax.set_ylabel("mean pairwise ARI (20 outer seeds)")
    ax.set_title("Z-score + spherical K-means++ multi-restart stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "ari_vs_restart_budget.png", dpi=150)
    plt.close(fig)

    meta = {
        "version_label": "latent_kmeanspp_multirestart_v1",
        "preprocess": "zscore_l2",
        "rollout_formula": "x_t = normalize((z_t - mu) / (sigma + eps))",
        "outer_seeds": OUTER_SEEDS,
        "inner_restarts": INNER_RESTARTS,
        "inner_seed_formula": "outer_seed * 100000 + restart_id",
        "r_budgets": list(R_BUDGETS),
        "kmeanspp_prob": "p_i ∝ 1 - max_c cos(x_i, c)",
        "max_iter": MAX_ITER,
        "rel_tol": REL_TOL,
        "patience": PATIENCE,
        "num_unique_timesteps": data_stats["num_unique_timesteps"],
        "latent_dim": data_stats["latent_dim"],
        "global_objective_max": global_objective_max,
        "git_commit": git_commit(),
        "device": str(torch_device),
        "load_sec": load_sec,
        "cluster_sec": cluster_sec,
        "wall_sec": time.perf_counter() - t0,
        "v2_random_baseline": V2_RANDOM_BASELINE,
        "budget_summaries": budget_rows,
    }
    with (args.out_dir / "experiment_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("\n==== restart budget summary ====", flush=True)
    for row in budget_rows:
        if row["init"] == "random_sample":
            print(
                f"  random (v2)  R=1  ARI={row['pairwise_ari_mean']:.4f}  "
                f"consensus_q={row['consensus_mean_q']:.3f}",
                flush=True,
            )
        else:
            print(
                f"  kmeans++  R={row['inner_restart']:2d}  "
                f"ARI={row['pairwise_ari_mean']:.4f}  "
                f"consensus_q={row['consensus_mean_q']:.3f}  "
                f"obj_gap={row['objective_gap_to_global_max']:.6f}",
                flush=True,
            )
    print(f"\n[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
