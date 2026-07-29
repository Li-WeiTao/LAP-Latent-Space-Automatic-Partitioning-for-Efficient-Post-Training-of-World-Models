#!/usr/bin/env python3
"""Converged spherical K-means on top-3 preprocessings (raw/center/zscore).

Follow-up to latent_preprocess_stability_v1: max_iter=1000 with relative
objective tolerance and label-change tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from latent_preprocess_stability import (  # noqa: E402
    DEFAULT_SEEDS,
    K,
    PreprocessState,
    build_all_preprocesses,
    consensus_stats,
    git_commit,
    hungarian_agreement,
    hungarian_centroid_cosine,
    make_figures,
    margin_stats,
    pairwise_stability_for_preprocess,
    per_run_metrics,
    row_l2,
    save_preprocess_params,
    sorted_cluster_fracs,
    spherical_objective,
    summarize_pairwise,
    write_csv,
)
from latent_cluster_common import (  # noqa: E402
    default_embed_dir,
    load_all_train_dedup_latent_vectors,
    resolve_cluster_torch_device,
)

from latent_spherical_kmeans_lib import (  # noqa: E402
    cluster_torch_spherical_kmeans_converged as cluster_converged_lib,
)

CONVERGE_PREPROCESS_NAMES = ("raw_l2", "center_l2", "zscore_l2")
MAX_ITER = 1000
REL_TOL = 1e-7
PATIENCE = 10


def cluster_torch_spherical_kmeans_converged(
    X: np.ndarray,
    *,
    num_clusters: int,
    seed: int,
    max_iter: int,
    rel_tol: float,
    patience: int,
    torch_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    data = torch.from_numpy(row_l2(X)).to(torch_device)
    centroids, labels, info = cluster_converged_lib(
        data,
        num_clusters=num_clusters,
        seed=seed,
        max_iter=max_iter,
        rel_tol=rel_tol,
        patience=patience,
        init_mode="random_k_samples",
    )
    return (
        centroids.detach().cpu().numpy().astype(np.float32),
        labels.detach().cpu().numpy().astype(np.int64),
        info,
    )


def select_preprocess_states(states: list[PreprocessState]) -> list[PreprocessState]:
    out = [s for s in states if s.name in CONVERGE_PREPROCESS_NAMES]
    if [s.name for s in out] != list(CONVERGE_PREPROCESS_NAMES):
        raise RuntimeError("Missing required preprocess states")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed-dir", type=Path, default=default_embed_dir())
    p.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results" / "latent_preprocess_convergence_k3",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--max-iter", type=int, default=MAX_ITER)
    p.add_argument("--rel-tol", type=float, default=REL_TOL)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--frameskip", type=int, default=5)
    return p.parse_args()


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
    load_sec = time.perf_counter() - t0

    mu = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = Z.std(axis=0, dtype=np.float64).astype(np.float32)
    from latent_preprocess_stability import fit_pca_stats  # noqa: E402

    pca = fit_pca_stats(Z)
    states = select_preprocess_states(build_all_preprocesses(Z))
    save_preprocess_params(args.out_dir, states, pca, mu, sigma)

    per_run_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    summaries: list[dict] = []
    ari_mats: dict[str, np.ndarray] = {}

    for st in states:
        print(f"\n==== preprocess={st.name} ====", flush=True)
        runs: list[dict] = []
        for seed in args.seeds:
            centroids, labels, cinfo = cluster_torch_spherical_kmeans_converged(
                st.X,
                num_clusters=K,
                seed=seed,
                max_iter=args.max_iter,
                rel_tol=args.rel_tol,
                patience=args.patience,
                torch_device=torch_device,
            )
            run = per_run_metrics(
                preprocess=st.name,
                seed=seed,
                labels=labels,
                centroids=centroids,
                X=st.X,
                cluster_info=cinfo,
                global_idx=global_idx,
            )
            runs.append(run)
            np.savez_compressed(
                args.out_dir / "labels" / f"{st.name}_seed{seed}.npz",
                global_idx=global_idx,
                labels=labels,
                centroids=centroids,
            )
            per_run_rows.append(
                {
                    "preprocess": st.name,
                    "seed": seed,
                    "converged": int(cinfo["converged"]),
                    "niter": cinfo["niter"],
                    "objective_final": run["objective_final"],
                    "objective_still_changing": int(cinfo["objective_still_changing"]),
                    "final_rel_obj_change": cinfo["final_rel_obj_change"],
                    "fit_sec": run["fit_sec"],
                    "mean_label_change_frac": cinfo["mean_label_change_frac"],
                    "final_label_change_frac": cinfo["final_label_change_frac"],
                    "max_label_change_frac": cinfo["max_label_change_frac"],
                    "cluster_frac_min": run["cluster_frac_min"],
                    "cluster_frac_mid": run["cluster_frac_mid"],
                    "cluster_frac_max": run["cluster_frac_max"],
                    "margin_mean": run["margin_mean"],
                    "margin_median": run["margin_median"],
                    "margin_p10": run["margin_p10"],
                    "margin_p25": run["margin_p25"],
                    "margin_lt_0p01_frac": run["margin_lt_0p01_frac"],
                }
            )
            print(
                f"  seed={seed:2d}  converged={cinfo['converged']}  "
                f"iters={cinfo['niter']:4d}  obj={run['objective_final']:.6f}  "
                f"final_label_chg={cinfo['final_label_change_frac']:.4f}",
                flush=True,
            )

        pw_rows, ari_mat, _ = pairwise_stability_for_preprocess(runs)
        pairwise_rows.extend(pw_rows)
        ari_mats[st.name] = ari_mat
        consensus = consensus_stats(runs)
        summaries.append(
            {
                "preprocess": st.name,
                **summarize_pairwise(pw_rows),
                **{k: v for k, v in consensus.items() if k != "per_seed_mean_ari"},
                "per_seed_mean_ari": consensus["per_seed_mean_ari"],
                "converged_runs": int(sum(r["converged"] for r in per_run_rows if r["preprocess"] == st.name)),
                "num_runs": len(args.seeds),
            }
        )

    write_csv(args.out_dir / "per_run_metrics.csv", per_run_rows, list(per_run_rows[0].keys()))
    write_csv(args.out_dir / "pairwise_stability.csv", pairwise_rows, list(pairwise_rows[0].keys()))
    summary_csv_rows = [{k: v for k, v in s.items() if k != "per_seed_mean_ari"} for s in summaries]
    write_csv(args.out_dir / "stability_summary.csv", summary_csv_rows, list(summary_csv_rows[0].keys()))

    meta = {
        "version_label": "latent_preprocess_convergence_v1",
        "parent_experiment": "latent_preprocess_stability_v1",
        "preprocess_names": list(CONVERGE_PREPROCESS_NAMES),
        "clustering_seeds": args.seeds,
        "max_iter": args.max_iter,
        "rel_tol": args.rel_tol,
        "patience": args.patience,
        "num_unique_timesteps": data_stats["num_unique_timesteps"],
        "latent_dim": data_stats["latent_dim"],
        "git_commit": git_commit(),
        "device": str(torch_device),
        "load_sec": load_sec,
        "wall_sec": time.perf_counter() - t0,
        "summaries": summaries,
    }
    with (args.out_dir / "preprocess_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    make_figures(args.out_dir, summaries, ari_mats, per_run_rows)

    print("\n==== converged stability summary ====", flush=True)
    for s in sorted(summaries, key=lambda x: -x["pairwise_ari_mean"]):
        print(
            f"  {s['preprocess']:12s}  ARI={s['pairwise_ari_mean']:.4f}±{s['pairwise_ari_std']:.4f}  "
            f"converged={s['converged_runs']}/{s['num_runs']}  "
            f"consensus_q1={s['consensus_q_eq_1_frac']*100:.1f}%",
            flush=True,
        )
    print(f"\n[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
