#!/usr/bin/env python3
"""Latent preprocessing stability experiment (K=3, 20 clustering seeds).

Compares 7 preprocessing variants under fixed full-data PyTorch spherical K-means.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from latent_cluster_common import (  # noqa: E402
    default_embed_dir,
    l2_normalize_rows,
    load_all_train_dedup_latent_vectors,
    resolve_cluster_torch_device,
)

EPS = 1e-6
PREPROCESS_NAMES = (
    "raw_l2",
    "center_l2",
    "zscore_l2",
    "pca64_l2",
    "pca128_l2",
    "pca128_shrink001_l2",
    "pca192_shrink001_l2",
)
DEFAULT_SEEDS = list(range(20))
K = 3


@dataclass
class PcaStats:
    mu: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray


@dataclass
class PreprocessState:
    name: str
    X: np.ndarray
    params: dict


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=THIS_DIR.parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def fit_pca_stats(Z: np.ndarray) -> PcaStats:
    mu = Z.mean(axis=0, dtype=np.float64)
    Zc = (Z - mu).astype(np.float64)
    cov = (Zc.T @ Zc) / Z.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return PcaStats(mu=mu.astype(np.float32), evals=evals[order].astype(np.float32), evecs=evecs[:, order].astype(np.float32))


def row_l2(X: np.ndarray) -> np.ndarray:
    return l2_normalize_rows(X.astype(np.float32))


def apply_raw_l2(Z: np.ndarray) -> tuple[np.ndarray, dict]:
    return row_l2(Z), {}


def apply_center_l2(Z: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, dict]:
    X = row_l2(Z - mu)
    return X, {"mu": mu}


def apply_zscore_l2(Z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, dict]:
    Zs = (Z - mu) / (sigma + EPS)
    return row_l2(Zs), {"mu": mu, "sigma": sigma, "eps": EPS}


def apply_pca_l2(Z: np.ndarray, stats: PcaStats, d: int) -> tuple[np.ndarray, dict]:
    Zc = Z - stats.mu
    Y = Zc @ stats.evecs[:, :d]
    return row_l2(Y), {"pca_dim": d, "mu": stats.mu, "evecs_d": stats.evecs[:, :d]}


def apply_shrink_whiten_l2(
    Z: np.ndarray,
    stats: PcaStats,
    d: int,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    Zc = (Z - stats.mu).astype(np.float64)
    evecs_d = stats.evecs[:, :d]
    evals_d = stats.evals[:d].astype(np.float64)
    lam_bar = float(stats.evals.astype(np.float64).mean())
    scale = np.sqrt(evals_d + alpha * lam_bar)
    Y = (Zc @ evecs_d) / scale[None, :]
    return row_l2(Y.astype(np.float32)), {
        "pca_dim": d,
        "alpha": alpha,
        "lam_bar": lam_bar,
        "mu": stats.mu,
        "evecs_d": evecs_d,
        "evals_d": evals_d.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def build_all_preprocesses(Z: np.ndarray) -> list[PreprocessState]:
    mu = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = Z.std(axis=0, dtype=np.float64).astype(np.float32)
    pca = fit_pca_stats(Z)

    builders: list[tuple[str, np.ndarray, dict]] = []
    X, p = apply_raw_l2(Z)
    builders.append(("raw_l2", X, p))

    X, p = apply_center_l2(Z, mu)
    builders.append(("center_l2", X, p))

    X, p = apply_zscore_l2(Z, mu, sigma)
    builders.append(("zscore_l2", X, p))

    for d, name in ((64, "pca64_l2"), (128, "pca128_l2")):
        X, p = apply_pca_l2(Z, pca, d)
        builders.append((name, X, p))

    X, p = apply_shrink_whiten_l2(Z, pca, 128, alpha=0.01)
    builders.append(("pca128_shrink001_l2", X, p))

    X, p = apply_shrink_whiten_l2(Z, pca, 192, alpha=0.01)
    builders.append(("pca192_shrink001_l2", X, p))

    assert [b[0] for b in builders] == list(PREPROCESS_NAMES)
    return [PreprocessState(name=n, X=X, params=p) for n, X, p in builders]


def spherical_objective(Xn: torch.Tensor, centroids: torch.Tensor) -> float:
    scores = Xn @ centroids.T
    return float(scores.max(dim=1).values.mean().detach().cpu())


def cluster_torch_spherical_kmeans_tracked(
    X: np.ndarray,
    *,
    num_clusters: int,
    seed: int,
    niter: int,
    torch_device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    g = torch.Generator(device=torch_device)
    g.manual_seed(seed)

    Xn_np = row_l2(X)
    data = torch.from_numpy(Xn_np).to(torch_device)
    n, _ = data.shape
    perm = torch.randperm(n, generator=g, device=torch_device)
    centroids = F.normalize(data[perm[:num_clusters]].clone(), dim=1)

    objectives: list[float] = []
    t0 = time.perf_counter()
    for _ in range(niter):
        scores = data @ centroids.T
        labels_t = scores.argmax(dim=1)

        new_centroids = []
        for k in range(num_clusters):
            mask = labels_t == k
            if int(mask.sum()) == 0:
                idx = int(torch.randint(0, n, (1,), generator=g, device=torch_device).item())
                new_centroids.append(data[idx])
            else:
                new_centroids.append(F.normalize(data[mask].mean(dim=0), dim=0))
        centroids = torch.stack(new_centroids, dim=0)
        objectives.append(spherical_objective(data, centroids))
    fit_sec = time.perf_counter() - t0

    scores = data @ centroids.T
    labels = scores.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    centroids_np = centroids.detach().cpu().numpy().astype(np.float32)

    obj_final = float(objectives[-1])
    obj_last10 = objectives[-10:]
    obj_delta_last10 = float(obj_last10[-1] - obj_last10[0])
    obj_still_changing = abs(obj_delta_last10) > 1e-5

    return centroids_np, labels, {
        "fit_sec": float(fit_sec),
        "objective_final": obj_final,
        "objective_last10": obj_last10,
        "objective_delta_last10": obj_delta_last10,
        "objective_still_changing": obj_still_changing,
        "niter": niter,
        "init": "random_k_samples",
        "backend": "torch_spherical_kmeans_full_data",
        "device": str(torch_device),
    }


def sorted_cluster_fracs(labels: np.ndarray, k: int = K) -> tuple[float, float, float]:
    fracs = np.bincount(labels, minlength=k).astype(np.float64) / len(labels)
    fracs.sort()
    return float(fracs[0]), float(fracs[1]), float(fracs[2])


def margin_stats(X: np.ndarray, centroids: np.ndarray) -> dict:
    Xn = row_l2(X)
    Cn = row_l2(centroids)
    scores = Xn @ Cn.T
    part = np.partition(scores, K - 2, axis=1)
    top2 = np.sort(part[:, -2:], axis=1)
    margins = top2[:, 1] - top2[:, 0]
    return {
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_p25": float(np.quantile(margins, 0.25)),
        "margin_lt_0p01_frac": float((margins < 0.01).mean()),
    }


def hungarian_label_map(labels_a: np.ndarray, labels_b: np.ndarray, k: int = K) -> np.ndarray:
    cont = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(labels_a, labels_b):
        cont[a, b] += 1
    row_ind, col_ind = linear_sum_assignment(-cont)
    b_to_a = np.zeros(k, dtype=np.int64)
    for a_label, b_label in zip(row_ind, col_ind):
        b_to_a[b_label] = a_label
    return b_to_a


def align_labels_b_to_a(labels_b: np.ndarray, b_to_a: np.ndarray) -> np.ndarray:
    return b_to_a[labels_b]


def hungarian_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, np.ndarray]:
    b_to_a = hungarian_label_map(labels_a, labels_b)
    aligned_b = align_labels_b_to_a(labels_b, b_to_a)
    agree = float((labels_a == aligned_b).mean())
    return agree, b_to_a


def hungarian_centroid_cosine(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    b_to_a: np.ndarray,
) -> float:
    Ca = row_l2(centroids_a)
    Cb = row_l2(centroids_b)
    sims = [float(np.dot(Ca[b_to_a[b]], Cb[b])) for b in range(K)]
    return float(np.mean(sims))


def per_run_metrics(
    *,
    preprocess: str,
    seed: int,
    labels: np.ndarray,
    centroids: np.ndarray,
    X: np.ndarray,
    cluster_info: dict,
    global_idx: np.ndarray,
) -> dict:
    fmin, fmid, fmax = sorted_cluster_fracs(labels)
    m = margin_stats(X, centroids)
    return {
        "preprocess": preprocess,
        "seed": seed,
        "objective_final": cluster_info["objective_final"],
        "objective_delta_last10": cluster_info["objective_delta_last10"],
        "objective_still_changing": cluster_info["objective_still_changing"],
        "fit_sec": cluster_info["fit_sec"],
        "cluster_frac_min": fmin,
        "cluster_frac_mid": fmid,
        "cluster_frac_max": fmax,
        **m,
        "global_idx": global_idx,
        "labels": labels,
        "centroids": centroids,
    }


def pairwise_stability_for_preprocess(runs: list[dict]) -> tuple[list[dict], np.ndarray, list[float]]:
    seeds = [r["seed"] for r in runs]
    seed_to_idx = {s: i for i, s in enumerate(seeds)}
    n = len(seeds)
    ari_mat = np.eye(n, dtype=np.float64)
    rows: list[dict] = []
    a_s: dict[int, list[float]] = {s: [] for s in seeds}

    for i, j in combinations(range(n), 2):
        ra, rb = runs[i], runs[j]
        la, lb = ra["labels"], rb["labels"]
        ari = adjusted_rand_score(la, lb)
        nmi = normalized_mutual_info_score(la, lb)
        agree, perm = hungarian_agreement(la, lb)
        cos = hungarian_centroid_cosine(ra["centroids"], rb["centroids"], perm)
        ari_mat[i, j] = ari_mat[j, i] = ari
        a_s[ra["seed"]].append(ari)
        a_s[rb["seed"]].append(ari)
        rows.append(
            {
                "preprocess": ra["preprocess"],
                "seed_a": ra["seed"],
                "seed_b": rb["seed"],
                "ari": ari,
                "nmi": nmi,
                "label_agreement": agree,
                "centroid_cosine": cos,
            }
        )

    mean_a_per_seed = {s: float(np.mean(v)) for s, v in a_s.items()}
    return rows, ari_mat, [mean_a_per_seed[s] for s in seeds]


def consensus_stats(runs: list[dict]) -> dict:
    mean_a = {}
    for r in runs:
        others = [adjusted_rand_score(r["labels"], o["labels"]) for o in runs if o["seed"] != r["seed"]]
        mean_a[r["seed"]] = float(np.mean(others))
    medoid_seed = max(mean_a, key=mean_a.get)
    medoid = next(r for r in runs if r["seed"] == medoid_seed)

    aligned_labels = []
    for r in runs:
        if r["seed"] == medoid_seed:
            aligned_labels.append(r["labels"])
        else:
            _, b_to_a = hungarian_agreement(medoid["labels"], r["labels"])
            aligned_labels.append(align_labels_b_to_a(r["labels"], b_to_a))

    votes = np.stack(aligned_labels, axis=1)
    max_counts = np.zeros(votes.shape[0], dtype=np.float64)
    for k in range(K):
        max_counts = np.maximum(max_counts, (votes == k).sum(axis=1))
    q = max_counts / len(runs)

    return {
        "medoid_seed": int(medoid_seed),
        "consensus_mean_q": float(q.mean()),
        "consensus_q_eq_1_frac": float((q == 1.0).mean()),
        "consensus_q_ge_0p9_frac": float((q >= 0.9).mean()),
        "consensus_q_lt_0p7_frac": float((q < 0.7).mean()),
        "per_seed_mean_ari": mean_a,
    }


def summarize_pairwise(rows: list[dict]) -> dict:
    aris = np.array([r["ari"] for r in rows], dtype=np.float64)
    nmis = np.array([r["nmi"] for r in rows], dtype=np.float64)
    agrees = np.array([r["label_agreement"] for r in rows], dtype=np.float64)
    coss = np.array([r["centroid_cosine"] for r in rows], dtype=np.float64)
    return {
        "pairwise_ari_mean": float(aris.mean()),
        "pairwise_ari_std": float(aris.std()),
        "pairwise_ari_median": float(np.median(aris)),
        "pairwise_ari_min": float(aris.min()),
        "pairwise_ari_max": float(aris.max()),
        "pairwise_nmi_mean": float(nmis.mean()),
        "pairwise_nmi_std": float(nmis.std()),
        "pairwise_label_agreement_mean": float(agrees.mean()),
        "pairwise_centroid_cosine_mean": float(coss.mean()),
        "num_pairs": int(len(rows)),
    }


def save_preprocess_params(out_dir: Path, states: list[PreprocessState], pca: PcaStats, mu: np.ndarray, sigma: np.ndarray) -> None:
    payload = {
        "mu": mu,
        "sigma": sigma,
        "pca_evals": pca.evals,
        "pca_evecs": pca.evecs,
    }
    for st in states:
        for k, v in st.params.items():
            payload[f"{st.name}__{k}"] = v
    np.savez_compressed(out_dir / "preprocess_params.npz", **payload)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


def make_figures(
    out_dir: Path,
    summaries: list[dict],
    ari_mats: dict[str, np.ndarray],
    per_run_rows: list[dict],
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = [s["preprocess"] for s in summaries]
    ari_means = [s["pairwise_ari_mean"] for s in summaries]
    ari_stds = [s["pairwise_ari_std"] for s in summaries]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        im = ax.imshow(ari_mats[name], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name)
        ax.set_xlabel("seed idx")
        ax.set_ylabel("seed idx")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[-1].axis("off")
    fig.suptitle("Pairwise ARI heatmaps (20 seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ari_heatmaps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, ari_means, yerr=ari_stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("mean pairwise ARI")
    ax.set_title("Stability by preprocessing")
    ax.axhline(0.40, color="gray", ls="--", lw=0.8, label="ARI=0.40")
    ax.axhline(0.70, color="gray", ls=":", lw=0.8, label="ARI=0.70")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "stability_by_preprocess.png", dpi=150)
    plt.close(fig)

    obj_by_name = {n: [] for n in names}
    ari_by_name = {n: [] for n in names}
    for row in per_run_rows:
        obj_by_name[row["preprocess"]].append(row["objective_final"])
    for s in summaries:
        ari_by_name[s["preprocess"]] = s["pairwise_ari_mean"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        ax.scatter(
            [ari_by_name[name]] * len(obj_by_name[name]),
            obj_by_name[name],
            label=name,
            alpha=0.7,
            s=20,
        )
    ax.set_xlabel("mean pairwise ARI (preprocess-level)")
    ax.set_ylabel("objective_final per seed")
    ax.set_title("Objective vs stability")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "objective_vs_stability.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embed-dir", type=Path, default=default_embed_dir())
    p.add_argument(
        "--out-dir",
        type=Path,
        default=THIS_DIR / "results" / "latent_preprocess_stability_k3",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--niter", type=int, default=100)
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
    print(f"[load] {len(Z)} vectors in {load_sec:.2f}s", flush=True)

    mu = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
    sigma = Z.std(axis=0, dtype=np.float64).astype(np.float32)
    pca = fit_pca_stats(Z)
    states = build_all_preprocesses(Z)
    save_preprocess_params(args.out_dir, states, pca, mu, sigma)

    per_run_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    summaries: list[dict] = []
    ari_mats: dict[str, np.ndarray] = {}
    runs_by_preprocess: dict[str, list[dict]] = {}

    for st in states:
        print(f"\n==== preprocess={st.name} ====", flush=True)
        runs: list[dict] = []
        for seed in args.seeds:
            centroids, labels, cinfo = cluster_torch_spherical_kmeans_tracked(
                st.X,
                num_clusters=K,
                seed=seed,
                niter=args.niter,
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

            label_path = args.out_dir / "labels" / f"{st.name}_seed{seed}.npz"
            np.savez_compressed(
                label_path,
                global_idx=global_idx,
                labels=labels,
                centroids=centroids,
            )

            per_run_rows.append(
                {
                    "preprocess": st.name,
                    "seed": seed,
                    "objective_final": run["objective_final"],
                    "objective_delta_last10": run["objective_delta_last10"],
                    "objective_still_changing": int(run["objective_still_changing"]),
                    "fit_sec": run["fit_sec"],
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
                f"  seed={seed:2d}  obj={run['objective_final']:.4f}  "
                f"fit={run['fit_sec']:.3f}s  fracs=({run['cluster_frac_min']:.3f},"
                f"{run['cluster_frac_mid']:.3f},{run['cluster_frac_max']:.3f})",
                flush=True,
            )

        runs_by_preprocess[st.name] = runs
        pw_rows, ari_mat, _ = pairwise_stability_for_preprocess(runs)
        pairwise_rows.extend(pw_rows)
        ari_mats[st.name] = ari_mat
        consensus = consensus_stats(runs)
        summary = {
            "preprocess": st.name,
            **summarize_pairwise(pw_rows),
            **{k: v for k, v in consensus.items() if k != "per_seed_mean_ari"},
            "per_seed_mean_ari": consensus["per_seed_mean_ari"],
        }
        summaries.append(summary)

    per_run_fields = list(per_run_rows[0].keys())
    pairwise_fields = list(pairwise_rows[0].keys())
    write_csv(args.out_dir / "per_run_metrics.csv", per_run_rows, per_run_fields)
    write_csv(args.out_dir / "pairwise_stability.csv", pairwise_rows, pairwise_fields)

    summary_csv_rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if k not in ("per_seed_mean_ari",)}
        summary_csv_rows.append(row)
    write_csv(
        args.out_dir / "stability_summary.csv",
        summary_csv_rows,
        list(summary_csv_rows[0].keys()),
    )

    meta = {
        "version_label": "latent_preprocess_stability_v1",
        "embed_dir": str(args.embed_dir),
        "out_dir": str(args.out_dir),
        "num_unique_timesteps": data_stats["num_unique_timesteps"],
        "latent_dim": data_stats["latent_dim"],
        "preprocess_names": list(PREPROCESS_NAMES),
        "pca_dims": {"pca64_l2": 64, "pca128_l2": 128, "pca128_shrink001_l2": 128, "pca192_shrink001_l2": 192},
        "shrinkage_alpha": 0.01,
        "clustering_seeds": args.seeds,
        "num_clusters": K,
        "niter": args.niter,
        "init": "random_k_samples",
        "cluster_backend": "torch_spherical_kmeans_full_data",
        "git_commit": git_commit(),
        "device": str(torch_device),
        "dtype": "float32",
        "load_sec": load_sec,
        "wall_sec": time.perf_counter() - t0,
        "num_runs": len(per_run_rows),
        "summaries": summaries,
    }
    with (args.out_dir / "preprocess_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    make_figures(args.out_dir, summaries, ari_mats, per_run_rows)

    print("\n==== stability summary (mean pairwise ARI) ====", flush=True)
    for s in sorted(summaries, key=lambda x: -x["pairwise_ari_mean"]):
        print(
            f"  {s['preprocess']:24s}  ARI={s['pairwise_ari_mean']:.4f}±{s['pairwise_ari_std']:.4f}  "
            f"consensus_q1={s['consensus_q_eq_1_frac']*100:.1f}%  "
            f"medoid_seed={s['medoid_seed']}",
            flush=True,
        )
    print(f"\n[done] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
