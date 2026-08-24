#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


TASKS = ("tworoom", "pusht", "reacher", "cube")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--clusters", default="2,3,4")
    p.add_argument("--partition-seeds", default="0,1,2")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--transition-stride", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--chunk", type=int, default=200000)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--audit-dir", type=Path, default=Path("/tmp/lap_k4_geometry_audit"))
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def csv_ints(value):
    return tuple(int(x) for x in value.split(",") if x)


def gate_manifest(repo, task, k):
    return repo / f"experiments/{task}/results/auto_gate_complete_k{k}/auto/partition/manifest.json"


def label_path(repo, task, k, seed):
    if k in (2, 4):
        return repo / f"experiments/{task}/matrix_k{k}/partitions/spectral/seed{seed}/cluster_labels.npz"
    if task == "tworoom":
        return repo / f"experiments/tworoom/results/latent_landmark_spectral_k3/spectral_M20000_k30_P16_seed{seed}/cluster_labels.npz"
    return repo / f"experiments/{task}/matrix/partitions/spectral/seed{seed}/cluster_labels.npz"


def load_unique(cache_path, frameskip):
    with np.load(cache_path, allow_pickle=False) as z:
        emb = np.asarray(z["emb"], dtype=np.float32)
        starts = np.asarray(z["region_starts"], dtype=np.int64)
    ids = (starts[:, None] + np.arange(emb.shape[1], dtype=np.int64)[None, :] * frameskip).reshape(-1)
    flat = emb.reshape(-1, emb.shape[-1])
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    flat = flat[order]
    keep = np.r_[True, ids[1:] != ids[:-1]]
    return flat[keep], ids[keep]


def load_labels(path, ids):
    with np.load(path, allow_pickle=False) as z:
        labels = np.asarray(z["labels"], dtype=np.int64)
        source_ids = None
        for key in ("sample_ids", "global_idx"):
            if key in z.files:
                candidate = np.asarray(z[key], dtype=np.int64)
                if len(candidate) == len(labels):
                    source_ids = candidate
                    break
    if source_ids is None:
        if len(labels) != len(ids):
            raise ValueError(f"cannot align labels in {path}")
        return labels
    if np.array_equal(source_ids, ids):
        return labels
    order = np.argsort(source_ids)
    pos = np.searchsorted(source_ids[order], ids)
    if np.any(pos == len(order)) or not np.array_equal(source_ids[order][pos], ids):
        raise ValueError(f"label ids do not cover latent ids in {path}")
    return labels[order][pos]


def transition_rows(ids, h5_path, stride):
    nxt = np.searchsorted(ids, ids + stride)
    valid = nxt < len(ids)
    valid[valid] &= ids[nxt[valid]] == ids[valid] + stride
    left = np.flatnonzero(valid)
    right = nxt[valid]
    with h5py.File(h5_path, "r", swmr=True) as h:
        ep_key = "episode_idx" if "episode_idx" in h else "ep_idx"
        ep0 = np.asarray(h[ep_key][ids[left]], dtype=np.int64)
        ep1 = np.asarray(h[ep_key][ids[right]], dtype=np.int64)
        same = ep0 == ep1
        left, right = left[same], right[same]
        actions = np.asarray(h["action"][ids[left]], dtype=np.float64)
    return left, right, actions


def sufficient_stats(x, left, right, actions, labels_by_seed, k, ridge, chunk):
    mean = actions.mean(axis=0)
    scale = actions.std(axis=0) + 1e-12
    a = (actions - mean) / scale
    p, d = a.shape[1], x.shape[1]
    xtx = np.zeros((p + 1, p + 1), dtype=np.float64)
    xty = np.zeros((p + 1, d), dtype=np.float64)
    y2 = 0.0
    region = {
        seed: [(np.zeros_like(xtx), np.zeros_like(xty), 0) for _ in range(k)]
        for seed in labels_by_seed
    }
    for begin in range(0, len(left), chunk):
        end = min(begin + chunk, len(left))
        aa = a[begin:end]
        xx = np.empty((len(aa), p + 1), dtype=np.float64)
        xx[:, 0] = 1.0
        xx[:, 1:] = aa
        yy = x[right[begin:end]].astype(np.float64) - x[left[begin:end]].astype(np.float64)
        xtx += xx.T @ xx
        xty += xx.T @ yy
        y2 += float(np.square(yy).sum(dtype=np.float64))
        for seed, labels in labels_by_seed.items():
            local = labels[left[begin:end]]
            for r in range(k):
                mask = local == r
                if not np.any(mask):
                    continue
                rxx, rxy, rn = region[seed][r]
                rx = xx[mask]
                ry = yy[mask]
                rxx += rx.T @ rx
                rxy += rx.T @ ry
                region[seed][r] = (rxx, rxy, rn + int(mask.sum()))
    penalty = np.eye(p + 1) * ridge
    penalty[0, 0] = 0.0
    global_coef = np.linalg.solve(xtx + penalty, xty)
    sse = y2 - 2.0 * float(np.sum(global_coef * xty)) + float(np.sum(global_coef * (xtx @ global_coef)))
    residual_energy = sse / len(left)
    cov = (a.T @ a) / len(a)
    vals, vecs = np.linalg.eigh(cov)
    sqrt_cov = (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
    fitted = {}
    for seed, stats in region.items():
        fitted[seed] = []
        for rxx, rxy, rn in stats:
            fitted[seed].append((np.linalg.solve(rxx + penalty, rxy), rn))
    return fitted, residual_energy, sqrt_cov, mean, scale


def jacobian_metrics(c1, c2, sqrt_cov):
    f1 = sqrt_cov @ c1[1:]
    f2 = sqrt_cov @ c2[1:]
    n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
    cosine = 1.0 - float(np.sum(f1 * f2) / max(n1 * n2, 1e-12))
    log_scale = float(abs(np.log(max(n1, 1e-12) / max(n2, 1e-12))))
    l1, l2 = f1.T, f2.T
    q1 = np.linalg.svd(l1, full_matrices=False)[0]
    q2 = np.linalg.svd(l2, full_matrices=False)[0]
    rank = min(q1.shape[1], q2.shape[1])
    chordal = float(np.linalg.norm(q1 @ q1.T - q2 @ q2.T, "fro") / np.sqrt(2.0 * rank))
    nuclear = float(np.linalg.svd(l1.T @ l2, compute_uv=False).sum())
    denom = n1 * n1 + n2 * n2 + 1e-12
    bures = float(max(n1 * n1 + n2 * n2 - 2.0 * nuclear, 0.0) / denom)
    return cosine, log_scale, chordal, bures


def fit_subset(x, ids, actions_by_id, labels, rows, k, ridge, transition_stride):
    p, d = actions_by_id.shape[1], x.shape[1]
    a = actions_by_id[rows]
    a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)
    cov = (a.T @ a) / len(a)
    vals, vecs = np.linalg.eigh(cov)
    sqrt_cov = (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
    nxt = np.searchsorted(ids, ids[rows] + transition_stride)
    ok = (nxt < len(ids)) & (ids[nxt] == ids[rows] + transition_stride)
    rows, nxt, a = rows[ok], nxt[ok], a[ok]
    coefs = []
    penalty = np.eye(p + 1) * ridge
    penalty[0, 0] = 0.0
    for r in range(k):
        mask = labels[rows] == r
        xx = np.c_[np.ones(mask.sum()), a[mask]]
        yy = x[nxt[mask]].astype(np.float64) - x[rows[mask]].astype(np.float64)
        coefs.append(np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy))
    return coefs, sqrt_cov


def analyze_task(repo, task, ks, seeds, frameskip, transition_stride, ridge, chunk,
                 audit_dir, cpu_threads):
    manifest = json.loads(gate_manifest(repo, task, 4).read_text())
    x, ids = load_unique(Path(manifest["cache_stats"]["cache"]), frameskip)
    left, right, actions = transition_rows(ids, Path(manifest["data_file"]), transition_stride)
    action_by_row = np.zeros((len(ids), actions.shape[1]), dtype=np.float64)
    action_by_row[left] = actions
    transition_mask = np.zeros(len(ids), dtype=bool)
    transition_mask[left] = True
    pair_rows, summary_rows, jac_rows = [], [], []
    for k in ks:
        labels = {seed: load_labels(label_path(repo, task, k, seed), ids) for seed in seeds}
        fitted, residual, sqrt_cov, _, _ = sufficient_stats(x, left, right, actions, labels, k, ridge, chunk)
        audit = np.load(audit_dir / f"{task}.npz", allow_pickle=False) if k == 4 else None
        for seed in seeds:
            contrasts = []
            jac_values = []
            for i in range(k):
                for j in range(i + 1, k):
                    ci, ni = fitted[seed][i]
                    cj, nj = fitted[seed][j]
                    di = ci[0] - cj[0]
                    ds = sqrt_cov @ (ci[1:] - cj[1:])
                    mean_part = float(np.sum(di * di) / residual)
                    slope_part = float(np.sum(ds * ds) / residual)
                    contrast = mean_part + slope_part
                    contrasts.append(contrast)
                    row = dict(task=task, num_clusters=k, partition_seed=seed, region_left=i,
                               region_right=j, left_transition_count=ni, right_transition_count=nj,
                               pairwise_response_contrast=contrast,
                               pairwise_mean_action_contrast=mean_part,
                               pairwise_action_slope_contrast=slope_part)
                    pair_rows.append(row)
                    vals = jacobian_metrics(ci, cj, sqrt_cov)
                    jac_values.append(vals)
                    jac_rows.append(dict(**row,
                        jacobian_cosine_distance=vals[0],
                        jacobian_log_scale_distance=vals[1],
                        jacobian_subspace_chordal_distance=vals[2],
                        jacobian_bures_distance=vals[3]))
            summary_rows.append(dict(task=task, num_clusters=k, partition_seed=seed,
                transition_count=len(left), action_dim=actions.shape[1],
                minimum_pairwise_response=min(contrasts),
                mean_pairwise_response=float(np.mean(contrasts)),
                pairwise_uniformity_min_over_mean=float(min(contrasts) / np.mean(contrasts))))
            if k == 4:
                audit_ids = np.asarray(audit["sample_ids"], dtype=np.int64)
                audit_x = np.asarray(audit["x"], dtype=np.float32)
                audit_labels = np.asarray(audit[f"y{seed}"], dtype=np.int64)
                z = (audit_x - audit_x.mean(axis=0)) / (audit_x.std(axis=0) + 1e-8)
                z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                nn = NearestNeighbors(n_neighbors=31, metric="cosine", n_jobs=cpu_threads).fit(z)
                neighbors = nn.kneighbors(z, return_distance=False)[:, 1:]
                mixed = np.any(audit_labels[neighbors] != audit_labels[:, None], axis=1)
                pos = np.searchsorted(ids, audit_ids)
                valid = (pos < len(ids)) & (ids[pos] == audit_ids) & mixed
                rows = pos[valid]
                rows = rows[transition_mask[rows]]
                boundary_coefs, boundary_cov = fit_subset(
                    x, ids, action_by_row, labels[seed], rows, k, ridge, transition_stride
                )
                boundary = [jacobian_metrics(boundary_coefs[i], boundary_coefs[j], boundary_cov)[3]
                            for i in range(k) for j in range(i + 1, k)]
                vals = np.asarray(jac_values)
                jac_rows.append(dict(task=task, num_clusters=4, partition_seed=seed,
                    region_left=-1, region_right=-1, left_transition_count=len(rows),
                    right_transition_count=len(rows), pairwise_response_contrast=np.nan,
                    pairwise_mean_action_contrast=np.nan, pairwise_action_slope_contrast=np.nan,
                    jacobian_cosine_distance=float(vals[:, 0].min()),
                    jacobian_log_scale_distance=float(vals[:, 1].min()),
                    jacobian_subspace_chordal_distance=float(vals[:, 2].min()),
                    jacobian_bures_distance=float(vals[:, 3].min()),
                    boundary_min_jacobian_bures_distance=float(min(boundary)),
                    boundary_transition_count=len(rows)))
    return pair_rows, summary_rows, jac_rows


def main():
    args = parse_args()
    repo = args.repo.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ks, seeds = csv_ints(args.clusters), csv_ints(args.partition_seeds)
    all_pair, all_summary, all_jac = [], [], []
    for task in tuple(x for x in args.tasks.split(",") if x):
        pair, summary, jac = analyze_task(
            repo, task, ks, seeds, args.frameskip, args.transition_stride,
            args.ridge, args.chunk, args.audit_dir, args.cpu_threads
        )
        all_pair.extend(pair)
        all_summary.extend(summary)
        all_jac.extend(jac)
        print(f"completed {task}", flush=True)
    pd.DataFrame(all_pair).to_csv(out / "response_geometry_cross_k_pairs.csv", index=False)
    pd.DataFrame(all_summary).to_csv(out / "response_geometry_cross_k_by_seed.csv", index=False)
    jac = pd.DataFrame(all_jac)
    pair = jac[jac["region_left"].ge(0)].drop(
        columns=["boundary_min_jacobian_bures_distance", "boundary_transition_count"],
        errors="ignore",
    )
    fixed_k_seed_summary = pair.groupby(
        ["task", "num_clusters", "partition_seed"], as_index=False
    )[[
        "jacobian_cosine_distance", "jacobian_log_scale_distance",
        "jacobian_subspace_chordal_distance", "jacobian_bures_distance",
    ]].min()
    seed_summary = jac[jac["region_left"].eq(-1)][[
        "task", "num_clusters", "partition_seed",
        "jacobian_cosine_distance", "jacobian_log_scale_distance",
        "jacobian_subspace_chordal_distance", "jacobian_bures_distance",
        "boundary_min_jacobian_bures_distance", "boundary_transition_count",
    ]]
    pair.to_csv(out / "jacobian_fixed_k_pair_metrics.csv", index=False)
    fixed_k_seed_summary.to_csv(out / "jacobian_fixed_k_seed_summary.csv", index=False)
    pair[pair["num_clusters"].eq(4)].to_csv(
        out / "k4_jacobian_pair_metrics.csv", index=False
    )
    seed_summary.to_csv(out / "k4_jacobian_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
