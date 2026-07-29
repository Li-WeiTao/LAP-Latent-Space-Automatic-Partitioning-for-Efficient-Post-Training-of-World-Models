"""Shared full-data spherical K-means (random / K-means++ init, converged)."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

InitMode = Literal["random_k_samples", "kmeanspp"]


def relative_objective_change(prev_obj: float, curr_obj: float) -> float:
    denom = max(abs(prev_obj), 1e-12)
    return abs(curr_obj - prev_obj) / denom


def spherical_objective_tensor(data: torch.Tensor, centroids: torch.Tensor) -> float:
    scores = data @ centroids.T
    return float(scores.max(dim=1).values.mean().detach().cpu())


def init_random_k_samples(
    data: torch.Tensor,
    num_clusters: int,
    g: torch.Generator,
) -> torch.Tensor:
    n = data.shape[0]
    perm = torch.randperm(n, generator=g, device=data.device)
    return F.normalize(data[perm[:num_clusters]].clone(), dim=1)


def init_kmeans_plus_plus_spherical(
    data: torch.Tensor,
    num_clusters: int,
    g: torch.Generator,
) -> torch.Tensor:
    """Spherical K-means++: p_i ∝ 1 - max_c cos(x_i, c)."""
    n = data.shape[0]
    idx0 = int(torch.randint(0, n, (1,), generator=g, device=data.device).item())
    centroids = [data[idx0]]
    for _ in range(1, num_clusters):
        C = torch.stack(centroids, dim=0)
        max_cos = (data @ C.T).max(dim=1).values
        weights = 1.0 - max_cos
        weights = torch.clamp(weights, min=0.0)
        total = weights.sum()
        if float(total) <= 0:
            idx = int(torch.randint(0, n, (1,), generator=g, device=data.device).item())
        else:
            idx = int(torch.multinomial(weights / total, 1, generator=g).item())
        centroids.append(data[idx])
    return F.normalize(torch.stack(centroids, dim=0), dim=1)


def iterate_spherical_kmeans(
    data: torch.Tensor,
    centroids: torch.Tensor,
    *,
    num_clusters: int,
    max_iter: int,
    rel_tol: float,
    patience: int,
    g: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    objectives: list[float] = []
    label_change_fracs: list[float] = []
    rel_obj_changes: list[float] = []
    prev_labels: torch.Tensor | None = None
    stable_rounds = 0
    converged = False
    stop_iter = max_iter

    for it in range(1, max_iter + 1):
        scores = data @ centroids.T
        labels_t = scores.argmax(dim=1)

        if prev_labels is not None:
            label_change_fracs.append(
                float((labels_t != prev_labels).float().mean().detach().cpu())
            )
        prev_labels = labels_t.clone()

        new_centroids = []
        for k in range(num_clusters):
            mask = labels_t == k
            if int(mask.sum()) == 0:
                idx = int(torch.randint(0, data.shape[0], (1,), generator=g, device=data.device).item())
                new_centroids.append(data[idx])
            else:
                new_centroids.append(F.normalize(data[mask].mean(dim=0), dim=0))
        centroids = torch.stack(new_centroids, dim=0)

        obj = spherical_objective_tensor(data, centroids)
        objectives.append(obj)
        if len(objectives) >= 2:
            rel_change = relative_objective_change(objectives[-2], objectives[-1])
            rel_obj_changes.append(rel_change)
            if rel_change < rel_tol:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= patience:
                converged = True
                stop_iter = it
                break

    final_scores = data @ centroids.T
    labels = final_scores.argmax(dim=1)
    objective_final = float(final_scores.max(dim=1).values.mean().detach().cpu())

    info = {
        "objective_final": objective_final,
        "objective_history_final": float(objectives[-1]) if objectives else objective_final,
        "objective_last10": objectives[-10:],
        "objective_delta_last10": float(objectives[-1] - objectives[-10]) if len(objectives) >= 10 else float("nan"),
        "objective_still_changing": not converged,
        "converged": converged,
        "niter": int(stop_iter),
        "max_iter": max_iter,
        "rel_tol": rel_tol,
        "patience": patience,
        "final_rel_obj_change": float(rel_obj_changes[-1]) if rel_obj_changes else float("nan"),
        "mean_label_change_frac": float(np.mean(label_change_fracs)) if label_change_fracs else 0.0,
        "final_label_change_frac": float(
            (labels != prev_labels).float().mean().detach().cpu()
        ) if prev_labels is not None else 0.0,
        "max_label_change_frac": float(np.max(label_change_fracs)) if label_change_fracs else 0.0,
    }
    return centroids, labels, info


def cluster_torch_spherical_kmeans_converged(
    data: torch.Tensor,
    *,
    num_clusters: int,
    seed: int,
    max_iter: int,
    rel_tol: float,
    patience: int,
    init_mode: InitMode = "random_k_samples",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    g = torch.Generator(device=data.device)
    g.manual_seed(seed)

    if init_mode == "random_k_samples":
        centroids = init_random_k_samples(data, num_clusters, g)
    elif init_mode == "kmeanspp":
        centroids = init_kmeans_plus_plus_spherical(data, num_clusters, g)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    t0 = time.perf_counter()
    centroids, labels, info = iterate_spherical_kmeans(
        data,
        centroids,
        num_clusters=num_clusters,
        max_iter=max_iter,
        rel_tol=rel_tol,
        patience=patience,
        g=g,
    )
    info["fit_sec"] = float(time.perf_counter() - t0)
    info["init"] = init_mode
    info["seed"] = seed
    info["backend"] = "torch_spherical_kmeans_full_data_converged"
    info["device"] = str(data.device)
    return centroids, labels, info


def assign_labels_from_centroids(data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    scores = data @ centroids.T
    return scores.argmax(dim=1)
