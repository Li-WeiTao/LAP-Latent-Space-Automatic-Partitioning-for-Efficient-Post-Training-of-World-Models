"""Vectorized Controlled-PARC fitting for state-only deployment routing.

This implements the Voronoi-separation variant of PARC for multi-output
controlled dynamics. Region assignment is informed by local prediction of a
latent response from state and action, while the deployable partition remains
a deterministic function of state alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ControlledPARCConfig:
    num_clusters: int
    seed: int
    alpha: float = 1.0e-5
    sigma: float = 1.0
    max_iter: int = 15
    cost_tol: float = 1.0e-4
    kmeans_n_init: int = 10
    min_cluster_size: int = 256
    cpu_threads: int = 4
    routable_updates: bool = False
    router_prototypes_per_region: int = 8


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1.0e-12))


def _soft_router_proposal(
    states: np.ndarray,
    fit_cost: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Optimize a smooth state-only routing objective, then hard-route later.

    The response residuals are fixed during this block-coordinate step. No
    action or next-state information enters the deployed routing function.
    """

    n, dim = states.shape
    k = len(centroids)
    baseline = max(float(np.mean(fit_cost[np.arange(n), labels])), 1.0e-12)
    costs = (fit_cost - fit_cost.min(axis=1, keepdims=True)) / baseline
    states64 = states.astype(np.float64, copy=False)
    costs64 = costs.astype(np.float64, copy=False)
    initial = centroids.astype(np.float64)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        raw = flat.reshape(k, dim)
        norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1.0e-12)
        unit = raw / norms
        logits = temperature * (states64 @ unit.T)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected = np.sum(probabilities * costs64, axis=1)
        loss = float(expected.mean() + 1.0e-3 * np.square(unit - initial).sum())
        logit_grad = probabilities * (costs64 - expected[:, None])
        unit_grad = (
            temperature * (logit_grad.T @ states64) / n
            + 2.0e-3 * (unit - initial)
        )
        raw_grad = (
            unit_grad - np.sum(unit_grad * unit, axis=1, keepdims=True) * unit
        ) / norms
        return loss, raw_grad.ravel()

    result = minimize(
        objective,
        initial.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 60, "ftol": 1.0e-9},
    )
    return _unit(result.x.reshape(k, dim))


def fit_controlled_parc(
    routing_states: np.ndarray,
    actions: np.ndarray,
    responses: np.ndarray,
    config: ControlledPARCConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a dynamics-informed spherical Voronoi partition.

    ``routing_states`` are the only features used by the deployable router.
    The local affine response models use both ``routing_states`` and
    ``actions``. The implementation follows PARC block-coordinate updates,
    with one vectorized multi-output ridge fit per region.
    """

    states = _unit(routing_states)
    actions = np.asarray(actions, dtype=np.float32)
    responses = np.asarray(responses, dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2 or responses.ndim != 2:
        raise ValueError("Controlled PARC inputs must all be rank-2 arrays")
    if not (len(states) == len(actions) == len(responses)):
        raise ValueError("Controlled PARC inputs must have equal row counts")
    if config.num_clusters < 2 or config.num_clusters > len(states):
        raise ValueError("invalid Controlled PARC cluster count")
    if config.min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")
    if config.router_prototypes_per_region < 1:
        raise ValueError("router_prototypes_per_region must be positive")

    # Ridge is solved in float64 because the frozen action embedding can be
    # rank deficient.  With the predeclared small ridge coefficient, float32
    # Cholesky can otherwise trigger a least-squares fallback.
    design = np.concatenate((states, actions), axis=1).astype(np.float64)
    response_fit = responses.astype(np.float64)
    with threadpool_limits(limits=config.cpu_threads):
        labels = KMeans(
            n_clusters=config.num_clusters,
            init="k-means++",
            n_init=config.kmeans_n_init,
            random_state=config.seed,
            algorithm="lloyd",
        ).fit_predict(states).astype(np.int64)

    router_owners = np.arange(config.num_clusters, dtype=np.int64)
    if config.routable_updates:
        # Every regression fit must use labels that the state-only router can
        # reproduce.  The unconstrained residual assignment below is only a
        # proposal for moving the Voronoi prototypes, never a training label.
        centroids = _unit(
            np.stack([states[labels == k].mean(axis=0) for k in range(config.num_clusters)])
        )
        labels = router_owners[(states @ centroids.T).argmax(axis=1)]

    effective_sigma = float(config.sigma) / float(len(states))
    history: list[dict] = []
    previous_cost = np.inf
    converged = False
    latent_labels = labels.copy()
    termination_reason = "max_iter"

    for iteration in range(1, config.max_iter + 1):
        counts = np.bincount(labels, minlength=config.num_clusters)
        if np.any(counts < config.min_cluster_size):
            raise RuntimeError(
                "Controlled PARC produced an undersized region before deployment: "
                f"{counts.tolist()}"
            )

        if not config.routable_updates:
            centroids = _unit(
                np.stack([states[labels == k].mean(axis=0) for k in range(config.num_clusters)])
            )
        fit_cost = np.empty((len(states), config.num_clusters), dtype=np.float32)
        with threadpool_limits(limits=config.cpu_threads):
            for k in range(config.num_clusters):
                mask = labels == k
                ridge = Ridge(
                    alpha=float(config.alpha) * float(counts[k]) / float(len(states)),
                    fit_intercept=True,
                    solver="cholesky",
                )
                ridge.fit(design[mask], response_fit[mask])
                residual = response_fit - ridge.predict(design)
                fit_cost[:, k] = np.square(residual).sum(axis=1).astype(np.float32)

        similarities = states @ centroids.T
        region_similarity = np.stack(
            [similarities[:, router_owners == k].max(axis=1)
             for k in range(config.num_clusters)], axis=1
        )
        separation_cost = np.maximum(
            np.float32(0.0),
            np.float32(2.0) - np.float32(2.0) * region_similarity,
        )
        joint_cost = fit_cost + np.float32(effective_sigma) * separation_cost
        new_labels = joint_cost.argmin(axis=1).astype(np.int64)
        if config.routable_updates:
            latent_labels = new_labels.copy()
            proposed_counts = np.bincount(new_labels, minlength=config.num_clusters)
            if np.any(proposed_counts < config.min_cluster_size):
                raise RuntimeError(
                    "Controlled PARC residual proposal produced an undersized region: "
                    f"{proposed_counts.tolist()}"
                )
            proposed = _unit(
                np.stack([states[new_labels == k].mean(axis=0) for k in range(config.num_clusters)])
            )
            current_cost = float(
                joint_cost[np.arange(len(states)), labels].sum(dtype=np.float64)
            )
            best = None
            # Project the residual proposal into the deployable Voronoi family.
            # A short line search prevents a non-routable proposal from being
            # silently accepted just because its oracle assignment has low loss.
            candidates = []
            if len(centroids) == config.num_clusters:
                candidates.append(("residual_centroid", proposed, router_owners))
                with threadpool_limits(limits=config.cpu_threads):
                    for temperature in (16.0, 48.0):
                        candidates.append(
                            (
                                f"soft_router_t{temperature:g}",
                                _soft_router_proposal(
                                    states, fit_cost, centroids, labels, temperature
                                ),
                                router_owners,
                            )
                        )
            if np.all(proposed_counts >= config.router_prototypes_per_region):
                with threadpool_limits(limits=config.cpu_threads):
                    multi = np.concatenate(
                        [KMeans(
                            n_clusters=config.router_prototypes_per_region,
                            init="k-means++", n_init=1, max_iter=100,
                            random_state=config.seed + 1000 * iteration + k,
                        ).fit(states[new_labels == k]).cluster_centers_
                         for k in range(config.num_clusters)], axis=0
                    )
                multi_owners = np.repeat(
                    np.arange(config.num_clusters, dtype=np.int64),
                    config.router_prototypes_per_region,
                )
                candidates.append(("response_supervised_prototypes", _unit(multi), multi_owners))
            for source, candidate, candidate_owners in candidates:
              for step in ((1.0, 0.5, 0.25, 0.125, 0.0625)
                           if candidate.shape == centroids.shape else (1.0,)):
                trial_centroids = _unit(
                    (1.0 - step) * centroids + step * candidate
                    if candidate.shape == centroids.shape else candidate
                )
                trial_similarity = states @ trial_centroids.T
                trial_labels = candidate_owners[
                    trial_similarity.argmax(axis=1)
                ].astype(np.int64)
                trial_counts = np.bincount(trial_labels, minlength=config.num_clusters)
                if np.any(trial_counts < config.min_cluster_size):
                    continue
                trial_region_similarity = np.stack(
                    [trial_similarity[:, candidate_owners == k].max(axis=1)
                     for k in range(config.num_clusters)], axis=1
                )
                trial_separation = np.maximum(
                    np.float32(0.0),
                    np.float32(2.0) - np.float32(2.0) * trial_region_similarity,
                )
                trial_cost = float(
                    (fit_cost + np.float32(effective_sigma) * trial_separation)[
                        np.arange(len(states)), trial_labels
                    ].sum(dtype=np.float64)
                )
                if trial_cost < current_cost - config.cost_tol and (
                    best is None or trial_cost < best[0]
                ):
                    best = (
                        trial_cost, trial_centroids, trial_labels,
                        step, source, candidate_owners,
                    )
            if best is None:
                converged = True
                termination_reason = "no_improving_routable_step"
                history.append(
                    {
                        "iteration": iteration,
                        "cost": current_cost,
                        "improvement": 0.0,
                        "changed_assignments": 0,
                        "cluster_counts": counts.tolist(),
                        "proposal_to_router_disagreement": float(
                            np.mean(latent_labels != labels)
                        ),
                        "accepted_step": 0.0,
                        "accepted_source": "none",
                    }
                )
                break
            (total_cost, centroids, new_labels, accepted_step,
             accepted_source, router_owners) = best
            improvement = float(current_cost - total_cost)
        else:
            total_cost = float(
                joint_cost[np.arange(len(states)), new_labels].sum(dtype=np.float64)
            )
            improvement = float(previous_cost - total_cost)
        changed = int(np.count_nonzero(new_labels != labels))
        entry = {
            "iteration": iteration,
            "cost": total_cost,
            "improvement": improvement,
            "changed_assignments": changed,
            "cluster_counts": np.bincount(
                new_labels, minlength=config.num_clusters
            ).tolist(),
        }
        if config.routable_updates:
            entry["proposal_to_router_disagreement"] = float(
                np.mean(latent_labels != new_labels)
            )
            entry["accepted_step"] = accepted_step
            entry["accepted_source"] = accepted_source
        history.append(entry)
        labels = new_labels
        if not config.routable_updates:
            latent_labels = labels.copy()
        if changed == 0 or (
            not config.routable_updates
            and np.isfinite(previous_cost)
            and improvement <= config.cost_tol
        ):
            converged = True
            termination_reason = "stable_router_labels" if changed == 0 else "cost_tol"
            break
        previous_cost = total_cost

    counts = np.bincount(
        labels if config.routable_updates else latent_labels,
        minlength=config.num_clusters,
    )
    if np.any(counts < config.min_cluster_size):
        raise RuntimeError(
            "Controlled PARC produced an undersized final latent region: "
            f"{counts.tolist()}"
        )
    if not config.routable_updates:
        centroids = _unit(
            np.stack(
                [states[latent_labels == k].mean(axis=0) for k in range(config.num_clusters)]
            )
        )
    deployed_labels = router_owners[(states @ centroids.T).argmax(axis=1)]
    if config.routable_updates and not np.array_equal(labels, deployed_labels):
        raise RuntimeError("routable Controlled PARC changed labels after fitting")
    deployed_counts = np.bincount(deployed_labels, minlength=config.num_clusters)
    if np.any(deployed_counts < config.min_cluster_size):
        raise RuntimeError(
            "Controlled PARC state-only router produced an undersized region: "
            f"{deployed_counts.tolist()}"
        )

    metadata = {
        "algorithm": (
            "controlled_parc_routable_voronoi"
            if config.routable_updates else "controlled_parc_voronoi"
        ),
        "implementation": "vectorized_multioutput_parc_updates",
        "num_clusters": config.num_clusters,
        "seed": config.seed,
        "alpha": config.alpha,
        "sigma": config.sigma,
        "effective_sigma": effective_sigma,
        "max_iter": config.max_iter,
        "cost_tol": config.cost_tol,
        "kmeans_n_init": config.kmeans_n_init,
        "min_cluster_size": config.min_cluster_size,
        "iterations": len(history),
        "converged": converged,
        "termination_reason": termination_reason,
        "routable_updates": config.routable_updates,
        "accepted_routable_updates": sum(
            row.get("accepted_source", "none") != "none" for row in history
        ),
        "router_prototype_region_ids": router_owners.tolist(),
        "router_prototypes_per_region": config.router_prototypes_per_region,
        "history": history,
        "latent_assignment_counts": counts.tolist(),
        "deployed_assignment_counts": deployed_counts.tolist(),
        "latent_to_deployed_disagreement": float(
            np.mean((labels if config.routable_updates else latent_labels) != deployed_labels)
        ),
        "last_proposal_to_deployed_disagreement": float(
            np.mean(latent_labels != deployed_labels)
        ),
        "regression_features": "[zscore_l2(z_t), zscore(act_emb_t)]",
        "response_target": "zscore(z_{t+1} - z_t)",
        "partition_features": "zscore_l2(z_t) only",
        "separation": "spherical_voronoi",
        "routing": "zscore_l2_spherical_voronoi",
    }
    return centroids.astype(np.float32), deployed_labels, metadata
