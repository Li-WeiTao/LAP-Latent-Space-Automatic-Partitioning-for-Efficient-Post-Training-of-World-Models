"""Hierarchical paired eval-block bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import CellData, MethodData, point_estimate


@dataclass
class BootstrapResult:
    method_id: str
    point_estimate: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_low: float
    ci_high: float
    n_train_seeds: int
    n_partition_seeds: int
    n_eval_blocks: int
    draws: np.ndarray | None = None


@dataclass
class ContrastResult:
    baseline_method: str
    point_difference: float
    ci_low: float
    ci_high: float
    pr_gt_zero: float
    n_common_blocks: int
    draws: np.ndarray | None = None


def _draws_official(
    blocks: np.ndarray,
    *,
    n_bootstrap: int,
    batch_size: int,
    shared_eval_idx: np.ndarray,
) -> np.ndarray:
    n_eval = blocks.shape[0]
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        e_idx = shared_eval_idx[start:end]
        draws[start:end] = blocks[e_idx].mean(axis=1)
    return draws


def _draws_learned(
    blocks: np.ndarray,
    *,
    n_bootstrap: int,
    batch_size: int,
    shared_eval_idx: np.ndarray,
    shared_train_idx: np.ndarray,
) -> np.ndarray:
    n_train, n_eval = blocks.shape
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        t_idx = shared_train_idx[start:end]
        e_idx = shared_eval_idx[start:end]
        train_selected = blocks[t_idx, :]  # (B, n_train, n_eval)
        e_expanded = e_idx[:, None, :].repeat(n_train, axis=1)
        gathered = np.take_along_axis(train_selected, e_expanded, axis=2)
        draws[start:end] = gathered.mean(axis=(1, 2))
    return draws


def _draws_episode(
    episodes: np.ndarray,
    *,
    official: bool,
    n_bootstrap: int,
    batch_size: int,
    shared_eval_idx: np.ndarray,
    shared_train_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    rates = episodes * 100.0
    if official:
        n_eval, n_ep = rates.shape
        draws = np.empty(n_bootstrap, dtype=np.float64)
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            size = end - start
            e_idx = shared_eval_idx[start:end]
            ep_idx = rng.integers(0, n_ep, size=(size, n_eval, n_ep))
            block_vals = np.take_along_axis(rates[e_idx], ep_idx, axis=2).mean(axis=2)
            draws[start:end] = block_vals.mean(axis=1)
        return draws

    n_train, n_eval, n_ep = rates.shape
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        size = end - start
        t_idx = shared_train_idx[start:end]
        e_idx = shared_eval_idx[start:end]
        train_selected = rates[t_idx, :]
        e_expanded = e_idx[:, None, :].repeat(n_train, axis=1)
        block_selected = np.take_along_axis(train_selected, e_expanded, axis=2)
        ep_idx = rng.integers(0, n_ep, size=(size, n_train, n_eval, n_ep))
        ep_gathered = np.take_along_axis(
            block_selected[:, :, :, None].repeat(n_ep, axis=3),
            ep_idx,
            axis=3,
        )
        draws[start:end] = ep_gathered.mean(axis=3).mean(axis=(1, 2))
    return draws


def _method_draws(
    method: MethodData,
    *,
    n_bootstrap: int,
    batch_size: int,
    resampling_unit: str,
    shared_eval_idx: np.ndarray,
    shared_train_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if resampling_unit == "episode":
        if method.episodes is None:
            raise ValueError(f"{method.method_id}: episode mode requires episode_successes")
        return _draws_episode(
            method.episodes,
            official=method.official,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            shared_eval_idx=shared_eval_idx,
            shared_train_idx=shared_train_idx,
            rng=rng,
        )
    if method.official:
        return _draws_official(
            method.blocks,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            shared_eval_idx=shared_eval_idx,
        )
    return _draws_learned(
        method.blocks,
        n_bootstrap=n_bootstrap,
        batch_size=batch_size,
        shared_eval_idx=shared_eval_idx,
        shared_train_idx=shared_train_idx,
    )


def bootstrap_cell_with_contrasts(
    cell: CellData,
    *,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
    resampling_unit: str,
    save_draws: bool = False,
) -> tuple[dict[str, BootstrapResult], list[ContrastResult]]:
    if cell.status == "pending":
        return {}, []
    if resampling_unit == "episode" and not cell.has_episode_data:
        raise ValueError(
            f"{cell.model}/{cell.task}/{cell.horizon}: episode mode unavailable"
        )

    rng = np.random.default_rng(seed)
    n_eval = len(cell.eval_seeds)
    n_train = len(cell.train_seeds)
    shared_eval_idx = rng.integers(0, n_eval, size=(n_bootstrap, n_eval))
    shared_train_idx = rng.integers(0, n_train, size=(n_bootstrap, n_train))

    raw_draws: dict[str, np.ndarray] = {}
    results: dict[str, BootstrapResult] = {}
    for mid, method in cell.methods.items():
        draws = _method_draws(
            method,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            resampling_unit=resampling_unit,
            shared_eval_idx=shared_eval_idx,
            shared_train_idx=shared_train_idx,
            rng=rng,
        )
        raw_draws[mid] = draws
        point = point_estimate(method.blocks, official=method.official)
        n_train_used = 0 if method.official else method.blocks.shape[0]
        results[mid] = BootstrapResult(
            method_id=mid,
            point_estimate=point,
            bootstrap_mean=float(draws.mean()),
            bootstrap_std=float(draws.std(ddof=0)),
            ci_low=float(np.quantile(draws, 0.025)),
            ci_high=float(np.quantile(draws, 0.975)),
            n_train_seeds=n_train_used,
            n_partition_seeds=method.n_partition_seeds,
            n_eval_blocks=method.blocks.shape[-1],
            draws=draws if save_draws else None,
        )

    contrasts: list[ContrastResult] = []
    if "autolap" in raw_draws:
        auto_draws = raw_draws["autolap"]
        for bid, base_draws in raw_draws.items():
            if bid == "autolap":
                continue
            delta = auto_draws - base_draws
            contrasts.append(
                ContrastResult(
                    baseline_method=bid,
                    point_difference=results["autolap"].point_estimate - results[bid].point_estimate,
                    ci_low=float(np.quantile(delta, 0.025)),
                    ci_high=float(np.quantile(delta, 0.975)),
                    pr_gt_zero=float((delta > 0).mean()),
                    n_common_blocks=results[bid].n_eval_blocks,
                    draws=delta if save_draws else None,
                )
            )

    return results, contrasts
