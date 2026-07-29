#!/usr/bin/env python3
"""Four-way open-loop latent rollout MSE on held-out test transitions.

Compares Official LeWM, Global-FT, rooms3 region-switch, and latent
cluster-switch under identical recorded actions and frozen encoder latents.

Latent routing modes are deliberately explicit:

``mpc``
    Route once from the observed latent at the rollout start and keep that
    expert fixed for the complete imagined rollout.
``step``
    Route the first prediction from the observed transition start, then route
    every later prediction from the preceding model-predicted latent.  No
    future ground-truth latent is consulted.
``oracle_gt_step``
    Diagnostic upper-bound protocol that routes from the recorded future GT
    latent at every step.  The word ``oracle`` is retained in every result key
    and metadata field so it cannot be mistaken for deployable routing.

Both centroid artifacts and multi-prototype spectral artifacts use the same
contract: nearest routing vector -> ``prototype_cluster_ids`` -> predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS, tworoom_geometry_thresholds  # noqa: E402
from latent_cluster_common import (  # noqa: E402
    resolve_cluster_source,
)
from predictor_rule_drift import (  # noqa: E402
    json_ready,
    load_transition_cache,
    predict_next,
    read_sequence_dataset,
)
from trajectory_deviation import load_predictor, resolve_device  # noqa: E402
from tworoom_success_rate_eval import (  # noqa: E402
    geometry_rooms3_key,
    predictor_manifest_provenance,
    validate_predictor_manifest_artifact,
)
from latent_cluster_train_predictors import manifest_file_lock  # noqa: E402

DEFAULT_CACHE = THIS_DIR / "cache" / "tworoom_trajectory_test_full_transitions.npz"
DEFAULT_OFFICIAL = Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt")
DEFAULT_GLOBAL_FT50 = (
    THIS_DIR / "results" / "tworoom_geometry_train_global_ft_50ep" / "P_train_global_ft_object.ckpt"
)
DEFAULT_REGION_DIR = THIS_DIR / "results" / "tworoom_geometry_train_region_predictors"
DEFAULT_KMEANS_DIR = THIS_DIR / "results" / "latent_kmeanspp_multirestart_k3"
ROOMS3_REGIONS = ("left_room", "doorway_corridor", "right_room")
REPORT_STEPS = (1, 3, 5, 10)
LATENT_ROUTING_MODES = ("mpc", "step", "oracle_gt_step")
ROOMS3_ROUTING_MODES = ("mpc", "oracle_gt_step")


class LatentPrototypeRouter:
    """Torch router shared by offline rollout and online deployment semantics.

    ``routing_vectors`` may contain one vector per cluster (legacy K-means) or
    any number of prototypes per cluster.  The nearest vector index is never
    treated as a predictor index directly; it is always mapped through
    ``prototype_cluster_ids``.
    """

    def __init__(self, artifact: dict, device: torch.device):
        vector_source = artifact.get("routing_vectors")
        if vector_source is None:
            vector_source = artifact.get("centroids")
        if vector_source is None:
            raise ValueError("artifact lacks routing_vectors/centroids")
        routing_vectors = np.asarray(vector_source, dtype=np.float32)
        if routing_vectors.ndim != 2 or not len(routing_vectors):
            raise ValueError("routing vectors must have shape (P,D) with P > 0")

        owners = np.asarray(
            artifact.get(
                "prototype_cluster_ids",
                np.arange(len(routing_vectors), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        if owners.shape != (len(routing_vectors),):
            raise ValueError(
                "prototype_cluster_ids must contain one owner per routing vector"
            )

        self.num_clusters = int(
            artifact.get("meta", {}).get(
                "num_clusters", int(owners.max()) + 1 if len(owners) else 0
            )
        )
        if self.num_clusters < 1:
            raise ValueError("artifact must define at least one cluster")
        if owners.min() < 0 or owners.max() >= self.num_clusters:
            raise ValueError("prototype owner is outside the artifact cluster range")
        if set(owners.tolist()) != set(range(self.num_clusters)):
            raise ValueError("every cluster must own at least one routing vector")
        if not np.isfinite(routing_vectors).all():
            raise ValueError("routing vectors contain NaN/Inf")

        self.device = device
        self.spherical = bool(artifact.get("meta", {}).get("spherical", True))
        vectors = torch.as_tensor(routing_vectors, device=device, dtype=torch.float32)
        self.routing_vectors = (
            torch.nn.functional.normalize(vectors, dim=1)
            if self.spherical
            else vectors
        )
        self.prototype_cluster_ids = torch.as_tensor(
            owners, device=device, dtype=torch.long
        )

        zscore = artifact.get("zscore")
        if zscore is None:
            self.zscore_mu = None
            self.zscore_sigma = None
            self.zscore_eps = 0.0
        else:
            mu = np.asarray(zscore["mu"], dtype=np.float32)
            sigma = np.asarray(zscore["sigma"], dtype=np.float32)
            if mu.shape != (routing_vectors.shape[1],) or sigma.shape != mu.shape:
                raise ValueError("Z-score parameters do not match routing dimension")
            self.zscore_mu = torch.as_tensor(mu, device=device)
            self.zscore_sigma = torch.as_tensor(sigma, device=device)
            self.zscore_eps = float(zscore["eps"])

    def assign(self, latent: torch.Tensor) -> torch.Tensor:
        """Return owner-cluster ids for a ``(B,D)`` latent tensor."""
        if latent.ndim != 2 or latent.shape[1] != self.routing_vectors.shape[1]:
            raise ValueError(
                "latent must have shape (B,D) matching the routing vectors"
            )
        routed = latent.to(self.device, dtype=torch.float32)
        if self.zscore_mu is not None:
            routed = (routed - self.zscore_mu) / (
                self.zscore_sigma + self.zscore_eps
            )
            routed = torch.nn.functional.normalize(routed, dim=1)
        if self.spherical:
            routed = torch.nn.functional.normalize(routed, dim=1)
            prototype_ids = (routed @ self.routing_vectors.T).argmax(dim=1)
        else:
            prototype_ids = torch.cdist(
                routed, self.routing_vectors, p=2
            ).argmin(dim=1)
        return self.prototype_cluster_ids[prototype_ids]


def load_proprio_sequences(
    h5_path: Path,
    spec,
    starts: np.ndarray,
    seq_len: int,
    frameskip: int,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        key = "proprio" if "proprio" in h5 else spec.state_keys[0]
        proprio = read_sequence_dataset(h5[key], starts, seq_len, frameskip)
    return proprio.astype(np.float64)


def region_masks_from_proprio(proprio: np.ndarray, route_index: int) -> dict[str, np.ndarray]:
    thresholds = tworoom_geometry_thresholds()
    n = proprio.shape[0]
    masks = {name: np.zeros(n, dtype=bool) for name in ROOMS3_REGIONS}
    for i in range(n):
        x = float(proprio[i, route_index, 0])
        y = float(proprio[i, route_index, 1])
        key = geometry_rooms3_key(x, y, thresholds)
        masks[key][i] = True
    return masks


@torch.no_grad()
def rollout_mse_single(
    model: torch.nn.Module,
    latents: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    max_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Return per-step per-sample MSE vs GT, shape (max_steps, n_samples)."""
    n_samples = latents.shape[0]
    out = np.zeros((max_steps, n_samples), dtype=np.float64)
    for offset in range(0, n_samples, batch_size):
        y_true = latents[offset : offset + batch_size]
        a_true = actions[offset : offset + batch_size]
        bsz = y_true.shape[0]
        emb = y_true[:, :history].clone()
        for step in range(max_steps):
            ctx_emb = emb[:, -history:]
            ctx_action = a_true[:, step : step + history]
            pred = predict_next(model, ctx_emb, ctx_action)
            target = y_true[:, history + step]
            mse = (pred - target).pow(2).mean(dim=1).detach().cpu().numpy()
            out[step, offset : offset + bsz] = mse
            emb = torch.cat([emb, pred[:, None]], dim=1)
    return out


@torch.no_grad()
def rollout_mse_switch(
    models: dict[str, torch.nn.Module],
    pick_key: Callable[[int, int], str],
    latents: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    max_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Dynamic predictor switch rollout. pick_key(global_sample_idx, step) -> model key."""
    n_samples = latents.shape[0]
    out = np.zeros((max_steps, n_samples), dtype=np.float64)
    for offset in range(0, n_samples, batch_size):
        y_true = latents[offset : offset + batch_size]
        a_true = actions[offset : offset + batch_size]
        bsz = y_true.shape[0]
        emb = y_true[:, :history].clone()
        for step in range(max_steps):
            keys = [pick_key(offset + local_i, step) for local_i in range(bsz)]
            pred_batch = torch.empty((bsz, emb.shape[-1]), device=emb.device, dtype=emb.dtype)
            for key in set(keys):
                local_idx = [i for i, k in enumerate(keys) if k == key]
                idx = torch.tensor(local_idx, device=emb.device, dtype=torch.long)
                model = models[key]
                ctx_emb = emb.index_select(0, idx)[:, -history:]
                ctx_action = a_true.index_select(0, idx)[:, step : step + history]
                pred = predict_next(model, ctx_emb, ctx_action)
                pred_batch.index_copy_(0, idx, pred)
            target = y_true[:, history + step]
            mse = (pred_batch - target).pow(2).mean(dim=1).detach().cpu().numpy()
            out[step, offset : offset + bsz] = mse
            emb = torch.cat([emb, pred_batch[:, None]], dim=1)
    return out


@torch.no_grad()
def rollout_mse_latent_switch(
    models: dict[str, torch.nn.Module],
    router: LatentPrototypeRouter,
    routing_mode: str,
    latents: torch.Tensor,
    actions: torch.Tensor,
    history: int,
    max_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Cluster-switch rollout with deployable and oracle modes kept distinct.

    ``mpc`` and ``step`` only route from information available at deployment.
    ``oracle_gt_step`` is deliberately explicit because it reads future recorded
    latents and is valid only as a diagnostic upper bound.
    """
    if routing_mode not in LATENT_ROUTING_MODES:
        raise ValueError(
            f"Unknown latent routing mode {routing_mode!r}; "
            f"expected one of {LATENT_ROUTING_MODES}"
        )
    expected_keys = {f"cluster{k}" for k in range(router.num_clusters)}
    missing_keys = sorted(expected_keys.difference(models))
    if missing_keys:
        raise KeyError(f"Missing predictor checkpoints for {missing_keys}")

    n_samples = latents.shape[0]
    out = np.zeros((max_steps, n_samples), dtype=np.float64)
    for offset in range(0, n_samples, batch_size):
        y_true = latents[offset : offset + batch_size]
        a_true = actions[offset : offset + batch_size]
        bsz = y_true.shape[0]
        emb = y_true[:, :history].clone()
        # Predictor experts are trained from transition-start labels (offset 0),
        # so the fixed rollout expert is selected from the first state in the
        # cached window, not from history-1.
        fixed_cluster_ids = router.assign(emb[:, 0]) if routing_mode == "mpc" else None

        for step in range(max_steps):
            if fixed_cluster_ids is not None:
                cluster_ids = fixed_cluster_ids
            elif routing_mode == "step":
                # The first route uses the user-defined rollout start (index 0).
                # Every later route uses the preceding model prediction, which
                # is the last element appended to emb.  Future GT never enters.
                route_latent = emb[:, 0] if step == 0 else emb[:, -1]
                cluster_ids = router.assign(route_latent)
            else:
                # Explicit diagnostic counterpart: first use index 0, then the
                # recorded GT state corresponding to the preceding prediction.
                route_t = 0 if step == 0 else history + step - 1
                cluster_ids = router.assign(y_true[:, route_t])

            if cluster_ids.shape != (bsz,):
                raise ValueError("router.assign must return one cluster id per sample")
            if cluster_ids.numel() and (
                int(cluster_ids.min()) < 0
                or int(cluster_ids.max()) >= router.num_clusters
            ):
                raise ValueError("router returned a cluster id outside the artifact range")

            pred_batch = torch.empty(
                (bsz, emb.shape[-1]), device=emb.device, dtype=emb.dtype
            )
            for cluster_id in torch.unique(cluster_ids).detach().cpu().tolist():
                key = f"cluster{int(cluster_id)}"
                idx = torch.nonzero(cluster_ids == cluster_id, as_tuple=False).flatten()
                ctx_emb = emb.index_select(0, idx)[:, -history:]
                ctx_action = a_true.index_select(0, idx)[:, step : step + history]
                pred = predict_next(models[key], ctx_emb, ctx_action)
                pred_batch.index_copy_(0, idx, pred)

            target = y_true[:, history + step]
            mse = (pred_batch - target).pow(2).mean(dim=1).detach().cpu().numpy()
            out[step, offset : offset + bsz] = mse
            emb = torch.cat([emb, pred_batch[:, None]], dim=1)
    return out


def summarize_overall(per_sample: np.ndarray, steps: tuple[int, ...]) -> list[dict]:
    rows = []
    for step in steps:
        idx = step - 1
        vals = per_sample[idx]
        rows.append(
            {
                "scope": "overall",
                "region": "all",
                "step": step,
                "mse_mean": float(vals.mean()),
                "mse_std": float(vals.std()),
                "n_samples": int(len(vals)),
            }
        )
    return rows


def summarize_by_region(
    per_sample: np.ndarray,
    region_masks: dict[str, np.ndarray],
    steps: tuple[int, ...],
) -> list[dict]:
    rows = []
    for step in steps:
        idx = step - 1
        vals = per_sample[idx]
        for region, mask in region_masks.items():
            if not mask.any():
                continue
            sub = vals[mask]
            rows.append(
                {
                    "scope": "region",
                    "region": region,
                    "step": step,
                    "mse_mean": float(sub.mean()),
                    "mse_std": float(sub.std()),
                    "n_samples": int(len(sub)),
                }
            )
    return rows


def delta_vs_reference(
    per_sample: np.ndarray,
    reference: np.ndarray,
    steps: tuple[int, ...],
) -> list[dict]:
    rows = []
    for step in steps:
        idx = step - 1
        delta = per_sample[idx] - reference[idx]
        rows.append(
            {
                "step": step,
                "delta_mean": float(delta.mean()),
                "delta_std": float(delta.std()),
                "frac_improved": float((delta < 0).mean()),
                "frac_worse": float((delta > 0).mean()),
            }
        )
    return rows


def load_region_switch_models(region_dir: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    models = {}
    for region in ROOMS3_REGIONS:
        path = region_dir / f"P_train_{region}_object.ckpt"
        if not path.exists():
            raise FileNotFoundError(path)
        models[region] = load_predictor(path, device)
    return models


def load_cluster_switch_models(
    base_ckpt: Path,
    predictor_dir: Path,
    num_clusters: int,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models = {}
    for k in range(num_clusters):
        path = predictor_dir / f"P_train_cluster{k}_object.ckpt"
        if not path.exists():
            raise FileNotFoundError(path)
        models[f"cluster{k}"] = load_predictor(path, device)
    # Keep official encoder bundle only for action_encoder consistency check; rollout uses cached latents.
    _ = load_predictor(base_ckpt, device)
    return models


def precompute_rooms3_keys(
    proprio: np.ndarray,
    history: int,
    max_steps: int,
    routing_mode: str,
) -> np.ndarray:
    if routing_mode not in ROOMS3_ROUTING_MODES:
        raise ValueError(
            f"Unknown rooms3 routing mode {routing_mode!r}; "
            f"expected one of {ROOMS3_ROUTING_MODES}"
        )
    thresholds = tworoom_geometry_thresholds()
    n = proprio.shape[0]
    keys = np.empty((n, max_steps), dtype=object)
    for step in range(max_steps):
        route_t = (
            0
            if routing_mode == "mpc" or step == 0
            else history + step - 1
        )
        for i in range(n):
            x = float(proprio[i, route_t, 0])
            y = float(proprio[i, route_t, 1])
            keys[i, step] = geometry_rooms3_key(x, y, thresholds)
    return keys


def make_precomputed_picker(keys: np.ndarray) -> Callable[[int, int], str]:
    def pick(global_i: int, step: int) -> str:
        val = keys[global_i, step]
        return val if isinstance(val, str) else f"cluster{int(val)}"

    return pick


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-test-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--official-ckpt", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--global-ft-ckpt", type=Path, default=DEFAULT_GLOBAL_FT50)
    parser.add_argument("--region-dir", type=Path, default=DEFAULT_REGION_DIR)
    parser.add_argument(
        "--cluster-artifact-dir",
        type=Path,
        default=None,
        help="Unified spectral/K-means cluster artifact directory.",
    )
    parser.add_argument(
        "--cluster-predictor-dir",
        type=Path,
        default=None,
        help="Directory containing P_train_cluster{ID}_object.ckpt.",
    )
    parser.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS_DIR)
    parser.add_argument("--cluster-outer-seeds", default="0,1,2")
    parser.add_argument(
        "--latent-routing",
        choices=LATENT_ROUTING_MODES,
        default="mpc",
        help=(
            "mpc=fixed from rollout start; step=reroute from predicted latent; "
            "oracle_gt_step=diagnostic future-GT routing"
        ),
    )
    parser.add_argument(
        "--rooms3-routing",
        choices=ROOMS3_ROUTING_MODES,
        default="mpc",
        help="mpc is deployable; oracle_gt_step reads recorded future proprio.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tworoom")
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to a source- and routing-specific result directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_steps = tuple(step for step in REPORT_STEPS if step <= args.max_steps)
    if not report_steps:
        raise ValueError("--max-steps must be at least 1")
    if args.out_dir is None:
        if args.cluster_artifact_dir is not None:
            source_tag = args.cluster_artifact_dir.name
        else:
            source_tag = f"kmeans_outer{args.cluster_outer_seeds.replace(',', '-') }"
        args.out_dir = (
            THIS_DIR
            / "results"
            / (
                f"tworoom_trajectory_switch_rollout_{source_tag}_"
                f"latent{args.latent_routing}_rooms3{args.rooms3_routing}"
            )
        )
    device = resolve_device(args.device)
    spec = DATASETS[args.dataset]
    h5_path = args.data_root / spec.default_file

    latents_np, actions_np, starts, cache_meta = load_transition_cache(args.load_test_cache)
    seq_len = int(cache_meta.get("seq_len", args.history_size + args.max_steps))
    if seq_len < args.history_size + args.max_steps:
        raise ValueError(f"Cache seq_len={seq_len} too short for rollout")
    latents_np = latents_np[:, :seq_len]
    actions_np = actions_np[:, :seq_len]

    frameskip = int(cache_meta.get("frameskip", 5))
    proprio = load_proprio_sequences(h5_path, spec, starts, seq_len, frameskip)
    start_region_masks = region_masks_from_proprio(proprio, 0)

    latents = torch.as_tensor(latents_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    n_samples = latents.shape[0]
    print(
        f"[data] {n_samples} test transitions, history={args.history_size}, "
        f"max_steps={args.max_steps}, frameskip={frameskip}",
        flush=True,
    )

    results: dict[str, np.ndarray] = {}

    print("[rollout] official_lewm", flush=True)
    official = load_predictor(args.official_ckpt, device)
    results["official_lewm"] = rollout_mse_single(
        official, latents, actions, args.history_size, args.max_steps, args.batch_size
    )

    print("[rollout] global_ft_50ep", flush=True)
    global_ft = load_predictor(args.global_ft_ckpt, device)
    results["global_ft_50ep"] = rollout_mse_single(
        global_ft, latents, actions, args.history_size, args.max_steps, args.batch_size
    )

    rooms3_result_key = f"rooms3_switch_50ep_{args.rooms3_routing}"
    print(f"[rollout] {rooms3_result_key}", flush=True)
    region_models = load_region_switch_models(args.region_dir, device)
    rooms3_keys = precompute_rooms3_keys(
        proprio,
        args.history_size,
        args.max_steps,
        args.rooms3_routing,
    )
    rooms3_pick = make_precomputed_picker(rooms3_keys)
    results[rooms3_result_key] = rollout_mse_switch(
        region_models,
        rooms3_pick,
        latents,
        actions,
        args.history_size,
        args.max_steps,
        args.batch_size,
    )

    if (args.cluster_artifact_dir is None) != (args.cluster_predictor_dir is None):
        raise ValueError(
            "--cluster-artifact-dir and --cluster-predictor-dir must be passed together"
        )

    outer_seeds: list[int] = []
    cluster_jobs: list[
        tuple[str, dict, Path, Path | None, Path | None]
    ] = []
    if args.cluster_artifact_dir is not None:
        artifact = resolve_cluster_source(
            cluster_artifact_dir=args.cluster_artifact_dir,
        )
        cluster_jobs.append(
            (
                "explicit_artifact",
                artifact,
                args.cluster_predictor_dir,
                args.cluster_artifact_dir,
                None,
            )
        )
    else:
        outer_seeds = [
            int(x) for x in args.cluster_outer_seeds.split(",") if x.strip()
        ]
        if not outer_seeds:
            raise ValueError("At least one --cluster-outer-seeds value is required")
        zscore_path = args.kmeans_dir / "zscore_params.npz"
        for outer in outer_seeds:
            label_npz = args.kmeans_dir / "labels" / f"kmeanspp_R50_outer{outer}.npz"
            pred_dir = (
                THIS_DIR
                / "results"
                / f"tworoom_latent_kmeanspp_kmeanspp_R50_outer{outer}"
            )
            artifact = resolve_cluster_source(
                kmeanspp_label_npz=label_npz,
                zscore_params_npz=zscore_path,
                require_kmeanspp_zscore=True,
            )
            cluster_jobs.append((f"outer={outer}", artifact, pred_dir, None, label_npz))

    cluster_runs: list[np.ndarray] = []
    cluster_source_records: list[dict] = []
    for source_label, artifact, pred_dir, artifact_dir, kmeans_label_npz in cluster_jobs:
        router = LatentPrototypeRouter(artifact, device)
        manifest_path = pred_dir / "manifest.json"
        # Authenticate and load the same immutable checkpoint bytes while
        # recovery/overwrite writers are excluded by their shared lock.
        with manifest_file_lock(manifest_path):
            predictor_manifest = validate_predictor_manifest_artifact(
                pred_dir,
                artifact=artifact,
                cluster_artifact_dir=artifact_dir,
                kmeanspp_label_npz=kmeans_label_npz,
                base_checkpoint=args.official_ckpt,
            )
            manifest_provenance = predictor_manifest_provenance(
                predictor_manifest
            )
            cluster_models = load_cluster_switch_models(
                args.official_ckpt, pred_dir, router.num_clusters, device
            )
        print(
            f"[rollout] latent_cluster_switch_{args.latent_routing} "
            f"source={source_label} K={router.num_clusters} "
            f"P={len(router.routing_vectors)}",
            flush=True,
        )
        cluster_runs.append(
            rollout_mse_latent_switch(
                cluster_models,
                router,
                args.latent_routing,
                latents,
                actions,
                args.history_size,
                args.max_steps,
                args.batch_size,
            )
        )
        cluster_source_records.append(
            {
                "source_label": source_label,
                "artifact": str(artifact.get("artifact_dir", "")),
                "predictor_dir": str(pred_dir),
                "predictor_manifest": (
                    str(predictor_manifest) if predictor_manifest else None
                ),
                "predictor_manifest_provenance": manifest_provenance,
                "num_clusters": router.num_clusters,
                "num_routing_prototypes": int(len(router.routing_vectors)),
                "prototype_cluster_ids": (
                    router.prototype_cluster_ids.detach().cpu().tolist()
                ),
                "artifact_meta": json_ready(artifact.get("meta", {})),
            }
        )

    latent_result_key = f"latent_cluster_switch_{args.latent_routing}"
    cluster_stack = np.stack(cluster_runs, axis=0)
    results[latent_result_key] = np.mean(cluster_stack, axis=0)
    if len(cluster_runs) > 1:
        results[f"{latent_result_key}_source_std"] = np.std(
            cluster_stack, axis=0
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []
    delta_rows: list[dict] = []
    ref = results["global_ft_50ep"]

    for config, per_sample in results.items():
        if config.endswith("_std"):
            continue
        for row in summarize_overall(per_sample, report_steps):
            row = {"config": config, **row}
            long_rows.append(row)
        for row in summarize_by_region(per_sample, start_region_masks, report_steps):
            row = {"config": config, **row}
            long_rows.append(row)
        if config != "global_ft_50ep":
            for row in delta_vs_reference(per_sample, ref, report_steps):
                delta_rows.append({"config": config, **row})

    long_csv = args.out_dir / "rollout_mse_4way.csv"
    with long_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    delta_csv = args.out_dir / "rollout_mse_delta_vs_global_ft_50ep.csv"
    with delta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)

    per_traj_path = args.out_dir / "per_trajectory_mse.npz"
    np.savez(
        per_traj_path,
        **{k: v for k, v in results.items() if not k.endswith("_std")},
        starts=starts,
        report_steps=np.array(report_steps, dtype=np.int64),
    )

    summary_table = []
    for config in (
        "official_lewm",
        "global_ft_50ep",
        rooms3_result_key,
        latent_result_key,
    ):
        row = {"config": config}
        for step in report_steps:
            row[f"step{step}_mse"] = float(results[config][step - 1].mean())
        summary_table.append(row)

    payload = {
        "protocol": {
            "goal": "4-way offline latent rollout MSE on identical held-out transitions",
            "history_size": args.history_size,
            "max_steps": args.max_steps,
            "report_steps": list(report_steps),
            "eval_budget_note": (
                "offline open-loop latent rollout under recorded actions; "
                "not control eval_budget"
            ),
            "routing": {
                "rooms3": {
                    "mode": args.rooms3_routing,
                    "deployable": args.rooms3_routing == "mpc",
                    "description": (
                        "route once from observed start proprio and keep the expert fixed"
                        if args.rooms3_routing == "mpc"
                        else "ORACLE diagnostic: reroute from recorded future GT proprio"
                    ),
                },
                "cluster": {
                    "mode": args.latent_routing,
                    "deployable": args.latent_routing != "oracle_gt_step",
                    "description": {
                        "mpc": (
                            "route once from observed start latent and keep the expert fixed"
                        ),
                        "step": (
                            "route first from observed transition start, then only from "
                            "the preceding model-predicted latent"
                        ),
                        "oracle_gt_step": (
                            "ORACLE diagnostic: reroute from recorded future GT latents"
                        ),
                    }[args.latent_routing],
                    "prototype_contract": (
                        "nearest routing-vector index is mapped through "
                        "prototype_cluster_ids before predictor lookup"
                    ),
                },
            },
            "region_stratification": "transition-start position region at index 0",
            "num_test_samples": n_samples,
            "cache": str(args.load_test_cache),
            "cache_metadata": json_ready(cache_meta),
        },
        "checkpoints": {
            "official_lewm": str(args.official_ckpt),
            "global_ft_50ep": str(args.global_ft_ckpt),
            "rooms3_regions": {
                r: str(args.region_dir / f"P_train_{r}_object.ckpt") for r in ROOMS3_REGIONS
            },
            "latent_cluster_outer_seeds": outer_seeds,
            "latent_cluster_sources": cluster_source_records,
        },
        "summary_overall": summary_table,
        "metrics_long": long_rows,
        "delta_vs_global_ft_50ep": delta_rows,
    }
    json_path = args.out_dir / "rollout_mse_4way.json"
    with json_path.open("w") as f:
        json.dump(json_ready(payload), f, indent=2)

    print(f"Wrote {long_csv}", flush=True)
    print(f"Wrote {delta_csv}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {per_traj_path}", flush=True)
    print("config  step1   step3   step5   step10", flush=True)
    for row in summary_table:
        print(
            f"{row['config']:22s}  "
            f"{row['step1_mse']:.4f}  {row['step3_mse']:.4f}  "
            f"{row['step5_mse']:.4f}  {row['step10_mse']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
