#!/usr/bin/env python3
"""LeWM task success-rate evaluation (dataset-seeded MPC, CEM).

Matches the selected official ``config/eval/<task>.yaml`` while loading the
serialized baseline, continued, global-FT, or LAP checkpoints used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import tworoom_geometry_thresholds  # noqa: E402
from jepa import JEPA, detach_clone  # noqa: E402
from latent_cluster_common import resolve_cluster_source  # noqa: E402
from latent_cluster_train_predictors import manifest_file_lock  # noqa: E402
from backends.lewm.routing import (  # noqa: E402
    route_voronoi_torch,
    transform_latent_torch,
)
from lap.partition import PartitionArtifact  # noqa: E402

GEOMETRY_TRAIN_PRED_DIR = THIS_DIR / "results" / "tworoom_geometry_train_region_predictors"
DEFAULT_BASELINE_STARTS = (
    THIS_DIR / "results" / "tworoom_success_rate_baseline_seed42" / "results.json"
)
DEFAULT_GLOBAL_CKPT = Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt")

PRIORITY5_ORDER = (
    "doorway_corridor",
    "near_wall",
    "common",
    "right_room",
    "left_room",
)


def _proprio_xy(proprio: torch.Tensor) -> tuple[float, float]:
    pos = proprio
    while pos.ndim > 2:
        pos = pos[:, 0]
    return pos[0, 0].item(), pos[0, 1].item()


def _standardize_geometry_thresholds(
    thresholds: dict[str, float], proprio_processor
) -> dict[str, float]:
    """Express physical geometry thresholds in standardized proprio coordinates."""
    mean = proprio_processor.mean_
    scale = proprio_processor.scale_
    standardized = dict(thresholds)
    for name in ("wall_lo", "wall_hi"):
        standardized[name] = (thresholds[name] - mean[0]) / scale[0]
    for name in thresholds:
        if name.startswith("x_"):
            standardized[name] = (thresholds[name] - mean[0]) / scale[0]
        elif name.startswith("y_"):
            standardized[name] = (thresholds[name] - mean[1]) / scale[1]
    return standardized


def geometry_rooms3_key(x: float, y: float, thresholds: dict[str, float]) -> str:
    if x < thresholds["wall_lo"]:
        return "left_room"
    if x <= thresholds["wall_hi"]:
        return "doorway_corridor"
    return "right_room"


def geometry_priority5_key(x: float, y: float, thresholds: dict[str, float]) -> str:
    """README exp3 priority on overlapping geometry regions."""
    doorway = thresholds["wall_lo"] <= x <= thresholds["wall_hi"]
    near_wall = (
        x <= thresholds["x_lo_wall"]
        or x >= thresholds["x_hi_wall"]
        or y <= thresholds["y_lo_wall"]
        or y >= thresholds["y_hi_wall"]
    )
    left_interior = (
        thresholds["x_lo_common"] <= x <= thresholds["x_hi_common_left"]
    )
    right_interior = (
        thresholds["x_lo_common_right"] <= x <= thresholds["x_hi_common"]
    )
    y_interior = thresholds["y_lo_common"] <= y <= thresholds["y_hi_common"]
    common = (left_interior or right_interior) and y_interior

    if doorway:
        return "doorway_corridor"
    if near_wall:
        return "near_wall"
    if common:
        return "common"
    if x > thresholds["wall_hi"]:
        return "right_room"
    return "left_room"


class RegionSwitchJEPA(JEPA):
    """Swap predictor at each MPC replan based on agent proprio."""

    def __init__(
        self,
        base: JEPA,
        region_models: dict[str, JEPA],
        proprio_processor,
        thresholds: dict[str, float],
        region_selector,
    ):
        super().__init__(
            base.encoder,
            base.predictor,
            base.action_encoder,
            base.projector,
            base.pred_proj,
        )
        self.region_predictor = torch.nn.ModuleDict(
            {name: m.predictor for name, m in region_models.items()}
        )
        self.region_pred_proj = torch.nn.ModuleDict(
            {name: m.pred_proj for name, m in region_models.items()}
        )
        self.thresholds = _standardize_geometry_thresholds(
            thresholds, proprio_processor
        )
        self.region_selector = region_selector
        self._cached_proprio: torch.Tensor | None = None

    def _select_predictor(self, proprio: torch.Tensor) -> str:
        # CEM reuses the same expanded proprio tensor for all optimization
        # iterations in one batch. Avoid repeating GPU→CPU synchronization and
        # region lookup on every get_cost call.
        if proprio is self._cached_proprio:
            return self._active_region

        x, y = _proprio_xy(proprio)
        key = self.region_selector(x, y, self.thresholds)
        self.predictor = self.region_predictor[key]
        self.pred_proj = self.region_pred_proj[key]
        self._cached_proprio = proprio
        self._active_region = key
        return key

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        self._select_predictor(info_dict["proprio"])
        return super().get_cost(info_dict, action_candidates)


class LatentClusterSwitchJEPA(JEPA):
    """Route latent-cluster experts either once per MPC replan or per imagined step."""

    def __init__(
        self,
        base: JEPA,
        cluster_models: dict[str, JEPA],
        centroids: np.ndarray,
        *,
        prototype_cluster_ids: np.ndarray | None = None,
        spherical: bool = True,
        zscore: dict | None = None,
        routing_mode: str = "mpc",
    ):
        super().__init__(
            base.encoder,
            base.predictor,
            base.action_encoder,
            base.projector,
            base.pred_proj,
        )
        self.cluster_predictor = torch.nn.ModuleDict(
            {name: m.predictor for name, m in cluster_models.items()}
        )
        self.cluster_pred_proj = torch.nn.ModuleDict(
            {name: m.pred_proj for name, m in cluster_models.items()}
        )
        if routing_mode not in ("mpc", "step"):
            raise ValueError(f"Unknown latent routing mode: {routing_mode}")
        self.routing_mode = routing_mode
        self.cluster_names = tuple(
            sorted(cluster_models, key=lambda name: int(name.removeprefix("cluster")))
        )
        expected_cluster_names = tuple(
            f"cluster{k}" for k in range(len(self.cluster_names))
        )
        if self.cluster_names != expected_cluster_names:
            raise ValueError(
                "Cluster predictors must use contiguous names cluster0..clusterK-1; "
                f"got {self.cluster_names}"
            )
        self.register_buffer(
            "cluster_centroids",
            torch.as_tensor(np.asarray(centroids, dtype=np.float32)),
        )
        if prototype_cluster_ids is None:
            prototype_cluster_ids = np.arange(len(centroids), dtype=np.int64)
        prototype_cluster_ids = np.asarray(prototype_cluster_ids, dtype=np.int64)
        if prototype_cluster_ids.shape != (len(centroids),):
            raise ValueError(
                "prototype_cluster_ids must contain one owner per routing vector"
            )
        if prototype_cluster_ids.size and (
            prototype_cluster_ids.min() < 0
            or prototype_cluster_ids.max() >= len(self.cluster_names)
        ):
            raise ValueError("prototype owner refers to a missing cluster predictor")
        self.register_buffer(
            "prototype_cluster_ids",
            torch.as_tensor(prototype_cluster_ids, dtype=torch.long),
        )
        self.spherical = spherical
        if zscore is None:
            self.register_buffer("zscore_mu", torch.empty(0, dtype=torch.float32))
            self.register_buffer("zscore_sigma", torch.empty(0, dtype=torch.float32))
            self.zscore_eps = 0.0
        else:
            self.register_buffer(
                "zscore_mu",
                torch.as_tensor(np.asarray(zscore["mu"], dtype=np.float32)),
            )
            self.register_buffer(
                "zscore_sigma",
                torch.as_tensor(np.asarray(zscore["sigma"], dtype=np.float32)),
            )
            self.zscore_eps = float(zscore["eps"])
        self._cached_pixels: torch.Tensor | None = None
        self._active_cluster: str = "cluster0"
        self._mpc_route_cache_key: tuple | None = None
        self._mpc_cached_pixels_ref: torch.Tensor | None = None
        self._mpc_cached_env_cluster_ids: torch.Tensor | None = None
        self.mpc_route_cache_hits = 0
        self.mpc_route_cache_misses = 0
        self.classify_count = 0
        self.classify_assignment_count = 0
        self.classify_time_sec = 0.0
        self.classify_timed_count = 0
        self.classify_timed_assignment_count = 0
        # CUDA events are intentionally sampled: retaining one event pair for
        # every CEM candidate-step route can otherwise become a memory leak in
        # long evaluations.  The sample still measures the deployed GPU path.
        self.classify_timing_sample_limit = 2048
        self._classify_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.route_call_count = 0
        self.route_assignment_count = 0
        self.register_buffer(
            "route_histogram",
            torch.zeros(len(self.cluster_names), dtype=torch.long),
        )
        self.register_buffer("route_switch_count", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "route_transition_count", torch.zeros((), dtype=torch.long)
        )

    def _transform_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Expose the backend transform for diagnostics and contract tests."""

        return transform_latent_torch(
            latent,
            mean=self.zscore_mu if self.zscore_mu.numel() else None,
            scale=self.zscore_sigma if self.zscore_sigma.numel() else None,
            eps=self.zscore_eps,
        )

    def _assign_clusters(self, latent: torch.Tensor) -> torch.Tensor:
        return route_voronoi_torch(
            latent,
            self.cluster_centroids,
            self.prototype_cluster_ids,
            mean=self.zscore_mu if self.zscore_mu.numel() else None,
            scale=self.zscore_sigma if self.zscore_sigma.numel() else None,
            eps=self.zscore_eps,
            spherical=self.spherical,
        )

    def _timed_assign_clusters(self, latent: torch.Tensor) -> torch.Tensor:
        """Assign clusters without synchronizing the online CUDA execution path."""
        should_time = self.classify_timed_count < self.classify_timing_sample_limit
        if latent.is_cuda and should_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            cluster_ids = self._assign_clusters(latent)
            end.record()
            self._classify_cuda_events.append((start, end))
        elif should_time:
            t0 = time.perf_counter()
            cluster_ids = self._assign_clusters(latent)
            self.classify_time_sec += time.perf_counter() - t0
        else:
            cluster_ids = self._assign_clusters(latent)
        self.classify_count += 1
        self.classify_assignment_count += int(latent.shape[0])
        if should_time:
            self.classify_timed_count += 1
            self.classify_timed_assignment_count += int(latent.shape[0])
        return cluster_ids

    def finalize_classify_timing(self) -> None:
        """Materialize deferred CUDA-event timings after evaluation has finished."""
        if not self._classify_cuda_events:
            return
        torch.cuda.synchronize(self.cluster_centroids.device)
        self.classify_time_sec += sum(
            start.elapsed_time(end) / 1000.0
            for start, end in self._classify_cuda_events
        )
        self._classify_cuda_events.clear()

    def _current_latent(self, info_dict: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        pixels = info_dict["pixels"].float().to(device)
        # CEM expands (B, T, C, H, W) -> (B, S, T, C, H, W); cluster routing is sample-invariant.
        if pixels.ndim == 6:
            pixels = pixels[:, 0]
        b, t_steps = pixels.shape[:2]
        flat = pixels.reshape(b * t_steps, *pixels.shape[2:])
        output = self.encoder(flat, interpolate_pos_encoding=True)
        emb = self.projector(output.last_hidden_state[:, 0])
        emb = emb.reshape(b, t_steps, -1)
        return emb[:, -1, :]

    @staticmethod
    def _tensor_identity(tensor: torch.Tensor) -> tuple:
        """Storage identity for reusing one route across repeated CEM calls.

        PyTorch inference tensors intentionally have no version counter.  CEM
        keeps the expanded observation tensor immutable during a solve, and the
        route cache keeps a strong reference to its storage, so the remaining
        storage metadata is sufficient to distinguish successive observations
        without allowing allocator pointer reuse.
        """
        try:
            version = int(tensor._version)
        except RuntimeError:
            if not tensor.is_inference():
                raise
            version = None
        return (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.storage_offset()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            version,
        )

    def _select_predictor(self, info_dict: dict) -> str:
        pixels = info_dict["pixels"]
        cache_key = pixels[:, 0] if pixels.ndim == 6 else pixels
        if cache_key is self._cached_pixels:
            return self._active_cluster

        latent = self._current_latent(info_dict)
        cluster_id = int(self._timed_assign_clusters(latent)[0].item())
        key = f"cluster{cluster_id}"
        self.predictor = self.cluster_predictor[key]
        self.pred_proj = self.cluster_pred_proj[key]
        self._cached_pixels = cache_key
        self._active_cluster = key
        return key

    def _predict_routed(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one next latent for each candidate with its assigned expert."""
        pred_emb = torch.empty(
            (emb.size(0), 1, emb.size(-1)),
            dtype=emb.dtype,
            device=emb.device,
        )
        for cluster_id, key in enumerate(self.cluster_names):
            indices = torch.nonzero(cluster_ids == cluster_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            raw = self.cluster_predictor[key](
                emb.index_select(0, indices),
                act_emb.index_select(0, indices),
            )
            projected = self.cluster_pred_proj[key](raw[:, -1, :]).unsqueeze(1)
            pred_emb.index_copy_(0, indices, projected)
        return pred_emb

    def _record_step_routes(
        self,
        cluster_ids: torch.Tensor,
        previous_cluster_ids: torch.Tensor | None,
    ) -> None:
        self.route_call_count += 1
        self.route_assignment_count += int(cluster_ids.numel())
        self.route_histogram.add_(
            torch.bincount(cluster_ids, minlength=len(self.cluster_names))
        )
        if previous_cluster_ids is not None:
            self.route_switch_count.add_((cluster_ids != previous_cluster_ids).sum())
            self.route_transition_count.add_(cluster_ids.numel())

    def rollout(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        history_steps = info["pixels"].size(2)
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        act_0, act_future = torch.split(
            action_sequence,
            [history_steps, horizon - history_steps],
            dim=2,
        )
        info["action"] = act_0
        num_future_steps = horizon - history_steps

        initial = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        initial = self.encode(initial)
        fixed_cluster_ids: torch.Tensor | None = None
        if self.routing_mode == "mpc":
            # One observed-latent route per environment, fixed for all CEM
            # candidates and all imagined steps in this open-loop rollout.
            # This remains correct when solver batch_size > 1; globally swapping
            # self.predictor would incorrectly share env0's expert across a batch.
            route_key = self._tensor_identity(info["pixels"])
            if (
                route_key == self._mpc_route_cache_key
                and self._mpc_cached_env_cluster_ids is not None
                and len(self._mpc_cached_env_cluster_ids) == batch_size
            ):
                env_cluster_ids = self._mpc_cached_env_cluster_ids
                self.mpc_route_cache_hits += 1
            else:
                # Online TwoRoom currently has a singleton observed time axis;
                # index 0 is the transition-start/MPC-start latent.
                env_cluster_ids = self._timed_assign_clusters(initial["emb"][:, 0, :])
                self._mpc_route_cache_key = route_key
                # Keep the backing storage alive so allocator pointer reuse
                # cannot turn a later observation into a false cache hit.
                self._mpc_cached_pixels_ref = info["pixels"]
                self._mpc_cached_env_cluster_ids = env_cluster_ids.detach().clone()
                self.mpc_route_cache_misses += 1
            fixed_cluster_ids = (
                env_cluster_ids[:, None]
                .expand(batch_size, num_samples)
                .reshape(-1)
            )
        emb = initial["emb"].unsqueeze(1).expand(
            batch_size, num_samples, -1, -1
        )
        info["emb"] = emb
        initial = {k: detach_clone(v) for k, v in initial.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        previous_cluster_ids: torch.Tensor | None = None
        for step in range(num_future_steps + 1):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -history_size:]
            act_trunc = act_emb[:, -history_size:]
            if fixed_cluster_ids is None:
                cluster_ids = self._timed_assign_clusters(emb[:, -1, :])
                self._record_step_routes(cluster_ids, previous_cluster_ids)
            else:
                cluster_ids = fixed_cluster_ids
            pred_emb = self._predict_routed(emb_trunc, act_trunc, cluster_ids)
            emb = torch.cat([emb, pred_emb], dim=1)
            previous_cluster_ids = cluster_ids

            if step < num_future_steps:
                next_act = act_future[:, step : step + 1, :]
                act = torch.cat([act, next_act], dim=1)

        info["predicted_emb"] = rearrange(
            emb,
            "(b s) ... -> b s ...",
            b=batch_size,
            s=num_samples,
        )
        return info

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        return super().get_cost(info_dict, action_candidates)


def build_latent_cluster_switch_model(
    base_ckpt: Path,
    cluster_ckpts: dict[str, Path],
    centroids: np.ndarray,
    device: torch.device,
    *,
    prototype_cluster_ids: np.ndarray | None = None,
    spherical: bool,
    zscore: dict | None = None,
    routing_mode: str = "mpc",
) -> LatentClusterSwitchJEPA:
    base = load_object_checkpoint(base_ckpt, device)
    validate_jepa_components(base, source=base_ckpt)
    cluster_models = {
        name: load_object_checkpoint(path, device) for name, path in cluster_ckpts.items()
    }
    for name, model in cluster_models.items():
        validate_jepa_components(model, source=cluster_ckpts[name])
    model = LatentClusterSwitchJEPA(
        base,
        cluster_models,
        centroids,
        prototype_cluster_ids=prototype_cluster_ids,
        spherical=spherical,
        zscore=zscore,
        routing_mode=routing_mode,
    )
    return model.to(device).eval()


def latent_cluster_train_predictor_dir(
    cluster_seed_name: str, num_clusters: int = 3
) -> Path:
    return (
        THIS_DIR
        / "results"
        / f"tworoom_latent_cluster{num_clusters}_{cluster_seed_name}"
    )


def latent_cluster_predictor_path(cluster_dir: Path, cluster_id: int) -> Path:
    return cluster_dir / f"P_train_cluster{cluster_id}_object.ckpt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(path_value: str | Path, manifest_path: Path) -> Path:
    """Resolve legacy relative manifest paths independently of caller CWD."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # New manifests record absolute paths.  Legacy experiment manifests used
    # repository-relative paths such as experiments/tworoom/... .
    candidates = (PROJECT_ROOT / path, manifest_path.parent / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_predictor_manifest_artifact(
    cluster_predictor_dir: Path,
    *,
    artifact: dict,
    cluster_artifact_dir: Path | None,
    kmeanspp_label_npz: Path | None,
    base_checkpoint: Path,
) -> Path | None:
    """Fail closed for schema-v2 predictor/artifact/provenance mismatches."""
    if (cluster_artifact_dir is None) == (kmeanspp_label_npz is None):
        raise ValueError(
            "Exactly one of cluster_artifact_dir or kmeanspp_label_npz is required"
        )
    meta = artifact["meta"]
    assignment_schema_version = int(meta.get("assignment_schema_version", 1))
    expected_offset = 0
    artifact_offset = int(meta.get("transition_label_offset_steps", 0))
    if artifact_offset != expected_offset:
        raise ValueError(
            "TwoRoom online MPC routes from the current observation, so the cluster "
            f"artifact must use transition_label_offset_steps=0, got {artifact_offset}"
        )
    manifest_path = cluster_predictor_dir / "manifest.json"
    if not manifest_path.exists():
        if assignment_schema_version >= 2:
            raise FileNotFoundError(
                "The cluster artifact uses assignment schema "
                f"v{assignment_schema_version} and therefore requires {manifest_path}"
            )
        return None
    manifest = json.loads(manifest_path.read_text())
    manifest_assignment_schema = int(manifest.get("assignment_schema_version", 1))
    manifest_contract_schema = int(manifest.get("manifest_schema_version", 1))
    strict_manifest = manifest_contract_schema >= 2 or assignment_schema_version >= 2
    if manifest_assignment_schema != assignment_schema_version:
        raise ValueError(
            "Predictor/artifact assignment-schema mismatch: "
            f"{manifest_assignment_schema} != {assignment_schema_version}"
        )
    manifest_offset = int(manifest.get("route_label_offset_steps", 0))
    if manifest_offset != artifact_offset:
        raise ValueError(
            "Predictor/artifact route offset mismatch: "
            f"{manifest_offset} != {artifact_offset}"
        )
    expected_source = (
        manifest.get("cluster_artifact_dir")
        if cluster_artifact_dir is not None
        else manifest.get("kmeanspp_label_npz")
    )
    actual_source = cluster_artifact_dir or kmeanspp_label_npz
    if strict_manifest and not expected_source:
        raise ValueError("Strict manifest lacks its cluster source path")
    if expected_source and actual_source:
        if (
            resolve_manifest_path(expected_source, manifest_path)
            != Path(actual_source).resolve()
        ):
            raise ValueError(
                "Predictor manifest was trained with a different cluster artifact: "
                f"{expected_source} != {actual_source}"
            )
    artifact_hashes = manifest.get("cluster_artifact_sha256", {})
    if strict_manifest and not artifact_hashes:
        raise ValueError("Strict manifest lacks cluster artifact SHA-256 records")
    normalized_artifact_hash_paths = {
        str(resolve_manifest_path(path_text, manifest_path))
        for path_text in artifact_hashes
    }
    required_artifact_paths: list[Path] = []
    if cluster_artifact_dir is not None:
        required_artifact_paths.extend(
            cluster_artifact_dir / name
            for name in ("centroids.npy", "cluster_labels.npz", "cluster_meta.json")
        )
        if (cluster_artifact_dir / "routing_prototypes.npy").exists():
            required_artifact_paths.extend(
                (
                    cluster_artifact_dir / "routing_prototypes.npy",
                    cluster_artifact_dir / "prototype_cluster_ids.npy",
                )
            )
        if artifact.get("zscore") is not None:
            required_artifact_paths.append(cluster_artifact_dir / "zscore_params.npz")
    else:
        required_artifact_paths.append(kmeanspp_label_npz)
        if artifact.get("zscore") is not None:
            required_artifact_paths.append(Path(artifact["zscore"]["path"]))
    if strict_manifest:
        unhashed = [
            str(path)
            for path in required_artifact_paths
            if str(path.resolve()) not in normalized_artifact_hash_paths
        ]
        if unhashed:
            raise ValueError(
                "Strict manifest does not bind every routing artifact file: "
                f"{unhashed}"
            )
    for path_text, expected_hash in artifact_hashes.items():
        path = resolve_manifest_path(path_text, manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest artifact file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Cluster artifact hash mismatch: {path}")

    num_clusters = int(meta.get("num_clusters", 3))
    expected_clusters = {f"cluster{k}" for k in range(num_clusters)}
    cluster_records = manifest.get("clusters", {})
    if strict_manifest and set(cluster_records) != expected_clusters:
        raise ValueError(
            "Incomplete predictor manifest: expected "
            f"{sorted(expected_clusters)}, found {sorted(cluster_records)}"
        )
    checkpoint_hashes = manifest.get("predictor_checkpoint_sha256", {})
    for name in sorted(expected_clusters):
        record = cluster_records.get(name, {})
        output = record.get("output_ckpt")
        if strict_manifest and (
            record.get("status") != "trained" or not output
        ):
            raise ValueError(f"Predictor manifest has no completed {name}")
        if not output:
            continue
        output_path = resolve_manifest_path(output, manifest_path)
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing predictor checkpoint: {output_path}")
        loaded_output_path = latent_cluster_predictor_path(
            cluster_predictor_dir, int(name.removeprefix("cluster"))
        )
        if output_path.resolve() != loaded_output_path.resolve():
            raise ValueError(
                f"Manifest {name} checkpoint is not the checkpoint selected for loading: "
                f"{output_path} != {loaded_output_path}"
            )
        expected_checkpoint_hash = (
            record.get("output_ckpt_sha256") or checkpoint_hashes.get(name)
        )
        if strict_manifest and not expected_checkpoint_hash:
            raise ValueError(f"Strict manifest lacks a hash for {name}")
        if expected_checkpoint_hash and sha256_file(output_path) != expected_checkpoint_hash:
            raise ValueError(f"Predictor checkpoint hash mismatch: {output_path}")

    base_record = manifest.get("base_checkpoint", {})
    expected_base_hash = manifest.get("base_checkpoint_sha256") or (
        base_record.get("sha256") if isinstance(base_record, dict) else None
    )
    expected_base_path = (
        base_record.get("path") if isinstance(base_record, dict) else None
    )
    if strict_manifest and not expected_base_hash:
        raise ValueError("Strict manifest lacks the base checkpoint SHA-256")
    if (
        expected_base_path
        and resolve_manifest_path(expected_base_path, manifest_path)
        != base_checkpoint.resolve()
    ):
        raise ValueError(
            "Base checkpoint path mismatch: "
            f"{expected_base_path} != {base_checkpoint}"
        )
    if expected_base_hash and sha256_file(base_checkpoint) != expected_base_hash:
        raise ValueError(
            f"Base checkpoint hash mismatch for {base_checkpoint}"
        )
    return manifest_path


def predictor_manifest_provenance(manifest_path: Path | None) -> dict | None:
    """Capture immutable hashes needed to reproduce an evaluation later."""

    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text())
    clusters = manifest.get("clusters", {})
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "immutable_config_sha256": manifest.get("immutable_config_sha256"),
        "cluster_artifact_sha256": manifest.get("cluster_artifact_sha256", {}),
        "base_checkpoint": manifest.get("base_checkpoint"),
        "clusters": {
            name: {
                "status": record.get("status"),
                "output_ckpt": record.get("output_ckpt"),
                "output_ckpt_sha256": record.get("output_ckpt_sha256"),
                "output_metadata": record.get("output_metadata"),
                "output_metadata_sha256": record.get("output_metadata_sha256"),
            }
            for name, record in clusters.items()
            if isinstance(record, dict)
        },
    }


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def load_object_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    return model


def validate_jepa_components(model: torch.nn.Module, *, source: Path) -> None:
    """Validate the shared module contract of official and vendored JEPA models.

    Official ``stable_worldmodel`` checkpoints serialize ``LeWM`` while older
    TwoRoom checkpoints serialize this repository's vendored ``JEPA`` class.
    The routing wrappers need the same five modules from either implementation,
    so checking the serialized Python class would reject a compatible model.
    """

    required = ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
    missing = [
        name
        for name in required
        if not isinstance(getattr(model, name, None), torch.nn.Module)
    ]
    if missing:
        raise TypeError(
            f"Checkpoint {source} does not implement the JEPA module contract; "
            f"missing components: {missing}"
        )


def build_region_switch_model(
    base_ckpt: Path,
    region_ckpts: dict[str, Path],
    proprio_processor,
    device: torch.device,
    region_selector,
) -> RegionSwitchJEPA:
    base = load_object_checkpoint(base_ckpt, device)
    validate_jepa_components(base, source=base_ckpt)
    region_models = {
        name: load_object_checkpoint(path, device) for name, path in region_ckpts.items()
    }
    for name, model in region_models.items():
        validate_jepa_components(model, source=region_ckpts[name])
    thresholds = tworoom_geometry_thresholds()
    return RegionSwitchJEPA(
        base,
        region_models,
        proprio_processor,
        thresholds,
        region_selector,
    )


def geometry_train_predictor_path(region: str) -> Path:
    return GEOMETRY_TRAIN_PRED_DIR / f"P_train_{region}_object.ckpt"


def load_eval_start_indices(
    path: Path | None, cfg, dataset, col_name: str, ep_indices
) -> np.ndarray:
    if path is not None and path.exists():
        with path.open() as f:
            data = json.load(f)
        starts = np.asarray(data["eval_start_indices"], dtype=np.int64)
        if len(starts) > cfg.eval.num_eval:
            starts = starts[: cfg.eval.num_eval]
        elif len(starts) < cfg.eval.num_eval:
            raise ValueError(
                f"eval_start_indices has {len(starts)} rows but num_eval={cfg.eval.num_eval}"
            )
        print(f"Reusing {len(starts)} eval starts from {path}", flush=True)
        return starts

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.", flush=True)
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    return np.sort(valid_indices[random_episode_indices])


def resolve_model(
    mode: str,
    checkpoint: Path,
    device: torch.device,
    proprio_processor,
    *,
    region_ckpt_overrides: dict[str, Path] | None = None,
    cluster_artifact_dir: Path | None = None,
    cluster_predictor_dir: Path | None = None,
    kmeanspp_label_npz: Path | None = None,
    zscore_params_npz: Path | None = None,
    lap_run_dir: Path | None = None,
    latent_routing: str = "mpc",
) -> tuple[torch.nn.Module, dict]:
    meta: dict = {"mode": mode}
    if mode == "baseline":
        meta["checkpoint"] = str(checkpoint)
        return load_object_checkpoint(checkpoint, device), meta

    if mode == "lap":
        if lap_run_dir is None:
            raise ValueError("--lap-run-dir is required for --mode lap")
        lap_run_dir = lap_run_dir.resolve(strict=True)
        artifact = PartitionArtifact.load(lap_run_dir / "partition")
        num_clusters = artifact.num_regions
        cluster_ckpts = {
            f"cluster{k}": lap_run_dir / f"P_train_cluster{k}_object.ckpt"
            for k in range(num_clusters)
        }
        for name, path in cluster_ckpts.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} predictor: {path}")
        manifest_path = lap_run_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        recorded = manifest.get("predictor_checkpoint_sha256", {})
        for name, path in cluster_ckpts.items():
            if name in recorded and sha256_file(path) != recorded[name]:
                raise ValueError(f"Predictor checkpoint hash mismatch: {path}")
        switch_model = build_latent_cluster_switch_model(
            checkpoint,
            cluster_ckpts,
            artifact.prototypes,
            device,
            prototype_cluster_ids=artifact.prototype_region_ids,
            spherical=True,
            zscore={
                "mu": artifact.mean,
                "sigma": artifact.scale,
                "eps": 1e-12,
            },
            routing_mode=latent_routing,
        )
        meta.update(
            {
                "base_encoder_checkpoint": str(checkpoint),
                "lap_run_dir": str(lap_run_dir),
                "lap_manifest": str(manifest_path) if manifest_path.exists() else None,
                "cluster_predictors": {
                    name: str(path) for name, path in cluster_ckpts.items()
                },
                "num_clusters": num_clusters,
                "num_routing_prototypes": int(len(artifact.prototypes)),
                "partition_metadata": artifact.metadata,
                "switch_timing": (
                    "each imagined rollout step from predicted latent"
                    if latent_routing == "step"
                    else "once per MPC replan from the observed latent"
                ),
                "latent_routing": latent_routing,
            }
        )
        return switch_model, meta

    if mode == "latent_cluster3":
        if (cluster_artifact_dir is None) == (kmeanspp_label_npz is None):
            raise ValueError(
                "Exactly one of --cluster-artifact-dir or --kmeanspp-label-npz "
                "is required for latent_cluster3 mode"
            )
        artifact = resolve_cluster_source(
            cluster_artifact_dir=cluster_artifact_dir,
            kmeanspp_label_npz=kmeanspp_label_npz,
            zscore_params_npz=zscore_params_npz,
        )
        num_clusters = int(artifact["meta"].get("num_clusters", 3))
        requires_zscore = (
            str(artifact["meta"].get("preprocess", "")).lower() == "zscore_l2"
            or "zscore" in str(artifact["meta"].get("method", "")).lower()
        )
        if requires_zscore and artifact.get("zscore") is None:
            raise FileNotFoundError(
                "Inference routing requires the Z-score parameters used to fit "
                "the cluster routing vectors"
            )
        if cluster_predictor_dir is None:
            if kmeanspp_label_npz is not None:
                cluster_predictor_dir = (
                    THIS_DIR / "results" / f"tworoom_latent_kmeanspp_{kmeanspp_label_npz.stem}"
                )
            else:
                cluster_predictor_dir = latent_cluster_train_predictor_dir(
                    cluster_artifact_dir.name, num_clusters
                )
        cluster_ckpts = {
            f"cluster{k}": latent_cluster_predictor_path(cluster_predictor_dir, k)
            for k in range(num_clusters)
        }
        manifest_path = cluster_predictor_dir / "manifest.json"
        # Recovery/overwrite jobs use the same lock.  Keep it across hash
        # validation and torch.load so the bytes loaded into memory are exactly
        # the bytes authenticated by the manifest (no validate->load TOCTOU).
        with manifest_file_lock(manifest_path):
            for name, path in cluster_ckpts.items():
                if not path.exists():
                    raise FileNotFoundError(f"Missing cluster predictor: {path}")
            predictor_manifest = validate_predictor_manifest_artifact(
                cluster_predictor_dir,
                artifact=artifact,
                cluster_artifact_dir=cluster_artifact_dir,
                kmeanspp_label_npz=kmeanspp_label_npz,
                base_checkpoint=checkpoint,
            )
            manifest_provenance = predictor_manifest_provenance(predictor_manifest)
            spherical = bool(artifact["meta"].get("spherical", True))
            switch_model = build_latent_cluster_switch_model(
                checkpoint,
                cluster_ckpts,
                artifact["centroids"],
                device,
                prototype_cluster_ids=artifact.get("prototype_cluster_ids"),
                spherical=spherical,
                zscore=artifact.get("zscore"),
                routing_mode=latent_routing,
            )
        meta.update(
            {
                "base_encoder_checkpoint": str(checkpoint),
                "cluster_artifact_dir": str(cluster_artifact_dir) if cluster_artifact_dir else None,
                "kmeanspp_label_npz": str(kmeanspp_label_npz) if kmeanspp_label_npz else None,
                "zscore_params_npz": (
                    artifact["zscore"].get("path")
                    if artifact.get("zscore")
                    else None
                ),
                "cluster_predictors": {k: str(v) for k, v in cluster_ckpts.items()},
                "cluster_predictor_manifest": (
                    str(predictor_manifest) if predictor_manifest else None
                ),
                "cluster_predictor_manifest_provenance": manifest_provenance,
                "cluster_rule": artifact["meta"].get("classification_rule"),
                "num_routing_prototypes": int(len(artifact["centroids"])),
                "prototype_cluster_ids": artifact[
                    "prototype_cluster_ids"
                ].tolist(),
                "cluster_fit_timing_sec": artifact["meta"].get("timing_sec"),
                "switch_timing": (
                    "each imagined rollout step from predicted latent nearest centroid"
                    if latent_routing == "step"
                    else (
                        "each MPC rollout: one observed-latent route per environment, "
                        "fixed across its CEM candidates and imagined steps"
                    )
                ),
                "latent_routing": latent_routing,
            }
        )
        return switch_model, meta

    if mode == "rooms3":
        region_ckpts = {
            "left_room": geometry_train_predictor_path("left_room"),
            "doorway_corridor": geometry_train_predictor_path("doorway_corridor"),
            "right_room": geometry_train_predictor_path("right_room"),
        }
        selector = geometry_rooms3_key
        meta_rule = "geometry x: left<107, doorway 107-117, right>117"
    elif mode == "priority5":
        region_ckpts = {
            region: geometry_train_predictor_path(region) for region in PRIORITY5_ORDER
        }
        selector = geometry_priority5_key
        meta_rule = (
            "geometry priority: doorway > near_wall > common > right/left by x"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if region_ckpt_overrides:
        for region, path in region_ckpt_overrides.items():
            if region in region_ckpts:
                region_ckpts[region] = Path(path)
            else:
                raise ValueError(f"--region-ckpt override for unknown region '{region}' (valid: {list(region_ckpts)})")

    for name, path in region_ckpts.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing region predictor: {path}")
    meta.update(
        {
            "base_encoder_checkpoint": str(checkpoint),
            "region_predictors": {k: str(v) for k, v in region_ckpts.items()},
            "region_rule": meta_rule,
            "switch_timing": "each MPC replan from current proprio",
            "geometry_thresholds": tworoom_geometry_thresholds(),
        }
    )
    return build_region_switch_model(
        checkpoint, region_ckpts, proprio_processor, device, selector
    ), meta


def run_eval(
    cfg,
    out_dir: Path,
    experiment_mode: str,
    *,
    checkpoint: Path = DEFAULT_GLOBAL_CKPT,
    eval_start_indices_path: Path | None = DEFAULT_BASELINE_STARTS,
    region_ckpt_overrides: dict[str, Path] | None = None,
    cluster_artifact_dir: Path | None = None,
    cluster_predictor_dir: Path | None = None,
    kmeanspp_label_npz: Path | None = None,
    zscore_params_npz: Path | None = None,
    lap_run_dir: Path | None = None,
    latent_routing: str = "mpc",
) -> dict:
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": img_transform(cfg.eval.img_size),
        "goal": img_transform(cfg.eval.img_size),
    }

    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    device = torch.device(str(cfg.solver.device))
    model, model_meta = resolve_model(
        experiment_mode, checkpoint, device, process["proprio"],
        region_ckpt_overrides=region_ckpt_overrides,
        cluster_artifact_dir=cluster_artifact_dir,
        cluster_predictor_dir=cluster_predictor_dir,
        kmeanspp_label_npz=kmeanspp_label_npz,
        zscore_params_npz=zscore_params_npz,
        lap_run_dir=lap_run_dir,
        latent_routing=latent_routing,
    )
    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )

    random_episode_indices = load_eval_start_indices(
        eval_start_indices_path, cfg, dataset, col_name, ep_indices
    )
    print(random_episode_indices, flush=True)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=out_dir,
    )
    elapsed = time.time() - start_time
    print(metrics, flush=True)

    classify_meta: dict = {}
    if isinstance(model, LatentClusterSwitchJEPA):
        model.finalize_classify_timing()
        timed_calls = max(model.classify_timed_count, 1)
        timed_assignments = max(model.classify_timed_assignment_count, 1)
        classify_meta = {
            "inference_classify_count": int(model.classify_count),
            "inference_classify_assignments": int(model.classify_assignment_count),
            "inference_classify_timed_calls": int(model.classify_timed_count),
            "inference_classify_timed_assignments": int(
                model.classify_timed_assignment_count
            ),
            "inference_classify_time_sample_sec": float(model.classify_time_sec),
            "inference_classify_per_call_ms": float(
                1000.0 * model.classify_time_sec / timed_calls
            ),
            "inference_classify_per_assignment_us": float(
                1_000_000.0 * model.classify_time_sec / timed_assignments
            ),
            "inference_classify_timing_backend": (
                "cuda_events" if model.cluster_centroids.is_cuda else "perf_counter"
            ),
            "inference_mpc_route_cache_hits": int(model.mpc_route_cache_hits),
            "inference_mpc_route_cache_misses": int(model.mpc_route_cache_misses),
        }
        if model.routing_mode == "step":
            route_histogram = model.route_histogram.detach().cpu().tolist()
            route_total = max(sum(route_histogram), 1)
            route_transitions = int(model.route_transition_count.item())
            route_switches = int(model.route_switch_count.item())
            classify_meta.update({
                "inference_route_calls": int(model.route_call_count),
                "inference_route_assignments": int(model.route_assignment_count),
                "inference_route_histogram": {
                    name: int(route_histogram[i])
                    for i, name in enumerate(model.cluster_names)
                },
                "inference_route_fraction": {
                    name: float(route_histogram[i] / route_total)
                    for i, name in enumerate(model.cluster_names)
                },
                "inference_route_transitions": route_transitions,
                "inference_route_switches": route_switches,
                "inference_route_switch_rate": float(
                    route_switches / max(route_transitions, 1)
                ),
            })

    result = {
        "mode": experiment_mode,
        "seed": int(cfg.seed),
        "num_eval": int(cfg.eval.num_eval),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "eval_budget": int(cfg.eval.eval_budget),
        **model_meta,
        **classify_meta,
        "metrics": {
            "success_rate": float(metrics["success_rate"]),
            "episode_successes": metrics["episode_successes"].astype(bool).tolist(),
            "seeds": (
                metrics["seeds"].astype(int).tolist()
                if metrics.get("seeds") is not None
                else None
            ),
        },
        "eval_start_indices": random_episode_indices.astype(int).tolist(),
        "evaluation_time_sec": elapsed,
    }

    with (out_dir / "results.json").open("w") as f:
        json.dump(result, f, indent=2)

    with (out_dir / "results.txt").open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== MODEL ====\n")
        f.write(json.dumps(model_meta, indent=2))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed} seconds\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=("baseline", "rooms3", "priority5", "latent_cluster3", "lap"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_GLOBAL_CKPT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults depend on --mode",
    )
    parser.add_argument(
        "--eval-start-indices",
        type=Path,
        default=None,
        help="Reuse dataset row indices from a prior results.json (for fair comparison)",
    )
    parser.add_argument(
        "--sample-eval-starts",
        action="store_true",
        help="Sample fresh eval_start_indices from --seed (ignores --eval-start-indices)",
    )
    parser.add_argument(
        "--region-ckpt",
        action="append",
        default=[],
        metavar="REGION=PATH",
        help="Override checkpoint for a specific region, e.g. doorway_corridor=/path/to/ckpt. Repeatable.",
    )
    parser.add_argument(
        "--config-name",
        default="tworoom",
        help="Hydra config under config/eval/ (without .yaml)",
    )
    parser.add_argument(
        "--dataset-tag",
        default="tworoom",
        help="Filesystem/result prefix only; the official task comes from --config-name.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=THIS_DIR / "results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory containing <eval.dataset_name>.h5.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Model and solver device; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--goal-offset",
        type=int,
        default=None,
        help="Override eval.goal_offset_steps from the config",
    )
    parser.add_argument(
        "--eval-budget",
        type=int,
        default=None,
        help="Override eval.eval_budget from the config",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=None,
        help="Override eval.num_eval (useful for smoke tests)",
    )
    parser.add_argument(
        "--cluster-artifact-dir",
        type=Path,
        default=None,
        help="Cluster centroids/labels dir for --mode latent_cluster3",
    )
    parser.add_argument(
        "--kmeanspp-label-npz",
        type=Path,
        default=None,
        help="K-means++ label npz for --mode latent_cluster3 (paper main config)",
    )
    parser.add_argument(
        "--zscore-params",
        type=Path,
        default=None,
        help="zscore_params.npz for K-means++ routing at inference",
    )
    parser.add_argument(
        "--cluster-predictor-dir",
        type=Path,
        default=None,
        help="Directory with P_train_cluster{k}_object.ckpt (defaults from artifact name)",
    )
    parser.add_argument(
        "--lap-run-dir",
        type=Path,
        default=None,
        help="Portable LAP training run with partition/ and P_train_cluster*.ckpt.",
    )
    parser.add_argument(
        "--latent-routing",
        choices=("mpc", "step"),
        default="mpc",
        help=(
            "For latent_cluster3: route once from the observed latent per MPC replan "
            "or independently at every imagined rollout step from the predicted latent"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        if args.mode == "baseline":
            args.out_dir = args.results_root / (
                f"{args.dataset_tag}_success_rate_baseline_seed{args.seed}"
            )
        elif args.mode == "lap":
            run_tag = args.lap_run_dir.name if args.lap_run_dir else "missing_lap_run"
            args.out_dir = args.results_root / (
                f"{args.dataset_tag}_success_rate_{run_tag}_{args.latent_routing}_"
                f"seed{args.seed}"
            )
        elif args.mode == "latent_cluster3":
            source_tag = (
                args.cluster_artifact_dir.name
                if args.cluster_artifact_dir is not None
                else (
                    args.kmeanspp_label_npz.stem
                    if args.kmeanspp_label_npz is not None
                    else "missing_artifact"
                )
            )
            predictor_tag = (
                args.cluster_predictor_dir.name
                if args.cluster_predictor_dir is not None
                else "default_predictors"
            )
            args.out_dir = (
                args.results_root
                / (
                    f"{args.dataset_tag}_success_rate_latent_{source_tag}_{predictor_tag}_"
                    f"{args.latent_routing}_seed{args.seed}"
                )
            )
        else:
            args.out_dir = args.results_root / (
                f"{args.dataset_tag}_success_rate_{args.mode}_seed{args.seed}"
            )

    eval_start_indices_path = args.eval_start_indices
    if args.sample_eval_starts:
        eval_start_indices_path = None
    elif eval_start_indices_path is None and args.mode != "baseline":
        eval_start_indices_path = (
            args.results_root
            / f"{args.dataset_tag}_success_rate_baseline_seed{args.seed}"
            / "results.json"
        )

    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "config" / "eval"),
    ):
        cfg = hydra.compose(config_name=args.config_name)
    cfg.seed = args.seed
    if args.cache_dir is not None:
        cfg.cache_dir = str(args.cache_dir.resolve(strict=True))
    cfg.solver.device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    if args.goal_offset is not None:
        cfg.eval.goal_offset_steps = args.goal_offset
    if args.eval_budget is not None:
        cfg.eval.eval_budget = args.eval_budget
    if args.num_eval is not None:
        cfg.eval.num_eval = args.num_eval
    region_ckpt_overrides = {}
    for item in args.region_ckpt:
        if "=" not in item:
            raise ValueError(f"--region-ckpt must be REGION=PATH, got: {item!r}")
        region, path = item.split("=", 1)
        region_ckpt_overrides[region] = Path(path)
    if args.kmeanspp_label_npz is not None and args.zscore_params is None:
        default_z = args.kmeanspp_label_npz.parent.parent / "zscore_params.npz"
        if default_z.exists():
            args.zscore_params = default_z

    run_eval(
        cfg,
        args.out_dir,
        args.mode,
        checkpoint=args.checkpoint,
        eval_start_indices_path=eval_start_indices_path,
        region_ckpt_overrides=region_ckpt_overrides or None,
        cluster_artifact_dir=args.cluster_artifact_dir,
        cluster_predictor_dir=args.cluster_predictor_dir,
        kmeanspp_label_npz=args.kmeanspp_label_npz,
        zscore_params_npz=args.zscore_params,
        lap_run_dir=args.lap_run_dir,
        latent_routing=args.latent_routing,
    )


if __name__ == "__main__":
    main()
