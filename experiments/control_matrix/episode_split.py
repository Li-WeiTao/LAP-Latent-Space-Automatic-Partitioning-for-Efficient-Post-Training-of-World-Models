"""Episode-level train/eval split utilities for formal region-risk experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from experiments.control_matrix.region_risk_lib import (
    atomic_write_json,
    episode_ids_at_starts,
    sha256_file,
)
from experiments.tworoom.gauge_drift import DATASETS, choose_state_key
from experiments.tworoom.predictor_rule_drift import valid_transition_starts

FORMAL_SPLIT_SEED = 20260801
FORMAL_TRAIN_FRACTION = 0.9
FORMAL_BOOTSTRAP_SEED = 20260801


@dataclass(frozen=True)
class EpisodeSplit:
    train_episode_ids: tuple[int, ...]
    eval_episode_ids: tuple[int, ...]
    train_starts: np.ndarray
    eval_starts: np.ndarray
    split_seed: int
    train_fraction: float
    data_file: Path
    dataset_name: str
    history_size: int
    num_preds: int
    frameskip: int
    valid_start_seed: int

    @property
    def train_eval_episode_disjoint(self) -> bool:
        return not set(self.train_episode_ids).intersection(self.eval_episode_ids)

    def to_manifest_payload(
        self,
        *,
        train_starts_path: Path,
        eval_starts_path: Path,
        action_norm_starts_path: Path,
        nominal_train_num_transitions: int,
        nominal_eval_num_transitions: int,
        written_train_num_transitions: int,
        written_eval_num_transitions: int,
        subsampled: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_name": self.dataset_name,
            "data_file": str(self.data_file.resolve()),
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "valid_start_seed": self.valid_start_seed,
            "history_size": self.history_size,
            "num_preds": self.num_preds,
            "frameskip": self.frameskip,
            "train_episode_ids": list(self.train_episode_ids),
            "eval_episode_ids": list(self.eval_episode_ids),
            "train_num_episodes": len(self.train_episode_ids),
            "eval_num_episodes": len(self.eval_episode_ids),
            "train_num_transitions": int(written_train_num_transitions),
            "eval_num_transitions": int(written_eval_num_transitions),
            "nominal_train_num_transitions": int(nominal_train_num_transitions),
            "written_train_num_transitions": int(written_train_num_transitions),
            "nominal_eval_num_transitions": int(nominal_eval_num_transitions),
            "written_eval_num_transitions": int(written_eval_num_transitions),
            "subsampled": bool(subsampled),
            "train_eval_episode_disjoint": self.train_eval_episode_disjoint,
            "paths": {
                "train_starts": str(train_starts_path.resolve()),
                "eval_starts": str(eval_starts_path.resolve()),
                "action_norm_starts": str(action_norm_starts_path.resolve()),
            },
            "sha256": {
                "train_starts": sha256_file(train_starts_path),
                "eval_starts": sha256_file(eval_starts_path),
                "action_norm_starts": sha256_file(action_norm_starts_path),
            },
        }


def compute_episode_split(
    data_file: Path,
    dataset_name: str,
    *,
    history_size: int = 3,
    num_preds: int = 1,
    frameskip: int = 5,
    train_fraction: float = FORMAL_TRAIN_FRACTION,
    split_seed: int = FORMAL_SPLIT_SEED,
    valid_start_seed: int = 0,
) -> EpisodeSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    spec = DATASETS[dataset_name]
    seq_len = history_size + num_preds
    data_file = data_file.resolve(strict=True)
    with h5py.File(data_file, "r") as handle:
        state_key = choose_state_key(handle, spec, None)
        all_starts = valid_transition_starts(
            handle,
            spec,
            state_key,
            seq_len,
            frameskip,
            0,
            valid_start_seed,
        )
        episode_ids = episode_ids_at_starts(data_file, all_starts)
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(unique_episodes)
    train_n = int(round(len(perm) * train_fraction))
    train_episode_set = set(map(int, perm[:train_n]))
    eval_episode_set = set(map(int, perm[train_n:]))
    train_mask = np.asarray([int(ep) in train_episode_set for ep in episode_ids], dtype=bool)
    eval_mask = np.asarray([int(ep) in eval_episode_set for ep in episode_ids], dtype=bool)
    train_starts = np.sort(all_starts[train_mask].astype(np.int64))
    eval_starts = np.sort(all_starts[eval_mask].astype(np.int64))
    if len(train_starts) == 0 or len(eval_starts) == 0:
        raise RuntimeError("episode split produced an empty train or eval start pool")
    if not train_episode_set.isdisjoint(eval_episode_set):
        raise RuntimeError("episode split is not episode-disjoint")
    return EpisodeSplit(
        train_episode_ids=tuple(sorted(train_episode_set)),
        eval_episode_ids=tuple(sorted(eval_episode_set)),
        train_starts=train_starts,
        eval_starts=eval_starts,
        split_seed=split_seed,
        train_fraction=train_fraction,
        data_file=data_file,
        dataset_name=dataset_name,
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
        valid_start_seed=valid_start_seed,
    )


def write_split_artifacts(
    split: EpisodeSplit,
    out_dir: Path,
    *,
    max_train_starts: int = 0,
    max_eval_starts: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    nominal_train = int(len(split.train_starts))
    nominal_eval = int(len(split.eval_starts))
    train_starts = split.train_starts
    eval_starts = split.eval_starts
    if max_train_starts > 0:
        train_starts = train_starts[:max_train_starts]
    if max_eval_starts > 0:
        eval_starts = eval_starts[:max_eval_starts]
    written_train = int(len(train_starts))
    written_eval = int(len(eval_starts))
    subsampled = written_train < nominal_train or written_eval < nominal_eval
    train_starts_path = out_dir / "train_starts.npy"
    eval_starts_path = out_dir / "eval_starts.npy"
    action_norm_starts_path = out_dir / "action_norm_starts.npy"
    np.save(train_starts_path, train_starts)
    np.save(eval_starts_path, eval_starts)
    np.save(action_norm_starts_path, split.train_starts)
    manifest = split.to_manifest_payload(
        train_starts_path=train_starts_path,
        eval_starts_path=eval_starts_path,
        action_norm_starts_path=action_norm_starts_path,
        nominal_train_num_transitions=nominal_train,
        nominal_eval_num_transitions=nominal_eval,
        written_train_num_transitions=written_train,
        written_eval_num_transitions=written_eval,
        subsampled=subsampled,
    )
    atomic_write_json(out_dir / "split_manifest.json", manifest)
    return manifest


def load_split_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported split manifest schema: {path}")
    return payload


def split_paths_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths = manifest.get("paths", {})
    return {
        key: Path(str(paths[key])).resolve(strict=True)
        for key in ("train_starts", "eval_starts", "action_norm_starts")
    }
