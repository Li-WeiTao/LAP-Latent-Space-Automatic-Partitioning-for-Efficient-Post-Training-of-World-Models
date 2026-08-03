#!/usr/bin/env python3
"""Compare Sub-JEPA matrix protocol against the canonical LeWM TwoRoom matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.jepa_checkpoint_probe import sha256_file
from experiments.control_matrix.resolve_jepa_matrix_config import (  # noqa: E402
    parse_args as parse_matrix_args,
    resolve_config,
)


ALLOWED_DIFF_FIELDS = frozenset(
    {
        "model_family",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_class",
        "cache_path",
        "cache_sha256",
        "latent_shape",
        "work_root",
        "cache_dir",
        "output_paths",
        "task_spec_path",
        "implementation_backend",
        # Sub-JEPA core matrix omits LeWM-only methods (random_voronoi, joint).
        "methods",
        "skip_joint",
    }
)

DEFAULT_DATASET = "/data/sicong/weitao/datasets/lewm/tworoom.h5"
DEFAULT_LEWM_CHECKPOINT = "/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"
DEFAULT_SUBJEPA_CHECKPOINT = (
    "/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt"
)
DEFAULT_TASK_SPEC = PROJECT_ROOT / "configs/experiments/tasks/tworoom.json"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "config/eval/tworoom.yaml"


def _sha256_starts(path: Path) -> str:
    array = np.load(path)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _paired_start_hashes(results_root: Path, eval_seeds: list[int]) -> dict[int, str]:
    hashes: dict[int, str] = {}
    for seed in eval_seeds:
        path = results_root / f"tworoom_success_rate_baseline_seed{seed}" / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        starts = payload["eval_start_indices"]
        digest = hashlib.sha256(
            json.dumps([int(v) for v in starts], separators=(",", ":")).encode("ascii")
        ).hexdigest()
        hashes[seed] = digest
    return hashes


def _load_eval_yaml() -> dict[str, Any]:
    with DEFAULT_EVAL_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_side(
    *,
    model_family: str,
    checkpoint: str,
    work_root: str,
    cache_dir: str,
    methods: str,
    skip_joint: bool,
) -> dict[str, Any]:
    env_backup = {
        key: os.environ.get(key)
        for key in ("SKIP_JOINT", "METHODS", "CACHE_DIR", "MODEL_FAMILY")
    }
    os.environ["SKIP_JOINT"] = "1" if skip_joint else "0"
    os.environ["METHODS"] = methods
    os.environ["CACHE_DIR"] = cache_dir
    os.environ["MODEL_FAMILY"] = model_family
    args = parse_matrix_args(
        [
            "--model-family",
            model_family,
            "--task-spec",
            str(DEFAULT_TASK_SPEC),
            "--dataset",
            DEFAULT_DATASET,
            "--checkpoint",
            checkpoint,
            "--eval-config-name",
            "tworoom",
            "--work-root",
            work_root,
            "--cache-dir",
            cache_dir,
            "--methods",
            methods,
            "--train-seeds",
            "0,42,625",
            "--partition-seeds",
            "0,1,2",
            "--eval-seeds",
            "0,1,2,3,4",
        ]
        + (["--skip-joint"] if skip_joint else [])
    )
    resolved = resolve_config(args)
    for key, value in env_backup.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return resolved.to_json()


def _training_protocol_from_code() -> dict[str, Any]:
    return {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
        "min_region_samples": 256,
        "history_size": 3,
        "num_preds": 1,
        "precision": "fp32",
        "checkpoint_selection_rule": (
            "minimum loss on same-region training cache"
        ),
        "predictor_loss": (
            "MSE between model.predict(ctx_emb, ctx_act) and tgt_emb "
            "(final-token latent regression; backends/lewm/finetuning.py::predictor_loss)"
        ),
        "trainable_parameter_allowlist": ["predictor", "pred_proj"],
        "frozen_parameter_prefixes": ["encoder", "projector", "action_encoder"],
        "action_embedding_usage": (
            "precomputed act_emb windows stored in embedding_cache.npz; "
            "training consumes cached act_emb alongside emb"
        ),
    }


def _cache_protocol(cache_dir: Path, starts_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prepare_script": "experiments/control_matrix/prepare_lewm_cache.py",
        "regions": ["common"],
        "restrict_to_train_split": True,
        "predictor_prefix": "train_",
        "transition_batch_size": 128,
        "frame_batch_size": 512,
        "encoding": "lossless_precomputed unique_frame_inverse_reconstruction",
    }
    if starts_path.is_file():
        payload["train_global_reference_starts_path"] = str(starts_path)
        payload["train_global_reference_starts_sha256"] = _sha256_starts(starts_path)
        payload["num_train_global_reference_starts"] = int(np.load(starts_path).size)
    starts_manifest = starts_path.parent / "starts_only_manifest.json"
    if starts_manifest.is_file():
        manifest = json.loads(starts_manifest.read_text(encoding="utf-8"))
        payload.update(
            {
                "num_all_starts": manifest.get("num_all_starts"),
                "train_fraction": manifest.get("train_config", {}).get(
                    "train_fraction"
                ),
                "split_seed": manifest.get("train_config", {}).get("split_seed"),
                "max_starts": manifest.get("train_config", {}).get("max_starts"),
            }
        )
    cache_path = cache_dir / "embedding_cache.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            payload["cache_path"] = str(cache_path)
            payload["cache_sha256"] = sha256_file(cache_path)
            payload["latent_shape"] = {
                key: list(data[key].shape) for key in ("emb", "act_emb")
            }
            payload["num_transitions"] = int(data["emb"].shape[0])
    return payload


def _eval_protocol(resolved: dict[str, Any], task_spec: dict[str, Any]) -> dict[str, Any]:
    eval_yaml = _load_eval_yaml()
    return {
        "short_goal_offset": resolved["short_goal_offset"],
        "long_goal_offset": resolved["long_goal_offset"],
        "eval_budget": resolved["eval_budget"],
        "num_eval": resolved["num_eval"],
        "eval_config_name": resolved["eval_config_name"],
        "plan_horizon": task_spec.get("plan_horizon", eval_yaml["plan_config"]["horizon"]),
        "plan_receding_horizon": task_spec.get(
            "plan_receding_horizon", eval_yaml["plan_config"]["receding_horizon"]
        ),
        "plan_action_block": task_spec.get(
            "plan_action_block", eval_yaml["plan_config"]["action_block"]
        ),
        "evaluator": "experiments/tworoom/tworoom_success_rate_eval.py",
        "latent_routing": "mpc",
        "paired_start_source_short": (
            "experiments/tworoom/results/tworoom_success_rate_baseline_seed{eval_seed}"
        ),
        "paired_start_source_long": (
            "experiments/tworoom/results/tworoom_success_rate_baseline_exp6_seed{eval_seed}"
        ),
    }


def _bootstrap_protocol() -> dict[str, Any]:
    return {
        "script": (
            "experiments/tworoom/subjepa/matrix/scripts/matrix_paired_bootstrap.py"
        ),
        "reps": 200_000,
        "unit": "paired eval seed blocks after averaging partition seeds per train seed",
        "comparisons": [
            "Global-FT50 vs Official baseline",
            "K-means++ K3-50 vs Global-FT50",
            "Spectral K3-50 vs Global-FT50",
        ],
        "ci": "2.5/97.5 percentile of paired block bootstrap delta in success-rate points",
    }


def _compare_field(
    field: str,
    reference: Any,
    candidate: Any,
) -> dict[str, Any]:
    leaf = field.rsplit(".", 1)[-1]
    if reference == candidate:
        status = "match"
    elif leaf in ALLOWED_DIFF_FIELDS or field in ALLOWED_DIFF_FIELDS:
        status = "allowed_diff"
    else:
        status = "mismatch"
    return {
        "field": field,
        "reference": reference,
        "candidate": candidate,
        "status": status,
    }


def build_protocol_bundle(
    *,
    resolved: dict[str, Any],
    cache_root: Path,
    starts_path: Path,
    task_spec: dict[str, Any],
    paired_start_hashes: dict[int, str] | None = None,
) -> dict[str, Any]:
    bundle = {
        "resolved_manifest": resolved,
        "dataset_transition_scope": _cache_protocol(cache_root, starts_path),
        "context_target_construction": {
            "history_size": resolved["history_size"],
            "num_preds": resolved["num_preds"],
            "frameskip": resolved["frameskip"],
            "sequence_length": resolved["history_size"] + resolved["num_preds"],
            "img_size": resolved["img_size"],
        },
        "training": _training_protocol_from_code(),
        "partition_config": {
            "num_clusters": 3,
            "partition_seeds": resolved["partition_seeds"],
            "methods": resolved["methods"],
            "fit_script": "experiments/control_matrix/fit_partition.py",
            "frameskip": resolved["frameskip"],
        },
        "matrix_scope": {
            "train_seeds": resolved["train_seeds"],
            "eval_seeds": resolved["eval_seeds"],
            "skip_joint": resolved["skip_joint"],
        },
        "evaluation": _eval_protocol(resolved, task_spec),
        "bootstrap_aggregation": _bootstrap_protocol(),
    }
    if paired_start_hashes is not None:
        bundle["paired_start_sha256_by_eval_seed"] = paired_start_hashes
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/tworoom/subjepa/matrix",
    )
    parser.add_argument(
        "--formal-cache-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/tworoom/subjepa/formal/preparation",
    )
    parser.add_argument(
        "--lewm-results-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/tworoom/results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/tworoom/subjepa/matrix/manifests/protocol_parity.json",
    )
    parser.add_argument(
        "--reference-work-root",
        default="experiments/tworoom/matrix",
    )
    parser.add_argument("--reference-cache-dir", default=str(Path.home() / ".stable_worldmodel"))
    args = parser.parse_args()
    args.matrix_root = (PROJECT_ROOT / args.matrix_root).resolve()
    args.formal_cache_root = (PROJECT_ROOT / args.formal_cache_root).resolve()
    args.lewm_results_root = (PROJECT_ROOT / args.lewm_results_root).resolve()
    args.out = (PROJECT_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out

    task_spec = json.loads(DEFAULT_TASK_SPEC.read_text(encoding="utf-8"))
    eval_seeds = [0, 1, 2, 3, 4]
    lewm_paired = _paired_start_hashes(args.lewm_results_root, eval_seeds)

    lewm_resolved = _resolve_side(
        model_family="lewm",
        checkpoint=DEFAULT_LEWM_CHECKPOINT,
        work_root=args.reference_work_root,
        cache_dir=args.reference_cache_dir,
        methods="random_voronoi,kmeanspp,spectral",
        skip_joint=False,
    )
    subjepa_resolved = _resolve_side(
        model_family="subjepa",
        checkpoint=DEFAULT_SUBJEPA_CHECKPOINT,
        work_root=str(args.matrix_root.relative_to(PROJECT_ROOT)),
        cache_dir=str(
            Path("/data/sicong/weitao/.stable_worldmodel/subjepa/tworoom")
        ),
        methods=os.environ.get("METHODS", "kmeanspp,spectral"),
        skip_joint=os.environ.get("SKIP_JOINT", "1") == "1",
    )

    starts_path = args.formal_cache_root / "train_global_reference_starts.npy"
    reference_bundle = build_protocol_bundle(
        resolved=lewm_resolved,
        cache_root=args.formal_cache_root,
        starts_path=starts_path,
        task_spec=task_spec,
        paired_start_hashes=lewm_paired,
    )
    candidate_bundle = build_protocol_bundle(
        resolved=subjepa_resolved,
        cache_root=args.formal_cache_root,
        starts_path=starts_path,
        task_spec=task_spec,
        paired_start_hashes=lewm_paired,
    )

    PROTOCOL_PATHS = (
        "context_target_construction.history_size",
        "context_target_construction.num_preds",
        "context_target_construction.frameskip",
        "context_target_construction.sequence_length",
        "context_target_construction.img_size",
        "dataset_transition_scope.regions",
        "dataset_transition_scope.restrict_to_train_split",
        "dataset_transition_scope.predictor_prefix",
        "dataset_transition_scope.transition_batch_size",
        "dataset_transition_scope.frame_batch_size",
        "dataset_transition_scope.encoding",
        "dataset_transition_scope.num_all_starts",
        "dataset_transition_scope.train_fraction",
        "dataset_transition_scope.split_seed",
        "dataset_transition_scope.max_starts",
        "dataset_transition_scope.num_train_global_reference_starts",
        "dataset_transition_scope.train_global_reference_starts_sha256",
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "training.weight_decay",
        "training.min_region_samples",
        "training.history_size",
        "training.num_preds",
        "training.precision",
        "training.checkpoint_selection_rule",
        "training.predictor_loss",
        "training.trainable_parameter_allowlist",
        "training.frozen_parameter_prefixes",
        "training.action_embedding_usage",
        "partition_config.num_clusters",
        "partition_config.partition_seeds",
        "partition_config.methods",
        "partition_config.frameskip",
        "matrix_scope.train_seeds",
        "matrix_scope.eval_seeds",
        "matrix_scope.skip_joint",
        "evaluation.short_goal_offset",
        "evaluation.long_goal_offset",
        "evaluation.eval_budget",
        "evaluation.num_eval",
        "evaluation.plan_horizon",
        "evaluation.plan_receding_horizon",
        "evaluation.plan_action_block",
        "evaluation.evaluator",
        "evaluation.latent_routing",
        "evaluation.paired_start_source_short",
        "evaluation.paired_start_source_long",
        "bootstrap_aggregation.reps",
        "bootstrap_aggregation.unit",
        "bootstrap_aggregation.ci",
    )

    def _get_path(obj: dict[str, Any], dotted: str) -> Any:
        cur: Any = obj
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    comparisons: list[dict[str, Any]] = []
    mismatches: list[str] = []
    allowed: list[str] = []
    for field in PROTOCOL_PATHS:
        reference = _get_path(reference_bundle, field)
        candidate = _get_path(candidate_bundle, field)
        row = _compare_field(field, reference, candidate)
        comparisons.append(row)
        if row["status"] == "mismatch":
            mismatches.append(field)
        elif row["status"] == "allowed_diff":
            allowed.append(field)

    allowed_rows = [
        _compare_field(
            "resolved_manifest.model_family",
            lewm_resolved["model_family"],
            subjepa_resolved["model_family"],
        ),
        _compare_field(
            "resolved_manifest.checkpoint",
            lewm_resolved["checkpoint"],
            subjepa_resolved["checkpoint"],
        ),
        _compare_field(
            "dataset_transition_scope.cache_sha256",
            reference_bundle["dataset_transition_scope"].get("cache_sha256"),
            candidate_bundle["dataset_transition_scope"].get("cache_sha256"),
        ),
        _compare_field(
            "dataset_transition_scope.latent_shape",
            reference_bundle["dataset_transition_scope"].get("latent_shape"),
            candidate_bundle["dataset_transition_scope"].get("latent_shape"),
        ),
        _compare_field(
            "resolved_manifest.work_root",
            lewm_resolved["work_root"],
            subjepa_resolved["work_root"],
        ),
        _compare_field(
            "resolved_manifest.cache_dir",
            lewm_resolved["cache_dir"],
            subjepa_resolved["cache_dir"],
        ),
    ]
    for row in allowed_rows:
        comparisons.append(row)
        if row["status"] == "allowed_diff":
            allowed.append(row["field"])
        elif row["status"] == "mismatch":
            mismatches.append(row["field"])

    # Verify matrix paired-start copies match LeWM baselines when present.
    for horizon, suffix in (("short", "baseline"), ("long", "baseline_exp6")):
        for seed in eval_seeds:
            src = (
                args.lewm_results_root
                / f"tworoom_success_rate_{suffix}_seed{seed}"
                / "results.json"
            )
            dst = (
                args.matrix_root
                / "paired_starts"
                / f"lewm_{horizon}"
                / "eval"
                / "official"
                / f"eval{seed}"
                / "results.json"
            )
            if not dst.is_file():
                mismatches.append(f"paired_start_copy_missing.{horizon}.eval{seed}")
                continue
            src_starts = json.loads(src.read_text(encoding="utf-8"))["eval_start_indices"]
            dst_starts = json.loads(dst.read_text(encoding="utf-8"))["eval_start_indices"]
            if src_starts != dst_starts:
                mismatches.append(f"paired_start_ids.{horizon}.eval{seed}")

    parity_status = "PASSED" if not mismatches else "FAILED"
    payload = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_matrix_protocol_parity",
        "reference": {
            "label": "canonical_lewm_tworoom_matrix",
            "resolved_manifest_source": "resolve_jepa_matrix_config.py",
            "canonical_entrypoint": (
                "experiments/tworoom/scripts/canonical/run_tworoom_main_matrix.sh"
            ),
            "bundle": reference_bundle,
        },
        "candidate": {
            "label": "subjepa_tworoom_matrix",
            "bundle": candidate_bundle,
        },
        "allowed_difference_fields": sorted(ALLOWED_DIFF_FIELDS),
        "parity_status": parity_status,
        "comparisons": comparisons,
        "mismatches": mismatches,
        "allowed_differences_observed": sorted(set(allowed)),
        "formal_gate_scope_note": (
            "Formal gate artifacts supply only gate decision, verified spectral "
            "partitions, deployment seed, and router metadata. They must not "
            "override training or evaluation protocol fields checked here."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "parity_status": parity_status,
                "mismatches": mismatches,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    if parity_status != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
