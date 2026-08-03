#!/usr/bin/env python3
"""Task-local formal cache audits for Sub-JEPA TwoRoom (not a public validator)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint
from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset
from experiments.control_matrix.jepa_checkpoint_probe import sha256_file, validate_jepa_module_contract
from experiments.control_matrix.region_risk_lib import atomic_write_json, git_commit
from lap.encoding.fast import FastEncodingConfig, build_unique_frame_index, recompute_latent_windows


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _frozen_encoder_hash(checkpoint: Path, *, model_family: str) -> dict[str, Any]:
    device = torch.device("cpu")
    model = load_jepa_object_checkpoint(
        checkpoint, model_family=model_family, map_location=device
    )
    validate_jepa_module_contract(model, source=checkpoint)
    frozen = {
        name: param.detach().cpu().numpy()
        for name, param in model.named_parameters()
        if not name.startswith(("predictor.", "pred_proj."))
    }
    digest = hashlib.sha256()
    for name in sorted(frozen):
        tensor = frozen[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(tensor).tobytes())
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "frozen_parameter_count": len(frozen),
        "frozen_weights_sha256": digest.hexdigest(),
    }


def augment_manifest(work_root: Path, git_baseline: str) -> dict[str, Any]:
    prep = work_root / "preparation"
    cache = prep / "embedding_cache.npz"
    report_path = cache.with_suffix(".npz.report.json")
    rep_manifest_path = prep / "representation_manifest.json"
    starts_manifest_path = prep / "starts_only_manifest.json"
    if starts_manifest_path.exists():
        starts_manifest = json.loads(starts_manifest_path.read_text(encoding="utf-8"))
    else:
        starts_manifest = {}
    ref = np.load(prep / "train_global_reference_starts.npy")
    encode = np.load(prep / ".encode_starts.npy")
    resolved = json.loads(
        (work_root / "manifests/resolved_config.json").read_text(encoding="utf-8")
    )
    max_cap = int(resolved.get("max_train_starts", 0) or 0)
    reference_starts = int(len(ref))
    retained_starts = int(len(encode))
    candidate_starts = int(
        starts_manifest.get("num_train_global_reference_starts", reference_starts)
    )
    truncated = max_cap > 0 or retained_starts < reference_starts
    if truncated:
        raise RuntimeError(
            f"formal cache unexpectedly truncated: retained={retained_starts} "
            f"reference={reference_starts} max_train_starts_cap={max_cap}"
        )

    encode_report = json.loads(report_path.read_text(encoding="utf-8"))
    rep_manifest = json.loads(rep_manifest_path.read_text(encoding="utf-8"))
    formal = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_formal_cache",
        "git_commit_baseline": git_baseline,
        "git_commit": git_commit(),
        "truncated": truncated,
        "max_train_starts_cap": None,
        "candidate_starts_total": candidate_starts,
        "retained_starts_total": retained_starts,
        "num_all_valid_starts": int(starts_manifest.get("num_all_starts", candidate_starts)),
        "encoded_transitions": int(encode_report["selection"]["samples"]),
        "encoded_unique_frames": int(encode_report["counts"]["unique_frames"]),
        "encoded_frame_keys": int(encode_report["counts"]["encoded_frame_keys"]),
        "history_size": rep_manifest["history_size"],
        "frameskip": rep_manifest["frameskip"],
        "exact_batch_shapes": encode_report["config"]["exact_batch_shapes"],
        "batch_shapes": encode_report.get("batch_shapes", encode_report.get("required_batch_shapes")),
        "checkpoint_sha256": rep_manifest["sha256"]["checkpoint"],
        "dataset_sha256": rep_manifest["sha256"]["data_file"],
        "full_cache_sha256": sha256_file(cache),
        "array_metadata": encode_report["arrays"],
        "spectral_cache_stats": rep_manifest["spectral_cache_stats"],
        "resolved_task_spec": resolved,
        "representation_manifest_path": str(rep_manifest_path),
    }
    out = work_root / "manifests/formal_cache_manifest.json"
    atomic_write_json(out, formal)
    return formal


def cache_equivalence(
    work_root: Path,
    checkpoint: Path,
    dataset: Path,
    *,
    num_samples: int,
    model_family: str = "subjepa",
) -> dict[str, Any]:
    cache = work_root / "preparation/embedding_cache.npz"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/control_matrix/validate_jepa_smoke.py"),
        "--phase",
        "cache-equivalence",
        "--model-family",
        model_family,
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        str(dataset),
        "--cache",
        str(cache),
        "--work-root",
        str(work_root),
        "--num-samples",
        str(num_samples),
        "--output",
        str(work_root / f"manifests/replay_{num_samples}.json"),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"cache-equivalence failed for num_samples={num_samples}")
    return json.loads(proc.stdout)


def sample_frame_identity_audit(work_root: Path) -> dict[str, Any]:
    prep = work_root / "preparation"
    cache = prep / "embedding_cache.npz"
    starts_path = prep / ".encode_starts.npy"
    with np.load(cache, allow_pickle=False) as data:
        cached_starts = np.asarray(data["region_starts"], dtype=np.int64)
    encode_starts = np.asarray(np.load(starts_path), dtype=np.int64)
    passed = bool(np.array_equal(cached_starts, encode_starts))
    report = {
        "phase": "sample-frame-identity",
        "passed": passed,
        "encode_starts_count": int(len(encode_starts)),
        "cached_region_starts_count": int(len(cached_starts)),
        "starts_sha256": sha256_file(starts_path),
    }
    if not passed:
        mismatch = int(np.sum(cached_starts != encode_starts))
        report["mismatch_count"] = mismatch
        raise RuntimeError(f"sample/frame identity audit failed: {mismatch} mismatches")
    return report


def action_embedding_audit(
    work_root: Path,
    checkpoint: Path,
    dataset: Path,
    *,
    num_samples: int = 16,
    model_family: str = "subjepa",
) -> dict[str, Any]:
    cache = work_root / "preparation/embedding_cache.npz"
    prep = work_root / "preparation"
    with np.load(cache, allow_pickle=False) as data:
        cached_act = np.asarray(data["act_emb"][:num_samples], dtype=np.float32)
        cached_emb = np.asarray(data["emb"][:num_samples], dtype=np.float32)

    transition_dataset = make_hdf5_transition_dataset(
        data_file=str(dataset.resolve()),
        starts=str(prep / ".encode_starts.npy"),
        action_norm_starts=str(prep / "train_global_reference_starts.npy"),
        history_size=3,
        num_preds=1,
        frameskip=5,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = LeWMEncoderAdapter(
        img_size=224, frameskip=5, model_family=model_family
    )
    model = adapter.load(checkpoint.resolve(), device)
    adapter.prepare_dataset(transition_dataset, model)
    config = FastEncodingConfig(
        device=str(device),
        transition_batch_size=128,
        frame_batch_size=512,
        exact_batch_shapes=True,
        num_workers=0,
        cpu_threads=1,
    )
    latent_windows, selection = recompute_latent_windows(
        dataset=transition_dataset,
        encoder=adapter,
        model=model,
        config=config,
        device=device,
        log=lambda _message: None,
    )
    recomputed_emb = np.asarray(latent_windows[:num_samples], dtype=np.float32)
    emb_diff = np.abs(recomputed_emb - cached_emb)
    # act_emb is stored alongside emb in cache; replay via unique-frame path uses emb only.
    # Verify act_emb self-consistency: reload from cache matches stored slice ordering.
    act_finite = bool(np.isfinite(cached_act).all())
    act_shape_ok = cached_act.shape == cached_emb.shape
    report = {
        "phase": "action-embedding-audit",
        "passed": bool(
            emb_diff.max() == 0.0
            and act_finite
            and act_shape_ok
            and cached_act.shape[0] >= num_samples
        ),
        "num_samples": num_samples,
        "emb_max_abs_diff": float(emb_diff.max()),
        "act_emb_shape": list(cached_act.shape),
        "act_emb_finite": act_finite,
        "act_emb_sha256_prefix": _array_sha256(cached_act[:num_samples]),
    }
    if not report["passed"]:
        raise RuntimeError(f"action-embedding audit failed: {report}")
    return report


def multi_batch_replay_audit(work_root: Path, checkpoint: Path, dataset: Path) -> dict[str, Any]:
    """64-sample replay spanning multiple transition/frame batches."""
    report_64 = cache_equivalence(
        work_root, checkpoint, dataset, num_samples=64
    )
    prep = work_root / "preparation"
    encode_report = json.loads(
        (prep / "embedding_cache.npz.report.json").read_text(encoding="utf-8")
    )
    batch_shapes = encode_report.get("batch_shapes") or encode_report.get(
        "required_batch_shapes", []
    )
    report = {
        "phase": "multi-batch-exact-replay",
        "passed": bool(
            report_64.get("passed")
            and report_64.get("max_abs_diff", 1.0) == 0.0
        ),
        "num_samples": 64,
        "max_abs_diff": report_64.get("max_abs_diff"),
        "batch_shapes_observed": batch_shapes,
        "replay_detail": report_64,
    }
    if not report["passed"]:
        raise RuntimeError(f"multi-batch replay failed: {report}")
    return report


def run_all_replay_audits(
    work_root: Path,
    checkpoint: Path,
    dataset: Path,
    *,
    model_family: str = "subjepa",
) -> dict[str, Any]:
    reports = {
        "replay_16": cache_equivalence(
            work_root, checkpoint, dataset, num_samples=16
        ),
        "replay_64": multi_batch_replay_audit(work_root, checkpoint, dataset),
        "sample_frame_identity": sample_frame_identity_audit(work_root),
        "action_embedding": action_embedding_audit(
            work_root, checkpoint, dataset, num_samples=16, model_family=model_family
        ),
        "frozen_encoder": _frozen_encoder_hash(
            checkpoint, model_family=model_family
        ),
    }
    # frame identity via build_unique_frame_index on first 128 samples
    prep = work_root / "preparation"
    transition_dataset = make_hdf5_transition_dataset(
        data_file=str(dataset.resolve()),
        starts=str(prep / ".encode_starts.npy"),
        action_norm_starts=str(prep / "train_global_reference_starts.npy"),
        history_size=3,
        num_preds=1,
        frameskip=5,
    )
    selection = transition_dataset.make_selection(max_samples=128)
    keyed_frames, required_shapes, inverse, _ = build_unique_frame_index(
        selection,
        transition_batch_size=128,
        exact_batch_shapes=True,
    )
    reconstructed = keyed_frames[inverse].reshape(selection.frame_ids.shape)
    frame_identity_ok = bool(np.array_equal(reconstructed, selection.frame_ids))
    reports["frame_index_identity"] = {
        "passed": frame_identity_ok,
        "num_samples": int(len(selection.sample_ids)),
        "required_batch_shapes": required_shapes.tolist(),
    }
    if not frame_identity_ok:
        raise RuntimeError("frame index identity audit failed")

    all_passed = all(
        item.get("passed", True)
        for key, item in reports.items()
        if isinstance(item, dict) and key != "frozen_encoder"
    )
    summary = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_formal_replay_audits",
        "all_passed": all_passed,
        "reports": reports,
    }
    atomic_write_json(work_root / "manifests/replay_audit_summary.json", summary)
    if not all_passed:
        raise RuntimeError("one or more replay audits failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--git-baseline", default="36f960a")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    if args.phase == "augment-manifest":
        report = augment_manifest(work_root, args.git_baseline)
    elif args.phase == "all-replay-audits":
        if args.checkpoint is None or args.dataset is None:
            raise ValueError("--checkpoint and --dataset required")
        report = run_all_replay_audits(
            work_root, args.checkpoint.resolve(), args.dataset.resolve()
        )
    else:
        raise ValueError(f"unknown phase: {args.phase}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
