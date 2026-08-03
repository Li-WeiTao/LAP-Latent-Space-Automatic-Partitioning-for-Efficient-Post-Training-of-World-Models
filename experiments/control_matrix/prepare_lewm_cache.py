#!/usr/bin/env python3
"""Prepare the five LeWM matrix cache artifacts consumed by ``run_lewm_matrix.sh``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWOROOM_DIR = PROJECT_ROOT / "experiments" / "tworoom"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TWOROOM_DIR))

from backends.lewm.encoding import (  # noqa: E402
    LeWMEncoderAdapter,
    _action_stats,
    _read_action_blocks,
    make_hdf5_transition_dataset,
)
from experiments.control_matrix.fit_partition import load_unique_latents  # noqa: E402
from experiments.control_matrix.region_risk_lib import (  # noqa: E402
    atomic_write_json,
    git_commit,
    sha256_file,
)
from experiments.control_matrix.backend_registry import (  # noqa: E402
    DEFAULT_MODEL_FAMILY,
    backend_metadata,
    normalize_model_family,
)
from lap.encoding.fast import FastEncodingConfig, FastLatentCacheEncoder  # noqa: E402


ARTIFACT_NAMES = (
    "embedding_cache.npz",
    "spectral_embedding_cache.npz",
    "representation_manifest.json",
    "action_norm_stats.npz",
    "action_norm_manifest.json",
)


@dataclass(frozen=True)
class PreparePaths:
    out_dir: Path
    starts: Path
    embedding_cache: Path
    spectral_cache: Path
    representation_manifest: Path
    action_norm_stats: Path
    action_norm_manifest: Path
    legacy_latent_cache: Path | None

    @classmethod
    def from_out_dir(
        cls, out_dir: Path, *, dataset_name: str | None = None
    ) -> "PreparePaths":
        out_dir = out_dir.resolve()
        legacy = (
            out_dir.parent / f"{dataset_name}_lewm_train_latent_cache.npz"
            if dataset_name
            else None
        )
        return cls(
            out_dir=out_dir,
            starts=out_dir / "train_global_reference_starts.npy",
            embedding_cache=out_dir / "embedding_cache.npz",
            spectral_cache=out_dir / "spectral_embedding_cache.npz",
            representation_manifest=out_dir / "representation_manifest.json",
            action_norm_stats=out_dir / "action_norm_stats.npz",
            action_norm_manifest=out_dir / "action_norm_manifest.json",
            legacy_latent_cache=legacy,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--regions", nargs="+", default=["common"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transition-batch-size", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Optional historical cache for exact array validation after encode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-starts",
        action="store_true",
        help="Assume train_global_reference_starts.npy already exists.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for trajectory.py start bootstrap.",
    )
    return parser.parse_args()


def all_artifacts_present(paths: PreparePaths) -> bool:
    required = (
        paths.starts,
        paths.embedding_cache,
        paths.spectral_cache,
        paths.representation_manifest,
        paths.action_norm_stats,
        paths.action_norm_manifest,
    )
    return all(path.exists() for path in required)


def run_prepare_starts(args: argparse.Namespace, paths: PreparePaths) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(TWOROOM_DIR / "trajectory.py"),
        "--dataset",
        args.dataset_name,
        "--data-file",
        str(args.data_file.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--out-dir",
        str(paths.out_dir),
        "--prepare-starts-only",
        "--regions",
        *args.regions,
        "--restrict-to-train-split",
        "--predictor-prefix",
        "train_",
        "--frameskip",
        str(args.frameskip),
        "--history-size",
        str(args.history_size),
        "--num-preds",
        str(args.num_preds),
        "--model-family",
        args.model_family,
    ]
    if args.max_starts > 0:
        command.extend(["--max-starts", str(args.max_starts)])
    subprocess.run(command, check=True)
    if not paths.starts.exists():
        raise FileExistsError(f"missing starts after bootstrap: {paths.starts}")


def load_starts(paths: PreparePaths, *, max_starts: int) -> np.ndarray:
    starts = np.asarray(np.load(paths.starts), dtype=np.int64)
    if max_starts > 0:
        starts = starts[:max_starts]
    if starts.ndim != 1 or len(starts) == 0:
        raise ValueError("starts must be a non-empty one-dimensional array")
    return starts


def compute_action_norm_stats(
    data_file: Path,
    starts: np.ndarray,
    starts_source: Path,
    *,
    history_size: int,
    num_preds: int,
    frameskip: int,
    action_key: str = "action",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action_steps = history_size + num_preds - 1
    with h5py.File(data_file, "r", swmr=True) as handle:
        if action_key not in handle:
            raise KeyError(f"dataset is missing action key {action_key!r}")
        actions = np.asarray(handle[action_key][:], dtype=np.float32)
    blocks = _read_action_blocks(actions, starts, action_steps, frameskip)
    mean, std = _action_stats(blocks)
    metadata = {
        "action_key": action_key,
        "history_size": history_size,
        "num_preds": num_preds,
        "frameskip": frameskip,
        "action_steps": action_steps,
        "normalization_samples": int(len(starts)),
        "starts_source": str(starts_source.resolve()),
    }
    return mean.astype(np.float32), std.astype(np.float32), metadata


def write_action_norm_artifacts(
    paths: PreparePaths,
    starts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    data_file: Path,
    checkpoint: Path,
) -> None:
    np.savez_compressed(
        paths.action_norm_stats,
        action_mean=mean,
        action_std=std,
        frameskip=np.int64(metadata["frameskip"]),
        normalization_samples=np.int64(metadata["normalization_samples"]),
    )
    manifest = {
        "schema_version": 1,
        "dataset_name": metadata.get("dataset_name"),
        "data_file": str(data_file.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "starts_file": str(paths.starts.resolve()),
        "starts_sha256": sha256_file(paths.starts),
        "num_starts": int(len(starts)),
        "action_mean_shape": list(mean.shape),
        "action_std_shape": list(std.shape),
        "frameskip": metadata["frameskip"],
        "history_size": metadata["history_size"],
        "num_preds": metadata["num_preds"],
        "action_steps": metadata["action_steps"],
        "normalization_samples": metadata["normalization_samples"],
        "sha256": {
            "action_norm_stats": None,
            "starts": sha256_file(paths.starts),
            "data_file": sha256_file(data_file),
            "checkpoint": sha256_file(checkpoint),
        },
    }
    atomic_write_json(paths.action_norm_manifest, manifest)
    manifest["sha256"]["action_norm_stats"] = sha256_file(paths.action_norm_stats)
    atomic_write_json(paths.action_norm_manifest, manifest)


def encode_embedding_cache(
    args: argparse.Namespace,
    paths: PreparePaths,
    starts: np.ndarray,
) -> dict[str, Any]:
    report_path = paths.embedding_cache.with_suffix(".npz.report.json")
    if paths.embedding_cache.exists() and not args.overwrite:
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"existing cache without report refuses overwrite: {paths.embedding_cache}"
        )
    if paths.embedding_cache.exists() and args.overwrite:
        for path in (paths.embedding_cache, report_path):
            if path.exists():
                path.unlink()

    starts_path = paths.out_dir / ".encode_starts.npy"
    np.save(starts_path, starts)
    dataset = make_hdf5_transition_dataset(
        data_file=str(args.data_file.resolve()),
        starts=str(starts_path),
        action_norm_starts=str(paths.starts.resolve()),
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    encoder = LeWMEncoderAdapter(
        img_size=args.img_size,
        frameskip=args.frameskip,
        model_family=args.model_family,
    )
    config = FastEncodingConfig(
        device=args.device,
        transition_batch_size=args.transition_batch_size,
        frame_batch_size=args.frame_batch_size,
        exact_batch_shapes=True,
        num_workers=args.num_workers,
        cpu_threads=args.cpu_threads,
    )
    return FastLatentCacheEncoder(config).encode(
        dataset=dataset,
        encoder=encoder,
        pretrained_model=str(args.checkpoint.resolve()),
        output=paths.embedding_cache,
        reference_cache=args.reference_cache,
    )


def build_spectral_embedding_cache(
    paths: PreparePaths,
    *,
    frameskip: int,
    overwrite: bool,
) -> dict[str, Any]:
    if paths.spectral_cache.exists() and not overwrite:
        with np.load(paths.spectral_cache, allow_pickle=False) as data:
            return {
                "num_unique_timesteps": int(len(data["emb"])),
                "latent_dim": int(data["emb"].shape[1]),
            }

    unique, sample_ids, cache_stats = load_unique_latents(
        paths.embedding_cache.resolve(strict=True), frameskip
    )
    np.savez_compressed(
        paths.spectral_cache,
        emb=unique.astype(np.float32, copy=False),
        global_timestep_ids=sample_ids.astype(np.int64, copy=False),
    )
    return cache_stats


def audit_against_reference(
    generated: Path,
    reference: Path,
) -> dict[str, Any]:
    with np.load(generated, allow_pickle=False) as gen, np.load(
        reference, allow_pickle=False
    ) as ref:
        keys = ("emb", "act_emb", "region_starts")
        missing = [key for key in keys if key not in gen.files or key not in ref.files]
        if missing:
            raise KeyError(f"reference audit missing keys: {missing}")
        report: dict[str, Any] = {"passed": True, "arrays": {}}
        for key in keys:
            actual = np.asarray(gen[key])
            expected = np.asarray(ref[key])
            exact = actual.shape == expected.shape and np.array_equal(actual, expected)
            semantic = exact
            if not exact and key == "emb":
                semantic = actual.shape == expected.shape and np.allclose(
                    actual, expected, rtol=1e-5, atol=1e-6
                )
            report["arrays"][key] = {
                "shape_equal": actual.shape == expected.shape,
                "exact_equal": exact,
                "semantic_equal": semantic,
                "generated_shape": list(actual.shape),
                "reference_shape": list(expected.shape),
            }
            report["passed"] = report["passed"] and semantic
        return report


def write_representation_manifest(
    paths: PreparePaths,
    *,
    args: argparse.Namespace,
    encode_report: Mapping[str, Any],
    spectral_stats: Mapping[str, Any],
    reference_audit: Mapping[str, Any] | None,
) -> None:
    manifest = {
        "schema_version": 2,
        **backend_metadata(normalize_model_family(getattr(args, "model_family", None))),
        "dataset_name": args.dataset_name,
        "data_file": str(args.data_file.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "starts_file": str(paths.starts.resolve()),
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "encode_report": {
            "method": encode_report.get("method"),
            "selection": encode_report.get("selection"),
            "arrays": encode_report.get("arrays"),
        },
        "spectral_cache_stats": dict(spectral_stats),
        "reference_audit": reference_audit,
        "git_commit": git_commit(),
        "artifacts": {
            name: str((paths.out_dir / name).resolve())
            for name in ARTIFACT_NAMES
        },
        "sha256": {
            "data_file": sha256_file(args.data_file),
            "checkpoint": sha256_file(args.checkpoint),
            "starts": sha256_file(paths.starts),
            "embedding_cache": sha256_file(paths.embedding_cache),
            "spectral_embedding_cache": sha256_file(paths.spectral_cache),
            "action_norm_stats": sha256_file(paths.action_norm_stats),
        },
    }
    atomic_write_json(paths.representation_manifest, manifest)


def link_legacy_cache(paths: PreparePaths) -> None:
    if paths.legacy_latent_cache is None:
        return
    if paths.legacy_latent_cache.exists() or paths.legacy_latent_cache.is_symlink():
        if paths.legacy_latent_cache.resolve() != paths.embedding_cache.resolve():
            paths.legacy_latent_cache.unlink()
        else:
            return
    os.symlink(paths.embedding_cache.name, paths.legacy_latent_cache)


def main() -> None:
    args = parse_args()
    paths = PreparePaths.from_out_dir(args.out_dir, dataset_name=args.dataset_name)
    if all_artifacts_present(paths) and not args.overwrite:
        print(f"[prepare] all artifacts present under {paths.out_dir}; skipping")
        link_legacy_cache(paths)
        return

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_starts and (
        not paths.starts.exists() or args.overwrite
    ):
        run_prepare_starts(args, paths)

    starts = load_starts(paths, max_starts=0)
    encode_starts = starts if args.max_starts <= 0 else starts[: args.max_starts]

    mean, std, norm_meta = compute_action_norm_stats(
        args.data_file.resolve(),
        starts,
        paths.starts,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
    )
    norm_meta["dataset_name"] = args.dataset_name
    if paths.action_norm_stats.exists() and args.overwrite:
        paths.action_norm_stats.unlink()
    if paths.action_norm_manifest.exists() and args.overwrite:
        paths.action_norm_manifest.unlink()
    write_action_norm_artifacts(
        paths,
        starts,
        mean,
        std,
        norm_meta,
        data_file=args.data_file,
        checkpoint=args.checkpoint,
    )

    encode_report = encode_embedding_cache(args, paths, encode_starts)
    spectral_stats = build_spectral_embedding_cache(
        paths, frameskip=args.frameskip, overwrite=args.overwrite
    )

    reference_audit = None
    if args.reference_cache is not None:
        reference_audit = audit_against_reference(
            paths.embedding_cache, args.reference_cache.resolve(strict=True)
        )
        if not reference_audit["passed"]:
            raise RuntimeError(
                "generated embedding cache failed reference audit: "
                f"{reference_audit}"
            )

    if paths.representation_manifest.exists() and args.overwrite:
        paths.representation_manifest.unlink()
    write_representation_manifest(
        paths,
        args=args,
        encode_report=encode_report,
        spectral_stats=spectral_stats,
        reference_audit=reference_audit,
    )
    link_legacy_cache(paths)
    print(f"[prepare] wrote matrix cache artifacts under {paths.out_dir}")


if __name__ == "__main__":
    main()
