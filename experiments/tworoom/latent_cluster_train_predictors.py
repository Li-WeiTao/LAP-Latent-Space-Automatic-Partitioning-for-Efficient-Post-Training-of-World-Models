#!/usr/bin/env python3
"""Fine-tune one predictor per latent cluster (50 epochs, best-by-eval).

Uses cluster labels at transition start global_idx and reuses cached embeddings.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import load_encoder  # noqa: E402
from latent_cluster_common import (  # noqa: E402
    CLUSTER_NAMES,
    cluster_labels_for_starts,
    default_embed_dir,
    resolve_cluster_source,
)
from trajectory import (  # noqa: E402
    GLOBAL_FT_EMBED_REGIONS,
    TrainConfig,
    embedding_cache_path,
    json_ready,
    load_embedding_cache,
    load_global_train_embeddings_from_region_caches,
    save_region_predictor,
    set_training_seed,
    starts_match_cached,
    train_region_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-artifact-dir",
        type=Path,
        default=None,
        help="Directory with centroids.npy, cluster_labels.npz, cluster_meta.json",
    )
    parser.add_argument(
        "--kmeanspp-label-npz",
        type=Path,
        default=None,
        help="K-means++ partition npz from latent_kmeanspp_multirestart labels/",
    )
    parser.add_argument(
        "--zscore-params",
        type=Path,
        default=None,
        help="zscore_params.npz (required for kmeanspp inference routing; optional for FT)",
    )
    parser.add_argument(
        "--embedding-source-dir",
        type=Path,
        default=default_embed_dir(),
        help="P_train_{region}_embeddings.npz source for full train merge",
    )
    parser.add_argument(
        "--train-starts",
        type=Path,
        default=default_embed_dir() / "train_global_reference_starts.npy",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to results/tworoom_latent_cluster3_<artifact_name>",
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--min-cluster-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Predictor FT shuffle seed")
    parser.add_argument(
        "--route-label-offset-steps",
        type=int,
        default=None,
        help=(
            "Global-timestep offset used to label each transition. Defaults to "
            "artifact metadata transition_label_offset_steps, otherwise 0 for "
            "legacy artifacts."
        ),
    )
    parser.add_argument(
        "--only-clusters",
        type=str,
        default=None,
        help="Comma-separated cluster ids to train (e.g. 2). Skips others; merges manifest.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifact coverage and print cluster counts without loading/training a model.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Allow a complete (non --only-clusters) job to replace an existing "
            "predictor set. Without this explicit flag, existing outputs fail closed."
        ),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MANIFEST_SCHEMA_VERSION = 2


def file_provenance(path: Path) -> dict:
    """Return content-addressed provenance for an input or output file."""
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def source_cache_lineage(path: Path) -> dict:
    """Record cheap source-cache lineage; the merged cache is content hashed."""
    resolved = Path(path).expanduser().resolve()
    result = {"path": str(resolved), "exists": resolved.is_file()}
    if result["exists"]:
        stat = resolved.stat()
        result.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return result


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        raise RuntimeError(f"Manifest exists but is empty: {path}")
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"Manifest root must be an object: {path}")
    return value


@contextmanager
def manifest_file_lock(manifest_path: Path, timeout_sec: float = 120.0):
    """Cross-platform advisory lock shared by cache and manifest writers."""
    lock_path = manifest_path.with_name(f"{manifest_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()

    deadline = time.monotonic() + timeout_sec
    locked = False
    try:
        while not locked:
            try:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring manifest lock: {lock_path}")
                time.sleep(0.1)
        yield
    finally:
        if locked:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def atomic_write_json(path: Path, value: dict) -> None:
    """Durably replace a JSON file without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(json_ready(value), f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                try:
                    os.fsync(dir_fd)
                except OSError as exc:
                    # Some NFS configurations reject directory fsync even
                    # after the atomic rename has succeeded.  At this point
                    # the new manifest is already visible; raising would make
                    # the caller roll checkpoints back while leaving that new
                    # manifest in place.  Preserve consistency and report that
                    # crash-durability could not be strengthened on this FS.
                    warnings.warn(
                        f"Directory fsync unavailable for {path.parent}: {exc}",
                        RuntimeWarning,
                    )
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_manifest_compatibility(
    existing: dict,
    expected: dict,
    *,
    manifest_path: Path,
) -> None:
    """Reject recovery from a manifest produced by different immutable inputs."""
    if existing.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Cannot merge {manifest_path}: manifest schema is missing or incompatible; "
            "run a complete training job into a fresh output directory first"
        )

    expected_hash = expected["immutable_config_sha256"]
    existing_hash = existing.get("immutable_config_sha256")
    existing_config = existing.get("immutable_config")
    expected_config = expected["immutable_config"]
    if existing_hash != expected_hash or existing_config != expected_config:
        existing_config = existing_config if isinstance(existing_config, dict) else {}
        keys = sorted(set(existing_config) | set(expected_config))
        changed = [
            key
            for key in keys
            if canonical_json_sha256(existing_config.get(key))
            != canonical_json_sha256(expected_config.get(key))
        ]
        raise RuntimeError(
            f"Refusing to mix predictor checkpoints in {manifest_path}: immutable "
            f"configuration differs in {changed or ['fingerprint']}"
        )


def validate_recovered_predictors(
    manifest: dict,
    *,
    ignore_clusters: set[str] | None = None,
) -> None:
    """Verify hashes for checkpoints that will be retained during recovery."""
    ignored = ignore_clusters or set()
    clusters = manifest.get("clusters", {})
    if not isinstance(clusters, dict):
        raise RuntimeError("Manifest clusters must be a JSON object")
    for name, entry in clusters.items():
        if name in ignored or not isinstance(entry, dict):
            continue
        if entry.get("status") != "trained":
            continue
        for path_key, hash_key in (
            ("output_ckpt", "output_ckpt_sha256"),
            ("output_metadata", "output_metadata_sha256"),
        ):
            output = entry.get(path_key)
            expected_hash = entry.get(hash_key)
            if not output or not expected_hash:
                raise RuntimeError(
                    f"Recovered {name} lacks content provenance ({path_key}/{hash_key})"
                )
            output_path = Path(output)
            if not output_path.is_file():
                raise RuntimeError(f"Recovered {name} output is missing: {output_path}")
            actual_hash = sha256_file(output_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Recovered {name} output hash mismatch for {output_path}"
                )


def load_recovery_manifest(
    manifest_path: Path,
    expected: dict,
    *,
    replacement_clusters: set[str],
) -> dict | None:
    """Read and validate a recovery manifest under the writer lock."""
    with manifest_file_lock(manifest_path):
        existing = read_manifest(manifest_path)
        if existing is not None:
            validate_manifest_compatibility(
                existing, expected, manifest_path=manifest_path
            )
            validate_recovered_predictors(
                existing, ignore_clusters=replacement_clusters
            )
        return existing


def write_manifest_with_cluster_merge(
    manifest_path: Path,
    base_manifest: dict,
    cluster_updates: dict,
    *,
    partial_recovery: bool,
) -> dict:
    """Atomically merge cluster results without losing concurrent partial jobs."""
    with manifest_file_lock(manifest_path):
        existing = read_manifest(manifest_path)
        retained: dict = {}
        if partial_recovery and existing is not None:
            validate_manifest_compatibility(
                existing, base_manifest, manifest_path=manifest_path
            )
            validate_recovered_predictors(
                existing, ignore_clusters=set(cluster_updates)
            )
            retained = dict(existing.get("clusters", {}))

        merged = copy.deepcopy(base_manifest)
        merged["clusters"] = retained
        merged["clusters"].update(cluster_updates)
        atomic_write_json(manifest_path, merged)
        return merged


def commit_staged_cluster_merge(
    manifest_path: Path,
    base_manifest: dict,
    cluster_updates: dict,
    staged_outputs: dict[str, dict[str, Path]],
    *,
    partial_recovery: bool,
    allow_full_overwrite: bool,
    staging_dir: Path,
) -> dict:
    """Promote checkpoints and the matching manifest under one writer lock.

    Training never touches canonical checkpoint paths.  This prevents two
    recovery jobs from invalidating the manifest while the other is still
    training.  Local exceptions during promotion roll the previous files back.
    """
    with manifest_file_lock(manifest_path):
        existing = read_manifest(manifest_path)
        retained: dict = {}
        if partial_recovery and existing is not None:
            validate_manifest_compatibility(
                existing, base_manifest, manifest_path=manifest_path
            )
            validate_recovered_predictors(
                existing, ignore_clusters=set(cluster_updates)
            )
            retained = dict(existing.get("clusters", {}))
        elif existing is not None:
            if not allow_full_overwrite:
                raise RuntimeError(
                    f"Complete predictor output already exists at {manifest_path}; "
                    "use --only-clusters for compatible recovery or explicitly pass "
                    "--overwrite-existing"
                )
            validate_manifest_compatibility(
                existing, base_manifest, manifest_path=manifest_path
            )

        committed_updates = copy.deepcopy(cluster_updates)
        promotions: list[tuple[Path, Path | None]] = []
        try:
            for name, staged in staged_outputs.items():
                final_ckpt = staged["final_ckpt"]
                final_metadata = staged["final_metadata"]
                for source, destination in (
                    (staged["staged_ckpt"], final_ckpt),
                    (staged["staged_metadata"], final_metadata),
                ):
                    backup: Path | None = None
                    if destination.exists():
                        backup = staging_dir / f"{destination.name}.previous"
                        os.replace(destination, backup)
                    promotions.append((destination, backup))
                    os.replace(source, destination)

                committed_updates[name].update(
                    {
                        "output_ckpt": str(final_ckpt.resolve(strict=True)),
                        "output_ckpt_sha256": sha256_file(final_ckpt),
                        "output_metadata": str(final_metadata.resolve(strict=True)),
                        "output_metadata_sha256": sha256_file(final_metadata),
                    }
                )

            merged = copy.deepcopy(base_manifest)
            merged["clusters"] = retained
            merged["clusters"].update(committed_updates)
            atomic_write_json(manifest_path, merged)
        except BaseException:
            for destination, backup in reversed(promotions):
                if destination.exists():
                    destination.unlink()
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
            raise
        else:
            for _destination, backup in promotions:
                if backup is not None and backup.exists():
                    backup.unlink()
            return merged


def main() -> None:
    args = parse_args()
    if (args.cluster_artifact_dir is None) == (args.kmeanspp_label_npz is None):
        raise SystemExit(
            "Exactly one cluster source is required: --cluster-artifact-dir XOR "
            "--kmeanspp-label-npz"
        )
    if args.kmeanspp_label_npz is not None and args.zscore_params is None:
        default_z = args.kmeanspp_label_npz.parent.parent / "zscore_params.npz"
        if default_z.exists():
            args.zscore_params = default_z
    if (
        args.kmeanspp_label_npz is not None
        and args.zscore_params is None
        and not args.dry_run
    ):
        raise SystemExit(
            "Deployable K-means++ predictor training requires the fitted "
            "zscore_params.npz so the manifest binds the exact inference transform"
        )

    if args.cluster_artifact_dir is not None:
        artifact_files = [
            args.cluster_artifact_dir / name
            for name in (
                "centroids.npy",
                "cluster_labels.npz",
                "routing_prototypes.npy",
                "prototype_cluster_ids.npy",
                "zscore_params.npz",
                "cluster_meta.json",
            )
            if (args.cluster_artifact_dir / name).exists()
        ]
    else:
        artifact_files = [args.kmeanspp_label_npz]
        if args.zscore_params is not None:
            artifact_files.append(args.zscore_params)
    artifact_snapshot_before = {
        str(path.expanduser().resolve(strict=True)): sha256_file(
            path.expanduser().resolve(strict=True)
        )
        for path in artifact_files
    }

    artifact = resolve_cluster_source(
        cluster_artifact_dir=args.cluster_artifact_dir,
        kmeanspp_label_npz=args.kmeanspp_label_npz,
        zscore_params_npz=args.zscore_params,
    )
    num_clusters = int(artifact["meta"].get("num_clusters", len(CLUSTER_NAMES)))
    cluster_names = tuple(f"cluster{k}" for k in range(num_clusters))

    only_clusters: set[int] | None = None
    if args.only_clusters is not None:
        only_clusters = {
            int(x.strip()) for x in args.only_clusters.split(",") if x.strip()
        }
        if not only_clusters:
            raise SystemExit("--only-clusters must contain at least one cluster id")
        invalid = sorted(k for k in only_clusters if k < 0 or k >= num_clusters)
        if invalid:
            raise SystemExit(
                f"--only-clusters contains ids outside [0,{num_clusters}): {invalid}"
            )

    if args.out_dir is None:
        if args.kmeanspp_label_npz is not None:
            args.out_dir = (
                THIS_DIR
                / "results"
                / f"tworoom_latent_kmeanspp_{args.kmeanspp_label_npz.stem}"
            )
        else:
            args.out_dir = (
                THIS_DIR
                / "results"
                / f"tworoom_latent_cluster{num_clusters}_{args.cluster_artifact_dir.name}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_starts = np.load(args.train_starts)
    route_label_offset_steps = args.route_label_offset_steps
    if route_label_offset_steps is None:
        route_label_offset_steps = int(
            artifact["meta"].get("transition_label_offset_steps", 0)
        )
    route_label_indices = train_starts + route_label_offset_steps
    cluster_ids = cluster_labels_for_starts(
        route_label_indices, artifact["lookup"]
    )
    artifact_provenance = {
        str(path.expanduser().resolve(strict=True)): sha256_file(
            path.expanduser().resolve(strict=True)
        )
        for path in artifact_files
    }
    if artifact_provenance != artifact_snapshot_before:
        raise RuntimeError(
            "Cluster artifact changed while its labels were being loaded; retry "
            "after the spectral artifact writer has finished"
        )
    print(
        f"[data] {len(train_starts)} train transitions, "
        f"cluster source={args.kmeanspp_label_npz or args.cluster_artifact_dir}, "
        f"route_label_offset_steps={route_label_offset_steps}",
        flush=True,
    )
    for k in range(num_clusters):
        print(f"  cluster{k}: {(cluster_ids == k).sum()} transitions", flush=True)

    if args.dry_run:
        print("[dry-run] artifact coverage and route-label assignment passed", flush=True)
        return

    set_training_seed(args.seed)

    cfg = TrainConfig(
        history_size=args.history_size,
        num_preds=args.num_preds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_region_samples=args.min_cluster_samples,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    manifest_path = args.out_dir / "manifest.json"
    # Fail before cache construction/model loading when a complete run would
    # overwrite an existing predictor set without explicit authorization.
    # The commit path repeats this check under the same lock to close the
    # time-of-check/time-of-use race.
    if only_clusters is None and not args.overwrite_existing:
        with manifest_file_lock(manifest_path):
            if read_manifest(manifest_path) is not None:
                raise FileExistsError(
                    f"Predictor output already has a manifest at {manifest_path}; "
                    "use --overwrite-existing only for an intentional same-config "
                    "replacement"
                )
    staging_dir = (
        args.out_dir
        / ".staging"
        / f"pid{os.getpid()}_{time.time_ns()}"
    )
    staging_dir.mkdir(parents=True, exist_ok=False)
    atexit.register(shutil.rmtree, staging_dir, ignore_errors=True)
    merged_cache = embedding_cache_path(args.out_dir, "global_merged", "train_")
    with manifest_file_lock(manifest_path):
        if merged_cache.exists():
            emb, act_emb, cached_starts = load_embedding_cache(merged_cache)
            if not starts_match_cached(cached_starts, train_starts):
                raise RuntimeError(f"Stale merged cache at {merged_cache}")
            cache_origin = "reused"
            print(f"  [cache] loaded merged embeddings from {merged_cache}", flush=True)
        else:
            emb, act_emb = load_global_train_embeddings_from_region_caches(
                args.embedding_source_dir,
                train_starts,
                name_prefix="train_",
            )
            from trajectory import save_embedding_cache  # local import avoids cycle

            tmp_cache = merged_cache.with_name(
                f".{merged_cache.stem}.{os.getpid()}.tmp.npz"
            )
            try:
                save_embedding_cache(tmp_cache, emb, act_emb, train_starts)
                os.replace(tmp_cache, merged_cache)
            finally:
                if tmp_cache.exists():
                    tmp_cache.unlink()
            cache_origin = "created_from_region_caches"
            print(f"  [cache] saved merged embeddings to {merged_cache}", flush=True)

    checkpoint_provenance = file_provenance(Path(args.checkpoint))
    train_starts_provenance = file_provenance(args.train_starts)
    merged_cache_provenance = file_provenance(merged_cache)
    source_cache_provenance = [
        source_cache_lineage(
            embedding_cache_path(args.embedding_source_dir, region, "train_")
        )
        for region in GLOBAL_FT_EMBED_REGIONS
    ]
    cluster_source = {
        "kind": "kmeanspp" if args.kmeanspp_label_npz is not None else "cluster_artifact",
        "path": str(
            (args.kmeanspp_label_npz or args.cluster_artifact_dir)
            .expanduser()
            .resolve(strict=True)
        ),
    }
    route_anchor = artifact["meta"].get(
        "route_anchor", "transition_start" if route_label_offset_steps == 0 else "custom"
    )
    assignment_schema_version = int(
        artifact["meta"].get("assignment_schema_version", 1)
    )
    immutable_config = {
        "dataset": "tworoom",
        "cluster_source": cluster_source,
        "cluster_artifact_sha256": artifact_provenance,
        "base_checkpoint": checkpoint_provenance,
        "train_starts": train_starts_provenance,
        "input_embedding_cache": merged_cache_provenance,
        "num_train_transitions": int(len(train_starts)),
        "num_clusters": int(num_clusters),
        "route_anchor": route_anchor,
        "route_label_offset_steps": int(route_label_offset_steps),
        "assignment_schema_version": assignment_schema_version,
        "training": {
            "history_size": args.history_size,
            "num_preds": args.num_preds,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "min_cluster_samples": args.min_cluster_samples,
            "seed": args.seed,
            "select_best_by_eval": True,
            "checkpoint_selection": {
                "criterion": "minimum_predictor_loss",
                "monitor_split": "same_cluster_training_embedding_pool",
                "is_held_out_validation": False,
            },
            "device": str(device),
        },
    }
    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "tworoom",
        "cluster_artifact_dir": (
            str(args.cluster_artifact_dir.expanduser().resolve())
            if args.cluster_artifact_dir
            else None
        ),
        "kmeanspp_label_npz": (
            str(args.kmeanspp_label_npz.expanduser().resolve())
            if args.kmeanspp_label_npz
            else None
        ),
        "zscore_params": (
            artifact["zscore"].get("path") if artifact.get("zscore") else None
        ),
        "cluster_meta": artifact["meta"],
        "cluster_artifact_sha256": artifact_provenance,
        "base_checkpoint": checkpoint_provenance,
        "train_starts": train_starts_provenance,
        "input_embedding_cache": {
            **merged_cache_provenance,
            "origin": cache_origin,
        },
        "embedding_source_dir": str(args.embedding_source_dir.expanduser().resolve()),
        "embedding_source_caches": source_cache_provenance,
        "num_train_transitions": int(len(train_starts)),
        "route_anchor": route_anchor,
        "route_label_offset_steps": int(route_label_offset_steps),
        "assignment_schema_version": assignment_schema_version,
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_cluster_samples": args.min_cluster_samples,
        "seed": args.seed,
        "select_best_by_eval": True,
        "checkpoint_selection": {
            "criterion": "minimum_predictor_loss",
            "monitor_split": "same_cluster_training_embedding_pool",
            "is_held_out_validation": False,
        },
        "immutable_config": immutable_config,
        "immutable_config_sha256": canonical_json_sha256(immutable_config),
        "clusters": {},
    }
    if only_clusters is None and args.overwrite_existing:
        # Validate an existing manifest before loading/training the model.  A
        # same-directory overwrite is only allowed for identical immutable
        # inputs; a different spectral artifact/config needs a different
        # fingerprinted output directory.
        with manifest_file_lock(manifest_path):
            existing = read_manifest(manifest_path)
            if existing is not None:
                validate_manifest_compatibility(
                    existing, manifest, manifest_path=manifest_path
                )
    if only_clusters is not None:
        load_recovery_manifest(
            manifest_path,
            manifest,
            replacement_clusters={f"cluster{k}" for k in only_clusters},
        )

    base_model = load_encoder(args.checkpoint, device, None)
    cluster_updates: dict[str, dict] = {}
    staged_outputs: dict[str, dict[str, Path]] = {}

    for k, name in enumerate(cluster_names):
        if only_clusters is not None and k not in only_clusters:
            print(f"[skip] {name}: not in --only-clusters", flush=True)
            continue
        mask = cluster_ids == k
        pool_emb = emb[mask]
        pool_act = act_emb[mask]
        n = int(mask.sum())
        cluster_updates[name] = {"num_samples": n}
        if n < cfg.min_region_samples:
            print(f"[skip] {name}: only {n} samples", flush=True)
            cluster_updates[name]["status"] = "skipped"
            continue

        # Different clusters may train concurrently, but two writers for the
        # same cluster would share the checkpoint filename and corrupt each
        # other's provenance.  Fail fast instead of serializing a duplicate
        # multi-hour job.
        cluster_job_lock = args.out_dir / f".{name}.training"
        with manifest_file_lock(cluster_job_lock, timeout_sec=1.0):
            print(f"[train] {name}: {n} transitions, {args.epochs} epochs", flush=True)
            model = copy.deepcopy(base_model)
            trained, stats = train_region_predictor(
                model,
                pool_emb,
                pool_act,
                cfg,
                device,
                region=name,
                name_prefix="train_",
                select_best_by_eval=True,
            )
            staged_ckpt_path = staging_dir / f"P_train_{name}_object.ckpt"
            save_region_predictor(
                trained,
                staged_ckpt_path,
                metadata={
                    "cluster": name,
                    "cluster_id": k,
                    "num_samples": n,
                    **stats,
                },
            )
            staged_ckpt_path = staged_ckpt_path.resolve(strict=True)
            staged_metadata_path = staged_ckpt_path.with_suffix(".json").resolve(
                strict=True
            )
            final_ckpt_path = (
                args.out_dir / f"P_train_{name}_object.ckpt"
            ).resolve()
            final_metadata_path = final_ckpt_path.with_suffix(".json")
            staged_outputs[name] = {
                "staged_ckpt": staged_ckpt_path,
                "staged_metadata": staged_metadata_path,
                "final_ckpt": final_ckpt_path,
                "final_metadata": final_metadata_path,
            }
            cluster_updates[name].update(
                {
                    "status": "trained",
                    "best_epoch": stats["best_epoch"],
                    "best_eval_loss": stats["best_eval_loss"],
                }
            )

    if only_clusters is None:
        expected = set(cluster_names)
        actual = set(cluster_updates)
        if actual != expected:
            raise RuntimeError(
                f"Incomplete predictor manifest: expected {sorted(expected)}, "
                f"found {sorted(actual)}"
            )
        missing_ckpts = [
            name
            for name in cluster_names
            if cluster_updates[name].get("status") != "trained"
            or name not in staged_outputs
            or not staged_outputs[name]["staged_ckpt"].is_file()
            or not staged_outputs[name]["staged_metadata"].is_file()
        ]
        if missing_ckpts:
            raise RuntimeError(
                f"Predictor training did not produce complete checkpoints: {missing_ckpts}"
            )

    commit_staged_cluster_merge(
        manifest_path,
        manifest,
        cluster_updates,
        staged_outputs,
        partial_recovery=only_clusters is not None,
        allow_full_overwrite=args.overwrite_existing,
        staging_dir=staging_dir,
    )
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
