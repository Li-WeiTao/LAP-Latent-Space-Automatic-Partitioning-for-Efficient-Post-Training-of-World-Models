"""Static validation for efficiency benchmark provenance and cache contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import sha256_file


LEWM_CHECKPOINT_SHA256: dict[str, str] = {
    "tworoom": "18b5764492c74de5487efdadb66adab11876cb230952765b17c0815fa87b13ff",
    "pusht": "e727d64a8b3535c3152dc72688bb7565c536c1b1317c56d04072cf7cc1183cc2",
    "reacher": "1fcf86a118e10d3c608088861c59a39c7d5bc39c2a9760e5f799202c70a80d6d",
    "cube": "2c00552d586b9cbc03ec43c216d84f2e0f38d47b6b5dda6d81ec35163731f581",
}

# Legacy alias
LEWM_TWOROOM_CHECKPOINT_SHA256 = LEWM_CHECKPOINT_SHA256["tworoom"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_task_checkpoint(task: str, checkpoint: Path) -> None:
    """Validate a task-specific LeWM baseline checkpoint when present."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing LeWM checkpoint for {task}: {checkpoint}")
    expected = LEWM_CHECKPOINT_SHA256.get(task)
    if expected is None:
        return
    digest = sha256_file(checkpoint)
    if digest != expected:
        raise ValueError(
            f"{task} LeWM checkpoint SHA256 mismatch: {checkpoint} "
            f"(got {digest}, expected {expected})"
        )


def validate_lewm_checkpoint(checkpoint: Path) -> None:
    """Backward-compatible TwoRoom-only helper."""
    validate_task_checkpoint("tworoom", checkpoint)


def assert_cache_matches_train_pool(
    cache_path: Path,
    pool_starts_path: Path,
) -> None:
    pool_starts_path = pool_starts_path.resolve(strict=True)
    pool = np.load(pool_starts_path)
    with np.load(cache_path, allow_pickle=False) as data:
        cache_starts = np.asarray(data["region_starts"], dtype=np.int64)
    if not np.array_equal(cache_starts, pool):
        if len(cache_starts) != len(pool) or not np.array_equal(
            np.sort(cache_starts), np.sort(pool)
        ):
            raise ValueError(
                f"training cache region_starts do not match train pool {pool_starts_path}"
            )


def validate_joint_train_pool_dataset(
    *,
    train_pool_starts: Path,
    training_latent_cache: Path,
    data_file: Path,
    dataset_name: str,
    history_size: int,
    num_preds: int,
    frameskip: int,
    img_size: int,
) -> dict[str, int | bool]:
    """Validate joint train-pool loader covers the LAP cache exactly."""
    assert_cache_matches_train_pool(training_latent_cache, train_pool_starts)
    with np.load(training_latent_cache, allow_pickle=False) as data:
        expected = int(len(data["region_starts"]))

    from .train_pool import build_joint_train_pool_dataset

    # Import path setup mirrors training benchmark entrypoints.
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[3]
    tworoom = repo_root / "experiments" / "tworoom"
    lewm_root = repo_root.parent / "le-wm"
    for path in (repo_root, tworoom, lewm_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from joint_continue_tworoom import resolve_state_key

    dataset = build_joint_train_pool_dataset(
        data_file=data_file,
        dataset_name=dataset_name,
        train_pool_starts=train_pool_starts,
        history_size=history_size,
        num_preds=num_preds,
        frameskip=frameskip,
        img_size=img_size,
        resolve_state_key=resolve_state_key,
    )
    if len(dataset) != expected:
        raise ValueError(
            f"joint train pool dataset length {len(dataset)} != LAP cache {expected}"
        )
    return {
        "num_windows": len(dataset),
        "matches_cache": True,
    }


def validate_training_latent_cache(
    cache_path: Path,
    *,
    partition_dir: Path,
    expected_model: str = "lewm",
    checkpoint: Path | None = None,
    task: str = "tworoom",
) -> dict[str, Any]:
    cache_path = cache_path.resolve(strict=True)
    with np.load(cache_path, allow_pickle=False) as data:
        missing = {"emb", "act_emb", "region_starts"} - set(data.files)
        if missing:
            raise ValueError(
                f"training latent cache {cache_path} missing keys {sorted(missing)}; "
                "routing-only caches cannot be used for Regional-FT timing"
            )
        emb = np.asarray(data["emb"])
        act_emb = np.asarray(data["act_emb"])
        starts = np.asarray(data["region_starts"], dtype=np.int64)
    if emb.ndim != 3 or act_emb.ndim != 3 or emb.shape[0] != len(starts):
        raise ValueError(f"invalid training cache shapes at {cache_path}")
    if emb.shape[1] < 4 or act_emb.shape[0] != emb.shape[0]:
        raise ValueError(
            f"training cache must contain full predictor windows (emb T>=4), got {emb.shape}"
        )

    labels_path = partition_dir / "cluster_labels.npz"
    with np.load(labels_path, allow_pickle=False) as part:
        id_key = "sample_ids" if "sample_ids" in part.files else "global_idx"
        partition_ids = np.asarray(part[id_key], dtype=np.int64)
        partition_labels = np.asarray(part["labels"], dtype=np.int64)

    positions = np.searchsorted(partition_ids, starts)
    valid = (positions < len(partition_ids)) & (partition_ids[positions] == starts)
    if not valid.all():
        missing = starts[~valid][:5].tolist()
        raise ValueError(
            f"training cache region_starts not covered by partition assignments: {missing}"
        )

    provenance: dict[str, Any] = {
        "cache_path": str(cache_path),
        "num_transitions": int(len(starts)),
        "emb_shape": list(emb.shape),
        "partition_num_samples": int(len(partition_ids)),
        "partition_cluster_counts": np.bincount(partition_labels).tolist(),
        "region_starts_sha256": sha256_file(cache_path),
    }

    sidecar = cache_path.with_suffix(".json")
    manifest_candidates = [
        cache_path.parent / "representation_manifest.json",
        cache_path.parent / "manifest.json",
        cache_path.parent.parent / "manifest.json",
    ]
    for candidate in (sidecar, *manifest_candidates):
        if not candidate.is_file():
            continue
        payload = _read_json(candidate)
        family = payload.get("model_family") or payload.get("implementation_backend")
        base = payload.get("base_checkpoint") or payload.get("checkpoint")
        if isinstance(base, dict):
            base_sha = base.get("sha256")
        else:
            base_sha = None
        if family and "subjepa" in str(family).lower():
            raise ValueError(
                f"training cache provenance points to Sub-JEPA ({candidate}), expected {expected_model}"
            )
        expected_sha = LEWM_CHECKPOINT_SHA256.get(task)
        if base_sha and expected_sha and base_sha != expected_sha:
            raise ValueError(
                f"training cache base checkpoint SHA mismatch in {candidate}: {base_sha}"
            )
        provenance["provenance_manifest"] = str(candidate)
        break

    if checkpoint is not None:
        validate_task_checkpoint(task, checkpoint)

    return provenance


def validate_lap_predictor_manifest(
    run_dir: Path,
    *,
    task: str,
    checkpoint: Path,
    expect_regional: bool,
    require_partition: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LAP manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    family = (
        manifest.get("model_family")
        or manifest.get("implementation_backend")
        or manifest.get("immutable_config", {}).get("model_family")
    )
    if family and "subjepa" in str(family).lower():
        raise ValueError(f"LAP run dir uses Sub-JEPA manifest: {manifest_path}")

    base_sha = None
    base = manifest.get("base_checkpoint") or manifest.get("pretrained_model")
    if isinstance(base, dict):
        base_sha = base.get("sha256")
    elif isinstance(base, str):
        base_path = Path(base)
        if base_path.is_file():
            base_sha = sha256_file(base_path)
    expected_sha = LEWM_CHECKPOINT_SHA256.get(task)
    if base_sha and expected_sha and base_sha != expected_sha:
        raise ValueError(
            f"LAP manifest base checkpoint SHA does not match {task} LeWM baseline: {manifest_path}"
        )
    if checkpoint.is_file() and expected_sha:
        if sha256_file(checkpoint) != expected_sha:
            raise ValueError(f"baseline checkpoint SHA mismatch for task {task}")

    num_regions = int(
        manifest.get("num_regions")
        or manifest.get("immutable_config", {}).get("num_clusters")
        or len(manifest.get("clusters", {}))
        or len(manifest.get("regions", {}))
        or 0
    )
    if expect_regional and num_regions < 2:
        raise ValueError(
            f"expected regional LAP deployment (K>=2) at {run_dir}, got num_regions={num_regions}"
        )
    if not expect_regional and num_regions != 1:
        raise ValueError(
            f"expected global LAP deployment (K=1) at {run_dir}, got num_regions={num_regions}"
        )

    for region_id in range(num_regions):
        ckpt = run_dir / f"P_train_cluster{region_id}_object.ckpt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing regional predictor checkpoint: {ckpt}")

    partition_root = run_dir / "partition"
    if expect_regional and require_partition and not partition_root.is_dir():
        raise FileNotFoundError(
            f"Regional LAP inference requires partition artifact dir: {partition_root}"
        )

    return {
        "manifest_path": str(manifest_path),
        "num_regions": num_regions,
        "model_family": family,
    }


def validate_tworoom_predictor_partition_provenance(
    predictor_dir: Path,
    partition_root: Path,
) -> None:
    """Ensure assembled TwoRoom LAP predictors match the deployed Auto-LAP partition."""
    predictor_manifest = _read_json(predictor_dir / "manifest.json")
    partition_manifest = _read_json(partition_root / "manifest.json")

    predictor_cluster = predictor_manifest.get("cluster_artifact_dir")
    if predictor_cluster:
        predictor_cluster = str(Path(predictor_cluster).resolve())

    deployed = partition_manifest.get("method_metadata", {}).get(
        "deployed_partition", {}
    )
    deployed_seed = partition_manifest.get("partition_seed")
    predictor_seed = predictor_manifest.get("seed")
    if deployed_seed is not None and predictor_seed is not None:
        if int(deployed_seed) != int(predictor_seed):
            raise ValueError(
                f"predictor seed {predictor_seed} != deployed partition seed {deployed_seed}"
            )

    predictor_sha = predictor_manifest.get("cluster_artifact_sha256")
    if isinstance(predictor_sha, dict) and predictor_cluster:
        cluster_meta = partition_root / "partition" / "cluster_meta.json"
        if cluster_meta.is_file():
            meta = _read_json(cluster_meta)
            if meta.get("seed") is not None and predictor_seed is not None:
                if int(meta["seed"]) != int(predictor_seed):
                    raise ValueError(
                        "partition artifact seed does not match predictor manifest seed"
                    )

    immutable = predictor_manifest.get("immutable_config", {})
    cluster_source = immutable.get("cluster_source", {})
    if cluster_source.get("path") and predictor_cluster:
        if str(Path(cluster_source["path"]).resolve()) != predictor_cluster:
            raise ValueError("predictor immutable cluster_source path mismatch")

    base = predictor_manifest.get("base_checkpoint", {})
    if isinstance(base, dict) and base.get("sha256"):
        if base["sha256"] != LEWM_CHECKPOINT_SHA256["tworoom"]:
            raise ValueError("predictor manifest base checkpoint is not TwoRoom LeWM")

    _ = deployed


def read_gate_branch(gate_manifest: Path) -> str:
    manifest = _read_json(gate_manifest.resolve(strict=True))
    meta = manifest.get("method_metadata", {})
    return str(
        meta.get("selected_post_training")
        or meta.get("automatic_gate", {}).get("selected_post_training")
        or ("regional_predictors" if manifest.get("num_clusters", 1) > 1 else "global_predictor")
    )


def materialize_tworoom_lap_run_dir(
    *,
    predictor_dir: Path,
    partition_root: Path,
    scratch_dir: Path,
) -> Path:
    """Assemble a lap_run_dir with predictors + Auto-LAP partition artifacts."""
    predictor_dir = predictor_dir.resolve(strict=True)
    partition_root = partition_root.resolve(strict=True)
    validate_tworoom_predictor_partition_provenance(predictor_dir, partition_root)
    validate_lap_predictor_manifest(
        predictor_dir,
        task="tworoom",
        checkpoint=Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"),
        expect_regional=True,
        require_partition=False,
    )

    out = scratch_dir / "lap_run_tworoom_seed0"
    out.mkdir(parents=True, exist_ok=True)
    partition_link = out / "partition"
    if partition_link.exists() or partition_link.is_symlink():
        if partition_link.is_symlink() or partition_link.is_dir():
            partition_link.unlink(missing_ok=True)
    partition_src = partition_root / "partition"
    if not partition_src.is_dir():
        raise FileNotFoundError(f"Missing partition artifact directory: {partition_src}")
    partition_link.symlink_to(partition_src, target_is_directory=True)

    for ckpt in predictor_dir.glob("P_train_cluster*_object.ckpt"):
        link = out / ckpt.name
        if link.exists() or link.is_symlink():
            link.unlink(missing_ok=True)
        link.symlink_to(ckpt)

    manifest_link = out / "manifest.json"
    if manifest_link.exists() or manifest_link.is_symlink():
        manifest_link.unlink(missing_ok=True)
    manifest_link.symlink_to(predictor_dir / "manifest.json")
    validate_lap_predictor_manifest(
        out,
        task="tworoom",
        checkpoint=Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"),
        expect_regional=True,
        require_partition=True,
    )
    return out
