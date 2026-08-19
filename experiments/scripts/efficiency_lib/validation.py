"""Static validation for efficiency benchmark provenance and cache contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import sha256_file


LEWM_TWOROOM_CHECKPOINT_SHA256 = (
    "18b5764492c74de5487efdadb66adab11876cb230952765b17c0815fa87b13ff"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lewm_checkpoint(checkpoint: Path) -> None:
    checkpoint = checkpoint.resolve(strict=True)
    digest = sha256_file(checkpoint)
    if digest != LEWM_TWOROOM_CHECKPOINT_SHA256:
        raise ValueError(
            f"TwoRoom LeWM checkpoint SHA256 mismatch: {checkpoint} "
            f"(got {digest}, expected {LEWM_TWOROOM_CHECKPOINT_SHA256})"
        )


def validate_training_latent_cache(
    cache_path: Path,
    *,
    partition_dir: Path,
    expected_model: str = "lewm",
    checkpoint: Path | None = None,
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
            base_path = base.get("path")
            base_sha = base.get("sha256")
        else:
            base_path = base
            base_sha = None
        if family and "subjepa" in str(family).lower():
            raise ValueError(
                f"training cache provenance points to Sub-JEPA ({candidate}), expected {expected_model}"
            )
        if base_sha and base_sha != LEWM_TWOROOM_CHECKPOINT_SHA256:
            raise ValueError(
                f"training cache base checkpoint SHA mismatch in {candidate}: {base_sha}"
            )
        provenance["provenance_manifest"] = str(candidate)
        break

    if checkpoint is not None:
        validate_lewm_checkpoint(checkpoint)

    return provenance


def validate_lap_predictor_manifest(
    run_dir: Path,
    *,
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
    if base_sha and base_sha != sha256_file(checkpoint.resolve(strict=True)):
        raise ValueError(
            f"LAP manifest base checkpoint SHA does not match LeWM baseline: {manifest_path}"
        )

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
    validate_lap_predictor_manifest(
        predictor_dir,
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
        checkpoint=Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"),
        expect_regional=True,
        require_partition=True,
    )
    return out
