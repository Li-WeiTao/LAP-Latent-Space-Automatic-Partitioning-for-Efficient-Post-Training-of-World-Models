#!/usr/bin/env python3
"""Materialize matrix spectral partitions with training-compatible sample IDs.

Formal gate nominal spectral partitions store cluster_labels keyed by dense
unique-latent indices (0..N-1).  Training consumes region_starts / global
timestep IDs from the full embedding cache.  This script copies the locked
partition artifact from formal gate and remaps labels onto global IDs without
modifying formal artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.fit_partition import load_unique_latents  # noqa: E402


def materialize_seed(
    *,
    formal_seed_dir: Path,
    matrix_seed_dir: Path,
    latent_cache: Path,
    frameskip: int,
) -> dict[str, object]:
    if not formal_seed_dir.is_dir():
        raise FileNotFoundError(f"missing formal spectral partition: {formal_seed_dir}")
    formal_labels_path = formal_seed_dir / "cluster_labels.npz"
    formal_partition_dir = formal_seed_dir / "partition"
    if not formal_labels_path.is_file() or not formal_partition_dir.is_dir():
        raise FileNotFoundError(f"incomplete formal spectral partition: {formal_seed_dir}")

    _, unique_ids, cache_stats = load_unique_latents(latent_cache.resolve(), frameskip)
    with np.load(formal_labels_path, allow_pickle=False) as data:
        index_ids = np.asarray(data["sample_ids"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)

    if index_ids.min() == unique_ids.min() and index_ids.max() == unique_ids.max():
        remapped_ids = index_ids
        mode = "already_global_ids"
    elif index_ids.min() >= 0 and index_ids.max() < len(unique_ids):
        remapped_ids = unique_ids[index_ids]
        mode = "remapped_from_dense_index"
    else:
        raise RuntimeError(
            "unexpected spectral sample_ids range: "
            f"min={index_ids.min()} max={index_ids.max()} "
            f"unique_count={len(unique_ids)}"
        )

    if matrix_seed_dir.is_symlink():
        matrix_seed_dir.unlink()
    elif matrix_seed_dir.exists():
        shutil.rmtree(matrix_seed_dir)
    matrix_seed_dir.mkdir(parents=True, exist_ok=True)

    dst_partition = matrix_seed_dir / "partition"
    if dst_partition.exists() or dst_partition.is_symlink():
        dst_partition.unlink()
    dst_partition.symlink_to(formal_partition_dir.resolve())

    np.savez_compressed(
        matrix_seed_dir / "cluster_labels.npz",
        sample_ids=remapped_ids,
        global_idx=remapped_ids,
        labels=labels,
    )

    manifest_src = formal_seed_dir / "manifest.json"
    manifest: dict[str, object]
    if manifest_src.is_file():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "dataset": "tworoom",
            "method": "spectral",
            "selected_method": "spectral",
        }
    manifest["matrix_materialization"] = {
        "source": str(formal_seed_dir.resolve()),
        "sample_id_mode": mode,
        "cache_stats": cache_stats,
    }
    (matrix_seed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with np.load(latent_cache) as cache:
        starts = np.asarray(cache["region_starts"], dtype=np.int64)
    positions = np.searchsorted(remapped_ids, starts)
    valid = positions < len(remapped_ids)
    valid[valid] &= remapped_ids[positions[valid]] == starts[valid]
    if not valid.all():
        missing = starts[~valid][:5].tolist()
        raise RuntimeError(
            f"remapped spectral labels still missing training starts: {missing}"
        )

    return {
        "seed_dir": str(matrix_seed_dir),
        "sample_id_mode": mode,
        "num_labels": int(len(remapped_ids)),
        "id_min": int(remapped_ids.min()),
        "id_max": int(remapped_ids.max()),
        "training_starts_covered": int(valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/tworoom/subjepa/formal/partitions/spectral",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/tworoom/subjepa/matrix/partitions/spectral",
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/tworoom/subjepa/formal/preparation/embedding_cache.npz",
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = []
    for seed in seeds:
        row = materialize_seed(
            formal_seed_dir=args.formal_root / f"seed{seed}",
            matrix_seed_dir=args.matrix_root / f"seed{seed}",
            latent_cache=args.latent_cache,
            frameskip=args.frameskip,
        )
        row["seed"] = seed
        rows.append(row)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
