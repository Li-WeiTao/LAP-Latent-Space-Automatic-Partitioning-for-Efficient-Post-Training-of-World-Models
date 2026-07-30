#!/usr/bin/env python3
"""Compute a shared train-fitted UMAP for rooms3 and priority5 visualizations.

The script samples the same held-out latent vectors for both labelings. Sampling
is balanced by the mutually exclusive priority5 labels so rare regions remain
visible. UMAP is fitted on training episodes and only transformed test points
are written to the plotting CSV.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


LABELS_PRIORITY5 = (
    "Doorway corridor",
    "Near wall",
    "Common interior",
    "Right room",
    "Left room",
)
REGIONS_PRIORITY5 = (
    "doorway_corridor",
    "near_wall",
    "common",
    "right_room",
    "left_room",
)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "data",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(
            os.environ.get(
                "LAP_TWOROOM_DATA",
                "/data/sicong/weitao/datasets/lewm/tworoom.h5",
            )
        ),
    )
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--embed-dir", type=Path, default=None)
    parser.add_argument("--samples-per-class", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.1)
    return parser.parse_args()


def balanced_indices(
    labels: np.ndarray,
    mask: np.ndarray,
    samples_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = []
    for class_id in range(len(LABELS_PRIORITY5)):
        candidates = np.flatnonzero(mask & (labels == class_id))
        n = min(samples_per_class, len(candidates))
        chosen.append(rng.choice(candidates, size=n, replace=False))
    return np.concatenate(chosen)


def read_xy(h5_path: Path, global_idx: np.ndarray) -> np.ndarray:
    unique_idx, inverse = np.unique(global_idx, return_inverse=True)
    with h5py.File(h5_path, "r") as h5:
        if "proprio" in h5:
            key = "proprio"
        elif "state" in h5:
            key = "state"
        else:
            raise KeyError("Could not find proprio/state coordinates in dataset")
        xy_unique = np.asarray(h5[key][unique_idx], dtype=np.float64)[:, :2]
    return xy_unique[inverse]


def rooms3_labels(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    labels = np.empty(len(x), dtype=object)
    labels[x < 107.0] = "Left room"
    labels[(x >= 107.0) & (x <= 117.0)] = "Doorway corridor"
    labels[x > 117.0] = "Right room"
    return labels


def load_selected_latents(
    embed_dir: Path,
    global_idx: np.ndarray,
    priority_labels: np.ndarray,
) -> np.ndarray:
    del priority_labels  # Labels do not determine which overlapping cache holds a frame.
    output = np.empty((len(global_idx), 192), dtype=np.float32)
    resolved = np.zeros(len(global_idx), dtype=bool)
    offsets = np.array([0, 5, 10, 15], dtype=np.int64)

    for region in REGIONS_PRIORITY5:
        output_positions = np.flatnonzero(~resolved)
        targets = global_idx[output_positions]
        cache_path = embed_dir / f"P_train_{region}_embeddings.npz"
        print(
            f"[latent] scanning {cache_path.name}; unresolved={len(targets)}",
            flush=True,
        )
        with np.load(cache_path) as data:
            starts = np.asarray(data["region_starts"], dtype=np.int64)
            flat_idx = (starts[:, None] + offsets[None, :]).reshape(-1)
            order = np.argsort(flat_idx, kind="stable")
            sorted_idx = flat_idx[order]
            where = np.searchsorted(sorted_idx, targets)
            found = (where < len(sorted_idx)) & (
                sorted_idx[np.minimum(where, len(sorted_idx) - 1)] == targets
            )
            if found.any():
                flat_positions = order[where[found]]
                embeddings = np.asarray(data["emb"], dtype=np.float32).reshape(-1, 192)
                matched_output_positions = output_positions[found]
                output[matched_output_positions] = embeddings[flat_positions]
                resolved[matched_output_positions] = True
                del embeddings
                print(f"[latent] resolved {int(found.sum())} frames", flush=True)
        del flat_idx, order, sorted_idx
        gc.collect()

    if not resolved.all():
        raise RuntimeError(f"Could not locate {int((~resolved).sum())} sampled timesteps")
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis_root = args.project_root / "experiments" / "tworoom"
    split_path = args.split_file or (
        analysis_root
        / "results/geometry_latent_probe_priority5/geometry_latent_probe_priority5_split.npz"
    )
    embed_dir = args.embed_dir or (
        analysis_root / "results/tworoom_geometry_train_region_predictors"
    )
    h5_path = args.data_file

    with np.load(split_path) as split:
        all_global_idx = np.asarray(split["global_idx"], dtype=np.int64)
        all_priority_y = np.asarray(split["y"], dtype=np.int64)
        train_mask = np.asarray(split["train_mask"], dtype=bool)
        test_mask = np.asarray(split["test_mask"], dtype=bool)

    rng = np.random.default_rng(args.seed)
    train_take = balanced_indices(
        all_priority_y, train_mask, args.samples_per_class, rng
    )
    test_take = balanced_indices(
        all_priority_y, test_mask, args.samples_per_class, rng
    )

    selected_take = np.concatenate([train_take, test_take])
    selected_global_idx = all_global_idx[selected_take]
    selected_priority_y = all_priority_y[selected_take]
    print(
        f"[sample] train={len(train_take)} test={len(test_take)} "
        f"total={len(selected_take)}",
        flush=True,
    )

    selected_xy = read_xy(h5_path, selected_global_idx)
    selected_latents = load_selected_latents(
        embed_dir, selected_global_idx, selected_priority_y
    )

    n_train = len(train_take)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(selected_latents[:n_train])
    test_scaled = scaler.transform(selected_latents[n_train:])

    pca = PCA(n_components=50, random_state=args.seed)
    train_reduced = pca.fit_transform(train_scaled)
    test_reduced = pca.transform(test_scaled)
    print(
        f"[pca] cumulative explained variance={pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
        n_jobs=1,
        verbose=True,
    )
    print("[umap] fitting on balanced training episodes", flush=True)
    reducer.fit(train_reduced)
    print("[umap] transforming held-out test episodes", flush=True)
    test_umap = reducer.transform(test_reduced)

    test_global_idx = selected_global_idx[n_train:]
    test_priority_y = selected_priority_y[n_train:]
    test_xy = selected_xy[n_train:]
    test_rooms3 = rooms3_labels(test_xy)

    csv_path = args.output_dir / "latent_umap_test_coordinates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("global_idx,umap_1,umap_2,x,y,rooms3,priority5\n")
        for i in range(len(test_global_idx)):
            handle.write(
                f"{int(test_global_idx[i])},{test_umap[i,0]:.8f},{test_umap[i,1]:.8f},"
                f"{test_xy[i,0]:.6f},{test_xy[i,1]:.6f},"
                f"{test_rooms3[i]},{LABELS_PRIORITY5[int(test_priority_y[i])]}\n"
            )

    metadata = {
        "sample_unit": "unique global timestep latent",
        "projection_protocol": "fit train episodes, transform held-out test episodes",
        "sampling": "priority5 class-balanced",
        "samples_per_class": args.samples_per_class,
        "train_samples": int(len(train_take)),
        "test_samples": int(len(test_take)),
        "seed": args.seed,
        "preprocessing": "train StandardScaler then train PCA(50)",
        "pca_explained_variance_50": float(pca.explained_variance_ratio_.sum()),
        "umap": {
            "version": umap.__version__,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "euclidean",
        },
        "output_csv": str(csv_path),
    }
    metadata_path = args.output_dir / "latent_umap_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote {csv_path}", flush=True)
    print(f"[done] wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
