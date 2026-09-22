"""Audit saved partition labels across seeds without refitting partitions."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--method", required=True)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated partition seeds")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_saved(root: Path, task: str, method: str, k: int, seed: int) -> dict:
    directory = root / "experiments" / task / f"matrix_k{k}" / "partitions" / method / f"seed{seed}"
    manifest_path = directory / "manifest.json"
    labels_path = directory / "cluster_labels.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(labels_path, allow_pickle=False) as archive:
        key = "sample_ids" if "sample_ids" in archive.files else "global_idx"
        sample_ids = np.asarray(archive[key], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    if manifest["method"] != method or int(manifest["num_clusters"]) != k:
        raise ValueError(f"method/K mismatch in {manifest_path}")
    if labels.shape != sample_ids.shape or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"sample ID mismatch or duplicates in {labels_path}")
    if len(labels) != sum(manifest["cluster_counts"]):
        raise ValueError(f"cluster counts mismatch in {manifest_path}")
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"label outside 0..{k-1} in {labels_path}")
    return {
        "directory": str(directory),
        "manifest": manifest,
        "sample_ids": sample_ids,
        "labels": labels,
    }


def compare_labels(left: np.ndarray, right: np.ndarray, k: int) -> dict:
    contingency = np.bincount(left * k + right, minlength=k * k).reshape(k, k)
    row, col = linear_sum_assignment(contingency, maximize=True)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(left, right)),
        "hungarian_aligned_agreement": float(contingency[row, col].sum() / len(left)),
        "alignment": {str(int(col_i)): int(row_i) for row_i, col_i in zip(row, col)},
        "contingency": contingency.tolist(),
    }


def main() -> None:
    args = parse_args()
    root = args.repo.resolve(strict=True)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(tasks) != len(set(tasks)) or len(seeds) != len(set(seeds)) or len(seeds) < 2:
        raise ValueError("tasks/seeds must be unique; at least two seeds are required")
    result = {
        "source": "saved deployment cluster_labels.npz; no partition fitting or evaluation",
        "method": args.method,
        "num_clusters": args.num_clusters,
        "seeds": seeds,
        "tasks": {},
    }
    for task in tasks:
        saved = {
            seed: load_saved(root, task, args.method, args.num_clusters, seed)
            for seed in seeds
        }
        reference = saved[seeds[0]]
        reference_ids = reference["sample_ids"]
        cache_hash = reference["manifest"]["latent_cache_sha256"]
        for seed, item in saved.items():
            if not np.array_equal(reference_ids, item["sample_ids"]):
                raise ValueError(f"{task}: seed {seed} has different sample IDs")
            if item["manifest"]["latent_cache_sha256"] != cache_hash:
                raise ValueError(f"{task}: seed {seed} has a different latent cache")
        pairs = {}
        for a, b in combinations(seeds, 2):
            pairs[f"{a}-{b}"] = compare_labels(
                saved[a]["labels"], saved[b]["labels"], args.num_clusters
            )
        scores = [entry["adjusted_rand_index"] for entry in pairs.values()]
        result["tasks"][task] = {
            "num_points": int(len(reference_ids)),
            "latent_cache_sha256": cache_hash,
            "seed_metadata": {
                str(seed): {
                    "path": item["directory"],
                    "cluster_counts": item["manifest"]["cluster_counts"],
                    "fit_to_deploy_disagreement": item["manifest"]["method_metadata"].get(
                        "latent_to_deployed_disagreement"
                    ),
                }
                for seed, item in saved.items()
            },
            "pairs": pairs,
            "ari_mean": float(np.mean(scores)),
            "ari_min": float(np.min(scores)),
            "ari_max": float(np.max(scores)),
        }
        print(f"{task}: n={len(reference_ids)} ARI " + ", ".join(
            f"{name}={entry['adjusted_rand_index']:.4f}"
            for name, entry in pairs.items()
        ) + f"; mean={np.mean(scores):.4f}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
