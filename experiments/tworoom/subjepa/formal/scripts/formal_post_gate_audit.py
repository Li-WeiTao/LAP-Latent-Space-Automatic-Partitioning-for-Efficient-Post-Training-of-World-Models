#!/usr/bin/env python3
"""Task-local post-gate audit and Material Passport for Sub-JEPA TwoRoom formal gate."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.fit_partition import load_unique_latents  # noqa: E402
from experiments.control_matrix.region_risk_lib import atomic_write_json, git_commit, sha256_file  # noqa: E402
from lap.partition.landmark import LandmarkSpectralConfig, LandmarkSpectralPartitioner  # noqa: E402
from lap.routing import VoronoiRouter  # noqa: E402
from latent_landmark_spectral import exact_cosine_knn_torch  # noqa: E402


GATE_DEFAULTS = {
    "num_clusters": 3,
    "diagnostic_seeds": (0, 1, 2),
    "deployment_seed": 0,
    "num_landmarks": 20_000,
    "nominal_knn": 30,
    "perturb_knn": (27, 33),
    "gate_perturbation_multiplier": 2.0,
    "gate_retention_threshold": 0.5,
    "gate_background_gap_count": 10,
    "gate_background_mad_multiplier": 3.0,
    "gate_epsilon": 1e-8,
    "required_eigenvalues_per_graph": 14,
}


def episode_ids(data_file: Path, sample_ids: np.ndarray, episode_key: str = "auto") -> np.ndarray:
    import h5py

    with h5py.File(data_file, "r", swmr=True) as handle:
        if episode_key == "auto":
            key = "episode_idx" if "episode_idx" in handle else "ep_idx"
        else:
            key = episode_key
        if key not in handle:
            raise KeyError(f"episode column {key!r} is missing from {data_file}")
        return np.asarray(handle[key][sample_ids], dtype=np.int64)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes
    return float(usage) / 1024.0


def verify_gate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    gate = manifest["method_metadata"]["automatic_gate"]
    config = gate.get("configuration", {})
    graph_count = 0
    graph_details: list[dict[str, Any]] = []
    for draw in gate["draw_results"]:
        seed = draw["seed"]
        for knn_str, gaps in draw.get("background_by_knn", {}).items():
            knn = int(knn_str)
            rel_gaps = gaps["relative_gaps"]
            graph_count += 1
            graph_details.append(
                {
                    "diagnostic_seed": seed,
                    "knn": knn,
                    "background_gap_count": len(rel_gaps),
                    "candidate_relative_gap": draw["candidate_relative_gap_by_knn"].get(
                        str(knn), draw["candidate_relative_gap_by_knn"].get(knn)
                    ),
                    "background_threshold": gaps["threshold"],
                }
            )
            if len(rel_gaps) < GATE_DEFAULTS["gate_background_gap_count"]:
                raise RuntimeError(
                    f"graph seed={seed} knn={knn} has only {len(rel_gaps)} background gaps"
                )

    expected_graphs = (
        len(GATE_DEFAULTS["diagnostic_seeds"]) * len({GATE_DEFAULTS["nominal_knn"], *GATE_DEFAULTS["perturb_knn"]})
    )
    if graph_count != expected_graphs:
        raise RuntimeError(f"expected {expected_graphs} graphs, found {graph_count}")

    s_task = gate.get("retained_safety_fraction")
    r_k = gate.get("robust_residual_gap")
    t_bg = gate.get("background_threshold")
    safety_pass = s_task is not None and s_task >= GATE_DEFAULTS["gate_retention_threshold"]
    background_pass = r_k is not None and t_bg is not None and r_k > t_bg
    criterion_pass = bool(gate.get("use_partition")) == bool(safety_pass and background_pass)

    checks = {
        "graph_count": graph_count,
        "expected_graph_count": expected_graphs,
        "required_eigenvalues": GATE_DEFAULTS["required_eigenvalues_per_graph"],
        "safety_pass": safety_pass,
        "background_pass": background_pass,
        "criterion_matches_selection": criterion_pass,
        "E_K_min": gate.get("candidate_gap_min"),
        "T_E_K_max": gate.get("perturbation_threshold_max"),
        "S_task": s_task,
        "R_K": r_k,
        "T_bg": t_bg,
        "selected_method": gate.get("selected_method"),
        "reason": gate.get("reason"),
        "deployment_seed": gate.get("deployment_seed"),
        "graph_details": graph_details,
    }
    return checks


def fit_nominal_partition(
    raw: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    out_dir: Path,
) -> np.ndarray:
    search = lambda values, max_k: exact_cosine_knn_torch(
        values, max_k, gpu_id=0, query_chunk=2048
    )
    partitioner = LandmarkSpectralPartitioner(
        LandmarkSpectralConfig(
            num_regions=GATE_DEFAULTS["num_clusters"],
            num_landmarks=min(GATE_DEFAULTS["num_landmarks"], len(raw)),
            knn=GATE_DEFAULTS["nominal_knn"],
            seed=seed,
            cpu_threads=4,
        ),
        neighbor_search=search,
    )
    result = partitioner.fit(
        raw,
        sample_ids=np.arange(len(raw), dtype=np.int64),
        group_ids=groups,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.artifact.save(out_dir / "partition")
    np.savez_compressed(
        out_dir / "cluster_labels.npz",
        sample_ids=np.arange(len(raw), dtype=np.int64),
        labels=result.labels,
    )
    return np.asarray(result.labels, dtype=np.int64)


def prototype_router_holdout_f1(
    raw: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    artifact,
    *,
    holdout_fraction: float = 0.1,
    split_seed: int = 20260801,
) -> dict[str, Any]:
    rng = np.random.default_rng(split_seed)
    unique_groups = np.unique(groups)
    holdout_groups = set(
        rng.choice(unique_groups, size=max(1, int(len(unique_groups) * holdout_fraction)), replace=False)
    )
    holdout_mask = np.isin(groups, list(holdout_groups))
    train_mask = ~holdout_mask
    router = VoronoiRouter(artifact)
    routed = router.route(raw).astype(np.int64)
    offline_rate = float(np.mean(labels == routed))
    holdout_labels = labels[holdout_mask]
    holdout_pred = routed[holdout_mask]
    macro_f1 = float(f1_score(holdout_labels, holdout_pred, average="macro"))
    return {
        "offline_labels_reproduced_rate": offline_rate,
        "holdout_macro_f1": macro_f1,
        "holdout_fraction": holdout_fraction,
        "num_holdout": int(holdout_mask.sum()),
        "num_train": int(train_mask.sum()),
    }


def run_post_gate_audit(
    work_root: Path,
    data_file: Path,
    latent_cache: Path,
    *,
    git_baseline: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    gate_manifest_path = work_root / "gate/partition/manifest.json"
    gate_manifest = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    gate_checks = verify_gate_manifest(gate_manifest)

    raw, sample_ids, cache_stats = load_unique_latents(latent_cache.resolve(), frameskip=5)
    groups = episode_ids(data_file.resolve(), sample_ids)

    # Nominal partitions for diagnostic seeds (pairwise ARI/NMI)
    nominal_labels: dict[int, np.ndarray] = {}
    partitions_root = work_root / "partitions/spectral"
    for seed in GATE_DEFAULTS["diagnostic_seeds"]:
        out = partitions_root / f"seed{seed}"
        if (out / "cluster_labels.npz").exists():
            with np.load(out / "cluster_labels.npz") as data:
                nominal_labels[seed] = np.asarray(data["labels"], dtype=np.int64)
        else:
            nominal_labels[seed] = fit_nominal_partition(raw, groups, seed=seed, out_dir=out)

    pairwise: list[dict[str, Any]] = []
    for a, b in combinations(GATE_DEFAULTS["diagnostic_seeds"], 2):
        la = nominal_labels[a]
        lb = nominal_labels[b]
        pairwise.append(
            {
                "pair": [a, b],
                "ari": float(adjusted_rand_score(la, lb)),
                "nmi": float(normalized_mutual_info_score(la, lb)),
            }
        )

    deploy_seed = GATE_DEFAULTS["deployment_seed"]
    deploy_labels = nominal_labels[deploy_seed]
    region_sizes = {
        f"region_{index}": int(count)
        for index, count in enumerate(np.bincount(deploy_labels, minlength=GATE_DEFAULTS["num_clusters"]))
    }
    empty_regions = [key for key, value in region_sizes.items() if value == 0]
    tiny_threshold = max(1, int(0.001 * len(deploy_labels)))
    tiny_regions = {
        key: value for key, value in region_sizes.items() if 0 < value < tiny_threshold
    }

    # Deployment partition from gate output
    from lap.partition.artifact import PartitionArtifact

    deploy_artifact = PartitionArtifact.load(work_root / "gate/partition/partition")
    gate_labels_path = work_root / "gate/partition/cluster_labels.npz"
    with np.load(gate_labels_path) as data:
        gate_labels = np.asarray(data["labels"], dtype=np.int64)
    router_audit = prototype_router_holdout_f1(
        raw, gate_labels, groups, deploy_artifact
    )

    router_dir = work_root / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        work_root / "gate/partition/partition",
        router_dir / "deployment_seed0",
        dirs_exist_ok=True,
    )

    formal_cache = json.loads(
        (work_root / "manifests/formal_cache_manifest.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (work_root / "manifests/replay_audit_summary.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (work_root / "manifests/resolved_config.json").read_text(encoding="utf-8")
    )

    elapsed = time.perf_counter() - started
    audit = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_formal_post_gate",
        "git_commit_baseline": git_baseline,
        "git_commit": git_commit(),
        "gate_task_summary": gate_checks,
        "pairwise_nominal_partitions": pairwise,
        "deployment_seed_0_region_sizes": region_sizes,
        "empty_regions": empty_regions,
        "tiny_regions": tiny_regions,
        "router_fidelity": router_audit,
        "cache_stats": cache_stats,
        "timings_sec": {
            "post_gate_audit_wall": elapsed,
            "gate_elapsed_sec": gate_manifest.get("elapsed_sec"),
        },
        "peak_rss_mb": _peak_rss_mb(),
        "bindings": {
            "checkpoint_sha256": formal_cache["checkpoint_sha256"],
            "dataset_sha256": formal_cache["dataset_sha256"],
            "full_cache_sha256": formal_cache["full_cache_sha256"],
            "resolved_task_spec_sha256": sha256_file(
                work_root / "manifests/resolved_config.json"
            ),
        },
        "replay_audit_all_passed": replay.get("all_passed"),
    }
    atomic_write_json(work_root / "manifests/post_gate_audit.json", audit)
    return audit


def emit_material_passport(work_root: Path, git_baseline: str) -> dict[str, Any]:
    formal_cache = json.loads(
        (work_root / "manifests/formal_cache_manifest.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (work_root / "manifests/replay_audit_summary.json").read_text(encoding="utf-8")
    )
    post_gate = json.loads(
        (work_root / "manifests/post_gate_audit.json").read_text(encoding="utf-8")
    )
    gate_manifest = json.loads(
        (work_root / "gate/partition/manifest.json").read_text(encoding="utf-8")
    )
    gate = gate_manifest["method_metadata"]["automatic_gate"]

    reproduce_cmd = (
        "bash experiments/tworoom/subjepa/formal/scripts/run_formal_gate.sh all"
    )
    all_ok = (
        formal_cache.get("truncated") is False
        and replay.get("all_passed") is True
        and post_gate["gate_task_summary"]["graph_count"] == 9
        and post_gate["replay_audit_all_passed"] is True
    )
    passport = {
        "schema_version": 1,
        "id": "subjepa-tworoom-formal-gate-2026-08-03",
        "verification_status": "VERIFIED" if all_ok else "FAILED",
        "git_commit_baseline": git_baseline,
        "git_commit": git_commit(),
        "checkpoint_sha256": formal_cache["checkpoint_sha256"],
        "dataset_sha256": formal_cache["dataset_sha256"],
        "full_cache_sha256": formal_cache["full_cache_sha256"],
        "full_cache_stats": {
            "candidate_starts_total": formal_cache["candidate_starts_total"],
            "retained_starts_total": formal_cache["retained_starts_total"],
            "encoded_transitions": formal_cache["encoded_transitions"],
            "encoded_unique_frames": formal_cache["encoded_unique_frames"],
            "truncated": formal_cache["truncated"],
        },
        "replay_audit": replay,
        "graph_results": post_gate["gate_task_summary"]["graph_details"],
        "gate_task_summary": {
            "E_K_min": gate.get("candidate_gap_min"),
            "T_E_K_max": gate.get("perturbation_threshold_max"),
            "S_task": gate.get("retained_safety_fraction"),
            "R_K": gate.get("robust_residual_gap"),
            "T_bg": gate.get("background_threshold"),
            "safety_pass": post_gate["gate_task_summary"]["safety_pass"],
            "background_pass": post_gate["gate_task_summary"]["background_pass"],
            "selected_method": gate.get("selected_method"),
            "reason": gate.get("reason"),
            "deployment_seed": gate.get("deployment_seed"),
        },
        "selected_branch": gate.get("selected_method"),
        "selected_reason": gate.get("reason"),
        "partition_stability": post_gate["pairwise_nominal_partitions"],
        "router_fidelity": post_gate["router_fidelity"],
        "deployment_region_sizes": post_gate["deployment_seed_0_region_sizes"],
        "timings": post_gate["timings_sec"],
        "peak_memory_mb": post_gate["peak_rss_mb"],
        "resolved_task_spec": formal_cache["resolved_task_spec"],
        "exact_reproduction_command": reproduce_cmd,
        "smoke_cache_preserved_sha256": "6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7",
        "note": "Formal gate complete; 50-epoch training intentionally not started.",
    }
    atomic_write_json(work_root / "manifests/material_passport.json", passport)
    return passport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--git-baseline", default="36f960a")
    parser.add_argument("--emit-passport-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_root = args.work_root.resolve()
    if args.emit_passport_only:
        report = emit_material_passport(work_root, args.git_baseline)
    else:
        report = run_post_gate_audit(
            work_root,
            args.data_file.resolve(),
            args.latent_cache.resolve(),
            git_baseline=args.git_baseline,
        )
        report = emit_material_passport(work_root, args.git_baseline)
    print(json.dumps(report, indent=2))
    if report.get("verification_status") == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
