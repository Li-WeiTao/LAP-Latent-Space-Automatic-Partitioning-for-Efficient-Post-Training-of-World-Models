"""Shared helpers for gate-only sensitivity audits (no predictor training)."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from lap.partition.gate import (
    SpectralDegeneracyGate,
    SpectralGateConfig,
    SpectralGateResult,
    evaluate_spectral_gate_spectra,
)
from lap.partition.landmark import (
    LandmarkSpectralConfig,
    NeighborSearch,
    _sample_landmarks,
    _zscore_l2,
    exact_cosine_neighbors_cpu,
)
from lap.partition.spectral import build_self_tuned_graph, l2_normalize_rows, spectral_labels

MAX_K = 5
MAX_BACKGROUND_GAP = 10
MAX_EIGENVALUES = MAX_K + MAX_BACKGROUND_GAP + 1
B_SEED_PREFIXES: dict[int, tuple[int, ...]] = {
    3: (0, 1, 2),
    5: (0, 1, 2, 3, 4),
    10: tuple(range(10)),
}
AUDIT_SAMPLE_SEED = 20_260_812
AUDIT_SAMPLE_SIZE = 20_000
PROTOTYPES_PER_CLUSTER = 16


@dataclass(frozen=True)
class PairSpec:
    model: str
    task: str
    manifest_rel: str

    @property
    def key(self) -> str:
        return f"{self.model}_{self.task}"


PAIR_SPECS: tuple[PairSpec, ...] = (
    PairSpec("lewm", "tworoom", "experiments/tworoom/results/auto_gate_complete_k3/auto/partition/manifest.json"),
    PairSpec("lewm", "pusht", "experiments/pusht/results/auto_gate_complete_k3/auto/partition/manifest.json"),
    PairSpec("lewm", "reacher", "experiments/reacher/results/auto_gate_complete_k3/auto/partition/manifest.json"),
    PairSpec("lewm", "cube", "experiments/cube/results/auto_gate_complete_k3/auto/partition/manifest.json"),
    PairSpec("subjepa", "tworoom", "experiments/tworoom/subjepa/formal/gate/partition/manifest.json"),
    PairSpec("subjepa", "pusht", "experiments/pusht/subjepa/formal/gate/partition/manifest.json"),
    PairSpec("subjepa", "reacher", "experiments/reacher/subjepa/formal/gate/partition/manifest.json"),
    PairSpec("subjepa", "cube", "experiments/cube/subjepa/formal/gate/partition/manifest.json"),
)


@dataclass(frozen=True)
class OATScenario:
    varied_factor: str
    varied_value: str
    config: SpectralGateConfig
    is_baseline: bool
    needs_new_spectra: bool
    needs_partition_ari: bool


@dataclass(frozen=True)
class ResolvedPairInputs:
    spec: PairSpec
    manifest_path: Path
    manifest: dict[str, Any]
    latent_cache: Path
    data_file: Path
    baseline_config: SpectralGateConfig
    manifest_gate: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _resolve_existing_path(repo_root: Path, raw: str) -> Path | None:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    alt = repo_root / raw
    if alt.is_file():
        return alt.resolve()
    # Manifests may reference the sibling repo name.
    swapped = raw.replace(
        "LAP-Latent-Space-Automatic-Partitioning-for-Efficient-Post-Training-of-World-Models",
        repo_root.name,
    )
    alt2 = Path(swapped)
    if alt2.is_file():
        return alt2.resolve()
    alt3 = repo_root / swapped
    if alt3.is_file():
        return alt3.resolve()
    return None


def config_from_manifest_dict(raw: dict[str, Any]) -> SpectralGateConfig:
    return SpectralGateConfig(
        num_regions=int(raw["num_regions"]),
        num_landmarks=int(raw["num_landmarks"]),
        nominal_knn=int(raw["nominal_knn"]),
        perturb_knn=tuple(int(v) for v in raw["perturb_knn"]),
        diagnostic_seeds=tuple(int(v) for v in raw["diagnostic_seeds"]),
        deployment_seed=int(raw["deployment_seed"]),
        perturbation_multiplier=float(raw["perturbation_multiplier"]),
        retention_threshold=float(raw["retention_threshold"]),
        background_gap_count=int(raw["background_gap_count"]),
        background_mad_multiplier=float(raw["background_mad_multiplier"]),
        epsilon=float(raw.get("epsilon", 1e-8)),
        eig_tol=float(raw.get("eig_tol", 1e-4)),
        eig_maxiter=int(raw.get("eig_maxiter", 20_000)),
        cpu_threads=int(raw.get("cpu_threads", 4)),
    )


def resolve_pair_inputs(repo_root: Path, spec: PairSpec) -> tuple[ResolvedPairInputs | None, list[str]]:
    issues: list[str] = []
    manifest_path = (repo_root / spec.manifest_rel).resolve()
    if not manifest_path.is_file():
        issues.append(f"missing manifest: {manifest_path}")
        return None, issues
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = manifest.get("method_metadata", {}).get("automatic_gate", {})
    if not gate:
        issues.append("manifest missing method_metadata.automatic_gate")
        return None, issues
    config_raw = gate.get("configuration")
    if not config_raw:
        issues.append("manifest missing automatic_gate.configuration")
        return None, issues

    cache_raw = (
        manifest.get("cache_stats", {}).get("cache")
        or manifest.get("latent_cache")
        or config_raw.get("latent_cache")
    )
    if not cache_raw:
        issues.append("manifest missing latent cache path")
        return None, issues
    latent_cache = _resolve_existing_path(repo_root, str(cache_raw))
    if latent_cache is None:
        issues.append(f"missing latent cache: {cache_raw}")

    data_raw = manifest.get("data_file")
    if not data_raw:
        issues.append("manifest missing data_file")
        return None, issues
    data_file = _resolve_existing_path(repo_root, str(data_raw))
    if data_file is None:
        issues.append(f"missing data file: {data_raw}")

    if issues:
        return None, issues

    baseline_config = config_from_manifest_dict(config_raw)
    return ResolvedPairInputs(
        spec=spec,
        manifest_path=manifest_path,
        manifest=manifest,
        latent_cache=latent_cache,  # type: ignore[arg-type]
        data_file=data_file,  # type: ignore[arg-type]
        baseline_config=baseline_config,
        manifest_gate=gate,
    ), []


def safety_margin(result: SpectralGateResult) -> float | None:
    if result.retained_safety_fraction is None:
        return None
    return float(result.retained_safety_fraction - result.configuration["retention_threshold"])


def prominence_margin(result: SpectralGateResult) -> float | None:
    if result.robust_residual_gap is None:
        return None
    return float(result.robust_residual_gap - result.background_threshold)


def margins_from_result(result: SpectralGateResult) -> dict[str, Any]:
    rho = float(result.configuration["retention_threshold"])
    s_val = result.retained_safety_fraction
    r_val = result.robust_residual_gap
    t_bg = float(result.background_threshold)
    return {
        "S": s_val,
        "rho": rho,
        "R_K": r_val,
        "T_bg": t_bg,
        "safety_margin": None if s_val is None else float(s_val - rho),
        "prominence_margin": None if r_val is None else float(r_val - t_bg),
        "safety_pass": None if s_val is None else bool(s_val >= rho),
        "background_pass": None if r_val is None else bool(r_val > t_bg),
    }


def compare_to_manifest(
    result: SpectralGateResult,
    manifest_gate: dict[str, Any],
    *,
    atol: float = 5e-4,
) -> list[str]:
    issues: list[str] = []
    if result.selected_method != manifest_gate.get("selected_method"):
        issues.append(
            f"decision mismatch: got {result.selected_method}, "
            f"expected {manifest_gate.get('selected_method')}"
        )
    if result.reason != manifest_gate.get("reason"):
        issues.append(f"reason mismatch: got {result.reason}, expected {manifest_gate.get('reason')}")
    for key in (
        "candidate_gap_min",
        "perturbation_threshold_max",
        "retained_safety_fraction",
        "robust_residual_gap",
        "background_threshold",
    ):
        got = getattr(result, key)
        expected = manifest_gate.get(key)
        if got is None or expected is None:
            continue
        if abs(float(got) - float(expected)) > atol:
            issues.append(f"{key} mismatch: got {got}, expected {expected}")
    return issues


def unique_positive_knn_triplet(k0: int) -> tuple[int, int, int]:
    center = int(k0)
    low = max(1, int(round(0.9 * center)))
    high = max(low + 1, int(round(1.1 * center)))
    values = sorted({low, center, high})
    while len(values) < 3:
        high += 1
        values = sorted(set(values) | {high})
    return tuple(values)  # type: ignore[return-value]


def knn_center_sweep(k0: int) -> list[tuple[int, tuple[int, int]]]:
    centers = unique_positive_knn_triplet(k0)
    out: list[tuple[int, tuple[int, int]]] = []
    for center in centers:
        p_low = max(1, int(round(0.9 * center)))
        p_high = max(p_low + 1, int(round(1.1 * center)))
        if p_low == center:
            p_low = max(1, center - 1)
        if p_high == center:
            p_high = center + 1
        if len({p_low, center, p_high}) < 3:
            p_high = center + 2
        out.append((center, (p_low, p_high)))
    return out


def landmark_sweep_values(baseline_m: int) -> list[int]:
    values = []
    for frac in (0.5, 0.75, 1.0):
        values.append(max(3, int(round(baseline_m * frac))))
    return sorted(set(values))


def enumerate_seed_subsets(
    *,
    universe: Iterable[int],
    subset_size: int,
    required_seed: int,
) -> list[tuple[int, ...]]:
    others = [seed for seed in sorted(set(universe)) if seed != required_seed]
    need = subset_size - 1
    if need < 0 or need > len(others):
        return []
    return [
        tuple(sorted((required_seed, *combo)))
        for combo in itertools.combinations(others, need)
    ]


def build_oat_scenarios(baseline: SpectralGateConfig) -> list[OATScenario]:
    scenarios: list[OATScenario] = []
    scenarios.append(
        OATScenario("baseline", "baseline", baseline, True, False, False)
    )

    for k in (2, 3, 4, 5):
        cfg = replace(baseline, num_regions=k)
        scenarios.append(
            OATScenario("K", str(k), cfg, k == baseline.num_regions, False, False)
        )

    for rho in (0.4, 0.5, 0.6):
        cfg = replace(baseline, retention_threshold=rho)
        scenarios.append(
            OATScenario(
                "rho",
                f"{rho:.1f}",
                cfg,
                abs(rho - baseline.retention_threshold) < 1e-12,
                False,
                False,
            )
        )

    for c_pert in (1.5, 2.0, 2.5):
        cfg = replace(baseline, perturbation_multiplier=c_pert)
        scenarios.append(
            OATScenario(
                "c_pert",
                f"{c_pert:.1f}",
                cfg,
                abs(c_pert - baseline.perturbation_multiplier) < 1e-12,
                False,
                False,
            )
        )

    for c_bg in (2.5, 3.0, 3.5):
        cfg = replace(baseline, background_mad_multiplier=c_bg)
        scenarios.append(
            OATScenario(
                "c_bg",
                f"{c_bg:.1f}",
                cfg,
                abs(c_bg - baseline.background_mad_multiplier) < 1e-12,
                False,
                False,
            )
        )

    for m in landmark_sweep_values(baseline.num_landmarks):
        cfg = replace(baseline, num_landmarks=m)
        scenarios.append(
            OATScenario(
                "M",
                str(m),
                cfg,
                m == baseline.num_landmarks,
                True,
                m != baseline.num_landmarks,
            )
        )

    for b, seeds in B_SEED_PREFIXES.items():
        cfg = replace(baseline, diagnostic_seeds=seeds)
        scenarios.append(
            OATScenario(
                "B",
                str(b),
                cfg,
                seeds == baseline.diagnostic_seeds,
                True,
                False,
            )
        )

    for center, perturb in knn_center_sweep(baseline.nominal_knn):
        cfg = replace(
            baseline,
            nominal_knn=center,
            perturb_knn=perturb,
        )
        is_base = center == baseline.nominal_knn and perturb == baseline.perturb_knn
        scenarios.append(
            OATScenario("kNN", f"center={center},perturb={perturb}", cfg, is_base, True, not is_base)
        )

    return scenarios


def spectrum_cache_key(
    *,
    cache_hash: str,
    model: str,
    task: str,
    num_landmarks: int,
    seed: int,
    knn: int,
) -> str:
    return f"{cache_hash}_{model}_{task}_M{num_landmarks}_s{seed}_k{knn}"


class SpectrumCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            return None
        with np.load(path, allow_pickle=False) as data:
            self.hits += 1
            return np.asarray(data["eigenvalues"], dtype=np.float64)

    def put(self, key: str, eigenvalues: np.ndarray) -> None:
        path = self.path_for(key)
        tmp = path.with_name(f"{path.stem}.writing.npz")
        np.savez_compressed(tmp, eigenvalues=np.asarray(eigenvalues, dtype=np.float64))
        tmp.replace(path)


class NeighborDrawCache:
    """Reuse one exact kNN search per (M, landmark seed) across kNN values."""

    def __init__(self, neighbor_search: NeighborSearch | None = None) -> None:
        self._draws: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.neighbor_search = neighbor_search or exact_cosine_neighbors_cpu

    def neighbors_for(
        self,
        transformed: np.ndarray,
        *,
        group_ids: np.ndarray | None,
        num_landmarks: int,
        seed: int,
        max_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (num_landmarks, seed)
        if key not in self._draws:
            landmark_indices = _sample_landmarks(
                len(transformed), num_landmarks, seed, group_ids
            )
            landmarks = np.ascontiguousarray(transformed[landmark_indices], dtype=np.float32)
            neighbors, similarities, _ = self.neighbor_search(landmarks, max_k)
            self._draws[key] = (neighbors, similarities)
        return self._draws[key]


def compute_landmark_spectrum(
    transformed: np.ndarray,
    *,
    group_ids: np.ndarray | None,
    num_landmarks: int,
    seed: int,
    knn: int,
    count: int,
    eig_seed: int,
    config: SpectralGateConfig,
    neighbor_cache: NeighborDrawCache | None = None,
    max_k: int | None = None,
    neighbor_search: NeighborSearch | None = None,
) -> np.ndarray:
    max_k = max_k if max_k is not None else knn
    search = neighbor_search or exact_cosine_neighbors_cpu
    if neighbor_cache is None:
        landmark_indices = _sample_landmarks(len(transformed), num_landmarks, seed, group_ids)
        landmarks = np.ascontiguousarray(transformed[landmark_indices], dtype=np.float32)
        neighbors, similarities, _ = search(landmarks, max_k)
    else:
        neighbors, similarities = neighbor_cache.neighbors_for(
            transformed,
            group_ids=group_ids,
            num_landmarks=num_landmarks,
            seed=seed,
            max_k=max_k,
        )
    graph, _ = build_self_tuned_graph(neighbors, similarities, knn)
    from lap.partition.gate import _smallest_normalized_laplacian_eigenvalues

    return _smallest_normalized_laplacian_eigenvalues(
        graph,
        count=count,
        seed=eig_seed,
        tol=config.eig_tol,
        maxiter=config.eig_maxiter,
    )


def collect_required_spectrum_requests(
    scenarios: Iterable[OATScenario],
    *,
    draw_universe_seeds: tuple[int, ...],
) -> set[tuple[int, int, int]]:
    """Return unique (M, seed, knn) requests."""

    needed: set[tuple[int, int, int]] = set()
    for scenario in scenarios:
        cfg = scenario.config
        seeds = set(cfg.diagnostic_seeds) | set(draw_universe_seeds)
        knns = cfg.graph_knn_values
        for seed in seeds:
            for knn in knns:
                needed.add((cfg.num_landmarks, seed, knn))
    return needed


def spectra_for_config(
    cache: SpectrumCache,
    *,
    cache_hash: str,
    model: str,
    task: str,
    transformed: np.ndarray,
    group_ids: np.ndarray | None,
    config: SpectralGateConfig,
    draw_universe_seeds: tuple[int, ...],
    eig_count: int = MAX_EIGENVALUES,
    neighbor_cache: NeighborDrawCache | None = None,
    all_knns: set[int] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    requests: set[tuple[int, int, int]] = set()
    for seed in set(config.diagnostic_seeds) | set(draw_universe_seeds):
        for knn in config.graph_knn_values:
            requests.add((config.num_landmarks, seed, knn))

    knn_pool = all_knns or {knn for _, _, knn in requests}
    max_k = max(knn_pool)

    out: dict[int, dict[int, np.ndarray]] = {}
    grouped: dict[tuple[int, int], set[int]] = {}
    for m, seed, knn in requests:
        grouped.setdefault((m, seed), set()).add(knn)

    for (m, seed), knns in sorted(grouped.items()):
        for knn in sorted(knns):
            key = spectrum_cache_key(
                cache_hash=cache_hash,
                model=model,
                task=task,
                num_landmarks=m,
                seed=seed,
                knn=knn,
            )
            values = cache.get(key)
            if values is None:
                values = compute_landmark_spectrum(
                    transformed,
                    group_ids=group_ids,
                    num_landmarks=m,
                    seed=seed,
                    knn=knn,
                    count=eig_count,
                    eig_seed=seed * 1000 + knn,
                    config=config,
                    neighbor_cache=neighbor_cache,
                    max_k=max_k,
                )
                cache.put(key, values)
            out.setdefault(seed, {})[knn] = values
    return out


def evaluate_config(
    config: SpectralGateConfig,
    spectra_bank: dict[int, dict[int, np.ndarray]],
) -> SpectralGateResult:
    selected = {
        seed: {knn: spectra_bank[seed][knn] for knn in config.graph_knn_values}
        for seed in config.diagnostic_seeds
    }
    return evaluate_spectral_gate_spectra(selected, config)


def result_row(
    *,
    model: str,
    task: str,
    scenario: OATScenario,
    result: SpectralGateResult,
    baseline_decision: str,
    elapsed_sec: float,
    cache_hit: bool,
) -> dict[str, Any]:
    margins = margins_from_result(result)
    decision = result.selected_method
    return {
        "model": model,
        "task": task,
        "varied_factor": scenario.varied_factor,
        "varied_value": scenario.varied_value,
        "K": int(result.configuration["num_regions"]),
        "rho": float(result.configuration["retention_threshold"]),
        "c_pert": float(result.configuration["perturbation_multiplier"]),
        "c_bg": float(result.configuration["background_mad_multiplier"]),
        "num_landmarks": int(result.configuration["num_landmarks"]),
        "B": len(result.configuration["diagnostic_seeds"]),
        "diagnostic_seeds": ",".join(str(v) for v in result.configuration["diagnostic_seeds"]),
        "nominal_knn": int(result.configuration["nominal_knn"]),
        "perturb_knn": ",".join(str(v) for v in result.configuration["perturb_knn"]),
        "candidate_gap_min": result.candidate_gap_min,
        "perturbation_threshold_max": result.perturbation_threshold_max,
        "S": margins["S"],
        "R_K": margins["R_K"],
        "T_bg": margins["T_bg"],
        "safety_margin": margins["safety_margin"],
        "prominence_margin": margins["prominence_margin"],
        "safety_pass": margins["safety_pass"],
        "background_pass": margins["background_pass"],
        "decision": decision,
        "reason": result.reason,
        "agreement_with_baseline": decision == baseline_decision,
        "elapsed_sec": elapsed_sec,
        "cache_hit": cache_hit,
    }


def hungarian_align(reference: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (reference, labels), 1)
    ref_rows, candidate_cols = linear_sum_assignment(-contingency)
    mapping = np.arange(num_clusters, dtype=np.int64)
    mapping[candidate_cols] = ref_rows
    return mapping[labels]


def fit_audit_labels(
    transformed: np.ndarray,
    audit_rows: np.ndarray,
    *,
    num_clusters: int,
    num_landmarks: int,
    knn: int,
    seed: int,
    group_ids: np.ndarray | None,
    cpu_threads: int,
) -> np.ndarray:
    landmark_indices = _sample_landmarks(len(transformed), num_landmarks, seed, group_ids)
    landmarks = np.ascontiguousarray(transformed[landmark_indices], dtype=np.float32)
    neighbors, similarities, _ = exact_cosine_neighbors_cpu(landmarks, knn)
    graph, _ = build_self_tuned_graph(neighbors, similarities, knn)
    landmark_labels, _, _, _ = spectral_labels(
        graph,
        num_clusters=num_clusters,
        seed=seed,
        eig_tol=1e-4,
        eig_maxiter=10_000,
        spectral_n_init=20,
        cpu_threads=cpu_threads,
    )
    prototypes: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for region_id in range(num_clusters):
        region = landmarks[landmark_labels == region_id]
        model = KMeans(
            n_clusters=PROTOTYPES_PER_CLUSTER,
            init="k-means++",
            n_init=5,
            random_state=seed * 1000 + region_id,
            algorithm="lloyd",
        ).fit(region)
        prototypes.append(l2_normalize_rows(model.cluster_centers_))
        owners.append(np.full(PROTOTYPES_PER_CLUSTER, region_id, dtype=np.int64))
    prototype_matrix = np.concatenate(prototypes).astype(np.float32)
    prototype_owner = np.concatenate(owners)
    audit_transformed = transformed[audit_rows]
    nearest = np.argmax(audit_transformed @ prototype_matrix.T, axis=1)
    return prototype_owner[nearest].astype(np.int64)


def choose_audit_rows(n_samples: int, landmark_indices: np.ndarray, *, size: int, seed: int) -> np.ndarray:
    landmark_set = set(int(v) for v in landmark_indices.tolist())
    candidates = np.array([idx for idx in range(n_samples) if idx not in landmark_set], dtype=np.int64)
    if len(candidates) < size:
        raise RuntimeError("not enough non-landmark samples for audit subset")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(candidates), size=size, replace=False)
    return candidates[pick]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
