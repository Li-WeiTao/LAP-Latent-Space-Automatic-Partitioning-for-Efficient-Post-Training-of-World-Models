"""Shared helpers for gate-only sensitivity audits (no predictor training)."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from lap.partition.gate import (
    SpectralGateConfig,
    SpectralGateResult,
    evaluate_spectral_gate_spectra,
)
from lap.partition.landmark import (
    NeighborSearch,
    _sample_landmarks,
    _zscore_l2,
    exact_cosine_neighbors_cpu,
)
from lap.partition.spectral import build_self_tuned_graph, l2_normalize_rows, spectral_labels

CACHE_SCHEMA_VERSION = 2
PREPROCESSING_VERSION = "zscore_l2_v1"
SOURCE_CODE_ID = "gate_audit_lib_v2"
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
NON_K_FACTORS = frozenset({"rho", "c_pert", "c_bg", "M", "B", "kNN"})


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


@dataclass(frozen=True)
class SpectrumCacheIdentity:
    schema_version: int
    latent_cache_sha256: str
    group_ids_hash: str
    model: str
    task: str
    num_landmarks: int
    landmark_seed: int
    knn: int
    eigenvalue_count: int
    eig_tol: float
    eig_maxiter: int
    preprocessing_version: str
    source_code_id: str

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


SpectraByM = dict[int, dict[int, dict[int, np.ndarray]]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_group_ids(group_ids: np.ndarray | None) -> str:
    if group_ids is None:
        return sha256_bytes(b"none")
    arr = np.asarray(group_ids, dtype=np.int64)
    return sha256_bytes(arr.tobytes())


AUDIT_SOURCE_REL_PATHS = (
    "experiments/control_matrix/gate_audit_lib.py",
    "experiments/control_matrix/gate_sensitivity_audit.py",
    "lap/partition/gate.py",
    "lap/partition/landmark.py",
)


def git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    # marked_final pins audit source at HEAD; ignore tracked result artifacts.
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(
            _run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    *AUDIT_SOURCE_REL_PATHS,
                ]
            )
        ),
    }


def audit_source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        rel_path: sha256_file(repo_root / rel_path)
        for rel_path in AUDIT_SOURCE_REL_PATHS
        if (repo_root / rel_path).is_file()
    }


def config_hash(config: SpectralGateConfig) -> str:
    return sha256_bytes(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out: list[str] = []
    for ch in str(text):
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _resolve_existing_path(repo_root: Path, raw: str) -> Path | None:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    alt = repo_root / raw
    if alt.is_file():
        return alt.resolve()
    swapped = raw.replace(
        "LAP-Latent-Space-Automatic-Partitioning-for-Efficient-Post-Training-of-World-Models",
        repo_root.name,
    )
    for probe in (Path(swapped), repo_root / swapped):
        if probe.is_file():
            return probe.resolve()
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

    return ResolvedPairInputs(
        spec=spec,
        manifest_path=manifest_path,
        manifest=manifest,
        latent_cache=latent_cache,  # type: ignore[arg-type]
        data_file=data_file,  # type: ignore[arg-type]
        baseline_config=config_from_manifest_dict(config_raw),
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
    scenarios: list[OATScenario] = [
        OATScenario("baseline", "baseline", baseline, True, False, False)
    ]

    for k in (2, 3, 4, 5):
        cfg = replace(baseline, num_regions=k)
        scenarios.append(OATScenario("K", str(k), cfg, k == baseline.num_regions, False, False))

    for rho in (0.4, 0.5, 0.6):
        cfg = replace(baseline, retention_threshold=rho)
        scenarios.append(
            OATScenario("rho", f"{rho:.1f}", cfg, abs(rho - baseline.retention_threshold) < 1e-12, False, False)
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
            OATScenario("M", str(m), cfg, m == baseline.num_landmarks, True, m != baseline.num_landmarks)
        )

    for b, seeds in B_SEED_PREFIXES.items():
        cfg = replace(baseline, diagnostic_seeds=seeds)
        scenarios.append(
            OATScenario("B", str(b), cfg, seeds == baseline.diagnostic_seeds, True, False)
        )

    for center, perturb in knn_center_sweep(baseline.nominal_knn):
        cfg = replace(baseline, nominal_knn=center, perturb_knn=perturb)
        is_base = center == baseline.nominal_knn and perturb == baseline.perturb_knn
        scenarios.append(
            OATScenario("kNN", f"center={center},perturb={perturb}", cfg, is_base, True, not is_base)
        )

    return scenarios


def collect_minimal_spectrum_requests(
    scenarios: Iterable[OATScenario],
    baseline: SpectralGateConfig,
) -> set[tuple[int, int, int]]:
    """Return unique (M, seed, knn) triples, avoiding unnecessary eigensolves."""

    baseline_m = baseline.num_landmarks
    requests: set[tuple[int, int, int]] = set()
    for scenario in scenarios:
        if scenario.varied_factor in {"rho", "c_pert", "c_bg", "K"}:
            continue
        cfg = scenario.config
        seeds = set(cfg.diagnostic_seeds)
        if cfg.num_landmarks == baseline_m:
            seeds.add(baseline.deployment_seed)
            if scenario.varied_factor in {"baseline", "B"} or scenario.is_baseline:
                seeds.update(range(10))
        for seed in seeds:
            for knn in cfg.graph_knn_values:
                requests.add((cfg.num_landmarks, seed, knn))
    return requests


def scenario_spectra_bank(spectra_by_m: SpectraByM, config: SpectralGateConfig) -> dict[int, dict[int, np.ndarray]]:
    m = config.num_landmarks
    if m not in spectra_by_m:
        raise KeyError(f"missing spectra bank for M={m}")
    bank = spectra_by_m[m]
    return {
        seed: {knn: bank[seed][knn] for knn in config.graph_knn_values}
        for seed in config.diagnostic_seeds
    }


def baseline_draw_bank(
    spectra_by_m: SpectraByM, baseline: SpectralGateConfig
) -> dict[int, dict[int, np.ndarray]]:
    m = baseline.num_landmarks
    bank = spectra_by_m[m]
    return {
        seed: {knn: bank[seed][knn] for knn in baseline.graph_knn_values}
        for seed in range(10)
        if seed in bank
    }


class VersionedSpectrumCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.eigensolves = 0

    def _paths(self, digest: str) -> tuple[Path, Path]:
        return self.root / f"{digest}.npz", self.root / f"{digest}.meta.json"

    def _read(self, identity: SpectrumCacheIdentity) -> np.ndarray | None:
        digest = identity.digest()
        npz_path, meta_path = self._paths(digest)
        if not npz_path.is_file() or not meta_path.is_file():
            self.misses += 1
            return None
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.misses += 1
            return None
        if stored != identity.to_metadata():
            self.misses += 1
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            values = np.asarray(data["eigenvalues"], dtype=np.float64)
        if values.shape[0] != identity.eigenvalue_count:
            self.misses += 1
            return None
        self.hits += 1
        return values

    def _write(self, identity: SpectrumCacheIdentity, eigenvalues: np.ndarray) -> None:
        digest = identity.digest()
        npz_path, meta_path = self._paths(digest)
        tmp_npz = npz_path.with_name(f"{digest}.writing.npz")
        tmp_meta = meta_path.with_name(f"{digest}.writing.meta.json")
        np.savez_compressed(tmp_npz, eigenvalues=np.asarray(eigenvalues, dtype=np.float64))
        tmp_meta.write_text(json.dumps(identity.to_metadata(), indent=2) + "\n", encoding="utf-8")
        tmp_npz.replace(npz_path)
        tmp_meta.replace(meta_path)

    def get_or_compute(
        self,
        identity: SpectrumCacheIdentity,
        compute_fn,
    ) -> tuple[np.ndarray, bool]:
        cached = self._read(identity)
        if cached is not None:
            return cached, True
        values = compute_fn()
        self.eigensolves += 1
        self._write(identity, values)
        return values, False


class NeighborDrawCache:
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
            landmark_indices = _sample_landmarks(len(transformed), num_landmarks, seed, group_ids)
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
) -> np.ndarray:
    max_k = max_k if max_k is not None else knn
    if neighbor_cache is None:
        landmark_indices = _sample_landmarks(len(transformed), num_landmarks, seed, group_ids)
        landmarks = np.ascontiguousarray(transformed[landmark_indices], dtype=np.float32)
        neighbors, similarities, _ = exact_cosine_neighbors_cpu(landmarks, max_k)
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


def build_spectra_by_m(
    cache: VersionedSpectrumCache,
    *,
    identity_base: dict[str, Any],
    transformed: np.ndarray,
    group_ids: np.ndarray | None,
    baseline_config: SpectralGateConfig,
    requests: set[tuple[int, int, int]],
    neighbor_cache: NeighborDrawCache,
    max_k: int,
) -> SpectraByM:
    spectra_by_m: SpectraByM = {}
    grouped: dict[int, dict[int, set[int]]] = {}
    for m, seed, knn in sorted(requests):
        grouped.setdefault(m, {}).setdefault(seed, set()).add(knn)

    for m, seed_map in grouped.items():
        for seed, knns in sorted(seed_map.items()):
            for knn in sorted(knns):
                identity = SpectrumCacheIdentity(
                    schema_version=CACHE_SCHEMA_VERSION,
                    latent_cache_sha256=identity_base["latent_cache_sha256"],
                    group_ids_hash=identity_base["group_ids_hash"],
                    model=identity_base["model"],
                    task=identity_base["task"],
                    num_landmarks=m,
                    landmark_seed=seed,
                    knn=knn,
                    eigenvalue_count=MAX_EIGENVALUES,
                    eig_tol=baseline_config.eig_tol,
                    eig_maxiter=baseline_config.eig_maxiter,
                    preprocessing_version=PREPROCESSING_VERSION,
                    source_code_id=SOURCE_CODE_ID,
                )

                def _compute(
                    m=m,
                    seed=seed,
                    knn=knn,
                ) -> np.ndarray:
                    return compute_landmark_spectrum(
                        transformed,
                        group_ids=group_ids,
                        num_landmarks=m,
                        seed=seed,
                        knn=knn,
                        count=MAX_EIGENVALUES,
                        eig_seed=seed * 1000 + knn,
                        config=baseline_config,
                        neighbor_cache=neighbor_cache,
                        max_k=max_k,
                    )

                values, _ = cache.get_or_compute(identity, _compute)
                spectra_by_m.setdefault(m, {}).setdefault(seed, {})[knn] = values
    return spectra_by_m


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


def decision_agreement_rows(
    oat_rows: list[dict[str, Any]],
    scenarios: list[OATScenario],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_by_key = {
        (row["model"], row["task"]): row["decision"]
        for row in oat_rows
        if row["varied_factor"] == "baseline"
    }
    scenario_lookup = {(s.varied_factor, s.varied_value): s for s in scenarios}
    by_pair_factor: dict[tuple[str, str, str], list[dict]] = {}
    for row in oat_rows:
        by_pair_factor.setdefault((row["model"], row["task"], row["varied_factor"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    non_k_agree = 0
    non_k_total = 0
    for (model, task, factor), rows in sorted(by_pair_factor.items()):
        if factor in {"baseline", "K"}:
            continue
        baseline_decision = baseline_by_key[(model, task)]
        alts = [
            row
            for row in rows
            if (row["varied_factor"], row["varied_value"]) in scenario_lookup
            and not scenario_lookup[(row["varied_factor"], row["varied_value"])].is_baseline
        ]
        if not alts:
            continue
        matches = sum(1 for row in alts if row["decision"] == baseline_decision)
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "varied_factor": factor,
                "n_alternatives": len(alts),
                "n_agree": matches,
                "agreement_rate": matches / len(alts),
                "baseline_decision": baseline_decision,
            }
        )
        non_k_agree += matches
        non_k_total += len(alts)

    overall = {
        "non_k_overall_agreement_rate": None if non_k_total == 0 else non_k_agree / non_k_total,
        "non_k_overall_agreement": f"{non_k_agree}/{non_k_total}",
    }
    return summary_rows, overall


def k_behavior_rows(oat_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in oat_rows:
        if row["varied_factor"] != "K" or row["varied_value"] == "baseline":
            continue
        rows.append(
            {
                "model": row["model"],
                "task": row["task"],
                "K": row["K"],
                "decision": row["decision"],
                "reason": row["reason"],
                "safety_margin": row["safety_margin"],
                "prominence_margin": row["prominence_margin"],
                "agreement_with_baseline_K": row["agreement_with_baseline"],
            }
        )
    return rows


def collect_non_k_boundary_cases(oat_rows: list[dict[str, Any]]) -> list[str]:
    cases: list[str] = []
    for row in oat_rows:
        if row["varied_factor"] == "baseline" or row["varied_factor"] == "K":
            continue
        if row["varied_factor"] not in NON_K_FACTORS:
            continue
        if not row["agreement_with_baseline"]:
            cases.append(
                f"{row['model']}/{row['task']} {row['varied_factor']}={row['varied_value']} "
                f"flipped to {row['decision']} (safety_margin={row['safety_margin']}, "
                f"prominence_margin={row['prominence_margin']})"
            )
    return cases


def draw_pass_rows(
    *,
    model: str,
    task: str,
    draw_bank: dict[int, dict[int, np.ndarray]],
    baseline: SpectralGateConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    universe = tuple(range(10))
    for b in (3, 5, 10):
        subsets = enumerate_seed_subsets(
            universe=universe, subset_size=b, required_seed=baseline.deployment_seed
        )
        passes = 0
        safety_margins: list[float] = []
        prominence_margins: list[float] = []
        for seeds in subsets:
            cfg = replace(baseline, diagnostic_seeds=seeds)
            result = evaluate_config(cfg, draw_bank)
            margins = margins_from_result(result)
            if result.use_partition:
                passes += 1
            if margins["safety_margin"] is not None:
                safety_margins.append(float(margins["safety_margin"]))
            if margins["prominence_margin"] is not None:
                prominence_margins.append(float(margins["prominence_margin"]))
        rows.append(
            {
                "model": model,
                "task": task,
                "B": b,
                "n_subsets": len(subsets),
                "pass_frequency": passes / len(subsets) if subsets else 0.0,
                "min_safety_margin": min(safety_margins) if safety_margins else None,
                "median_safety_margin": float(np.median(safety_margins)) if safety_margins else None,
                "min_prominence_margin": min(prominence_margins) if prominence_margins else None,
                "median_prominence_margin": float(np.median(prominence_margins))
                if prominence_margins
                else None,
                "note": "draw-subset sensitivity diagnostic; baseline-M bank only",
            }
        )
    return rows


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


def landmark_indices_for_config(
    n_samples: int,
    config: SpectralGateConfig,
    group_ids: np.ndarray | None,
) -> np.ndarray:
    return _sample_landmarks(
        n_samples,
        config.num_landmarks,
        config.deployment_seed,
        group_ids,
    )


def choose_held_out_audit_rows(
    n_samples: int,
    *,
    excluded_indices: set[int],
    size: int,
    seed: int,
) -> np.ndarray:
    candidates = np.array(
        [idx for idx in range(n_samples) if idx not in excluded_indices],
        dtype=np.int64,
    )
    if len(candidates) < size:
        raise RuntimeError("not enough held-out samples for audit subset")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(candidates), size=size, replace=False)
    return candidates[pick]


def build_excluded_landmark_indices(
    n_samples: int,
    baseline: SpectralGateConfig,
    scenarios: Iterable[OATScenario],
    group_ids: np.ndarray | None,
) -> set[int]:
    excluded: set[int] = set()
    excluded.update(
        int(v)
        for v in landmark_indices_for_config(n_samples, baseline, group_ids).tolist()
    )
    for scenario in scenarios:
        if scenario.needs_partition_ari and not scenario.is_baseline:
            excluded.update(
                int(v)
                for v in landmark_indices_for_config(
                    n_samples, scenario.config, group_ids
                ).tolist()
            )
    return excluded


def partition_stability_summaries(partition_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in partition_rows:
        key = (row["model"], row["task"], row["varied_factor"])
        grouped.setdefault(key, []).append(float(row["adjusted_rand_index"]))
    summaries: list[dict[str, object]] = []
    for (model, task, factor), aris in sorted(grouped.items()):
        summaries.append(
            {
                "model": model,
                "task": task,
                "varied_factor": factor,
                "comparison_count": len(aris),
                "mean_ari": float(np.mean(aris)),
                "min_ari": float(np.min(aris)),
            }
        )
    return summaries


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
