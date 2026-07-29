#!/usr/bin/env python3
"""Check the reusable LAP spectral primitives against the migrated reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "tworoom"))

from lap.partition.spectral import (  # noqa: E402
    build_self_tuned_graph,
    select_k_from_laplacian_eigenvalues,
    spectral_labels,
)
from latent_landmark_spectral import (  # noqa: E402
    build_self_tuned_graph as reference_build_graph,
    select_k_from_laplacian_eigenvalues as reference_select_k,
    spectral_labels as reference_spectral_labels,
)


def main() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(64, 12)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = vectors @ vectors.T
    np.fill_diagonal(scores, -np.inf)
    neighbors = np.argsort(-scores, axis=1)[:, :8]
    similarities = np.take_along_axis(scores, neighbors, axis=1)

    graph, graph_meta = build_self_tuned_graph(neighbors, similarities, 8)
    reference_graph, reference_meta = reference_build_graph(neighbors, similarities, 8)
    if not np.allclose(graph.toarray(), reference_graph.toarray(), atol=0, rtol=0):
        raise AssertionError("extracted graph differs from the reference implementation")
    if graph_meta != reference_meta:
        raise AssertionError("extracted graph metadata differs from the reference")

    output = spectral_labels(graph, num_clusters=3, seed=7)
    reference = reference_spectral_labels(
        graph,
        num_clusters=3,
        seed=7,
        eig_tol=1e-6,
        eig_maxiter=10_000,
        spectral_n_init=20,
        cpu_threads=4,
    )
    for actual, expected in zip(output[:3], reference[:3]):
        if not np.allclose(actual, expected, atol=0, rtol=0):
            raise AssertionError("extracted spectral result differs from the reference")

    eigenvalues = output[2]
    if len(eigenvalues) >= 5:
        if select_k_from_laplacian_eigenvalues(
            eigenvalues, k_min=2, k_max=4
        ) != reference_select_k(eigenvalues, k_min=2, k_max=4):
            raise AssertionError("extracted eigengap rule differs from the reference")
    print("spectral extraction: exact match")


if __name__ == "__main__":
    main()
