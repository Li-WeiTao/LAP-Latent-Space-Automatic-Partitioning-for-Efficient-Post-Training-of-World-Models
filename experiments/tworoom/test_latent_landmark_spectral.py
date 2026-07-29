#!/usr/bin/env python3
"""Fast contract tests for landmark spectral artifacts and prototype routing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from jepa import JEPA
from latent_cluster_common import (
    load_cluster_artifact,
    load_kmeanspp_label_artifact,
    sha256_file,
    transform_zscore_l2,
)
from latent_landmark_spectral import (
    assign_prototype_owners_numpy,
    select_k_from_laplacian_eigenvalues,
)
from tworoom_success_rate_eval import LatentClusterSwitchJEPA


class TwoArgIdentity(nn.Module):
    def forward(self, emb, _act):
        return emb


def dummy_jepa() -> JEPA:
    return JEPA(
        nn.Identity(),
        TwoArgIdentity(),
        nn.Identity(),
        nn.Identity(),
        nn.Identity(),
    )


class SpectralArtifactContractTest(unittest.TestCase):
    def test_legacy_kmeanspp_identity_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "kmeanspp_R50_outer0.npz"
            np.savez_compressed(
                path,
                global_idx=np.arange(6, dtype=np.int64),
                labels=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
                centroids=np.eye(3, dtype=np.float32),
            )
            artifact = load_kmeanspp_label_artifact(path)
            np.testing.assert_array_equal(
                artifact["prototype_cluster_ids"], np.arange(3)
            )
            np.testing.assert_array_equal(
                artifact["routing_vectors"], artifact["centroids"]
            )

    def test_spectral_artifact_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "centroids.npy", np.eye(3, 4, dtype=np.float32))
            routing = np.array(
                [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0],
                 [0, 0.9, 0.1, 0], [0, 0, 1, 0], [0.1, 0, 0.9, 0]],
                dtype=np.float32,
            )
            owners = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
            np.save(root / "routing_prototypes.npy", routing)
            np.save(root / "prototype_cluster_ids.npy", owners)
            np.savez_compressed(
                root / "cluster_labels.npz",
                global_idx=np.arange(9, dtype=np.int64),
                labels=np.repeat(np.arange(3), 3),
            )
            np.savez_compressed(
                root / "zscore_params.npz",
                mu=np.zeros(4, dtype=np.float32),
                sigma=np.ones(4, dtype=np.float32),
                eps=np.float32(1e-6),
            )
            (root / "cluster_meta.json").write_text(
                json.dumps(
                    {
                        "num_clusters": 3,
                        "spherical": True,
                        "assignment_schema_version": 2,
                        "transition_label_offset_steps": 10,
                    }
                )
            )
            artifact = load_cluster_artifact(root)
            np.testing.assert_array_equal(artifact["centroids"], routing)
            np.testing.assert_array_equal(
                artifact["prototype_cluster_ids"], owners
            )
            self.assertIsNotNone(artifact["zscore"])
            self.assertEqual(artifact["meta"]["num_clusters"], 3)
            self.assertEqual(
                artifact["meta"]["transition_label_offset_steps"], 10
            )

    def test_invalid_prototype_owner_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "centroids.npy", np.eye(3, dtype=np.float32))
            np.save(root / "routing_prototypes.npy", np.eye(3, dtype=np.float32))
            np.save(
                root / "prototype_cluster_ids.npy",
                np.array([0, 1, 3], dtype=np.int64),
            )
            np.savez_compressed(
                root / "cluster_labels.npz",
                global_idx=np.arange(3, dtype=np.int64),
                labels=np.arange(3, dtype=np.int64),
            )
            (root / "cluster_meta.json").write_text(
                json.dumps({"num_clusters": 3})
            )
            with self.assertRaisesRegex(ValueError, "prototype owners out of range"):
                load_cluster_artifact(root)

    def test_zscore_space_without_params_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "centroids.npy", np.eye(3, dtype=np.float32))
            np.save(root / "routing_prototypes.npy", np.eye(3, dtype=np.float32))
            np.save(
                root / "prototype_cluster_ids.npy", np.arange(3, dtype=np.int64)
            )
            np.savez_compressed(
                root / "cluster_labels.npz",
                global_idx=np.arange(3, dtype=np.int64),
                labels=np.arange(3, dtype=np.int64),
            )
            (root / "cluster_meta.json").write_text(
                json.dumps(
                    {
                        "num_clusters": 3,
                        "method": "zscore_l2_landmark_spectral_prototype",
                        "preprocess": "zscore_l2",
                    }
                )
            )
            with self.assertRaisesRegex(FileNotFoundError, "Z-score space"):
                load_cluster_artifact(root)

    def test_numpy_and_torch_prototype_owner_mapping_match(self) -> None:
        rng = np.random.default_rng(20260716)
        raw = rng.normal(size=(128, 4)).astype(np.float32)
        mu = rng.normal(size=4).astype(np.float32)
        sigma = rng.uniform(0.5, 2.0, size=4).astype(np.float32)
        transformed = transform_zscore_l2(raw, mu, sigma, 1e-6)
        prototypes = transformed[[3, 7, 11, 19, 23, 29]]
        owners = np.array([2, 0, 2, 1, 0, 1], dtype=np.int64)
        expected = assign_prototype_owners_numpy(
            transformed, prototypes, owners, chunk_size=17
        )

        base = dummy_jepa()
        clusters = {f"cluster{k}": dummy_jepa() for k in range(3)}
        model = LatentClusterSwitchJEPA(
            base,
            clusters,
            prototypes,
            prototype_cluster_ids=owners,
            spherical=True,
            zscore={"mu": mu, "sigma": sigma, "eps": 1e-6},
        )
        actual = model._assign_clusters(torch.from_numpy(raw)).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isin(actual, [0, 1, 2]).all())

    def test_mpc_routing_is_fixed_per_environment_not_shared_per_batch(self) -> None:
        base = dummy_jepa()
        clusters = {f"cluster{k}": dummy_jepa() for k in range(3)}
        prototypes = np.eye(3, dtype=np.float32)
        model = LatentClusterSwitchJEPA(
            base,
            clusters,
            prototypes,
            prototype_cluster_ids=np.arange(3),
            spherical=True,
            routing_mode="mpc",
        )

        def fake_encode(self, info):
            info["emb"] = info["pixels"].float()
            return info

        seen: list[np.ndarray] = []

        def fake_predict(self, emb, _act_emb, cluster_ids):
            seen.append(cluster_ids.detach().cpu().numpy().copy())
            return emb[:, -1:, :]

        model.encode = types.MethodType(fake_encode, model)
        model._predict_routed = types.MethodType(fake_predict, model)
        # B=2 environments, S=3 CEM candidates, H=2 history steps.
        pixels = torch.tensor(
            [
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 3,
                [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]] * 3,
            ]
        )
        action_sequence = torch.zeros(2, 3, 3, 1)
        model.rollout({"pixels": pixels}, action_sequence, history_size=2)
        expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        self.assertGreater(len(seen), 0)
        for routed in seen:
            np.testing.assert_array_equal(routed, expected)

    def test_auto_k_uses_predeclared_largest_eigengap(self) -> None:
        eigenvalues = np.array([0.0, 0.01, 0.02, 0.50, 0.51, 0.52, 0.53])
        selected, meta = select_k_from_laplacian_eigenvalues(
            eigenvalues, k_min=2, k_max=5
        )
        self.assertEqual(selected, 3)
        self.assertEqual(meta["selected_num_clusters"], 3)

    def test_recorded_output_hash_rejects_artifact_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "centroids.npy", np.eye(2, dtype=np.float32))
            np.savez_compressed(
                root / "cluster_labels.npz",
                global_idx=np.arange(4, dtype=np.int64),
                labels=np.array([0, 0, 1, 1], dtype=np.int64),
            )
            diagnostic = root / "landmark_diagnostics.npz"
            diagnostic.write_bytes(b"original diagnostics")
            (root / "cluster_meta.json").write_text(
                json.dumps(
                    {
                        "num_clusters": 2,
                        "output_file_sha256": {
                            diagnostic.name: sha256_file(diagnostic)
                        },
                        "output_file_size_bytes": {
                            diagnostic.name: diagnostic.stat().st_size
                        },
                    }
                )
            )
            diagnostic.write_bytes(b"corrupted diagnostics")
            with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
                load_cluster_artifact(root)


if __name__ == "__main__":
    unittest.main()
