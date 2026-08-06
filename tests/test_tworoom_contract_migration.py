from __future__ import annotations

import numpy as np
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TwoRoomMigrationContractTest(unittest.TestCase):
    def test_transition_starts_accept_official_episode_idx_name(self):
        import h5py
        from predictor_rule_drift import valid_transition_starts

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official_pusht_schema.h5"
            with h5py.File(path, "w") as handle:
                handle["state"] = np.zeros((10, 2), dtype=np.float32)
                handle["action"] = np.zeros((10, 2), dtype=np.float32)
                handle["pixels"] = np.zeros((10, 1, 1, 3), dtype=np.uint8)
                handle["episode_idx"] = np.repeat(
                    np.arange(2, dtype=np.int64), 5
                )
                handle["step_idx"] = np.tile(
                    np.arange(5, dtype=np.int64), 2
                )
                handle["ep_len"] = np.full(2, 5, dtype=np.int32)
            with h5py.File(path, "r") as handle:
                starts = valid_transition_starts(
                    handle,
                    SimpleNamespace(pixel_key="pixels"),
                    "state",
                    seq_len=2,
                    step_stride=1,
                    max_samples=0,
                    seed=0,
                )
        np.testing.assert_array_equal(
            starts, np.asarray([0, 1, 2, 3, 5, 6, 7, 8])
        )

    def test_spectral_entrypoint_resolves_to_lap_core(self):
        import latent_landmark_spectral as experiment
        from lap.partition import spectral as core

        self.assertIs(experiment.build_self_tuned_graph, core.build_self_tuned_graph)
        self.assertIs(
            experiment.select_k_from_laplacian_eigenvalues,
            core.select_k_from_laplacian_eigenvalues,
        )
        self.assertIs(experiment.spectral_labels, core.spectral_labels)
        self.assertIs(experiment.l2_normalize_rows, core.l2_normalize_rows)

    def test_training_entrypoint_resolves_to_lewm_backend(self):
        import trajectory
        from backends.lewm import finetuning

        self.assertIs(
            trajectory.train_region_predictor,
            finetuning.train_region_predictor,
        )
        self.assertIs(trajectory.eval_predictor_loss, finetuning.eval_predictor_loss)
        self.assertIs(trajectory.save_region_predictor, finetuning.save_region_predictor)
        self.assertIs(trajectory.TrainConfig, finetuning.LeWMTrainConfig)

    def test_numpy_and_torch_voronoi_contracts_match(self):
        import torch
        from backends.lewm.routing import route_voronoi_torch
        from lap.partition import PartitionArtifact
        from lap.routing import VoronoiRouter

        rng = np.random.default_rng(17)
        latents = rng.normal(size=(31, 7)).astype(np.float32)
        prototypes = rng.normal(size=(8, 7)).astype(np.float32)
        owners = np.asarray([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
        mean = rng.normal(size=7).astype(np.float32)
        scale = rng.uniform(0.3, 2.0, size=7).astype(np.float32)
        artifact = PartitionArtifact(prototypes, owners, mean, scale, {})

        expected = VoronoiRouter(artifact).route(latents)
        actual = route_voronoi_torch(
            torch.from_numpy(latents),
            torch.from_numpy(prototypes),
            torch.from_numpy(owners),
            mean=torch.from_numpy(mean),
            scale=torch.from_numpy(scale),
            eps=0.0,
            spherical=True,
        ).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_resolve_torch_device_auto_falls_back_to_cpu(self):
        import trajectory
        from unittest import mock

        with mock.patch.object(trajectory.torch.cuda, "is_available", return_value=False):
            self.assertEqual(str(trajectory.resolve_torch_device("auto")), "cpu")
        with mock.patch.object(trajectory.torch.cuda, "is_available", return_value=True):
            self.assertEqual(str(trajectory.resolve_torch_device("auto")), "cuda")
        self.assertEqual(str(trajectory.resolve_torch_device("cpu")), "cpu")
