from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
import torch.nn as nn

from backends.lewm.cache import LeWMLatentCache
from backends.lewm.routing import route_voronoi_torch
from experiments.control_matrix.region_risk_lib import (
    aggregate_region_metrics,
    anchor_in_same_episode,
    audit_episode_disjointness,
    collect_rollout_anchors,
    load_cache_contract,
    nested_paired_bootstrap_ci,
    one_step_losses,
    open_loop_rollout_losses,
    paired_bootstrap_ci,
    resolve_action_norm_starts,
    start_index_map,
    weighted_summary,
    wrong_expert_losses,
)
from lap.partition import PartitionArtifact


class MockPredictor(nn.Module):
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return emb + self.shift


class StepPredictor(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        out = emb.clone()
        out[:, -1] = emb[:, -1] + self.delta
        return out


class RegionConditionalRiskTest(unittest.TestCase):
    def test_wrong_expert_excludes_correct(self) -> None:
        loss_matrix = np.asarray(
            [
                [0.1, 0.4, 0.5],
                [0.2, 0.3, 0.6],
                [0.7, 0.8, 0.2],
            ],
            dtype=np.float64,
        )
        regions = np.asarray([0, 1, 2], dtype=np.int64)
        wrong_mean, wrong_best = wrong_expert_losses(loss_matrix, regions, 3)
        self.assertAlmostEqual(wrong_mean[0], np.mean([0.3, 0.6]))
        self.assertAlmostEqual(wrong_best[0], 0.4)
        self.assertAlmostEqual(wrong_mean[1], 0.4)
        self.assertAlmostEqual(wrong_best[1], 0.2)

    def test_h1_open_loop_matches_one_step(self) -> None:
        emb = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        act_emb = emb.clone()
        cache = LeWMLatentCache(
            emb,
            act_emb,
            np.asarray([0, 10], dtype=np.int64),
            route_index=0,
        )
        contract = load_cache_contract(
            cache, history_size=3, num_preds=1, frameskip=1
        )
        models = [MockPredictor(0.0), MockPredictor(1.0)]
        one_step = one_step_losses(
            models,
            emb,
            act_emb,
            history_size=3,
            num_preds=1,
            device=torch.device("cpu"),
            batch_size=2,
        )
        rollout = open_loop_rollout_losses(
            models,
            cache,
            np.asarray([0, 10], dtype=np.int64),
            horizon=1,
            contract=contract,
            start_map=start_index_map(cache.sample_ids),
            device=torch.device("cpu"),
        )
        np.testing.assert_allclose(one_step, rollout.mean_trajectory_mse, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(one_step, rollout.terminal_mse, rtol=1e-6, atol=1e-6)

    def test_open_loop_does_not_write_back_ground_truth(self) -> None:
        model = MockPredictor(100.0)
        emb = torch.zeros((2, 4, 2), dtype=torch.float32)
        emb[0, 3] = torch.tensor([9.0, 9.0])
        emb[1, 3] = torch.tensor([9.0, 9.0])
        act_emb = emb.clone()
        cache = LeWMLatentCache(
            emb,
            act_emb,
            np.asarray([0, 1], dtype=np.int64),
            route_index=0,
        )
        contract = load_cache_contract(
            cache, history_size=3, num_preds=1, frameskip=1
        )
        start_map = {0: 0, 1: 1}
        losses = open_loop_rollout_losses(
            [model],
            cache,
            np.asarray([0], dtype=np.int64),
            horizon=2,
            contract=contract,
            start_map=start_map,
            device=torch.device("cpu"),
        )
        self.assertGreater(float(losses.mean_trajectory_mse[0, 0]), 1.0)
        self.assertGreater(float(losses.terminal_mse[0, 0]), 1.0)

    def test_open_loop_models_are_independent(self) -> None:
        emb = torch.zeros((3, 4, 2), dtype=torch.float32)
        emb[0, 3] = torch.tensor([1.0, 0.0])
        emb[1, 3] = torch.tensor([2.0, 0.0])
        emb[2, 3] = torch.tensor([3.0, 0.0])
        act_emb = emb.clone()
        cache = LeWMLatentCache(
            emb,
            act_emb,
            np.asarray([0, 1, 2], dtype=np.int64),
            route_index=0,
        )
        contract = load_cache_contract(
            cache, history_size=3, num_preds=1, frameskip=1
        )
        start_map = start_index_map(cache.sample_ids)
        models = [StepPredictor(1.0), StepPredictor(10.0), StepPredictor(100.0)]
        joint = open_loop_rollout_losses(
            models,
            cache,
            np.asarray([0], dtype=np.int64),
            horizon=2,
            contract=contract,
            start_map=start_map,
            device=torch.device("cpu"),
        )
        for model_index, model in enumerate(models):
            solo = open_loop_rollout_losses(
                [model],
                cache,
                np.asarray([0], dtype=np.int64),
                horizon=2,
                contract=contract,
                start_map=start_map,
                device=torch.device("cpu"),
            )
            self.assertAlmostEqual(
                float(joint.terminal_mse[0, model_index]),
                float(solo.terminal_mse[0, 0]),
                places=6,
            )
            self.assertAlmostEqual(
                float(joint.mean_trajectory_mse[0, model_index]),
                float(solo.mean_trajectory_mse[0, 0]),
                places=6,
            )

    def test_open_loop_model_order_only_permutates_columns(self) -> None:
        emb = torch.zeros((3, 4, 2), dtype=torch.float32)
        emb[0, 3] = torch.tensor([1.0, 0.0])
        emb[1, 3] = torch.tensor([2.0, 0.0])
        emb[2, 3] = torch.tensor([3.0, 0.0])
        act_emb = emb.clone()
        cache = LeWMLatentCache(
            emb,
            act_emb,
            np.asarray([0, 1, 2], dtype=np.int64),
            route_index=0,
        )
        contract = load_cache_contract(
            cache, history_size=3, num_preds=1, frameskip=1
        )
        start_map = start_index_map(cache.sample_ids)
        models_a = [StepPredictor(1.0), StepPredictor(10.0), StepPredictor(100.0)]
        models_b = [models_a[2], models_a[0], models_a[1]]
        losses_a = open_loop_rollout_losses(
            models_a,
            cache,
            np.asarray([0], dtype=np.int64),
            horizon=2,
            contract=contract,
            start_map=start_map,
            device=torch.device("cpu"),
        )
        losses_b = open_loop_rollout_losses(
            models_b,
            cache,
            np.asarray([0], dtype=np.int64),
            horizon=2,
            contract=contract,
            start_map=start_map,
            device=torch.device("cpu"),
        )
        for perm_index, source_index in enumerate((2, 0, 1)):
            np.testing.assert_allclose(
                losses_a.terminal_mse[:, source_index],
                losses_b.terminal_mse[:, perm_index],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                losses_a.mean_trajectory_mse[:, source_index],
                losses_b.mean_trajectory_mse[:, perm_index],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_resolve_action_norm_starts_from_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starts = root / "train_starts.npy"
            np.save(starts, np.asarray([0, 5, 10], dtype=np.int64))
            cache_path = root / "cache.npz"
            np.savez_compressed(
                cache_path,
                emb=np.zeros((3, 4, 2), dtype=np.float32),
                act_emb=np.zeros((3, 4, 2), dtype=np.float32),
                region_starts=np.asarray([0, 5, 10], dtype=np.int64),
            )
            report_path = Path(f"{cache_path}.report.json")
            report_path.write_text(
                json.dumps({"selection": {"starts_source": str(starts)}}),
                encoding="utf-8",
            )
            resolved = resolve_action_norm_starts(cache_path)
            self.assertEqual(resolved, starts.resolve())

    def test_rollout_anchor_respects_episode_boundary(self) -> None:
        start_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        episode_lookup = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
        self.assertTrue(
            anchor_in_same_episode(0, 2, 1, 3, episode_lookup, start_map)
        )
        self.assertFalse(
            anchor_in_same_episode(2, 2, 1, 3, episode_lookup, start_map)
        )
        anchors = collect_rollout_anchors(
            np.asarray([0, 2, 3], dtype=np.int64),
            horizon=2,
            frameskip=1,
            history_size=3,
            start_map=start_map,
            episode_lookup=episode_lookup,
        )
        np.testing.assert_array_equal(anchors, [0])

    def test_frameskip_row_stitching(self) -> None:
        starts = np.asarray([0, 5, 10], dtype=np.int64)
        start_map = start_index_map(starts)
        self.assertEqual(start_map[5], 1)
        self.assertEqual(start_map[10], 2)

    def test_weighted_risk(self) -> None:
        rows = [
            {
                "region_weight": 0.25,
                "global_mse": 1.0,
                "correct_mse": 0.5,
                "wrong_mean_mse": 0.8,
                "wrong_best_mse": 0.7,
            },
            {
                "region_weight": 0.75,
                "global_mse": 2.0,
                "correct_mse": 1.0,
                "wrong_mean_mse": 1.5,
                "wrong_best_mse": 1.2,
            },
        ]
        summary = weighted_summary(rows, task="t", train_seed=0, horizon=1)
        self.assertAlmostEqual(summary["global_mse"], 1.75)
        self.assertAlmostEqual(summary["correct_mse"], 0.875)

    def test_paired_bootstrap_reproducible(self) -> None:
        samples = {
            "global": np.asarray([1.0, 2.0, 3.0]),
            "correct": np.asarray([0.5, 1.5, 2.5]),
            "wrong_mean": np.asarray([1.2, 2.2, 3.2]),
            "wrong_best": np.asarray([1.1, 2.1, 3.1]),
        }
        first = paired_bootstrap_ci(samples, reps=1000, seed=7)
        second = paired_bootstrap_ci(samples, reps=1000, seed=7)
        self.assertEqual(first, second)

    def test_overlap_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("ep_offset", data=np.asarray([0, 5], dtype=np.int64))
                handle.create_dataset("ep_len", data=np.asarray([5, 5], dtype=np.int64))
            with self.assertRaises(RuntimeError):
                audit_episode_disjointness(
                    data_file=path,
                    train_starts=np.asarray([0, 1], dtype=np.int64),
                    eval_starts=np.asarray([1, 2], dtype=np.int64),
                    require_disjoint=True,
                )

    def test_router_matches_deployment_contract(self) -> None:
        artifact = PartitionArtifact(
            prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            prototype_region_ids=np.asarray([0, 1], dtype=np.int64),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            metadata={},
        )
        latent = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        assigned = route_voronoi_torch(
            latent,
            torch.as_tensor(artifact.prototypes),
            torch.as_tensor(artifact.prototype_region_ids),
            mean=torch.as_tensor(artifact.mean),
            scale=torch.as_tensor(artifact.scale),
            spherical=True,
        )
        self.assertEqual(assigned.tolist(), [0, 1])

    def test_mock_predictor_manual_mse(self) -> None:
        emb = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]],
            dtype=torch.float32,
        )
        act_emb = emb.clone()
        model = MockPredictor(1.0)
        pred = model.predict(emb[:, :3], act_emb[:, :3])
        target = emb[:, -1]
        manual = float(((pred[:, -1] - target) ** 2).mean())
        auto = one_step_losses(
            [model],
            emb,
            act_emb,
            history_size=3,
            num_preds=1,
            device=torch.device("cpu"),
            batch_size=1,
        )[0, 0]
        self.assertAlmostEqual(manual, auto)


if __name__ == "__main__":
    unittest.main()
