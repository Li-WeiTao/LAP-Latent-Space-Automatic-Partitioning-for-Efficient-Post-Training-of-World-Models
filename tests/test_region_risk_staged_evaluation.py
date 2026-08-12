from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from backends.lewm.cache import LeWMLatentCache
from experiments.control_matrix.evaluate_region_conditional_risk import (
    _bootstrap_chunk_job,
    _resume_matches,
    parse_args,
    run_finalize_stage,
)
from experiments.control_matrix.region_risk_lib import (
    PUBLIC_ANALYSIS_NAME,
    PUBLIC_ANALYSIS_SHORT_NAME,
    atomic_savez_compressed,
    atomic_write_json,
    load_cache_contract,
    multi_horizon_open_loop_rollout_losses,
    nested_paired_bootstrap_draws,
    precompute_episode_summaries,
    stable_json_sha256,
    start_index_map,
)


class CountingPredictor(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta
        self.predict_calls = 0

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        self.predict_calls += 1
        result = emb.clone()
        result[:, -1] = emb[:, -1] + self.delta
        return result


def legacy_nested_draws(
    blocks: list[dict[str, np.ndarray]], *, reps: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = ("global", "correct", "wrong_mean", "wrong_best")
    values = {key: [] for key in keys}
    for _ in range(reps):
        chosen_seeds = rng.choice(len(blocks), size=len(blocks), replace=True)
        episode_universe = np.unique(
            np.concatenate([blocks[int(index)]["episode_ids"] for index in chosen_seeds])
        )
        chosen_episodes = rng.choice(
            episode_universe, size=len(episode_universe), replace=True
        )
        replicate = {key: [] for key in keys}
        for seed_index in chosen_seeds:
            block = blocks[int(seed_index)]
            rows: list[int] = []
            for episode in chosen_episodes:
                rows.extend(np.flatnonzero(block["episode_ids"] == episode).tolist())
            if not rows:
                continue
            for key in keys:
                replicate[key].append(float(np.mean(block[key][rows])))
        for key in keys:
            values[key].append(float(np.mean(replicate[key])))
    return {
        "correct_minus_global": np.asarray(values["correct"]) - np.asarray(values["global"]),
        "correct_minus_wrong_mean": np.asarray(values["correct"])
        - np.asarray(values["wrong_mean"]),
        "correct_minus_wrong_best": np.asarray(values["correct"])
        - np.asarray(values["wrong_best"]),
    }


class StagedRegionRiskTest(unittest.TestCase):
    def test_h10_call_returns_h1_h5_h10_without_second_rollout(self) -> None:
        emb = torch.zeros((12, 4, 2), dtype=torch.float32)
        for row in range(len(emb)):
            emb[row, -1] = torch.tensor([float(row + 1), 0.0])
        cache = LeWMLatentCache(
            emb,
            emb.clone(),
            np.arange(len(emb), dtype=np.int64),
            route_index=0,
        )
        contract = load_cache_contract(cache, history_size=3, num_preds=1, frameskip=1)
        model = CountingPredictor(1.0)
        result = multi_horizon_open_loop_rollout_losses(
            [model],
            cache,
            np.asarray([0, 1], dtype=np.int64),
            horizons=[1, 5, 10],
            contract=contract,
            start_map=start_index_map(cache.sample_ids),
            device=torch.device("cpu"),
            batch_size=8,
        )
        self.assertEqual(set(result.by_horizon), {1, 5, 10})
        self.assertEqual(model.predict_calls, 10)
        np.testing.assert_array_equal(
            result.by_horizon[1].terminal_mse,
            result.by_horizon[1].mean_trajectory_mse,
        )

    def test_raw_resume_requires_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            fingerprint = stable_json_sha256({"task": "tworoom", "seed": 0})
            atomic_savez_compressed(
                path,
                metadata_json=np.asarray(json.dumps({"fingerprint": fingerprint})),
                values=np.asarray([1.0]),
            )
            self.assertTrue(
                _resume_matches(path, fingerprint, resume=True, kind="raw rollout")
            )
            with self.assertRaises(ValueError):
                _resume_matches(path, "different", resume=True, kind="raw rollout")

    def test_episode_preaggregation_matches_legacy_bootstrap(self) -> None:
        blocks = [
            {
                "global": np.asarray([1.0, 2.0, 4.0, 5.0]),
                "correct": np.asarray([0.8, 1.7, 3.8, 4.6]),
                "wrong_mean": np.asarray([1.2, 2.3, 4.4, 5.5]),
                "wrong_best": np.asarray([1.1, 2.1, 4.2, 5.2]),
                "episode_ids": np.asarray([10, 10, 11, 12]),
            },
            {
                "global": np.asarray([2.0, 3.0, 6.0]),
                "correct": np.asarray([1.5, 2.5, 5.4]),
                "wrong_mean": np.asarray([2.3, 3.4, 6.6]),
                "wrong_best": np.asarray([2.1, 3.2, 6.2]),
                "episode_ids": np.asarray([10, 11, 13]),
            },
        ]
        expected = legacy_nested_draws(blocks, reps=500, seed=19)
        actual, estimates = nested_paired_bootstrap_draws(
            blocks, reps=500, rng=np.random.default_rng(19)
        )
        self.assertAlmostEqual(
            estimates["correct"],
            np.mean([np.mean(block["correct"]) for block in blocks]),
        )
        for metric in expected:
            np.testing.assert_allclose(
                np.quantile(actual[metric], [0.025, 0.975]),
                np.quantile(expected[metric], [0.025, 0.975]),
                rtol=0,
                atol=1e-12,
            )

    def test_chunk_seed_is_deterministic_and_resume_safe(self) -> None:
        blocks = [
            {
                "global": np.asarray([1.0, 2.0]),
                "correct": np.asarray([0.5, 1.5]),
                "wrong_mean": np.asarray([1.2, 2.2]),
                "wrong_best": np.asarray([1.1, 2.1]),
                "episode_ids": np.asarray([0, 1]),
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "fingerprint": stable_json_sha256({"chunk": 0}),
                "horizon": 5,
            }
            common = {
                "summaries": precompute_episode_summaries(
                    blocks,
                    metric_keys=("global", "correct", "wrong_mean", "wrong_best"),
                ),
                "estimates": {
                    key: float(np.mean(blocks[0][key]))
                    for key in ("global", "correct", "wrong_mean", "wrong_best")
                },
                "reps": 100,
                "bootstrap_seed": 7,
                "horizon": 5,
                "loss_kind_id": 0,
                "chunk_id": 0,
                "metadata": metadata,
            }
            first = root / "first.npz"
            second = root / "second.npz"
            _bootstrap_chunk_job({**common, "path": str(first)})
            _bootstrap_chunk_job({**common, "path": str(second)})
            with np.load(first, allow_pickle=False) as left, np.load(
                second, allow_pickle=False
            ) as right:
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
            self.assertTrue(
                _resume_matches(
                    first, metadata["fingerprint"], resume=True, kind="bootstrap chunk"
                )
            )

    def test_finalize_preserves_outputs_and_public_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            raw_dir = out_dir / "raw"
            chunk_dir = out_dir / "bootstrap_chunks"
            metadata = {
                "train_seed": 0,
                "support_horizon": 1,
                "requested_horizons": [1],
            }
            matrix = np.asarray(
                [
                    [1.0, 0.5, 1.2, 1.3],
                    [2.0, 2.2, 1.5, 2.3],
                    [3.0, 3.2, 3.3, 2.5],
                ],
                dtype=np.float64,
            )
            atomic_savez_compressed(
                raw_dir / "trainseed0_h1_valid.npz",
                metadata_json=np.asarray(json.dumps(metadata)),
                anchors=np.asarray([0, 1, 2]),
                episode_ids=np.asarray([0, 1, 2]),
                region_ids=np.asarray([0, 1, 2]),
                h1_mean_loss_matrix=matrix,
                h1_terminal_loss_matrix=matrix,
            )
            atomic_write_json(
                out_dir / "rollout_manifest.json",
                {"audit": {"formal": True}, "num_regions": 3, "horizons": [1]},
            )
            chunk = chunk_dir / "h1_mean_trajectory_chunk00000.npz"
            atomic_savez_compressed(
                chunk,
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "estimates": {
                                "global": 2.0,
                                "correct": 1.5,
                                "wrong_mean": 2.2,
                                "wrong_best": 2.0,
                            }
                        }
                    )
                ),
                correct_minus_global=np.linspace(-0.7, -0.3, 100),
                correct_minus_wrong_mean=np.linspace(-0.9, -0.5, 100),
                correct_minus_wrong_best=np.linspace(-0.7, -0.3, 100),
            )
            atomic_write_json(
                out_dir / "bootstrap_index.json",
                {
                    "entries": [
                        {
                            "horizon": 1,
                            "loss_kind": "mean_trajectory",
                            "chunks": [str(chunk)],
                        }
                    ]
                },
            )
            run_finalize_stage(
                argparse.Namespace(task="tworoom", out_dir=out_dir, gate_summary=None)
            )
            required = {
                "audit.json",
                "sample_metrics.npz",
                "episode_metrics.csv",
                "region_summary.csv",
                "weighted_summary.csv",
                "bootstrap_summary.csv",
                "region_risk.pdf",
                "manifest.json",
            }
            self.assertTrue(required.issubset({path.name for path in out_dir.iterdir()}))
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["public_analysis_name"], PUBLIC_ANALYSIS_NAME)
            self.assertEqual(
                manifest["public_analysis_short_name"], PUBLIC_ANALYSIS_SHORT_NAME
            )
            self.assertIn(
                PUBLIC_ANALYSIS_SHORT_NAME.encode("ascii"),
                (out_dir / "region_risk.pdf").read_bytes(),
            )

    def test_cli_description_uses_public_name(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), mock.patch(
            "sys.argv", ["evaluate_region_conditional_risk.py", "--help"]
        ), self.assertRaises(SystemExit):
            parse_args()
        self.assertIn(PUBLIC_ANALYSIS_NAME, stream.getvalue())

if __name__ == "__main__":
    unittest.main()
