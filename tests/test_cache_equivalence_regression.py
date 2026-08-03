"""Regression tests for production cache-equivalence validation semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lap.encoding.fast import (
    FastEncodingConfig,
    FastLatentCacheEncoder,
    build_unique_frame_index,
    recompute_latent_windows,
)
from tests.fixtures_encoding import (
    BatchShapeSensitiveEncoder,
    FakeEncodingDataset,
    make_batch_shape_sensitive_encoder,
    make_dataset,
    make_encoder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TWOROOM_SUBJEPA_CACHE = (
    REPO_ROOT / "experiments/tworoom/subjepa/preparation/embedding_cache.npz"
)
TWOROOM_SUBJEPA_STARTS = (
    REPO_ROOT / "experiments/tworoom/subjepa/preparation/.encode_starts.npy"
)
TWOROOM_SUBJEPA_EMB_SHA256 = (
    "982fa6f190e189505393649ccee5b906a84be8673bbedfc65b537ea64ea47c7c"
)
TWOROOM_SUBJEPA_CACHE_SHA256 = (
    "6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7"
)


def _direct_window_encode(
    dataset: FakeEncodingDataset,
    encoder,
    model: dict,
    *,
    num_samples: int,
) -> np.ndarray:
    selection = dataset.make_selection(max_samples=num_samples)
    seq_len = selection.frame_ids.shape[1]
    rows = []
    for sample_index in range(len(selection.sample_ids)):
        frame_dataset = dataset.make_frame_dataset(
            selection.frame_ids[sample_index], chunk_aware=False
        )
        frames = torch.stack(
            [frame_dataset[index][0] for index in range(seq_len)], dim=0
        )
        rows.append(encoder.encode_frames(model, frames, torch.device("cpu")))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


class CacheEquivalenceRegressionTest(unittest.TestCase):
    def test_production_unique_frame_path_matches_reload_and_recompute(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "cache.npz"
            dataset = make_dataset(offset=0)
            encoder = make_encoder(scale=2.0)
            config = FastEncodingConfig(
                device="cpu",
                transition_batch_size=2,
                frame_batch_size=4,
                exact_batch_shapes=True,
                num_workers=0,
                cpu_threads=1,
            )
            report = FastLatentCacheEncoder(config, log=lambda *_: None).encode(
                dataset=dataset,
                encoder=encoder,
                pretrained_model="fake-checkpoint",
                output=output,
            )
            with np.load(output, allow_pickle=False) as data:
                writer_before_save = np.asarray(data["emb"], dtype=np.float32)
            recomputed, selection = recompute_latent_windows(
                dataset=dataset,
                encoder=encoder,
                model={"model": "fake-checkpoint"},
                config=config,
                device=torch.device("cpu"),
                log=lambda *_: None,
            )
            np.testing.assert_array_equal(writer_before_save, recomputed)
            self.assertEqual(
                report["method"], "unique_frame_inverse_reconstruction"
            )
            self.assertTrue(report["config"]["exact_batch_shapes"])
            self.assertEqual(
                report["selection"]["source_count"], len(selection.sample_ids)
            )

    def test_sample_and_frame_identity_matches_selection(self):
        dataset = make_dataset(offset=5)
        selection = dataset.make_selection(max_samples=0)
        keyed_frames, required_shapes, inverse, _ = build_unique_frame_index(
            selection,
            transition_batch_size=2,
            exact_batch_shapes=True,
        )
        reconstructed = keyed_frames[inverse].reshape(selection.frame_ids.shape)
        np.testing.assert_array_equal(reconstructed, selection.frame_ids)
        np.testing.assert_array_equal(
            selection.sample_ids, np.asarray([5, 6, 7], dtype=np.int64)
        )
        self.assertEqual(set(required_shapes.tolist()), {2, 4})

    def test_changing_batch_shape_produces_known_numerical_difference(self):
        dataset = make_dataset(offset=0)
        encoder = make_batch_shape_sensitive_encoder(scale=1.0)
        model = {"model": "fake-checkpoint"}
        config = FastEncodingConfig(
            device="cpu",
            transition_batch_size=2,
            frame_batch_size=4,
            exact_batch_shapes=True,
            num_workers=0,
            cpu_threads=1,
        )
        production, _selection = recompute_latent_windows(
            dataset=dataset,
            encoder=encoder,
            model=model,
            config=config,
            device=torch.device("cpu"),
            log=lambda *_: None,
        )
        direct = _direct_window_encode(
            dataset, encoder, model, num_samples=len(_selection.sample_ids)
        )
        diff = np.abs(production - direct)
        self.assertGreater(float(diff.max()), 1e-3)
        self.assertFalse(np.allclose(production, direct, rtol=1e-5, atol=1e-6))
        self.assertGreater(float(direct[0, :, 1].max()), 1.0)
        self.assertGreater(float(production[0, :, 1].max()), float(direct[0, :, 1].max()))

    @unittest.skipUnless(
        TWOROOM_SUBJEPA_CACHE.exists() and TWOROOM_SUBJEPA_STARTS.exists(),
        "TwoRoom Sub-JEPA smoke cache not present",
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Sub-JEPA cache replay")
    def test_tworoom_subjepa_verified_cache_recomputes_exact(self):
        from backends.lewm.encoding import LeWMEncoderAdapter, make_hdf5_transition_dataset
        from experiments.control_matrix.jepa_checkpoint_probe import sha256_file

        self.assertEqual(sha256_file(TWOROOM_SUBJEPA_CACHE), TWOROOM_SUBJEPA_CACHE_SHA256)
        with np.load(TWOROOM_SUBJEPA_CACHE, allow_pickle=False) as data:
            self.assertEqual(
                json.loads(
                    (
                        REPO_ROOT
                        / "experiments/tworoom/subjepa/preparation/representation_manifest.json"
                    ).read_text(encoding="utf-8")
                )["encode_report"]["arrays"]["emb"]["sha256"],
                TWOROOM_SUBJEPA_EMB_SHA256,
            )
            cached_emb = np.asarray(data["emb"][:16], dtype=np.float32)
            cached_starts = np.asarray(data["region_starts"][:16], dtype=np.int64)

        dataset = make_hdf5_transition_dataset(
            data_file="/data/sicong/weitao/datasets/lewm/tworoom.h5",
            starts=str(TWOROOM_SUBJEPA_STARTS.resolve()),
            history_size=3,
            num_preds=1,
            frameskip=5,
        )
        device = torch.device("cuda")
        adapter = LeWMEncoderAdapter(
            img_size=224, frameskip=5, model_family="subjepa"
        )
        model = adapter.load(
            "/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt",
            device,
        )
        adapter.prepare_dataset(dataset, model)
        selection = dataset.make_selection(max_samples=16)
        np.testing.assert_array_equal(selection.sample_ids, cached_starts)
        config = FastEncodingConfig(
            device="cuda",
            transition_batch_size=128,
            frame_batch_size=512,
            exact_batch_shapes=True,
            num_workers=0,
            cpu_threads=1,
        )
        recomputed, full_selection = recompute_latent_windows(
            dataset=dataset,
            encoder=adapter,
            model=model,
            config=config,
            device=device,
            log=lambda *_: None,
        )
        self.assertEqual(full_selection.source_count, 3686)
        np.testing.assert_allclose(
            recomputed[:16],
            cached_emb,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
