from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lap import (
    EncodedTransitions,
    IndexedPartitioner,
    LAP,
    LAPConfig,
    PartitionArtifact,
    PredictorTrainingResult,
    RegionalTrainingConfig,
)


class FakeBackend:
    def __init__(self, model):
        self.model = model
        self.frozen = False

    def freeze_encoder(self):
        self.frozen = True

    def encode(self, observations):
        return np.asarray(observations)

    def routing_latent(self, encoded_context):
        return np.asarray(encoded_context)[:, -1]

    def predict(self, latent_context, actions, predictor=None):
        del actions
        return np.asarray(latent_context) + (predictor or 0)

    def load_predictor(self, checkpoint):
        return checkpoint

    def clone_predictor(self, predictor):
        return predictor

    def save_predictor(self, predictor, checkpoint):
        Path(checkpoint).write_text(str(predictor), encoding="utf-8")


class FakeFactory:
    def load(self, pretrained_model):
        return FakeBackend(pretrained_model)


@dataclass
class ArrayPayload:
    values: np.ndarray

    def subset(self, indices):
        return ArrayPayload(self.values[indices])


class FakeLatentCache:
    def __init__(self, latents, sample_ids):
        self.latents = latents
        self.sample_ids = sample_ids

    def load(self):
        return EncodedTransitions(
            routing_latents=self.latents,
            sample_ids=self.sample_ids,
            payload=ArrayPayload(self.latents.copy()),
        )


class FakeTrainer:
    def fit_region(self, backend, region_id, transitions, config):
        assert backend.frozen
        return PredictorTrainingResult(
            predictor=float(transitions.payload.values.mean()),
            metrics={"region": region_id, "n": len(transitions.sample_ids)},
        )


class GenericPipelineTest(unittest.TestCase):
    def test_latent_cache_and_pretrained_model_drive_end_to_end_lap(self):
        latents = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        sample_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        artifact = PartitionArtifact(
            prototypes=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            prototype_region_ids=np.asarray([0, 1], dtype=np.int64),
            mean=np.zeros(2, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            metadata={"num_clusters": 2},
        )
        partitioner = IndexedPartitioner(
            artifact,
            sample_ids,
            np.asarray([0, 0, 1, 1], dtype=np.int64),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lap_run"
            method = LAP(
                backend_factory=FakeFactory(),
                partitioner=partitioner,
                trainer=FakeTrainer(),
                config=LAPConfig(
                    training=RegionalTrainingConfig(
                        epochs=1,
                        batch_size=2,
                        min_region_samples=1,
                    ),
                    output_directory=output,
                ),
            )
            result = method.fit(
                FakeLatentCache(latents, sample_ids), "checkpoint.ckpt"
            )
            self.assertEqual(result.backend.model, "checkpoint.ckpt")
            np.testing.assert_array_equal(result.partition.labels, [0, 0, 1, 1])
            self.assertEqual(result.regional_predictors[0].metrics["n"], 2)
            self.assertEqual(result.regional_predictors[1].metrics["n"], 2)
            np.testing.assert_array_equal(result.route(latents), [0, 0, 1, 1])
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "P_train_cluster0_object.ckpt").is_file())
