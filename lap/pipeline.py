"""End-to-end, backend-neutral LAP interface."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from lap.finetuning import fit_regional_predictors
from lap.interfaces import (
    LatentCache,
    PredictorTrainingResult,
    RegionalPredictorTrainer,
    RegionalTrainingConfig,
    WorldModelBackendFactory,
)
from lap.partition import LatentPartitioner, PartitionResult
from lap.routing import VoronoiRouter


@dataclass(frozen=True)
class LAPConfig:
    """Configuration owned by LAP rather than a particular world model."""

    training: RegionalTrainingConfig = field(
        default_factory=RegionalTrainingConfig
    )
    output_directory: Path | None = None

    def validate(self) -> None:
        self.training.validate()


@dataclass
class LAPFitResult:
    backend: Any
    partition: PartitionResult
    regional_predictors: dict[int, PredictorTrainingResult]
    sample_ids: np.ndarray

    @property
    def router(self) -> VoronoiRouter:
        return VoronoiRouter(self.partition.artifact)

    def route(self, latents: np.ndarray) -> np.ndarray:
        return self.router.route(latents)

    def predictor_for_latent(self, latent: np.ndarray) -> Any:
        region_id = int(self.route(np.asarray(latent)[None])[0])
        return self.regional_predictors[region_id].predictor


class LAP:
    """Fit LAP from a prepared latent cache and a pretrained-model parameter."""

    def __init__(
        self,
        *,
        backend_factory: WorldModelBackendFactory,
        partitioner: LatentPartitioner,
        trainer: RegionalPredictorTrainer,
        config: LAPConfig | None = None,
    ) -> None:
        self.backend_factory = backend_factory
        self.partitioner = partitioner
        self.trainer = trainer
        self.config = config or LAPConfig()

    def fit(
        self, latent_cache: LatentCache, pretrained_model: Any
    ) -> LAPFitResult:
        """Run action-free partitioning and regional training from a latent cache."""

        self.config.validate()
        output = (
            None
            if self.config.output_directory is None
            else Path(self.config.output_directory)
        )
        if output is not None and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"LAP output directory is not empty: {output}")
        backend = self.backend_factory.load(pretrained_model)
        backend.freeze_encoder()
        transitions = latent_cache.load()
        transitions.validate()
        partition = self.partitioner.fit(
            transitions.routing_latents,
            sample_ids=transitions.sample_ids,
            group_ids=transitions.group_ids,
        )
        partition.validate(len(transitions.routing_latents))
        predictors = fit_regional_predictors(
            backend,
            transitions,
            partition.labels,
            self.trainer,
            self.config.training,
            num_regions=partition.artifact.num_regions,
        )
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            partition.artifact.save(output / "partition", overwrite=False)
            np.savez_compressed(
                output / "cluster_labels.npz",
                sample_ids=np.asarray(transitions.sample_ids),
                labels=np.asarray(partition.labels, dtype=np.int64),
            )
            regions: dict[str, Any] = {}
            for region_id, trained in predictors.items():
                checkpoint = output / f"P_train_cluster{region_id}_object.ckpt"
                backend.save_predictor(trained.predictor, checkpoint)
                metrics_path = output / f"P_train_cluster{region_id}_metrics.json"
                metrics_path.write_text(
                    json.dumps(trained.metrics, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                regions[f"cluster{region_id}"] = {
                    "num_samples": int((partition.labels == region_id).sum()),
                    "checkpoint": str(checkpoint),
                    "metrics": str(metrics_path),
                }
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": "LAP",
                        "pretrained_model": str(pretrained_model),
                        "num_transitions": len(transitions.sample_ids),
                        "num_regions": partition.artifact.num_regions,
                        "partition_metadata": partition.metadata,
                        "cache_metadata": transitions.metadata,
                        "regions": regions,
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        return LAPFitResult(
            backend=backend,
            partition=partition,
            regional_predictors=predictors,
            sample_ids=np.asarray(transitions.sample_ids),
        )
