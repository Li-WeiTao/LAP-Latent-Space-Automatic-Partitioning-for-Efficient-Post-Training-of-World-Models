#!/usr/bin/env python3
"""Bitwise-compare the migrated LeWM trainer with a legacy trajectory module."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import torch
from torch import nn

from backends.lewm import finetuning as migrated


class TwoArgPredictor(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.linear(emb + act_emb)


class TinyWorldModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.encoder = nn.Linear(width, width)
        self.projector = nn.Linear(width, width)
        self.action_encoder = nn.Linear(width, width)
        self.predictor = TwoArgPredictor(width)
        self.pred_proj = nn.Linear(width, width)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        raw = self.predictor(emb, act_emb)
        batch, steps, width = raw.shape
        return self.pred_proj(raw.reshape(batch * steps, width)).reshape(
            batch, steps, width
        )


def load_legacy(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("lap_legacy_trajectory", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load legacy module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_trajectory", type=Path)
    args = parser.parse_args()
    legacy = load_legacy(args.legacy_trajectory.resolve())

    torch.manual_seed(123)
    base = TinyWorldModel(5)
    generator = torch.Generator().manual_seed(999)
    emb = torch.randn(17, 4, 5, generator=generator)
    act = torch.randn(17, 3, 5, generator=generator)
    shared = dict(
        history_size=3,
        num_preds=1,
        batch_size=4,
        epochs=3,
        lr=5e-5,
        weight_decay=1e-3,
        seed=42,
    )
    legacy_config = legacy.TrainConfig(**shared)
    migrated_config = migrated.LeWMTrainConfig(**shared)
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_model, legacy_stats = legacy.train_region_predictor(
            base,
            emb,
            act,
            legacy_config,
            torch.device("cpu"),
            select_best_by_eval=True,
        )
        migrated_model, migrated_stats = migrated.train_region_predictor(
            base,
            emb,
            act,
            migrated_config,
            torch.device("cpu"),
            select_best_by_eval=True,
        )
    for module_name in ("predictor", "pred_proj"):
        legacy_state = getattr(legacy_model, module_name).state_dict()
        migrated_state = getattr(migrated_model, module_name).state_dict()
        if legacy_state.keys() != migrated_state.keys():
            raise AssertionError(f"state keys differ for {module_name}")
        for key in legacy_state:
            if not torch.equal(legacy_state[key], migrated_state[key]):
                raise AssertionError(f"parameter mismatch: {module_name}.{key}")
    if legacy_stats != migrated_stats:
        raise AssertionError("trainer metric histories differ")
    print("old_vs_new_trainer_bitwise: PASS")
    print(f"best_epoch: {migrated_stats['best_epoch']}")
    print(f"best_eval_loss: {migrated_stats['best_eval_loss']:.12f}")


if __name__ == "__main__":
    main()
