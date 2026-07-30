#!/usr/bin/env python3
"""Convert an official LeWM weights.pt/config.json pair to an object checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def migrate_vit_key(key: str) -> str:
    """Map the official pre-Transformers-5 ViT names to current names."""

    key = key.replace("encoder.encoder.layer.", "encoder.layers.")
    key = key.replace(".attention.attention.query.", ".attention.q_proj.")
    key = key.replace(".attention.attention.key.", ".attention.k_proj.")
    key = key.replace(".attention.attention.value.", ".attention.v_proj.")
    key = key.replace(".attention.output.dense.", ".attention.o_proj.")
    key = key.replace(".intermediate.dense.", ".mlp.fc1.")
    return key.replace(".output.dense.", ".mlp.fc2.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config_path = (args.model_dir / "config.json").resolve(strict=True)
    weights_path = (args.model_dir / "weights.pt").resolve(strict=True)
    config = OmegaConf.create(json.loads(config_path.read_text()))
    model = instantiate(config)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    migrated = {migrate_vit_key(key): value for key, value in state.items()}
    model.load_state_dict(migrated, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
    provenance = {
        "official_config": str(config_path),
        "official_weights": str(weights_path),
        "official_weights_sha256": sha256(weights_path),
        "object_checkpoint": str(args.output.resolve()),
        "object_checkpoint_sha256": sha256(args.output),
        "key_migration": "legacy Hugging Face ViT names to current Transformers names",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
