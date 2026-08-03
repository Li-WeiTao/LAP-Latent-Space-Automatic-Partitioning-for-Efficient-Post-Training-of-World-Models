#!/usr/bin/env python3
"""One-step latent MSE on a fixed cache holdout for matrix training jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint
from experiments.control_matrix.jepa_checkpoint_probe import sha256_file
from experiments.control_matrix.region_risk_lib import one_step_losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model-family", default="subjepa")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock["sha256"]["full_cache"] != sha256_file(args.cache):
        raise SystemExit("cache sha256 does not match pre_execution_lock")

    cache = np.load(args.cache)
    emb = cache["emb"].astype(np.float32)
    act_emb = cache["act_emb"].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(emb))
    rng.shuffle(indices)
    holdout_count = max(1, int(len(indices) * args.holdout_fraction))
    holdout = np.sort(indices[:holdout_count])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(args.work_root.glob("training/**/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for region_name, region in manifest.get("regions", {}).items():
            ckpt_path = Path(region["checkpoint"])
            if not ckpt_path.is_file():
                ckpt_path = PROJECT_ROOT / region["checkpoint"]
            model = load_jepa_object_checkpoint(
                ckpt_path, model_family=args.model_family, map_location=device
            )
            model.eval()
            losses = one_step_losses(
                [model],
                emb[holdout],
                act_emb[holdout],
                history_size=args.history_size,
                num_preds=args.num_preds,
                device=device,
                batch_size=args.batch_size,
            )[:, 0]
            rows.append(
                {
                    "run_dir": str(manifest_path.parent.relative_to(args.work_root)),
                    "region": region_name,
                    "checkpoint": str(ckpt_path),
                    "holdout_count": int(len(holdout)),
                    "one_step_mse_mean": float(np.mean(losses)),
                    "one_step_mse_std": float(np.std(losses)),
                }
            )

    report = {
        "schema_version": 1,
        "scope": "subjepa_tworoom_matrix_one_step_mse",
        "git_commit": lock.get("git_commit"),
        "cache_sha256": sha256_file(args.cache),
        "holdout_fraction": args.holdout_fraction,
        "holdout_seed": args.seed,
        "runs": rows,
    }
    out = args.work_root / "manifests/one_step_mse.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"num_runs": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
