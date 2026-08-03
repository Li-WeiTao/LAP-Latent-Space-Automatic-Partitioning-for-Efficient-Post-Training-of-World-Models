#!/usr/bin/env python3
"""Task-agnostic JEPA object checkpoint compatibility probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.control_matrix.jepa_checkpoint_probe import (  # noqa: E402
    probe_checkpoint,
    write_probe_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--encoder-config", default=None)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = probe_checkpoint(
        model_family=args.model_family,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        frameskip=args.frameskip,
        history_size=args.history_size,
        num_preds=args.num_preds,
        img_size=args.img_size,
        max_samples=args.max_samples,
        device=args.device,
    )
    if args.dataset_config is not None:
        report["dataset_config"] = str(args.dataset_config)
    if args.encoder_config is not None:
        report["encoder_config"] = str(args.encoder_config)
    write_probe_report(report, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
