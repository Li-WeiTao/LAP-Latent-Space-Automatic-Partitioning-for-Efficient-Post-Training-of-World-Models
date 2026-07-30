"""Command line interface for backend-driven latent-cache construction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from lap.encoding.fast import FastEncodingConfig, FastLatentCacheEncoder
from lap.interfaces.encoding import EncodingDataset, LatentEncoderAdapter


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a JSON object: {path}")

    def expand(item: Any) -> Any:
        if isinstance(item, str):
            return os.path.expandvars(os.path.expanduser(item))
        if isinstance(item, list):
            return [expand(part) for part in item]
        if isinstance(item, dict):
            return {key: expand(part) for key, part in item.items()}
        return item

    return expand(value)


def _load_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("factory must use MODULE:CALLABLE syntax")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"factory is not callable: {spec}")
    return factory


def _parse_overrides(values: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("adapter overrides must use KEY=VALUE syntax")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError("adapter override key cannot be empty")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        if isinstance(parsed, str):
            parsed = os.path.expandvars(os.path.expanduser(parsed))
        result[key] = parsed
    return result


def _construct(
    spec: str,
    config_path: str | Path | None,
    overrides: list[str] | None,
) -> Any:
    config = _load_json(config_path)
    config.update(_parse_overrides(overrides))
    return _load_factory(spec)(**config)


def _add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-factory",
        required=True,
        help="Import path MODULE:CALLABLE returning an EncodingDataset adapter.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="JSON keyword arguments passed to the dataset factory.",
    )
    parser.add_argument(
        "--dataset-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dataset-factory override; JSON values are parsed.",
    )
    parser.add_argument(
        "--encoder-factory",
        required=True,
        help="Import path MODULE:CALLABLE returning a LatentEncoderAdapter.",
    )
    parser.add_argument(
        "--encoder-config",
        type=Path,
        default=None,
        help="Optional JSON keyword arguments passed to the encoder factory.",
    )
    parser.add_argument(
        "--encoder-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable encoder-factory override; JSON values are parsed.",
    )
    parser.add_argument(
        "--pretrained-model",
        required=True,
        help="Checkpoint path or official model ID passed unchanged to the adapter.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lap-cache",
        description=(
            "Build a frozen-encoder latent cache without coupling LAP to one "
            "task dataset or world-model encoder."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit one stable JSON object on stdout."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Instantiate and validate dataset/encoder adapter contracts."
    )
    _add_adapter_arguments(doctor)

    encode = subparsers.add_parser(
        "encode", help="Build a lossless cache with unique-frame acceleration."
    )
    _add_adapter_arguments(encode)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--report", type=Path, default=None)
    encode.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Optional NPZ whose generated arrays must match exactly.",
    )
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--transition-batch-size", type=int, default=128)
    encode.add_argument("--frame-batch-size", type=int, default=512)
    encode.add_argument(
        "--batch-shape-mode",
        choices=("exact", "fixed"),
        default="exact",
        help=(
            "exact preserves legacy visual batch shapes for bitwise reproduction; "
            "fixed maximizes throughput for new caches."
        ),
    )
    encode.add_argument("--num-workers", type=int, default=4)
    encode.add_argument("--prefetch-factor", type=int, default=2)
    encode.add_argument("--cpu-threads", type=int, default=4)
    encode.add_argument("--log-every", type=int, default=50)
    encode.add_argument("--start-offset", type=int, default=0)
    encode.add_argument("--max-samples", type=int, default=0)
    encode.add_argument("--no-chunk-aware-read", action="store_true")
    return parser


def _adapter_status(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    dataset = _construct(
        args.dataset_factory, args.dataset_config, args.dataset_arg
    )
    encoder = _construct(
        args.encoder_factory, args.encoder_config, args.encoder_arg
    )
    dataset_ok = isinstance(dataset, EncodingDataset)
    encoder_ok = isinstance(encoder, LatentEncoderAdapter)
    result = {
        "ok": dataset_ok and encoder_ok,
        "dataset": {
            "factory": args.dataset_factory,
            "contract_ok": dataset_ok,
            "description": dict(dataset.describe()) if dataset_ok else None,
        },
        "encoder": {
            "factory": args.encoder_factory,
            "contract_ok": encoder_ok,
            "description": dict(encoder.describe()) if encoder_ok else None,
        },
        "pretrained_model": args.pretrained_model,
    }
    return dataset, encoder, result


def _write_result(value: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, sort_keys=True))
        return
    for key, item in value.items():
        print(f"{key}: {item}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset, encoder, status = _adapter_status(args)
    if not status["ok"]:
        raise TypeError("dataset or encoder factory does not satisfy its protocol")
    if args.command == "doctor":
        return status
    config = FastEncodingConfig(
        device=args.device,
        transition_batch_size=args.transition_batch_size,
        frame_batch_size=args.frame_batch_size,
        exact_batch_shapes=args.batch_shape_mode == "exact",
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        cpu_threads=args.cpu_threads,
        chunk_aware=not args.no_chunk_aware_read,
        log_every=args.log_every,
        start_offset=args.start_offset,
        max_samples=args.max_samples,
    )
    log = (
        (lambda message: print(message, file=sys.stderr, flush=True))
        if args.json
        else (lambda message: print(message, flush=True))
    )
    report = FastLatentCacheEncoder(config, log=log).encode(
        dataset=dataset,
        encoder=encoder,
        pretrained_model=args.pretrained_model,
        output=args.output,
        report=args.report,
        reference_cache=args.reference_cache,
    )
    return {
        "ok": True,
        "output": str(args.output),
        "report": str(
            args.report
            if args.report is not None
            else args.output.with_suffix(args.output.suffix + ".report.json")
        ),
        "samples": report["selection"]["samples"],
        "actual_speedup": report["efficiency"][
            "actual_kernel_frame_speedup_after_padding"
        ],
        "validation_passed": (
            None
            if report["validation"] is None
            else report["validation"]["passed"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as error:  # CLI boundary returns stable machine-readable errors.
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": type(error).__name__,
                        "message": str(error),
                    },
                    sort_keys=True,
                )
            )
            return 1
        raise
    _write_result(result, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
