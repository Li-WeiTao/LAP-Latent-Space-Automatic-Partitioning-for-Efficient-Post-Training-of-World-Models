#!/usr/bin/env python3
"""Losslessly re-encode JEPA transition caches by encoding each frame once.

The reference cache stores overlapping transition windows with shape ``(N,T,D)``.
Adjacent windows repeatedly encode the same image.  This script computes the
sorted union of their global frame indices, encodes every unique frame once,
and reconstructs the original windows through an inverse index.  Action
normalization, action encoding, transition order, output keys and dtypes follow
``trajectory.py``.

"Completely identical" means array-level identity for ``emb``, ``act_emb`` and
``region_starts``.  A compressed NPZ file is not expected to have the same byte
hash because ZIP metadata can differ between writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import hdf5plugin  # noqa: F401  # Register optional HDF5 compression filters.
except ImportError:
    pass

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import load_encoder, preprocess_pixels  # noqa: E402
from predictor_rule_drift import action_block_stats, infer_frameskip  # noqa: E402
from trajectory import read_action_blocks_vectorized  # noqa: E402


DEFAULT_DATA = Path("/data/sicong/weitao/datasets/lewm/tworoom.h5")
DEFAULT_CHECKPOINT = Path(
    "/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"
)
DEFAULT_GLOBAL_STARTS = Path(
    "experiments/tworoom/results/"
    "tworoom_geometry_train_region_predictors/train_global_reference_starts.npy"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--starts",
        type=Path,
        default=None,
        help=(
            "Transition-start .npy. Required unless --reference-cache supplies "
            "region_starts."
        ),
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help=(
            "Optional original NPZ. Its region_starts become the requested starts "
            "and all three output arrays are checked for exact equality."
        ),
    )
    parser.add_argument(
        "--action-norm-starts",
        type=Path,
        default=DEFAULT_GLOBAL_STARTS,
        help="Global training starts used by the original action normalizer.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON timing/equivalence report; defaults to OUTPUT.report.json.",
    )
    parser.add_argument("--pixel-key", default="pixels")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument(
        "--frameskip", type=int, default=0, help="0 infers it from action_encoder."
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--transition-batch-size",
        type=int,
        default=128,
        help="Action-encoder batch size; 128 exactly matches the reference path.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=0,
        help=(
            "Reserved diagnostic override. Exact mode requires 0 so that every "
            "frame is evaluated under its original visual batch shape."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Slice offset within the source starts (mainly for exact validation).",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--no-chunk-aware-read",
        action="store_true",
        help="Disable grouped HDF5-chunk reads (diagnostic only).",
    )
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_ready(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class UniqueFrameDataset(Dataset):
    """Lazy HDF5 dataset with optional per-batch, chunk-coalesced reads."""

    def __init__(
        self,
        h5_path: Path,
        pixel_key: str,
        frame_indices: np.ndarray,
        *,
        chunk_aware: bool,
    ) -> None:
        self.h5_path = str(h5_path)
        self.pixel_key = pixel_key
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.chunk_aware = bool(chunk_aware)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r", swmr=True) as h5:
            dataset = h5[self.pixel_key]
            self.num_frames = int(dataset.shape[0])
            self.chunk_len = int(dataset.chunks[0]) if dataset.chunks else 1

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )
        return self._h5

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except (OSError, RuntimeError):
                pass

    def __getitem__(self, position: int) -> tuple[np.ndarray, np.int64]:
        frame = int(self.frame_indices[position])
        pixels = np.asarray(self._open()[self.pixel_key][frame])
        return pixels, np.int64(frame)

    def __getitems__(
        self, positions: list[int]
    ) -> list[tuple[np.ndarray, np.int64]]:
        """Read each touched HDF5 chunk once while preserving sampler order."""
        if not self.chunk_aware:
            return [self[position] for position in positions]

        positions_array = np.asarray(positions, dtype=np.int64)
        frames = self.frame_indices[positions_array]
        chunk_ids = frames // self.chunk_len
        output: list[tuple[np.ndarray, np.int64] | None] = [None] * len(frames)
        dataset = self._open()[self.pixel_key]

        for chunk_id in np.unique(chunk_ids):
            members = np.flatnonzero(chunk_ids == chunk_id)
            lo = int(chunk_id * self.chunk_len)
            hi = min(lo + self.chunk_len, self.num_frames)
            block = np.asarray(dataset[lo:hi])
            for member in members:
                frame = int(frames[member])
                output[int(member)] = (block[frame - lo], np.int64(frame))

        if any(item is None for item in output):
            raise RuntimeError("Internal error: not every requested frame was read")
        return output  # type: ignore[return-value]


def load_starts(args: argparse.Namespace) -> tuple[np.ndarray, str, int]:
    if args.reference_cache is not None:
        with np.load(args.reference_cache) as reference:
            if "region_starts" not in reference.files:
                raise KeyError(
                    f"{args.reference_cache} does not contain region_starts"
                )
            all_starts = np.asarray(reference["region_starts"], dtype=np.int64)
        source = str(args.reference_cache) + "::region_starts"
    else:
        if args.starts is None:
            raise ValueError("Pass --starts or --reference-cache")
        all_starts = np.asarray(np.load(args.starts), dtype=np.int64)
        source = str(args.starts)

    total_count = len(all_starts)
    if args.start_offset < 0 or args.start_offset >= total_count:
        raise ValueError(
            f"start-offset must be in [0, {total_count}), got {args.start_offset}"
        )
    stop = total_count
    if args.max_starts > 0:
        stop = min(args.start_offset + args.max_starts, total_count)
    starts = all_starts[args.start_offset:stop]
    if starts.ndim != 1 or len(starts) == 0:
        raise ValueError(f"Expected non-empty one-dimensional starts, got {starts.shape}")
    return starts, source, total_count


def build_exact_unique_index(
    starts: np.ndarray,
    sequence_length: int,
    frameskip: int,
    transition_batch_size: int,
    source_transition_offset: int,
    source_transition_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Deduplicate frames while retaining the legacy CUDA batch shape.

    The released model is mathematically sample-wise at inference, but CUDA
    kernels can produce different floating-point bits for different batch
    shapes.  Full legacy transition batches contain ``B*T`` images while the
    final partial batch can be smaller.  A key is therefore ``(frame, B*T)``.
    In normal batches this still encodes each global frame exactly once; only a
    frame crossing into the final partial batch can have a second shape key.
    """
    offsets = np.arange(sequence_length, dtype=np.int64) * frameskip
    frame_slots = starts[:, None] + offsets[None, :]
    transition_ids = source_transition_offset + np.arange(len(starts), dtype=np.int64)
    batch_offsets = (transition_ids // transition_batch_size) * transition_batch_size
    batch_transition_counts = np.minimum(
        transition_batch_size, source_transition_count - batch_offsets
    )
    visual_batch_shapes = np.repeat(
        batch_transition_counts * sequence_length, sequence_length
    )
    keys = np.column_stack(
        (frame_slots.reshape(-1), visual_batch_shapes.astype(np.int64, copy=False))
    )
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    global_unique_count = int(np.unique(frame_slots).size)
    return unique_keys[:, 0], unique_keys[:, 1], inverse, global_unique_count


@torch.inference_mode()
def encode_unique_frames(
    model: torch.nn.Module,
    dataset: UniqueFrameDataset,
    device: torch.device,
    *,
    img_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    log_every: int,
) -> tuple[np.ndarray, dict[str, float]]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    loader = DataLoader(dataset, **loader_kwargs)
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    frame_embeddings: np.ndarray | None = None
    wait_sec = 0.0
    compute_sec = 0.0
    total_t0 = time.perf_counter()
    iterator = iter(loader)

    for batch_idx in range(num_batches):
        wait_t0 = time.perf_counter()
        pixels_cpu, returned_frames = next(iterator)
        wait_sec += time.perf_counter() - wait_t0

        offset = batch_idx * batch_size
        stop = offset + len(returned_frames)
        expected = dataset.frame_indices[offset:stop]
        if not np.array_equal(returned_frames.numpy(), expected):
            raise RuntimeError("DataLoader changed unique-frame ordering")

        compute_t0 = time.perf_counter()
        pixels_np = pixels_cpu.numpy()
        valid_count = pixels_np.shape[0]
        # The reference path always sends B*T images to the encoder. CUDA GEMM
        # and preprocessing path. Their floating-point kernels are shape-
        # dependent, so pad the raw uint8 batch before preprocessing and discard
        # only the synthetic outputs. Padding after normalization is numerically
        # close but not bitwise identical on this host.
        if valid_count < batch_size:
            pixels_np = np.concatenate(
                (
                    pixels_np,
                    np.repeat(pixels_np[-1:], batch_size - valid_count, axis=0),
                ),
                axis=0,
            )
        pixels = preprocess_pixels(pixels_np, device, img_size)
        output = model.encode({"pixels": pixels.unsqueeze(1)})
        embeddings = output["emb"][:valid_count, 0].detach().cpu().numpy()
        if frame_embeddings is None:
            frame_embeddings = np.empty(
                (len(dataset), embeddings.shape[1]), dtype=embeddings.dtype
            )
        frame_embeddings[offset:stop] = embeddings
        compute_sec += time.perf_counter() - compute_t0

        if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == num_batches:
            print(
                f"  [unique-frame] batch {batch_idx + 1}/{num_batches} "
                f"({stop}/{len(dataset)} frames)",
                flush=True,
            )

    if frame_embeddings is None:
        raise RuntimeError("No frame embeddings were produced")
    total_sec = time.perf_counter() - total_t0
    timing = {
        "total_sec": total_sec,
        "loader_wait_sec": wait_sec,
        "preprocess_encode_d2h_sec": compute_sec,
    }
    print(
        "  [unique-frame] timing "
        f"total={total_sec:.2f}s loader_wait={wait_sec:.2f}s "
        f"preprocess+encode+d2h={compute_sec:.2f}s",
        flush=True,
    )
    return frame_embeddings, timing


@torch.inference_mode()
def encode_exact_frame_keys(
    model: torch.nn.Module,
    h5_path: Path,
    pixel_key: str,
    keyed_frames: np.ndarray,
    required_batch_shapes: np.ndarray,
    device: torch.device,
    *,
    img_size: int,
    num_workers: int,
    prefetch_factor: int,
    log_every: int,
    chunk_aware: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode each ``(frame, legacy visual batch shape)`` exactly once."""
    all_embeddings: np.ndarray | None = None
    groups: dict[str, Any] = {}
    aggregate = {
        "total_sec": 0.0,
        "loader_wait_sec": 0.0,
        "preprocess_encode_d2h_sec": 0.0,
    }
    padded_encoder_slots = 0
    for required_shape in np.unique(required_batch_shapes):
        positions = np.flatnonzero(required_batch_shapes == required_shape)
        dataset = UniqueFrameDataset(
            h5_path,
            pixel_key,
            keyed_frames[positions],
            chunk_aware=chunk_aware,
        )
        print(
            f"[batch-shape] visual_batch={required_shape} "
            f"unique_keys={len(positions)}",
            flush=True,
        )
        group_embeddings, timing = encode_unique_frames(
            model,
            dataset,
            device,
            img_size=img_size,
            batch_size=int(required_shape),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            log_every=log_every,
        )
        if all_embeddings is None:
            all_embeddings = np.empty(
                (len(keyed_frames), group_embeddings.shape[-1]),
                dtype=group_embeddings.dtype,
            )
        all_embeddings[positions] = group_embeddings
        group_padded_slots = (
            (len(positions) + int(required_shape) - 1)
            // int(required_shape)
            * int(required_shape)
        )
        padded_encoder_slots += group_padded_slots
        groups[str(int(required_shape))] = {
            "unique_keys": len(positions),
            "padded_encoder_slots": group_padded_slots,
            "timing_sec": timing,
        }
        for key in aggregate:
            aggregate[key] += timing[key]

    if all_embeddings is None:
        raise RuntimeError("No keyed frame embeddings were produced")
    return all_embeddings, {
        "aggregate": aggregate,
        "padded_encoder_slots": padded_encoder_slots,
        "groups": groups,
    }


def action_normalization_stats(
    all_actions: np.ndarray,
    norm_starts_path: Path,
    action_steps: int,
    frameskip: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    norm_starts = np.asarray(np.load(norm_starts_path), dtype=np.int64)
    norm_blocks = read_action_blocks_vectorized(
        all_actions, norm_starts, action_steps, frameskip
    )
    mean, std = action_block_stats(norm_blocks)
    return mean, std, len(norm_starts)


@torch.inference_mode()
def encode_actions(
    model: torch.nn.Module,
    all_actions: np.ndarray,
    starts: np.ndarray,
    action_steps: int,
    sequence_length: int,
    frameskip: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    log_every: int,
    source_transition_offset: int,
    source_transition_count: int,
) -> np.ndarray:
    raw = read_action_blocks_vectorized(
        all_actions, starts, action_steps, frameskip
    ).astype(np.float32, copy=False)
    raw = (raw - action_mean.astype(np.float32)) / action_std.astype(np.float32)
    pad = np.zeros((len(starts), 1, raw.shape[-1]), dtype=np.float32)
    normalized = np.concatenate([raw, pad], axis=1)
    if normalized.shape[1] != sequence_length:
        raise RuntimeError("Action sequence has the wrong temporal length")

    action_embeddings: np.ndarray | None = None
    if source_transition_offset % batch_size != 0:
        raise ValueError(
            "Exact action validation requires --start-offset to be aligned to "
            f"transition-batch-size={batch_size}"
        )
    batch_specs: list[tuple[int, int, int]] = []
    offset = 0
    while offset < len(starts):
        source_batch_start = source_transition_offset + offset
        required_count = min(
            batch_size, source_transition_count - source_batch_start
        )
        valid_count = min(required_count, len(starts) - offset)
        batch_specs.append((offset, valid_count, required_count))
        offset += valid_count

    num_batches = len(batch_specs)
    for batch_idx, (offset, valid_count, required_count) in enumerate(batch_specs):
        stop = offset + valid_count
        action_np = normalized[offset:stop]
        if valid_count < required_count:
            action_np = np.concatenate(
                (
                    action_np,
                    np.repeat(
                        action_np[-1:], required_count - valid_count, axis=0
                    ),
                ),
                axis=0,
            )
        action = torch.as_tensor(
            action_np, device=device, dtype=torch.float32
        )
        embeddings = (
            model.action_encoder(action)[:valid_count].detach().cpu().numpy()
        )
        if action_embeddings is None:
            action_embeddings = np.empty(
                (len(starts), *embeddings.shape[1:]), dtype=embeddings.dtype
            )
        action_embeddings[offset:stop] = embeddings
        if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == num_batches:
            print(
                f"  [action] batch {batch_idx + 1}/{num_batches} "
                f"({stop}/{len(starts)} transitions)",
                flush=True,
            )

    if action_embeddings is None:
        raise RuntimeError("No action embeddings were produced")
    return action_embeddings


def validate_reference(
    reference_path: Path,
    emb: np.ndarray,
    act_emb: np.ndarray,
    starts: np.ndarray,
    reference_offset: int,
) -> dict[str, Any]:
    generated = {
        "emb": emb,
        "act_emb": act_emb,
        "region_starts": starts,
    }
    arrays: dict[str, Any] = {}
    passed = True
    with np.load(reference_path) as reference:
        expected_keys = ["emb", "act_emb", "region_starts"]
        missing = [key for key in expected_keys if key not in reference.files]
        if missing:
            raise KeyError(f"Reference cache is missing keys: {missing}")
        for key, actual in generated.items():
            expected = np.asarray(
                reference[key][reference_offset : reference_offset + len(starts)]
            )
            same_shape = expected.shape == actual.shape
            same_dtype = expected.dtype == actual.dtype
            exact = same_shape and same_dtype and np.array_equal(expected, actual)
            max_abs = None
            mismatch_count = None
            if same_shape and np.issubdtype(actual.dtype, np.number):
                delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
                max_abs = float(delta.max(initial=0.0))
                mismatch_count = int(np.count_nonzero(expected != actual))
            arrays[key] = {
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "shape_equal": same_shape,
                "dtype_equal": same_dtype,
                "array_equal": exact,
                "mismatch_count": mismatch_count,
                "max_abs_error": max_abs,
                "generated_sha256": sha256_array(actual),
                "reference_sha256": sha256_array(expected),
            }
            passed = passed and exact
    return {"passed": passed, "arrays": arrays}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.report is None:
        args.report = args.output.with_suffix(args.output.suffix + ".report.json")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    if args.history_size < 1 or args.num_preds < 1:
        raise ValueError("history-size and num-preds must be positive")
    if args.transition_batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch sizes must be positive and num-workers nonnegative")
    if args.frame_batch_size != 0:
        raise ValueError(
            "Exact-output mode determines visual batch shapes from the original "
            "transition batches; leave --frame-batch-size at 0"
        )
    if args.start_offset % args.transition_batch_size != 0:
        raise ValueError(
            "--start-offset must align to --transition-batch-size in exact mode"
        )
    if args.prefetch_factor < 1:
        raise ValueError("prefetch-factor must be positive")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.data_file = args.data_file.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.action_norm_starts = args.action_norm_starts.resolve(strict=True)
    if args.reference_cache is not None:
        args.reference_cache = args.reference_cache.resolve(strict=True)
    if args.starts is not None:
        args.starts = args.starts.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    run_t0 = time.perf_counter()
    model_t0 = time.perf_counter()
    model = load_encoder(str(args.checkpoint), device, cache_dir=None)
    model_load_sec = time.perf_counter() - model_t0

    starts, starts_source, source_transition_count = load_starts(args)
    sequence_length = args.history_size + args.num_preds
    action_steps = sequence_length - 1
    with h5py.File(args.data_file, "r", swmr=True) as h5:
        if args.pixel_key not in h5 or args.action_key not in h5:
            raise KeyError(
                f"Required keys {args.pixel_key!r}, {args.action_key!r} are not both present"
            )
        raw_action_dim = int(h5[args.action_key].shape[1])
        num_pixels = int(h5[args.pixel_key].shape[0])
        all_actions = np.asarray(h5[args.action_key][:], dtype=np.float32)
    frameskip = infer_frameskip(model, raw_action_dim, args.frameskip)
    index_t0 = time.perf_counter()
    keyed_frames, required_batch_shapes, inverse, global_unique_count = (
        build_exact_unique_index(
            starts,
            sequence_length,
            frameskip,
            args.transition_batch_size,
            args.start_offset,
            source_transition_count,
        )
    )
    index_sec = time.perf_counter() - index_t0
    if keyed_frames[0] < 0 or keyed_frames[-1] >= num_pixels:
        raise IndexError(
            f"Frame range [{keyed_frames[0]}, {keyed_frames[-1]}] exceeds dataset"
        )
    frame_slots = int(len(starts) * sequence_length)
    exact_key_count = int(len(keyed_frames))
    exact_key_speedup = frame_slots / exact_key_count
    print(
        f"[index] transitions={len(starts)} frame_slots={frame_slots} "
        f"unique_frames={global_unique_count} exact_shape_keys={exact_key_count} "
        f"duplicate_fraction={1.0 - global_unique_count / frame_slots:.4%} "
        f"visual_work_ratio={exact_key_speedup:.3f}x",
        flush=True,
    )

    norm_t0 = time.perf_counter()
    action_mean, action_std, norm_start_count = action_normalization_stats(
        all_actions,
        args.action_norm_starts,
        action_steps,
        frameskip,
    )
    action_norm_sec = time.perf_counter() - norm_t0

    frame_embeddings, frame_timing = encode_exact_frame_keys(
        model,
        args.data_file,
        args.pixel_key,
        keyed_frames,
        required_batch_shapes,
        device,
        img_size=args.img_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        log_every=args.log_every,
        chunk_aware=not args.no_chunk_aware_read,
    )
    padded_encoder_slots = int(frame_timing["padded_encoder_slots"])

    action_t0 = time.perf_counter()
    action_embeddings = encode_actions(
        model,
        all_actions,
        starts,
        action_steps,
        sequence_length,
        frameskip,
        action_mean,
        action_std,
        device,
        args.transition_batch_size,
        args.log_every,
        args.start_offset,
        source_transition_count,
    )
    action_encode_sec = time.perf_counter() - action_t0

    reconstruct_t0 = time.perf_counter()
    embeddings = frame_embeddings[inverse].reshape(
        len(starts), sequence_length, frame_embeddings.shape[-1]
    )
    reconstruct_sec = time.perf_counter() - reconstruct_t0

    validation = None
    if args.reference_cache is not None:
        validation_t0 = time.perf_counter()
        validation = validate_reference(
            args.reference_cache,
            embeddings,
            action_embeddings,
            starts,
            args.start_offset,
        )
        validation_sec = time.perf_counter() - validation_t0
        print(
            f"[validate] exact_array_identity={validation['passed']}", flush=True
        )
    else:
        validation_sec = 0.0

    save_t0 = time.perf_counter()
    np.savez_compressed(
        args.output,
        emb=embeddings,
        act_emb=action_embeddings,
        region_starts=starts,
    )
    save_sec = time.perf_counter() - save_t0
    total_sec = time.perf_counter() - run_t0
    report = {
        "schema_version": 1,
        "method": "unique_timestep_exact_reconstruction",
        "output_semantics": "array-level exact; NPZ byte hash may differ",
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "h5py": h5py.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "config": vars(args),
        "starts_source": starts_source,
        "counts": {
            "transitions": len(starts),
            "source_transition_offset": args.start_offset,
            "source_transition_count": source_transition_count,
            "sequence_length": sequence_length,
            "frameskip": frameskip,
            "frame_slots": frame_slots,
            "unique_frames": global_unique_count,
            "exact_frame_batch_shape_keys": exact_key_count,
            "actual_padded_encoder_frame_slots": padded_encoder_slots,
            "extra_shape_keys_for_bitwise_exactness": exact_key_count
            - global_unique_count,
            "eliminated_duplicate_frames": frame_slots - exact_key_count,
            "duplicate_fraction": 1.0 - global_unique_count / frame_slots,
            "action_norm_starts": norm_start_count,
        },
        "efficiency": {
            "ideal_unique_frame_speedup": frame_slots / global_unique_count,
            "exact_shape_key_speedup_before_padding": exact_key_speedup,
            "actual_kernel_frame_speedup_after_padding": frame_slots
            / padded_encoder_slots,
            "actual_kernel_frame_reduction_fraction": 1.0
            - padded_encoder_slots / frame_slots,
            "exact_shape_key_reduction_fraction_before_padding": 1.0
            - exact_key_count / frame_slots,
        },
        "timing_sec": {
            "model_load": model_load_sec,
            "unique_index": index_sec,
            "action_normalization_stats": action_norm_sec,
            "unique_frame_encode": frame_timing,
            "action_encode": action_encode_sec,
            "inverse_reconstruct": reconstruct_sec,
            "reference_validation": validation_sec,
            "npz_save": save_sec,
            "total": total_sec,
            "core_precompute": frame_timing["aggregate"]["total_sec"]
            + action_encode_sec
            + reconstruct_sec,
        },
        "arrays": {
            "emb": {
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "sha256": sha256_array(embeddings),
            },
            "act_emb": {
                "shape": list(action_embeddings.shape),
                "dtype": str(action_embeddings.dtype),
                "sha256": sha256_array(action_embeddings),
            },
            "region_starts": {
                "shape": list(starts.shape),
                "dtype": str(starts.dtype),
                "sha256": sha256_array(starts),
            },
        },
        "validation": validation,
    }
    atomic_write_json(args.report, report)
    print(f"[saved] cache={args.output}", flush=True)
    print(f"[saved] report={args.report}", flush=True)
    print(f"[timing] total={total_sec:.2f}s", flush=True)

    if validation is not None and not validation["passed"]:
        raise SystemExit("Reference validation failed; see JSON report")


if __name__ == "__main__":
    main()
