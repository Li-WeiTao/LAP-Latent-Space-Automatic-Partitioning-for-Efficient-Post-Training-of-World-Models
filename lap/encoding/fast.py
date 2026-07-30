"""Dataset- and model-agnostic unique-frame latent-cache encoder."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from lap.interfaces.encoding import (
    EncodingDataset,
    EncodingSelection,
    LatentEncoderAdapter,
)

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class FastEncodingConfig:
    """Hardware and exactness settings independent of dataset/model identity."""

    device: str = "cuda"
    transition_batch_size: int = 128
    frame_batch_size: int = 512
    exact_batch_shapes: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    cpu_threads: int = 4
    chunk_aware: bool = True
    log_every: int = 50
    start_offset: int = 0
    max_samples: int = 0

    def validate(self) -> None:
        if self.transition_batch_size < 1 or self.frame_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.num_workers < 0 or self.cpu_threads < 1:
            raise ValueError("num_workers must be nonnegative and cpu_threads positive")
        if self.prefetch_factor < 1 or self.log_every < 1:
            raise ValueError("prefetch_factor and log_every must be positive")
        if self.start_offset < 0 or self.max_samples < 0:
            raise ValueError("start_offset and max_samples must be nonnegative")
        if self.exact_batch_shapes and self.start_offset % self.transition_batch_size:
            raise ValueError(
                "exact batch-shape mode requires start_offset aligned to "
                "transition_batch_size"
            )


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_unique_frame_index(
    selection: EncodingSelection,
    *,
    transition_batch_size: int,
    exact_batch_shapes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build unique frame keys plus inverse indices for exact reconstruction."""

    selection.validate()
    frame_windows = np.asarray(selection.frame_ids, dtype=np.int64)
    flat_frames = frame_windows.reshape(-1)
    global_unique_count = int(np.unique(flat_frames).size)
    if not exact_batch_shapes:
        unique_frames, inverse = np.unique(flat_frames, return_inverse=True)
        required_batch_shapes = np.zeros(len(unique_frames), dtype=np.int64)
        return unique_frames, required_batch_shapes, inverse, global_unique_count

    source_count = (
        len(selection.sample_ids)
        if selection.source_count is None
        else int(selection.source_count)
    )
    sequence_length = frame_windows.shape[1]
    transition_ids = selection.source_offset + np.arange(
        len(selection.sample_ids), dtype=np.int64
    )
    batch_offsets = (
        transition_ids // transition_batch_size
    ) * transition_batch_size
    transition_counts = np.minimum(
        transition_batch_size, source_count - batch_offsets
    )
    visual_batch_shapes = np.repeat(
        transition_counts * sequence_length, sequence_length
    )
    keys = np.column_stack((flat_frames, visual_batch_shapes))
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    return (
        unique_keys[:, 0],
        unique_keys[:, 1],
        inverse,
        global_unique_count,
    )


def _batch_length(batch: Any) -> int:
    if isinstance(batch, Mapping):
        lengths = {_batch_length(value) for value in batch.values()}
        if len(lengths) != 1:
            raise ValueError("structured frame batch fields have different lengths")
        return lengths.pop()
    return len(batch)


def _pad_batch(batch: Any, target: int) -> Any:
    current = _batch_length(batch)
    if current == target:
        return batch
    if current == 0 or current > target:
        raise ValueError(f"cannot pad batch of length {current} to {target}")
    if torch.is_tensor(batch):
        repeats = [1] * batch.ndim
        repeats[0] = target - current
        return torch.cat((batch, batch[-1:].repeat(*repeats)), dim=0)
    if isinstance(batch, np.ndarray):
        return np.concatenate(
            (batch, np.repeat(batch[-1:], target - current, axis=0)), axis=0
        )
    if isinstance(batch, Mapping):
        return {key: _pad_batch(value, target) for key, value in batch.items()}
    raise TypeError(
        "frame batches must be tensors, numpy arrays, or mappings of those types"
    )


def _encode_frame_group(
    dataset: EncodingDataset,
    encoder: LatentEncoderAdapter,
    model: Any,
    frame_ids: np.ndarray,
    *,
    target_batch_size: int,
    device: torch.device,
    config: FastEncodingConfig,
    log: LogFn,
) -> tuple[np.ndarray, dict[str, float]]:
    frame_dataset = dataset.make_frame_dataset(
        frame_ids, chunk_aware=config.chunk_aware
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": target_batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=config.prefetch_factor
        )
    loader = DataLoader(frame_dataset, **loader_kwargs)
    num_batches = (len(frame_dataset) + target_batch_size - 1) // target_batch_size
    result: np.ndarray | None = None
    wait_sec = 0.0
    compute_sec = 0.0
    total_start = time.perf_counter()

    iterator = iter(loader)
    for batch_index in range(num_batches):
        wait_start = time.perf_counter()
        frames, returned_ids = next(iterator)
        wait_sec += time.perf_counter() - wait_start
        offset = batch_index * target_batch_size
        valid_count = len(returned_ids)
        stop = offset + valid_count
        returned = np.asarray(returned_ids)
        expected = np.asarray(frame_ids[offset:stop])
        if not np.array_equal(returned, expected):
            raise RuntimeError("frame dataset changed unique-frame ordering")

        compute_start = time.perf_counter()
        padded = _pad_batch(frames, target_batch_size)
        encoded = np.asarray(encoder.encode_frames(model, padded, device))
        if encoded.ndim != 2 or encoded.shape[0] != target_batch_size:
            raise ValueError("encoder.encode_frames must return shape [B, D]")
        encoded = encoded[:valid_count]
        if result is None:
            result = np.empty((len(frame_dataset), encoded.shape[1]), encoded.dtype)
        result[offset:stop] = encoded
        compute_sec += time.perf_counter() - compute_start
        if (batch_index + 1) % config.log_every == 0 or batch_index + 1 == num_batches:
            log(
                f"[frames] batch {batch_index + 1}/{num_batches} "
                f"({stop}/{len(frame_dataset)})"
            )

    if result is None:
        raise RuntimeError("no frame embeddings were produced")
    return result, {
        "total_sec": time.perf_counter() - total_start,
        "loader_wait_sec": wait_sec,
        "preprocess_encode_d2h_sec": compute_sec,
    }


def _validate_reference(
    reference_path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    offset: int,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    with np.load(reference_path, allow_pickle=False) as reference:
        missing = [key for key in arrays if key not in reference.files]
        if missing:
            raise KeyError(f"reference cache is missing generated keys: {missing}")
        sample_count = len(next(iter(arrays.values())))
        for key, actual_value in arrays.items():
            actual = np.asarray(actual_value)
            expected = np.asarray(reference[key][offset : offset + sample_count])
            same_shape = actual.shape == expected.shape
            same_dtype = actual.dtype == expected.dtype
            exact = same_shape and same_dtype and np.array_equal(actual, expected)
            details[key] = {
                "shape_equal": same_shape,
                "dtype_equal": same_dtype,
                "array_equal": exact,
                "generated_sha256": _sha256_array(actual),
                "reference_sha256": _sha256_array(expected),
                "mismatch_count": (
                    int(np.count_nonzero(actual != expected)) if same_shape else None
                ),
            }
            passed = passed and exact
    return {"passed": passed, "arrays": details}


class FastLatentCacheEncoder:
    """Build a backend cache from arbitrary dataset and encoder adapters."""

    def __init__(self, config: FastEncodingConfig, *, log: LogFn = print):
        config.validate()
        self.config = config
        self.log = log

    def encode(
        self,
        *,
        dataset: EncodingDataset,
        encoder: LatentEncoderAdapter,
        pretrained_model: Any,
        output: str | Path,
        report: str | Path | None = None,
        reference_cache: str | Path | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output)
        report_path = (
            output_path.with_suffix(output_path.suffix + ".report.json")
            if report is None
            else Path(report)
        )
        for path in (output_path, report_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")

        torch.set_num_threads(self.config.cpu_threads)
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        total_start = time.perf_counter()

        model_start = time.perf_counter()
        model = encoder.load(pretrained_model, device)
        encoder.prepare_dataset(dataset, model)
        model_sec = time.perf_counter() - model_start

        selection = dataset.make_selection(
            start_offset=self.config.start_offset,
            max_samples=self.config.max_samples,
        )
        selection.validate()
        index_start = time.perf_counter()
        keyed_frames, required_shapes, inverse, global_unique_count = (
            build_unique_frame_index(
                selection,
                transition_batch_size=self.config.transition_batch_size,
                exact_batch_shapes=self.config.exact_batch_shapes,
            )
        )
        index_sec = time.perf_counter() - index_start
        frame_slots = int(np.asarray(selection.frame_ids).size)
        self.log(
            f"[index] samples={len(selection.sample_ids)} frame_slots={frame_slots} "
            f"unique_frames={global_unique_count} encoded_keys={len(keyed_frames)}"
        )

        all_embeddings: np.ndarray | None = None
        group_reports: dict[str, Any] = {}
        padded_slots = 0
        if self.config.exact_batch_shapes:
            group_shapes = np.unique(required_shapes)
        else:
            group_shapes = np.asarray([self.config.frame_batch_size], dtype=np.int64)
        frame_total_start = time.perf_counter()
        for target_shape in group_shapes:
            if self.config.exact_batch_shapes:
                positions = np.flatnonzero(required_shapes == target_shape)
            else:
                positions = np.arange(len(keyed_frames))
            target_batch_size = int(target_shape)
            self.log(
                f"[batch-shape] batch={target_batch_size} keys={len(positions)}"
            )
            group_embeddings, timing = _encode_frame_group(
                dataset,
                encoder,
                model,
                keyed_frames[positions],
                target_batch_size=target_batch_size,
                device=device,
                config=self.config,
                log=self.log,
            )
            if all_embeddings is None:
                all_embeddings = np.empty(
                    (len(keyed_frames), group_embeddings.shape[1]),
                    dtype=group_embeddings.dtype,
                )
            all_embeddings[positions] = group_embeddings
            padded = (
                (len(positions) + target_batch_size - 1)
                // target_batch_size
                * target_batch_size
            )
            padded_slots += padded
            group_reports[str(target_batch_size)] = {
                "encoded_keys": len(positions),
                "padded_encoder_slots": padded,
                "timing_sec": timing,
            }
        if all_embeddings is None:
            raise RuntimeError("no latent embeddings were produced")
        frame_total_sec = time.perf_counter() - frame_total_start

        reconstruct_start = time.perf_counter()
        latent_windows = all_embeddings[inverse].reshape(
            len(selection.sample_ids),
            selection.frame_ids.shape[1],
            all_embeddings.shape[1],
        )
        reconstruct_sec = time.perf_counter() - reconstruct_start

        auxiliary_start = time.perf_counter()
        auxiliary, auxiliary_report = encoder.encode_auxiliary(
            model,
            dataset,
            selection,
            device=device,
            batch_size=self.config.transition_batch_size,
            exact_batch_shapes=self.config.exact_batch_shapes,
            log_every=self.config.log_every,
        )
        auxiliary_sec = time.perf_counter() - auxiliary_start
        arrays = {
            str(key): np.asarray(value)
            for key, value in encoder.cache_arrays(
                latent_windows, selection, auxiliary
            ).items()
        }
        if not arrays:
            raise ValueError("encoder.cache_arrays returned no arrays")
        lengths = {key: len(value) for key, value in arrays.items()}
        if set(lengths.values()) != {len(selection.sample_ids)}:
            raise ValueError(f"cache arrays have inconsistent sample counts: {lengths}")

        validation = None
        validation_sec = 0.0
        if reference_cache is not None:
            validation_start = time.perf_counter()
            validation = _validate_reference(
                Path(reference_cache), arrays, offset=selection.source_offset
            )
            validation_sec = time.perf_counter() - validation_start
            self.log(f"[validate] exact_array_identity={validation['passed']}")

        save_start = time.perf_counter()
        _atomic_save_npz(output_path, arrays)
        save_sec = time.perf_counter() - save_start
        total_sec = time.perf_counter() - total_start
        report_value: dict[str, Any] = {
            "schema_version": 2,
            "method": "unique_frame_inverse_reconstruction",
            "dataset": dict(dataset.describe()),
            "encoder": dict(encoder.describe()),
            "pretrained_model": str(pretrained_model),
            "config": asdict(self.config),
            "selection": {
                **selection.metadata,
                "samples": len(selection.sample_ids),
                "source_offset": selection.source_offset,
                "source_count": selection.source_count,
                "sequence_length": selection.frame_ids.shape[1],
            },
            "counts": {
                "frame_slots": frame_slots,
                "unique_frames": global_unique_count,
                "encoded_frame_keys": len(keyed_frames),
                "padded_encoder_slots": padded_slots,
                "eliminated_duplicate_frames": frame_slots - len(keyed_frames),
            },
            "efficiency": {
                "ideal_unique_frame_speedup": frame_slots / global_unique_count,
                "encoded_key_speedup_before_padding": frame_slots / len(keyed_frames),
                "actual_kernel_frame_speedup_after_padding": frame_slots / padded_slots,
                "actual_kernel_frame_reduction_fraction": 1.0 - padded_slots / frame_slots,
            },
            "timing_sec": {
                "model_load_and_prepare": model_sec,
                "unique_index": index_sec,
                "unique_frame_encode": frame_total_sec,
                "inverse_reconstruct": reconstruct_sec,
                "auxiliary_encode": auxiliary_sec,
                "reference_validation": validation_sec,
                "npz_save": save_sec,
                "total": total_sec,
            },
            "frame_groups": group_reports,
            "auxiliary": dict(auxiliary_report),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _sha256_array(value),
                }
                for key, value in arrays.items()
            },
            "validation": validation,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
            },
        }
        _atomic_write_json(report_path, report_value)
        self.log(f"[saved] cache={output_path}")
        self.log(f"[saved] report={report_path}")
        if validation is not None and not validation["passed"]:
            raise RuntimeError("reference cache validation failed")
        return report_value
