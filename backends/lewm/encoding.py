"""LeWM adapters for the generic accelerated latent-cache encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from lap.interfaces.encoding import EncodingSelection

try:
    import hdf5plugin  # noqa: F401  # Registers optional HDF5 filters.
except ImportError:
    pass


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HDF5FrameDataset(Dataset):
    """Lazy HDF5 reader that coalesces all frames from each touched chunk."""

    def __init__(
        self,
        h5_path: str | Path,
        pixel_key: str,
        frame_ids: np.ndarray,
        *,
        chunk_aware: bool,
    ) -> None:
        self.h5_path = str(Path(h5_path).resolve())
        self.pixel_key = str(pixel_key)
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.chunk_aware = bool(chunk_aware)
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r", swmr=True) as handle:
            dataset = handle[self.pixel_key]
            self.num_frames = int(dataset.shape[0])
            self.chunk_length = int(dataset.chunks[0]) if dataset.chunks else 1

    def __len__(self) -> int:
        return len(self.frame_ids)

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
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError):
                pass

    def __getitem__(self, position: int) -> tuple[np.ndarray, np.int64]:
        frame_id = int(self.frame_ids[position])
        frame = np.asarray(self._open()[self.pixel_key][frame_id])
        return frame, np.int64(frame_id)

    def __getitems__(
        self, positions: list[int]
    ) -> list[tuple[np.ndarray, np.int64]]:
        if not self.chunk_aware:
            return [self[position] for position in positions]
        positions_array = np.asarray(positions, dtype=np.int64)
        frame_ids = self.frame_ids[positions_array]
        chunk_ids = frame_ids // self.chunk_length
        output: list[tuple[np.ndarray, np.int64] | None] = [None] * len(frame_ids)
        dataset = self._open()[self.pixel_key]
        for chunk_id in np.unique(chunk_ids):
            members = np.flatnonzero(chunk_ids == chunk_id)
            lower = int(chunk_id * self.chunk_length)
            upper = min(lower + self.chunk_length, self.num_frames)
            block = np.asarray(dataset[lower:upper])
            for member in members:
                frame_id = int(frame_ids[member])
                output[int(member)] = (block[frame_id - lower], np.int64(frame_id))
        if any(item is None for item in output):
            raise RuntimeError("not every requested HDF5 frame was read")
        return output  # type: ignore[return-value]


class LeWMHDF5TransitionDataset:
    """Configurable LeWM task dataset; no task name or geometry is assumed."""

    def __init__(
        self,
        *,
        data_file: str | Path,
        starts: str | Path | None = None,
        reference_cache: str | Path | None = None,
        action_norm_starts: str | Path | None = None,
        pixel_key: str = "pixels",
        action_key: str = "action",
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 0,
    ) -> None:
        if starts is None and reference_cache is None:
            raise ValueError("dataset config requires starts or reference_cache")
        if starts is not None and reference_cache is not None:
            raise ValueError("pass only one of starts and reference_cache")
        if history_size < 1 or num_preds < 1 or frameskip < 0:
            raise ValueError("history_size/num_preds must be positive; frameskip nonnegative")
        self.data_file = Path(data_file).expanduser().resolve(strict=True)
        self.starts_path = (
            None if starts is None else Path(starts).expanduser().resolve(strict=True)
        )
        self.reference_cache = (
            None
            if reference_cache is None
            else Path(reference_cache).expanduser().resolve(strict=True)
        )
        self.action_norm_starts = (
            None
            if action_norm_starts is None
            else Path(action_norm_starts).expanduser().resolve(strict=True)
        )
        self.pixel_key = str(pixel_key)
        self.action_key = str(action_key)
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        with h5py.File(self.data_file, "r", swmr=True) as handle:
            missing = [
                key for key in (self.pixel_key, self.action_key) if key not in handle
            ]
            if missing:
                raise KeyError(f"dataset is missing required HDF5 keys: {missing}")
            self.num_frames = int(handle[self.pixel_key].shape[0])
            self.raw_action_dim = int(handle[self.action_key].shape[1])

    @property
    def sequence_length(self) -> int:
        return self.history_size + self.num_preds

    def set_frameskip(self, frameskip: int) -> None:
        if frameskip < 1:
            raise ValueError("resolved frameskip must be positive")
        self.frameskip = int(frameskip)

    def _all_starts(self) -> tuple[np.ndarray, str]:
        if self.reference_cache is not None:
            with np.load(self.reference_cache, allow_pickle=False) as data:
                if "region_starts" not in data.files:
                    raise KeyError("reference cache does not contain region_starts")
                starts = np.asarray(data["region_starts"], dtype=np.int64)
            return starts, f"{self.reference_cache}::region_starts"
        assert self.starts_path is not None
        return np.asarray(np.load(self.starts_path), dtype=np.int64), str(self.starts_path)

    def describe(self) -> Mapping[str, Any]:
        return {
            "adapter": f"{type(self).__module__}:{type(self).__name__}",
            "data_file": str(self.data_file),
            "pixel_key": self.pixel_key,
            "action_key": self.action_key,
            "history_size": self.history_size,
            "num_preds": self.num_preds,
            "frameskip": self.frameskip,
        }

    def make_selection(
        self, *, start_offset: int = 0, max_samples: int = 0
    ) -> EncodingSelection:
        if self.frameskip < 1:
            raise RuntimeError("encoder adapter did not resolve frameskip")
        all_starts, source = self._all_starts()
        if all_starts.ndim != 1 or len(all_starts) == 0:
            raise ValueError("starts must be a non-empty one-dimensional array")
        if start_offset < 0 or start_offset >= len(all_starts):
            raise ValueError("start_offset is outside the source transition pool")
        stop = len(all_starts)
        if max_samples > 0:
            stop = min(start_offset + max_samples, stop)
        selected = all_starts[start_offset:stop]
        frame_offsets = (
            np.arange(self.sequence_length, dtype=np.int64) * self.frameskip
        )
        frame_ids = selected[:, None] + frame_offsets[None, :]
        if frame_ids.min() < 0 or frame_ids.max() >= self.num_frames:
            raise IndexError("selected transition window exceeds the pixel dataset")
        selection = EncodingSelection(
            sample_ids=selected,
            frame_ids=frame_ids,
            source_offset=start_offset,
            source_count=len(all_starts),
            metadata={"starts_source": source},
        )
        selection.validate()
        return selection

    def make_frame_dataset(
        self, frame_ids: np.ndarray, *, chunk_aware: bool
    ) -> HDF5FrameDataset:
        return HDF5FrameDataset(
            self.data_file,
            self.pixel_key,
            frame_ids,
            chunk_aware=chunk_aware,
        )

    def read_actions(self) -> np.ndarray:
        with h5py.File(self.data_file, "r", swmr=True) as handle:
            return np.asarray(handle[self.action_key][:], dtype=np.float32)

    def normalization_starts(self, fallback: np.ndarray) -> np.ndarray:
        if self.action_norm_starts is None:
            return np.asarray(fallback, dtype=np.int64)
        return np.asarray(np.load(self.action_norm_starts), dtype=np.int64)


def _read_action_blocks(
    actions: np.ndarray,
    starts: np.ndarray,
    num_steps: int,
    frameskip: int,
) -> np.ndarray:
    offsets = (
        np.arange(num_steps, dtype=np.int64)[:, None] * frameskip
        + np.arange(frameskip, dtype=np.int64)[None, :]
    ).reshape(-1)
    rows = actions[np.asarray(starts, dtype=np.int64)[:, None] + offsets[None, :]]
    return rows.reshape(len(starts), num_steps, -1)


def _action_stats(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = blocks.reshape(-1, blocks.shape[-1]).astype(np.float64)
    flat = flat[np.isfinite(flat).all(axis=1)]
    return flat.mean(axis=0, keepdims=True), flat.std(axis=0, keepdims=True) + 1e-8


class LeWMEncoderAdapter:
    """Official/file LeWM encoder with exact historical cache semantics."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        frameskip: int = 0,
        checkpoint_cache_dir: str | None = None,
        model_family: str = "lewm",
    ) -> None:
        if img_size < 1 or frameskip < 0:
            raise ValueError("img_size must be positive and frameskip nonnegative")
        self.img_size = int(img_size)
        self.frameskip = int(frameskip)
        self.checkpoint_cache_dir = checkpoint_cache_dir
        self.model_family = model_family

    def describe(self) -> Mapping[str, Any]:
        return {
            "adapter": f"{type(self).__module__}:{type(self).__name__}",
            "img_size": self.img_size,
            "requested_frameskip": self.frameskip,
        }

    def load(self, pretrained_model: Any, device: torch.device) -> Any:
        if isinstance(pretrained_model, torch.nn.Module):
            model = pretrained_model
        else:
            checkpoint = Path(str(pretrained_model)).expanduser()
            if checkpoint.exists() and checkpoint.suffix == ".ckpt":
                from backends.lewm.checkpoint_compat import load_jepa_object_checkpoint

                model = load_jepa_object_checkpoint(
                    checkpoint,
                    model_family=self.model_family,
                    map_location="cpu",
                )
            else:
                import stable_worldmodel as stable_wm

                model = stable_wm.wm.utils.load_pretrained(
                    str(pretrained_model), cache_dir=self.checkpoint_cache_dir
                )
        model = model.to(device).eval()
        model.requires_grad_(False)
        if hasattr(model, "interpolate_pos_encoding"):
            model.interpolate_pos_encoding = True
        encoder = getattr(model, "encoder", None)
        config = getattr(encoder, "config", None)
        if config is not None:
            for attr, value in (
                ("output_attentions", False),
                ("output_hidden_states", False),
                ("return_dict", True),
                ("torchscript", False),
                ("use_bfloat16", False),
            ):
                if not hasattr(config, attr):
                    setattr(config, attr, value)
        return model

    def prepare_dataset(
        self, dataset: LeWMHDF5TransitionDataset, model: Any
    ) -> None:
        if not isinstance(dataset, LeWMHDF5TransitionDataset):
            raise TypeError("LeWMEncoderAdapter requires a LeWM HDF5 dataset adapter")
        frameskip = self.frameskip or dataset.frameskip
        if frameskip < 1:
            patch_embed = getattr(
                getattr(model, "action_encoder", None), "patch_embed", None
            )
            expected_dim = getattr(patch_embed, "in_channels", None)
            if expected_dim is None:
                frameskip = 1
            elif expected_dim % dataset.raw_action_dim:
                raise ValueError(
                    "cannot infer frameskip: action encoder input is not divisible "
                    "by raw action dimension"
                )
            else:
                frameskip = max(1, int(expected_dim // dataset.raw_action_dim))
        dataset.set_frameskip(frameskip)

    @torch.inference_mode()
    def encode_frames(
        self, model: Any, frames: torch.Tensor, device: torch.device
    ) -> np.ndarray:
        if not torch.is_tensor(frames) or frames.ndim != 4:
            raise ValueError("LeWM frames must have shape [B, H, W, C]")
        pixels = frames.permute(0, 3, 1, 2).float().div_(255.0)
        if tuple(pixels.shape[-2:]) != (self.img_size, self.img_size):
            pixels = functional.interpolate(
                pixels,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = IMAGENET_MEAN.to(dtype=pixels.dtype, device=pixels.device)
        std = IMAGENET_STD.to(dtype=pixels.dtype, device=pixels.device)
        pixels = ((pixels - mean) / std).to(device)
        output = model.encode({"pixels": pixels.unsqueeze(1)})
        return output["emb"][:, 0].detach().cpu().numpy()

    @torch.inference_mode()
    def encode_auxiliary(
        self,
        model: Any,
        dataset: LeWMHDF5TransitionDataset,
        selection: EncodingSelection,
        *,
        device: torch.device,
        batch_size: int,
        exact_batch_shapes: bool,
        log_every: int,
    ) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        actions = dataset.read_actions()
        action_steps = dataset.sequence_length - 1
        norm_starts = dataset.normalization_starts(selection.sample_ids)
        norm_blocks = _read_action_blocks(
            actions, norm_starts, action_steps, dataset.frameskip
        )
        action_mean, action_std = _action_stats(norm_blocks)
        raw = _read_action_blocks(
            actions, selection.sample_ids, action_steps, dataset.frameskip
        ).astype(np.float32, copy=False)
        normalized = (raw - action_mean.astype(np.float32)) / action_std.astype(
            np.float32
        )
        normalized = np.concatenate(
            (
                normalized,
                np.zeros((len(normalized), 1, normalized.shape[-1]), np.float32),
            ),
            axis=1,
        )
        if normalized.shape[1] != dataset.sequence_length:
            raise RuntimeError("normalized LeWM action sequence has wrong length")
        if exact_batch_shapes and selection.source_offset % batch_size:
            raise ValueError(
                "exact LeWM action encoding requires source_offset aligned to batch_size"
            )

        source_count = (
            len(selection.sample_ids)
            if selection.source_count is None
            else selection.source_count
        )
        result: np.ndarray | None = None
        offset = 0
        batch_index = 0
        while offset < len(selection.sample_ids):
            source_batch_start = selection.source_offset + offset
            required_count = (
                min(batch_size, source_count - source_batch_start)
                if exact_batch_shapes
                else batch_size
            )
            valid_count = min(required_count, len(selection.sample_ids) - offset)
            stop = offset + valid_count
            action_batch = normalized[offset:stop]
            if valid_count < required_count:
                action_batch = np.concatenate(
                    (
                        action_batch,
                        np.repeat(
                            action_batch[-1:], required_count - valid_count, axis=0
                        ),
                    ),
                    axis=0,
                )
            action_tensor = torch.as_tensor(
                action_batch, device=device, dtype=torch.float32
            )
            encoded = (
                model.action_encoder(action_tensor)[:valid_count]
                .detach()
                .cpu()
                .numpy()
            )
            if result is None:
                result = np.empty(
                    (len(selection.sample_ids), *encoded.shape[1:]), encoded.dtype
                )
            result[offset:stop] = encoded
            offset = stop
            batch_index += 1
            # The generic CLI owns progress output so --json keeps stdout clean.
        if result is None:
            raise RuntimeError("no LeWM action embeddings were produced")
        return {"act_emb": result}, {
            "frameskip": dataset.frameskip,
            "normalization_samples": len(norm_starts),
        }

    def cache_arrays(
        self,
        latent_windows: np.ndarray,
        selection: EncodingSelection,
        auxiliary: Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        return {
            "emb": np.asarray(latent_windows),
            "act_emb": np.asarray(auxiliary["act_emb"]),
            "region_starts": np.asarray(selection.sample_ids, dtype=np.int64),
        }


def make_hdf5_transition_dataset(**config: Any) -> LeWMHDF5TransitionDataset:
    """Factory entry point used by ``lap-cache --dataset-factory``."""

    return LeWMHDF5TransitionDataset(**config)


def make_encoder(**config: Any) -> LeWMEncoderAdapter:
    """Factory entry point used by ``lap-cache --encoder-factory``."""

    return LeWMEncoderAdapter(**config)
