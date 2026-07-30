"""Small importable adapters used to test the public cache CLI."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from lap.interfaces.encoding import EncodingSelection


class _Frames(Dataset):
    def __init__(self, frame_ids: np.ndarray):
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int):
        frame_id = int(self.frame_ids[index])
        return torch.tensor([frame_id], dtype=torch.float32), np.int64(frame_id)


class FakeEncodingDataset:
    def __init__(self, *, offset: int = 0):
        self.offset = int(offset)

    def describe(self) -> Mapping[str, Any]:
        return {"adapter": "fake", "offset": self.offset}

    def make_selection(self, *, start_offset: int = 0, max_samples: int = 0):
        frame_ids = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        sample_ids = np.arange(3, dtype=np.int64) + self.offset
        stop = (
            len(sample_ids)
            if max_samples == 0
            else min(start_offset + max_samples, len(sample_ids))
        )
        return EncodingSelection(
            sample_ids=sample_ids[start_offset:stop],
            frame_ids=frame_ids[start_offset:stop],
            source_offset=start_offset,
            source_count=len(sample_ids),
        )

    def make_frame_dataset(self, frame_ids, *, chunk_aware: bool):
        return _Frames(frame_ids)


class FakeEncoderAdapter:
    def __init__(self, *, scale: float = 1.0):
        self.scale = float(scale)

    def describe(self) -> Mapping[str, Any]:
        return {"adapter": "fake", "scale": self.scale}

    def load(self, pretrained_model: Any, device: Any) -> Any:
        return {"model": pretrained_model}

    def prepare_dataset(self, dataset, model) -> None:
        return None

    def encode_frames(self, model, frames, device) -> np.ndarray:
        value = frames.detach().cpu().numpy()[:, 0] * self.scale
        return np.column_stack((value, value**2)).astype(np.float32)

    def encode_auxiliary(
        self,
        model,
        dataset,
        selection,
        *,
        device,
        batch_size,
        exact_batch_shapes,
        log_every,
    ):
        return {"condition": selection.sample_ids[:, None].astype(np.float32)}, {}

    def cache_arrays(self, latent_windows, selection, auxiliary):
        return {
            "emb": latent_windows,
            "condition": auxiliary["condition"],
            "sample_ids": selection.sample_ids,
        }


def make_dataset(**config: Any) -> FakeEncodingDataset:
    return FakeEncodingDataset(**config)


def make_encoder(**config: Any) -> FakeEncoderAdapter:
    return FakeEncoderAdapter(**config)
