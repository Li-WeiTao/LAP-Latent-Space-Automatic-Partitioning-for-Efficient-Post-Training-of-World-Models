"""Joint-training dataset aligned to LAP cache global reference starts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import Dataset


class GlobalReferenceStartDataset(Dataset):
    """Load predictor windows by global HDF5 frame id (LeWM cache semantics).

    ``stable_worldmodel`` clip indices require a contiguous ``num_steps * frameskip``
    span inside each episode.  The official LeWM train pool includes ~36k transitions
    near episode tails that satisfy ``valid_transition_starts`` but not that stricter
    span rule.  This loader matches the encoding path used to build LAP caches:
    strided pixel/state frames plus ``num_steps - 1`` action blocks with a zero-padded
    final action step.
    """

    def __init__(
        self,
        h5_path: Path,
        global_starts: np.ndarray,
        *,
        num_steps: int,
        frameskip: int,
        keys_to_load: list[str],
        keys_to_cache: dict[str, np.ndarray],
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2 for LeWM windows")
        self.h5_path = Path(h5_path).expanduser().resolve(strict=True)
        self.global_starts = np.asarray(global_starts, dtype=np.int64)
        if self.global_starts.ndim != 1:
            raise ValueError("global_starts must be one-dimensional")
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.keys_to_load = list(keys_to_load)
        self.keys_to_cache = keys_to_cache
        self.transform = transform
        self._h5: h5py.File | None = None
        self._action_offsets = (
            np.arange(self.num_steps - 1, dtype=np.int64)[:, None] * self.frameskip
            + np.arange(self.frameskip, dtype=np.int64)[None, :]
        )

    def __len__(self) -> int:
        return len(self.global_starts)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def _load_column_frames(self, col: str, frame_ids: np.ndarray) -> np.ndarray:
        if col in self.keys_to_cache:
            return np.asarray(self.keys_to_cache[col][frame_ids])
        self._open()
        return np.asarray(self._h5[col][frame_ids])

    def _load_window(self, global_start: int) -> dict[str, torch.Tensor]:
        start = int(global_start)
        frame_ids = start + np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        steps: dict[str, torch.Tensor] = {}
        for col in self.keys_to_load:
            if col == "action":
                block_rows = start + self._action_offsets
                blocks = self._load_column_frames(col, block_rows.reshape(-1))
                blocks = blocks.reshape(self.num_steps - 1, self.frameskip, -1)
                action = blocks.reshape(self.num_steps - 1, -1).astype(np.float32)
                action = np.concatenate(
                    [
                        action,
                        np.zeros((1, action.shape[-1]), dtype=np.float32),
                    ],
                    axis=0,
                )
                steps[col] = torch.from_numpy(action)
                continue

            data = self._load_column_frames(col, frame_ids)
            tensor = torch.from_numpy(np.asarray(data))
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
            steps[col] = tensor

        if self.transform is not None:
            steps = self.transform(steps)
        return steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._load_window(int(self.global_starts[idx]))


def build_joint_train_pool_dataset(
    *,
    data_file: Path,
    dataset_name: str,
    train_pool_starts: Path,
    history_size: int,
    num_preds: int,
    frameskip: int,
    img_size: int,
    resolve_state_key,
) -> GlobalReferenceStartDataset:
    """Build a joint-training dataset over the curated LAP train pool."""
    data_file = data_file.resolve(strict=True)
    num_steps = history_size + num_preds
    pool_starts = np.load(train_pool_starts)

    state_key = resolve_state_key(data_file, dataset_name)
    keys_to_load = ["pixels", "action"]
    keys_to_cache = ["action"]
    if state_key is not None:
        keys_to_load.append(state_key)
        keys_to_cache.append(state_key)

    reference = swm.data.load_dataset(
        str(data_file),
        transform=None,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
    )
    cached = {key: reference.get_col_data(key) for key in keys_to_cache}

    from utils import get_column_normalizer, get_img_preprocessor

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    ]
    for col in keys_to_cache:
        transforms.append(get_column_normalizer(reference, col, col))
    transform = spt.data.transforms.Compose(*transforms)

    return GlobalReferenceStartDataset(
        data_file,
        pool_starts,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=cached,
        transform=transform,
    )
