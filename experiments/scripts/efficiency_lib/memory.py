"""GPU memory measurement utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUMemorySnapshot:
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    @property
    def peak_allocated_gb(self) -> float:
        return self.peak_allocated_bytes / (1024**3)

    @property
    def peak_reserved_gb(self) -> float:
        return self.peak_reserved_bytes / (1024**3)

    def as_dict(self) -> dict[str, float]:
        return {
            "peak_allocated_bytes": float(self.peak_allocated_bytes),
            "peak_reserved_bytes": float(self.peak_reserved_bytes),
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
        }


def _cuda_index(device: torch.device | str | None) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None or str(device) == "cpu":
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    # Ensure a CUDA context exists on this device before peak-stat calls.
    _ = torch.zeros(1, device=f"cuda:{index}")
    return index


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    index = _cuda_index(device)
    if index is None:
        return
    torch.cuda.reset_peak_memory_stats(index)


def read_peak_memory(device: torch.device | str | None = None) -> GPUMemorySnapshot:
    index = _cuda_index(device)
    if index is None:
        return GPUMemorySnapshot(0, 0)
    return GPUMemorySnapshot(
        int(torch.cuda.max_memory_allocated(index)),
        int(torch.cuda.max_memory_reserved(index)),
    )


def release_training_gpu_state(device: torch.device | str | None = None) -> None:
    """Drop cached allocations and reset peak counters before the next benchmark phase."""
    index = _cuda_index(device)
    if index is None:
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)
