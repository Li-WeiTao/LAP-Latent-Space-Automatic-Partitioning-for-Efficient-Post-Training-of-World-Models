#!/usr/bin/env python3
"""Region-specific predictor finetuning for the trajectory deviation experiment.

Step 1 (README): define natural regions on the full dataset (train-global is
only used as the action-normalization reference). Step 2: finetune one predictor
per region on all transitions in that region, with a frozen encoder.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import (  # noqa: E402
    DATASETS,
    choose_state_key,
    finite_rows,
    load_encoder,
    read_columns,
    resolve_tworoom_region_splits,
)
from predictor_rule_drift import (  # noqa: E402
    action_block_stats,
    infer_frameskip,
    read_action_blocks,
    read_sequence_dataset,
    valid_transition_starts,
)
from gauge_drift import preprocess_pixels  # noqa: E402
from backends.lewm.finetuning import (  # noqa: E402
    LeWMTrainConfig as TrainConfig,
    eval_predictor_loss,
    freeze_encoder_path,
    save_region_predictor,
    set_training_seed,
    train_region_predictor,
    trainable_predictor_params,
    unfreeze_predictor_path,
)


REGION_SPLITS = (
    "common",
    "doorway_corridor",
    "left_room",
    "near_wall",
    "right_room",
)
GLOBAL_FT_EMBED_REGIONS = ("left_room", "doorway_corridor", "right_room")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def seq_len(cfg: TrainConfig) -> int:
    return cfg.history_size + cfg.num_preds


def train_global_pool(all_starts: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.split_seed)
    perm = rng.permutation(all_starts)
    train_n = int(round(len(perm) * cfg.train_fraction))
    return np.sort(perm[:train_n])


def region_masks_at_starts(
    h5_path: Path,
    spec,
    starts: np.ndarray,
    state_key: str,
    *,
    split_mode: str = "quantile",
    quantile_reference_starts: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float] | None]:
    """Natural-region masks at transition starts.

    ``split_mode='geometry'`` uses fixed TwoRoom wall/border constants.
    For ``split_mode='quantile'``, pass ``quantile_reference_starts`` to compute
    boundaries from train split only (no test leak).
    """
    with h5py.File(h5_path, "r") as h5:
        n = h5[state_key].shape[0]
        split_keys = sorted(
            set(spec.state_keys)
            | {"pos_agent", "pos_target", "goal_state", "goal_proprio", state_key}
        )
        full_cols = read_columns(h5, split_keys, np.arange(n))
        state_at_starts = np.asarray(h5[state_key][starts], dtype=np.float64)

        if spec.name == "tworoom":
            ref_pos = None
            if quantile_reference_starts is not None:
                ref_pos = np.asarray(h5[state_key][quantile_reference_starts], dtype=np.float64)
            raw, thresholds = resolve_tworoom_region_splits(
                full_cols,
                split_mode=split_mode,
                quantile_reference_pos=ref_pos,
            )
        else:
            thresholds = None
            raw = spec.split_fn(full_cols)

    masks = {
        name: np.asarray(mask[starts], dtype=bool) & finite_rows(state_at_starts)
        for name, mask in raw.items()
        if len(mask) == n
    }
    return masks, thresholds


def embedding_cache_path(out_dir: Path, region: str, name_prefix: str = "") -> Path:
    return out_dir / f"P_{name_prefix}{region}_embeddings.npz"


def save_embedding_cache(
    path: Path,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    region_starts: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        emb=emb.numpy(),
        act_emb=act_emb.numpy(),
        region_starts=region_starts,
    )


def load_embedding_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    data = np.load(path)
    return (
        torch.from_numpy(np.asarray(data["emb"])),
        torch.from_numpy(np.asarray(data["act_emb"])),
        np.asarray(data["region_starts"]),
    )


def starts_match_cached(cached_starts: np.ndarray, region_starts: np.ndarray) -> bool:
    if len(cached_starts) != len(region_starts):
        return False
    if len(region_starts) == 0:
        return True
    return np.array_equal(cached_starts, region_starts)


def read_action_blocks_vectorized(
    actions: np.ndarray,
    starts: np.ndarray,
    num_steps: int,
    frameskip: int,
) -> np.ndarray:
    """Read flattened action blocks with the exact legacy ordering, without Python loops."""
    actions = np.asarray(actions)
    starts = np.asarray(starts, dtype=np.int64)
    step_offsets = (
        np.arange(num_steps, dtype=np.int64)[:, None] * frameskip
        + np.arange(frameskip, dtype=np.int64)[None, :]
    ).reshape(-1)
    rows = actions[starts[:, None] + step_offsets[None, :]]
    return rows.reshape(len(starts), num_steps, -1)


class TransitionSequenceDataset(Dataset):
    """Exact-start, lazy-per-worker HDF5 reader for embedding precomputation.

    Unlike the official dataset split, this dataset consumes the already selected
    ``starts`` verbatim.  It therefore changes only the I/O backend, not the
    experiment's transition pool, ordering, frameskip, or action representation.
    """

    def __init__(
        self,
        h5_path: Path,
        pixel_key: str,
        starts: np.ndarray,
        seq_len: int,
        frameskip: int,
    ) -> None:
        self.h5_path = str(h5_path)
        self.pixel_key = pixel_key
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.frameskip = int(frameskip)
        self.act_steps = self.seq_len - 1
        self._h5: h5py.File | None = None

        # Actions are small for the supported datasets and the official loader
        # follows the same cache-in-memory strategy. This removes action HDF5 I/O
        # from every worker item while preserving float32 source values exactly.
        with h5py.File(self.h5_path, "r", swmr=True) as h5:
            self.actions = np.asarray(h5["action"][:], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.starts)

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )
        return self._h5

    def __getstate__(self) -> dict:
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

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.int64]:
        start = int(self.starts[index])
        pixel_stop = start + (self.seq_len - 1) * self.frameskip + 1
        pixels = np.asarray(
            self._open()[self.pixel_key][start:pixel_stop:self.frameskip]
        )
        if len(pixels) != self.seq_len:
            raise RuntimeError(
                f"Short pixel sequence at start={start}: "
                f"expected {self.seq_len}, got {len(pixels)}"
            )

        action_stop = start + self.act_steps * self.frameskip
        actions = self.actions[start:action_stop].reshape(self.act_steps, -1)
        return pixels, actions, np.int64(start)


def load_global_train_embeddings_from_region_caches(
    source_dir: Path,
    pool_starts: np.ndarray,
    *,
    name_prefix: str = "train_",
    regions: tuple[str, ...] = GLOBAL_FT_EMBED_REGIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge rooms3-exclusive region caches to cover the full train split."""
    emb_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    for region in regions:
        path = embedding_cache_path(source_dir, region, name_prefix)
        if not path.exists():
            raise FileNotFoundError(f"Missing region embedding cache: {path}")
        data = np.load(path)
        emb_parts.append(np.asarray(data["emb"]))
        act_parts.append(np.asarray(data["act_emb"]))
        start_parts.append(np.asarray(data["region_starts"], dtype=np.int64))
        print(f"  [embed-cache] {region}: {len(start_parts[-1])} transitions from {path.name}", flush=True)

    starts = np.concatenate(start_parts)
    order = np.argsort(starts, kind="stable")
    merged_starts = starts[order]
    if not starts_match_cached(merged_starts, pool_starts):
        raise RuntimeError(
            "Merged region embedding caches do not match requested train pool "
            f"({len(merged_starts)} cached vs {len(pool_starts)} requested)"
        )

    emb = torch.from_numpy(np.concatenate(emb_parts, axis=0)[order])
    act_emb = torch.from_numpy(np.concatenate(act_parts, axis=0)[order])
    print(
        f"  [embed-cache] merged {len(pool_starts)} train transitions from {source_dir}",
        flush=True,
    )
    return emb, act_emb


@torch.no_grad()
def precompute_embeddings_legacy(
    model: torch.nn.Module,
    h5_path: Path,
    spec,
    starts: np.ndarray,
    cfg: TrainConfig,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    log_every: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frozen (emb, act_emb) tensors for transition starts."""
    model.eval()
    t_len = seq_len(cfg)
    act_steps = t_len - 1
    emb_chunks: list[torch.Tensor] = []
    act_chunks: list[torch.Tensor] = []
    num_batches = (len(starts) + cfg.batch_size - 1) // cfg.batch_size

    with h5py.File(h5_path, "r") as h5:
        frameskip = infer_frameskip(model, int(h5["action"].shape[1]), cfg.frameskip)
        for batch_idx, offset in enumerate(range(0, len(starts), cfg.batch_size)):
            batch_starts = starts[offset : offset + cfg.batch_size]
            pixels_np = read_sequence_dataset(
                h5[spec.pixel_key], batch_starts, t_len, frameskip
            )
            actions_raw = read_action_blocks(
                h5["action"], batch_starts, act_steps, frameskip
            ).astype(np.float32)
            actions_raw = (actions_raw - action_mean.astype(np.float32)) / action_std.astype(
                np.float32
            )
            pad = np.zeros((actions_raw.shape[0], 1, actions_raw.shape[2]), dtype=np.float32)
            actions_np = np.concatenate([actions_raw, pad], axis=1)

            b, t_steps = pixels_np.shape[:2]
            flat_pixels = pixels_np.reshape(b * t_steps, *pixels_np.shape[2:])
            pixels = preprocess_pixels(flat_pixels, device, cfg.img_size)
            pixels = pixels.reshape(b, t_steps, *pixels.shape[1:])

            batch = {
                "pixels": pixels,
                "action": torch.as_tensor(actions_np, device=device, dtype=pixels.dtype),
            }
            out = model.encode(batch)
            emb_chunks.append(out["emb"].detach().cpu())
            act_chunks.append(out["act_emb"].detach().cpu())
            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == num_batches:
                print(
                    f"  [encode] batch {batch_idx + 1}/{num_batches} "
                    f"({min(offset + cfg.batch_size, len(starts))}/{len(starts)} samples)",
                    flush=True,
                )

    return torch.cat(emb_chunks, dim=0), torch.cat(act_chunks, dim=0)


@torch.no_grad()
def precompute_embeddings_dataloader(
    model: torch.nn.Module,
    h5_path: Path,
    spec,
    starts: np.ndarray,
    cfg: TrainConfig,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    log_every: int = 50,
    *,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute embeddings with exact-start DataLoader HDF5 prefetching."""
    model.eval()
    t_len = seq_len(cfg)
    dataset = TransitionSequenceDataset(
        h5_path=h5_path,
        pixel_key=spec.pixel_key,
        starts=starts,
        seq_len=t_len,
        frameskip=infer_frameskip(model, dataset_action_dim(h5_path), cfg.frameskip),
    )
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory and device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    loader = DataLoader(dataset, **loader_kwargs)

    emb_chunks: list[torch.Tensor] = []
    act_chunks: list[torch.Tensor] = []
    num_batches = (len(starts) + cfg.batch_size - 1) // cfg.batch_size
    action_mean32 = action_mean.astype(np.float32)
    action_std32 = action_std.astype(np.float32)
    load_sec = 0.0
    encode_sec = 0.0
    total_start = time.perf_counter()
    iterator = iter(loader)

    for batch_idx in range(num_batches):
        wait_start = time.perf_counter()
        pixels_cpu, actions_cpu, returned_starts = next(iterator)
        load_sec += time.perf_counter() - wait_start

        offset = batch_idx * cfg.batch_size
        expected_starts = starts[offset : offset + len(returned_starts)]
        if not np.array_equal(returned_starts.numpy(), expected_starts):
            raise RuntimeError("DataLoader changed transition-start ordering")

        encode_start = time.perf_counter()
        pixels_np = pixels_cpu.numpy()
        actions_raw = actions_cpu.numpy().astype(np.float32, copy=False)
        actions_raw = (actions_raw - action_mean32) / action_std32
        pad = np.zeros((actions_raw.shape[0], 1, actions_raw.shape[2]), dtype=np.float32)
        actions_np = np.concatenate([actions_raw, pad], axis=1)

        b, t_steps = pixels_np.shape[:2]
        flat_pixels = pixels_np.reshape(b * t_steps, *pixels_np.shape[2:])
        pixels = preprocess_pixels(flat_pixels, device, cfg.img_size)
        pixels = pixels.reshape(b, t_steps, *pixels.shape[1:])
        batch = {
            "pixels": pixels,
            "action": torch.as_tensor(actions_np, device=device, dtype=pixels.dtype),
        }
        out = model.encode(batch)
        emb_chunks.append(out["emb"].detach().cpu())
        act_chunks.append(out["act_emb"].detach().cpu())
        encode_sec += time.perf_counter() - encode_start

        if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == num_batches:
            done = min(offset + cfg.batch_size, len(starts))
            print(
                f"  [encode:dataloader] batch {batch_idx + 1}/{num_batches} "
                f"({done}/{len(starts)} samples)",
                flush=True,
            )

    total_sec = time.perf_counter() - total_start
    print(
        f"  [encode:dataloader] timing total={total_sec:.2f}s "
        f"loader_wait={load_sec:.2f}s preprocess+encode={encode_sec:.2f}s "
        f"workers={num_workers} prefetch={prefetch_factor}",
        flush=True,
    )
    return torch.cat(emb_chunks, dim=0), torch.cat(act_chunks, dim=0)


def dataset_action_dim(h5_path: Path) -> int:
    with h5py.File(h5_path, "r", swmr=True) as h5:
        return int(h5["action"].shape[1])


def precompute_embeddings(
    model: torch.nn.Module,
    h5_path: Path,
    spec,
    starts: np.ndarray,
    cfg: TrainConfig,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    log_every: int = 50,
    *,
    backend: str = "dataloader",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the exact legacy or accelerated embedding backend."""
    if backend == "legacy":
        return precompute_embeddings_legacy(
            model,
            h5_path,
            spec,
            starts,
            cfg,
            action_mean,
            action_std,
            device,
            log_every,
        )
    if backend == "dataloader":
        return precompute_embeddings_dataloader(
            model,
            h5_path,
            spec,
            starts,
            cfg,
            action_mean,
            action_std,
            device,
            log_every,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tworoom")
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--state-key", default=None)
    parser.add_argument(
        "--checkpoint",
        default="/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt",
    )
    parser.add_argument("--checkpoint-cache-dir", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/tworoom/results/tworoom_trajectory_predictors"),
    )
    parser.add_argument("--regions", nargs="+", default=list(REGION_SPLITS))
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42, help="Predictor FT RNG seed (shuffle + torch)")
    parser.add_argument("--max-starts", type=int, default=0, help="Cap train-global starts (0=all)")
    parser.add_argument("--min-region-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--restrict-to-train-split",
        action="store_true",
        help="Train on natural_region ∩ train_global split (not full dataset)",
    )
    parser.add_argument(
        "--region-split-mode",
        choices=("quantile", "geometry"),
        default="quantile",
        help="TwoRoom region definition: data quantiles (legacy) or fixed task geometry",
    )
    parser.add_argument(
        "--train-only-region-quantiles",
        action="store_true",
        help="(quantile mode) Compute region boundaries from train split starts only (avoids test leakage)",
    )
    parser.add_argument(
        "--predictor-prefix",
        default="",
        help="Insert prefix into saved predictor names, e.g. 'train_' -> P_train_common_object.ckpt",
    )
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Ignore cached region embeddings and re-run encoder precompute",
    )
    parser.add_argument(
        "--encode-log-every",
        type=int,
        default=50,
        help="Print embedding precompute progress every N batches",
    )
    parser.add_argument(
        "--embedding-loader",
        choices=("dataloader", "legacy"),
        default="dataloader",
        help="Embedding I/O backend; dataloader preserves starts/order but prefetches HDF5 reads",
    )
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=4,
        help="DataLoader workers for embedding precompute (default: 4)",
    )
    parser.add_argument(
        "--encode-prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per embedding DataLoader worker",
    )
    parser.add_argument(
        "--no-encode-pin-memory",
        action="store_true",
        help="Disable pinned host memory for embedding precompute",
    )
    parser.add_argument(
        "--save-epochs",
        default="",
        help="Comma-separated epochs to save intermediate checkpoints, e.g. 20,30,40,50",
    )
    parser.add_argument(
        "--select-best-by-eval",
        action="store_true",
        help="Export P_*_object.ckpt from the epoch with minimum eval_loss (not last epoch)",
    )
    parser.add_argument(
        "--train-global-predictor",
        action="store_true",
        help="Finetune one predictor on the full train split (compute-matched global FT).",
    )
    parser.add_argument(
        "--global-predictor-name",
        default="global_ft",
        help="Region label used in output filenames for --train-global-predictor",
    )
    parser.add_argument(
        "--embedding-source-dir",
        type=Path,
        default=None,
        help="Reuse P_train_{region}_embeddings.npz from this dir (rooms3 merge for global FT)",
    )
    parser.add_argument(
        "--prepare-starts-only",
        action="store_true",
        help=(
            "Save the global training starts and one P_{prefix}{region}_starts.npy "
            "file per requested region, then exit before encoding or training. "
            "This bootstraps the lossless unique-timestep cache builder."
        ),
    )
    return parser.parse_args()


def parse_save_epochs(text: str) -> list[int]:
    if not text.strip():
        return []
    epochs = [int(x.strip()) for x in text.split(",") if x.strip()]
    if any(e <= 0 for e in epochs):
        raise ValueError("--save-epochs must contain positive integers")
    return sorted(set(epochs))


def train_one_predictor(
    *,
    base_model: torch.nn.Module,
    h5_path: Path,
    spec,
    pool_starts: np.ndarray,
    region: str,
    split_label: str,
    cfg: TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    name_prefix: str,
    manifest: dict,
) -> None:
    ckpt_name = f"P_{name_prefix}{region}_object.ckpt"
    manifest["regions"][region] = {
        "num_samples": int(len(pool_starts)),
        "output_ckpt": str(args.out_dir / ckpt_name),
    }

    if len(pool_starts) < cfg.min_region_samples:
        print(
            f"[skip] {region}: only {len(pool_starts)} samples "
            f"(min={cfg.min_region_samples})",
            flush=True,
        )
        manifest["regions"][region]["status"] = "skipped"
        return

    print(f"[train] {region}: {len(pool_starts)} {split_label} transitions", flush=True)
    cache_path = embedding_cache_path(args.out_dir, region, name_prefix)
    if (
        args.embedding_source_dir is not None
        and args.train_global_predictor
        and region == args.global_predictor_name
    ):
        emb, act_emb = load_global_train_embeddings_from_region_caches(
            args.embedding_source_dir,
            pool_starts,
            name_prefix=name_prefix,
        )
        save_embedding_cache(cache_path, emb, act_emb, pool_starts)
        print(f"  [cache] saved merged embeddings to {cache_path}", flush=True)
    elif not args.force_reencode and cache_path.exists():
        emb, act_emb, cached_starts = load_embedding_cache(cache_path)
        if starts_match_cached(cached_starts, pool_starts):
            print(f"  [cache] loaded embeddings from {cache_path}", flush=True)
        else:
            print(f"  [cache] stale cache at {cache_path}, re-encoding", flush=True)
            emb, act_emb = precompute_embeddings(
                base_model,
                h5_path,
                spec,
                pool_starts,
                cfg,
                action_mean,
                action_std,
                device,
                log_every=args.encode_log_every,
                backend=args.embedding_loader,
                num_workers=args.encode_workers,
                prefetch_factor=args.encode_prefetch_factor,
                pin_memory=not args.no_encode_pin_memory,
            )
            save_embedding_cache(cache_path, emb, act_emb, pool_starts)
            print(f"  [cache] saved embeddings to {cache_path}", flush=True)
    else:
        emb, act_emb = precompute_embeddings(
            base_model,
            h5_path,
            spec,
            pool_starts,
            cfg,
            action_mean,
            action_std,
            device,
            log_every=args.encode_log_every,
            backend=args.embedding_loader,
            num_workers=args.encode_workers,
            prefetch_factor=args.encode_prefetch_factor,
            pin_memory=not args.no_encode_pin_memory,
        )
        save_embedding_cache(cache_path, emb, act_emb, pool_starts)
        print(f"  [cache] saved embeddings to {cache_path}", flush=True)

    if device.type == "cuda":
        base_model.to("cpu")
        torch.cuda.empty_cache()

    save_epochs = parse_save_epochs(args.save_epochs)
    region_model, stats = train_region_predictor(
        base_model,
        emb,
        act_emb,
        cfg,
        device,
        save_epochs=save_epochs,
        checkpoint_dir=args.out_dir,
        region=region,
        name_prefix=name_prefix,
        select_best_by_eval=args.select_best_by_eval,
    )

    if device.type == "cuda":
        base_model.to(device)
    ckpt_path = args.out_dir / ckpt_name
    save_region_predictor(
        region_model,
        ckpt_path,
        metadata={
            "region": region,
            "num_samples": int(len(pool_starts)),
            "region_starts": pool_starts,
            "training_data": split_label,
            "action_norm_reference": "train_global",
            "stats": stats,
            "action_mean": action_mean,
            "action_std": action_std,
        },
    )
    np.save(args.out_dir / f"P_{name_prefix}{region}_starts.npy", pool_starts)
    manifest["regions"][region]["status"] = "trained"
    manifest["regions"][region]["final_loss"] = stats["final_loss"]
    manifest["regions"][region]["best_epoch"] = stats.get("best_epoch")
    manifest["regions"][region]["best_eval_loss"] = stats.get("best_eval_loss")
    manifest["regions"][region]["saved_checkpoints"] = stats.get("saved_checkpoints", {})
    print(
        f"  saved {ckpt_path}  best_epoch={stats.get('best_epoch')}  "
        f"best_eval_loss={stats['final_loss']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.encode_workers < 0:
        raise ValueError("--encode-workers must be non-negative")
    if args.encode_prefetch_factor <= 0:
        raise ValueError("--encode-prefetch-factor must be positive")
    cfg = TrainConfig(
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        seed=args.seed,
        max_starts=args.max_starts,
        min_region_samples=args.min_region_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    set_training_seed(cfg.seed)

    spec = DATASETS[args.dataset]
    h5_path = args.data_file or (args.data_root / spec.default_file)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    base_model = load_encoder(args.checkpoint, device, args.checkpoint_cache_dir)
    base_model.train()

    with h5py.File(h5_path, "r") as h5:
        state_key = choose_state_key(h5, spec, args.state_key)
        all_starts = valid_transition_starts(
            h5,
            spec,
            state_key,
            seq_len(cfg),
            infer_frameskip(base_model, int(h5["action"].shape[1]), cfg.frameskip),
            cfg.max_starts,
            cfg.seed,
        )

    train_starts = train_global_pool(all_starts, cfg)
    train_mask = np.isin(all_starts, train_starts)
    split_mode = args.region_split_mode
    if split_mode == "geometry" and args.train_only_region_quantiles:
        print(
            "[split] --train-only-region-quantiles ignored in geometry mode (fixed constants)",
            flush=True,
        )
    quantile_ref = train_starts if args.train_only_region_quantiles and split_mode == "quantile" else None
    region_masks, region_thresholds = region_masks_at_starts(
        h5_path,
        spec,
        all_starts,
        state_key,
        split_mode=split_mode,
        quantile_reference_starts=quantile_ref,
    )
    if split_mode == "geometry":
        print("[split] TwoRoom regions from fixed task geometry (wall_center=112, wall_width=10)", flush=True)
    elif args.train_only_region_quantiles:
        print(
            f"[split] region quantiles from train split only ({len(train_starts)} reference starts)",
            flush=True,
        )

    name_prefix = args.predictor_prefix
    if args.restrict_to_train_split and not name_prefix:
        name_prefix = "train_"

    if args.prepare_starts_only:
        if args.train_global_predictor:
            raise ValueError(
                "--prepare-starts-only is for regional cache preparation and "
                "cannot be combined with --train-global-predictor"
            )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if region_thresholds is not None:
            threshold_name = (
                "geometry_region_thresholds.npy"
                if split_mode == "geometry"
                else "train_region_thresholds.npy"
            )
            np.save(args.out_dir / threshold_name, region_thresholds)
        np.save(args.out_dir / "train_global_reference_starts.npy", train_starts)

        prepared_regions: dict[str, dict[str, object]] = {}
        for region in args.regions:
            if region not in region_masks:
                raise KeyError(
                    f"Region '{region}' not found. Available: {sorted(region_masks)}"
                )
            region_mask = region_masks[region]
            if args.restrict_to_train_split:
                region_mask = region_mask & train_mask
            region_starts = all_starts[region_mask]
            starts_path = args.out_dir / f"P_{name_prefix}{region}_starts.npy"
            np.save(starts_path, region_starts)
            prepared_regions[region] = {
                "num_starts": int(len(region_starts)),
                "starts_file": str(starts_path),
            }
            print(
                f"[prepare-starts] {region}: {len(region_starts)} -> {starts_path}",
                flush=True,
            )

        starts_manifest = {
            "mode": "prepare_starts_only",
            "dataset": args.dataset,
            "data_file": str(h5_path),
            "checkpoint": args.checkpoint,
            "train_config": asdict(cfg),
            "num_all_starts": int(len(all_starts)),
            "num_train_global_reference_starts": int(len(train_starts)),
            "training_data": (
                "train_split_intersection"
                if args.restrict_to_train_split
                else "full_dataset"
            ),
            "predictor_prefix": name_prefix,
            "region_split_mode": split_mode,
            "region_thresholds": region_thresholds,
            "regions": prepared_regions,
        }
        with (args.out_dir / "starts_only_manifest.json").open("w") as handle:
            json.dump(json_ready(starts_manifest), handle, indent=2)
            handle.write("\n")
        print("[prepare-starts] complete; no embeddings or predictors were trained")
        return

    with h5py.File(h5_path, "r") as h5:
        frameskip = infer_frameskip(base_model, int(h5["action"].shape[1]), cfg.frameskip)
        all_actions = np.asarray(h5["action"][:], dtype=np.float32)
    action_blocks = read_action_blocks_vectorized(
        all_actions,
        train_starts,
        seq_len(cfg) - 1,
        frameskip,
    )
    action_mean, action_std = action_block_stats(action_blocks)
    del action_blocks, all_actions

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "data_file": str(h5_path),
        "checkpoint": args.checkpoint,
        "train_config": asdict(cfg),
        "num_all_starts": int(len(all_starts)),
        "num_train_global_reference_starts": int(len(train_starts)),
        "action_norm_reference": "train_global",
        "training_data": (
            "train_split_global_ft"
            if args.train_global_predictor
            else ("train_split_intersection" if args.restrict_to_train_split else "full_dataset")
        ),
        "predictor_prefix": name_prefix,
        "region_split_mode": split_mode,
        "train_global_predictor": args.train_global_predictor,
        "global_predictor_name": args.global_predictor_name if args.train_global_predictor else None,
        "embedding_source_dir": str(args.embedding_source_dir) if args.embedding_source_dir else None,
        "embedding_loader": args.embedding_loader,
        "encode_workers": args.encode_workers,
        "encode_prefetch_factor": args.encode_prefetch_factor,
        "encode_pin_memory": not args.no_encode_pin_memory,
        "compute_matched_equiv_epochs": 64.2 if args.train_global_predictor and cfg.epochs == 65 else None,
        "region_quantile_reference": (
            "train_split_starts"
            if args.train_only_region_quantiles and split_mode == "quantile"
            else ("fixed_geometry" if split_mode == "geometry" else "full_dataset")
        ),
        "region_thresholds": region_thresholds,
        "regions": {},
    }
    if region_thresholds is not None:
        threshold_name = (
            "geometry_region_thresholds.npy"
            if split_mode == "geometry"
            else "train_region_thresholds.npy"
        )
        np.save(args.out_dir / threshold_name, region_thresholds)
    np.save(args.out_dir / "train_global_reference_starts.npy", train_starts)

    if args.train_global_predictor:
        if not args.restrict_to_train_split:
            raise ValueError("--train-global-predictor requires --restrict-to-train-split")
        train_one_predictor(
            base_model=base_model,
            h5_path=h5_path,
            spec=spec,
            pool_starts=train_starts,
            region=args.global_predictor_name,
            split_label="train-global",
            cfg=cfg,
            args=args,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
            name_prefix=name_prefix,
            manifest=manifest,
        )
    else:
        for region in args.regions:
            if region not in region_masks:
                raise KeyError(f"Region '{region}' not found. Available: {sorted(region_masks)}")

            region_mask = region_masks[region]
            if args.restrict_to_train_split:
                region_mask = region_mask & train_mask
            region_starts = all_starts[region_mask]
            split_label = "train∩region" if args.restrict_to_train_split else "full-dataset"
            train_one_predictor(
                base_model=base_model,
                h5_path=h5_path,
                spec=spec,
                pool_starts=region_starts,
                region=region,
                split_label=split_label,
                cfg=cfg,
                args=args,
                device=device,
                action_mean=action_mean,
                action_std=action_std,
                name_prefix=name_prefix,
                manifest=manifest,
            )

    with (args.out_dir / "manifest.json").open("w") as f:
        json.dump(json_ready(manifest), f, indent=2)
    print(f"Wrote manifest to {args.out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
