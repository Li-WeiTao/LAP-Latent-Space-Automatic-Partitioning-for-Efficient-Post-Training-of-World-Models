#!/usr/bin/env python3
"""Linear decodability of geometry region labels from LeWM latent representations.

Each sample is one encoded 192-d latent vector (deduplicated by global timestep).
Partitions:
  - rooms3: left_room, doorway_corridor, right_room
  - priority5: mutually exclusive doorway, near_wall\\doorway, common\\doorway,
    left_room\\(near_wall∪common), right_room\\(near_wall∪common) via priority cascade
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS, choose_state_key, finite_rows, tworoom_geometry_thresholds  # noqa: E402

ROOMS3_LABELS = ("left_room", "doorway_corridor", "right_room")
PRIORITY5_LABELS = ("doorway_corridor", "near_wall", "common", "right_room", "left_room")
PARTITIONS = {
    "rooms3": {
        "labels": ROOMS3_LABELS,
        "cache_regions": ROOMS3_LABELS,
    },
    "priority5": {
        "labels": PRIORITY5_LABELS,
        "cache_regions": PRIORITY5_LABELS,
    },
}
DEFAULT_EMBED_DIR = THIS_DIR / "results" / "tworoom_geometry_train_region_predictors"
DEFAULT_FRAME_SKIP = 5


def rooms3_labels_from_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = tworoom_geometry_thresholds()
    x = xy[:, 0]
    y = xy[:, 1]
    wall_lo = thresholds["wall_lo"]
    wall_hi = thresholds["wall_hi"]
    left = x < wall_lo
    right = x > wall_hi
    doorway = (x >= wall_lo) & (x <= wall_hi)
    valid = left | doorway | right
    labels = np.full(len(xy), -1, dtype=np.int64)
    labels[left] = 0
    labels[doorway] = 1
    labels[right] = 2
    overlap = int((left.astype(int) + doorway.astype(int) + right.astype(int) > 1).sum())
    if overlap:
        raise RuntimeError(f"rooms3 masks overlap on {overlap} timesteps")
    return valid, labels


def priority5_exclusive_labels_from_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mutually exclusive priority5 regions (same cascade as MPC priority5 selector)."""
    thresholds = tworoom_geometry_thresholds()
    x = xy[:, 0]
    y = xy[:, 1]
    wall_lo = thresholds["wall_lo"]
    wall_hi = thresholds["wall_hi"]
    doorway = (x >= wall_lo) & (x <= wall_hi)
    near_wall = (
        (x <= thresholds["x_lo_wall"])
        | (x >= thresholds["x_hi_wall"])
        | (y <= thresholds["y_lo_wall"])
        | (y >= thresholds["y_hi_wall"])
    )
    left_interior = (x >= thresholds["x_lo_common"]) & (x <= thresholds["x_hi_common_left"])
    right_interior = (x >= thresholds["x_lo_common_right"]) & (x <= thresholds["x_hi_common"])
    y_interior = (y >= thresholds["y_lo_common"]) & (y <= thresholds["y_hi_common"])
    common = (left_interior | right_interior) & y_interior

    labels = np.full(len(xy), -1, dtype=np.int64)
    labels[doorway] = 0  # doorway_corridor
    rem = ~doorway
    labels[rem & near_wall] = 1  # near_wall \ doorway
    rem2 = rem & ~near_wall
    labels[rem2 & common] = 2  # common \ (doorway ∪ near_wall)
    rem3 = rem2 & ~common
    labels[rem3 & (x > wall_hi)] = 3  # right_room \ (near_wall ∪ common)
    labels[rem3 & ~(x > wall_hi)] = 4  # left_room \ (near_wall ∪ common)
    valid = labels >= 0
    if int((~valid).sum()):
        raise RuntimeError(f"unlabeled priority5 timesteps: {(~valid).sum()}")
    return valid, labels


def labels_from_xy(xy: np.ndarray, partition: str) -> tuple[np.ndarray, np.ndarray]:
    if partition == "rooms3":
        return rooms3_labels_from_xy(xy)
    if partition == "priority5":
        return priority5_exclusive_labels_from_xy(xy)
    raise ValueError(f"Unknown partition: {partition!r}")


def proprio_at_indices(h5_path: Path, spec, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    unique_idx, inverse = np.unique(indices, return_inverse=True)
    with h5py.File(h5_path, "r") as h5:
        state_key = choose_state_key(h5, spec, None)
        proprio = np.asarray(h5[state_key][unique_idx], dtype=np.float64)
    if not finite_rows(proprio).all():
        raise ValueError("Invalid proprio rows for timestep indices")
    return proprio[inverse, :2]


def episode_ids_at_indices(h5_path: Path, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    unique_idx, inverse = np.unique(indices, return_inverse=True)
    with h5py.File(h5_path, "r") as h5:
        if "ep_idx" not in h5:
            raise KeyError("Dataset missing ep_idx; cannot split by episode")
        ep = np.asarray(h5["ep_idx"][unique_idx], dtype=np.int64)
    return ep[inverse]


def expand_latent_vectors(
    emb: np.ndarray,
    starts: np.ndarray,
    *,
    frameskip: int,
) -> tuple[np.ndarray, np.ndarray]:
    if emb.ndim != 3:
        raise ValueError(f"Expected (N,T,D) embeddings, got shape {emb.shape}")
    n, t_steps, dim = emb.shape
    offsets = np.arange(t_steps, dtype=np.int64) * frameskip
    global_idx = starts[:, None] + offsets[None, :]
    return emb.reshape(n * t_steps, dim), global_idx.reshape(n * t_steps)


def filter_labeled_latent_vectors(
    X: np.ndarray,
    global_idx: np.ndarray,
    h5_path: Path,
    spec,
    *,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = proprio_at_indices(h5_path, spec, global_idx)
    valid, y = labels_from_xy(xy, partition)
    return X[valid], global_idx[valid], y[valid]


def deduplicate_by_global_idx(
    X: np.ndarray,
    global_idx: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep one latent vector per global timestep (sliding windows overlap)."""
    order = np.argsort(global_idx, kind="stable")
    sorted_idx = global_idx[order]
    unique_take = np.concatenate([[True], sorted_idx[1:] != sorted_idx[:-1]])
    keep_pos = order[unique_take]
    dropped = int(len(global_idx) - len(keep_pos))
    return X[keep_pos], global_idx[keep_pos], y[keep_pos], dropped


def load_cached_latent_vectors(
    embed_dir: Path,
    *,
    h5_path: Path,
    spec,
    frameskip: int,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cfg = PARTITIONS[partition]
    region_labels = cfg["labels"]
    cache_regions = cfg["cache_regions"]
    X_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    transition_counts: dict[str, int] = {}

    for region in cache_regions:
        path = embed_dir / f"P_train_{region}_embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached embeddings: {path}")
        data = np.load(path)
        emb = np.asarray(data["emb"], dtype=np.float32)
        starts = np.asarray(data["region_starts"], dtype=np.int64)
        transition_counts[region] = int(len(starts))
        X_seq, global_idx = expand_latent_vectors(emb, starts, frameskip=frameskip)
        X_parts.append(X_seq)
        idx_parts.append(global_idx)
        print(
            f"  [cache] {region}: {transition_counts[region]} transitions, "
            f"{len(X_seq)} latent vectors from {path.name}",
            flush=True,
        )

    X_all = np.concatenate(X_parts, axis=0)
    global_idx = np.concatenate(idx_parts, axis=0)
    X, kept_idx, y = filter_labeled_latent_vectors(
        X_all, global_idx, h5_path, spec, partition=partition
    )
    before_dedup = len(y)
    print(
        f"  [data] {len(X_all)} expanded vectors -> {before_dedup} {partition} latent vectors",
        flush=True,
    )
    X, kept_idx, y, dropped = deduplicate_by_global_idx(X, kept_idx, y)
    print(
        f"  [dedup] unique global timesteps={len(y)} dropped_duplicates={dropped}",
        flush=True,
    )
    stats = {
        "partition": partition,
        "transition_counts": transition_counts,
        "num_expanded_vectors": int(len(X_all)),
        "num_vectors_before_dedup": before_dedup,
        "num_dropped_duplicate_timesteps": dropped,
        "num_unique_timesteps": int(len(y)),
    }
    return X, kept_idx, y, stats


def split_episodes(
    episode_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[set[int], set[int], set[int], dict]:
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Episode split fractions must sum to 1.0, got {total}")

    unique_eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_eps)
    n = len(perm)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    n_test = max(0, n - n_train - n_val)

    train_eps = set(perm[:n_train].tolist())
    val_eps = set(perm[n_train : n_train + n_val].tolist())
    test_eps = set(perm[n_train + n_val : n_train + n_val + n_test].tolist())

    if train_eps & val_eps or train_eps & test_eps or val_eps & test_eps:
        raise RuntimeError("Episode splits are not disjoint")

    info = {
        "episode_split_seed": seed,
        "episode_train_fraction": train_fraction,
        "episode_val_fraction": val_fraction,
        "episode_test_fraction": test_fraction,
        "num_unique_episodes": int(n),
        "num_train_episodes": len(train_eps),
        "num_val_episodes": len(val_eps),
        "num_test_episodes": len(test_eps),
    }
    return train_eps, val_eps, test_eps, info


def masks_for_episode_sets(episode_ids: np.ndarray, train_eps: set[int], val_eps: set[int], test_eps: set[int]):
    train_mask = np.isin(episode_ids, list(train_eps))
    val_mask = np.isin(episode_ids, list(val_eps))
    test_mask = np.isin(episode_ids, list(test_eps))
    if int((train_mask | val_mask | test_mask).sum()) != len(episode_ids):
        raise RuntimeError("Some latent vectors were not assigned to an episode split")
    return train_mask, val_mask, test_mask


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    class_ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": per_class,
    }


class TorchLinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TorchRBFFeatureMap(nn.Module):
    def __init__(self, in_dim: int, n_components: int, gamma: float, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(in_dim, n_components, generator=gen) * np.sqrt(2.0 * gamma)
        bias = 2.0 * np.pi * torch.rand(n_components, generator=gen)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.scale = float(np.sqrt(2.0 / n_components))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(x @ self.weight + self.bias)


@torch.no_grad()
def predict_torch(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for offset in range(0, len(X), batch_size):
        batch = torch.as_tensor(X[offset : offset + batch_size], device=device)
        logits = model(batch)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def fit_torch_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    scaler_mean = X_train.mean(axis=0, keepdims=True)
    scaler_std = X_train.std(axis=0, keepdims=True) + 1e-8
    Xn = (X_train - scaler_mean) / scaler_std

    model = TorchLinearClassifier(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(Xn, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    t0 = time.perf_counter()
    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    train_sec = time.perf_counter() - t0

    def transform(X: np.ndarray) -> np.ndarray:
        return ((X - scaler_mean) / scaler_std).astype(np.float32)

    return model, {"transform": transform, "train_sec": train_sec, "method": "linear_softmax_probe", "architecture": "Linear + CrossEntropyLoss + AdamW"}


def fit_torch_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_classes: int,
    device: torch.device,
    n_components: int,
    gamma: float | None,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    scaler_mean = X_train.mean(axis=0, keepdims=True)
    scaler_std = X_train.std(axis=0, keepdims=True) + 1e-8
    if gamma is None:
        gamma = 1.0 / X_train.shape[1]

    def transform(X: np.ndarray) -> np.ndarray:
        return ((X - scaler_mean) / scaler_std).astype(np.float32)

    feature_map = TorchRBFFeatureMap(X_train.shape[1], n_components, gamma, seed).to(device)
    classifier = TorchLinearClassifier(n_components, n_classes).to(device)
    model = nn.Sequential(feature_map, classifier).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    Xn = transform(X_train)
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(Xn, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    t0 = time.perf_counter()
    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    train_sec = time.perf_counter() - t0

    return model, {
        "transform": transform,
        "train_sec": train_sec,
        "gamma": gamma,
        "n_components": n_components,
        "implementation": "rff_rbf_probe",
        "architecture": "RandomFourierFeatures + Linear + CrossEntropyLoss + AdamW",
        "kernel": "rbf",
    }


def eval_torch(model, transform, X_train: np.ndarray, y_train: np.ndarray, splits: dict[str, tuple[np.ndarray, np.ndarray]], meta: dict, *, labels: tuple[str, ...], device: torch.device, batch_size: int) -> dict:
    out = {
        "train_fit_sec": float(meta["train_sec"]),
        "train_accuracy": float(
            accuracy_score(y_train, predict_torch(model, transform(X_train), device, batch_size))
        ),
    }
    if "method" in meta or "implementation" in meta:
        out["method"] = meta.get("method") or meta.get("implementation")
        out["architecture"] = meta.get("architecture")
        out["kernel"] = meta.get("kernel")
        out["n_components"] = meta.get("n_components")
        out["gamma"] = meta.get("gamma")
    for name, (X, y) in splits.items():
        y_pred = predict_torch(model, transform(X), device, batch_size)
        out[name] = split_metrics(y, y_pred, labels)
    return out


def print_split_metrics(split_name: str, m: dict, labels: tuple[str, ...]) -> None:
    print(
        f"  [{split_name}] acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
        f"macro_f1={m['macro_f1']:.4f}",
        flush=True,
    )
    for region in labels:
        c = m["per_class"][region]
        print(f"    {region}: P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1']:.4f}", flush=True)


def class_counts_dict(y: np.ndarray, labels: tuple[str, ...]) -> dict[str, int]:
    return {labels[i]: int((y == i).sum()) for i in range(len(labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tworoom", choices=DATASETS.keys())
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--partition", choices=tuple(PARTITIONS), default="rooms3")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--frameskip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--episode-split-seed", type=int, default=20260711)
    parser.add_argument("--episode-train-fraction", type=float, default=0.7)
    parser.add_argument("--episode-val-fraction", type=float, default=0.15)
    parser.add_argument("--episode-test-fraction", type=float, default=0.15)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-epochs", type=int, default=5)
    parser.add_argument("--torch-batch-size", type=int, default=16384)
    parser.add_argument("--torch-lr", type=float, default=1e-2)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--predict-batch-size", type=int, default=16384)
    parser.add_argument("--rbf-gamma", type=float, default=None, help="default 1 / n_features")
    parser.add_argument("--rbf-n-components", type=int, default=8192)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = THIS_DIR / "results" / f"geometry_latent_probe_{args.partition}"

    label_names = PARTITIONS[args.partition]["labels"]
    n_classes = len(label_names)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    spec = DATASETS[args.dataset]
    h5_path = args.data_root / spec.default_file
    thresholds = tworoom_geometry_thresholds()

    print(f"[data] partition={args.partition} loading P_train embeddings from {args.embedding_dir}", flush=True)
    X_latent, global_idx, y, data_stats = load_cached_latent_vectors(
        args.embedding_dir,
        h5_path=h5_path,
        spec=spec,
        frameskip=args.frameskip,
        partition=args.partition,
    )
    episode_ids = episode_ids_at_indices(h5_path, global_idx)

    train_eps, val_eps, test_eps, episode_info = split_episodes(
        episode_ids,
        seed=args.episode_split_seed,
        train_fraction=args.episode_train_fraction,
        val_fraction=args.episode_val_fraction,
        test_fraction=args.episode_test_fraction,
    )
    train_mask, val_mask, test_mask = masks_for_episode_sets(episode_ids, train_eps, val_eps, test_eps)

    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    print(
        f"[split] episodes train/val/test = "
        f"{episode_info['num_train_episodes']}/{episode_info['num_val_episodes']}/{episode_info['num_test_episodes']}",
        flush=True,
    )
    print(f"  vectors train/val/test = {len(y_train)}/{len(y_val)}/{len(y_test)}", flush=True)
    print(f"  train counts: {class_counts_dict(y_train, label_names)}", flush=True)
    print(f"  val   counts: {class_counts_dict(y_val, label_names)}", flush=True)
    print(f"  test  counts: {class_counts_dict(y_test, label_names)}", flush=True)

    eval_splits = {
        "val": (X_latent[val_mask], y_val),
        "test": (X_latent[test_mask], y_test),
    }

    results = {
        "dataset": args.dataset,
        "embedding_dir": str(args.embedding_dir),
        "sample_unit": "single_latent_vector",
        "split_unit": "episode",
        "partition": args.partition,
        "label_names": list(label_names),
        "rooms3_thresholds": thresholds,
        "data_source": f"P_train_geometry_{args.partition}_cache_expanded_dedup",
        "data_stats": data_stats,
        "frameskip": args.frameskip,
        "episode_split": episode_info,
        "vector_class_counts": {
            "train": class_counts_dict(y_train, label_names),
            "val": class_counts_dict(y_val, label_names),
            "test": class_counts_dict(y_test, label_names),
        },
        "model_seed": args.model_seed,
        "device": str(device),
        "models": {},
    }

    print("[model] linear_softmax_probe (GPU)", flush=True)
    model, meta = fit_torch_linear(
        X_latent[train_mask], y_train,
        n_classes=n_classes, device=device, epochs=args.torch_epochs, batch_size=args.torch_batch_size,
        lr=args.torch_lr, weight_decay=args.torch_weight_decay, seed=args.model_seed,
    )
    results["models"]["linear_softmax_probe"] = eval_torch(
        model, meta["transform"], X_latent[train_mask], y_train, eval_splits, meta,
        labels=label_names, device=device, batch_size=args.predict_batch_size,
    )
    for split_name in ("val", "test"):
        print_split_metrics(split_name, results["models"]["linear_softmax_probe"][split_name], label_names)

    print("[model] rff_rbf_probe (GPU)", flush=True)
    model, meta = fit_torch_rbf(
        X_latent[train_mask], y_train,
        n_classes=n_classes, device=device, n_components=args.rbf_n_components, gamma=args.rbf_gamma,
        epochs=args.torch_epochs, batch_size=args.torch_batch_size,
        lr=args.torch_lr, weight_decay=args.torch_weight_decay, seed=args.model_seed,
    )
    results["models"]["rff_rbf_probe"] = eval_torch(
        model, meta["transform"], X_latent[train_mask], y_train, eval_splits, meta,
        labels=label_names, device=device, batch_size=args.predict_batch_size,
    )
    for split_name in ("val", "test"):
        print_split_metrics(split_name, results["models"]["rff_rbf_probe"][split_name], label_names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"geometry_latent_probe_{args.partition}"
    np.savez_compressed(
        args.out_dir / f"{stem}_split.npz",
        global_idx=global_idx,
        y=y,
        episode_ids=episode_ids,
        train_episodes=np.array(sorted(train_eps), dtype=np.int64),
        val_episodes=np.array(sorted(val_eps), dtype=np.int64),
        test_episodes=np.array(sorted(test_eps), dtype=np.int64),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    out_json = args.out_dir / f"{stem}.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    out_csv = args.out_dir / f"{stem}_metrics.csv"
    with out_csv.open("w") as f:
        f.write(
            "model,split,accuracy,balanced_accuracy,macro_f1,"
            + ",".join(f"{r}_f1" for r in label_names)
            + "\n"
        )
        for name, model_res in results["models"].items():
            for split_name in ("val", "test"):
                s = model_res[split_name]
                row = [
                    name,
                    split_name,
                    f"{s['accuracy']:.6f}",
                    f"{s['balanced_accuracy']:.6f}",
                    f"{s['macro_f1']:.6f}",
                ]
                row.extend(f"{s['per_class'][r]['f1']:.6f}" for r in label_names)
                f.write(",".join(row) + "\n")

    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {args.out_dir / f'{stem}_split.npz'}", flush=True)


if __name__ == "__main__":
    main()
