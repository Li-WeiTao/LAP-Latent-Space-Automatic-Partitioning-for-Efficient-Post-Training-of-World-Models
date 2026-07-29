#!/usr/bin/env python3
"""Multi-seed geometry latent probe evaluation.

Each seed jointly controls:
  - episode train/val/test split
  - Linear Probe initialization and DataLoader minibatch order
  - RFF random Fourier features (omega_i, b_i)

Within a seed, Linear and RFF-RBF share the same episode split.
Reports test metrics as mean ± std across seeds and paired
  Delta_s = MacroF1_RFF,s - MacroF1_Linear,s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from gauge_drift import DATASETS, tworoom_geometry_thresholds  # noqa: E402

from geometry_latent_svm_rooms3 import (  # noqa: E402
    DEFAULT_EMBED_DIR,
    DEFAULT_FRAME_SKIP,
    PARTITIONS,
    class_counts_dict,
    episode_ids_at_indices,
    eval_torch,
    fit_torch_linear,
    fit_torch_rbf,
    load_cached_latent_vectors,
    masks_for_episode_sets,
    print_split_metrics,
    split_episodes,
)


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "values": [float(v) for v in values]}


def aggregate_test_metrics(
    per_seed: list[dict],
    label_names: tuple[str, ...],
) -> dict:
    out: dict = {}
    for model_key in ("linear_softmax_probe", "rff_rbf_probe"):
        out[model_key] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            out[model_key][metric] = mean_std(
                [run["models"][model_key]["test"][metric] for run in per_seed]
            )
        out[model_key]["per_class_f1"] = {}
        for label in label_names:
            out[model_key]["per_class_f1"][label] = mean_std(
                [run["models"][model_key]["test"]["per_class"][label]["f1"] for run in per_seed]
            )
    return out


def paired_macro_f1_delta(per_seed: list[dict]) -> dict:
    deltas = [
        run["models"]["rff_rbf_probe"]["test"]["macro_f1"]
        - run["models"]["linear_softmax_probe"]["test"]["macro_f1"]
        for run in per_seed
    ]
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": "macro_f1_rff_minus_linear",
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": [float(v) for v in deltas],
        "all_positive": bool((arr > 0).all()),
        "num_positive": int((arr > 0).sum()),
        "num_seeds": len(deltas),
    }


def run_one_seed(
    *,
    seed: int,
    X_latent: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
    label_names: tuple[str, ...],
    n_classes: int,
    device,
    args,
) -> dict:
    train_eps, val_eps, test_eps, episode_info = split_episodes(
        episode_ids,
        seed=seed,
        train_fraction=args.episode_train_fraction,
        val_fraction=args.episode_val_fraction,
        test_fraction=args.episode_test_fraction,
    )
    train_mask, val_mask, test_mask = masks_for_episode_sets(episode_ids, train_eps, val_eps, test_eps)
    y_train = y[train_mask]
    eval_splits = {
        "val": (X_latent[val_mask], y[val_mask]),
        "test": (X_latent[test_mask], y[test_mask]),
    }

    print(f"[seed={seed}] vectors train/val/test = {int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}", flush=True)

    result = {
        "seed": seed,
        "episode_split": episode_info,
        "vector_class_counts": {
            "train": class_counts_dict(y_train, label_names),
            "val": class_counts_dict(y[val_mask], label_names),
            "test": class_counts_dict(y[test_mask], label_names),
        },
        "models": {},
    }

    print(f"[seed={seed}] linear_softmax_probe", flush=True)
    model, meta = fit_torch_linear(
        X_latent[train_mask], y_train,
        n_classes=n_classes, device=device, epochs=args.torch_epochs, batch_size=args.torch_batch_size,
        lr=args.torch_lr, weight_decay=args.torch_weight_decay, seed=seed,
    )
    result["models"]["linear_softmax_probe"] = eval_torch(
        model, meta["transform"], X_latent[train_mask], y_train, eval_splits, meta,
        labels=label_names, device=device, batch_size=args.predict_batch_size,
    )
    print_split_metrics("test", result["models"]["linear_softmax_probe"]["test"], label_names)

    print(f"[seed={seed}] rff_rbf_probe", flush=True)
    model, meta = fit_torch_rbf(
        X_latent[train_mask], y_train,
        n_classes=n_classes, device=device, n_components=args.rbf_n_components, gamma=args.rbf_gamma,
        epochs=args.torch_epochs, batch_size=args.torch_batch_size,
        lr=args.torch_lr, weight_decay=args.torch_weight_decay, seed=seed,
    )
    result["models"]["rff_rbf_probe"] = eval_torch(
        model, meta["transform"], X_latent[train_mask], y_train, eval_splits, meta,
        labels=label_names, device=device, batch_size=args.predict_batch_size,
    )
    print_split_metrics("test", result["models"]["rff_rbf_probe"]["test"], label_names)

    delta = (
        result["models"]["rff_rbf_probe"]["test"]["macro_f1"]
        - result["models"]["linear_softmax_probe"]["test"]["macro_f1"]
    )
    print(f"[seed={seed}] paired Delta macro-F1 (RFF - Linear) = {delta:+.4f}", flush=True)
    result["paired_delta_macro_f1"] = float(delta)
    return result


def format_pct(stat: dict) -> str:
    return f"{stat['mean'] * 100:.2f}% ± {stat['std'] * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tworoom", choices=DATASETS.keys())
    parser.add_argument("--data-root", type=Path, default=Path("/data/sicong/weitao/datasets/lewm"))
    parser.add_argument("--partition", choices=tuple(PARTITIONS), default="priority5")
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--frameskip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episode-train-fraction", type=float, default=0.7)
    parser.add_argument("--episode-val-fraction", type=float, default=0.15)
    parser.add_argument("--episode-test-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-epochs", type=int, default=5)
    parser.add_argument("--torch-batch-size", type=int, default=16384)
    parser.add_argument("--torch-lr", type=float, default=1e-2)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--predict-batch-size", type=int, default=16384)
    parser.add_argument("--rbf-gamma", type=float, default=None)
    parser.add_argument("--rbf-n-components", type=int, default=8192)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = THIS_DIR / "results" / f"geometry_latent_probe_{args.partition}_multiseed"

    import torch

    label_names = PARTITIONS[args.partition]["labels"]
    n_classes = len(label_names)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    spec = DATASETS[args.dataset]
    h5_path = args.data_root / spec.default_file
    thresholds = tworoom_geometry_thresholds()

    print(f"[data] partition={args.partition} seeds={args.seeds}", flush=True)
    X_latent, global_idx, y, data_stats = load_cached_latent_vectors(
        args.embedding_dir,
        h5_path=h5_path,
        spec=spec,
        frameskip=args.frameskip,
        partition=args.partition,
    )
    episode_ids = episode_ids_at_indices(h5_path, global_idx)

    per_seed = [
        run_one_seed(
            seed=seed,
            X_latent=X_latent,
            y=y,
            episode_ids=episode_ids,
            label_names=label_names,
            n_classes=n_classes,
            device=device,
            args=args,
        )
        for seed in args.seeds
    ]

    summary = {
        "dataset": args.dataset,
        "partition": args.partition,
        "label_names": list(label_names),
        "seeds": args.seeds,
        "seed_controls": [
            "episode_split",
            "linear_probe_init_and_minibatch_order",
            "rff_random_fourier_features",
        ],
        "rooms3_thresholds": thresholds,
        "data_stats": data_stats,
        "frameskip": args.frameskip,
        "device": str(device),
        "per_seed": per_seed,
        "test_aggregate": aggregate_test_metrics(per_seed, label_names),
        "paired_delta": paired_macro_f1_delta(per_seed),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"geometry_latent_probe_{args.partition}_multiseed"
    out_json = args.out_dir / f"{stem}.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    out_csv = args.out_dir / f"{stem}_summary.csv"
    agg = summary["test_aggregate"]
    with out_csv.open("w") as f:
        f.write("model,metric,mean,std\n")
        for model_key in ("linear_softmax_probe", "rff_rbf_probe"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                stat = agg[model_key][metric]
                f.write(f"{model_key},{metric},{stat['mean']:.6f},{stat['std']:.6f}\n")
            for label in label_names:
                stat = agg[model_key]["per_class_f1"][label]
                f.write(f"{model_key},{label}_f1,{stat['mean']:.6f},{stat['std']:.6f}\n")
        pd = summary["paired_delta"]
        f.write(f"paired_delta,macro_f1_rff_minus_linear,{pd['mean']:.6f},{pd['std']:.6f}\n")

    print("\n=== test aggregate (mean ± std) ===", flush=True)
    for model_key, title in (
        ("linear_softmax_probe", "Linear Softmax Probe"),
        ("rff_rbf_probe", "RFF-RBF Probe"),
    ):
        print(f"{title}:", flush=True)
        print(f"  accuracy      {format_pct(agg[model_key]['accuracy'])}", flush=True)
        print(f"  balanced acc  {format_pct(agg[model_key]['balanced_accuracy'])}", flush=True)
        print(f"  macro-F1      {format_pct(agg[model_key]['macro_f1'])}", flush=True)
        for label in label_names:
            stat = agg[model_key]["per_class_f1"][label]
            print(f"  {label} F1     {format_pct(stat)}", flush=True)

    pd = summary["paired_delta"]
    print("\n=== paired Delta macro-F1 (RFF - Linear) ===", flush=True)
    print(f"  mean ± std = {pd['mean'] * 100:+.2f}pp ± {pd['std'] * 100:.2f}pp", flush=True)
    print(f"  per-seed   = {[f'{v * 100:+.2f}pp' for v in pd['values']]}", flush=True)
    print(f"  RFF > Linear on {pd['num_positive']}/{pd['num_seeds']} seeds", flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
