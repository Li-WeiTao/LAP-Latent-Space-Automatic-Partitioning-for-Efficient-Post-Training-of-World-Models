from pathlib import Path
import csv
import math
import statistics


BASE = Path("experiments/tworoom/results/tworoom_encoder_v2_seed_sweep")

METRICS = [
    "n",
    "dim",
    "pca_residual",
    "pca_drift",
    "pca_residual_ratio",
    "pca_drift_to_residual_ratio",
    "frame_residual",
    "frame_drift",
    "frame_residual_ratio",
    "frame_drift_to_residual_ratio",
    "state_to_latent_r2",
    "latent_to_state_r2",
    "orthogonality_error",
    "orthogonal_recovery_error",
    "reference_overlap_n",
    "reference_overlap_ratio",
]

RATIO_METRICS = [
    "pca_drift_over_iid",
    "frame_drift_over_iid",
    "pca_residual_over_iid",
    "frame_residual_over_iid",
]


def as_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_ratio(a, b):
    if b and not math.isnan(a) and not math.isnan(b):
        return a / b
    return float("nan")


def mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    rows = []
    seed_dirs = [path for path in BASE.glob("seed_*") if path.is_dir()]
    for directory in sorted(seed_dirs, key=lambda p: int(p.name.split("_")[1])):
        seed = int(directory.name.split("_")[1])
        path = directory / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {"seed": seed, "split": row["split"]}
                for metric in METRICS:
                    parsed[metric] = as_float(row.get(metric, ""))
                rows.append(parsed)

    iid_by_seed = {row["seed"]: row for row in rows if row["split"] == "iid_nonoverlap"}
    for row in rows:
        iid = iid_by_seed.get(row["seed"])
        if iid is None:
            for metric in RATIO_METRICS:
                row[metric] = float("nan")
            continue
        row["pca_drift_over_iid"] = safe_ratio(row["pca_drift"], iid["pca_drift"])
        row["frame_drift_over_iid"] = safe_ratio(row["frame_drift"], iid["frame_drift"])
        row["pca_residual_over_iid"] = safe_ratio(row["pca_residual"], iid["pca_residual"])
        row["frame_residual_over_iid"] = safe_ratio(row["frame_residual"], iid["frame_residual"])

    summary = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        summary_row = {"split": split, "seeds": len({row["seed"] for row in split_rows})}
        for metric in METRICS + RATIO_METRICS:
            mean, std = mean_std([row[metric] for row in split_rows])
            summary_row[f"{metric}_mean"] = mean
            summary_row[f"{metric}_std"] = std
        summary.append(summary_row)

    fields = ["split", "seeds"] + [
        field for metric in METRICS + RATIO_METRICS for field in (f"{metric}_mean", f"{metric}_std")
    ]
    with (BASE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    key_fields = [
        "split",
        "seeds",
        "n_mean",
        "pca_residual_mean",
        "pca_residual_std",
        "pca_drift_mean",
        "pca_drift_std",
        "pca_residual_over_iid_mean",
        "pca_residual_over_iid_std",
        "pca_drift_over_iid_mean",
        "pca_drift_over_iid_std",
        "frame_residual_mean",
        "frame_residual_std",
        "frame_drift_mean",
        "frame_drift_std",
        "frame_residual_over_iid_mean",
        "frame_residual_over_iid_std",
        "frame_drift_over_iid_mean",
        "frame_drift_over_iid_std",
        "state_to_latent_r2_mean",
        "state_to_latent_r2_std",
        "latent_to_state_r2_mean",
        "latent_to_state_r2_std",
        "reference_overlap_ratio_mean",
        "reference_overlap_ratio_std",
    ]
    with (BASE / "summary_key.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in key_fields} for row in summary])

    print(f"wrote {BASE / 'summary.csv'}")
    print(f"wrote {BASE / 'summary_key.csv'}")
    print("\nKEY SUMMARY")
    for row in summary:
        if row["split"] == "reference":
            continue
        print(
            f"{row['split']:18s} seeds={row['seeds']:2d} "
            f"pca_drift={row['pca_drift_mean']:.4f}+/-{row['pca_drift_std']:.4f} "
            f"pca_drift/IID={row['pca_drift_over_iid_mean']:.1f}+/-{row['pca_drift_over_iid_std']:.1f} "
            f"pca_resid/IID={row['pca_residual_over_iid_mean']:.2f}+/-{row['pca_residual_over_iid_std']:.2f} "
            f"frame_drift={row['frame_drift_mean']:.4f}+/-{row['frame_drift_std']:.4f} "
            f"frame_drift/IID={row['frame_drift_over_iid_mean']:.1f}+/-{row['frame_drift_over_iid_std']:.1f} "
            f"frame_resid/IID={row['frame_residual_over_iid_mean']:.2f}+/-{row['frame_residual_over_iid_std']:.2f} "
            f"lat2stateR2={row['latent_to_state_r2_mean']:.4f}+/-{row['latent_to_state_r2_std']:.4f}"
        )


if __name__ == "__main__":
    main()
