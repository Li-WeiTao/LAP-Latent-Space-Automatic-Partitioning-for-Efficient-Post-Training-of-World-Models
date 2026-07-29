from pathlib import Path
import csv
import math
import statistics


BASE = Path("experiments/tworoom/results/tworoom_resample_check")

METRICS = [
    "n",
    "pca_residual",
    "pca_drift",
    "frame_residual",
    "frame_drift",
    "state_to_latent_r2",
    "latent_to_state_r2",
    "reference_overlap_ratio",
]


def as_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def mean_std(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def safe_ratio(a, b):
    if b and not math.isnan(a) and not math.isnan(b):
        return a / b
    return float("nan")


def read_rows():
    rows = []
    for sample_dir in sorted(BASE.glob("sample_seed_*"), key=lambda p: int(p.name.split("_")[-1])):
        sample_seed = int(sample_dir.name.split("_")[-1])
        analysis_dirs = [path for path in sample_dir.glob("analysis_seed_*") if path.is_dir()]
        for analysis_dir in sorted(analysis_dirs, key=lambda p: int(p.name.split("_")[-1])):
            analysis_seed = int(analysis_dir.name.split("_")[-1])
            path = analysis_dir / "metrics.csv"
            if not path.exists():
                continue
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    parsed = {
                        "sample_seed": sample_seed,
                        "analysis_seed": analysis_seed,
                        "split": row["split"],
                    }
                    for metric in METRICS:
                        parsed[metric] = as_float(row.get(metric, ""))
                    rows.append(parsed)
    return rows


def add_iid_ratios(rows):
    iid = {
        (row["sample_seed"], row["analysis_seed"]): row
        for row in rows
        if row["split"] == "iid_nonoverlap"
    }
    for row in rows:
        baseline = iid.get((row["sample_seed"], row["analysis_seed"]))
        row["pca_drift_over_iid"] = safe_ratio(row["pca_drift"], baseline["pca_drift"]) if baseline else float("nan")
        row["pca_residual_over_iid"] = safe_ratio(row["pca_residual"], baseline["pca_residual"]) if baseline else float("nan")
        row["frame_drift_over_iid"] = safe_ratio(row["frame_drift"], baseline["frame_drift"]) if baseline else float("nan")
        row["frame_residual_over_iid"] = safe_ratio(row["frame_residual"], baseline["frame_residual"]) if baseline else float("nan")


def summarize(rows, keys):
    metric_names = METRICS + [
        "pca_drift_over_iid",
        "pca_residual_over_iid",
        "frame_drift_over_iid",
        "frame_residual_over_iid",
    ]
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group_rows in sorted(groups.items()):
        summary = {name: value for name, value in zip(keys, key)}
        summary["runs"] = len(group_rows) // max(1, len({r["split"] for r in group_rows}))
        for metric in metric_names:
            mean, std = mean_std([row[metric] for row in group_rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        out.append(summary)
    return out, metric_names


def write_csv(path, rows, keys, metric_names):
    fields = keys + ["runs"] + [field for metric in metric_names for field in (f"{metric}_mean", f"{metric}_std")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_rows()
    if not rows:
        raise RuntimeError(f"No analysis metrics found under {BASE}")
    add_iid_ratios(rows)

    by_sample, metrics = summarize(rows, ["sample_seed", "split"])
    write_csv(BASE / "summary_by_sample_seed.csv", by_sample, ["sample_seed", "split"], metrics)

    combined, metrics = summarize(rows, ["split"])
    write_csv(BASE / "summary_combined_resamples.csv", combined, ["split"], metrics)

    print("wrote", BASE / "summary_by_sample_seed.csv")
    print("wrote", BASE / "summary_combined_resamples.csv")
    print("\nBY SAMPLE SEED")
    for row in by_sample:
        if row["split"] in {"reference"}:
            continue
        print(
            f"sample={int(row['sample_seed'])} {row['split']:18s} "
            f"pca_drift={row['pca_drift_mean']:.4f}+/-{row['pca_drift_std']:.4f} "
            f"pca_resid/IID={row['pca_residual_over_iid_mean']:.2f}+/-{row['pca_residual_over_iid_std']:.2f} "
            f"frame_drift={row['frame_drift_mean']:.4f}+/-{row['frame_drift_std']:.4f} "
            f"frame_resid/IID={row['frame_residual_over_iid_mean']:.2f}+/-{row['frame_residual_over_iid_std']:.2f}"
        )

    print("\nCOMBINED RESAMPLES")
    for row in combined:
        if row["split"] in {"reference"}:
            continue
        print(
            f"{row['split']:18s} "
            f"pca_drift={row['pca_drift_mean']:.4f}+/-{row['pca_drift_std']:.4f} "
            f"pca_resid/IID={row['pca_residual_over_iid_mean']:.2f}+/-{row['pca_residual_over_iid_std']:.2f} "
            f"frame_drift={row['frame_drift_mean']:.4f}+/-{row['frame_drift_std']:.4f} "
            f"frame_resid/IID={row['frame_residual_over_iid_mean']:.2f}+/-{row['frame_residual_over_iid_std']:.2f}"
        )


if __name__ == "__main__":
    main()
