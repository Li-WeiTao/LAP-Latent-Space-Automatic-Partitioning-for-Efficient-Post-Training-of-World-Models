#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/tworoom/results"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value) -> float:
    try:
        if value == "" or value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def predictor_paths() -> list[tuple[int, Path]]:
    return [
        (0, RESULTS / "tworoom_predictor_rule_train_test_5k_j512_b20/predictor_rule_metrics.csv"),
        (1, RESULTS / "tworoom_predictor_rule_train_test_5k_j512_b20_seed_1/predictor_rule_metrics.csv"),
        (2, RESULTS / "tworoom_predictor_rule_train_test_5k_j512_b20_seed_2/predictor_rule_metrics.csv"),
        (3, RESULTS / "tworoom_predictor_rule_train_test_5k_j512_b20_seed_3/predictor_rule_metrics.csv"),
        (4, RESULTS / "tworoom_predictor_rule_train_test_5k_j512_b20_seed_4/predictor_rule_metrics.csv"),
    ]


def aggregate_predictor() -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for seed, path in predictor_paths():
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_csv(path):
            row = dict(row)
            row["seed"] = seed
            rows.append(row)

    metrics = [
        "n",
        "one_step_mse",
        "rollout_h5_mse",
        "rollout_h10_mse",
        "rule_drift_to_train",
        "test_iid_bootstrap_mean",
        "test_iid_bootstrap_std",
        "rule_excess_vs_test_iid",
        "rule_z_vs_test_iid",
    ]
    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)

    summary = []
    summary_by_split = {}
    for split in sorted(by_split):
        out = {"split": split, "seeds": len(by_split[split])}
        for metric in metrics:
            mean, std = mean_std([fnum(row.get(metric)) for row in by_split[split]])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary.append(out)
        summary_by_split[split] = out

    fieldnames = ["split", "seeds"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(
        RESULTS / "tworoom_predictor_rule_train_test_5seed_summary.csv",
        summary,
        fieldnames,
    )
    return summary, summary_by_split


def load_encoder_summary() -> dict[str, dict]:
    path = RESULTS / "tworoom_encoder_v2_seed_sweep/summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return {row["split"]: row for row in read_csv(path)}


def make_joint_summary(pred_by_split: dict[str, dict], enc_by_split: dict[str, dict]) -> list[dict]:
    rows = []
    for split in ["common", "doorway_corridor", "goal_other_side", "left_room", "near_wall", "right_room"]:
        pred = pred_by_split.get(split, {})
        enc = enc_by_split.get(split, {})
        row = {
            "split": split,
            "encoder_frame_drift_over_iid": fnum(enc.get("frame_drift_over_iid_mean")),
            "encoder_frame_residual_over_iid": fnum(enc.get("frame_residual_over_iid_mean")),
            "encoder_pca_drift_over_iid": fnum(enc.get("pca_drift_over_iid_mean")),
            "encoder_pca_residual_over_iid": fnum(enc.get("pca_residual_over_iid_mean")),
            "predictor_rule_drift_to_train": fnum(pred.get("rule_drift_to_train_mean")),
            "predictor_rule_excess_vs_test_iid": fnum(pred.get("rule_excess_vs_test_iid_mean")),
            "predictor_rule_z_vs_test_iid": fnum(pred.get("rule_z_vs_test_iid_mean")),
            "predictor_rollout_h10_mse": fnum(pred.get("rollout_h10_mse_mean")),
            "predictor_one_step_mse": fnum(pred.get("one_step_mse_mean")),
        }
        rows.append(row)
    fieldnames = [
        "split",
        "encoder_frame_drift_over_iid",
        "encoder_frame_residual_over_iid",
        "encoder_pca_drift_over_iid",
        "encoder_pca_residual_over_iid",
        "predictor_rule_drift_to_train",
        "predictor_rule_excess_vs_test_iid",
        "predictor_rule_z_vs_test_iid",
        "predictor_rollout_h10_mse",
        "predictor_one_step_mse",
    ]
    write_csv(RESULTS / "tworoom_stage1_encoder_predictor_joint_summary.csv", rows, fieldnames)
    return rows


def make_markdown(pred_summary: list[dict], joint_rows: list[dict]) -> str:
    pred_order = ["test_all", "doorway_corridor", "goal_other_side", "left_room", "near_wall", "right_room"]
    pred_map = {row["split"]: row for row in pred_summary}

    lines = []
    lines.append("## Stage 1 TwoRoom Summary: Encoder Drift and Predictor Rule Drift")
    lines.append("")
    lines.append("This section summarizes the current Stage 1 TwoRoom evidence after correcting the predictor-side reference. The predictor diagnostic now uses a train-global reference instead of the earlier common-region reference.")
    lines.append("")
    lines.append("### Predictor-Side Design")
    lines.append("")
    lines.append("- Train reference: global train split reconstructed from the LeWM config (`train_split=0.9`, `seed=3072`).")
    lines.append("- Test reports: all metrics are reported on held-out test transitions and natural test regions.")
    lines.append("- Seeds: five sampling seeds (`0,1,2,3,4`) with fixed `split_seed=3072`.")
    lines.append("- Per run: `train_max_samples=5000`, `test_max_samples=5000`, `jacobian_samples=512`, `iid_bootstrap_trials=20`.")
    lines.append("- Caveat: this approximates the LeWM train/test split from config; it is not the exact saved training index set from the checkpoint.")
    lines.append("")
    lines.append("### Predictor Rule Drift Across 5 Seeds")
    lines.append("")
    lines.append("| test split | rule drift to train | excess vs test IID | z vs test IID | rollout h10 MSE | one-step MSE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in pred_order:
        row = pred_map.get(split)
        if not row:
            continue
        lines.append(
            "| "
            + split
            + " | "
            + fmt(row["rule_drift_to_train_mean"])
            + " +- "
            + fmt(row["rule_drift_to_train_std"])
            + " | "
            + fmt(row["rule_excess_vs_test_iid_mean"])
            + " +- "
            + fmt(row["rule_excess_vs_test_iid_std"])
            + " | "
            + fmt(row["rule_z_vs_test_iid_mean"])
            + " +- "
            + fmt(row["rule_z_vs_test_iid_std"])
            + " | "
            + fmt(row["rollout_h10_mse_mean"])
            + " +- "
            + fmt(row["rollout_h10_mse_std"])
            + " | "
            + fmt(row["one_step_mse_mean"])
            + " +- "
            + fmt(row["one_step_mse_std"])
            + " |"
        )
    lines.append("")
    lines.append("Interpretation: `excess vs test IID` subtracts the test-IID bootstrap baseline from a region's rule drift. `z vs test IID` expresses that excess in units of the bootstrap standard deviation. Positive, large z means the region's predictor dynamics rule differs from train-global dynamics more than ordinary held-out test sampling noise.")
    lines.append("")
    lines.append("### Joint Encoder/Predictor Stage 1 View")
    lines.append("")
    lines.append("| split | encoder frame drift / IID | encoder frame residual / IID | predictor rule z | predictor excess | rollout h10 MSE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in joint_rows:
        lines.append(
            "| "
            + row["split"]
            + " | "
            + fmt(row["encoder_frame_drift_over_iid"])
            + " | "
            + fmt(row["encoder_frame_residual_over_iid"])
            + " | "
            + fmt(row["predictor_rule_z_vs_test_iid"])
            + " | "
            + fmt(row["predictor_rule_excess_vs_test_iid"])
            + " | "
            + fmt(row["predictor_rollout_h10_mse"])
            + " |"
        )
    lines.append("")
    lines.append("Stage 1 reading:")
    lines.append("")
    lines.append("- Encoder side: natural TwoRoom regions show much larger state-aligned frame drift than IID. Doorway/corridor is the cleanest encoder-side case because its frame residual stays close to IID while frame drift is high. Right room, left room, near wall, and goal-other-side also drift strongly, but their residual ratios are higher, so they mix coordinate drift with representation distortion or state-proxy mismatch.")
    lines.append("- Predictor side: with the corrected train-global reference, physical/dynamics rule inconsistency is strongest in `near_wall`, `right_room`, and `left_room`. These regions have large positive excess and z scores across five seeds, meaning their local predictor dynamics differ from train-global dynamics beyond ordinary test-IID variation.")
    lines.append("- Doorway/corridor remains important for encoder gauge drift, but it is not currently the strongest predictor-rule-drift region. Its predictor-side z is positive but weaker than wall/room-side regions. This means the cleanest encoder gauge drift does not automatically imply the largest predictor dynamics mismatch.")
    lines.append("- The current Stage 1 evidence therefore supports a more careful claim: natural regions can show both encoder-side state-aligned coordinate drift and predictor-side dynamics-rule drift, but the two effects are region-dependent and not one-to-one.")
    lines.append("- For the project's main hypothesis, the most useful next test is to check whether regions with high predictor rule drift also show higher rollout/planning failure after controlling for ordinary test-IID drift, and then test whether gauge-aware predictor interventions reduce that excess.")
    lines.append("")
    lines.append("Generated files:")
    lines.append("")
    lines.append("- `results/tworoom_predictor_rule_train_test_5seed_summary.csv`")
    lines.append("- `results/tworoom_stage1_encoder_predictor_joint_summary.csv`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    pred_summary, pred_by_split = aggregate_predictor()
    enc_by_split = load_encoder_summary()
    joint_rows = make_joint_summary(pred_by_split, enc_by_split)
    markdown = make_markdown(pred_summary, joint_rows)
    out_md = RESULTS / "tworoom_stage1_encoder_predictor_joint_summary.md"
    out_md.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
