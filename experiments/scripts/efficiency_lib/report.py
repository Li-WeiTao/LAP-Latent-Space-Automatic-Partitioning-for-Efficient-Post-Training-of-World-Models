"""Aggregate efficiency benchmark outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _pct_reduction(base: float, value: float) -> float:
    if base <= 0:
        return float("nan")
    return (1.0 - value / base) * 100.0


def _pct_overhead(base: float, value: float) -> float:
    if base <= 0:
        return float("nan")
    return (value - base) / base * 100.0


def build_reports(
    *,
    output_dir: Path,
    joint: dict[str, Any] | None,
    lap: dict[str, Any] | None,
    gate_partition: dict[str, Any] | None,
    inference: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_lines: list[dict[str, Any]] = []

    if joint:
        for row in joint.get("epochs", []):
            raw_lines.append({"benchmark": "training", **row})
    if lap:
        for row in lap.get("expert_epochs", []):
            raw_lines.append({"benchmark": "training_expert", **row})
        for row in lap.get("lap_epochs", []):
            raw_lines.append(
                {
                    "benchmark": "training_lap_epoch",
                    "lap_epoch": row.get("lap_epoch"),
                    "total_wall_sec": row.get("total_wall_sec"),
                    "peak_allocated_bytes": row.get("peak_allocated_bytes"),
                }
            )
    for item in inference:
        if item.get("status") != "ok":
            raw_lines.append({"benchmark": "inference", **item})
            continue
        for idx, sec in enumerate(item.get("planning_latency_sec", [])):
            raw_lines.append(
                {
                    "benchmark": "inference_planning",
                    "task": item["task"],
                    "mode": item["mode"],
                    "repeat": idx,
                    "planning_latency_sec": sec,
                }
            )
        for idx, sec in enumerate(item.get("routing_latency_sec", [])):
            raw_lines.append(
                {
                    "benchmark": "inference_routing",
                    "task": item["task"],
                    "mode": item["mode"],
                    "repeat": idx,
                    "routing_latency_sec": sec,
                }
            )

    raw_path = output_dir / "efficiency_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_lines:
            handle.write(json.dumps(row) + "\n")

    training_rows = []
    if joint and lap:
        joint_stable = joint["stable_epoch_summary"]["mean_sec"]
        lap_stable = lap["stable_epoch_summary"]["mean_sec"]
        joint_peak = joint["peak_memory"]["peak_allocated_gb"]
        lap_peak = lap["peak_memory"]["peak_allocated_gb"]
        training_rows.append(
            {
                "method": "joint",
                "time_per_epoch_sec": joint_stable,
                "peak_gpu_memory_gb": joint_peak,
                "training_speedup_x": 1.0,
                "time_reduction_pct": 0.0,
                "memory_reduction_pct": 0.0,
            }
        )
        training_rows.append(
            {
                "method": "lap_regional",
                "time_per_epoch_sec": lap_stable,
                "peak_gpu_memory_gb": lap_peak,
                "training_speedup_x": joint_stable / lap_stable if lap_stable else float("nan"),
                "time_reduction_pct": _pct_reduction(joint_stable, lap_stable),
                "memory_reduction_pct": _pct_reduction(joint_peak, lap_peak),
            }
        )

    training_csv = output_dir / "training_comparison.csv"
    if training_rows:
        with training_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(training_rows[0].keys()))
            writer.writeheader()
            writer.writerows(training_rows)

    inference_rows = []
    breakdown_rows = []
    by_task: dict[str, dict[str, Any]] = {}
    for item in inference:
        task = item["task"]
        by_task.setdefault(task, {})[item["mode"]] = item
    for task, modes in by_task.items():
        base = modes.get("baseline")
        lap_item = modes.get("lap")
        if not base or base.get("status") != "ok":
            inference_rows.append({"task": task, "status": "pending", "reason": "missing baseline"})
            continue
        if not lap_item or lap_item.get("status") != "ok":
            inference_rows.append({"task": task, "status": "pending", "reason": "missing lap"})
            continue
        base_mean = base["planning_summary"]["mean"]
        lap_mean = lap_item["planning_summary"]["mean"]
        route_mean = (
            lap_item["routing_summary"]["mean"]
            if lap_item.get("routing_summary")
            else float("nan")
        )
        inference_rows.append(
            {
                "task": task,
                "original_lewm_planning_sec": base_mean,
                "lap_planning_sec": lap_mean,
                "routing_sec": route_mean,
                "absolute_overhead_sec": lap_mean - base_mean,
                "relative_overhead_pct": _pct_overhead(base_mean, lap_mean),
                "original_peak_gpu_gb": base["peak_memory"]["peak_allocated_gb"],
                "lap_peak_gpu_gb": lap_item["peak_memory"]["peak_allocated_gb"],
                "status": "ok",
            }
        )
        breakdown_rows.append(
            {
                "task": task,
                "component": "planning_total",
                "lewm_sec": base_mean,
                "lap_sec": lap_mean,
            }
        )
        breakdown_rows.append(
            {
                "task": task,
                "component": "routing_only",
                "lewm_sec": float("nan"),
                "lap_sec": route_mean,
            }
        )

    with (output_dir / "inference_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        if inference_rows:
            writer = csv.DictWriter(handle, fieldnames=list(inference_rows[0].keys()))
            writer.writeheader()
            writer.writerows(inference_rows)
    with (output_dir / "inference_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        if breakdown_rows:
            writer = csv.DictWriter(handle, fieldnames=list(breakdown_rows[0].keys()))
            writer.writeheader()
            writer.writerows(breakdown_rows)

    summary_rows: list[dict[str, Any]] = []
    if training_rows:
        for row in training_rows:
            summary_rows.append({"section": "training", **row})
    for row in inference_rows:
        summary_rows.append({"section": "inference", **row})
    with (output_dir / "efficiency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            fieldnames: list[str] = []
            for row in summary_rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    tex = output_dir / "efficiency_table.tex"
    lines = [
        "% Auto-generated LAP efficiency tables",
        "% Panel (a): Training efficiency on fixed TwoRoom",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Time / epoch (s) & Peak GPU (GB) & Speedup & Memory reduction \\\\",
        "\\midrule",
    ]
    for row in training_rows:
        if row["method"] == "joint":
            lines.append(
                f"Joint training & {row['time_per_epoch_sec']:.1f} & {row['peak_gpu_memory_gb']:.2f} & -- & -- \\\\"
            )
        else:
            lines.append(
                f"LAP Regional-FT & {row['time_per_epoch_sec']:.1f} & {row['peak_gpu_memory_gb']:.2f} & "
                f"{row['training_speedup_x']:.2f}$\\times$ & {row['memory_reduction_pct']:.1f}\\% \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    if gate_partition:
        lines.extend(
            [
                "% One-time LAP costs",
                "\\begin{tabular}{lr}",
                "\\toprule",
                "One-time LAP cost & Time (s) \\\\",
                "\\midrule",
                f"Gate & {gate_partition.get('gate_wall_sec', 'N/A')} \\\\",
                f"Partition & {gate_partition.get('partition_wall_sec', 'N/A')} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        )
    lines.extend(
        [
            "% Panel (b): Planning latency",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Task & Original LeWM & LAP & Routing & Overhead \\\\",
            "\\midrule",
        ]
    )
    for row in inference_rows:
        if row.get("status") != "ok":
            lines.append(f"{row['task']} & pending & pending & pending & pending \\\\")
            continue
        lines.append(
            f"{row['task']} & {row['original_lewm_planning_sec']:.3f} & {row['lap_planning_sec']:.3f} & "
            f"{row['routing_sec']:.4f} & {row['relative_overhead_pct']:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
