"""Uncertainty-aware branch-selection diagnostics from paired bootstrap draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .loader import CellData, point_estimate

PILOT_PAIRS: frozenset[tuple[str, str]] = frozenset({("lewm", "tworoom"), ("lewm", "pusht")})

AGGREGATION_LABELS: dict[str, str] = {
    "none": "post-training-seed mean",
    "average": "partition-seed average",
    "deployment": "deployment partition seed",
}

MARGIN_PP_DEFAULT = 2.0
MARGIN_STATUS = "post-hoc practical-sensitivity margin"


def pilot_status(model: str, task: str) -> str:
    return "pilot" if (model, task) in PILOT_PAIRS else "non-pilot"


def resolve_selected_branch(gate_info: dict[str, Any]) -> str:
    branch = gate_info.get("branch")
    if branch not in {"global", "spectral"}:
        raise ValueError(f"gate manifest missing supported branch: {branch!r}")
    return str(branch)


def resolve_rejected_method(gate_info: dict[str, Any]) -> str:
    branch = resolve_selected_branch(gate_info)
    if branch == "spectral":
        return "global"
    return "spectral"


def aggregation_label(partition_policy: str) -> str:
    return AGGREGATION_LABELS.get(partition_policy, partition_policy)


def classify_selection_risk(
    ci_low_pp: float,
    ci_high_pp: float,
    *,
    margin_pp: float,
) -> str:
    if ci_low_pp > margin_pp:
        return "materially_better"
    if ci_low_pp > -margin_pp:
        return "practically_noninferior"
    if ci_high_pp < -margin_pp:
        return "materially_worse"
    return "statistically_unresolved"


@dataclass(frozen=True)
class SelectionRiskSummary:
    model: str
    task: str
    horizon: str
    pilot_status: str
    selected_branch: str
    rejected_branch: str
    selected_aggregation: str
    rejected_aggregation: str
    delta_mean_pp: float
    ci_low_pp: float
    ci_high_pp: float
    point_estimate_favors_selected: bool
    p_harm: float
    eol_pp: float
    practical_eol_pp: float
    margin_pp: float
    classification: str
    n_bootstrap: int
    rng_seed: int
    gate_source: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "horizon": self.horizon,
            "pilot_status": self.pilot_status,
            "selected_branch": self.selected_branch,
            "rejected_branch": self.rejected_branch,
            "selected_aggregation": self.selected_aggregation,
            "rejected_aggregation": self.rejected_aggregation,
            "delta_mean_pp": self.delta_mean_pp,
            "ci_low_pp": self.ci_low_pp,
            "ci_high_pp": self.ci_high_pp,
            "point_estimate_favors_selected": self.point_estimate_favors_selected,
            "p_harm": self.p_harm,
            "eol_pp": self.eol_pp,
            "practical_eol_pp": self.practical_eol_pp,
            "margin_pp": self.margin_pp,
            "classification": self.classification,
            "n_bootstrap": self.n_bootstrap,
            "rng_seed": self.rng_seed,
            "gate_source": self.gate_source,
        }


def summarize_selection_risk(
    delta_samples: np.ndarray,
    *,
    margin_pp: float = MARGIN_PP_DEFAULT,
    model: str,
    task: str,
    horizon: str,
    gate_info: dict[str, Any],
    cell: CellData,
    n_bootstrap: int,
    rng_seed: int,
) -> SelectionRiskSummary:
    delta = np.asarray(delta_samples, dtype=np.float64)
    if delta.ndim != 1:
        raise ValueError("delta_samples must be one-dimensional")
    if delta.shape[0] != n_bootstrap:
        raise ValueError(
            f"expected {n_bootstrap} bootstrap draws, got {delta.shape[0]}"
        )

    selected = resolve_selected_branch(gate_info)
    rejected = resolve_rejected_method(gate_info)
    ci_low_pp, ci_high_pp = (float(v) for v in np.quantile(delta, [0.025, 0.975]))
    delta_mean_pp = float(delta.mean())

    return SelectionRiskSummary(
        model=model,
        task=task,
        horizon=horizon,
        pilot_status=pilot_status(model, task),
        selected_branch=selected,
        rejected_branch=rejected,
        selected_aggregation=aggregation_label(cell.methods["autolap"].partition_policy),
        rejected_aggregation=aggregation_label(cell.methods[rejected].partition_policy),
        delta_mean_pp=delta_mean_pp,
        ci_low_pp=ci_low_pp,
        ci_high_pp=ci_high_pp,
        point_estimate_favors_selected=delta_mean_pp > 0.0,
        p_harm=float(np.mean(delta < -margin_pp)),
        eol_pp=float(np.mean(np.maximum(-delta, 0.0))),
        practical_eol_pp=float(np.mean(np.maximum(-margin_pp - delta, 0.0))),
        margin_pp=margin_pp,
        classification=classify_selection_risk(
            ci_low_pp, ci_high_pp, margin_pp=margin_pp
        ),
        n_bootstrap=n_bootstrap,
        rng_seed=rng_seed,
        gate_source=gate_info.get("source"),
    )


def selection_delta_draws(
    cell: CellData,
    raw_draws: dict[str, np.ndarray],
) -> np.ndarray:
    if "autolap" not in raw_draws:
        raise ValueError("missing autolap bootstrap draws")
    rejected = resolve_rejected_method(cell.gate_info)
    if rejected not in raw_draws:
        raise KeyError(f"missing rejected-method draws: {rejected}")
    return raw_draws["autolap"] - raw_draws[rejected]


def verify_point_estimate_agreement(summary: SelectionRiskSummary, cell: CellData) -> bool:
    auto = point_estimate(cell.methods["autolap"].blocks, official=False)
    rejected = resolve_rejected_method(cell.gate_info)
    base = point_estimate(cell.methods[rejected].blocks, official=False)
    observed_favors = auto > base
    return observed_favors == summary.point_estimate_favors_selected


def aggregate_selection_risk_summary(
    rows: list[SelectionRiskSummary],
    *,
    margin_pp: float = MARGIN_PP_DEFAULT,
) -> dict[str, Any]:
    long_rows = [r for r in rows if r.horizon == "long"]
    pilot_rows = [r for r in long_rows if r.pilot_status == "pilot"]
    non_pilot_rows = [r for r in long_rows if r.pilot_status == "non-pilot"]

    classifications: dict[str, int] = {}
    for row in long_rows:
        classifications[row.classification] = classifications.get(row.classification, 0) + 1

    max_p_harm_row = max(long_rows, key=lambda r: r.p_harm)
    max_eol_row = max(long_rows, key=lambda r: r.eol_pp)

    ci_crosses_zero = [
        {
            "model": r.model,
            "task": r.task,
            "ci_low_pp": r.ci_low_pp,
            "ci_high_pp": r.ci_high_pp,
        }
        for r in long_rows
        if r.ci_low_pp <= 0.0 <= r.ci_high_pp
    ]

    return {
        "margin_pp": margin_pp,
        "margin_status": MARGIN_STATUS,
        "margin_interpretation": (
            "Each 50-start evaluation block corresponds to 2 percentage points per "
            "episode success; the 2 pp threshold is a post-hoc sensitivity analysis "
            "and is not predeclared, preregistered, or confirmatory."
        ),
        "long_horizon_cell_count": len(long_rows),
        "pilot_cell_count": len(pilot_rows),
        "non_pilot_cell_count": len(non_pilot_rows),
        "overall_point_estimate_agreement_count": sum(
            1 for r in long_rows if r.point_estimate_favors_selected
        ),
        "overall_point_estimate_total": len(long_rows),
        "pilot_point_estimate_agreement_count": sum(
            1 for r in pilot_rows if r.point_estimate_favors_selected
        ),
        "pilot_point_estimate_total": len(pilot_rows),
        "non_pilot_point_estimate_agreement_count": sum(
            1 for r in non_pilot_rows if r.point_estimate_favors_selected
        ),
        "non_pilot_point_estimate_total": len(non_pilot_rows),
        "non_pilot_mean_eol_pp": float(np.mean([r.eol_pp for r in non_pilot_rows])),
        "non_pilot_mean_practical_eol_pp": float(
            np.mean([r.practical_eol_pp for r in non_pilot_rows])
        ),
        "classification_counts": classifications,
        "max_p_harm": {
            "value": max_p_harm_row.p_harm,
            "model": max_p_harm_row.model,
            "task": max_p_harm_row.task,
        },
        "max_eol_pp": {
            "value": max_eol_row.eol_pp,
            "model": max_eol_row.model,
            "task": max_eol_row.task,
        },
        "ci_crosses_zero": ci_crosses_zero,
        "subjepa_tworoom_long": next(
            (
                r.to_row()
                for r in long_rows
                if r.model == "subjepa" and r.task == "tworoom"
            ),
            None,
        ),
    }


def render_selection_risk_report(
    rows: list[SelectionRiskSummary],
    summary: dict[str, Any],
) -> str:
    long_rows = sorted(
        [r for r in rows if r.horizon == "long"],
        key=lambda r: (r.model, r.task),
    )
    lines: list[str] = []
    lines.append("# LAP branch-selection uncertainty and decision-risk diagnostics")
    lines.append("")
    lines.append(
        "This appendix-level sensitivity analysis supplements the descriptive "
        "7-of-8 point-estimate agreement reported in the main text. It does **not** "
        "redefine the empirically better-performing branch, alter gate decisions, or "
        "constitute a predeclared or confirmatory analysis."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The **7/8** headline counts pairs where Auto-LAP's selected dynamics-modeling "
        "branch has the higher **observed mean** success rate at long horizon."
    )
    lines.append(
        "- **Paired bootstrap CIs** quantify evaluation uncertainty; a CI that crosses "
        "zero does not overturn the observed-mean ordering."
    )
    lines.append(
        "- **p_harm** and **EOL** are supplementary decision-risk diagnostics for the "
        "selected-vs-rejected dynamics-modeling contrast."
    )
    lines.append(
        f"- **margin = {summary['margin_pp']:.1f} pp** is a "
        f"**{summary['margin_status']}** (one episode success ≈ 2 pp within a "
        "50-start evaluation block)."
    )
    lines.append("")
    lines.append("## Long-horizon diagnostics")
    lines.append("")
    lines.append(
        "| Model | Task | Pilot | Selected | Rejected | Δ mean (pp) | 95% CI | "
        "Point favors selected | p_harm | EOL (pp) | Practical EOL (pp) | Classification |"
    )
    lines.append(
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|"
    )
    for row in long_rows:
        ci = f"[{row.ci_low_pp:.2f}, {row.ci_high_pp:.2f}]"
        lines.append(
            f"| {row.model} | {row.task} | {row.pilot_status} | {row.selected_branch} "
            f"| {row.rejected_branch} | {row.delta_mean_pp:.2f} | {ci} | "
            f"{str(row.point_estimate_favors_selected).lower()} | {row.p_harm:.4f} | "
            f"{row.eol_pp:.4f} | {row.practical_eol_pp:.4f} | {row.classification} |"
        )
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")
    lines.append(
        f"- Overall point-estimate agreement: "
        f"{summary['overall_point_estimate_agreement_count']}/"
        f"{summary['overall_point_estimate_total']}"
    )
    lines.append(
        f"- Pilot pairs: {summary['pilot_point_estimate_agreement_count']}/"
        f"{summary['pilot_point_estimate_total']}"
    )
    lines.append(
        f"- Non-pilot pairs: {summary['non_pilot_point_estimate_agreement_count']}/"
        f"{summary['non_pilot_point_estimate_total']}"
    )
    lines.append(
        f"- Non-pilot mean EOL: {summary['non_pilot_mean_eol_pp']:.4f} pp"
    )
    lines.append(
        f"- Non-pilot mean practical EOL: "
        f"{summary['non_pilot_mean_practical_eol_pp']:.4f} pp"
    )
    lines.append(
        f"- Max p_harm: {summary['max_p_harm']['value']:.4f} "
        f"({summary['max_p_harm']['model']}/{summary['max_p_harm']['task']})"
    )
    lines.append(
        f"- Max EOL: {summary['max_eol_pp']['value']:.4f} pp "
        f"({summary['max_eol_pp']['model']}/{summary['max_eol_pp']['task']})"
    )
    lines.append("")
    lines.append("## LaTeX appendix snippet")
    lines.append("")
    lines.append("```latex")
    lines.append(
        "Because the empirically better-performing branch is defined by the higher "
        "observed mean, the 7-of-8 result is a point-estimate comparison. We supplement "
        "it with paired-bootstrap decision-risk diagnostics. Using a post-hoc practical "
        "sensitivity margin of 2 percentage points, we report the probability that the "
        "selected branch is materially worse and its expected opportunity loss. These "
        "diagnostics quantify uncertainty but do not alter the gate or the reported "
        "branch decisions."
    )
    lines.append("")
    lines.append(
        "\\caption{Paired-bootstrap uncertainty and decision-risk diagnostics for "
        "LAP's long-horizon dynamics-modeling choices.}"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
