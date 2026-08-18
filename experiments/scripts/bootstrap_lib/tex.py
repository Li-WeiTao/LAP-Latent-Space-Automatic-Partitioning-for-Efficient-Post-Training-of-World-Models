"""LaTeX table generation for bootstrap results."""

from __future__ import annotations

from typing import Any


def _fmt(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value:.2f}"


def render_tex_tables(rows_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for key in sorted(rows_by_cell):
        model, task, horizon = key
        cell_rows = rows_by_cell[key]
        pending = bool(cell_rows) and cell_rows[0].get("status") == "pending"
        lines.append(f"% {model}-{task}-{horizon}")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(
            f"\\caption{{Bootstrap success rates (\\%) for {model.upper()} on "
            f"{task.capitalize()} ({horizon} horizon).}}"
        )
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Method & Point & Mean & 95\\% CI & $n_b$ \\\\")
        lines.append("\\midrule")
        if pending or not cell_rows:
            lines.append("\\multicolumn{5}{c}{\\textit{Pending}} \\\\")
        else:
            for row in cell_rows:
                method = str(row["method"]).replace("_", "\\_")
                ci = f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
                lines.append(
                    f"{method} & {_fmt(row['point_estimate'])} & "
                    f"{_fmt(row['bootstrap_mean'])} & {ci} & {row['n_bootstrap']} \\\\"
                )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    return "\n".join(lines)
