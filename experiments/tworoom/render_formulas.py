"""Render gauge-drift metric formulas as high-quality figures for notebook display."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
from matplotlib import mathtext

try:
    from IPython.display import Image, display
except ImportError:  # pragma: no cover - plain python fallback
    Image = None
    display = print

FORMULA_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Encoder 侧",
        [
            r"Q_D^* = \arg\min_{Q^\top Q = I}\;"
            r"\frac{1}{|D|}\sum_{i \in D} \left\|S_i Q - Y_i\right\|_2^2",
            r"e_{\mathrm{frame}}(D) = "
            r"\frac{\left\|Q_D^* - Q_{\mathrm{ref}}^*\right\|_F}{\sqrt{d}}",
            r"\mathrm{Res}_{\mathrm{frame}}(D) = "
            r"\frac{1}{|D|}\sum_{i \in D} \left\|S_i Q_D^* - Y_i\right\|_2^2",
            r"r_{\mathrm{frame}}(D) = "
            r"\frac{\mathrm{Res}_{\mathrm{frame}}(D)}"
            r"{\mathrm{Res}_{\mathrm{frame}}(\mathrm{ref}) + \epsilon}",
        ],
    ),
    (
        "Predictor 侧",
        [
            r"\hat{y}_{t+1} = P(y_t, a_t)",
            r"J_t = \left["
            r"\frac{\partial P}{\partial y}(y_t, a_t),\;"
            r"\frac{\partial P}{\partial a}(y_t, a_t)\right]",
            r"\bar{J}_D = \frac{1}{|D|}\sum_{t \in D} J_t",
            r"\delta_{\mathrm{rule}}(D) = "
            r"\frac{\left\|\bar{J}_D - \bar{J}_{\mathrm{train}}\right\|_F}"
            r"{\left\|\bar{J}_{\mathrm{train}}\right\|_F + \epsilon}",
            r"\mathrm{excess}(D) = "
            r"\delta_{\mathrm{rule}}(D) - "
            r"\mathbb{E}\!\left[\delta_{\mathrm{rule}}(\mathrm{IID})\right]",
            r"z(D) = "
            r"\frac{\mathrm{excess}(D)}"
            r"{\mathrm{Std}\!\left[\delta_{\mathrm{rule}}(\mathrm{IID})\right] + \epsilon}",
        ],
    ),
]


def _validate(tex: str) -> None:
    mathtext.MathTextParser("agg").parse(f"${tex}$", dpi=120)


def render_formula(tex: str, fontsize: int = 22) -> bytes:
    _validate(tex)
    fig = plt.figure(figsize=(11, 0.95), facecolor="white")
    fig.text(
        0.5,
        0.5,
        f"${tex}$",
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight", pad_inches=0.14, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def show_all() -> None:
    for title, formulas in FORMULA_SECTIONS:
        if display is print:
            print(f"\n=== {title} ===")
        else:
            display({"text/plain": title}, raw=True)
        for tex in formulas:
            png = render_formula(tex)
            if Image is None:
                print(tex)
            else:
                display(Image(data=png))


if __name__ == "__main__":
    show_all()
