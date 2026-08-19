"""Summary statistics for repeated timing measurements."""

from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = [float(v) for v in values]
    means = []
    n = len(samples)
    for _ in range(n_bootstrap):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(draw))
    means.sort()
    low_idx = max(0, int((alpha / 2) * len(means)) - 1)
    high_idx = min(len(means) - 1, int((1 - alpha / 2) * len(means)))
    return means[low_idx], means[high_idx]


def summarize(values: Sequence[float], *, seed: int = 0) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    samples = [float(v) for v in values]
    ci_low, ci_high = bootstrap_ci(samples, seed=seed)
    return {
        "count": float(len(samples)),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p5": percentile(samples, 5),
        "p95": percentile(samples, 95),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
