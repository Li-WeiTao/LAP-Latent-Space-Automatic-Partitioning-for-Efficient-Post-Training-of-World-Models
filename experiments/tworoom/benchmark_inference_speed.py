"""Benchmark planning inference speed: baseline vs rooms3 vs priority5.

Each CEMSolver.solve call plans for all 50 parallel envs (one MPC replan each,
horizon=5 plan steps, 300 candidates, 30 CEM iterations), i.e. exactly
"50 inferences x 5 steps". We time every solve call during a real evaluation
run and compare the per-solve wall time across modes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import hydra
import torch

from stable_worldmodel.solver import CEMSolver

from tworoom_success_rate_eval import (
    PROJECT_ROOT,
    THIS_DIR,
    run_eval,
)

SOLVE_TIMES: list[float] = []
_ORIG_SOLVE = CEMSolver.solve


def _timed_solve(self, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _ORIG_SOLVE(self, *args, **kwargs)
    torch.cuda.synchronize()
    SOLVE_TIMES.append(time.perf_counter() - t0)
    return out


CEMSolver.solve = _timed_solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--doorway80-ckpt",
        type=Path,
        default=THIS_DIR
        / "results/tworoom_geometry_train_region_predictors_doorway80ep/P_train_doorway_corridor_object.ckpt",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=THIS_DIR / "results" / "tworoom_inference_speed_benchmark.json",
    )
    args = parser.parse_args()

    baseline_starts = (
        THIS_DIR / "results" / f"tworoom_success_rate_baseline_seed{args.seed}" / "results.json"
    )
    overrides = {"doorway_corridor": args.doorway80_ckpt}

    report: dict = {"seed": args.seed, "modes": {}}
    for mode in ["baseline", "rooms3", "priority5"]:
        with hydra.initialize_config_dir(
            version_base=None, config_dir=str(PROJECT_ROOT / "config" / "eval")
        ):
            cfg = hydra.compose(config_name="tworoom")
        cfg.seed = args.seed

        out_dir = THIS_DIR / "results" / f"tworoom_speed_benchmark_{mode}_seed{args.seed}"
        SOLVE_TIMES.clear()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            run_eval(
                cfg,
                out_dir,
                mode,
                eval_start_indices_path=baseline_starts,
                region_ckpt_overrides=overrides if mode != "baseline" else None,
            )
        total = time.perf_counter() - t0

        times = list(SOLVE_TIMES)
        # Each run produces eval_budget / (receding_horizon * action_block) solve
        # calls. Only the FIRST solve of each run replans all num_eval envs at
        # once (identical workload across modes); later solves only cover envs
        # still alive, which differs per mode. So we compare first-solve times.
        per_run = len(times) // args.repeats
        first_solves = [times[i * per_run] for i in range(args.repeats)]
        entry = {
            "n_solve_calls": len(times),
            "solves_per_run": per_run,
            "num_envs_per_solve": int(cfg.eval.num_eval),
            "raw_solve_times_sec": times,
            "full_replan_mean_sec": statistics.mean(first_solves),
            "full_replan_std_sec": (
                statistics.stdev(first_solves) if len(first_solves) > 1 else 0.0
            ),
            "per_env_replan_ms": statistics.mean(first_solves)
            / cfg.eval.num_eval
            * 1000,
            "eval_wall_time_sec": total,
        }
        report["modes"][mode] = entry
        print(
            f"[{mode}] full_replan(50 envs)={entry['full_replan_mean_sec']:.3f}s "
            f"(±{entry['full_replan_std_sec']:.3f}) "
            f"per_env_replan={entry['per_env_replan_ms']:.1f}ms",
            flush=True,
        )

    base = report["modes"]["baseline"]["full_replan_mean_sec"]
    for mode in ["rooms3", "priority5"]:
        m = report["modes"][mode]
        m["overhead_vs_baseline_pct"] = (m["full_replan_mean_sec"] / base - 1) * 100
        print(
            f"[{mode}] overhead vs baseline: {m['overhead_vs_baseline_pct']:+.2f}%",
            flush=True,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
