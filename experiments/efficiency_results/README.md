# LAP Efficiency Benchmark Results

## Status (2026-08-19)

| Phase | Status |
|---|---|
| LAP Regional-FT (TwoRoom) | **Pending / invalid preliminary run** — prior scratch used Sub-JEPA cache; do not cite |
| Gate / Partition (one-time) | **Complete** — committed manifest via `--skip-gate-rerun` |
| Joint training (5 epochs, discard epoch 1) | **Pending** — waits for idle GPU via `run_efficiency_joint_when_ready.sh` |
| Inference (4 tasks × 50 repeats) | **Pending** — use `run_efficiency_inference_when_ready.sh` after script fixes |

Formal inference/training require an **idle GPU**: free ≥ 20 GiB and utilization ≤ 10% (configurable via `MIN_FREE_MIB`, `MAX_UTIL_PCT`).

## Reproduction

Full benchmark:

```bash
cd /data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models
PYTHONPATH=experiments/scripts:.:experiments/tworoom \
/data/sicong/weitao/le-wm/.venv/bin/python experiments/scripts/benchmark_efficiency.py \
  --measure train,gate,partition,inference \
  --skip-gate-rerun \
  --training-methods joint,lap \
  --warmup 20 --repeats 50 --seed 20260819 --device cuda:0 \
  --joint-epochs 5 --lap-epochs 5 --discard-warmup-epochs 1 \
  --inference-tasks tworoom,pusht,reacher,cube \
  --output-dir experiments/efficiency_results
```

Split runs:

```bash
# LAP regional training only
PYTHONPATH=experiments/scripts:.:experiments/tworoom \
/data/sicong/weitao/le-wm/.venv/bin/python experiments/scripts/benchmark_efficiency.py \
  --measure train --training-methods lap --device cuda:0 --output-dir experiments/efficiency_results

# Joint when a GPU is idle (default: GPU 0)
experiments/scripts/run_efficiency_joint_when_ready.sh

# Formal inference: 20 warmup + 50 repeats, on first idle GPU
experiments/scripts/run_efficiency_inference_when_ready.sh
```

Do **not** run the 50-repeat inference benchmark on a shared/busy GPU; partial results are kept in `inference_run_shared_gpu.partial.log` only.

Primary metrics:

- **Training:** pure Joint training epoch vs sum of pure LAP regional predictor-training epochs (all K experts), peak GPU memory; epoch 1 discarded; setup/eval excluded.
- **Inference:** original LeWM vs deployed Auto-LAP on the same machine; each timed MPC cycle uses a fresh observation clone; global-gate tasks deploy Global-FT without a router.

Gate and partition are one-time costs and are excluded from seconds/epoch.

Outputs: `efficiency_raw.jsonl`, `training_comparison.csv`, `inference_comparison.csv`, `inference_breakdown.csv`, `efficiency_summary.csv`, `efficiency_table.tex`, `metadata.json`
