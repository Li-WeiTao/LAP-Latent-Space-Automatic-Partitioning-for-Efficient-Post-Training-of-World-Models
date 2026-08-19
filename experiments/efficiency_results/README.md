# LAP Efficiency Benchmark Results

## Status (2026-08-19)

| Phase | Status |
|---|---|
| CPU dry-run (provenance + pool alignment) | **Complete** — `experiments/scripts/dry_run_efficiency.py` |
| Joint + LAP training (same GPU, 5 epochs) | **Pending** — `run_efficiency_training_when_ready.sh` |
| Gate / Partition (one-time) | **Complete** — committed manifest via `--skip-gate-rerun` |
| Inference (4 tasks × 50 repeats, B=1 planning) | **Pending** — `run_efficiency_inference_when_ready.sh` |

CPU validation before GPU:

```bash
cd /data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models
/data/sicong/weitao/le-wm/.venv/bin/python experiments/scripts/dry_run_efficiency.py
```

Formal inference/training require an **idle GPU**: free ≥ 20 GiB and utilization ≤ 10% (configurable via `MIN_FREE_MIB`, `MAX_UTIL_PCT`).

## Reproduction

Full benchmark (single idle GPU session):

```bash
cd /data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models
PYTHONPATH=experiments/scripts:.:experiments/tworoom:/data/sicong/weitao/le-wm \
/data/sicong/weitao/le-wm/.venv/bin/python experiments/scripts/benchmark_efficiency.py \
  --measure train,gate,partition,inference \
  --skip-gate-rerun \
  --training-methods joint,lap \
  --warmup 20 --repeats 50 --seed 20260819 --device cuda:0 \
  --joint-epochs 5 --lap-epochs 5 --discard-warmup-epochs 1 \
  --inference-tasks tworoom,pusht,reacher,cube \
  --output-dir experiments/efficiency_results
```

Recommended split runs (immutable scratch artifacts + aggregate):

```bash
# Joint + LAP training together on one GPU (preferred)
experiments/scripts/run_efficiency_training_when_ready.sh

# Inference only (single-env planning latency; merges with scratch training)
experiments/scripts/run_efficiency_inference_when_ready.sh

# Rebuild tables/CSVs from scratch without rerunning GPU work
PYTHONPATH=experiments/scripts:.:experiments/tworoom:/data/sicong/weitao/le-wm \
/data/sicong/weitao/le-wm/.venv/bin/python experiments/scripts/benchmark_efficiency.py \
  --aggregate-only \
  --output-dir experiments/efficiency_results
```

Scratch artifacts (never overwritten across split runs):

- `scratch/training/joint_training.json`
- `scratch/training/lap_regional_training.json`
- `scratch/gate_partition/gate_partition.json`
- `scratch/inference/inference_<task>_<mode>.json`

Do **not** run the 50-repeat inference benchmark on a shared/busy GPU; partial results are kept in `inference_run_shared_gpu.partial.log` only.

Primary metrics:

- **Training:** pure Joint training epoch vs sum of pure LAP regional predictor-training epochs (all K experts), peak GPU memory = max over experts after releasing each predictor; epoch 1 discarded; setup/eval excluded.
- **Inference:** single-environment complete CEM planning latency (`timed_num_envs=1`); original LeWM vs deployed Auto-LAP on the same machine; each timed MPC cycle uses a fresh B=1 observation clone; global-gate tasks deploy Global-FT without a router (routing column = `N/A`).

Gate and partition are one-time costs and are excluded from seconds/epoch.

Outputs: `efficiency_raw.jsonl`, `training_comparison.csv`, `inference_comparison.csv`, `inference_breakdown.csv`, `efficiency_summary.csv`, `efficiency_table.tex`, `metadata.json`
