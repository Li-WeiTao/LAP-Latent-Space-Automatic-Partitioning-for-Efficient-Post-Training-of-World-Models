# Sub-JEPA TwoRoom 50-Epoch Matrix

Task-local orchestration for the **Sub-JEPA formal five-method** comparison.
Training and evaluation **protocol** (seeds, horizons, paired starts, hyperparameters)
is locked to the canonical LeWM TwoRoom matrix via `protocol_parity.json`. Sub-JEPA
**method scope** intentionally omits LeWM-only baselines (Random-Voronoi, Joint-Continue,
Human partition).

Formal gate artifacts supply spectral partitions, deployment seed 0, and router metadata.

**Do not write into** `experiments/tworoom/matrix` or other LeWM result trees.

## Five comparison methods

| # | Method | Training | Eval |
|---|--------|----------|------|
| 1 | **Official Sub-JEPA** | None (original checkpoint) | short + long |
| 2 | **Global-FT50** | 3 train seeds: 0, 42, 625 | short + long |
| 3 | **K-means++ K3-50** | 3 partition seeds × 3 train seeds = 9 jobs | short + long |
| 4 | **Spectral K3-50** (forced) | 9 jobs; spectral partitions reused from formal gate | short + long |
| 5 | **Auto-LAP** | No extra training — symlinks to Spectral partition seed **0** | short + long (identical to deployed Spectral) |

**Not in scope:** Random-Voronoi, Human partition, Joint-Continue.

**Training jobs:** 3 global + 9 kmeanspp + 9 spectral = **21** (50 epochs each).

## Prerequisites

| Item | Location |
|------|----------|
| Formal gate | `../formal/` — `verification_status: VERIFIED`, spectral seed 0 |
| Full latent cache | `../formal/preparation/embedding_cache.npz` (693,728 transitions) |
| Sub-JEPA checkpoint | `.stable_worldmodel/tworoom/subjepa_object.ckpt` |
| Dataset | `tworoom.h5` (same file as LeWM matrix) |
| LeWM paired eval starts | `experiments/tworoom/results/tworoom_success_rate_baseline_seed{0-4}` (short), `..._baseline_exp6_seed{0-4}` (long) |
| Python env | `le-wm/.venv` (Sub-JEPA checkpoint compat) |

## Directory layout

```
matrix/
├── preparation/          → symlink to ../formal/preparation (read-only cache)
├── partitions/
│   ├── global/seed0/     → fitted once on full cache
│   ├── spectral/seed{0,1,2}/ → symlink to ../formal/partitions/spectral/...
│   └── kmeanspp/seed{0,1,2}/
├── auto/
│   ├── partition/        → symlink to ../formal/gate/partition
│   ├── training/train*   → symlinks to spectral/partition0_train*
│   └── eval/train*/      → symlinks to spectral/partition0_train*/eval*
├── paired_starts/        → copies of LeWM baseline results.json per eval seed
├── training/             → 50-epoch predictor runs
├── eval_short/ eval_long/ → snapshots after short/long MPC eval
├── manifests/
│   ├── protocol_parity.json
│   ├── pre_execution_lock.json
│   └── ...
└── logs/
```

## Protocol parity (required before run)

Compares Sub-JEPA against the LeWM canonical matrix on **training/eval protocol**
fields. Allowed diffs include model family, checkpoint, cache, and Sub-JEPA method
scope (`methods=kmeanspp,spectral`, `skip_joint=1`).

```bash
PYTHON=/data/sicong/weitao/le-wm/.venv/bin/python \
  experiments/tworoom/subjepa/matrix/scripts/protocol_parity.py \
  --out experiments/tworoom/subjepa/matrix/manifests/protocol_parity.json
```

Exit code must be 0 (`parity_status: PASSED`).

## How to run

All commands from the **repository root**.

### Recommended: detached from IDE (nohup + setsid)

```bash
bash experiments/tworoom/subjepa/matrix/scripts/launch_matrix_detached.sh
```

Monitor:

```bash
tail -f experiments/tworoom/subjepa/matrix/logs/detached_<RUN_ID>.log
watch -n5 nvidia-smi
```

Stop:

```bash
kill $(cat experiments/tworoom/subjepa/matrix/logs/detached_<RUN_ID>.pid)
```

### Stages

| Command | What it does |
|---------|----------------|
| `setup` | Gate check, protocol parity, cache/partition links, LeWM paired starts |
| `training` | setup + partition (kmeanspp only; spectral reused) + **21** training jobs |
| `eval-short` | MPC eval goal_offset=25, snapshot → `eval_short/` |
| `eval-long` | MPC eval goal_offset=50, snapshot → `eval_long/` |
| `audit` | Frozen-parameter audit + one-step latent MSE |
| `aggregate` | Success-rate tables (includes Auto-LAP row) |
| `bootstrap` | 200k paired block bootstrap |
| `all-post-train` | eval-short → eval-long → audit → aggregate → bootstrap |

After training:

```bash
bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh all-post-train
```

## GPU scheduling and wall time

8 GPUs, round-robin queue, ~2.6 jobs/GPU for 21 jobs.

| Job type | Count | Approx. time each |
|----------|-------|-------------------|
| Regional 50ep (3 clusters) | 18 | ~1.5–2 h |
| Global 50ep | 3 | ~75–90 min |

**Training wall time (8×4090): ~4–6 hours.**

Post-train eval/audit/bootstrap adds several hours depending on eval parallelism.

## Resume behavior

Completed jobs skip when `training/**/manifest.json` exists. Safe to re-run
`launch_matrix_detached.sh` after interruption. Auto-LAP symlinks refresh via
`scripts/link_auto_lap.sh` after training/eval.

## Smoke (separate)

Reduced 4096-cap smoke lives under `../`. See `../README.md`.
