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
| Python env | Repo `.venv` (Python 3.10; see `../README.md` server runbook) |
| Extra pip deps | `hdf5plugin`, `imageio[ffmpeg]` |

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
| `eval-short` | MPC eval goal_offset=25 (Official Sub-JEPA + Global + Regional), snapshot → `eval_short/` |
| `eval-long` | MPC eval goal_offset=50 (same methods), snapshot → `eval_long/` |
| `audit` | Frozen-parameter audit + one-step latent MSE |
| `aggregate` | Success-rate tables (includes Auto-LAP row) |
| `bootstrap` | 200k paired block bootstrap |
| `all-post-train` | eval-short → eval-long → audit → aggregate → bootstrap |

After training:

```bash
bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh all-post-train
```

### Eval only (short + long, no audit/aggregate/bootstrap)

When training is already complete and you only need MPC success-rate evaluation
(Official + Global-FT + K-means++ + Spectral):

```bash
GPU_IDS=0,1,2,3,4,5,6 \
bash experiments/tworoom/subjepa/matrix/scripts/launch_eval_detached.sh
```

# Long only (after short is done)
bash experiments/tworoom/subjepa/matrix/scripts/launch_eval_long_detached.sh
```

This runs `run_eval_only.sh` → `eval-short` then `eval-long` with **7-GPU parallel**
dispatch (22 tasks per horizon). Logs under `matrix/logs/eval_only_<RUN_ID>.log`.
Outputs snapshot to `eval_short/` and `eval_long/`.

Long eval passes horizon via `EVAL_GOAL_OFFSET` in `run_jepa_matrix_parallel.sh`
(not bare `GOAL_OFFSET`, which conflicts with task-spec `short_goal_offset`).

Requires `imageio[ffmpeg]` in the active venv.

### What `audit` does (optional post-train stage)

Not part of success-rate evaluation. The `audit` stage runs two checks on completed
training jobs:

1. **`matrix_frozen_audit.py`**: every trained checkpoint must leave encoder (and all
   non-predictor) weights identical to the original Sub-JEPA checkpoint; only
   `predictor` / `pred_proj` may change.
2. **`matrix_one_step_mse.py`**: one-step latent prediction MSE on a fixed 5% holdout
   of `embedding_cache.npz` for each regional predictor.

Use audit for reproducibility reporting; skip it if you only need MPC eval numbers.

## Results (sicong, Aug 2026)

Training, short/long MPC eval, and aggregate completed on 7× RTX 3090. Gate
artifacts were produced on another machine (same protocol) and committed under
`../formal/`; this server re-encoded the full latent cache locally and reused
formal spectral partitions.

Raw outputs:

| Artifact | Path |
|----------|------|
| Short eval snapshots | `eval_short/` (110 `results.json`) |
| Long eval snapshots | `eval_long/` (110 `results.json`) |
| Aggregate tables | `manifests/matrix_summary_short.json`, `matrix_summary_long.json` |
| Raw CSV | `manifests/matrix_raw_short.csv`, `matrix_raw_long.csv` |

Reproduce aggregate after eval:

```bash
PYTHON=$PWD/.venv/bin/python bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh aggregate
```

### Formal gate

Source: `../formal/manifests/material_passport.json` (`id`:
`subjepa-tworoom-formal-gate-2026-08-03`).

| Field | Value |
|-------|-------|
| `verification_status` | `VERIFIED` |
| `selected_branch` | `spectral` |
| `selected_reason` | `spectrally_nondegenerate` |
| `deployment_seed` | `0` |

Gate criteria (`lap/partition/gate.py`): safety `S_task ≥ 0.5` and background
`R_K > T_bg`, where `R_K = E_K_min − T_E_K_max`.

| Metric | Value | Notes |
|--------|-------|-------|
| `E_K_min` | 0.489 | Minimum nominal relative eigengap across 3 diagnostic seeds |
| `T_E_K_max` | 0.0055 | Max kNN perturbation (×2 multiplier) |
| `S_task` | 0.989 | Safety retention — **wide margin** above 0.5 |
| `R_K` | 0.483 | Robust residual gap |
| `T_bg` | 0.475 | Background threshold (max over 9 diagnostic graphs) |
| **`R_K − T_bg`** | **0.008** | **Narrow pass** (~1.7% above threshold) |

Tightest single graph: diagnostic seed **2**, kNN **33** (candidate gap 0.490 vs
background threshold 0.475, margin ≈ 0.015). Seeds 0 and 1 had much larger margins.

Deployed partition stability (pairwise ARI across partition seeds 0/1/2):
**0.94–0.96**. Router holdout macro-F1: **1.0**.

Spectral structure at deployment (seed 0): `eigengap_after_k ≈ 0.00069`; first
Laplacian eigenvalues after the zero mode are very small — partition signal exists
but is weak. TwoRoom has low intrinsic dimensionality; Sub-JEPA’s global Gaussian
regularization may already impose strong structure, leaving little benefit from hard
K=3 splits.

Detailed gate diagnostics: `../formal/gate/partition/manifest.json`,
`../formal/manifests/post_gate_audit.json`.

### Matrix MPC success rates

From `manifests/matrix_summary_{short,long}.json` after paired MPC eval (50 episodes
× 5 eval seeds; train seeds 0, 42, 625; partition seeds 0, 1, 2). Error bars:
sample SD across fine-tuning seeds.

**Short horizon** (`goal_offset_steps = 25`):

| Method | Mean | SD (FT seeds) |
|--------|------|---------------|
| Official baseline | 94.0% | — |
| Global-FT50 | **94.8%** | ±0.0% |
| K-means++ | 93.9% | ±0.1% |
| Spectral | 94.0% | ±0.3% |
| Auto-LAP | 93.9% | ±0.2% |

**Long horizon** (`goal_offset_steps = 50`):

| Method | Mean | SD (FT seeds) |
|--------|------|---------------|
| Official baseline | 57.2% | — |
| Global-FT50 | **58.5%** | ±0.6% |
| K-means++ | 57.9% | ±0.8% |
| Spectral | 58.3% | ±0.5% |
| Auto-LAP | 58.7% | ±1.4% |

### Interpretation

1. **Post-training helps.** Official is lowest on both horizons; Global-FT and
   regional predictors gain ~0.8–1.3 pp (short) and ~1–1.5 pp (long).
2. **Regional methods ≈ Global-FT.** K-means++, Spectral, and Auto-LAP do not
   consistently beat Global-FT50; differences are within fine-tuning seed noise.
3. **Consistent with a narrow gate.** The background check passed by only
   `R_K − T_bg ≈ 0.008`, so latent space barely supports a confident K=3 spectral
   split. Expect limited uplift from regional post-training on this task/checkpoint.
4. **Auto-LAP on long** has the highest mean (58.7%) but the largest FT-seed SD
   (±1.4%), matching unstable routing gains when partition signal is weak.

This run is best read as a **neutral / weak-positive** LAP scenario for Sub-JEPA
TwoRoom, not a strong spectral win. Compare against LeWM TwoRoom gate margins and
matrix numbers when available.

## GPU scheduling and wall time

7–8 GPUs, round-robin queue, ~3 jobs/GPU for 21 training jobs; ~3.1 waves for 22 eval
tasks per horizon.

| Job type | Count | Approx. time each |
|----------|-------|-------------------|
| Regional 50ep (3 clusters) | 18 | ~1.5–2 h |
| Global 50ep | 3 | ~75–90 min |

**Training wall time (8×4090): ~4–6 hours; observed ~10.5 h on 7×3090 (sicong, Aug 2026).**

Eval-only (short + long, 7 GPUs): ~4–6 hours. Audit/aggregate/bootstrap add further
time if enabled.

## Resume behavior

Completed jobs skip when `training/**/manifest.json` exists. Safe to re-run
`launch_matrix_detached.sh` after interruption. Auto-LAP symlinks refresh via
`scripts/link_auto_lap.sh` after training/eval.

## Smoke (separate)

Reduced 4096-cap smoke lives under `../`. See `../README.md`.
