# Sub-JEPA PushT

**Smoke: `VERIFIED` (sicong 2026-08-06)** — see `manifests/verification_status.json`.
**Formal + 50-epoch matrix + paired MPC eval: complete (sicong 2026-08-09).**

Cache-equivalence uses the same production unique-frame / `exact_batch_shapes` replay
as TwoRoom. Preserved smoke cache SHA-256:
`3d2e75d1e347c1826b94ab47a474d8e97af0eb92a7a9f6f63a733dcb3177ec3e`
(also recorded in `manifests/verification_status.json` and `env.sh`).

Reduced smoke eval proves pipeline wiring only; formal method conclusions require
full paired short/long evaluation.

Smoke artifacts protected during formal (`verification_status.json`,
`smoke_cache-equivalence.json`, `smoke_frozen-audit.json`, `smoke_route-equivalence.json`);
formal writes only under `formal/`, never `preparation/` at smoke root.

## Reproduction

```bash
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export DATASET=/home/sicong/weitao/datasets/lewm/pusht_expert_train.h5
export CHECKPOINT=/data/sicong/weitao/.stable_worldmodel/pusht/subjepa_object.ckpt

bash experiments/pusht/subjepa/scripts/launch_smoke_detached.sh
# or inline:
bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec configs/experiments/tasks/pusht.json \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --eval-config-name pusht \
  --work-root experiments/pusht/subjepa \
  --cache-dir "${CACHE_DIR:-/data/sicong/weitao/.stableworldmodel/subjepa/pusht}" \
  --max-train-starts 4096 \
  --phase smoke
```

Formal pipeline (after smoke VERIFIED; same structure as `experiments/tworoom/subjepa/`):

```bash
# Stage 1: full cache + LAP gate + passport
bash experiments/pusht/subjepa/scripts/launch_formal_detached.sh all
# or inline:
bash experiments/pusht/subjepa/formal/scripts/run_formal_gate.sh all

# Stage 2: 50-epoch matrix training (preflight must pass)
bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh preflight
bash experiments/pusht/subjepa/matrix/scripts/launch_matrix_detached.sh

# Stage 3: paired short/long eval + audit + aggregate
bash experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh all-post-train
```

Audit/materialization scripts are reused from `experiments/tworoom/subjepa/` (task-agnostic
via `--work-root`). Paired eval starts copy from `experiments/pusht/matrix/` (short) and
`experiments/pusht/matrix_long/` (long) official eval.

See `experiments/tworoom/subjepa/README.md` and `experiments/tworoom/subjepa/matrix/README.md`
for the reference layout.

## Server runbook (sicong, Aug 2026)

| Artifact | Path |
|----------|------|
| Dataset | `/home/sicong/weitao/datasets/lewm/pusht_expert_train.h5` (44 GiB) |
| Sub-JEPA checkpoint | `/data/sicong/weitao/.stable_worldmodel/pusht/subjepa_object.ckpt` |
| Smoke work root | `experiments/pusht/subjepa/` |
| Formal cache + gate | `experiments/pusht/subjepa/formal/` |
| 50-epoch matrix | `experiments/pusht/subjepa/matrix/` |

PushT eval uses `config/eval/pusht.yaml` (`state` / `goal_state`, not TwoRoom
`proprio` / `goal_proprio`). Shared evaluator:
`experiments/tworoom/tworoom_success_rate_eval.py` with `--config-name pusht`.

**Environment:** install PushT eval deps with `uv pip install -r requirements/pusht.txt`
(includes `pygame`, `pymunk`, `shapely`, `opencv-python-headless`).

Check smoke: `tail -f experiments/pusht/subjepa/logs/smoke_*.log`

## Results summary (sicong, Aug 2026)

Paired MPC eval: 50 episodes × 5 eval seeds; train seeds 0, 42, 625; partition
seeds 0, 1, 2. Error bars: sample SD across fine-tuning seeds (after averaging
partition and eval seeds). Full tables:
`matrix/manifests/matrix_summary_{short,long}.json`,
`matrix/manifests/matrix_raw_{short,long}.csv`. Eval snapshots:
`matrix/eval_short/`, `matrix/eval_long/`.

### Formal gate (`formal/`, 2026-08-07)

| Field | Value |
|-------|-------|
| Status | `VERIFIED` |
| Selected branch | **global** (`safety_and_background_checks_failed`) |
| Deployment seed | **0** |
| Safety retention `S_task` | 0.206 (threshold 0.5 — **failed**) |
| Background gap `R_K` vs `T_bg` | 0.031 vs 0.159 (**failed**) |
| Artifacts | `formal/manifests/material_passport.json`, `formal/gate/partition/manifest.json` |

Spectral partitions were materialized for the matrix ablation, but the gate did
not pass safety/background checks on PushT Sub-JEPA latents. Auto-LAP symlinks
the **Global-FT50** branch (not spectral).

### 50-epoch matrix MPC success rates

**Short horizon** (`goal_offset_steps = 25`):

| Method | Mean | SD (FT seeds) |
|--------|------|---------------|
| Official baseline | 92.4% | — |
| Global-FT50 | **94.5%** | ±1.0% |
| K-means++ | 94.5% | ±0.2% |
| Spectral | 94.4% | ±0.3% |
| Auto-LAP | **94.5%** | ±1.0% |

**Long horizon** (`goal_offset_steps = 50`):

| Method | Mean | SD (FT seeds) |
|--------|------|---------------|
| Official baseline | 38.8% | — |
| Global-FT50 | **46.4%** | ±0.4% |
| K-means++ | 42.9% | ±0.7% |
| Spectral | 44.1% | ±0.9% |
| Auto-LAP | **46.4%** | ±0.4% |

### Interpretation

1. **Post-training helps on both horizons.** Official is lowest; 50-epoch
   fine-tuning gains ~2 pp (short) and ~8 pp (long) over the frozen baseline.
2. **Short horizon: methods tie.** Global-FT, K-means++, and Spectral cluster
   around 94.4–94.5%; differences are within fine-tuning seed noise.
3. **Long horizon: Global-FT wins.** Regional LAP (K-means++ ~43%, Spectral ~44%)
   underperforms Global-FT (~46%). Hard K=3 splits plus MPC routing do not beat a
   single global predictor on the harder long-horizon task.
4. **Auto-LAP matches Global-FT exactly**, as expected from the gate selecting
   `global`. This is the deployed recommendation for PushT Sub-JEPA on this run.
5. **Consistent with a failed gate.** Weak safety retention and background gap
   on PushT latents predict limited benefit from regional post-training; the long
   horizon confirms Global-FT as the stronger default.

### Run notes

| Stage | Status | Notes |
|-------|--------|-------|
| Formal re-encode + gate | complete | 1,850,815 transitions; gate → `global` |
| Matrix training | 21/21 | global(3) + kmeanspp(9) + spectral(9), 50 epochs |
| Eval short + long | 22/22 each | GPU 1+3 (avoid GPU4; skip nearly-full cards) |
| Aggregate | complete | `run_full_matrix.sh aggregate` |

Matrix layout mirrors `experiments/tworoom/subjepa/matrix/` (Sub-JEPA TwoRoom
structure, not LeWM protocol parity).
