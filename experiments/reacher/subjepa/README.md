# Sub-JEPA Reacher

Isolated Reacher experiment tree for Sub-JEPA LAP (Latent-space Automatic
Partitioning), mirroring the structure and protocol of
`experiments/pusht/subjepa/` but with Reacher-specific dataset, checkpoint,
eval config, and canonical evaluation starts. It writes only under
`experiments/reacher/subjepa/` and never touches:

- `experiments/pusht/subjepa/` (PushT Sub-JEPA),
- `experiments/reacher/matrix/`, `experiments/reacher/matrix_long/` (existing
  Reacher LeWM results — read-only reused for canonical paired starts),
- any other already-completed experiment directory.

**Status as of this change: code, config, and static checks only. No cache
preparation, gate, partition fit, training, evaluation, or aggregation has
been run.** No GPU was used; the Sub-JEPA Reacher checkpoint is still
downloading and its final path is not yet confirmed.

## Why this is not a copy-paste of PushT

- PushT Sub-JEPA's `formal`/`matrix` scripts are wired to a smoke-verification
  + material-passport + replay-audit workflow (`verification_status.json`,
  `SMOKE_CACHE_SHA256`, `pusht_formal_lib.py`, `matrix_prelock.py`,
  `matrix_preflight.py`, and the TwoRoom `formal_cache_audit.py` /
  `formal_post_gate_audit.py` / `matrix_frozen_audit.py` /
  `matrix_one_step_mse.py` / `matrix_paired_bootstrap.py` /
  `materialize_spectral_partitions.py` scripts it depends on). Per this
  migration's scope, **none of that audit/passport/checksum machinery is
  reproduced here** — only the LAP method itself (empirical spectral
  degeneracy gate, K=3 region partitioning, 50-epoch predictor training,
  paired short/long evaluation, aggregation) is carried over.
- The Auto-LAP branch (`global` vs `spectral`) is read directly from this
  task's own gate manifest, `formal/gate/partition/manifest.json`
  (`selected_method` field, written by `fit_partition.py --method auto`
  itself). There is no `pre_execution_lock.json`, no `material_passport.json`,
  and no dependency on Reacher LeWM's gate decision.
- Region partitions (K-means++ K3, Spectral K3) are fit directly in the
  matrix work root via the generic driver (`fit_partition.py`), the same way
  the existing Reacher **LeWM** matrix (`experiments/control_matrix/scripts/run_lewm_matrix.sh`)
  already does — not via PushT's formal-cache pre-fit + TwoRoom remap
  (`materialize_spectral_partitions.py`), which this migration intentionally
  does not depend on.
- Reacher eval state comes from `config/eval/reacher.yaml`: `qpos` / `qvel` /
  `goal_qpos` on `swm/ReacherDMControl-v0`, **not** PushT's `state` /
  `goal_state`. `--config-name reacher` is passed explicitly everywhere.

## Files

| Path | Purpose |
|---|---|
| `env.sh` | Shared paths/env vars. `CHECKPOINT` has **no default** (see below). |
| `formal/scripts/run_reacher_gate.sh` | `prepare` (full-cache encode) and `gate` (LAP empirical spectral gate, `fit_partition.py --method auto`). |
| `matrix/scripts/setup_matrix.sh` | Directory scaffolding, read-only cache symlink, canonical paired-start copy (from Reacher LeWM), global partition fit, Auto-LAP branch symlink from the gate manifest. |
| `matrix/scripts/link_auto_lap.sh` | Symlinks `matrix/auto/{training,eval}` to the gate-selected branch (`global` or `spectral partition{deployment_seed}`). |
| `matrix/scripts/run_full_matrix.sh` | Stage controller: `setup`, `partition`, `training`, `eval-short`, `eval-long`, `aggregate`, `all-post-train`. |
| `matrix/scripts/run_eval_only.sh` | Runs `eval-short` then `eval-long`. |
| `scripts/run_reacher_subjepa.sh` | Top-level dispatcher across all 8 stages (`prepare`, `gate`, `partition`, `training`, `eval-short`, `eval-long`, `aggregate`, `all-post-train`). |
| `scripts/launch_formal_detached.sh` | Detached `formal` gate launcher (nohup+setsid). **Created only, not run.** |
| `matrix/scripts/launch_matrix_detached.sh` | Detached matrix training launcher. **Created only, not run.** |
| `matrix/scripts/launch_eval_detached.sh` | Detached eval-short+long launcher. **Created only, not run.** |
| `matrix/scripts/launch_eval_long_detached.sh` | Detached eval-long-only launcher. **Created only, not run.** |
| `../../../configs/experiments/tasks/reacher.json` | Reacher task spec (schema matches `experiments/control_matrix/task_spec.py`). |

## Checkpoint: not yet provided

Every script sources `env.sh` and then calls `require_checkpoint` right
before the checkpoint is actually needed (never merely on source). If
`CHECKPOINT` is unset, the script fails immediately with a clear message; it
never guesses, downloads, loads, or hardcodes a path. Set it explicitly once
the Sub-JEPA Reacher checkpoint is available:

```bash
export CHECKPOINT=/path/to/subjepa_reacher_object.ckpt
```

## Protocol (must match the current PushT Sub-JEPA formal scripts; unchanged here)

- Train seeds: `0, 42, 625`
- Partition seeds: `0, 1, 2`
- Evaluation seeds: `0, 1, 2, 3, 4`
- `K = 3` regions
- 50 epochs per predictor
- Gate diagnostic seeds: `0, 1, 2`; deployment seed: `0`
- Nominal kNN: `30`; perturbed kNN: `27, 33`
- Landmarks: `20000`
- Retained-safety-fraction gap threshold: `0.5`
- Background gap count: `10`; background MAD multiplier: `3.0`
- Gate perturbation multiplier: `2.0`; gate epsilon: `1e-8`

These are the exact values passed to `fit_partition.py --method auto` in
`experiments/pusht/subjepa/formal/scripts/run_formal_gate.sh` on this branch
and are reproduced identically in `formal/scripts/run_reacher_gate.sh`.

## Canonical paired evaluation starts

Reused read-only from the existing Reacher LeWM matrix so Sub-JEPA and LeWM
evaluate from identical initial `qpos`/`qvel`/`goal_qpos` states:

- Short (`goal_offset=25`): `experiments/reacher/matrix/eval/official` (override with `CANON_SHORT`)
- Long (`goal_offset=50`): `experiments/reacher/matrix_long/eval/official` (override with `CANON_LONG`)

`setup_matrix.sh` copies (never resamples) these into
`matrix/paired_starts/canon_{short,long}/eval/official/eval{0..4}/results.json`.
`PAIRED_START_ROOT_SHORT` / `PAIRED_START_ROOT_LONG` are also accepted
directly by the generic driver if you want to point at a different root.

## Reproduction (after checkpoint download completes and a GPU is available)

**Not executed in this change.** Recommended order once `CHECKPOINT` is set:

```bash
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"   # override to use a different interpreter
export DATASET=/home/sicong/weitao/datasets/lewm/reacher.h5
export CHECKPOINT=/path/to/subjepa_reacher_object.ckpt    # must be provided; see above

# 1. Formal: full-cache preparation + LAP empirical spectral gate
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh prepare
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh gate
# or detached:
bash experiments/reacher/subjepa/scripts/launch_formal_detached.sh all

# 2. Matrix: region partitions (K-means++ K3, Spectral K3) + Global partition
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh partition

# 3. Matrix: 50-epoch training (Global-FT50, K-means++ K3-50, Spectral K3-50)
#    + Auto-LAP symlink from the gate-selected branch
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh training
# or detached:
bash experiments/reacher/subjepa/matrix/scripts/launch_matrix_detached.sh

# 4. Paired short/long evaluation (official + Global-FT + K-means++ + Spectral + Auto-LAP)
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh eval-short
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh eval-long
# or detached:
bash experiments/reacher/subjepa/matrix/scripts/launch_eval_detached.sh

# 5. Aggregate both horizons into matrix_summary_{short,long}.json
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh aggregate

# ...or steps 4+5 together:
bash experiments/reacher/subjepa/scripts/run_reacher_subjepa.sh all-post-train
```

Direct calls to the generic driver also work (this is what the scripts above
call under the hood):

```bash
bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec configs/experiments/tasks/reacher.json \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --eval-config-name reacher \
  --work-root experiments/reacher/subjepa/formal \
  --cache-dir "${CACHE_DIR:-/data/sicong/weitao/.stableworldmodel/subjepa/reacher}" \
  --phase prepare
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `PYTHON` | `${REPO_ROOT}/.venv/bin/python` | Override to use a different interpreter. |
| `DATASET` | `/home/sicong/weitao/datasets/lewm/reacher.h5` | Fixed per task instructions; confirmed present (~98 GB) on this host. |
| `CHECKPOINT` | *(none — required)* | Must be set explicitly; scripts error clearly if unset. |
| `TASK_SPEC` | `configs/experiments/tasks/reacher.json` | |
| `CACHE_DIR` | `/data/sicong/weitao/.stableworldmodel/subjepa/reacher` | Isolated from PushT (`.../subjepa/pusht`) and from LeWM Reacher's cache dir. |
| `WORK_ROOT` / `SMOKE_ROOT` | `experiments/reacher/subjepa` | |
| `FORMAL` | `experiments/reacher/subjepa/formal` | |
| `MATRIX` | `experiments/reacher/subjepa/matrix` | |
| `CANON_SHORT` | `experiments/reacher/matrix/eval/official` | Reacher LeWM short-horizon official eval (read-only). |
| `CANON_LONG` | `experiments/reacher/matrix_long/eval/official` | Reacher LeWM long-horizon official eval (read-only). |
| `GPU_IDS`, `GPU_ID`, `CPU_THREADS` | `0`, `0`, `4` | Override for the target host's hardware. |
| `TRAIN_SEEDS`, `PARTITION_SEEDS`, `EVAL_SEEDS`, `MATRIX_METHODS`, `DEPLOYMENT_SEED` | see Protocol above | |
| `RUN_ID` | UTC timestamp at invocation time | Never hardcoded. |

## What was intentionally not added

Per task scope, this migration adds **no new audit workflow**: no
`verification_status.json`, `material_passport.json`, replay audit, frozen
audit, route-equivalence audit, cache-equivalence audit, smoke-cache SHA
lock, `SMOKE_CACHE_SHA256`, `verify_smoke_untouched`/`require_smoke_verified`
gate, dependency on TwoRoom audit/materialization scripts, or new
checksum/passport/`VERIFIED` pre-execution gate. The LAP empirical spectral
gate (`fit_partition.py --method auto`) is preserved in full — it is the
experimental method, not an audit step.
