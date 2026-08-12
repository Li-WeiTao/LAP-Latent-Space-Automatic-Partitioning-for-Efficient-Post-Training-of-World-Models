# OGBench Cube

Internal task name: `cube`. Human-readable name: **OGBench Cube**
(`swm/OGBCube-v0`).

Mirrors the PushT / Reacher control-matrix setup: Le-WM comparison matrix
(`experiments/cube/`) and a Sub-JEPA comparison matrix
(`experiments/cube/subjepa/`), both thin task-local wrappers around the
generic, task-agnostic `experiments/control_matrix/` driver. No new audit
scripts were added or copied for this task — see "Excluded audits" below.

## Data

```
/home/sicong/weitao/datasets/lewm/cube_single_expert.h5
```

2,010,000 transitions / 10,000 episodes of length 201. Key schema (verified
with `h5py`, read-only):

| key | shape | dtype |
|---|---|---|
| `action` | (2010000, 5) | float32 |
| `observation` | (2010000, 28) | float64 |
| `pixels` | (2010000, 224, 224, 3) | uint8 |
| `qpos` | (2010000, 21) | float64 |
| `qvel` | (2010000, 20) | float64 |
| `privileged_block_0_pos` | (2010000, 3) | float64 |
| `privileged_block_0_quat` | (2010000, 4) | float64 |
| `privileged_target_block_pos` | (2010000, 3) | float64 |
| `ep_idx` / `ep_len` / `ep_offset` | — | int32/int32/int64 |

`ep_len` is uniformly 201; `ep_offset` gives per-episode start row.
`experiments/tworoom/gauge_drift.py` registers `"cube"` in `DATASETS`
(`default_file="cube_single_expert.h5"`, `pixel_key="pixels"`,
`state_keys=("observation", "qpos")`, `split_fn=cube_splits`). Like Reacher,
`cube_splits` only emits a `"common"` region (no manually designed
partition); LAP's automatic spectral gate produces any regional structure
from the latent cache, not from this file.

## Checkpoints

- **Sub-JEPA Cube object checkpoint (found):**
  `/data/sicong/weitao/.stable_worldmodel/cube/subjepa_object.ckpt`
  — verified with `scripts/probe_jepa_checkpoint.py` (status `PASSED`;
  encoder/predictor/planning cost all finite, `action_encoder_input_dim=25`
  = 5 actions × frameskip 5). Set automatically by
  `experiments/cube/subjepa/env.sh`; override via `CHECKPOINT` if needed.
- **Le-WM Cube object checkpoint (NOT found):** searched
  `/home/sicong/weitao/` and `/data/sicong/weitao/` (including
  `.stable_worldmodel/`); only `pusht/`, `tworoom/`, `reacher/`, and `cube/`
  Sub-JEPA checkpoints exist under `.stable_worldmodel/`, no `lewm_object.ckpt`
  for any task. `experiments/cube/scripts/run_cube_matrix.sh` and
  `launch_cube_matrix_parallel.sh` require `CHECKPOINT` to be set explicitly
  and fail loudly (not silently) if it is missing — verified:
  `bash experiments/cube/scripts/run_cube_matrix.sh` with no `CHECKPOINT` set
  exits 1 with "set CHECKPOINT to the official Le-WM Cube object checkpoint
  ... None was found ... see experiments/cube/README.md 'Known blockers'."

## Cube evaluator (`config/eval/cube.yaml`)

Base copied from the authoritative `/data/sicong/weitao/config/eval/cube.yaml`
(the adjacent reference Le-WM installation's own Cube config), **not**
guessed from PushT fields. One task-agnostic compatibility addition on top:
`dataset.keys_to_cache` adds `observation` (this repo's shared evaluator,
`experiments/tworoom/tworoom_success_rate_eval.py`, eagerly builds a
`StandardScaler` over every key in `keys_to_cache` and requires at least one
of `proprio/state/qpos/observation/finger_pos`; the upstream config only
cached `action`). No env/callable/dataset semantics were changed.

- `world.env_name: swm/OGBCube-v0`, `env_type: single`, `ob_type: states`,
  `width`/`height: 224` (matches `img_size: 224` in `cube.json` and the
  checkpoint's trained resolution).
- `eval.callables`:
  - `set_state(qpos=<qpos>, qvel=<qvel>)`
  - `set_target_pos(cube_id=0 [literal, not from dataset], target_pos=<goal_privileged_block_0_pos>, target_quat=<goal_privileged_block_0_quat>)`
    (`goal_<key>` reads the dataset's `<key>` column at the sampled goal
    offset; `privileged_block_0_pos`/`_quat` are real HDF5 columns, verified
    above).
- `world.terminate_at_goal: true` — same success/termination behavior as the
  upstream Cube evaluator; not modified.
- `eval.dataset_name: ogbench/cube_single_expert` — resolved through
  `$CACHE_DIR/datasets/ogbench/cube_single_expert.h5`. The generic driver's
  `prepare()` (in `run_jepa_matrix.sh`) creates this symlink itself and
  refuses to proceed if the path already resolves elsewhere; verified during
  smoke: `/data/sicong/weitao/.stableworldmodel/subjepa/cube/datasets/ogbench/cube_single_expert.h5`
  → `/home/sicong/weitao/datasets/lewm/cube_single_expert.h5`.

Verified: `gym.make("swm/OGBCube-v0", env_type="single", ob_type="states")`
creates successfully in this repo's `.venv`; `obs.shape == (28,)` (matches
`observation` in the HDF5); `hasattr(env.unwrapped, "set_state")` and
`hasattr(env.unwrapped, "set_target_pos")` are both `True`.

## Dependencies

`swm/OGBCube-v0` needs `mujoco`, `dm_control`, `ogbench`, `dm-env`, which
this repo's `.venv` did not have. Installed on 2026-08-10 (versions recorded
from the adjacent, already-working Le-WM installation at
`/data/sicong/weitao/.venv`; no conflicts with existing `torch`/`numpy`
pins — checked with `uv pip install --dry-run` first):

```bash
uv pip install --python .venv/bin/python mujoco==3.10.0 dm_control==1.0.43 ogbench==1.2.1 dm-env==1.6
```

Pinned in `requirements/cube.txt` (extends `requirements/tworoom.txt`).

## Cache directory layout

- Le-WM matrix cache root: `CACHE_DIR` (default
  `/data/sicong/weitao/.stable_worldmodel`); stable-worldmodel appends
  `datasets/<eval_dataset_name>.h5`.
- Sub-JEPA cache root: `CACHE_DIR` (default
  `/data/sicong/weitao/.stableworldmodel/subjepa/cube`), set in
  `experiments/cube/subjepa/env.sh`.
- Both are created/symlinked automatically and safely by the generic
  driver's `prepare()` step (existing-path conflict → hard error, never
  silently overwritten).

## Naming and directory structure

```
experiments/cube/README.md
experiments/cube/scripts/run_cube_matrix.sh              # Le-WM task-local wrapper
experiments/cube/scripts/launch_cube_matrix_parallel.sh  # Le-WM detached multi-GPU launcher
experiments/cube/subjepa/env.sh                          # Sub-JEPA shared paths (no PushT paths/SHAs)
experiments/cube/subjepa/scripts/run_cube_subjepa.sh      # Sub-JEPA main dispatcher
experiments/cube/subjepa/scripts/run_cube_smoke.sh        # restricted smoke test
experiments/cube/subjepa/scripts/launch_formal_detached.sh
experiments/cube/subjepa/scripts/run_pipeline_after_formal.sh
experiments/cube/subjepa/formal/scripts/run_cube_gate.sh  # prepare + LAP auto gate
experiments/cube/subjepa/matrix/scripts/setup_matrix.sh
experiments/cube/subjepa/matrix/scripts/bootstrap_canonical_starts.sh
experiments/cube/subjepa/matrix/scripts/link_auto_lap.sh
experiments/cube/subjepa/matrix/scripts/run_full_matrix.sh
experiments/cube/subjepa/matrix/scripts/run_eval_only.sh
experiments/cube/subjepa/matrix/scripts/launch_matrix_detached.sh
experiments/cube/subjepa/matrix/scripts/launch_eval_detached.sh
experiments/cube/subjepa/matrix/scripts/launch_eval_long_detached.sh
```

No results/logs/manifests/caches/partitions/checkpoints/verification_status
were copied from PushT; no PushT SHA256 constants; no PushT paired starts.
Every script is a thin wrapper that sets Cube-specific defaults and execs
into `experiments/control_matrix/scripts/run_lewm_matrix.sh`,
`run_lewm_matrix_parallel.sh`, `run_subjepa_matrix.sh`
(`run_jepa_matrix.sh`), or `run_jepa_matrix_parallel.sh` — no large chunks of
generic Python logic were duplicated.

## Common config

- `configs/experiments/tasks/cube.json` — same numeric protocol as PushT
  (`frameskip: 5`, `history_size: 3`, `num_preds: 1`, `img_size: 224`,
  `short_goal_offset: 25`, `long_goal_offset: 50`, `eval_budget: 50`,
  `num_eval: 50`, `plan_horizon/receding_horizon/action_block: 5`),
  `task_name/dataset_name: cube`, `eval_config_name: cube`,
  `eval_dataset_name: ogbench/cube_single_expert`.
- `config/eval/cube.yaml` — see "Cube evaluator" above.

## Le-WM

`experiments/cube/scripts/run_cube_matrix.sh` sets Cube defaults
(`DATASET_NAME=cube`, `DATA_FILE=.../cube_single_expert.h5`,
`EVAL_CONFIG=cube`, `EVAL_DATASET_NAME=ogbench/cube_single_expert`,
`WORK_ROOT=experiments/cube/matrix`, `CACHE_DIR`, `CHECKPOINT` required,
`PYTHON`, `GPU_IDS`, `CPU_THREADS=4`) and `exec`s into the generic
`run_lewm_matrix.sh`, which implements the same protocol as PushT:
official pretrained baseline, Joint-Continue FP32 (3 epochs), Global-FT (50
epochs), Random-Voronoi K3 (50 epochs), K-means++ K3 (50 epochs), Spectral K3
(50 epochs), and the Auto-LAP gate, with train seeds `0,42,625`, partition
seeds `0,1,2`, eval seeds `0,1,2,3,4`, deployment partition seed `0`, and the
same spectral gate parameters as the task brief (K=3, diagnostic seeds
`0,1,2`, 20,000 landmarks, nominal kNN 30, perturbed kNN 27/33, retention
threshold 0.5, background gap count 10, MAD multiplier 3.0 — all hardcoded
in `run_cube_gate.sh` / passed through from `run_lewm_matrix.sh`'s own
`GATE_*` defaults, unchanged from PushT/Reacher).

`launch_cube_matrix_parallel.sh` is the detached, multi-GPU wrapper around
`run_lewm_matrix_parallel.sh` (one worker per listed GPU).

Both scripts are resumable: `run_lewm_matrix.sh`'s `prepare`/partition/train
functions all check for an existing `manifest.json`/cache before doing any
work.

**Blocked on:** no Le-WM Cube checkpoint found on this server (see
"Checkpoints"). Scripts are otherwise complete and preflight-clean (see
"Verification"); they will run as soon as `CHECKPOINT` is set to a real
Le-WM Cube object checkpoint.

## Sub-JEPA

`experiments/cube/subjepa/env.sh` holds shared paths: `DATASET`,
`CHECKPOINT` (defaults to the found Sub-JEPA checkpoint above), `TASK_SPEC`,
`CACHE_DIR`, `SMOKE_ROOT`, `FORMAL`, `MATRIX`, `CANON_SHORT`, `CANON_LONG`.
No PushT checkpoint, SHA, path, or paired starts appear anywhere in it.

Stages (all independently resumable — every step checks for an existing
manifest/cache/results file before doing work):

1. **smoke** — restricted end-to-end validation, writes only under
   `experiments/cube/subjepa/smoke/` (see "Verification"; already run).
2. **prepare** — full-scale latent cache (`experiments/cube/subjepa/formal/preparation/`).
3. **gate** — LAP automatic spectral gate (`fit_partition.py --method auto`),
   same parameters as Le-WM above. This *is* the method under test, not an
   audit step, and is unmodified from the Reacher/PushT gate call.
4. **partition / training** — Global-FT50, K-means++ K3-50, Spectral K3-50,
   Auto-LAP (branch selected by the gate's own manifest, not hardcoded).
5. **eval-short** (goal_offset 25) / **eval-long** (goal_offset 50) — paired
   across official baseline + every method, using the *same* 5 canonical
   starts per horizon (see "Paired starts" below); no method resamples
   independently.
6. **aggregate** — `matrix_summary_{short,long}.json` /
   `matrix_raw_{short,long}.csv` via the shared `aggregate_matrix.py`.

**Paired starts:** Reacher/PushT Sub-JEPA reuse an already-completed Le-WM
matrix's official eval results as the canonical short/long starts. There is
no completed Le-WM Cube matrix on this server (see "Checkpoints"), so
`experiments/cube/subjepa/matrix/scripts/setup_matrix.sh` instead
**bootstraps Cube's own** canonical starts once, via
`bootstrap_canonical_starts.sh`, which calls the exact same shared evaluator
every method's eval call already uses
(`experiments/tworoom/tworoom_success_rate_eval.py --mode baseline
--sample-eval-starts`) directly — no new audit, no duplicated Python logic.
If `CANON_SHORT`/`CANON_LONG` are later pointed at a completed Le-WM Cube
matrix, `setup_matrix.sh` will copy those in instead of bootstrapping. Cube
never reads, copies, or resamples from PushT's (or any other task's) starts
— verified (see "Verification").

## Sub-JEPA results

The full Sub-JEPA matrix completed on 2026-08-12. Both `eval_short/` and
`eval_long/` contain 110/110 result cells.

### Paired control success rate

Short uses `goal_offset=25`; long uses `goal_offset=50`. Both use five
evaluation seeds and 50 episodes per seed. Error bars are sample SD across
the three fine-tuning seeds after averaging partition and evaluation seeds.

| Method | Short success (%) | Long success (%) |
|---|---:|---:|
| Official baseline | 69.20 | 51.20 |
| Global-FT50 | 64.40 ± 0.69 | 49.07 ± 1.97 |
| K-means++ K3-50 | 66.80 ± 1.04 | 46.98 ± 1.37 |
| Spectral K3-50 | 67.20 ± 0.58 | 48.00 ± 0.83 |
| Auto-LAP | 64.40 ± 0.69 | 49.07 ± 1.97 |

Sources:
`subjepa/matrix/manifests/matrix_summary_short.json` and
`subjepa/matrix/manifests/matrix_summary_long.json`.

The automatic gate selected `global`, so Auto-LAP is expected to equal
Global-FT50. `deployment_seed=0` is the deployment partition seed, not a
predictor training seed; Auto-LAP still evaluates train seeds `0,42,625`.

### Held-out one-step latent prediction MSE

This uses the matrix training protocol's transition-level 90/10 split
(`split_seed=3072`): 1,674,000 training transitions and 186,000 held-out
transitions with zero transition overlap. It is not episode-disjoint.
Official and all three Auto-LAP/global fine-tuned predictors are evaluated on
the same Sub-JEPA-encoded held-out cache (`history_size=3`, `num_preds=1`,
`frameskip=5`).

| Predictor | Held-out one-step MSE |
|---|---:|
| Official baseline | 0.00145984 |
| Auto-LAP train seed 0 | 0.00057817 |
| Auto-LAP train seed 42 | 0.00057285 |
| Auto-LAP train seed 625 | 0.00057323 |
| Auto-LAP mean ± sample SD | 0.00057475 ± 0.00000297 |

Auto-LAP reduces mean held-out one-step MSE by **60.63%** relative to the
Official baseline. Results are stored under
`subjepa/matrix/heldout/official_vs_auto_lap_train{0,42,625}_one_step.json`;
the shared held-out latent cache is
`subjepa/matrix/heldout/cube_subjepa_heldout_eval_latent_cache.npz`.

The one-step MSE improvement does not by itself imply better MPC success:
the latent prediction objective and downstream planning/control objective
are distinct, and the control results above should be reported separately.

## Excluded audits (explicitly not added/copied for this task)

- Frozen audit, one-step MSE audit, cache-equivalence audit
  (`validate_jepa_smoke.py`'s `frozen-audit`/`cache-equivalence`/
  `route-equivalence` phases) — the smoke script calls the generic driver's
  individual phases (`prepare`, `partition_global`, `partition_regions`,
  direct `train_predictors.py`, `eval-short`) directly instead of the
  built-in `PHASE=smoke` composite, specifically to avoid these.
- Repository audit / audit passport / material passport / verification-status
  audits (`formal_cache_audit.py`, `formal_post_gate_audit.py`,
  `*_formal_lib.py` smoke-untouched checks) — not used; Cube's
  `run_cube_gate.sh` has no `passport` phase.
- The `audit)` case and its `pre_execution_lock`/`matrix_one_step_mse.py`
  dependencies from PushT's `run_full_matrix.sh` — dropped entirely from
  Cube's `run_full_matrix.sh`; `all-post-train` only runs `eval-short` →
  `eval-long` → `aggregate`.
- Paired bootstrap (`matrix_paired_bootstrap.py`) — PushT's formal protocol
  runs this as a statistics step, but it is **not** included for Cube per
  explicit instruction; Cube's `run_full_matrix.sh` has no `bootstrap` case.
- The LAP automatic spectral gate (`fit_partition.py --method auto`) is
  **not** an audit — it is the method itself — and is preserved unmodified.

## Verification performed

The checks below describe the initial setup validation. The full 50-epoch
Sub-JEPA matrix was subsequently run; its results are reported above.

1. `bash -n` on all 16 new shell scripts under `experiments/cube/` — all
   passed.
2. `python -m json.tool configs/experiments/tasks/cube.json` and
   `yaml.safe_load(config/eval/cube.yaml)` — both parse.
3. `resolve_jepa_matrix_config.py --task-spec configs/experiments/tasks/cube.json ... --dry-run --emit-shell` — resolves `DATASET_NAME=cube`,
   `EVAL_DATASET_NAME=ogbench/cube_single_expert`, all task-spec numeric
   fields, with no errors.
4. `h5py` read-only schema check on `cube_single_expert.h5` — see "Data".
5. `scripts/probe_jepa_checkpoint.py` on the Sub-JEPA Cube checkpoint —
   `status: PASSED` (encoder/predictor/planning outputs all finite; no
   NaN/Inf).
6. `gym.make("swm/OGBCube-v0", env_type="single", ob_type="states")` — env
   creates, `obs.shape == (28,)`, `set_state`/`set_target_pos` present.
7. Backend hardcoding scan (`task_spec.py`, `prepare_lewm_cache.py`,
   `fit_partition.py`, `train_predictors.py`,
   `resolve_jepa_matrix_config.py`, `backend_registry.py`,
   `tworoom_success_rate_eval.py`, `gauge_drift.py`) — no task-name
   allowlists reject `"cube"`; only additive change needed was a generic
   `--num-eval` CLI override on `resolve_jepa_matrix_config.py` (used by
   smoke; verified no regression — PushT's resolver output for `NUM_EVAL`
   is unchanged at `50` when `--num-eval`/`NUM_EVAL` are not passed).
8. **Restricted smoke test actually executed** (GPU, not a dry run — see
   below), single GPU, single train/partition/eval seed (`0`), 300 max train
   starts, 2 training epochs, 3 eval episodes, `--min-region-samples 32`
   (K=3 spectral split of a 300-start cache otherwise falls under the
   default 256-sample minimum — smoke-only override, driver already exposes
   this flag). Ran the full chain: prepare → partition (global + spectral)
   → train (global + spectral, direct `train_predictors.py` calls) →
   bootstrap official starts → eval-short (official + global + spectral) →
   aggregate. All phases completed; output success rates (tiny-scale, not
   meaningful): Official baseline 66.7%, Global-FT 66.7%, Spectral 33.3%.
   Output under `experiments/cube/subjepa/smoke/`; `experiments/cube/subjepa/formal/`
   and `experiments/cube/subjepa/matrix/` (formal/real-scale dirs) were never
   touched.
9. `bash experiments/cube/scripts/run_cube_matrix.sh` with `CHECKPOINT`
   unset — fails loudly with the expected "set CHECKPOINT ..." message
   (Le-WM preflight; confirms the required-env-var gate works and that no
   silent fallback to a wrong-family checkpoint occurs).
10. Compatibility scan —
    `rg -n "pusht|PushT|pusht_expert_train|state|goal_state" experiments/cube configs/experiments/tasks/cube.json config/eval/cube.yaml`
    — only comments citing PushT as a source, `ob_type: states`/
    `set_state` (legitimate Cube-native config fields, not PushT dataset
    keys), and smoke run artifacts (binary checkpoints, unrelated log
    warnings).
11. Final scan —
    `rg -n "experiments/pusht|config-name pusht|dataset-name pusht" experiments/cube`
    — only one hit, a comment in `run_cube_matrix.sh` citing
    `experiments/pusht/README.md` as the source of the wrapper pattern. No
    runtime PushT hardcoding.

## GPU policy used during validation

GPU 4 was used briefly for the checkpoint probe and part of the smoke run,
then **stopped using GPU 4 entirely per operator instruction**; all
subsequent Cube validation (region training fix, eval-short, aggregate) ran
on GPU 6. Do not use GPU 4 for Cube commands going forward.

## Copy-paste commands

### Smoke (already run; safe to re-run — resumable)

```bash
bash experiments/cube/subjepa/scripts/run_cube_smoke.sh
# outputs: experiments/cube/subjepa/smoke/
```

### Le-WM full matrix (blocked: needs a real Le-WM Cube checkpoint)

```bash
CHECKPOINT=/path/to/lewm_object.ckpt \
GPU_IDS=5,6 GPU_ID=5 \
bash experiments/cube/scripts/run_cube_matrix.sh
# or, detached / multi-GPU:
CHECKPOINT=/path/to/lewm_object.ckpt GPU_IDS=5,6 \
bash experiments/cube/scripts/launch_cube_matrix_parallel.sh
```

### Sub-JEPA gate / training / eval (completed; reproduction commands)

```bash
# 1) formal prepare + LAP auto gate (full-scale latent cache)
GPU_ID=6 bash experiments/cube/subjepa/scripts/run_cube_subjepa.sh formal-all
# or detached:
GPU_ID=6 bash experiments/cube/subjepa/scripts/launch_formal_detached.sh all

# 2) matrix partition + training (Global-FT50 / K-means++ K3-50 / Spectral K3-50 / Auto-LAP)
GPU_IDS=5,6 GPU_ID=5 bash experiments/cube/subjepa/scripts/run_cube_subjepa.sh training
# or detached:
GPU_IDS=5,6 bash experiments/cube/subjepa/matrix/scripts/launch_matrix_detached.sh

# 3) paired short + long eval, then aggregate
GPU_IDS=5,6 bash experiments/cube/subjepa/scripts/run_cube_subjepa.sh all-post-train
# or detached, eval only:
GPU_IDS=5,6 bash experiments/cube/subjepa/matrix/scripts/launch_eval_detached.sh
```

`GPU_ID`/`GPU_IDS` default to `0` if unset — always pass them explicitly on
this shared server, and do not use GPU 4.

## Result directories

- Le-WM: `experiments/cube/matrix/` (preparation, partitions, training, eval,
  manifests) — created on first run of `run_cube_matrix.sh`.
- Sub-JEPA formal (gate): `experiments/cube/subjepa/formal/`
  (`preparation/`, `gate/partition/`, `logs/`, `manifests/`).
- Sub-JEPA matrix: `experiments/cube/subjepa/matrix/` (`preparation` symlink,
  `partitions/`, `training/`, `eval/` + `eval_short/`/`eval_long/`
  snapshots, `auto/`, `paired_starts/`, `manifests/matrix_summary_{short,long}.json`).
- Sub-JEPA smoke: `experiments/cube/subjepa/smoke/` (already populated by
  verification above).

## Resuming safely

Every phase in every script checks for an existing manifest / cache /
`results.json` before doing work and skips if already present — re-running
any command above (formal, matrix, smoke) after an interruption (OOM, kill,
network) will pick up where it left off, never overwrite formal/matrix
outputs from a smoke run or vice versa, and never silently switch to a
different checkpoint/dataset than the one already recorded in
`manifests/resolved_config.json`.
