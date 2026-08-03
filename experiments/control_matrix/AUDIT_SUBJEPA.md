# Sub-JEPA Integration Audit

Date: 2026-08-03  
Baseline commit: `0babf1687afaa1c6667ca6d877cc85e4509f1131`  
Pre-change hash file: `/tmp/lap_pre_subjepa_hashes.txt` (2255 files)

## 1. Generic JEPA logic vs LeWM naming

| Component | Generic? | Notes |
|-----------|----------|-------|
| `lap/interfaces/{cache,encoding,training,world_model}.py` | Yes | Backend-neutral contracts |
| `lap/encoding/fast.py` | Yes | Accelerated cache encoder |
| `lap/partition/*` | Yes | Spectral gate, landmark partition |
| `backends/lewm/*` | Shared runtime | Implements `jepa.JEPA + ARPredictor` for both LeWM and Sub-JEPA |
| `prepare_lewm_cache.py` | Generic path | Name is historical; now accepts `--model-family` |
| `train_predictors.py` | Generic path | Predictor-only fine-tuning |
| `run_lewm_matrix.sh` | Unchanged | Still the canonical LeWM entry point |
| `run_jepa_matrix.sh` | New generic driver | Adds model-family, task-spec resolution, probe/smoke phases |

## 2. LeWM-specific coupling

- Checkpoint loader assumes serialized `torch.nn.Module` JEPA object (`encoder`, `predictor`, `action_encoder`, `projector`, `pred_proj`).
- Action normalization uses HDF5 `action` blocks and frameskip from task spec.
- Evaluator remains `experiments/tworoom/tworoom_success_rate_eval.py` (Hydra multi-task).

## 3. New additive surfaces

- `experiments/control_matrix/backend_registry.py`
- `experiments/control_matrix/task_spec.py`
- `experiments/control_matrix/resolve_jepa_matrix_config.py`
- `configs/experiments/tasks/{tworoom,pusht}.json`
- `scripts/probe_jepa_checkpoint.py`
- `experiments/control_matrix/scripts/run_jepa_matrix.sh`
- `experiments/control_matrix/scripts/run_subjepa_matrix.sh`

## 4. CLI parameters (`run_jepa_matrix.sh`)

`--model-family`, `--task-spec`, `--task-name`, `--dataset`, `--checkpoint`,
`--dataset-config`, `--encoder-config`, `--eval-config-name`, `--eval-config`,
`--work-root`, `--cache-dir`, `--paired-start-root`, `--paired-start-root-short`,
`--paired-start-root-long`, `--phase`, `--max-train-starts`, `--dry-run`,
plus training/eval overrides (`--train-seeds`, `--partition-seeds`, `--eval-seeds`,
`--methods`, `--skip-joint`, `--python`, `--cpu-threads`, `--gpu-id`).

## 5. Legacy environment variables (unchanged for LeWM)

`DATASET_NAME`, `DATA_FILE`, `CHECKPOINT`, `EVAL_CONFIG`, `EVAL_DATASET_NAME`,
`CACHE_DIR`, `WORK_ROOT`, `PHASE`, `GPU_ID`, `TRAIN_SEEDS`, `PARTITION_SEEDS`,
`EVAL_SEEDS`, `METHODS`, `CPU_THREADS`, `SKIP_JOINT`, `PREPARE_MAX_STARTS`,
`GOAL_OFFSET`, `EVAL_BUDGET`, gate variables (`GATE_*`).

## 6. Parameter precedence

1. Explicit CLI
2. Legacy environment variables
3. Task spec JSON
4. Historical defaults from `run_lewm_matrix.sh`

Conflicts on task name, dataset name, eval config, frameskip, horizons fail fast.

## 7. Cache schema

`embedding_cache.npz`: `emb`, `act_emb`, `region_starts`  
Manifest binds checkpoint SHA-256, dataset SHA-256, frameskip, model_family.

## 8. Trainable allowlist

`predictor`, `pred_proj` only (`backends/lewm/finetuning.py`).

## 9. Evaluator checkpoint loading

Official: base object checkpoint.  
Global-FT: `P_train_cluster0_object.ckpt`.  
Regional/LAP: base checkpoint + `--lap-run-dir` regional predictors via routing wrappers.

## 11. TwoRoom Sub-JEPA smoke verification (2026-08-03)

**Status: `VERIFIED`**

- Smoke manifest: `experiments/tworoom/subjepa/manifests/verification_status.json`
- Original cache retained (not rebuilt); SHA-256 `6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7`
- Initial cache-equivalence failure was a **validator false positive**: direct per-window
  `encode_frames` (batch=4) was compared against production unique-frame /
  `exact_batch_shapes` cache (~0.013 max diff). Production writer and NPZ reload
  agreed to floating-point noise; recomputation via `recompute_latent_windows()` is exact.
- Regression coverage: `tests/test_cache_equivalence_regression.py`
- Reduced eval at 100% for official/global/spectral proves wiring only, not method quality.

## 12. Next integration steps

1. PushT Sub-JEPA smoke (task spec / paths only; no generic driver changes)
2. Full TwoRoom cache + formal gate + method matrix with paired evaluation
