# Sub-JEPA Experiments (TwoRoom)

This directory holds Sub-JEPA outputs isolated from historical LeWM results.

**Verification status: `VERIFIED` (smoke, 2026-08-03)**  
See `manifests/verification_status.json`.

The initial cache-equivalence failure (`max_abs_diff ≈ 0.013`) was a **validator false
positive**: the old smoke check used direct per-window `encode_frames`, while production
cache uses unique-frame encoding with `exact_batch_shapes`. The preserved cache SHA-256
is `6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7` (not rebuilt).

Reduced smoke eval at 100% for official/global/spectral proves pipeline wiring only,
not method comparison. Use full paired short/long evaluation for formal conclusions.

## Reproduction (smoke)

```bash
export PYTHON="${PYTHON:-python}"
export TASK_SPEC="configs/experiments/tasks/tworoom.json"
export DATASET="${DATASET:?path to tworoom.h5}"
export CHECKPOINT="${CHECKPOINT:?path to subjepa_object.ckpt}"

bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec "$TASK_SPEC" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --eval-config-name tworoom \
  --work-root experiments/tworoom/subjepa \
  --cache-dir "${CACHE_DIR:-$HOME/.stable_worldmodel}/subjepa/tworoom" \
  --max-train-starts 4096 \
  --phase smoke
```

Do not write into `experiments/tworoom/matrix` or other LeWM result directories.

## Full 50-epoch matrix

Formal gate, protocol parity, detached launch, training, paired short/long eval, audit,
and bootstrap are documented in **`matrix/README.md`**.

Quick start (detached, 8 GPUs):

```bash
bash experiments/tworoom/subjepa/matrix/scripts/launch_matrix_detached.sh
```
