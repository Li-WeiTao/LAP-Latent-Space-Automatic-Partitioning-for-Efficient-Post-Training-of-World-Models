# Sub-JEPA PushT Smoke

**Verification status: `VERIFIED` (smoke, sicong 2026-08-06)**  
See `manifests/verification_status.json`.

Cache-equivalence uses the same production unique-frame / `exact_batch_shapes` replay
as TwoRoom. Preserved cache SHA-256:
`3276baa78353564f6baccbdea423137ee8472901841da6d1840d9b6353c4fc67`.

Reduced smoke eval proves pipeline wiring only; formal method conclusions require
full paired short/long evaluation.

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

# Stage 2: 50-epoch matrix training
bash experiments/pusht/subjepa/matrix/scripts/launch_matrix_detached.sh

# Stage 3: paired short/long eval + audit + aggregate + bootstrap
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
