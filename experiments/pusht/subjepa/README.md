# Sub-JEPA PushT Smoke

**Verification status: `VERIFIED` (smoke, 2026-08-03)**  
See `manifests/verification_status.json`.

Cache-equivalence uses the same production unique-frame / `exact_batch_shapes` replay
as TwoRoom. Preserved cache SHA-256:
`3276baa78353564f6baccbdea423137ee8472901841da6d1840d9b6353c4fc67`.

Reduced smoke eval proves pipeline wiring only; formal method conclusions require
full paired short/long evaluation.

## Reproduction

```bash
export PYTHON="${PYTHON:-python}"

bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec configs/experiments/tasks/pusht.json \
  --dataset "${DATASET:?path to pusht_expert_train.h5}" \
  --checkpoint "${CHECKPOINT:?path to subjepa_object.ckpt}" \
  --eval-config-name pusht \
  --work-root experiments/pusht/subjepa \
  --cache-dir "${CACHE_DIR:-$HOME/.stableworldmodel}/subjepa/pusht" \
  --max-train-starts 4096 \
  --phase smoke
```

See `experiments/tworoom/subjepa/README.md` for the TwoRoom reference and validator
false-failure notes.
