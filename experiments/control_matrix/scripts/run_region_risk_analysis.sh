#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"
PYTHON="${PYTHON:-/data/sicong/weitao/le-wm/.venv/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES="$GPU"

COMMON=(
  --train-seeds 0,42,625
  --horizons 1,5,10
  --bootstrap-reps 50000
  --bootstrap-seed 20260801
  --batch-size 256
  --device cuda
  --allow-in-cache
)

echo "[region-risk] PushT eval cache + analysis"
$PYTHON experiments/control_matrix/evaluate_region_conditional_risk.py \
  --task pusht \
  --data-file /data/sicong/weitao/datasets/lewm/pusht_expert_train.h5 \
  --train-latent-cache experiments/pusht/matrix/pusht_lewm_train_latent_cache.npz \
  --eval-latent-cache experiments/control_matrix/assets/region_risk/pusht/pusht_eval_latent_cache.npz \
  --partition-dir experiments/pusht/matrix/partitions/spectral/seed0 \
  --regional-runs experiments/pusht/matrix/training/spectral \
  --global-runs experiments/pusht/matrix/training/global \
  --pretrained-model /data/sicong/weitao/.stable_worldmodel/pusht/lewm_object.ckpt \
  --build-eval-cache \
  "${COMMON[@]}" \
  --out-dir experiments/control_matrix/assets/region_risk/pusht

echo "[region-risk] TwoRoom eval cache + analysis"
$PYTHON experiments/control_matrix/evaluate_region_conditional_risk.py \
  --task tworoom \
  --data-file /data/sicong/weitao/datasets/lewm/tworoom.h5 \
  --train-latent-cache /data/sicong/weitao/le-wm/experiments/real_gauge_drift/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed0/P_train_global_merged_embeddings.npz \
  --eval-latent-cache experiments/control_matrix/assets/region_risk/tworoom/tworoom_eval_latent_cache.npz \
  --partition-dir experiments/tworoom/results/auto_gate_complete_k3/auto/partition/partition \
  --regional-runs /data/sicong/weitao/le-wm/experiments/real_gauge_drift/results \
  --global-runs /data/sicong/weitao/le-wm/experiments/real_gauge_drift/results \
  --pretrained-model /data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt \
  --build-eval-cache \
  "${COMMON[@]}" \
  --out-dir experiments/control_matrix/assets/region_risk/tworoom

echo "[region-risk] complete"
