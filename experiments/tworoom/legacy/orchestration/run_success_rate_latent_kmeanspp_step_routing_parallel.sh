#!/usr/bin/env bash
# Run outer seeds 0/1/2 in parallel on three GPUs after the real-checkpoint smoke test passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
ROOT="experiments/tworoom"
SCRIPT="${ROOT}/scripts/run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh"
GPUS=("${GPU0:-0}" "${GPU1:-1}" "${GPU2:-2}")
PIDS=()

for outer_seed in 0 1 2; do
  gpu="${GPUS[${outer_seed}]}"
  wrapper_log="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_R50_outer${outer_seed}_step_routing_wrapper.log"
  echo "Launching outer=${outer_seed} on physical GPU ${gpu} at $(date)"
  CUDA_VISIBLE_DEVICES="${gpu}" OUTER_SEED="${outer_seed}" \
    bash "${SCRIPT}" > "${wrapper_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for i in 0 1 2; do
  if ! wait "${PIDS[${i}]}"; then
    echo "outer=${i} failed; see its wrapper/master logs" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "All three latent K-means++ step-routing outer seeds finished at $(date)"
