#!/usr/bin/env bash
# Run per-imagined-step latent routing for outer seeds 0, 1 and 2 sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== latent kmeanspp step routing all outer started at $(date) ===="

for OUTER_SEED in 0 1 2; do
  echo "==== outer=${OUTER_SEED} started at $(date) ===="
  OUTER_SEED="${OUTER_SEED}" bash "${SCRIPT_DIR}/run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh"
  echo "==== outer=${OUTER_SEED} finished at $(date) ===="
done

echo "==== latent kmeanspp step routing all outer DONE at $(date) ===="
