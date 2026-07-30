#!/usr/bin/env bash
# Step 3: long-range success rate for latent kmeanspp (outer seeds 0,1,2 × eval seeds 0-4).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== latent kmeanspp longrange all outer started at $(date) ===="

for OUTER_SEED in 0 1 2; do
  echo "==== outer=${OUTER_SEED} started at $(date) ===="
  OUTER_SEED="${OUTER_SEED}" bash "${SCRIPT_DIR}/../internal/run_success_rate_5seed_latent_kmeanspp_longrange.sh"
  echo "==== outer=${OUTER_SEED} finished at $(date) ===="
done

echo "==== latent kmeanspp longrange all outer DONE at $(date) ===="
