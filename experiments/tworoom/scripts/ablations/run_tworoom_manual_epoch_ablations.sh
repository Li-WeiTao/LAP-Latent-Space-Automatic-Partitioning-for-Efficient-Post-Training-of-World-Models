#!/usr/bin/env bash
# Historical manual-partition epoch ablations retained as an explicit profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

bash experiments/tworoom/scripts/ablations/run_geometry_train_region_predictors.sh
bash experiments/tworoom/scripts/ablations/run_geometry_train_region_predictors_50ep_best.sh
bash experiments/tworoom/scripts/ablations/run_geometry_train_doorway_80ep_best.sh
bash experiments/tworoom/scripts/ablations/run_success_rate_5seed_30ep_rooms3_priority5.sh
bash experiments/tworoom/scripts/ablations/run_success_rate_5seed_50ep_rooms3_priority5.sh
bash experiments/tworoom/scripts/ablations/run_success_rate_5seed_exp5_doorway80.sh
