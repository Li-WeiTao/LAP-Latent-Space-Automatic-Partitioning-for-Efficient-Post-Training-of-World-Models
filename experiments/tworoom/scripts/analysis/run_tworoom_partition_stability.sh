#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

bash experiments/tworoom/scripts/analysis/run_latent_preprocess_stability.sh
bash experiments/tworoom/scripts/analysis/run_latent_preprocess_convergence.sh
bash experiments/tworoom/scripts/analysis/run_latent_kmeanspp_multirestart.sh
bash experiments/tworoom/scripts/internal/run_random_voronoi_k3.sh
GPU="${GPU:?set GPU}" bash experiments/tworoom/scripts/analysis/run_latent_landmark_spectral.sh
