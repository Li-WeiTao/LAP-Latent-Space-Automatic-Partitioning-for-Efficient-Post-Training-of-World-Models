#!/usr/bin/env bash
# Global-FT strong baseline for latent main experiment: 50ep on full train split.
# Same recipe as per-cluster FT (lr/batch/seed/select_best); compute-matched to cluster 50ep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
EPOCHS=50
SEED=42
OUT_DIR="experiments/tworoom/results/tworoom_geometry_train_global_ft_50ep"

export GPU EPOCHS SEED OUT_DIR
exec bash "${SCRIPT_DIR}/run_geometry_train_global_ft.sh"
