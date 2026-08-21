#!/usr/bin/env bash
# PushT Sub-JEPA matrix setup — pre-lock, cache link, forced-Spectral materialization.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

K3_MATRIX="${K3_MATRIX:-$MATRIX}"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/control_matrix/scripts/subjepa_k_variant.sh"

PUSH_MATRIX_SCRIPTS="$REPO_ROOT/experiments/pusht/subjepa/matrix/scripts"
FORMAL_LIB="$REPO_ROOT/experiments/pusht/subjepa/formal/scripts/pusht_formal_lib.py"

mkdir -p "$MATRIX/manifests" "$MATRIX/logs" "$MATRIX/partitions" "$MATRIX/training"

"$PYTHON" "$FORMAL_LIB" \
  --phase verify-smoke \
  --smoke-root "$SMOKE_ROOT" \
  --formal-root "$FORMAL" \
  --expected-smoke-cache-sha256 "$SMOKE_CACHE_SHA256" >/dev/null

# --- read-only cache link (formal only; never smoke preparation/) ---
PREP="$MATRIX/preparation"
rm -rf "$PREP"
ln -sfn "$(realpath "$FORMAL/preparation")" "$PREP"

# --- forced-Spectral control partitions (independent of Auto-LAP branch) ---
if [[ "$SKIP_SPECTRAL_MATERIALIZE" != "1" ]]; then
  "$PYTHON" "$TWOROOM_MATRIX_SCRIPTS/materialize_spectral_partitions.py" \
    --formal-root "$FORMAL/partitions/spectral" \
    --matrix-root "$MATRIX/partitions/spectral" \
    --latent-cache "$FORMAL/preparation/embedding_cache.npz" \
    --seeds 0,1,2
fi

# --- canonical paired evaluation starts ---
if ! reuse_k3_paired_starts "$K3_MATRIX" "$MATRIX"; then
  PAIR_SHORT="$MATRIX/paired_starts/canon_short/eval/official"
  PAIR_LONG="$MATRIX/paired_starts/canon_long/eval/official"
  mkdir -p "$PAIR_SHORT" "$PAIR_LONG"
  for seed in 0 1 2 3 4; do
    short_src="$CANON_SHORT/eval${seed}/results.json"
    long_src="$CANON_LONG/eval${seed}/results.json"
    [[ -f "$short_src" ]] || { echo "missing canonical short eval seed $seed: $short_src" >&2; exit 1; }
    [[ -f "$long_src" ]] || { echo "missing canonical long eval seed $seed: $long_src" >&2; exit 1; }
    mkdir -p "$PAIR_SHORT/eval${seed}" "$PAIR_LONG/eval${seed}"
    cp "$short_src" "$PAIR_SHORT/eval${seed}/results.json"
    cp "$long_src" "$PAIR_LONG/eval${seed}/results.json"
  done
fi

# --- pre-lock (gate branch read from passport; no CLI override) ---
LOCK="$MATRIX/manifests/pre_execution_lock.json"
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  "$PYTHON" "$PUSH_MATRIX_SCRIPTS/matrix_prelock.py" \
    --repo-root "$REPO_ROOT" \
    --formal-root "$FORMAL" \
    --matrix-root "$MATRIX" \
    --smoke-root "$SMOKE_ROOT" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --task-spec "$TASK_SPEC" \
    --canon-short "$CANON_SHORT" \
    --canon-long "$CANON_LONG" \
    --expected-smoke-cache-sha256 "$SMOKE_CACHE_SHA256" \
    --out "$LOCK"

  read -r SELECTED_BRANCH AUTO_PARTITION_SRC <<EOF
$("$PYTHON" -c "
import json
lock = json.load(open('$LOCK'))
print(lock['gate_selected_branch'], lock['auto_partition_symlink_source'])
")
EOF
  echo "[setup] gate_selected_branch=$SELECTED_BRANCH"
else
  write_k_variant_lock "$MATRIX"
  SELECTED_BRANCH=""
  AUTO_PARTITION_SRC=""
fi

# --- global partition on full cache (once; independent of gate branch) ---
if ! reuse_k3_global_artifacts "$K3_MATRIX" "$MATRIX"; then
  if [[ ! -f "$MATRIX/partitions/global/seed0/manifest.json" ]]; then
    echo "[setup] fitting global partition on full cache"
    CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$PYTHON" experiments/control_matrix/fit_partition.py \
      --method global \
      --dataset-name pusht \
      --latent-cache "$FORMAL/preparation/embedding_cache.npz" \
      --frameskip 5 \
      --cpu-threads 4 \
      --out-dir "$MATRIX/partitions/global/seed0"
  fi
fi

# --- Auto-LAP deployment partition symlink (from gate output, not re-fit) ---
if [[ "$INCLUDE_AUTO_LAP" == "1" ]]; then
  AUTO="$MATRIX/auto/partition"
  rm -rf "$AUTO"
  mkdir -p "$(dirname "$AUTO")"
  ln -sfn "$(realpath "$AUTO_PARTITION_SRC")" "$AUTO"
  echo "[setup] auto_lap branch=$SELECTED_BRANCH auto/partition -> $AUTO_PARTITION_SRC"
fi

echo "[setup] matrix ready under $MATRIX"
echo "[setup] pre_execution_lock=$LOCK"
