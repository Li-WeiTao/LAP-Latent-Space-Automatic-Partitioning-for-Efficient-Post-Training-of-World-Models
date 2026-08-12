#!/usr/bin/env bash
# Reacher Sub-JEPA matrix setup — directory scaffolding, read-only cache
# link, canonical paired-start reuse, and Auto-LAP branch symlink.
#
# Unlike experiments/pusht/subjepa/matrix/scripts/setup_matrix.sh, this does
# NOT depend on any smoke-verification, material-passport, or TwoRoom
# audit/materialization scripts. The Auto-LAP branch is read directly from
# this task's own gate manifest (formal/gate/partition/manifest.json), never
# from LeWM Reacher's gate decision and never hardcoded.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/reacher/subjepa/env.sh"

WORK_ROOT="$MATRIX"
GATE_MANIFEST="$FORMAL/gate/partition/manifest.json"

mkdir -p "$WORK_ROOT/manifests" "$WORK_ROOT/logs" "$WORK_ROOT/partitions" "$WORK_ROOT/training"

# --- read-only cache link (matrix reuses the formal full cache; never writes
#     into formal/preparation) ---
PREP="$WORK_ROOT/preparation"
if [[ -d "$FORMAL/preparation" ]]; then
  rm -rf "$PREP"
  ln -sfn "$(realpath "$FORMAL/preparation")" "$PREP"
else
  echo "[setup] warning: formal preparation not found yet: $FORMAL/preparation (run formal prepare first)" >&2
fi

# --- canonical paired evaluation starts (reused from Reacher LeWM matrix;
#     never resampled here) ---
PAIR_SHORT="$WORK_ROOT/paired_starts/canon_short/eval/official"
PAIR_LONG="$WORK_ROOT/paired_starts/canon_long/eval/official"
mkdir -p "$PAIR_SHORT" "$PAIR_LONG"
for seed in 0 1 2 3 4; do
  short_src="$CANON_SHORT/eval${seed}/results.json"
  long_src="$CANON_LONG/eval${seed}/results.json"
  [[ -f "$short_src" ]] || { echo "missing canonical short eval seed $seed: $short_src" >&2; exit 1; }
  [[ -f "$long_src" ]] || { echo "missing canonical long eval seed $seed: $long_src" >&2; exit 1; }
  mkdir -p "$PAIR_SHORT/eval${seed}" "$PAIR_LONG/eval${seed}"
  [[ -f "$PAIR_SHORT/eval${seed}/results.json" ]] || cp "$short_src" "$PAIR_SHORT/eval${seed}/results.json"
  [[ -f "$PAIR_LONG/eval${seed}/results.json" ]] || cp "$long_src" "$PAIR_LONG/eval${seed}/results.json"
done

# --- global partition on full cache (once; independent of gate branch) ---
if [[ ! -f "$WORK_ROOT/partitions/global/seed0/manifest.json" && -f "$FORMAL/preparation/embedding_cache.npz" ]]; then
  echo "[setup] fitting global partition on full cache"
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$PYTHON" experiments/control_matrix/fit_partition.py \
    --method global \
    --dataset-name reacher \
    --latent-cache "$FORMAL/preparation/embedding_cache.npz" \
    --frameskip 5 \
    --gpu-id 0 \
    --cpu-threads "${CPU_THREADS:-4}" \
    --out-dir "$WORK_ROOT/partitions/global/seed0"
fi

# --- Auto-LAP branch: read directly from this task's own gate manifest ---
if [[ -f "$GATE_MANIFEST" ]]; then
  SELECTED_BRANCH="$("$PYTHON" -c "import json; print(json.load(open('$GATE_MANIFEST'))['selected_method'])")"
  echo "[setup] gate_selected_branch=$SELECTED_BRANCH (from $GATE_MANIFEST)"

  AUTO="$WORK_ROOT/auto/partition"
  mkdir -p "$(dirname "$AUTO")"
  rm -rf "$AUTO"
  ln -sfn "$(realpath "$FORMAL/gate/partition")" "$AUTO"
  echo "[setup] auto/partition -> $FORMAL/gate/partition"
else
  echo "[setup] warning: gate manifest not found yet: $GATE_MANIFEST (run formal gate first)" >&2
fi

echo "[setup] matrix ready under $WORK_ROOT"
