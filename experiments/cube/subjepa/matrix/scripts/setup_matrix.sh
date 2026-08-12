#!/usr/bin/env bash
# OGBench Cube Sub-JEPA matrix setup — directory scaffolding, read-only cache
# link, canonical paired-start reuse (best-effort), and Auto-LAP branch
# symlink.
#
# Unlike Reacher/PushT, there is no completed Le-WM Cube matrix on this
# server yet (see experiments/cube/README.md "Known blockers"), so
# CANON_SHORT/CANON_LONG (experiments/cube/subjepa/env.sh) point at this same
# Sub-JEPA matrix's own eval/official output by default: the first eval-short
# / eval-long run self-generates its official paired starts once (fixed by
# eval seed) and every method within that horizon and this work root reuses
# them — this is the paired-start invariant, no different than reusing an
# externally-computed Le-WM matrix, just self-contained. If CANON_SHORT /
# CANON_LONG are later pointed at a completed Le-WM Cube matrix (to align
# Sub-JEPA and Le-WM on identical starts), this script will copy them in;
# until then it skips that copy step instead of failing, and never resamples
# or borrows another task's (e.g. PushT's) starts.
#
# No smoke-verification, material-passport, or TwoRoom audit/materialization
# scripts. The Auto-LAP branch is read directly from this task's own gate
# manifest (formal/gate/partition/manifest.json).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

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

# --- canonical paired evaluation starts: best-effort copy from CANON_SHORT /
#     CANON_LONG if a complete 5-seed set already exists there (e.g. a later
#     completed Le-WM Cube matrix). If none exists yet, skip — training does
#     not need paired starts; eval-short/eval-long generate or reuse official
#     starts under $WORK_ROOT/eval/official/ at eval time (see run_full_matrix.sh).
PAIR_SHORT_ROOT="$WORK_ROOT/paired_starts/canon_short"
PAIR_LONG_ROOT="$WORK_ROOT/paired_starts/canon_long"
PAIR_SHORT="$PAIR_SHORT_ROOT/eval/official"
PAIR_LONG="$PAIR_LONG_ROOT/eval/official"

canon_source_complete() {
  local root=$1
  [[ -n "$root" ]] || return 1
  local seed
  for seed in 0 1 2 3 4; do
    [[ -f "$root/eval${seed}/results.json" ]] || return 1
  done
  return 0
}

if canon_source_complete "$CANON_SHORT" && canon_source_complete "$CANON_LONG"; then
  mkdir -p "$PAIR_SHORT" "$PAIR_LONG"
  for seed in 0 1 2 3 4; do
    mkdir -p "$PAIR_SHORT/eval${seed}" "$PAIR_LONG/eval${seed}"
    [[ -f "$PAIR_SHORT/eval${seed}/results.json" ]] || cp "$CANON_SHORT/eval${seed}/results.json" "$PAIR_SHORT/eval${seed}/results.json"
    [[ -f "$PAIR_LONG/eval${seed}/results.json" ]] || cp "$CANON_LONG/eval${seed}/results.json" "$PAIR_LONG/eval${seed}/results.json"
  done
  echo "[setup] canonical paired starts copied from CANON_SHORT=$CANON_SHORT CANON_LONG=$CANON_LONG"
else
  echo "[setup] no complete external canonical paired-start set found (CANON_SHORT=$CANON_SHORT, CANON_LONG=$CANON_LONG)"
  echo "[setup] skipping paired-start setup (not required for partition/training; eval-short/eval-long handle official starts separately)"
fi

# --- global partition on full cache (once; independent of gate branch) ---
if [[ ! -f "$WORK_ROOT/partitions/global/seed0/manifest.json" && -f "$FORMAL/preparation/embedding_cache.npz" ]]; then
  echo "[setup] fitting global partition on full cache"
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$PYTHON" experiments/control_matrix/fit_partition.py \
    --method global \
    --dataset-name cube \
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
