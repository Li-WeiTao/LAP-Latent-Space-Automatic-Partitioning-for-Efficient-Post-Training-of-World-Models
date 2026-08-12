#!/usr/bin/env bash
# Shared runtime for Held-out Region-Risk Analysis.
# "formal" in this filename is an internal compatibility term.
# Source this file; do not execute directly.
set -euo pipefail

ROOT="${ROOT:-/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models}"
LEWM_VENV="${LEWM_VENV:-/data/sicong/weitao/le-wm/.venv/bin/python}"

export ROOT
export PYTHON="${PYTHON:-$LEWM_VENV}"
export PYTHONPATH="${ROOT}:${ROOT}/experiments/tworoom${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[held-out-region-risk] missing python: $PYTHON" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import os
import sys
import torch
import stable_worldmodel  # noqa: F401

print(
    f"[held-out-region-risk] python={sys.executable} "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"stable_worldmodel=ok"
)
if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
    if not torch.cuda.is_available():
        raise SystemExit(f"CUDA unavailable for {sys.executable}")
    print(f"[held-out-region-risk] device={torch.cuda.get_device_name(0)} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("[held-out-region-risk] per-worker CUDA_VISIBLE_DEVICES set by parallel controller")
PY
