# Canonical paper experiment

`run_tworoom_main_matrix.sh` is the only supported full TwoRoom paper-matrix
entrypoint. `run_tworoom_human_rooms3_matrix.sh` is its TwoRoom-specific human
partition arm and is called automatically.

Prefer `python experiments/tworoom/reproduce.py run --profile main` so external
input preflight is performed before any GPU work starts.
