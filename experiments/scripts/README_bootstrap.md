# Main-experiment bootstrap

CPU-only hierarchical paired eval-block bootstrap for all committed main
control-matrix success-rate tables.

## Statistical unit

- **Default resampling unit:** `eval-block` — each of the five evaluation seeds
  is one block (50 paired starts kept together).
- **Hierarchy (learned methods):** resample predictor-training seeds with
  replacement, then resample eval blocks with replacement. The same train/eval
  index draws are shared across methods for paired comparisons.
- **Official baseline:** resample eval blocks only (no training seed).
- **Partitioned baselines:** average partition seeds 0/1/2 within each
  `(train_seed, eval_block)` cell **before** bootstrap (matching
  `aggregate_matrix.py` / `aggregate_tworoom_main.py`).
- **Auto-LAP:** follows the gate manifest — spectral branch uses deployment
  partition seed only; global branch reuses Global-FT eval paths.

## Command

From the repository root (NumPy only; no venv required):

```bash
python experiments/scripts/bootstrap_main_results.py \
  --repo-root . \
  --n-bootstrap 50000 \
  --seed 20260818 \
  --workers auto \
  --resampling-unit eval-block \
  --output-dir experiments/bootstrap_results
```

Smoke test (100 bootstrap draws, all code paths):

```bash
python experiments/scripts/bootstrap_main_results.py \
  --repo-root . --smoke-test --workers 4 \
  --output-dir experiments/bootstrap_results_smoke
```

## Outputs

| File | Description |
|------|-------------|
| `bootstrap_summary.csv` | Per-method point estimates and 95% percentile CIs |
| `bootstrap_contrasts.csv` | Auto-LAP minus each baseline (percentage points) |
| `bootstrap_metadata.json` | Provenance, validation, pairing checks, per-cell timing |
| `bootstrap_tables.tex` | LaTeX tables for all 16 model–task–horizon cells |

## Configuration

`bootstrap_config.json` maps each model–task pair to its results root and main
methods. LeWM–Cube is marked pending when `experiments/cube/matrix*` is absent;
non-`--strict` runs mark those cells **Pending** and continue with the other 14.

## Tests

```bash
python -m pytest tests/test_bootstrap_main_results.py -q
```

Optional: save all bootstrap replicate draws to compressed NPZ files under
`{output_dir}/replicates/{model}_{task}_{horizon}/` with `--save-replicates`.
