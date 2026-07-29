# Reproducibility contract

## Scope

The migrated scope begins at `# trajectory deviation experiment` in the original
TwoRoom development README and includes every subsequent experiment:

- trajectory deviation and predictor-switch rollout;
- geometry rooms3 and priority5 regional predictors;
- global predictor fine-tuning and Joint-Continue controls;
- latent linear separability probes;
- spherical K-means, preprocessing, and K-means++ restart studies;
- Random-Voronoi and landmark spectral K3 partitions;
- per-partition predictor training;
- MPC and step-routing evaluations;
- long- and short-horizon success-rate aggregation;
- inference timing and lossless unique-timestep re-encoding.

## Fixed experimental factors

- Fine-tuning seeds: 0, 42, and 625 for the reported main comparison.
- Partition seeds: 0, 1, and 2 where partition stability is evaluated.
- Evaluation: the same five evaluation seeds and paired starts used by the
  original LeWM TwoRoom protocol.
- Encoder: frozen for Global-FT and all regional predictor methods.
- Precision: FP32 for the compute comparison against Joint-Continue.
- Main automatic configuration: K=3, landmark spectral partition, 50 predictor
  epochs, and per-MPC routing.

## External inputs

The repository never silently substitutes external inputs. Set the dataset and
official checkpoint paths explicitly:

```bash
export LAP_TWOROOM_DATA=/path/to/tworoom.h5
export LAP_LEWM_CHECKPOINT=/path/to/lewm_object.ckpt
```

Every new run should record:

- Git commit and working-tree status;
- dataset and checkpoint SHA-256;
- command line and environment overrides;
- CUDA, PyTorch, NumPy, SciPy, and scikit-learn versions;
- partition seed, predictor fine-tuning seed, and evaluation seed;
- output manifest and result-file checksums.

## What is and is not precomputed

Compact CSV/JSON summaries, partition artifacts needed for online routing, and
figures are committed. Model checkpoints, dense embedding caches, videos, and
large per-timestep arrays are omitted. The checked-in summaries can be audited
without rerunning training; reproducing a model from raw trajectories requires
the external dataset and official checkpoint.

## Verification levels

- `static`: Python syntax, imports, shell syntax, relative paths, and manifest
  integrity are checked.
- `smoke`: a small synthetic partition/router test runs without the TwoRoom data.
- `result-audit`: checked-in summary tables are recomputed from committed compact
  per-run results where available.
- `full`: encoding, partitioning, regional training, and paired planning
  evaluation are rerun. Full verification is intentionally not executed during
  repository migration.
