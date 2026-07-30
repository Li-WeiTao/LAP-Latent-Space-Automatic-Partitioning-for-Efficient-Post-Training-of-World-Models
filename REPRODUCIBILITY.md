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

## Supported TwoRoom entrypoints

The machine-readable authority is
`experiments/tworoom/reproduction_manifest.json`. It separates the canonical
paper matrix from analysis, ablation, and validation profiles. Preflight does
not start training:

```bash
python experiments/tworoom/reproduce.py list --profile full
python experiments/tworoom/reproduce.py check --profile main
```

After setting the external paths and GPU, the complete seven-method main matrix
is launched with:

```bash
python experiments/tworoom/reproduce.py run --profile main
```

The entrypoint includes Official baseline, Joint-Continue 3ep, Global-FT50,
Random-Voronoi K3-50, K-means++ K3-50, Spectral K3-50, and human rooms3-50.
Automatic methods use the dataset/model-parameterized control-matrix runner;
the human partition remains a TwoRoom-only comparison. Development-time queue,
`nohup`, recovery, and partial-rerun launchers are retained under
`experiments/tworoom/scripts/legacy/` for provenance and are not called.

## External inputs

The repository never silently substitutes external inputs. Set the dataset and
official checkpoint paths explicitly:

```bash
export LAP_TWOROOM_DATA=/path/to/tworoom.h5
export LAP_TWOROOM_STARTS=/path/to/train_global_reference_starts.npy
export LAP_LEWM_CHECKPOINT=/path/to/lewm_object.ckpt
export GPU=0
```

The checkpoint used by the committed runs has SHA-256
`18b5764492c74de5487efdadb66adab11876cb230952765b17c0815fa87b13ff`.
Record the dataset SHA-256 locally before a new full run because the dataset is
not distributed by this repository.

Every new run should record:

- Git commit and working-tree status;
- dataset and checkpoint SHA-256;
- command line and environment overrides;
- CUDA, PyTorch, NumPy, SciPy, and scikit-learn versions;
- partition seed, predictor fine-tuning seed, and evaluation seed;
- output manifest and result-file checksums.

## What is and is not precomputed

Compact per-run JSON, CSV summaries, deployable routing artifacts, figure source
tables, plotting programs, and figures are committed. Model checkpoints, dense
embedding caches, videos, and large per-timestep arrays are omitted. The
checked-in summaries can be audited without rerunning training. The omitted
spectral inputs have a complete rebuild path:

```bash
GPU=0 bash experiments/tworoom/scripts/internal/prepare_tworoom_spectral_inputs.sh
```

Encoding is an explicit upstream preprocessing stage, not part of `LAP.fit`.
The architecture-neutral method receives a prepared latent cache and the
pretrained model. For TwoRoom/LeWM, the cache stores exact `emb`, `act_emb`, and
`region_starts` arrays. The cache can be packaged as one file with:

```bash
python experiments/tworoom/build_lap_latent_cache.py \
  --embedding-source-dir "$EMBED_DIR" \
  --train-starts "$EMBED_DIR/train_global_reference_starts.npy" \
  --output "$EMBED_DIR/tworoom_lewm_train_latent_cache.npz"
```

An official cache may replace this step if its backend adapter supplies the same
semantic fields and records its provenance.

If an official cache is unavailable, use the model- and task-parameterized
upstream encoder. The command itself has no TwoRoom/Push-T switch and imports no
fixed encoder architecture:

```bash
lap-cache encode \
  --dataset-factory backends.lewm.encoding:make_hdf5_transition_dataset \
  --dataset-config configs/encoding/tworoom_lewm_dataset.json \
  --encoder-factory backends.lewm.encoding:make_encoder \
  --encoder-config configs/encoding/lewm_encoder.json \
  --pretrained-model "$LAP_LEWM_CHECKPOINT" \
  --output "$EMBED_DIR/tworoom_lewm_train_latent_cache.npz" \
  --batch-shape-mode exact --device cuda
```

`--batch-shape-mode exact` is the publication-reproduction setting: it keys a
frame by both its dataset ID and legacy visual batch shape. For a new model with
no historical bitwise cache contract, `fixed` retains the same lossless
unique-frame/inverse-index logic and permits an independently tuned
`--frame-batch-size`. Dataset and encoder configuration, Git commit, checkpoint
hash, cache report, and adapter import paths are required run metadata.

The geometry-named files created by this command are storage shards only. The
automatic partitioner concatenates and deduplicates their latent vectors and
does not consume the geometry labels.

## Verification levels

- `static`: Python syntax, imports, shell syntax, relative paths, and manifest
  integrity are checked.
- `smoke`: a small synthetic partition/router test runs without the TwoRoom data.
- `cache-contract`: the new LAP latent-cache and indexed-partition interfaces
  reproduce all 693,728 historical TwoRoom training assignments exactly.
- `result-audit`: checked-in summary tables are recomputed from committed compact
  per-run results where available.
- `full`: encoding, partitioning, regional training, and paired planning
  evaluation are rerun. Full verification is intentionally not executed during
  repository migration.

The repository validator includes a 185-file main-result audit and verifies
that every method uses the same `eval_start_indices` for each evaluation seed.

## Migration equivalence

The independent API is not a parallel reimplementation beside the TwoRoom
code. The executable experiment entrypoints delegate their result-affecting
operations to the new modules:

- graph construction, eigengap selection, and spectral labels resolve to
  `lap.partition.spectral`;
- predictor-only FP32 optimization resolves to `backends.lewm.finetuning`;
- online Torch routing resolves to `backends.lewm.routing`, whose assignments
  are tested against the NumPy `lap.routing.VoronoiRouter`;
- the historical full-data spectral assignments are replayed through
  `LeWMLatentCache` and `IndexedPartitioner` before any predictor is trained.

Supported historical script names remain as experiment-specific
CLI/configuration helpers and are no longer separate algorithm implementations.
Development-only orchestration launchers have been quarantined under
`scripts/legacy`; the canonical registry never resolves to them.
