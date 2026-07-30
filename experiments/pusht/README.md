# PushT experiment record

This directory records the PushT reproduction of the LAP control matrix. The
matrix uses the same method families as the TwoRoom main experiment, except
that it does not include a manually designed partition:

- official pretrained baseline;
- Joint-Continue for 3 epochs;
- Global-FT for 50 epochs;
- Random-Voronoi;
- K-means++;
- spectral partitioning.

## Frozen-encoder cache

The completed PushT LeWM cache is stored at
`experiments/pusht/matrix/pusht_lewm_train_latent_cache.npz`; its encoder report
is the adjacent `.report.json` file. The measured run used FP32 inference and
the legacy transition batch size of 128.

| Quantity | Value |
|---|---:|
| Transition windows | 1,850,815 |
| Latent frames per window | 4 |
| Reconstructed frame slots | 7,403,260 |
| Unique timesteps | 2,315,671 |
| Encoded `(timestep, batch shape)` keys | 2,315,686 |
| Canonical partition-input latents | 2,315,671 |
| Cross-shape timesteps | 15 |
| Total cache-build time | 3,861.662 s (64.36 min) |
| Unique-frame encoding time | 3,600.669 s |
| NPZ save time | 242.735 s |
| Measured reduction relative to repeated window encoding | 3.1965x |

The final NPZ reconstructs the original overlapping window layout. Therefore,
the same timestep can appear in multiple output slots even though its encoding
was computed only once per valid encoding key. Repeated output slots are not
repeated encoder calls.

## Canonical-first deduplication contract

The cache audit shows that GPU batch shape can introduce small FP32 numerical
differences. These differences must not turn one observation into multiple
latent states. LAP therefore uses the following partition-input rule:

1. Flatten cached windows in their original order.
2. Stable-sort occurrences by timestep.
3. Retain the first occurrence of each timestep as its canonical latent.
4. Discard every later occurrence, including the 15 cases encoded under a
   different visual batch shape.

The first occurrence is representative because almost all later occurrences
were encoded with the same batch shape and are bitwise identical. For the 15
cross-shape cases, the observed difference is numerical noise rather than a
new environment state.

### PushT measurement

The audit reconstructed the original frame indices and the legacy visual batch
assignment, then compared every embedding for the 15 timesteps that occur in
both the 252-frame final batch and a 512-frame full batch.

| Check | Result |
|---|---:|
| Cross-shape timesteps tested | 15 |
| Bitwise-equal 252-vs-512 pairs | 0 / 15 |
| Pairs passing absolute tolerance `1e-7` | 0 / 15 |
| Pairs passing absolute tolerance `1e-6` | 0 / 15 |
| Pairs passing relative tolerance `1e-5`, absolute tolerance `1e-7` | 0 / 15 |
| Maximum absolute component difference | 0.0010030866 |
| Maximum L2 difference | 0.0040290930 |
| Maximum relative L2 difference | 0.0003500833 |
| Minimum cosine similarity | 0.9999999420 |
| Same-shape repeated embeddings that were bitwise equal | 15 / 15 per shape |

The differences are directionally tiny even though they are not bitwise equal:
the minimum cosine similarity is above 0.99999994. The completed cache does not
need to be rebuilt. Its windows preserve the legacy encoder output, while the
partition loader deterministically canonicalizes them to one latent per
timestep using the first occurrence.

The stopped partition run was caused by the old loader requiring all repeated
slots to be bitwise equal. The corrected loader no longer interprets
batch-shape noise as a second state or as a fatal cache inconsistency; it keeps
the stable first occurrence and records this policy in the partition manifest.

## Eight-GPU matrix command

The parallel controller uses one bounded worker per listed GPU, four CPU
threads per worker, disjoint output directories, and a dependency barrier
between partitioning, training, official-start generation, model evaluation,
and aggregation:

```bash
DATASET_NAME=pusht \
DATA_FILE=/path/to/pusht_expert_train.h5 \
CHECKPOINT=/path/to/lewm_object.ckpt \
EVAL_CONFIG=pusht \
EVAL_DATASET_NAME=pusht_expert_train \
CACHE_DIR=/path/to/stable_worldmodel_cache \
WORK_ROOT=experiments/pusht/matrix \
PYTHON=/path/to/lewm/python \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS=4 \
bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh
```

Each run records its resolved settings, Git commit, controller PID, and
per-task logs under `experiments/pusht/matrix/logs/parallel_<RUN_ID>/`.
