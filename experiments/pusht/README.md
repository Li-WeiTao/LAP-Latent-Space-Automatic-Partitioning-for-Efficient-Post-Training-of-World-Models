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
| Cross-shape timesteps | 15 |
| Total cache-build time | 3,861.662 s (64.36 min) |
| Unique-frame encoding time | 3,600.669 s |
| NPZ save time | 242.735 s |
| Measured reduction relative to repeated window encoding | 3.1965x |

The final NPZ reconstructs the original overlapping window layout. Therefore,
the same timestep can appear in multiple output slots even though its encoding
was computed only once per valid encoding key. Repeated output slots are not
repeated encoder calls.

## Batch-shape-aware deduplication contract

Deduplication must not assume that the encoder is invariant to batch shape.
The correct rule for each dataset/encoder pair is:

1. For repeated occurrences with the same timestep and the same visual batch
   shape, encode once and reuse the result.
2. For a timestep occurring under different visual batch shapes, compare the
   corresponding FP32 encoder outputs.
3. If changing batch shape does not change the embedding under the required
   reproduction criterion, retain one embedding for that timestep.
4. If changing batch shape changes the embedding, retain each distinct
   `(timestep, batch shape)` key.

This rule follows the measured behavior of the encoder; preserving legacy
batch shapes is not, by itself, a justification for retaining duplicates.

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

The differences are directionally small, but they are real FP32 output
differences: all 192 components differed in every cross-shape pair. Under the
bitwise reproduction contract, the 15 extra keys must therefore be retained.
The completed cache already does this correctly and does not need to be
rebuilt.

The stopped partition run exposed a downstream contract mismatch rather than
an encoding failure: the partition loader collapsed records by timestep and
required strict equality, even though the cache intentionally retained
batch-shape-sensitive outputs. Before resuming the PushT matrix, that loader
must consume the cache's encoding keys without replacing the two outputs by an
arbitrary first occurrence or weakening the comparison tolerance merely to
force them to merge.
