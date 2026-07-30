# Migration validation report

## Material Passport

- ID: `lap-tworoom-migration-2026-07-30`
- Type: reproducibility migration
- Status: `PARTIALLY_REPRODUCED` (compact results verified; full GPU training not rerun)
- Source commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- Source path: `/data/sicong/weitao/le-wm`
- Target path: `/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models`

## Completed checks

| Check | Result |
|---|---:|
| Python syntax failures | 0 |
| Bash syntax failures | 0 |
| Invalid compact JSON files | 0 |
| Invalid committed spectral artifacts | 0/3 |
| Main CLI `--help` checks | 5/5 passed |
| Landmark spectral contract tests | 8/8 passed |
| Additional routing/manifest/rollout tests | 18/18 passed |
| Extracted architecture-neutral spectral primitives | exact numerical match |
| Main-result JSON files independently aggregated | 185/185 |
| Main plot seed values reproduced | 19/19 exact |
| Paired evaluation-start checks | 5/5 eval seeds pass across all methods |
| Main/UMAP/probe PNG rerender hashes | 4/4 exact |
| Starts-only raw-data bootstrap smoke test | passed |
| Wheel compatibility modules | 5/5 included |
| Top-level TwoRoom Python files copied | 29/29 |
| TwoRoom launcher files copied | 80/80 |
| TwoRoom test files copied | 3/3 |
| Figure/source asset files copied | 43/43, SHA-256 identical |
| Executable references to old repository | 0 |

Main CLI checks covered landmark spectral partitioning, regional predictor
fine-tuning, TwoRoom success-rate evaluation, unique-timestep re-encoding, and
Joint-Continue.

The starts-only smoke test used the configured TwoRoom dataset and official
checkpoint without training or dense re-encoding. It reproduced 693,728 global
training starts and the expected storage-shard counts:

| Storage shard | Starts |
|---|---:|
| doorway_corridor | 38,552 |
| near_wall | 31,533 |
| common | 297,211 |
| right_room | 324,933 |
| left_room | 330,243 |

The generated region names are not consumed as labels by spectral partitioning;
the loader concatenates and deduplicates their latent vectors.

## Verification boundary

The migration verifies repository structure, syntax, imports, compact results,
artifact contracts, routing behavior, manifest transactions, result aggregation,
figure rendering, packaging, and numerical equivalence of the extracted
spectral primitives. It also verifies that the newly documented raw-data path
reconstructs every start-index input needed by the lossless cache builder.

It does not claim a new full reproduction of dense GPU encoding, predictor
training, or MPC evaluation; those stages require multiple GPU hours. Their
executables, fixed configurations, external-input contract, and rebuild chain
are present, while the original completed per-run results remain available for
audit.

## Source preservation

The source repository was never edited. Its final Git status matched the initial
status observed before migration:

```text
?? %ln
?? .gitignore
?? cd
?? experiments/
```
