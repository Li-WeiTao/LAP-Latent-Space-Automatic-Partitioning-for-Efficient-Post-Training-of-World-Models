# Migration validation report

## Material Passport

- ID: `lap-tworoom-migration-2026-07-30`
- Type: reproducibility migration
- Status: `ANALYZED` (full training was not rerun)
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
| Top-level TwoRoom Python files copied | 29/29 |
| TwoRoom launcher files copied | 80/80 |
| TwoRoom test files copied | 3/3 |
| Figure/source asset files copied | 43/43, SHA-256 identical |
| Executable references to old repository | 0 |

Main CLI checks covered landmark spectral partitioning, regional predictor
fine-tuning, TwoRoom success-rate evaluation, unique-timestep re-encoding, and
Joint-Continue.

## Verification boundary

The migration verifies repository structure, syntax, imports, compact results,
artifact contracts, routing behavior, manifest transactions, and numerical
equivalence of the extracted spectral primitives. It does not claim a full
reproduction of GPU encoding, predictor training, or MPC evaluation because
those runs require the external dataset, official checkpoint, and multiple GPU
hours. The original completed summaries and figures are included for audit.

## Source preservation

The source repository was never edited. Its final Git status matched the initial
status observed before migration:

```text
?? %ln
?? .gitignore
?? cd
?? experiments/
```
