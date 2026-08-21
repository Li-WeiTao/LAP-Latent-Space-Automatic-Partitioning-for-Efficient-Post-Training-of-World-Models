# Gate-only sensitivity audit

This audit reuses frozen latent caches and recomputes kNN landmark graphs,
spectra, and gate arithmetic only. **No predictors were trained** and **no
planning / MPC evaluation** was run. Thresholds were not adjusted post hoc.

## Baseline margins

| Model | Task | Decision | Safety margin | Prominence margin |
|---|---|---|---:|---:|
| lewm | tworoom | spectral | 0.4950 | 0.2349 |
| lewm | pusht | global | -0.4880 | -0.2223 |
| lewm | reacher | global | 0.2729 | -0.3688 |
| lewm | cube | global | 0.4874 | -0.1218 |

## Non-K OAT decision agreement

Overall agreement (excludes K sweep): **48/48**

- lewm/tworoom B: 2/2 (100.0%) vs baseline=spectral
- lewm/tworoom K: 1/3 (33.3%) vs baseline=spectral
- lewm/tworoom M: 2/2 (100.0%) vs baseline=spectral
- lewm/tworoom c_bg: 2/2 (100.0%) vs baseline=spectral
- lewm/tworoom c_pert: 2/2 (100.0%) vs baseline=spectral
- lewm/tworoom kNN: 2/2 (100.0%) vs baseline=spectral
- lewm/tworoom rho: 2/2 (100.0%) vs baseline=spectral
- lewm/pusht B: 2/2 (100.0%) vs baseline=global
- lewm/pusht K: 3/3 (100.0%) vs baseline=global
- lewm/pusht M: 2/2 (100.0%) vs baseline=global
- lewm/pusht c_bg: 2/2 (100.0%) vs baseline=global
- lewm/pusht c_pert: 2/2 (100.0%) vs baseline=global
- lewm/pusht kNN: 2/2 (100.0%) vs baseline=global
- lewm/pusht rho: 2/2 (100.0%) vs baseline=global
- lewm/reacher B: 2/2 (100.0%) vs baseline=global
- lewm/reacher K: 2/3 (66.7%) vs baseline=global
- lewm/reacher M: 2/2 (100.0%) vs baseline=global
- lewm/reacher c_bg: 2/2 (100.0%) vs baseline=global
- lewm/reacher c_pert: 2/2 (100.0%) vs baseline=global
- lewm/reacher kNN: 2/2 (100.0%) vs baseline=global
- lewm/reacher rho: 2/2 (100.0%) vs baseline=global
- lewm/cube B: 2/2 (100.0%) vs baseline=global
- lewm/cube K: 2/3 (66.7%) vs baseline=global
- lewm/cube M: 2/2 (100.0%) vs baseline=global
- lewm/cube c_bg: 2/2 (100.0%) vs baseline=global
- lewm/cube c_pert: 2/2 (100.0%) vs baseline=global
- lewm/cube kNN: 2/2 (100.0%) vs baseline=global
- lewm/cube rho: 2/2 (100.0%) vs baseline=global

## Draw-subset pass frequency (seeds 0–9)

- lewm/tworoom B=3: pass=1.000, min safety margin=0.4945036115956747, min prominence margin=0.22743937107276047
- lewm/tworoom B=5: pass=1.000, min safety margin=0.4945036115956747, min prominence margin=0.2272044528835373
- lewm/tworoom B=10: pass=1.000, min safety margin=0.4945036115956747, min prominence margin=0.2272044528835373
- lewm/pusht B=3: pass=0.000, min safety margin=-3.329227960455585, min prominence margin=-0.5094056127110256
- lewm/pusht B=5: pass=0.000, min safety margin=-3.329227960455585, min prominence margin=-0.5273058334516084
- lewm/pusht B=10: pass=0.000, min safety margin=-3.329227960455585, min prominence margin=-0.5273058334516084
- lewm/reacher B=3: pass=0.000, min safety margin=0.08269044708614726, min prominence margin=-0.3855006534231061
- lewm/reacher B=5: pass=0.000, min safety margin=0.08269044708614726, min prominence margin=-0.38600958922699063
- lewm/reacher B=10: pass=0.000, min safety margin=0.08269044708614726, min prominence margin=-0.38600958922699063
- lewm/cube B=3: pass=0.000, min safety margin=0.4848286662952329, min prominence margin=-0.1632090848882175
- lewm/cube B=5: pass=0.000, min safety margin=0.4848286662952329, min prominence margin=-0.16519267039529284
- lewm/cube B=10: pass=0.000, min safety margin=0.4848286662952329, min prominence margin=-0.16519267039529284

## Partition stability (graph/landmark-changing OAT only)

Mean ARI=0.9108, min ARI=0.7239 (16 comparisons)

## Boundary / abstention flips

- lewm/tworoom K=4 flipped to global (safety_margin=0.3206968639988833, prominence_margin=-0.294942405009621)
- lewm/tworoom K=5 flipped to global (safety_margin=0.484379344562608, prominence_margin=-0.21269030095076408)
- lewm/reacher K=2 flipped to spectral (safety_margin=0.4974938569666636, prominence_margin=0.3570040784725116)
- lewm/cube K=2 flipped to spectral (safety_margin=0.4980330337132345, prominence_margin=0.16481186479378118)

## Why no predictor training?

The spectral gate consumes only frozen state latents. All sweeps re-evaluate
cached spectra or rebuild kNN graphs; partition labels are diagnosis-only.

## What this audit cannot prove

- Planning performance under alternate K or flipped gate decisions.
- Sub-JEPA pairs without latent caches (preflight lists exact missing paths).
- Draw-subset diagnostics do not modify the predeclared gate definition.
