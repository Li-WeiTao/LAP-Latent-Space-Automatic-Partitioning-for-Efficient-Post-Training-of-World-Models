# Gate-only sensitivity audit

This audit reuses frozen latent caches and recomputes kNN landmark graphs,
spectra, and gate arithmetic only. **No predictors were trained** and **no
planning / MPC evaluation** was run. Thresholds were not adjusted post hoc.

Provenance: commit `a7c5cfa7dabc3d77c6ce6178f4fe378e4123d17e`, dirty=False, final=True, source_digest=`0b5c3aaf01686c233229dc6cf0b76df269333e2165b2bb4fc96943faef0fe93d`.

## Baseline margins

| Model | Task | Decision | Safety margin | Prominence margin |
|---|---|---|---:|---:|
| lewm | tworoom | spectral | 0.4950 | 0.2349 |
| lewm | pusht | global | -0.4880 | -0.2222 |
| lewm | reacher | global | 0.2729 | -0.3687 |
| lewm | cube | global | 0.4874 | -0.1218 |

## Non-K OAT decision agreement

Overall agreement (excludes K sweep): **48/48**
- lewm/tworoom B: 2/2 (100.0%)
- lewm/tworoom M: 2/2 (100.0%)
- lewm/tworoom c_bg: 2/2 (100.0%)
- lewm/tworoom c_pert: 2/2 (100.0%)
- lewm/tworoom kNN: 2/2 (100.0%)
- lewm/tworoom rho: 2/2 (100.0%)
- lewm/pusht B: 2/2 (100.0%)
- lewm/pusht M: 2/2 (100.0%)
- lewm/pusht c_bg: 2/2 (100.0%)
- lewm/pusht c_pert: 2/2 (100.0%)
- lewm/pusht kNN: 2/2 (100.0%)
- lewm/pusht rho: 2/2 (100.0%)
- lewm/reacher B: 2/2 (100.0%)
- lewm/reacher M: 2/2 (100.0%)
- lewm/reacher c_bg: 2/2 (100.0%)
- lewm/reacher c_pert: 2/2 (100.0%)
- lewm/reacher kNN: 2/2 (100.0%)
- lewm/reacher rho: 2/2 (100.0%)
- lewm/cube B: 2/2 (100.0%)
- lewm/cube M: 2/2 (100.0%)
- lewm/cube c_bg: 2/2 (100.0%)
- lewm/cube c_pert: 2/2 (100.0%)
- lewm/cube kNN: 2/2 (100.0%)
- lewm/cube rho: 2/2 (100.0%)

## Behavior across candidate region counts (K sweep)

- lewm/tworoom K=2: decision=spectral, prominence_margin=0.38856489801366434
- lewm/tworoom K=3: decision=spectral, prominence_margin=0.234870989579664
- lewm/tworoom K=4: decision=global, prominence_margin=-0.29493369343421155
- lewm/tworoom K=5: decision=global, prominence_margin=-0.21269040595783711
- lewm/pusht K=2: decision=global, prominence_margin=-0.21848744757367405
- lewm/pusht K=3: decision=global, prominence_margin=-0.22224302851474734
- lewm/pusht K=4: decision=global, prominence_margin=-0.12203299455211705
- lewm/pusht K=5: decision=global, prominence_margin=-0.09907507465107217
- lewm/reacher K=2: decision=spectral, prominence_margin=0.35697253156641173
- lewm/reacher K=3: decision=global, prominence_margin=-0.36873056503914603
- lewm/reacher K=4: decision=global, prominence_margin=-0.3247768434184574
- lewm/reacher K=5: decision=global, prominence_margin=-0.060401419673151546
- lewm/cube K=2: decision=spectral, prominence_margin=0.16481191841139087
- lewm/cube K=3: decision=global, prominence_margin=-0.12182655119173552
- lewm/cube K=4: decision=global, prominence_margin=-0.2893948143370122
- lewm/cube K=5: decision=global, prominence_margin=-0.04108877977813244

## Draw-subset pass frequency (baseline M, seeds 0–9)

- lewm/tworoom B=3: pass=1.000, min prominence=0.2274578611243454
- lewm/tworoom B=5: pass=1.000, min prominence=0.22722301746879436
- lewm/tworoom B=10: pass=1.000, min prominence=0.22722301746879436
- lewm/pusht B=3: pass=0.000, min prominence=-0.5094008372491288
- lewm/pusht B=5: pass=0.000, min prominence=-0.5272946642421927
- lewm/pusht B=10: pass=0.000, min prominence=-0.5272946642421927
- lewm/reacher B=3: pass=0.000, min prominence=-0.38555550011778533
- lewm/reacher B=5: pass=0.000, min prominence=-0.386064435927985
- lewm/reacher B=10: pass=0.000, min prominence=-0.386064435927985
- lewm/cube B=3: pass=0.000, min prominence=-0.16320908174709842
- lewm/cube B=5: pass=0.000, min prominence=-0.16519266725348974
- lewm/cube B=10: pass=0.000, min prominence=-0.16519266725348974

## Partition stability by pair/factor

- lewm/tworoom M: mean ARI=0.9590, min ARI=0.9542, n=2
- lewm/tworoom kNN: mean ARI=0.9776, min ARI=0.9757, n=2
- lewm/pusht M: mean ARI=0.7591, min ARI=0.7239, n=2
- lewm/pusht kNN: mean ARI=0.8749, min ARI=0.8615, n=2
- lewm/reacher M: mean ARI=0.8833, min ARI=0.8815, n=2
- lewm/reacher kNN: mean ARI=0.9432, min ARI=0.9284, n=2
- lewm/cube M: mean ARI=0.9341, min ARI=0.9210, n=2
- lewm/cube kNN: mean ARI=0.9549, min ARI=0.9491, n=2

## Non-K boundary / abstention flips

_No non-K OAT flips observed._

## What this audit cannot prove

- Planning performance under alternate K or flipped gate decisions.
- Draw-subset diagnostics do not modify the predeclared gate definition.
