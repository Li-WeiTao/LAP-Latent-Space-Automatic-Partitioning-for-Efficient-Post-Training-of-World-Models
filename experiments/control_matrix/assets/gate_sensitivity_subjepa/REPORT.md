# Gate-only sensitivity audit

This audit reuses frozen latent caches and recomputes kNN landmark graphs,
spectra, and gate arithmetic only. **No predictors were trained** and **no
planning / MPC evaluation** was run. Thresholds were not adjusted post hoc.

Provenance: commit `a7c5cfa7dabc3d77c6ce6178f4fe378e4123d17e`, dirty=False, final=True, source_digest=`0b5c3aaf01686c233229dc6cf0b76df269333e2165b2bb4fc96943faef0fe93d`.

## Baseline margins

| Model | Task | Decision | Safety margin | Prominence margin |
|---|---|---|---:|---:|
| subjepa | tworoom | spectral | 0.4888 | 0.0081 |
| subjepa | pusht | global | -0.2937 | -0.1287 |
| subjepa | reacher | global | 0.1537 | -0.3043 |
| subjepa | cube | global | 0.4800 | -0.0360 |

## Non-K OAT decision agreement

Overall agreement (excludes K sweep): **46/48**
- subjepa/tworoom B: 2/2 (100.0%)
- subjepa/tworoom M: 1/2 (50.0%)
- subjepa/tworoom c_bg: 1/2 (50.0%)
- subjepa/tworoom c_pert: 2/2 (100.0%)
- subjepa/tworoom kNN: 2/2 (100.0%)
- subjepa/tworoom rho: 2/2 (100.0%)
- subjepa/pusht B: 2/2 (100.0%)
- subjepa/pusht M: 2/2 (100.0%)
- subjepa/pusht c_bg: 2/2 (100.0%)
- subjepa/pusht c_pert: 2/2 (100.0%)
- subjepa/pusht kNN: 2/2 (100.0%)
- subjepa/pusht rho: 2/2 (100.0%)
- subjepa/reacher B: 2/2 (100.0%)
- subjepa/reacher M: 2/2 (100.0%)
- subjepa/reacher c_bg: 2/2 (100.0%)
- subjepa/reacher c_pert: 2/2 (100.0%)
- subjepa/reacher kNN: 2/2 (100.0%)
- subjepa/reacher rho: 2/2 (100.0%)
- subjepa/cube B: 2/2 (100.0%)
- subjepa/cube M: 2/2 (100.0%)
- subjepa/cube c_bg: 2/2 (100.0%)
- subjepa/cube c_pert: 2/2 (100.0%)
- subjepa/cube kNN: 2/2 (100.0%)
- subjepa/cube rho: 2/2 (100.0%)

## Behavior across candidate region counts (K sweep)

- subjepa/tworoom K=2: decision=global, prominence_margin=-0.1061143423934513
- subjepa/tworoom K=3: decision=spectral, prominence_margin=0.008057418831305008
- subjepa/tworoom K=4: decision=global, prominence_margin=-0.3305993127777221
- subjepa/tworoom K=5: decision=global, prominence_margin=-0.28639300951531865
- subjepa/pusht K=2: decision=global, prominence_margin=-0.041421762100867815
- subjepa/pusht K=3: decision=global, prominence_margin=-0.1287278359573476
- subjepa/pusht K=4: decision=global, prominence_margin=-0.3263305348947749
- subjepa/pusht K=5: decision=global, prominence_margin=-0.19889858906175648
- subjepa/reacher K=2: decision=spectral, prominence_margin=0.3009356531654765
- subjepa/reacher K=3: decision=global, prominence_margin=-0.30430387090303435
- subjepa/reacher K=4: decision=spectral, prominence_margin=0.05007674398691364
- subjepa/reacher K=5: decision=global, prominence_margin=-0.2834274537393086
- subjepa/cube K=2: decision=spectral, prominence_margin=0.3283684026376352
- subjepa/cube K=3: decision=global, prominence_margin=-0.03596668709162054
- subjepa/cube K=4: decision=global, prominence_margin=-0.18418224964464588
- subjepa/cube K=5: decision=global, prominence_margin=-0.14424302075925147

## Draw-subset pass frequency (baseline M, seeds 0–9)

- subjepa/tworoom B=3: pass=1.000, min prominence=0.0036862394019559486
- subjepa/tworoom B=5: pass=1.000, min prominence=0.003141657366913353
- subjepa/tworoom B=10: pass=1.000, min prominence=0.003141657366913353
- subjepa/pusht B=3: pass=0.000, min prominence=-0.27644760306763233
- subjepa/pusht B=5: pass=0.000, min prominence=-0.31213851186936414
- subjepa/pusht B=10: pass=0.000, min prominence=-0.31213851186936414
- subjepa/reacher B=3: pass=0.000, min prominence=-0.3923381938155201
- subjepa/reacher B=5: pass=0.000, min prominence=-0.3927446938565787
- subjepa/reacher B=10: pass=0.000, min prominence=-0.3927446938565787
- subjepa/cube B=3: pass=0.000, min prominence=-0.10311871024689512
- subjepa/cube B=5: pass=0.000, min prominence=-0.10311871024689512
- subjepa/cube B=10: pass=0.000, min prominence=-0.10311871024689512

## Partition stability by pair/factor

- subjepa/tworoom M: mean ARI=0.9684, min ARI=0.9623, n=2
- subjepa/tworoom kNN: mean ARI=0.9780, min ARI=0.9764, n=2
- subjepa/pusht M: mean ARI=0.4211, min ARI=0.3975, n=2
- subjepa/pusht kNN: mean ARI=0.6363, min ARI=0.4454, n=2
- subjepa/reacher M: mean ARI=0.2840, min ARI=0.2833, n=2
- subjepa/reacher kNN: mean ARI=0.6709, min ARI=0.4013, n=2
- subjepa/cube M: mean ARI=0.9121, min ARI=0.9110, n=2
- subjepa/cube kNN: mean ARI=0.9218, min ARI=0.9217, n=2

## Non-K boundary / abstention flips

- subjepa/tworoom c_bg=3.5 flipped to global (safety_margin=0.4887743728774375, prominence_margin=-0.04912629396523005)
- subjepa/tworoom M=10000 flipped to global (safety_margin=0.47920756047316615, prominence_margin=-0.040103325112946286)

## What this audit cannot prove

- Planning performance under alternate K or flipped gate decisions.
- Draw-subset diagnostics do not modify the predeclared gate definition.
