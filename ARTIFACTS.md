# Large artifact policy

The source experiment directory occupied approximately 43 GiB. Git tracks code,
compact numerical results, routing artifacts, and figures, but not replaceable
training products.

## Excluded classes

| Class | Examples | Reason | Rebuild path |
|---|---|---|---|
| Dataset | `tworoom.h5` | external benchmark data | configure `LAP_TWOROOM_DATA` |
| Official checkpoint | `lewm_object.ckpt` | upstream model artifact | configure `LAP_LEWM_CHECKPOINT` |
| Predictor checkpoints | `*.ckpt` | roughly 72 MB each | regional/global training scripts |
| Embedding caches | `*_embeddings.npz` | up to roughly 1.27 GB each | `unique_timestep_reencode.py` |
| Evaluation cache | transition NPZ files | hundreds of MB | trajectory cache command in experiment README |
| Videos | `*.mp4` | generated qualitative output | evaluation scripts |
| Temporary state | `_work/`, locks, PIDs | non-portable runtime state | regenerated automatically |

## Included routing artifacts

The small deployable artifacts for the final K3 spectral and Random-Voronoi
partitions are committed: normalization statistics, routing prototypes,
prototype owner IDs, centroids, cluster metadata, diagnostics, and compressed
labels. The selected K-means++ R=50 artifacts for outer seeds 0, 1, and 2 are
also retained for the reported comparison.

Large artifacts may later be published as a versioned release or external data
record. They should not be added directly to normal Git history.
