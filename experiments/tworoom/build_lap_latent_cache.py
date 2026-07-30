#!/usr/bin/env python3
"""Package lossless TwoRoom LeWM shards as one LAP latent-cache input."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from trajectory import (
    load_global_train_embeddings_from_region_caches,
    save_embedding_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-source-dir", type=Path, required=True)
    parser.add_argument("--train-starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"latent cache already exists: {args.output}; use --overwrite explicitly"
        )
    starts = np.load(args.train_starts)
    emb, act_emb = load_global_train_embeddings_from_region_caches(
        args.embedding_source_dir, starts, name_prefix="train_"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.stem}.",
        suffix=".tmp.npz",
        dir=args.output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_embedding_cache(temporary, emb, act_emb, starts)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"[done] LAP latent cache: {args.output} "
        f"shape={tuple(emb.shape)} transitions={len(starts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
