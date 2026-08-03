"""Fast, lossless latent-cache construction."""

from .fast import FastEncodingConfig, FastLatentCacheEncoder, recompute_latent_windows

__all__ = ["FastEncodingConfig", "FastLatentCacheEncoder", "recompute_latent_windows"]
