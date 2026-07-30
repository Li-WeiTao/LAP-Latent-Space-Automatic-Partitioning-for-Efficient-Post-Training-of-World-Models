"""Torch implementation of LAP's deployable Voronoi routing contract."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def transform_latent_torch(
    latent: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply LAP's fitted Z-score and L2 transform to a latent batch."""

    if mean is None or mean.numel() == 0:
        return latent
    if scale is None or scale.numel() == 0:
        raise ValueError("a normalization scale is required with a mean")
    return F.normalize((latent.float() - mean) / (scale + eps), dim=1)


def route_voronoi_torch(
    latent: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_region_ids: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    eps: float = 1e-12,
    spherical: bool = True,
) -> torch.Tensor:
    """Route tensors with the same transform and ownership rule as LAP core."""

    routed = transform_latent_torch(
        latent, mean=mean, scale=scale, eps=eps
    ).float()
    if spherical:
        routed = F.normalize(routed, dim=1)
        normalized_prototypes = F.normalize(prototypes.float(), dim=1)
        prototype_ids = (routed @ normalized_prototypes.T).argmax(dim=1)
    else:
        prototype_ids = torch.cdist(
            routed, prototypes.float(), p=2
        ).argmin(dim=1)
    return prototype_region_ids[prototype_ids]
