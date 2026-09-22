"""Exact full-covariance GMM MAP routing on already transformed latents."""
import numpy as np


def route_numpy(values, parameters, chunk_size=8192):
    means = np.asarray(parameters['means'], dtype=np.float64)
    precision = np.asarray(parameters['precisions_cholesky'], dtype=np.float64)
    weights = np.asarray(parameters['weights'], dtype=np.float64)
    shape = values.shape[:-1]
    flat = np.asarray(values).reshape(-1, means.shape[-1])
    logdet = np.log(np.diagonal(precision, axis1=-2, axis2=-1)).sum(axis=1)
    output = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), chunk_size):
        x = flat[start:start + chunk_size].astype(np.float64)
        scores = np.column_stack([
            -0.5 * np.square((x - m) @ p).sum(axis=1) + d + np.log(w)
            for m, p, d, w in zip(means, precision, logdet, weights)
        ])
        output[start:start + len(x)] = scores.argmax(axis=1)
    return output.reshape(shape)


def attach_torch(model, parameters):
    import torch
    device = model.cluster_centroids.device
    for key in ('means', 'precisions_cholesky', 'weights'):
        model.register_buffer('gmm_' + key, torch.as_tensor(parameters[key], dtype=torch.float64, device=device))


def route_torch(values, means, precision, weights):
    import torch
    x = values.to(dtype=means.dtype)
    logdet = torch.log(torch.diagonal(precision, dim1=-2, dim2=-1)).sum(dim=-1)
    scores = torch.stack([
        -0.5 * ((x - means[k]) @ precision[k]).square().sum(dim=-1)
        + logdet[k] + torch.log(weights[k]) for k in range(len(means))
    ], dim=-1)
    return scores.argmax(dim=-1)
