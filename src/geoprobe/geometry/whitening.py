from __future__ import annotations

import torch


def fit_whitener(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(eps))) @ evecs.T
    return mean, inv_sqrt


def apply_whitener(x: torch.Tensor, mean: torch.Tensor, inv_sqrt: torch.Tensor) -> torch.Tensor:
    return (x - mean) @ inv_sqrt
