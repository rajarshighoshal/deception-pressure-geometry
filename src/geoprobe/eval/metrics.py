from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred == y).float().mean().item()


def binary_rates(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = pred.bool()
    y = y.bool()
    tp = (pred & y).sum().item()
    tn = (~pred & ~y).sum().item()
    fp = (pred & ~y).sum().item()
    fn = (~pred & y).sum().item()
    denom_pos = max(tp + fn, 1)
    denom_neg = max(tn + fp, 1)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "tpr": tp / denom_pos,
        "fpr": fp / denom_neg,
        "tnr": tn / denom_neg,
        "fnr": fn / denom_pos,
    }
