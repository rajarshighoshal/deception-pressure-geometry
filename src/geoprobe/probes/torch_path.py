"""Torch path-probes (binary MLP + diagonal-Riemannian path probe) and their CV harness.

Extracted from the trajectory_baselines CLI; that module re-exports these. The torch
probes use a train-and-score interface distinct from the sklearn build_probe estimators.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


class TorchBinaryProbe(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None):
        super().__init__()
        if hidden_features is None:
            self.net = nn.Linear(in_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TorchDiagonalRiemannianPathProbe(nn.Module):
    def __init__(self, n_turns: int, activation_dim: int, hidden_features: int):
        super().__init__()
        self.n_turns = n_turns
        self.activation_dim = activation_dim
        self.metric_net = nn.Sequential(
            nn.Linear(activation_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, activation_dim),
        )
        n_steps = n_turns - 1
        geometry_dim = n_steps * 4 + 4
        self.classifier = nn.Sequential(
            nn.Linear(geometry_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        path = features.reshape(-1, self.n_turns, self.activation_dim)
        velocities = path[:, 1:] - path[:, :-1]
        midpoints = 0.5 * (path[:, 1:] + path[:, :-1])
        metric_diag = F.softplus(self.metric_net(midpoints)) + 1e-6
        metric_energy = (metric_diag * velocities.square()).sum(dim=-1)
        metric_length = torch.sqrt(metric_energy + 1e-6)
        euclidean_speed = velocities.norm(dim=-1)

        if velocities.shape[1] > 1:
            left = velocities[:, :-1]
            right = velocities[:, 1:]
            denom = left.norm(dim=-1) * right.norm(dim=-1) + 1e-6
            turn_cosine = ((left * right).sum(dim=-1) / denom).clamp(-1.0, 1.0)
            turn_cosine = F.pad(turn_cosine, (0, 1))
        else:
            turn_cosine = torch.zeros_like(metric_energy)

        displacement = path[:, -1] - path[:, 0]
        meanpoint = path.mean(dim=1)
        displacement_metric = F.softplus(self.metric_net(meanpoint)) + 1e-6
        displacement_energy = (displacement_metric * displacement.square()).sum(dim=-1, keepdim=True)
        euclidean_displacement = displacement.norm(dim=-1, keepdim=True)

        geometry = torch.cat(
            [
                metric_energy,
                metric_length,
                euclidean_speed,
                turn_cosine,
                metric_energy.sum(dim=-1, keepdim=True),
                metric_length.sum(dim=-1, keepdim=True),
                displacement_energy,
                euclidean_displacement,
            ],
            dim=-1,
        )
        return self.classifier(geometry).squeeze(-1)


def torch_train_scores(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test).to(device)).detach().cpu().numpy()


def torch_cross_validated_auroc(
    probe_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    hidden_features: int,
    path_shape: tuple[int, int] | None,
) -> float | None:
    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()), len(set(groups.tolist())))
    if n_splits < 2:
        return None
    if probe_name == "torch_riemannian" and path_shape is None:
        return None

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    out_of_fold = np.full(len(labels), np.nan)
    hidden = None if probe_name == "torch_linear" else hidden_features

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, labels, groups)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx]).astype(np.float32)
        x_test = scaler.transform(features[test_idx]).astype(np.float32)
        y_train = labels[train_idx].astype(np.float32)

        if probe_name == "torch_riemannian":
            assert path_shape is not None
            model = TorchDiagonalRiemannianPathProbe(
                n_turns=path_shape[0],
                activation_dim=path_shape[1],
                hidden_features=hidden_features,
            )
        else:
            model = TorchBinaryProbe(x_train.shape[1], hidden)

        out_of_fold[test_idx] = torch_train_scores(
            model,
            x_train,
            y_train,
            x_test,
            device,
            epochs,
            learning_rate,
            seed=fold_idx,
        )

    valid = ~np.isnan(out_of_fold)
    if valid.sum() == 0 or len(set(labels[valid].tolist())) < 2:
        return None
    return float(roc_auc_score(labels[valid], out_of_fold[valid]))


__all__ = ["TorchBinaryProbe", "TorchDiagonalRiemannianPathProbe", "torch_train_scores", "torch_cross_validated_auroc"]
