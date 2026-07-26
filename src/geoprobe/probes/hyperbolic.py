"""Hyperbolic logistic regression (Ganea et al. 2018) on the Poincare ball.

Built on geoopt's validated PoincareBall primitives (mobius_add, expmap0, dist),
so the manifold math is library-backed, not hand-rolled. The classifier fits one
geodesic decision hyperplane per class via Riemannian Adam.

This is a genuine non-Euclidean probe: decision boundaries are geodesics of the
Poincare ball (circular arcs in Euclidean coordinates), not flat hyperplanes.

Pipeline (sklearn-compatible): StandardScaler -> PCA(dim) -> expmap0(scale*.)
into the ball -> hyperbolic MLR. dim/scale/c are hyperparameters.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import geoopt
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


class _HypMLR(torch.nn.Module if _HAVE else object):
    def __init__(self, dim: int, n_classes: int, ball):
        super().__init__()
        self.ball = ball
        self.p = geoopt.ManifoldParameter(
            ball.random_normal(n_classes, dim, std=1e-2), manifold=ball
        )
        self.a = torch.nn.Parameter(torch.randn(n_classes, dim) * 1e-2)

    def forward(self, x):  # x: (B, dim) points already in the ball
        c = self.ball.c
        sqrt_c = c.sqrt()
        logits = []
        for k in range(self.p.shape[0]):
            pk = self.p[k:k + 1]
            ak = self.a[k:k + 1]
            mob = self.ball.mobius_add(-pk, x)                      # (B,dim)
            anorm = ak.norm(dim=-1).clamp_min(1e-15)
            num = 2 * sqrt_c * (mob * ak).sum(-1)
            den = (1 - c * mob.pow(2).sum(-1)).clamp_min(1e-15) * anorm
            lam = 2 / (1 - c * pk.pow(2).sum(-1)).clamp_min(1e-15)
            logits.append(lam * anorm / sqrt_c * torch.asinh(num / den))
        return torch.stack(logits, dim=1)                          # (B, n_classes)


class HyperbolicProbe(ClassifierMixin, BaseEstimator):
    def __init__(self, dim: int = 10, c: float = 1.0, scale: float = 1.0,
                 epochs: int = 300, lr: float = 1e-2, seed: int = 0):
        self.dim = dim
        self.c = c
        self.scale = scale
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def _to_ball(self, x: np.ndarray, fit: bool):
        if fit:
            self.scaler_ = StandardScaler().fit(x)
            xs = self.scaler_.transform(x)
            self.pca_ = PCA(n_components=min(self.dim, x.shape[1]), random_state=self.seed).fit(xs)
        else:
            xs = self.scaler_.transform(x)
        z = self.pca_.transform(xs)
        # normalize scale of tangent vectors so expmap0 doesn't saturate the boundary
        if fit:
            self.tnorm_ = np.median(np.linalg.norm(z, axis=1)) + 1e-9
        z = z / self.tnorm_ * self.scale
        t = torch.tensor(z, dtype=torch.float32)
        return self.ball_.expmap0(t)

    def fit(self, x: np.ndarray, y: np.ndarray):
        if not _HAVE:
            raise RuntimeError("torch+geoopt required for HyperbolicProbe")
        torch.manual_seed(self.seed)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        yi = np.searchsorted(self.classes_, y)
        self.ball_ = geoopt.PoincareBall(c=self.c)
        xb = self._to_ball(np.asarray(x, dtype=np.float64), fit=True)
        yt = torch.tensor(yi, dtype=torch.long)
        self.model_ = _HypMLR(xb.shape[1], len(self.classes_), self.ball_)
        opt = geoopt.optim.RiemannianAdam(self.model_.parameters(), lr=self.lr)
        # class-balanced loss (mirror sklearn class_weight="balanced") so severe
        # imbalance doesn't collapse the probe to majority-class prediction.
        counts = np.array([(yi == k).sum() for k in range(len(self.classes_))], dtype=np.float64)
        w = (len(yi) / (len(self.classes_) * np.maximum(counts, 1))).astype(np.float32)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w))
        self.model_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(self.model_(xb), yt)
            loss.backward()
            opt.step()
        self.model_.eval()
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        xb = self._to_ball(np.asarray(x, dtype=np.float64), fit=False)
        with torch.no_grad():
            logit = self.model_(xb).numpy()
        return logit[:, list(self.classes_).index(1)] - logit[:, list(self.classes_).index(0)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = self._to_ball(np.asarray(x, dtype=np.float64), fit=False)
        with torch.no_grad():
            idx = self.model_(xb).argmax(dim=1).numpy()
        return self.classes_[idx]
