"""Covariant calibration / re-expression layer: the trained recipe travels, not the frozen numbers.

Naming note (claim discipline): z-affine moment matching is a CALIBRATION transport, not a
mathematically complete equivariant map. Say "covariant calibration/re-expression", never "full
equivariance".

Source-trained structures (charts, fields, route boundaries) are re-expressed on a target dataset's
own activation map instead of pushing raw target states through source-frozen coordinates:

- ``ZFrameAligner``: per-dim affine recalibration in a frozen projection frame (z-space), fit from
  UNLABELED activations only. Identity within-domain by construction.
- ``AmbientDiagonalAligner``: diagonal moment-matching in ambient space (pre-map for estimators whose
  scaler lives in ambient coordinates, e.g. the source honesty route boundary).
- ``CovariantRouteFeaturizer``: the covariance contract — the SAME feature definitions computed on
  both domains, each calibrated against its own unlabeled pool (ECDF-rank margins, chart membership
  in the aligned frame, kNN local-scale ranks). No fabricated constants, no OOV categoricals.
- Abstention thresholds: pure functions of SOURCE data (self-distance / winning-score percentiles),
  frozen before any target outcome is observed. No target tuning anywhere.

Zero-shot contract: target-side inputs are unlabeled activations only. Anything that consumes target
labels (e.g. EcbHonestyBoundaryDoseEstimator) stays in explicitly declared target_calibrated arms.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class CovariantTransferConfig:
    mode: str = "z_affine"  # off | z_affine | ambient_diag | ambient_z
    scale_clip: tuple[float, float] = (0.1, 10.0)
    coverage_percentile: float = 95.0
    score_percentile: float = 10.0
    scale_k_neighbors: int = 12
    holdout_family_fraction: float = 0.2
    seed: int = 20260620

    def __post_init__(self) -> None:
        if self.mode not in {"off", "z_affine", "ambient_diag", "ambient_z"}:
            raise ValueError(f"unknown covariant mode {self.mode!r}")


class ZFrameAligner:
    """Affine recalibration z' = (z - mu_t)/sigma_t * sigma_s + mu_s in a frozen projection frame.

    Fit from unlabeled source/target z-clouds. When source == target the map is the identity, which
    gives the within-domain sanity check for free.
    """

    @staticmethod
    def _robust_moments(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Median + scaled MAD (sigma-consistent). Robust so OOD rows in the target pool cannot
        poison the alignment — those rows are exactly what the coverage gate must later reject."""
        mu = np.median(z, axis=0)
        mad = np.median(np.abs(z - mu[None, :]), axis=0)
        sigma = 1.4826 * mad
        fallback = z.std(axis=0)
        sigma = np.where(sigma <= 1e-12, fallback, sigma)
        return mu, sigma

    def fit(self, source_z: np.ndarray, target_z: np.ndarray, *, scale_clip: tuple[float, float] = (0.1, 10.0)) -> "ZFrameAligner":
        source_z = np.asarray(source_z, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        if source_z.ndim != 2 or target_z.ndim != 2 or source_z.shape[1] != target_z.shape[1]:
            raise ValueError("source_z/target_z must be 2-D with matching width")
        mu_s, sigma_s = self._robust_moments(source_z)
        mu_t, sigma_t = self._robust_moments(target_z)
        sigma_t = np.where(sigma_t <= 1e-12, 1.0, sigma_t)
        sigma_s = np.where(sigma_s <= 1e-12, 1.0, sigma_s)
        ratio = sigma_s / sigma_t
        lo, hi = float(scale_clip[0]), float(scale_clip[1])
        clipped = np.clip(ratio, lo, hi)
        self.clip_hits_ = int(np.sum(ratio != clipped))
        self.scale_ = clipped
        self.shift_ = mu_s - mu_t * clipped
        self.n_source_, self.n_target_, self.dim_ = len(source_z), len(target_z), source_z.shape[1]
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return z * self.scale_ + self.shift_

    def summary(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim_),
            "n_source": int(self.n_source_),
            "n_target": int(self.n_target_),
            "clip_hits": int(self.clip_hits_),
            "clip_hit_fraction": float(self.clip_hits_ / max(self.dim_, 1)),
            "scale_median": float(np.median(self.scale_)),
            "shift_abs_median": float(np.median(np.abs(self.shift_))),
        }

    def degenerate(self, *, max_clip_fraction: float = 0.25) -> bool:
        return (self.clip_hits_ / max(self.dim_, 1)) > max_clip_fraction


class AmbientDiagonalAligner:
    """Diagonal moment-matching in ambient activation space: x' = mu_s + (x - mu_t) * sigma_s/sigma_t."""

    def fit(self, source_x: np.ndarray, target_x: np.ndarray, *, scale_clip: tuple[float, float] = (0.1, 10.0)) -> "AmbientDiagonalAligner":
        aligner = ZFrameAligner().fit(source_x, target_x, scale_clip=scale_clip)
        self.scale_, self.shift_ = aligner.scale_, aligner.shift_
        self.summary_ = aligner.summary()
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.scale_ + self.shift_

    def transform_state_vectors(self, states: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
        return {key: self(vec) for key, vec in states.items()}


def apply_z_postmap(obj: Any, z: np.ndarray) -> np.ndarray:
    """Shared seam hook: apply an installed covariant post-map to frozen-frame z coordinates."""
    postmap = getattr(obj, "z_postmap_", None)
    if postmap is None:
        return z
    single = z.ndim == 1
    out = postmap(z[None, :] if single else z)
    return out[0] if single else out


def knn_local_scales(vectors_by_id: dict[str, np.ndarray], *, k_neighbors: int = 12,
                     min_scale: float = 1e-6) -> dict[str, float]:
    """Median kNN distance per point over its own pool — a dataset-relative local scale.

    Standalone twin of ``ecb_lcf_transfer.local_activation_scales`` (kept dependency-free here so the
    ecb modules can import this module without a cycle).
    """
    ids = sorted(vectors_by_id)
    if not ids:
        return {}
    pool = np.vstack([np.asarray(vectors_by_id[cid], dtype=np.float64) for cid in ids])
    center = np.median(pool, axis=0)
    fallback = float(np.median(np.linalg.norm(pool - center[None, :], axis=1)))
    if not np.isfinite(fallback) or fallback <= min_scale:
        fallback = 1.0
    if len(ids) < 2:
        return {cid: fallback for cid in ids}
    n_neighbors = min(max(int(k_neighbors), 1) + 1, len(ids))
    model = NearestNeighbors(metric="euclidean").fit(pool)
    distances, _ = model.kneighbors(pool, n_neighbors=n_neighbors)
    out = {}
    for idx, cid in enumerate(ids):
        d = distances[idx, 1:]  # exclude self
        scale = float(np.median(d)) if d.size else fallback
        out[cid] = scale if (np.isfinite(scale) and scale > min_scale) else fallback
    return out


def source_self_distance_threshold(z_pool: np.ndarray, *, k: int, percentile: float) -> float:
    """P{percentile} of source exclude-self k-th-neighbor distances: 'answer only where the query is
    as in-support as source points were'."""
    z_pool = np.asarray(z_pool, dtype=np.float64)
    if len(z_pool) < 3:
        return float("inf")
    n_neighbors = min(int(k) + 1, len(z_pool))
    model = NearestNeighbors(metric="euclidean").fit(z_pool)
    distances, _ = model.kneighbors(z_pool, n_neighbors=n_neighbors)
    kth = distances[:, -1]  # k-th excluding self (column 0 is self)
    return float(np.percentile(kth, percentile))


def ecdf_rank(sorted_pool: np.ndarray, value: float) -> float:
    """Fraction of the pool <= value (own-domain ECDF; invariant to monotone miscalibration)."""
    if sorted_pool.size == 0:
        return 0.5
    return float(np.searchsorted(sorted_pool, value, side="right") / sorted_pool.size)


@dataclass
class CovariantRouteFeaturizer:
    """The covariance contract: identical feature definitions on both domains, each calibrated
    against its own unlabeled pool. Per-cid features are computed once at fit time and merged into
    rows by state id."""

    scale_k_neighbors: int = 12
    source_features_: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_features_: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _build(self, *, margins: dict[str, float], chart_membership: dict[str, np.ndarray] | None,
               scales: dict[str, float]) -> dict[str, dict[str, Any]]:
        margin_pool = np.sort(np.asarray(list(margins.values()), dtype=np.float64))
        scale_pool = np.sort(np.asarray(list(scales.values()), dtype=np.float64))
        scale_median = float(np.median(scale_pool)) if scale_pool.size else 1.0
        out: dict[str, dict[str, Any]] = {}
        for cid, margin in margins.items():
            p = ecdf_rank(margin_pool, float(margin))
            feats: dict[str, Any] = {
                "base_margin": 2.0 * p - 1.0,
                "gate_score_PASS_minus_FAIL": 2.0 * p - 1.0,
                "gate_proba_PASS": p,
                "reported_status_before": "PASS" if p >= 0.5 else "FAIL",
            }
            scale = scales.get(cid)
            if scale is not None:
                feats["cov_local_scale_rank"] = ecdf_rank(scale_pool, float(scale))
                feats["cov_log_scale_ratio"] = float(math.log(max(float(scale), 1e-12) / max(scale_median, 1e-12)))
            member = (chart_membership or {}).get(cid)
            if member is not None:
                member = np.asarray(member, dtype=np.float64)
                top = int(np.argmax(member))
                probs = member[member > 0]
                feats["cov_chart_top_id"] = f"chart_{top}"
                feats["cov_chart_top_weight"] = float(member[top])
                feats["cov_chart_entropy"] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
            out[cid] = feats
        return out

    def fit_source(self, *, margins: dict[str, float], chart_membership: dict[str, np.ndarray] | None,
                   scales: dict[str, float]) -> "CovariantRouteFeaturizer":
        self.source_features_ = self._build(margins=margins, chart_membership=chart_membership, scales=scales)
        return self

    def fit_target(self, *, margins: dict[str, float], chart_membership: dict[str, np.ndarray] | None,
                   scales: dict[str, float]) -> "CovariantRouteFeaturizer":
        self.target_features_ = self._build(margins=margins, chart_membership=chart_membership, scales=scales)
        return self

    def apply_to_rows(self, rows: list[dict], *, domain: str) -> list[dict]:
        table = self.source_features_ if domain == "source" else self.target_features_
        out = []
        for row in rows:
            cid = str(row.get("state_id") or row.get("conversation_id") or "")
            feats = table.get(cid)
            out.append({**row, **feats} if feats else dict(row))
        return out

    def feature_summary(self, *, domain: str) -> dict[str, Any]:
        table = self.source_features_ if domain == "source" else self.target_features_
        numeric: dict[str, list[float]] = {}
        for feats in table.values():
            for key, value in feats.items():
                if isinstance(value, (int, float)):
                    numeric.setdefault(key, []).append(float(value))
        summary = {}
        for key, values in sorted(numeric.items()):
            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            summary[key] = {
                "n_unique": int(len(np.unique(np.round(arr, 12)))),
                "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else (float("inf") if std > 0 else 0.0),
                "mean": mean,
                "std": std,
            }
        return summary


def selection_report(selected: list[str]) -> dict[str, Any]:
    """Selection histogram + entropy + collapse flag for exporter manifests."""
    counts = Counter(str(item) for item in selected)
    total = sum(counts.values())
    probs = [c / total for c in counts.values() if c > 0] if total else []
    entropy = float(-sum(p * math.log2(p) for p in probs)) if probs else 0.0
    return {
        "selection_histogram": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "selection_entropy_bits": entropy,
        "collapsed": bool(total > 0 and (max(counts.values()) / total) >= 0.99),
    }


PostMap = Callable[[np.ndarray], np.ndarray]

__all__ = [
    "AmbientDiagonalAligner",
    "CovariantRouteFeaturizer",
    "CovariantTransferConfig",
    "ZFrameAligner",
    "apply_z_postmap",
    "ecdf_rank",
    "knn_local_scales",
    "selection_report",
    "source_self_distance_threshold",
]
