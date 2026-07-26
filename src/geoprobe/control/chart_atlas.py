"""Reusable local-chart atlas features for control selectors."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from geoprobe.control import geometry_map as gm


@dataclass(frozen=True)
class ChartAtlasConfig:
    chart_count: int | str = "auto"
    pca_dim: int = 12
    top_charts: int = 3
    head: str = "mean"
    ridge_alpha: float = 1.0
    min_chart_support: float = 4.0
    min_action_support: float = 2.0
    fallbacks: tuple[str, ...] = ("full", "method_layer", "method", "route")
    include_response: bool = False
    objective: str = "reward"
    seed: int = 0


class ChartAtlas:
    """Fold-local atlas of activation states plus chart/action support tables."""

    def __init__(self, config: ChartAtlasConfig | None = None) -> None:
        self.config = config or ChartAtlasConfig()
        if self.config.head not in {"mean", "ridge"}:
            raise ValueError(f"unknown chart head {self.config.head!r}")

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "ChartAtlas":
        rows = [row for row in gm.candidate_rows(rows) if gm.state_layer_key(row) in state_vectors]
        if not rows:
            raise ValueError("no train candidate rows with state vectors")
        self.rows_ = rows
        self.state_keys_ = sorted({gm.state_layer_key(row) for row in rows if gm.state_layer_key(row) is not None})
        x = np.vstack([state_vectors[key] for key in self.state_keys_])
        self.state_scaler_ = StandardScaler()
        x_scaled = self.state_scaler_.fit_transform(x)
        n_components = max(1, min(int(self.config.pca_dim), x_scaled.shape[0] - 1, x_scaled.shape[1]))
        self.pca_ = PCA(n_components=n_components, whiten=True, random_state=self.config.seed)
        z_states = self.pca_.fit_transform(x_scaled)
        self.state_z_ = {key: z_states[idx] for idx, key in enumerate(self.state_keys_)}
        self.chart_count_ = self._resolve_chart_count(len(self.state_keys_))
        if len(self.state_keys_) <= 1:
            self.centers_ = z_states.copy()
            labels = np.zeros(len(self.state_keys_), dtype=int)
        else:
            km = KMeans(n_clusters=self.chart_count_, n_init=10, random_state=self.config.seed)
            labels = km.fit_predict(z_states)
            self.centers_ = km.cluster_centers_
        self.bandwidths_ = self._chart_bandwidths(z_states, labels)
        y = np.asarray([gm.target_value(row, self.config.objective) for row in rows], dtype=np.float64)
        row_z = np.vstack([self.state_z_[gm.state_layer_key(row)] for row in rows])
        memberships = self.membership_from_z(row_z)
        self.chart_support_ = [float(memberships[:, idx].sum()) for idx in range(self.chart_count_)]
        self.tables_ = {
            granularity: [dict() for _ in range(self.chart_count_)]
            for granularity in self.config.fallbacks
        }
        self.models_: list[Ridge | None] = [None for _ in range(self.chart_count_)]
        if self.config.head == "ridge":
            self.action_vectorizer_ = DictVectorizer(sparse=False)
            action_x = self.action_vectorizer_.fit_transform([
                gm.action_features(row, include_response=self.config.include_response)
                for row in rows
            ])
            action_x = np.nan_to_num(action_x, nan=0.0, posinf=0.0, neginf=0.0)
            for chart_idx in range(self.chart_count_):
                weights = memberships[:, chart_idx]
                if float(weights.sum()) < self.config.min_chart_support:
                    continue
                local_coords = row_z - self.centers_[chart_idx][None, :]
                design = np.hstack([action_x, local_coords])
                model = Ridge(alpha=self.config.ridge_alpha)
                model.fit(design, y, sample_weight=weights)
                self.models_[chart_idx] = model
        else:
            self.action_vectorizer_ = None
            for row_idx, row in enumerate(rows):
                for chart_idx in range(self.chart_count_):
                    weight = float(memberships[row_idx, chart_idx])
                    if weight <= 1e-8:
                        continue
                    for granularity in self.config.fallbacks:
                        key = gm.action_key(row, granularity)
                        bucket = self.tables_[granularity][chart_idx].setdefault(
                            key,
                            {"weight": 0.0, "reward": 0.0, "fix": 0.0, "harm": 0.0, "count": 0.0},
                        )
                        bucket["weight"] += weight
                        bucket["reward"] += weight * float(y[row_idx])
                        bucket["fix"] += weight * gm.fix_value(row)
                        bucket["harm"] += weight * gm.harm_value(row)
                        bucket["count"] += 1.0
        return self

    def _resolve_chart_count(self, n_states: int) -> int:
        if self.config.chart_count == "auto":
            return max(2, min(24, int(math.ceil(math.sqrt(max(n_states, 1))))))
        return max(1, min(int(self.config.chart_count), max(n_states, 1)))

    def _chart_bandwidths(self, z_states: np.ndarray, labels: np.ndarray) -> np.ndarray:
        d_all = np.linalg.norm(z_states[:, None, :] - self.centers_[None, :, :], axis=2)
        global_scale = float(np.median(d_all[d_all > 1e-12])) if np.any(d_all > 1e-12) else 1.0
        bandwidths = []
        for chart_idx in range(self.centers_.shape[0]):
            assigned = d_all[labels == chart_idx, chart_idx]
            assigned = assigned[assigned > 1e-12]
            scale = float(np.median(assigned)) if assigned.size else global_scale
            bandwidths.append(max(scale, 1e-6))
        return np.asarray(bandwidths, dtype=np.float64)

    def state_to_z(self, vec: np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float64)[None, :]
        z = self.pca_.transform(self.state_scaler_.transform(x))[0]
        postmap = getattr(self, "z_postmap_", None)  # covariant transfer hook (default off)
        return z if postmap is None else postmap(z[None, :])[0]

    def membership_from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        d = np.linalg.norm(z[:, None, :] - self.centers_[None, :, :], axis=2)
        weights = np.exp(-0.5 * (d / self.bandwidths_[None, :]) ** 2)
        top = max(1, min(int(self.config.top_charts), weights.shape[1]))
        if top < weights.shape[1]:
            keep = np.argpartition(weights, -top, axis=1)[:, -top:]
            mask = np.zeros_like(weights, dtype=bool)
            row_idx = np.arange(weights.shape[0])[:, None]
            mask[row_idx, keep] = True
            weights = np.where(mask, weights, 0.0)
        denom = weights.sum(axis=1, keepdims=True)
        denom[denom <= 1e-12] = 1.0
        return weights / denom

    def features_one(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict[str, str | float]:
        score, meta = self.score_one(row, state_vectors)
        features: dict[str, str | float] = {
            "chart_score": score,
            "chart_count": float(self.chart_count_),
            "chart_pca_dim": float(self.pca_.n_components_),
            "chart_reason": str(meta.get("chart_reason") or "ok"),
            "chart_prediction_std": float(meta.get("chart_prediction_std", 0.0)),
        }
        top = meta.get("chart_top") or []
        for idx in range(3):
            item = top[idx] if idx < len(top) else {}
            features[f"chart{idx}_id"] = str(item.get("chart", "NONE"))
            features[f"chart{idx}_weight"] = float(item.get("weight", 0.0))
            features[f"chart{idx}_support"] = float(item.get("support", 0.0))
            features[f"chart{idx}_action_support"] = float(item.get("action_support", 0.0))
            features[f"chart{idx}_action_count"] = float(item.get("action_count", 0.0))
            features[f"chart{idx}_granularity"] = str(item.get("granularity", "NONE"))
            features[f"chart{idx}_prediction"] = float(item.get("prediction", 0.0))
            features[f"chart{idx}_fix_rate"] = float(item.get("fix_rate", 0.0))
            features[f"chart{idx}_harm_rate"] = float(item.get("harm_rate", 0.0))
        return features

    def score_one(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> tuple[float, dict]:
        key = gm.state_layer_key(row)
        if key not in state_vectors:
            return -1e9, {"chart_reason": "missing_state"}
        z = self.state_to_z(state_vectors[key])
        membership = self.membership_from_z(z)[0]
        if self.config.head == "mean":
            return self._score_mean(row, membership)
        return self._score_ridge(row, z, membership)

    def _score_ridge(self, row: dict, z: np.ndarray, membership: np.ndarray) -> tuple[float, dict]:
        action_x = self.action_vectorizer_.transform([gm.action_features(row, include_response=self.config.include_response)])
        action_x = np.nan_to_num(action_x, nan=0.0, posinf=0.0, neginf=0.0)
        preds = []
        weights = []
        chart_meta = []
        for chart_idx, model in enumerate(self.models_):
            weight = float(membership[chart_idx])
            if model is None or weight <= 1e-12:
                continue
            local = (z - self.centers_[chart_idx])[None, :]
            pred = float(model.predict(np.hstack([action_x, local]))[0])
            preds.append(pred)
            weights.append(weight)
            chart_meta.append({
                "chart": int(chart_idx),
                "weight": weight,
                "support": float(self.chart_support_[chart_idx]),
                "prediction": pred,
            })
        if not preds:
            return -1e9, {"chart_reason": "no_supported_chart"}
        return self._weighted_score(preds, weights, chart_meta)

    def _score_mean(self, row: dict, membership: np.ndarray) -> tuple[float, dict]:
        preds = []
        weights = []
        chart_meta = []
        for chart_idx, weight in enumerate(membership):
            if weight <= 1e-12 or self.chart_support_[chart_idx] < self.config.min_chart_support:
                continue
            used = None
            for granularity in self.config.fallbacks:
                bucket = self.tables_[granularity][chart_idx].get(gm.action_key(row, granularity))
                if bucket and bucket["weight"] >= self.config.min_action_support:
                    used = (granularity, bucket)
                    break
            if used is None:
                continue
            granularity, bucket = used
            pred = float(bucket["reward"] / max(bucket["weight"], 1e-12))
            preds.append(pred)
            weights.append(float(weight))
            chart_meta.append({
                "chart": int(chart_idx),
                "weight": float(weight),
                "support": float(self.chart_support_[chart_idx]),
                "action_support": float(bucket["weight"]),
                "action_count": int(bucket["count"]),
                "granularity": granularity,
                "prediction": pred,
                "fix_rate": float(bucket["fix"] / max(bucket["weight"], 1e-12)),
                "harm_rate": float(bucket["harm"] / max(bucket["weight"], 1e-12)),
            })
        if not preds:
            return -1e9, {"chart_reason": "no_supported_chart_action"}
        return self._weighted_score(preds, weights, chart_meta)

    @staticmethod
    def _weighted_score(preds: list[float], weights: list[float], chart_meta: list[dict]) -> tuple[float, dict]:
        weights_arr = np.asarray(weights, dtype=np.float64)
        preds_arr = np.asarray(preds, dtype=np.float64)
        score = float(np.sum(weights_arr * preds_arr) / max(float(weights_arr.sum()), 1e-12))
        variance = float(np.sum(weights_arr * (preds_arr - score) ** 2) / max(float(weights_arr.sum()), 1e-12))
        return score, {
            "chart_top": sorted(chart_meta, key=lambda item: item["weight"], reverse=True)[:3],
            "chart_prediction_std": math.sqrt(max(variance, 0.0)),
        }
