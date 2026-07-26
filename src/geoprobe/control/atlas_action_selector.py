"""Chart/atlas action-selector: a DISTINCT scorer from geoprobe.control.chart_atlas.ChartAtlas
(different feature/row semantics + selector surface; do NOT merge/alias the two). Promoted from
the geometric_chart_atlas_selector CLI (Phase 4A). Class ChartAtlasSelector was renamed to
AtlasActionSelector; the experiment path keeps ChartAtlasSelector as a compat alias.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from geoprobe.control.policy_eval import (
    baseline_row,
    grouped_by_conversation,
    safe_float,
    slim_choice,
    summarize_choices,
)


def parse_chart_counts(value: str) -> list[int | str]:
    out: list[int | str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append("auto" if item == "auto" else int(item))
    return out


def action_key(row: dict, granularity: str) -> tuple:
    route = str(row.get("route_action") or "unknown")
    target = str(row.get("target_status") or "NONE")
    method = str(row.get("method") or "unknown")
    layer = row.get("layer")
    alpha = float(safe_float(row.get("alpha")))
    if granularity == "full":
        return (route, target, method, layer, alpha)
    if granularity == "method_layer":
        return (route, target, method, layer)
    if granularity == "method":
        return (route, target, method)
    if granularity == "route":
        return (route, target)
    raise ValueError(f"unknown action granularity {granularity!r}")


def state_key(row: dict) -> tuple[str, int] | None:
    layer = row.get("layer")
    if layer is None:
        return None
    return (str(row.get("conversation_id")), int(layer))


def action_features(row: dict, *, include_response_margin: bool) -> dict[str, str | float]:
    target = str(row.get("target_status") or "NONE")
    target_sign = 1.0 if target == "PASS" else -1.0 if target == "FAIL" else 0.0
    margin = safe_float(row.get("decision_margin"))
    out: dict[str, str | float] = {
        "method": str(row.get("method") or "unknown"),
        "route_action": str(row.get("route_action") or "unknown"),
        "target_status": target,
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "gate_score": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "abs_gate_score": abs(safe_float(row.get("gate_score_PASS_minus_FAIL"))),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
    }
    if include_response_margin:
        out.update({
            "decision_margin": margin,
            "abs_decision_margin": abs(margin),
            "target_aligned_decision_margin": target_sign * margin,
            "decision_forced_status": str(row.get("decision_forced_status") or "UNKNOWN"),
        })
    return out


def candidate_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if str(row.get("method")) != "baseline"
        and str(row.get("route_action")) != "abstain"
        and row.get("target_status") is not None
        and row.get("layer") is not None
    ]


class AtlasActionSelector:
    def __init__(
        self,
        *,
        chart_count: int | str,
        pca_dim: int,
        top_charts: int,
        ridge_alpha: float,
        include_response_margin: bool,
        objective: str,
        min_chart_support: float,
        min_action_support: float,
        fallbacks: list[str],
        head: str,
        seed: int,
    ) -> None:
        self.chart_count_arg = chart_count
        self.pca_dim_arg = int(pca_dim)
        self.top_charts = int(top_charts)
        self.ridge_alpha = float(ridge_alpha)
        self.include_response_margin = bool(include_response_margin)
        self.objective = objective
        self.min_chart_support = float(min_chart_support)
        self.min_action_support = float(min_action_support)
        self.fallbacks = fallbacks
        self.head = head
        self.seed = int(seed)
        if self.head not in {"ridge", "mean"}:
            raise ValueError(f"unknown chart head {self.head!r}")

    def fit(self, rows: list[dict], state_vectors: dict[tuple[str, int], np.ndarray]) -> "AtlasActionSelector":
        rows = candidate_rows(rows)
        rows = [row for row in rows if state_key(row) in state_vectors]
        if not rows:
            raise ValueError("no train candidate rows with activation states")
        self.rows_ = rows
        state_keys = sorted({state_key(row) for row in rows if state_key(row) is not None})
        self.train_state_keys_ = state_keys
        x = np.vstack([state_vectors[key] for key in state_keys])
        self.state_scaler_ = StandardScaler()
        x_scaled = self.state_scaler_.fit_transform(x)
        n_components = max(1, min(self.pca_dim_arg, x_scaled.shape[0] - 1, x_scaled.shape[1]))
        self.pca_ = PCA(n_components=n_components, whiten=True, random_state=self.seed)
        z_states = self.pca_.fit_transform(x_scaled)
        self.state_z_ = {key: z_states[idx] for idx, key in enumerate(state_keys)}
        n_charts = self._resolve_chart_count(len(state_keys))
        self.chart_count_ = n_charts
        if len(state_keys) <= 1:
            self.centers_ = z_states.copy()
            labels = np.zeros(len(state_keys), dtype=int)
        else:
            km = KMeans(n_clusters=n_charts, n_init=10, random_state=self.seed)
            labels = km.fit_predict(z_states)
            self.centers_ = km.cluster_centers_
        self.bandwidths_ = self._chart_bandwidths(z_states, labels)
        y = np.asarray([safe_float(row.get(self.objective)) for row in rows], dtype=np.float64)
        row_z = np.vstack([self.state_z_[state_key(row)] for row in rows])
        memberships = self.membership_from_z(row_z)
        self.models_: list[Ridge | None] = []
        self.tables_: dict[str, list[dict[tuple, dict[str, float]]]] = {
            granularity: [dict() for _ in range(n_charts)]
            for granularity in self.fallbacks
        }
        self.chart_support_: list[float] = []
        if self.head == "ridge":
            self.action_vectorizer_ = DictVectorizer(sparse=False)
            action_x = self.action_vectorizer_.fit_transform([
                action_features(row, include_response_margin=self.include_response_margin)
                for row in rows
            ])
            action_x = np.nan_to_num(action_x, nan=0.0, posinf=0.0, neginf=0.0)
            for chart_idx in range(n_charts):
                weights = memberships[:, chart_idx]
                support = float(weights.sum())
                self.chart_support_.append(support)
                if support < self.min_chart_support:
                    self.models_.append(None)
                    continue
                local_coords = row_z - self.centers_[chart_idx][None, :]
                design = np.hstack([action_x, local_coords])
                model = Ridge(alpha=self.ridge_alpha)
                model.fit(design, y, sample_weight=weights)
                self.models_.append(model)
        else:
            self.action_vectorizer_ = None
            self.models_ = [None for _ in range(n_charts)]
            self.chart_support_ = [float(memberships[:, j].sum()) for j in range(n_charts)]
            for row_idx, row in enumerate(rows):
                for chart_idx in range(n_charts):
                    weight = float(memberships[row_idx, chart_idx])
                    if weight <= 1e-8:
                        continue
                    for granularity in self.fallbacks:
                        key = action_key(row, granularity)
                        bucket = self.tables_[granularity][chart_idx].setdefault(
                            key, {"weight": 0.0, "reward": 0.0, "count": 0.0}
                        )
                        bucket["weight"] += weight
                        bucket["reward"] += weight * float(y[row_idx])
                        bucket["count"] += 1.0
        return self

    def _resolve_chart_count(self, n_states: int) -> int:
        if self.chart_count_arg == "auto":
            return max(2, min(24, int(math.ceil(math.sqrt(max(n_states, 1))))))
        return max(1, min(int(self.chart_count_arg), max(n_states, 1)))

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
        return self.pca_.transform(self.state_scaler_.transform(x))[0]

    def membership_from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        d = np.linalg.norm(z[:, None, :] - self.centers_[None, :, :], axis=2)
        weights = np.exp(-0.5 * (d / self.bandwidths_[None, :]) ** 2)
        top = max(1, min(self.top_charts, weights.shape[1]))
        if top < weights.shape[1]:
            keep = np.argpartition(weights, -top, axis=1)[:, -top:]
            mask = np.zeros_like(weights, dtype=bool)
            row_idx = np.arange(weights.shape[0])[:, None]
            mask[row_idx, keep] = True
            weights = np.where(mask, weights, 0.0)
        denom = weights.sum(axis=1, keepdims=True)
        denom[denom <= 1e-12] = 1.0
        return weights / denom

    def score_one(self, row: dict, state_vectors: dict[tuple[str, int], np.ndarray]) -> tuple[float, dict]:
        key = state_key(row)
        if key not in state_vectors:
            return -1e9, {"chart_reason": "missing_state"}
        z = self.state_to_z(state_vectors[key])
        membership = self.membership_from_z(z)[0]
        if self.head == "mean":
            return self._score_mean(row, membership)
        action_x = self.action_vectorizer_.transform([
            action_features(row, include_response_margin=self.include_response_margin)
        ])
        action_x = np.nan_to_num(action_x, nan=0.0, posinf=0.0, neginf=0.0)
        preds = []
        weights = []
        chart_meta = []
        for chart_idx, model in enumerate(self.models_):
            w = float(membership[chart_idx])
            if model is None or w <= 1e-12:
                continue
            local = (z - self.centers_[chart_idx])[None, :]
            design = np.hstack([action_x, local])
            pred = float(model.predict(design)[0])
            preds.append(pred)
            weights.append(w)
            chart_meta.append({
                "chart": int(chart_idx),
                "weight": w,
                "support": float(self.chart_support_[chart_idx]),
                "prediction": pred,
            })
        if not preds:
            return -1e9, {"chart_reason": "no_supported_chart"}
        weights_arr = np.asarray(weights, dtype=np.float64)
        preds_arr = np.asarray(preds, dtype=np.float64)
        score = float(np.sum(weights_arr * preds_arr) / max(float(weights_arr.sum()), 1e-12))
        variance = float(np.sum(weights_arr * (preds_arr - score) ** 2) / max(float(weights_arr.sum()), 1e-12))
        return score, {
            "chart_top": sorted(chart_meta, key=lambda item: item["weight"], reverse=True)[:3],
            "chart_prediction_std": math.sqrt(max(variance, 0.0)),
            "chart_count": int(self.chart_count_),
            "chart_pca_dim": int(self.pca_.n_components_),
        }

    def _score_mean(self, row: dict, membership: np.ndarray) -> tuple[float, dict]:
        preds = []
        weights = []
        chart_meta = []
        for chart_idx, weight in enumerate(membership):
            if weight <= 1e-12 or self.chart_support_[chart_idx] < self.min_chart_support:
                continue
            used = None
            for granularity in self.fallbacks:
                bucket = self.tables_[granularity][chart_idx].get(action_key(row, granularity))
                if bucket and bucket["weight"] >= self.min_action_support:
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
            })
        if not preds:
            return -1e9, {"chart_reason": "no_supported_chart_action"}
        weights_arr = np.asarray(weights, dtype=np.float64)
        preds_arr = np.asarray(preds, dtype=np.float64)
        score = float(np.sum(weights_arr * preds_arr) / max(float(weights_arr.sum()), 1e-12))
        variance = float(np.sum(weights_arr * (preds_arr - score) ** 2) / max(float(weights_arr.sum()), 1e-12))
        return score, {
            "chart_top": sorted(chart_meta, key=lambda item: item["weight"], reverse=True)[:3],
            "chart_prediction_std": math.sqrt(max(variance, 0.0)),
            "chart_count": int(self.chart_count_),
            "chart_pca_dim": int(self.pca_.n_components_),
        }


def choose_chart(
    candidates: list[dict],
    selector: AtlasActionSelector,
    state_vectors: dict[tuple[str, int], np.ndarray],
    *,
    threshold: float,
) -> dict:
    if str(candidates[0].get("route_action")) == "abstain":
        return baseline_row(candidates)
    scored = []
    for row in candidates:
        if str(row.get("method")) == "baseline":
            continue
        score, meta = selector.score_one(row, state_vectors)
        scored.append((score, meta, row))
    if not scored:
        return baseline_row(candidates)
    score, meta, row = max(scored, key=lambda item: item[0])
    if float(score) <= threshold:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = float(score)
        chosen["policy_abstained_by_threshold"] = True
        chosen.update(meta)
        return chosen
    chosen = dict(row)
    chosen["policy_score"] = float(score)
    chosen["policy_abstained_by_threshold"] = False
    chosen.update(meta)
    return chosen


def evaluate_chart_atlas(
    rows: list[dict],
    *,
    state_vectors: dict[tuple[str, int], np.ndarray],
    chart_count: int | str,
    pca_dim: int,
    top_charts: int,
    ridge_alpha: float,
    include_response_margin: bool,
    threshold: float,
    min_chart_support: float,
    min_action_support: float,
    fallbacks: list[str],
    head: str,
    seed: int,
) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({str(row["family"]) for row in rows})
    choices = []
    folds = {}
    for family in families:
        train = [row for row in rows if str(row["family"]) != family]
        selector = AtlasActionSelector(
            chart_count=chart_count,
            pca_dim=pca_dim,
            top_charts=top_charts,
            ridge_alpha=ridge_alpha,
            include_response_margin=include_response_margin,
            objective="strict_reward",
            min_chart_support=min_chart_support,
            min_action_support=min_action_support,
            fallbacks=fallbacks,
            head=head,
            seed=seed,
        ).fit(train, state_vectors)
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) != family:
                continue
            fold_choices.append(choose_chart(candidates, selector, state_vectors, threshold=threshold))
        choices.extend(fold_choices)
        folds[family] = {
            "summary": summarize_choices(fold_choices),
            "chart_count": int(selector.chart_count_),
            "pca_dim": int(selector.pca_.n_components_),
            "chart_support": [float(x) for x in selector.chart_support_],
        }
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def build_chart_policies(
    rows: list[dict],
    *,
    state_vectors: dict[tuple[str, int], np.ndarray],
    chart_counts: list[int | str],
    pca_dim: int,
    top_charts: int,
    ridge_alpha: float,
    threshold: float,
    min_chart_support: float,
    min_action_support: float,
    fallbacks: list[str],
    heads: list[str],
    seed: int,
) -> dict[str, dict]:
    out = {}
    for head in heads:
        for include_response_margin, mode in [(False, "context"), (True, "response")]:
            for chart_count in chart_counts:
                count_name = str(chart_count)
                name = f"chart_{head}_{mode}_c{count_name}_d{pca_dim}_strict"
                out[name] = evaluate_chart_atlas(
                    rows,
                    state_vectors=state_vectors,
                    chart_count=chart_count,
                    pca_dim=pca_dim,
                    top_charts=top_charts,
                    ridge_alpha=ridge_alpha,
                    include_response_margin=include_response_margin,
                    threshold=threshold,
                    min_chart_support=min_chart_support,
                    min_action_support=min_action_support,
                    fallbacks=fallbacks,
                    head=head,
                    seed=seed,
                )
    return out


def slim_chart_choice(row: dict) -> dict:
    out = slim_choice(row)
    for key in ("policy_score", "policy_abstained_by_threshold", "chart_prediction_std", "chart_count", "chart_pca_dim"):
        if key in row:
            out[key] = row.get(key)
    if "chart_top" in row:
        out["chart_top"] = row["chart_top"]
    return out


def best_policy_name(policies: dict[str, dict]) -> str:
    def key(item):
        name, value = item
        s = value["summary"]
        return (
            s["deceptive_strict_fixes"],
            s["deceptive_status_fixes"],
            -s["honest_status_harms"],
            s["parse_success"],
            name,
        )

    return max(policies.items(), key=key)[0] if policies else ""


__all__ = [
    "parse_chart_counts", "action_key", "state_key", "action_features", "candidate_rows",
    "AtlasActionSelector", "choose_chart", "evaluate_chart_atlas", "build_chart_policies",
    "slim_chart_choice", "best_policy_name",
]
