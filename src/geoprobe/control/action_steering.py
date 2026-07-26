"""Action-steering primitives: reward/margin signs, steering-spec builder, and the point-cloud
context index/features + result-row builder. Extracted from the decision_token_action_response CLI (Phase 3)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy.spatial.distance import cdist

from geoprobe.models.interface import ResidualSteeringSpec


STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL")


def true_margin_sign(true_status: str) -> int:
    status = str(true_status).upper()
    if status == "PASS":
        return 1
    if status == "FAIL":
        return -1
    raise ValueError(f"bad true_status {true_status!r}")


def target_margin_sign(target_status: str) -> int:
    return true_margin_sign(target_status)


def margin_correct(true_status: str, value: float) -> bool:
    sign = true_margin_sign(true_status)
    return value > 0 if sign > 0 else value < 0


def desired_status_for_class(status_class: str) -> str:
    if status_class in {"false_FAIL", "honest_PASS"}:
        return "PASS"
    if status_class in {"false_PASS", "honest_FAIL"}:
        return "FAIL"
    raise ValueError(f"bad status_class {status_class!r}")


def action_reward(status_class: str, correct_before: bool, correct_after: bool) -> float:
    """Reward for a candidate action used by the learned policy script.

    Deceptive rows get +1 only if fixed. Honest rows get -1 if harmed and 0 if
    preserved.  Deceptive rows that remain wrong get 0. This avoids rewarding
    arbitrary steering on already-honest rows.
    """
    if status_class.startswith("false_"):
        return 1.0 if (not correct_before and correct_after) else 0.0
    return -1.0 if correct_before and not correct_after else 0.0


def public_direction_info(direction_info: dict | None) -> dict | None:
    if direction_info is None:
        return None
    return {
        "target_status": direction_info.get("target_status"),
        "direction_convention": direction_info.get("direction_convention"),
        "direction_levels": direction_info.get("direction_levels"),
        "n_train_points": direction_info.get("n_train_points"),
        "n_honest": direction_info.get("n_honest"),
        "n_false": direction_info.get("n_false"),
        "n_mixed_scenario_level_pairs": direction_info.get("n_mixed_scenario_level_pairs"),
    }


def make_spec(layer: int, direction: np.ndarray, alpha: float) -> ResidualSteeringSpec:
    clean = np.nan_to_num(np.asarray(direction, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return ResidualSteeringSpec(
        layer=layer,
        direction=torch.tensor(clean, dtype=torch.float32),
        alpha=float(alpha),
    )


def projection_features(projection: dict | None) -> dict:
    if not projection:
        return {
            "projection_fraction": None,
            "cos_to_raw": None,
            "neighbor_distance_mean": None,
            "neighbor_distance_max": None,
        }
    return {
        "projection_fraction": projection.get("projection_fraction"),
        "cos_to_raw": projection.get("cos_to_raw"),
        "neighbor_distance_mean": projection.get("neighbor_distance_mean"),
        "neighbor_distance_max": projection.get("neighbor_distance_max"),
    }


class PointcloudContextIndex:
    """Pre-built activation matrix for fast kNN context lookups.

    Builds the training-set numpy array and centroid cache ONCE per
    (layer, heldout_family) pair so that per-query lookups are just
    a matrix-vector distance computation + argsort.
    """

    def __init__(
        self,
        region_rows: list[dict],
        *,
        heldout_family: str | None = None,
        k: int,
    ) -> None:
        train = [
            row for row in region_rows
            if (heldout_family is None or str(row.get("family")) != heldout_family)
            and row.get("status_class") in STATUS_CLASSES
        ]
        self.k = int(k)
        self.n_train = len(train)
        self.heldout_family = heldout_family
        if train:
            self.xs = np.vstack([
                np.nan_to_num(np.asarray(row["x"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                for row in train
            ])
            self.labels = [str(row["status_class"]) for row in train]
            self.centroids: dict[str, np.ndarray] = {}
            for cls in STATUS_CLASSES:
                idx = [i for i, label in enumerate(self.labels) if label == cls]
                if idx:
                    self.centroids[cls] = self.xs[idx].mean(axis=0)
        else:
            self.xs = None
            self.labels = []
            self.centroids = {}

    def _empty_features(self) -> dict:
        out: dict[str, float | int | None] = {"pc_n_train": self.n_train, "pc_knn_k": self.k}
        for cls in STATUS_CLASSES:
            out[f"pc_knn_frac_{cls}"] = None
            out[f"pc_centroid_dist_{cls}"] = None
        out["pc_knn_entropy"] = None
        out["pc_knn_mean_dist"] = None
        out["pc_false_frac"] = None
        out["pc_honest_frac"] = None
        return out

    def _features_from_dists(self, dists: np.ndarray, q: np.ndarray) -> dict:
        """Build the context-feature dict from precomputed train distances.

        ``dists`` is the distance from one query point to every train point (same
        ordering as ``self.xs``); ``q`` is that query (for centroid distances).
        Shared by query() and query_batch() so the two are identical by
        construction — only the distance computation differs upstream.
        """
        out = self._empty_features()
        if self.xs is None:
            return out
        order = np.argsort(dists)
        kk = min(self.k, len(order))
        neigh = order[:kk]
        counts = Counter(self.labels[i] for i in neigh)
        probs = []
        out["pc_knn_mean_dist"] = float(np.mean(dists[neigh])) if kk else None
        for cls in STATUS_CLASSES:
            frac = float(counts.get(cls, 0) / kk) if kk else 0.0
            out[f"pc_knn_frac_{cls}"] = frac
            if frac > 0:
                probs.append(frac)
            if cls in self.centroids:
                out[f"pc_centroid_dist_{cls}"] = float(np.linalg.norm(self.centroids[cls] - q))
        out["pc_knn_entropy"] = float(-sum(p * np.log(p + 1e-12) for p in probs)) if probs else 0.0
        out["pc_false_frac"] = float(out["pc_knn_frac_false_FAIL"] + out["pc_knn_frac_false_PASS"])
        out["pc_honest_frac"] = float(out["pc_knn_frac_honest_PASS"] + out["pc_knn_frac_honest_FAIL"])
        return out

    def query(self, query_x: np.ndarray) -> dict:
        """Compute context features for one query point."""
        if self.xs is None:
            return self._empty_features()
        q = np.nan_to_num(np.asarray(query_x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        dists = np.linalg.norm(self.xs - q[None, :], axis=1)
        return self._features_from_dists(dists, q)

    def query_batch(self, query_xs: np.ndarray) -> list[dict]:
        """Context features for many query points at once.

        Computes the full (n_query x n_train) Euclidean distance matrix in one
        ``cdist`` call instead of a per-row ``np.linalg.norm`` allocation, then
        reuses the shared per-row feature extraction. Result is identical to
        calling ``query()`` on each row (cdist matches the per-row norm to ~1e-13
        with no change in the kNN neighbor set or order).
        """
        Q = np.nan_to_num(np.asarray(query_xs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if Q.ndim == 1:
            Q = Q[None, :]
        if self.xs is None:
            return [self._empty_features() for _ in range(Q.shape[0])]
        dmat = cdist(Q, self.xs, metric="euclidean")
        return [self._features_from_dists(dmat[i], Q[i]) for i in range(Q.shape[0])]


def pointcloud_context_features(
    *,
    query_x: np.ndarray,
    region_rows: list[dict],
    heldout_family: str | None = None,
    heldout_scenario_ids: set[str] | None = None,
    k: int,
    _index: PointcloudContextIndex | None = None,
) -> dict:
    """Leak-safe local point-cloud context around a query activation.

    A neighbor is excluded if it shares the query's ``heldout_family`` (family
    split) or matches any of ``heldout_scenario_ids`` (within-scenario split).
    Either, both, or neither may be given; the exclusions AND together.

    If ``_index`` is provided AND ``heldout_scenario_ids`` is None, uses the
    pre-built index for fast lookups.
    """
    if _index is not None and heldout_scenario_ids is None:
        return _index.query(query_x)
    train = [
        row for row in region_rows
        if (heldout_family is None or str(row.get("family")) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
        and row.get("status_class") in STATUS_CLASSES
    ]
    out: dict[str, float | int | None] = {"pc_n_train": len(train), "pc_knn_k": int(k)}
    for cls in STATUS_CLASSES:
        out[f"pc_knn_frac_{cls}"] = None
        out[f"pc_centroid_dist_{cls}"] = None
    out["pc_knn_entropy"] = None
    out["pc_knn_mean_dist"] = None
    out["pc_false_frac"] = None
    out["pc_honest_frac"] = None
    if not train:
        return out

    q = np.nan_to_num(np.asarray(query_x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xs = np.vstack([
        np.nan_to_num(np.asarray(row["x"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        for row in train
    ])
    labels = [str(row["status_class"]) for row in train]
    dists = np.linalg.norm(xs - q[None, :], axis=1)
    order = np.argsort(dists)
    kk = min(int(k), len(order))
    neigh = order[:kk]
    counts = Counter(labels[i] for i in neigh)
    probs = []
    out["pc_knn_mean_dist"] = float(np.mean(dists[neigh])) if kk else None
    for cls in STATUS_CLASSES:
        frac = float(counts.get(cls, 0) / kk) if kk else 0.0
        out[f"pc_knn_frac_{cls}"] = frac
        if frac > 0:
            probs.append(frac)
        idx = [i for i, label in enumerate(labels) if label == cls]
        if idx:
            out[f"pc_centroid_dist_{cls}"] = float(np.linalg.norm(xs[idx].mean(axis=0) - q))
    out["pc_knn_entropy"] = float(-sum(p * np.log(p + 1e-12) for p in probs)) if probs else 0.0
    out["pc_false_frac"] = float(out["pc_knn_frac_false_FAIL"] + out["pc_knn_frac_false_PASS"])
    out["pc_honest_frac"] = float(out["pc_knn_frac_honest_PASS"] + out["pc_knn_frac_honest_FAIL"])
    return out


def build_result_row(
    *,
    row: dict,
    method: str,
    target_status: str | None,
    layer: int | None,
    alpha: float,
    base_margin: float,
    final_margin: float,
    direction_info: dict | None,
    projection: dict | None,
    route: dict | None,
    context: dict | None = None,
) -> dict:
    status_class = str(row["status_class"])
    correct_before = margin_correct(str(row["true_status"]), base_margin)
    correct_after = margin_correct(str(row["true_status"]), final_margin)
    is_false = status_class.startswith("false_")
    proj = projection_features(projection)
    return {
        "conversation_id": str(row["conversation_id"]),
        "scenario_id": str(row.get("scenario_id", "")),
        "family": str(row["family"]),
        "arm": str(row["arm"]),
        "sample_seed": row.get("sample_seed"),
        "true_status": str(row["true_status"]).upper(),
        "reported_status_before": str(row.get("reported_status", "")).upper(),
        "status_class": status_class,
        "desired_status": desired_status_for_class(status_class),
        "method": method,
        "target_status": target_status,
        "layer": layer,
        "alpha": float(alpha),
        "base_margin": float(base_margin),
        "final_margin": float(final_margin),
        "delta_margin": float(final_margin - base_margin),
        "abs_base_margin": float(abs(base_margin)),
        "correct_before": bool(correct_before),
        "correct_after": bool(correct_after),
        "fixes_error": bool(is_false and (not correct_before) and correct_after),
        "harms_honest": bool((not is_false) and correct_before and (not correct_after)),
        "reward": action_reward(status_class, correct_before, correct_after),
        "desired_margin_sign": true_margin_sign(str(row["true_status"])),
        "target_margin_sign": target_margin_sign(target_status) if target_status else 0,
        "route_action": route.get("action") if route else None,
        "gate_score_PASS_minus_FAIL": route.get("score_PASS_minus_FAIL") if route else None,
        "gate_proba_PASS": route.get("proba_PASS") if route else None,
        **(context or {}),
        "projection": projection,
        **proj,
        "direction_info": public_direction_info(direction_info),
    }


__all__ = [
    "STATUS_CLASSES",
    "true_margin_sign",
    "target_margin_sign",
    "margin_correct",
    "desired_status_for_class",
    "action_reward",
    "public_direction_info",
    "make_spec",
    "projection_features",
    "PointcloudContextIndex",
    "pointcloud_context_features",
    "build_result_row",
]
