"""Train-fold dose-response models over measured alpha grids."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoprobe.control.action_response import safe_float


FEATURE_NAMES = (
    "target_is_pass",
    "route_to_pass",
    "route_to_fail",
    "reported_pass_before",
    "reported_fail_before",
    "base_margin",
    "signed_base_margin",
    "required_margin_gain",
    "abs_base_margin",
    "gate_score_PASS_minus_FAIL",
    "gate_proba_PASS",
    "pc_predicted_flow_norm",
    "pc_n_train",
    "pc_neighbor_distance_mean",
    "pc_neighbor_distance_min",
    "pc_neighbor_distance_max",
    "pc_boundary_directional_slope",
    "pc_boundary_alpha",
    "pc_boundary_alpha_raw",
    "pc_boundary_positive_slope",
    "pc_causal_alpha",
    "pc_causal_predicted_gain",
    "pc_causal_required_margin_gain",
)


@dataclass(frozen=True)
class DoseCurve:
    key: tuple[str, int, str, str]
    conversation_id: str
    family: str
    scenario_id: str
    status_class: str
    target_status: str
    route_action: str
    base_margin: float
    rows_by_alpha: dict[float, dict]
    abstain_row: dict | None

    @property
    def target_sign(self) -> float:
        return target_sign(self.target_status)

    @property
    def signed_base_margin(self) -> float:
        return self.target_sign * self.base_margin

    @property
    def required_margin_gain(self) -> float:
        return max(0.0 - self.signed_base_margin, 0.0)

    def numeric_alphas(self) -> list[float]:
        return sorted(self.rows_by_alpha)

    def representative_row(self) -> dict:
        for alpha in self.numeric_alphas():
            return self.rows_by_alpha[alpha]
        if self.abstain_row is not None:
            return self.abstain_row
        raise ValueError(f"empty dose curve for {self.key}")


@dataclass(frozen=True)
class DenseDoseConfig:
    target: str = "stable_crossing"
    prediction_mode: str = "raw_alpha"
    target_margin: float = 0.0
    safety_quantile: float = 0.80
    ridge_alpha: float = 3.0
    max_alpha: float = 128.0
    calibration_folds: int = 0


@dataclass(frozen=True)
class DosePrediction:
    alpha: float
    raw_alpha: float
    reason: str
    anchor_alpha: float | None = None
    multiplier: float | None = None


class DenseDoseResponseModel:
    """Predict a conservative alpha from context features and dense-grid labels."""

    def __init__(self, config: DenseDoseConfig | None = None) -> None:
        self.config = config or DenseDoseConfig()

    def fit(self, curves: list[DoseCurve]) -> "DenseDoseResponseModel":
        examples, targets, grid_values = self._examples(curves)
        if not examples:
            raise ValueError("no positive-alpha dense dose examples")
        self.alpha_grid_ = sorted(float(a) for a in grid_values if 0.0 < float(a) <= float(self.config.max_alpha))
        x = np.vstack([feat for feat, _y in examples]).astype(np.float64)
        y = np.asarray([y for _feat, y in examples], dtype=np.float64)
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=float(self.config.ridge_alpha), fit_intercept=True))
        self.model_.fit(x, y)
        residual = self._calibration_residuals(curves)
        self.calibration_mode_ = "family_block_oof" if residual else "train_in_sample"
        if not residual:
            train_pred = np.asarray(self.model_.predict(x), dtype=np.float64)
            residual = list(y - train_pred)
        q = min(max(float(self.config.safety_quantile), 0.0), 1.0)
        self.log_residual_quantile_ = float(np.quantile(np.asarray(residual, dtype=np.float64), q))
        self.n_train_ = int(len(examples))
        self.n_calibration_ = int(len(residual))
        self.target_alpha_summary_ = numeric_summary(targets)
        self.calibration_residual_summary_ = numeric_summary([float(x) for x in residual])
        return self

    def _examples(self, curves: list[DoseCurve]) -> tuple[list[tuple[np.ndarray, float]], list[float], set[float]]:
        examples: list[tuple[np.ndarray, float]] = []
        targets: list[float] = []
        grid_values: set[float] = set()
        for curve in curves:
            grid_values.update(curve.numeric_alphas())
            target = target_alpha(curve, self.config.target, target_margin=self.config.target_margin)
            if target is None or target <= 0.0:
                continue
            transformed = transform_target_alpha(curve, float(target), self.config.prediction_mode)
            if transformed is None:
                continue
            examples.append((feature_vector(curve), transformed))
            targets.append(float(target))
        return examples, targets, grid_values

    def predict(self, curve: DoseCurve) -> DosePrediction:
        if curve.required_margin_gain <= 0.0:
            return DosePrediction(alpha=0.0, raw_alpha=0.0, reason="already_target_side", anchor_alpha=0.0, multiplier=None)
        raw_log = self._predict_value(curve)
        calibrated = raw_log + float(self.log_residual_quantile_)
        raw_alpha = inverse_transform_alpha(curve, calibrated, self.config.prediction_mode)
        if raw_alpha is None:
            return DosePrediction(alpha=0.0, raw_alpha=0.0, reason=f"missing_{self.config.prediction_mode}_anchor")
        alpha = snap_alpha(raw_alpha, self.alpha_grid_, max_alpha=float(self.config.max_alpha))
        anchor = alpha_anchor(curve, self.config.prediction_mode)
        multiplier = None if anchor is None or anchor <= 0.0 else alpha / anchor
        return DosePrediction(
            alpha=alpha,
            raw_alpha=raw_alpha,
            reason=f"dense_dose_response_{self.config.prediction_mode}",
            anchor_alpha=anchor,
            multiplier=multiplier,
        )

    def _predict_value(self, curve: DoseCurve) -> float:
        return float(self.model_.predict(feature_vector(curve).astype(np.float64)[None, :])[0])

    def _calibration_residuals(self, curves: list[DoseCurve]) -> list[float]:
        n_folds = int(self.config.calibration_folds)
        families = sorted({curve.family for curve in curves})
        if n_folds <= 1 or len(families) <= 1:
            return []
        inner_config = DenseDoseConfig(
            target=self.config.target,
            prediction_mode=self.config.prediction_mode,
            target_margin=self.config.target_margin,
            safety_quantile=self.config.safety_quantile,
            ridge_alpha=self.config.ridge_alpha,
            max_alpha=self.config.max_alpha,
            calibration_folds=0,
        )
        residuals: list[float] = []
        for heldout_families in make_family_folds(curves, min(n_folds, len(families))):
            heldout = set(heldout_families)
            train = [curve for curve in curves if curve.family not in heldout]
            test = [curve for curve in curves if curve.family in heldout]
            if not train or not test:
                continue
            try:
                model = DenseDoseResponseModel(inner_config).fit(train)
            except ValueError:
                continue
            for curve in test:
                target = target_alpha(curve, self.config.target, target_margin=self.config.target_margin)
                if target is None or target <= 0.0:
                    continue
                transformed = transform_target_alpha(curve, float(target), self.config.prediction_mode)
                if transformed is None:
                    continue
                residuals.append(float(transformed - model._predict_value(curve)))
        return residuals


def extract_dose_curves(rows: list[dict], *, method: str = "local_control_flow") -> list[DoseCurve]:
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    abstain_by_cid: dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("conversation_id") or "")
        if str(row.get("method")) == "abstain":
            abstain_by_cid[cid] = row
            continue
        if str(row.get("method")) != method:
            continue
        if row.get("layer") is None or row.get("target_status") is None:
            continue
        alpha = finite_float(row.get("alpha"))
        if alpha is None:
            continue
        key = (cid, int(row["layer"]), str(row["method"]), str(row["target_status"]))
        grouped[key].append(row)

    curves: list[DoseCurve] = []
    for key, group in sorted(grouped.items()):
        by_alpha: dict[float, dict] = {}
        for row in group:
            alpha = finite_float(row.get("alpha"))
            if alpha is not None:
                by_alpha[float(alpha)] = row
        if not by_alpha:
            continue
        rep = by_alpha[sorted(by_alpha)[0]]
        curves.append(DoseCurve(
            key=key,
            conversation_id=str(rep.get("conversation_id")),
            family=str(rep.get("family") or "unknown"),
            scenario_id=str(rep.get("scenario_id") or "unknown"),
            status_class=str(rep.get("status_class") or "unknown"),
            target_status=str(rep.get("target_status") or "NONE"),
            route_action=str(rep.get("route_action") or "unknown"),
            base_margin=safe_float(rep.get("base_margin")),
            rows_by_alpha=by_alpha,
            abstain_row=abstain_by_cid.get(str(rep.get("conversation_id"))),
        ))
    return curves


def curve_from_context_row(row: dict, *, method: str = "local_control_flow") -> DoseCurve:
    cid = str(row.get("conversation_id") or "")
    target = str(row.get("target_status") or "NONE")
    layer = int(row.get("layer") or 0)
    rep = dict(row)
    rep.setdefault("method", method)
    rep.setdefault("alpha", 8.0)
    return DoseCurve(
        key=(cid, layer, method, target),
        conversation_id=cid,
        family=str(row.get("family") or "unknown"),
        scenario_id=str(row.get("scenario_id") or "unknown"),
        status_class=str(row.get("status_class") or "unknown"),
        target_status=target,
        route_action=str(row.get("route_action") or "unknown"),
        base_margin=safe_float(row.get("base_margin")),
        rows_by_alpha={safe_float(rep.get("alpha"), 8.0): rep},
        abstain_row=None,
    )


def target_alpha(curve: DoseCurve, target: str, *, target_margin: float = 0.0) -> float | None:
    if curve.required_margin_gain <= 0.0:
        return 0.0
    if target == "first_crossing":
        return first_crossing_alpha(curve, target_margin=target_margin)
    if target == "stable_crossing":
        return stable_crossing_alpha(curve, target_margin=target_margin)
    if target == "best_reward":
        return best_reward_alpha(curve)
    raise ValueError(f"unknown dense dose target {target!r}")


def first_crossing_alpha(curve: DoseCurve, *, target_margin: float = 0.0) -> float | None:
    for alpha in curve.numeric_alphas():
        row = curve.rows_by_alpha[alpha]
        if signed_final_margin(row) >= float(target_margin):
            return float(alpha)
    return None


def stable_crossing_alpha(curve: DoseCurve, *, target_margin: float = 0.0) -> float | None:
    alphas = curve.numeric_alphas()
    ok = [signed_final_margin(curve.rows_by_alpha[alpha]) >= float(target_margin) for alpha in alphas]
    for idx, alpha in enumerate(alphas):
        if all(ok[idx:]):
            return float(alpha)
    return None


def best_reward_alpha(curve: DoseCurve) -> float | None:
    if not curve.rows_by_alpha:
        return None
    row = max(curve.rows_by_alpha.values(), key=lambda item: (safe_float(item.get("reward")), -safe_float(item.get("alpha"))))
    return finite_float(row.get("alpha"))


def transform_target_alpha(curve: DoseCurve, target_alpha_value: float, prediction_mode: str) -> float | None:
    if target_alpha_value <= 0.0:
        return None
    if prediction_mode == "raw_alpha":
        return math.log(float(target_alpha_value))
    anchor = alpha_anchor(curve, prediction_mode)
    if anchor is None or anchor <= 0.0:
        return None
    return math.log(float(target_alpha_value) / float(anchor))


def inverse_transform_alpha(curve: DoseCurve, predicted_value: float, prediction_mode: str) -> float | None:
    if prediction_mode == "raw_alpha":
        return float(math.exp(predicted_value))
    anchor = alpha_anchor(curve, prediction_mode)
    if anchor is None or anchor <= 0.0:
        return None
    return float(anchor) * float(math.exp(predicted_value))


def alpha_anchor(curve: DoseCurve, prediction_mode: str) -> float | None:
    if prediction_mode == "raw_alpha":
        return 1.0
    if prediction_mode == "boundary_ratio":
        return first_positive_feature(
            curve,
            "pc_boundary_alpha",
            "pc_causal_alpha",
            "pc_boundary_alpha_raw",
            "pc_causal_alpha_raw",
        )
    if prediction_mode == "causal_ratio":
        return first_positive_feature(
            curve,
            "pc_causal_alpha",
            "pc_boundary_alpha",
            "pc_causal_alpha_raw",
            "pc_boundary_alpha_raw",
        )
    raise ValueError(f"unknown dense dose prediction_mode {prediction_mode!r}")


def first_positive_feature(curve: DoseCurve, *keys: str) -> float | None:
    row = curve.representative_row()
    for key in keys:
        value = feature_float(row, key, -1.0)
        if value > 0.0 and math.isfinite(value):
            return float(value)
    return None


def feature_vector(curve: DoseCurve) -> np.ndarray:
    row = curve.representative_row()
    target = curve.target_status
    route = curve.route_action
    reported = str(row.get("reported_status_before") or "UNKNOWN")
    boundary_slope = feature_float(row, "pc_boundary_directional_slope", 0.0)
    values = [
        1.0 if target == "PASS" else 0.0,
        1.0 if route == "steer_to_PASS" else 0.0,
        1.0 if route == "steer_to_FAIL" else 0.0,
        1.0 if reported == "PASS" else 0.0,
        1.0 if reported == "FAIL" else 0.0,
        curve.base_margin,
        curve.signed_base_margin,
        curve.required_margin_gain,
        abs(curve.base_margin),
        feature_float(row, "gate_score_PASS_minus_FAIL", 0.0),
        feature_float(row, "gate_proba_PASS", 0.5),
        feature_float(row, "pc_predicted_flow_norm", 0.0),
        feature_float(row, "pc_n_train", 0.0),
        feature_float(row, "pc_neighbor_distance_mean", -1.0),
        feature_float(row, "pc_neighbor_distance_min", -1.0),
        feature_float(row, "pc_neighbor_distance_max", -1.0),
        boundary_slope,
        feature_float(row, "pc_boundary_alpha", -1.0),
        feature_float(row, "pc_boundary_alpha_raw", -1.0),
        1.0 if boundary_slope > 0.0 else 0.0,
        feature_float(row, "pc_causal_alpha", -1.0),
        feature_float(row, "pc_causal_predicted_gain", -1.0),
        feature_float(row, "pc_causal_required_margin_gain", -1.0),
    ]
    return np.asarray(values, dtype=np.float32)


def evaluate_predictions(curves: list[DoseCurve], predictions: dict[tuple[str, int, str, str], DosePrediction]) -> dict:
    rows = []
    for curve in curves:
        prediction = predictions.get(curve.key)
        if prediction is None:
            continue
        rows.append(row_for_alpha(curve, prediction.alpha))
    return summarize_rows(rows)


def evaluate_fixed_alpha(curves: list[DoseCurve], alpha: float) -> dict:
    return summarize_rows([row_for_alpha(curve, float(alpha)) for curve in curves])


def summarize_rows(rows: list[dict]) -> dict:
    by_status: dict[str, dict[str, int]] = {}
    for status in sorted({str(row.get("status_class")) for row in rows}):
        status_rows = [row for row in rows if str(row.get("status_class")) == status]
        by_status[status] = {
            "rows": len(status_rows),
            "fixes_error": int(sum(bool(row.get("fixes_error")) for row in status_rows)),
            "harms_honest": int(sum(bool(row.get("harms_honest")) for row in status_rows)),
        }
    false_rows = [row for row in rows if str(row.get("status_class", "")).startswith("false")]
    honest_rows = [row for row in rows if str(row.get("status_class", "")).startswith("honest")]
    alphas = [safe_float(row.get("alpha")) for row in rows if str(row.get("method")) != "abstain"]
    return {
        "rows": len(rows),
        "false_rows": len(false_rows),
        "honest_rows": len(honest_rows),
        "fixes_error": int(sum(bool(row.get("fixes_error")) for row in false_rows)),
        "harms_honest": int(sum(bool(row.get("harms_honest")) for row in honest_rows)),
        "reward_sum": float(sum(safe_float(row.get("reward")) for row in rows)),
        "reward_per_row": float(sum(safe_float(row.get("reward")) for row in rows)) / max(len(rows), 1),
        "by_status_class": by_status,
        "alpha": numeric_summary(alphas),
    }


def row_for_alpha(curve: DoseCurve, alpha: float) -> dict:
    if alpha <= 0.0 and curve.abstain_row is not None:
        row = dict(curve.abstain_row)
        row["selected_alpha"] = 0.0
        return row
    alpha = snap_alpha(alpha, curve.numeric_alphas(), max_alpha=max(curve.numeric_alphas()))
    row = dict(curve.rows_by_alpha[alpha])
    row["selected_alpha"] = float(alpha)
    return row


def snap_alpha(alpha: float, grid: list[float], *, max_alpha: float) -> float:
    clean = sorted(float(a) for a in grid if 0.0 < float(a) <= float(max_alpha))
    if not clean:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    for candidate in clean:
        if candidate + 1e-8 >= float(alpha):
            return float(candidate)
    return float(clean[-1])


def make_family_folds(curves: list[DoseCurve], n_folds: int) -> list[list[str]]:
    counts = Counter(curve.family for curve in curves)
    folds: list[list[str]] = [[] for _ in range(max(1, min(int(n_folds), len(counts))))]
    sizes = [0 for _ in folds]
    for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        idx = min(range(len(folds)), key=lambda i: (sizes[i], i))
        folds[idx].append(family)
        sizes[idx] += int(count)
    return [sorted(fold) for fold in folds]


def signed_final_margin(row: dict) -> float:
    return target_sign(str(row.get("target_status") or "")) * safe_float(row.get("final_margin"))


def target_sign(target_status: str) -> float:
    if target_status == "PASS":
        return 1.0
    if target_status == "FAIL":
        return -1.0
    return 0.0


def feature_float(row: dict, key: str, default: float) -> float:
    value = finite_float(row.get(key))
    if value is not None:
        return value
    context = row.get("context")
    if isinstance(context, dict):
        value = finite_float(context.get(key))
        if value is not None:
            return value
    return float(default)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "q1": None, "q3": None, "max": None}
    quartiles = statistics.quantiles(clean, n=4) if len(clean) >= 4 else [None, None, None]
    return {
        "count": len(clean),
        "min": min(clean),
        "q1": quartiles[0],
        "median": statistics.median(clean),
        "q3": quartiles[2],
        "mean": statistics.mean(clean),
        "max": max(clean),
    }
