"""Dataset-agnostic control-map row helpers.

These helpers define the small row protocol used by reusable learned geometry
selectors. Dataset-specific scripts may keep extra fields, but selectors should
only rely on the accessors here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from geoprobe.control.action_response import aligned_delta_margin, aligned_margin, safe_float

ActionKey = tuple[str, str | None, int | None, float | None]
StateLayerKey = tuple[str, int]


def state_id(row: dict) -> str:
    return str(row.get("state_id") or row.get("conversation_id"))


def split_group(row: dict) -> str:
    return str(row.get("split_group") or row.get("family") or "unknown")


def scenario_id(row: dict) -> str:
    return str(row.get("scenario_id") or state_id(row))


def is_baseline(row: dict) -> bool:
    return str(row.get("method")) in {"abstain", "baseline"}


def state_layer_key(row: dict) -> StateLayerKey | None:
    layer = row.get("layer")
    if layer is None or is_baseline(row):
        return None
    return (state_id(row), int(layer))


def action_key(row: dict, granularity: str = "full") -> tuple:
    route = str(row.get("route_action") or "unknown")
    target = str(row.get("target_status") or "NONE")
    method = str(row.get("method") or "unknown")
    layer = None if row.get("layer") is None else int(row["layer"])
    alpha = None if row.get("alpha") is None else float(row["alpha"])
    if granularity == "full":
        return (route, target, method, layer, alpha)
    if granularity == "method_layer":
        return (route, target, method, layer)
    if granularity == "method":
        return (route, target, method)
    if granularity == "route":
        return (route, target)
    raise ValueError(f"unknown action granularity {granularity!r}")


def compact_action_key(row: dict) -> ActionKey:
    layer = None if row.get("layer") is None else int(row["layer"])
    alpha = None if row.get("alpha") is None else float(row["alpha"])
    target = None if row.get("target_status") is None else str(row.get("target_status"))
    return (str(row.get("method") or ""), target, layer, alpha)


def z2_partner_key(row: dict) -> ActionKey | None:
    method, target, layer, alpha = compact_action_key(row)
    if target == "PASS":
        return (method, "FAIL", layer, alpha)
    if target == "FAIL":
        return (method, "PASS", layer, alpha)
    return None


def candidate_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if not is_baseline(row)
        and row.get("target_status") in {"PASS", "FAIL"}
        and row.get("layer") is not None
    ]


def grouped_by_state(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[state_id(row)].append(row)
    return dict(out)


def baseline_row(candidates: list[dict]) -> dict:
    for row in candidates:
        if is_baseline(row):
            return row
    raise ValueError("candidate group has no baseline/abstain row")


def target_value(row: dict, objective: str) -> float:
    if objective in {"reward", "strict_reward", "status_reward"}:
        return safe_float(row.get(objective))
    if objective == "aligned_margin":
        return aligned_margin(row)
    if objective == "aligned_delta_margin":
        return aligned_delta_margin(row)
    if objective == "fix_minus_harm":
        return float(bool(row.get("fixes_error") or row.get("fixes_strict"))) - float(
            bool(row.get("harms_honest") or row.get("harms_honest_strict"))
        )
    raise ValueError(f"unknown objective {objective!r}")


def fix_value(row: dict) -> float:
    return float(bool(row.get("fixes_error") or row.get("fixes_strict")))


def harm_value(row: dict) -> float:
    return float(bool(row.get("harms_honest") or row.get("harms_honest_strict")))


def route_features(row: dict) -> dict[str, str | float]:
    out: dict[str, str | float] = {
        "route_action": str(row.get("route_action") or "unknown"),
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "base_margin": safe_float(row.get("base_margin")),
        "abs_base_margin": abs(safe_float(row.get("base_margin"))),
        "gate_score_PASS_minus_FAIL": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
    }
    # covariant transfer features: emitted only when the covariant featurizer stamped them on the
    # row, so every legacy path stays byte-identical.
    for key in ("cov_chart_top_weight", "cov_chart_entropy", "cov_local_scale_rank", "cov_log_scale_ratio"):
        if key in row:
            out[key] = safe_float(row.get(key))
    if "cov_chart_top_id" in row:
        out["cov_chart_top_id"] = str(row.get("cov_chart_top_id"))
    return out


def action_features(row: dict, *, include_response: bool) -> dict[str, str | float]:
    out: dict[str, str | float] = {
        "method": str(row.get("method") or "unknown"),
        "target_status": str(row.get("target_status") or "NONE"),
        "route_action": str(row.get("route_action") or "unknown"),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "target_margin_sign": safe_float(row.get("target_margin_sign")),
        "projection_fraction": safe_float(row.get("projection_fraction"), -1.0),
        "cos_to_raw": safe_float(row.get("cos_to_raw"), -1.0),
        "neighbor_distance_mean": safe_float(row.get("neighbor_distance_mean"), -1.0),
        "neighbor_distance_max": safe_float(row.get("neighbor_distance_max"), -1.0),
    }
    direction_info = row.get("direction_info") or {}
    if isinstance(direction_info, dict):
        out["direction_convention"] = str(direction_info.get("direction_convention") or "NONE")
        out["direction_train_points"] = safe_float(direction_info.get("n_train_points"), -1.0)
        out["direction_mixed_pairs"] = safe_float(direction_info.get("n_mixed_scenario_level_pairs"), -1.0)
    for key in sorted(row):
        if key.startswith("pc_"):
            out[key] = safe_float(row.get(key), 0.0)
    if include_response:
        out.update({
            "final_margin": safe_float(row.get("final_margin") or row.get("decision_margin")),
            "delta_margin": safe_float(row.get("delta_margin")),
            "aligned_margin": aligned_margin(row),
            "aligned_delta_margin": aligned_delta_margin(row),
            "correct_after": float(bool(row.get("correct_after") or row.get("status_correct"))),
        })
    return out


def choose_by_scores(candidates: list[dict], scores: np.ndarray, *, threshold: float) -> dict:
    if len(scores) != len(candidates) or not candidates:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = -1e9
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score <= threshold:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = best_score
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    chosen = dict(candidates[best_idx])
    chosen["policy_score"] = best_score
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def slim_choice(row: dict) -> dict:
    return {
        "conversation_id": row.get("conversation_id") or row.get("state_id"),
        "scenario_id": row.get("scenario_id"),
        "family": row.get("family") or row.get("split_group"),
        "status_class": row.get("status_class"),
        "route_action": row.get("route_action"),
        "method": row.get("method"),
        "target_status": row.get("target_status"),
        "layer": row.get("layer"),
        "alpha": row.get("alpha"),
        "reward": row.get("reward") if "reward" in row else row.get("strict_reward"),
        "fixes_error": row.get("fixes_error") if "fixes_error" in row else row.get("fixes_strict"),
        "harms_honest": row.get("harms_honest") if "harms_honest" in row else row.get("harms_honest_strict"),
        "policy_score": row.get("policy_score"),
        "policy_abstained_by_threshold": row.get("policy_abstained_by_threshold"),
        "explanation": row.get("explanation"),
    }


def summarize_choices(choices: list[dict]) -> dict[str, Any]:
    deceptive = [row for row in choices if str(row.get("status_class", "")).startswith("false_")]
    honest = [row for row in choices if str(row.get("status_class", "")).startswith("honest_")]
    rewards = [
        safe_float(row.get("reward") if "reward" in row else row.get("strict_reward"))
        for row in choices
    ]
    return {
        "n": len(choices),
        "deceptive_n": len(deceptive),
        "honest_n": len(honest),
        "fixes_error": int(sum(fix_value(row) for row in deceptive)),
        "honest_harms": int(sum(harm_value(row) for row in honest)),
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_aligned_margin": float(np.mean([aligned_margin(row) for row in choices])) if choices else None,
        "chosen_methods": dict(Counter(str(row.get("method")) for row in choices)),
    }
