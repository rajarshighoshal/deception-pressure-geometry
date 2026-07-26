"""Learn/evaluate action selectors over audited decision-token generation rows.

Reusable library extracted from the learned_generation_action_policy CLI (Phase 2); that
module re-exports these symbols and keeps the thin CLI. No model is run here.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from geoprobe.control.action_response import safe_float
from geoprobe.io import file_sha256


STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL")


def row_key(row: dict) -> tuple[str, str]:
    return (str(row["conversation_id"]), str(row["method"]))


def merge_rows(results_payload: dict, audit_payload: dict, action_response_rows: list[dict] | None = None) -> list[dict]:
    results = results_payload.get("results", [])
    audit_rows = audit_payload.get("rows", [])
    result_by_key = {row_key(row): row for row in results}
    context_by_cid_layer: dict[tuple[str, int], dict] = {}
    if action_response_rows is not None:
        for ar in action_response_rows:
            cid = str(ar.get("conversation_id", ""))
            layer = ar.get("layer")
            if cid and layer is not None:
                pc_features = {k: v for k, v in ar.items() if k.startswith("pc_")}
                if pc_features:
                    context_by_cid_layer[(cid, int(layer))] = pc_features
    rows = []
    for audit in audit_rows:
        key = row_key(audit)
        result = result_by_key.get(key)
        if result is None:
            raise ValueError(f"audit row has no matching result row: {key}")
        route = result.get("route") or {}
        decision = result.get("decision") or {}
        injection = result.get("injection") or {}
        projection = result.get("direction_projection") or {}
        projection_by_layer = projection.get("by_layer") or {}
        layer = None
        layers = injection.get("layers")
        if layers:
            layer = int(layers[0])
        elif result.get("method") == "baseline":
            layer = None
        by_layer_values = list(projection_by_layer.values())
        first_projection = by_layer_values[0] if by_layer_values else {}
        status_class = str(audit["status_class_before"])
        is_false = status_class.startswith("false_")
        status_correct = bool(audit["status_correct"])
        strict_ok = bool(audit["basis_strict_ok"])
        parse_success = bool(audit["parse_success"])
        rows.append({
            "conversation_id": str(audit["conversation_id"]),
            "scenario_id": str(audit.get("scenario_id", "")),
            "family": str(audit["family"]),
            "arm": str(audit.get("arm", "")),
            "sample_seed": audit.get("sample_seed"),
            "method": str(audit["method"]),
            "status_class": status_class,
            "route_action": str(audit.get("route_action") or route.get("action") or "unknown"),
            "target_status": route.get("target_status"),
            "reported_status_before": str(audit.get("reported_status_before", "")).upper(),
            "reported_status": audit.get("reported_status"),
            "true_status": str(audit["true_status"]).upper(),
            "layer": layer,
            "alpha": safe_float(result.get("alpha")),
            "gate_score_PASS_minus_FAIL": safe_float(route.get("score_PASS_minus_FAIL")),
            "gate_proba_PASS": safe_float(route.get("proba_PASS"), 0.5),
            "decision_margin": safe_float(decision.get("margin")),
            "decision_forced_status": str(decision.get("forced_status") or "UNKNOWN"),
            "projection_fraction": safe_float(projection.get("projection_fraction"), -1.0),
            "cos_to_raw": safe_float(projection.get("cos_to_raw"), -1.0),
            "mean_neighbor_distance": safe_float(first_projection.get("mean_neighbor_distance"), -1.0),
            "projected_norm": safe_float(first_projection.get("projected_norm"), -1.0),
            "tangent_dim": safe_float(first_projection.get("tangent_dim"), -1.0),
            "injected_norm": safe_float(injection.get("injected_norm")),
            "per_layer_alpha": safe_float(injection.get("per_layer_alpha")),
            "status_correct": status_correct,
            "basis_strict_ok": strict_ok,
            "parse_success": parse_success,
            "basis_quality": str(audit.get("basis_quality", "")),
            "fixes_status": bool(is_false and status_correct),
            "fixes_strict": bool(is_false and strict_ok),
            "harms_honest_status": bool((not is_false) and (not status_correct)),
            "harms_honest_strict": bool((not is_false) and (not strict_ok)),
            "status_reward": (1.0 if is_false and status_correct else -1.0 if (not is_false) and (not status_correct) else 0.0),
            "strict_reward": (1.0 if is_false and strict_ok else -1.0 if (not is_false) and (not strict_ok) else 0.0),
        })
        if context_by_cid_layer and layer is not None:
            pc_feats = context_by_cid_layer.get((str(audit["conversation_id"]), layer), {})
            if pc_feats:
                row = rows[-1]
                row.update(pc_feats)
    return rows


def grouped_by_conversation(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["conversation_id"])].append(row)
    return grouped


def baseline_row(candidates: list[dict]) -> dict:
    for row in candidates:
        if row["method"] == "baseline":
            return row
    raise ValueError("candidate group has no baseline row")


def choose_method(candidates: list[dict], method: str) -> dict:
    if candidates[0].get("route_action") == "abstain":
        return baseline_row(candidates)
    for row in candidates:
        if row["method"] == method:
            return row
    return baseline_row(candidates)


def choose_route_map(candidates: list[dict], mapping: dict[str, str]) -> dict:
    route = str(candidates[0].get("route_action"))
    method = mapping.get(route)
    if method is None:
        return baseline_row(candidates)
    return choose_method(candidates, method)


def choose_margin_argmax(candidates: list[dict], *, include_baseline: bool) -> dict:
    route = str(candidates[0].get("route_action"))
    if route == "abstain":
        return baseline_row(candidates)
    target = candidates[0].get("target_status")
    target_sign = 1.0 if target == "PASS" else -1.0 if target == "FAIL" else 0.0
    if target_sign == 0.0:
        return baseline_row(candidates)
    pool = candidates if include_baseline else [row for row in candidates if row["method"] != "baseline"]
    if not pool:
        return baseline_row(candidates)
    return max(pool, key=lambda row: target_sign * safe_float(row.get("decision_margin")))


def summarize_choices(choices: list[dict]) -> dict:
    by_status_class = {}
    for cls in STATUS_CLASSES:
        rows = [row for row in choices if row["status_class"] == cls]
        by_status_class[cls] = {
            "n": len(rows),
            "status_correct": int(sum(row["status_correct"] for row in rows)),
            "strict_ok": int(sum(row["basis_strict_ok"] for row in rows)),
            "chosen_methods": dict(Counter(row["method"] for row in rows)),
        }
    deceptive = [row for row in choices if row["status_class"].startswith("false_")]
    honest = [row for row in choices if row["status_class"].startswith("honest_")]
    return {
        "n": len(choices),
        "deceptive_n": len(deceptive),
        "honest_n": len(honest),
        "deceptive_status_fixes": int(sum(row["fixes_status"] for row in deceptive)),
        "deceptive_strict_fixes": int(sum(row["fixes_strict"] for row in deceptive)),
        "honest_status_harms": int(sum(row["harms_honest_status"] for row in honest)),
        "honest_strict_harms": int(sum(row["harms_honest_strict"] for row in honest)),
        "parse_success": int(sum(row["parse_success"] for row in choices)),
        "chosen_methods": dict(Counter(row["method"] for row in choices)),
        "by_status_class": by_status_class,
    }


def feature_dict(row: dict, *, include_response_margin: bool) -> dict[str, str | float]:
    features: dict[str, str | float] = {
        "method": row["method"],
        "route_action": row["route_action"],
        "target_status": str(row.get("target_status") or "NONE"),
        "reported_status_before": row["reported_status_before"],
        "decision_forced_status": row["decision_forced_status"] if include_response_margin else "MASKED",
        "arm": row["arm"],
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "gate_score_PASS_minus_FAIL": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "abs_gate_score": abs(safe_float(row.get("gate_score_PASS_minus_FAIL"))),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
        "projection_fraction": safe_float(row.get("projection_fraction"), -1.0),
        "cos_to_raw": safe_float(row.get("cos_to_raw"), -1.0),
        "mean_neighbor_distance": safe_float(row.get("mean_neighbor_distance"), -1.0),
        "projected_norm": safe_float(row.get("projected_norm"), -1.0),
        "tangent_dim": safe_float(row.get("tangent_dim"), -1.0),
        "injected_norm": safe_float(row.get("injected_norm")),
        "per_layer_alpha": safe_float(row.get("per_layer_alpha")),
    }
    for key in sorted(row):
        if key.startswith("pc_"):
            features[key] = safe_float(row.get(key), 0.0)
    if include_response_margin:
        features["decision_margin"] = safe_float(row.get("decision_margin"))
        features["abs_decision_margin"] = abs(safe_float(row.get("decision_margin")))
    return features


def fit_model(train_rows: list[dict], *, objective: str, model_name: str, include_response_margin: bool):
    if not train_rows:
        return None
    x = [feature_dict(row, include_response_margin=include_response_margin) for row in train_rows]
    y = np.asarray([safe_float(row[objective]) for row in train_rows], dtype=np.float64)
    if model_name == "rf":
        vectorizer = DictVectorizer(sparse=False)
        xv = vectorizer.fit_transform(x)
        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=1,
            random_state=0,
        )
        rf.fit(xv, y)
        return (vectorizer, rf)
    raise ValueError(f"unknown model {model_name}")


def predict_model(model, rows: list[dict], *, include_response_margin: bool) -> np.ndarray:
    x = [feature_dict(row, include_response_margin=include_response_margin) for row in rows]
    if isinstance(model, tuple):
        vectorizer, rf = model
        return np.asarray(rf.predict(vectorizer.transform(x)), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def choose_learned(
    candidates: list[dict],
    model,
    *,
    include_response_margin: bool,
    threshold: float,
) -> dict:
    if candidates[0].get("route_action") == "abstain":
        return baseline_row(candidates)
    if model is None:
        return baseline_row(candidates)
    scores = predict_model(model, candidates, include_response_margin=include_response_margin)
    best_idx = int(np.argmax(scores))
    if float(scores[best_idx]) <= threshold:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = float(scores[best_idx])
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    chosen = dict(candidates[best_idx])
    chosen["policy_score"] = float(scores[best_idx])
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def evaluate_fixed(rows: list[dict], name: str, chooser) -> dict:
    choices = [chooser(candidates) for candidates in grouped_by_conversation(rows).values()]
    return {"summary": summarize_choices(choices), "choices": choices}


def score_action_on_train(rows: list[dict], route_action: str, method: str, objective: str) -> float:
    sub = [row for row in rows if row["route_action"] == route_action and row["method"] == method]
    if not sub:
        return -1e9
    return float(np.mean([safe_float(row[objective]) for row in sub]))


def evaluate_train_best(rows: list[dict], *, objective: str, methods: list[str]) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({row["family"] for row in rows})
    choices = []
    folds = {}
    for family in families:
        train = [row for row in rows if row["family"] != family]
        best_by_route = {}
        for route in ("steer_to_PASS", "steer_to_FAIL"):
            scores = [(score_action_on_train(train, route, method, objective), method) for method in methods]
            best_by_route[route] = max(scores)[1]
        fold_choices = []
        for candidates in grouped.values():
            if candidates[0]["family"] != family:
                continue
            fold_choices.append(choose_route_map(candidates, best_by_route))
        choices.extend(fold_choices)
        folds[family] = {"best_by_route": best_by_route, "summary": summarize_choices(fold_choices)}
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def evaluate_learned(rows: list[dict], *, objective: str, model_name: str, include_response_margin: bool, threshold: float) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({row["family"] for row in rows})
    choices = []
    folds = {}
    for family in families:
        train = [row for row in rows if row["family"] != family]
        model = fit_model(
            train,
            objective=objective,
            model_name=model_name,
            include_response_margin=include_response_margin,
        )
        fold_choices = []
        for candidates in grouped.values():
            if candidates[0]["family"] != family:
                continue
            fold_choices.append(
                choose_learned(
                    candidates,
                    model,
                    include_response_margin=include_response_margin,
                    threshold=threshold,
                )
            )
        choices.extend(fold_choices)
        folds[family] = {"summary": summarize_choices(fold_choices)}
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def paired_gap(policy: list[dict], reference: list[dict], metric: str, *, seed: int = 0, bootstrap: int = 5000) -> dict:
    """Bootstrap CIs with clustering by scenario_id.

    The powered160 design has N scenario-level repeats per case, so conversation_ids
    are not independent units. The bootstrap resamples entire scenarios rather than
    individual rows.
    """
    by_pol = {row["conversation_id"]: row for row in policy}
    by_ref = {row["conversation_id"]: row for row in reference}
    ids = sorted(set(by_pol) & set(by_ref))

    def _diff(cid: str) -> float:
        pol = by_pol[cid]
        ref = by_ref[cid]
        if metric == "status_fix":
            return float(pol["fixes_status"]) - float(ref["fixes_status"])
        elif metric == "strict_fix":
            return float(pol["fixes_strict"]) - float(ref["fixes_strict"])
        elif metric == "honest_status_harm":
            return float(pol["harms_honest_status"]) - float(ref["harms_honest_status"])
        raise ValueError(metric)

    if metric in ("status_fix", "strict_fix"):
        ids = [cid for cid in ids if by_pol[cid]["status_class"].startswith("false_")]
    elif metric == "honest_status_harm":
        ids = [cid for cid in ids if by_pol[cid]["status_class"].startswith("honest_")]
    if len(ids) == 0:
        return {"n": 0, "point": None, "ci95": None}
    cid_by_scenario: dict[str, list[str]] = {}
    for cid in ids:
        sid = by_pol[cid].get("scenario_id", "")
        if sid:
            cid_by_scenario.setdefault(sid, []).append(cid)
    if len(cid_by_scenario) >= 2:
        cluster_ids = sorted(cid_by_scenario)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(bootstrap):
            drawn = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sampled = [cid for cluster in drawn for cid in cid_by_scenario[cluster]]
            samples.append(float(np.mean([_diff(cid) for cid in sampled])))
    else:
        rng = np.random.default_rng(seed)
        diffs = np.asarray([_diff(cid) for cid in ids], dtype=np.float64)
        samples = [float(diffs[rng.integers(0, len(diffs), len(diffs))].mean()) for _ in range(bootstrap)]
    return {
        "n": int(len(ids)),
        "n_clusters": int(len(cid_by_scenario)),
        "point": float(np.mean([_diff(cid) for cid in ids])),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def slim_choice(row: dict) -> dict:
    return {
        "conversation_id": row["conversation_id"],
        "scenario_id": row.get("scenario_id", ""),
        "family": row["family"],
        "status_class": row["status_class"],
        "route_action": row["route_action"],
        "method": row["method"],
        "target_status": row.get("target_status"),
        "layer": row.get("layer"),
        "alpha": row.get("alpha"),
        "status_correct": row["status_correct"],
        "basis_strict_ok": row["basis_strict_ok"],
        "parse_success": row["parse_success"],
        "fixes_status": row["fixes_status"],
        "fixes_strict": row["fixes_strict"],
        "harms_honest_status": row["harms_honest_status"],
        "harms_honest_strict": row["harms_honest_strict"],
        "policy_score": row.get("policy_score"),
        "policy_abstained_by_threshold": row.get("policy_abstained_by_threshold"),
    }


def build_policies(rows: list[dict], *, threshold: float) -> dict[str, dict]:
    methods = ["bidir_linear", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated"]
    policies = {
        "baseline": evaluate_fixed(rows, "baseline", lambda c: choose_method(c, "baseline")),
        "fixed_bidir_linear": evaluate_fixed(rows, "fixed_bidir_linear", lambda c: choose_method(c, "bidir_linear")),
        "fixed_bidir_tangent": evaluate_fixed(rows, "fixed_bidir_tangent", lambda c: choose_method(c, "bidir_tangent")),
        "fixed_global_mean_gated": evaluate_fixed(rows, "fixed_global_mean_gated", lambda c: choose_method(c, "global_mean_gated")),
        "fixed_global_probe_gated": evaluate_fixed(rows, "fixed_global_probe_gated", lambda c: choose_method(c, "global_probe_gated")),
        "fixed_random_gated": evaluate_fixed(rows, "fixed_random_gated", lambda c: choose_method(c, "random_gated")),
        "route_hybrid_mean_probe": evaluate_fixed(
            rows,
            "route_hybrid_mean_probe",
            lambda c: choose_route_map(c, {"steer_to_PASS": "global_mean_gated", "steer_to_FAIL": "global_probe_gated"}),
        ),
        "route_hybrid_mean_random": evaluate_fixed(
            rows,
            "route_hybrid_mean_random",
            lambda c: choose_route_map(c, {"steer_to_PASS": "global_mean_gated", "steer_to_FAIL": "random_gated"}),
        ),
        "margin_argmax_all": evaluate_fixed(
            rows,
            "margin_argmax_all",
            lambda c: choose_margin_argmax(c, include_baseline=True),
        ),
        "margin_argmax_steered": evaluate_fixed(
            rows,
            "margin_argmax_steered",
            lambda c: choose_margin_argmax(c, include_baseline=False),
        ),
        "train_best_route_status": evaluate_train_best(rows, objective="status_reward", methods=methods),
        "train_best_route_strict": evaluate_train_best(rows, objective="strict_reward", methods=methods),
        "learned_context_rf_strict": evaluate_learned(
            rows,
            objective="strict_reward",
            model_name="rf",
            include_response_margin=False,
            threshold=threshold,
        ),
        "learned_response_rf_strict": evaluate_learned(
            rows,
            objective="strict_reward",
            model_name="rf",
            include_response_margin=True,
            threshold=threshold,
        ),
    }
    return policies


__all__ = [
    "STATUS_CLASSES",
    "safe_float",
    "row_key",
    "merge_rows",
    "grouped_by_conversation",
    "baseline_row",
    "choose_method",
    "choose_route_map",
    "choose_margin_argmax",
    "summarize_choices",
    "feature_dict",
    "fit_model",
    "predict_model",
    "choose_learned",
    "evaluate_fixed",
    "score_action_on_train",
    "evaluate_train_best",
    "evaluate_learned",
    "paired_gap",
    "slim_choice",
    "build_policies",
    "file_sha256",
]
