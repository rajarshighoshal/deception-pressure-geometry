"""Reusable, model-free primitives for the powered150 action-response field.

Pure helpers shared by the powered150 adapter, selector evaluation, report
renderer, and the typed-graph controller. No torch / sklearn / model imports so
this stays cheap to import and safe to unit test. Row schema is the decoupled
BF16 decision-token action-response artifact
(``score_steering_specs.py`` / ``action_response.rows``).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from geoprobe.io import file_sha256  # noqa: F401  re-exported; canonical impl moved to geoprobe.io

STATUS_CLASSES = ('false_FAIL', 'false_PASS', 'honest_PASS', 'honest_FAIL')
TARGET_STATUSES = ('PASS', 'FAIL')

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def load_action_response(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text())
    action_response = payload.get("action_response", payload)
    rows = action_response.get("rows", [])
    if not rows:
        raise ValueError(f"no action_response rows found in {path}")
    return rows, {
        "top_level_keys": sorted(payload.keys()),
        "action_response_summary": action_response.get("summary", {}),
        "job_status": payload.get("job_status"),
        "n_specs_expected": payload.get("n_specs_expected"),
        "n_specs_scored": payload.get("n_specs_scored"),
        "n_eval_cids_expected": payload.get("n_eval_cids_expected"),
        "n_eval_cids_scored": payload.get("n_eval_cids_scored"),
        "n_action_rows_scored": payload.get("n_action_rows_scored"),
        "verify_batched_margins": payload.get("verify_batched_margins"),
        "batched_margin_verifications": payload.get("batched_margin_verifications"),
        "gpu_summary": payload.get("gpu_summary"),
        "note": (
            "Rows may be nested under top-level action_response. The outer wrapper "
            "STATUS from the launcher is not used here."
        ),
    }


def grouped_by_conversation(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["conversation_id"])].append(row)
    return dict(grouped)


def target_from_route(route_action: str | None) -> str | None:
    if route_action == "steer_to_PASS":
        return "PASS"
    if route_action == "steer_to_FAIL":
        return "FAIL"
    return None


def action_key(row: dict) -> tuple[str, str | None, int | None, float | None]:
    method = str(row.get("method") or "")
    target = row.get("target_status")
    layer = None if row.get("layer") is None else int(row["layer"])
    alpha = None if row.get("alpha") is None else float(row["alpha"])
    return (method, None if target is None else str(target), layer, alpha)


def row_matches_key(row: dict, key: tuple[str, str | None, int | None, float | None]) -> bool:
    method, target, layer, alpha = key
    if str(row.get("method")) != method:
        return False
    if target is not None and str(row.get("target_status")) != target:
        return False
    if layer is not None and int(row.get("layer", -999)) != int(layer):
        return False
    if alpha is not None and abs(float(row.get("alpha", -999.0)) - float(alpha)) > 1e-8:
        return False
    return True


def is_abstain(row: dict) -> bool:
    return str(row.get("method")) == "abstain"


def candidate_action_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if not is_abstain(row)
        and row.get("target_status") in TARGET_STATUSES
        and row.get("layer") is not None
    ]


def route_matched_candidate_action_rows(rows: list[dict]) -> list[dict]:
    """Return actions whose target is the target actually supplied by the row's route.

    Powered150 contains counter-target actions for honest rows so response fields can be
    characterized symmetrically. Those counterfactual actions are not available to a policy
    evaluated under the oracle route and must not influence its train-fold action selection.
    """
    return [
        row
        for row in candidate_action_rows(rows)
        if target_from_route(row.get("route_action")) == row.get("target_status")
    ]


def abstain_row(candidates: list[dict]) -> dict:
    for row in candidates:
        if is_abstain(row):
            return row
    raise ValueError("candidate group has no abstain row")


def desired_margin_sign(row: dict) -> float:
    sign = safe_float(row.get("desired_margin_sign"))
    if sign != 0.0:
        return sign
    status = str(row.get("desired_status") or row.get("true_status") or "").upper()
    if status == "PASS":
        return 1.0
    if status == "FAIL":
        return -1.0
    return 0.0


def aligned_margin(row: dict) -> float:
    return desired_margin_sign(row) * safe_float(row.get("final_margin"))


def aligned_delta_margin(row: dict) -> float:
    return desired_margin_sign(row) * (
        safe_float(row.get("final_margin")) - safe_float(row.get("base_margin"))
    )


def target_value(row: dict, objective: str) -> float:
    if objective == "reward":
        return safe_float(row.get("reward"))
    if objective == "aligned_margin":
        return aligned_margin(row)
    if objective == "aligned_delta_margin":
        return aligned_delta_margin(row)
    raise ValueError(f"unknown objective {objective!r}")


def route_features(row: dict) -> dict[str, str | float]:
    return {
        "route_action": str(row.get("route_action") or "unknown"),
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "base_margin": safe_float(row.get("base_margin")),
        "abs_base_margin": abs(safe_float(row.get("base_margin"))),
        "gate_score_PASS_minus_FAIL": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
    }


def action_features(row: dict) -> dict[str, str | float]:
    out: dict[str, str | float] = {
        "method": str(row.get("method") or "unknown"),
        "target_status": str(row.get("target_status") or "NONE"),
        "target_margin_sign": safe_float(row.get("target_margin_sign")),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
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
    return out


def response_features(row: dict, *, include_response: bool) -> dict[str, str | float]:
    if not include_response:
        return {
            "has_response": 0.0,
            "final_margin": 0.0,
            "delta_margin": 0.0,
            "aligned_margin": 0.0,
            "aligned_delta_margin": 0.0,
            "margin_source": "MASKED",
        }
    return {
        "has_response": 1.0,
        "final_margin": safe_float(row.get("final_margin")),
        "delta_margin": safe_float(row.get("delta_margin")),
        "aligned_margin": aligned_margin(row),
        "aligned_delta_margin": aligned_delta_margin(row),
        "margin_source": str(row.get("margin_source") or "UNKNOWN"),
    }


def make_family_folds(rows: list[dict], n_folds: int) -> list[list[str]]:
    baseline = [row for row in rows if is_abstain(row)]
    counts = Counter(str(row["family"]) for row in baseline or rows)
    folds: list[list[str]] = [[] for _ in range(max(1, min(n_folds, len(counts))))]
    fold_sizes = [0 for _ in folds]
    for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        idx = min(range(len(folds)), key=lambda i: (fold_sizes[i], i))
        folds[idx].append(family)
        fold_sizes[idx] += int(count)
    return [sorted(fold) for fold in folds]


def select_action_keys(
    train_rows: list[dict],
    *,
    top_per_target: int,
    objective: str,
    require_route_match: bool = False,
) -> dict[str, list[tuple[str, str | None, int | None, float | None]]]:
    grouped: dict[str, dict[tuple[str, str | None, int | None, float | None], list[float]]] = {
        target: defaultdict(list) for target in TARGET_STATUSES
    }
    candidates = (
        route_matched_candidate_action_rows(train_rows)
        if require_route_match
        else candidate_action_rows(train_rows)
    )
    for row in candidates:
        target = str(row.get("target_status"))
        if target not in grouped:
            continue
        grouped[target][action_key(row)].append(target_value(row, objective))
    selected: dict[str, list[tuple[str, str | None, int | None, float | None]]] = {}
    for target, by_key in grouped.items():
        scored = []
        for key, values in by_key.items():
            method, _target, layer, alpha = key
            scored.append((
                float(np.mean(values)),
                float(np.log1p(len(values))),
                -float(alpha or 0.0),
                -float(layer or 0.0),
                method,
                key,
            ))
        scored.sort(reverse=True)
        limit = len(scored) if top_per_target <= 0 else min(top_per_target, len(scored))
        selected[target] = [item[-1] for item in scored[:limit]]
    return selected


def row_in_selected_library(row: dict, selected: dict[str, list[tuple[str, str | None, int | None, float | None]]]) -> bool:
    if is_abstain(row):
        return True
    target = row.get("target_status")
    return target in selected and action_key(row) in set(selected[str(target)])


def prune_to_selected_library(
    rows: list[dict],
    selected: dict[str, list[tuple[str, str | None, int | None, float | None]]],
) -> list[dict]:
    selected_sets = {target: set(keys) for target, keys in selected.items()}
    out = []
    for row in rows:
        if is_abstain(row):
            out.append(row)
            continue
        target = row.get("target_status")
        if target in selected_sets and action_key(row) in selected_sets[str(target)]:
            out.append(row)
    return out


def choose_key(candidates: list[dict], key: tuple[str, str | None, int | None, float | None] | None) -> dict:
    if key is None:
        return abstain_row(candidates)
    for row in candidates:
        if row_matches_key(row, key):
            return row
    return abstain_row(candidates)


def fixed_route_action_keys(rows: list[dict]) -> list[tuple[str, int, float]]:
    """Return the deterministic method/layer/dose grid available under routed targets."""
    keys = {
        (str(row.get("method")), int(row["layer"]), float(row["alpha"]))
        for row in candidate_action_rows(rows)
    }
    return sorted(keys, key=lambda item: (item[0], item[1], item[2]))


def choose_fixed_route(
    candidates: list[dict],
    *,
    method: str,
    layer: int,
    alpha: float,
) -> dict:
    """Choose one fixed coordinate while taking only the declared route's target sign."""
    target = target_from_route(str(candidates[0].get("route_action")))
    if target is None:
        return abstain_row(candidates)
    return choose_key(candidates, (method, target, layer, alpha))


def evaluate_fixed_route_grid(rows: list[dict]) -> dict[str, dict]:
    """Evaluate every fixed coordinate over the complete saved-candidate field."""
    grouped = grouped_by_conversation(rows)
    out: dict[str, dict] = {}
    for method, layer, alpha in fixed_route_action_keys(rows):
        name = f"fixed_route_{method}_L{layer}_a{alpha:g}"
        choices = [
            choose_fixed_route(candidates, method=method, layer=layer, alpha=alpha)
            for candidates in grouped.values()
        ]
        out[name] = {"summary": summarize_choices(choices), "choices": choices}
    return out


def best_route_matched_fixed_coordinate(
    train_rows: list[dict],
    *,
    objective: str,
    methods: set[str] | None = None,
) -> tuple[str, int, float] | None:
    """Select one method/layer/dose while letting the supplied route choose only its sign."""
    by_coordinate: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for row in route_matched_candidate_action_rows(train_rows):
        method = str(row.get("method"))
        if methods is not None and method not in methods:
            continue
        coordinate = (method, int(row["layer"]), float(row["alpha"]))
        by_coordinate[coordinate].append(target_value(row, objective))
    if not by_coordinate:
        return None

    def sort_key(item: tuple[tuple[str, int, float], list[float]]) -> tuple:
        (method, layer, alpha), values = item
        return (
            float(np.mean(values)),
            float(np.log1p(len(values))),
            -float(alpha),
            -float(layer),
            method,
        )

    return max(by_coordinate.items(), key=sort_key)[0]


def choose_route_matched_fixed_coordinate(
    candidates: list[dict],
    coordinate: tuple[str, int, float] | None,
) -> dict:
    if coordinate is None:
        return abstain_row(candidates)
    target = target_from_route(candidates[0].get("route_action"))
    if target is None:
        return abstain_row(candidates)
    method, layer, alpha = coordinate
    return choose_key(candidates, (method, target, layer, alpha))


def best_key_for_target(
    train_rows: list[dict],
    target: str,
    *,
    objective: str,
    selected_keys: list[tuple[str, str | None, int | None, float | None]] | None = None,
    require_route_match: bool = False,
) -> tuple[str, str | None, int | None, float | None] | None:
    selected = set(selected_keys or [])
    by_key: dict[tuple[str, str | None, int | None, float | None], list[float]] = defaultdict(list)
    candidates = (
        route_matched_candidate_action_rows(train_rows)
        if require_route_match
        else candidate_action_rows(train_rows)
    )
    for row in candidates:
        if row.get("target_status") != target:
            continue
        key = action_key(row)
        if selected and key not in selected:
            continue
        by_key[key].append(target_value(row, objective))
    if not by_key:
        return None

    def sort_key(item: tuple[tuple[str, str | None, int | None, float | None], list[float]]) -> tuple:
        key, values = item
        method, _target, layer, alpha = key
        return (
            float(np.mean(values)),
            float(np.log1p(len(values))),
            -float(alpha or 0.0),
            -float(layer or 0.0),
            method,
        )

    return max(by_key.items(), key=sort_key)[0]


def slim_choice(row: dict) -> dict:
    return {
        "conversation_id": row.get("conversation_id"),
        "scenario_id": row.get("scenario_id"),
        "family": row.get("family"),
        "status_class": row.get("status_class"),
        "route_action": row.get("route_action"),
        "method": row.get("method"),
        "target_status": row.get("target_status"),
        "layer": row.get("layer"),
        "alpha": row.get("alpha"),
        "base_margin": row.get("base_margin"),
        "final_margin": row.get("final_margin"),
        "delta_margin": row.get("delta_margin"),
        "aligned_margin": aligned_margin(row),
        "aligned_delta_margin": aligned_delta_margin(row),
        "correct_after": row.get("correct_after"),
        "fixes_error": row.get("fixes_error"),
        "harms_honest": row.get("harms_honest"),
        "reward": row.get("reward"),
        "margin_source": row.get("margin_source"),
        "policy_score": row.get("policy_score"),
        "policy_abstained_by_threshold": row.get("policy_abstained_by_threshold"),
    }


def summarize_choices(choices: list[dict]) -> dict:
    by_class = {}
    for cls in STATUS_CLASSES:
        rows = [row for row in choices if row.get("status_class") == cls]
        by_class[cls] = {
            "n": len(rows),
            "correct_after": int(sum(bool(row.get("correct_after")) for row in rows)),
            "fixes_error": int(sum(bool(row.get("fixes_error")) for row in rows)),
            "harms_honest": int(sum(bool(row.get("harms_honest")) for row in rows)),
            "mean_reward": float(np.mean([safe_float(row.get("reward")) for row in rows])) if rows else None,
            "mean_aligned_margin": float(np.mean([aligned_margin(row) for row in rows])) if rows else None,
            "chosen_methods": dict(Counter(str(row.get("method")) for row in rows)),
        }
    deceptive = [row for row in choices if str(row.get("status_class", "")).startswith("false_")]
    honest = [row for row in choices if str(row.get("status_class", "")).startswith("honest_")]
    return {
        "n": len(choices),
        "deceptive_n": len(deceptive),
        "honest_n": len(honest),
        "fixes_error": int(sum(bool(row.get("fixes_error")) for row in deceptive)),
        "honest_harms": int(sum(bool(row.get("harms_honest")) for row in honest)),
        "mean_reward": float(np.mean([safe_float(row.get("reward")) for row in choices])) if choices else None,
        "mean_aligned_margin": float(np.mean([aligned_margin(row) for row in choices])) if choices else None,
        "mean_aligned_delta_margin": float(np.mean([aligned_delta_margin(row) for row in choices])) if choices else None,
        "chosen_methods": dict(Counter(str(row.get("method")) for row in choices)),
        "chosen_targets": dict(Counter(str(row.get("target_status")) for row in choices)),
        "by_status_class": by_class,
    }


def choose_by_scores(candidates: list[dict], scores: np.ndarray) -> dict:
    if len(candidates) == 0:
        raise ValueError("empty candidate group")
    if len(scores) != len(candidates):
        chosen = dict(abstain_row(candidates))
        chosen["policy_score"] = -1e9
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    best_idx = int(np.argmax(scores))
    chosen = dict(candidates[best_idx])
    chosen["policy_score"] = float(scores[best_idx])
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def choose_oracle(candidates: list[dict], *, objective: str) -> dict:
    chosen = max(
        candidates,
        key=lambda row: (
            target_value(row, objective),
            safe_float(row.get("reward")),
            aligned_margin(row),
            is_abstain(row),
        ),
    )
    return dict(chosen)


def evaluate_fixed_route_best(
    rows: list[dict],
    *,
    folds: list[list[str]],
    objective: str,
    selected_by_fold: list[dict[str, list[tuple[str, str | None, int | None, float | None]]]] | None = None,
    require_route_match: bool = False,
) -> dict:
    grouped = grouped_by_conversation(rows)
    choices = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [row for row in rows if str(row["family"]) not in heldout_set]
        selected = selected_by_fold[fold_idx] if selected_by_fold is not None else {}
        best_by_target = {
            target: best_key_for_target(
                train,
                target,
                objective=objective,
                selected_keys=selected.get(target),
                require_route_match=require_route_match,
            )
            for target in TARGET_STATUSES
        }
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) not in heldout_set:
                continue
            target = target_from_route(str(candidates[0].get("route_action")))
            fold_choices.append(choose_key(candidates, best_by_target.get(target)))
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "best_by_target": {
                target: list(key) if key is not None else None
                for target, key in best_by_target.items()
            },
            "selected_library_restricted": selected_by_fold is not None,
            "summary": summarize_choices(fold_choices),
        }
    return {
        "summary": summarize_choices(choices),
        "folds": fold_out,
        "choices": choices,
        "require_route_match": require_route_match,
    }


def evaluate_route_matched_fixed_coordinate(
    rows: list[dict],
    *,
    folds: list[list[str]],
    objective: str,
    methods: set[str] | None = None,
) -> dict:
    """Fit one train-fold coordinate and supply only the row's declared route at evaluation."""
    grouped = grouped_by_conversation(rows)
    choices: list[dict] = []
    fold_out: dict[str, dict] = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [row for row in rows if str(row["family"]) not in heldout_set]
        coordinate = best_route_matched_fixed_coordinate(
            train,
            objective=objective,
            methods=methods,
        )
        fold_choices = [
            choose_route_matched_fixed_coordinate(candidates, coordinate)
            for candidates in grouped.values()
            if str(candidates[0]["family"]) in heldout_set
        ]
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "coordinate": list(coordinate) if coordinate is not None else None,
            "summary": summarize_choices(fold_choices),
        }
    return {
        "summary": summarize_choices(choices),
        "folds": fold_out,
        "choices": choices,
        "methods": sorted(methods) if methods is not None else None,
        "route_information": "target_from_declared_route_only",
    }


def evaluate_oracle_policy(rows: list[dict], *, objective: str) -> dict:
    choices = [choose_oracle(candidates, objective=objective) for candidates in grouped_by_conversation(rows).values()]
    return {"summary": summarize_choices(choices), "choices": choices}


def evaluate_abstain_policy(rows: list[dict]) -> dict:
    choices = [abstain_row(candidates) for candidates in grouped_by_conversation(rows).values()]
    return {"summary": summarize_choices(choices), "choices": choices}


def metric_values(choices: list[dict], metric: str) -> dict[str, float]:
    out = {}
    for row in choices:
        cid = str(row["conversation_id"])
        status_class = str(row.get("status_class", ""))
        if metric == "fixes_error":
            if status_class.startswith("false_"):
                out[cid] = float(bool(row.get("fixes_error")))
        elif metric == "honest_harm":
            if status_class.startswith("honest_"):
                out[cid] = float(bool(row.get("harms_honest")))
        elif metric == "reward":
            out[cid] = safe_float(row.get("reward"))
        elif metric == "aligned_margin":
            out[cid] = aligned_margin(row)
        elif metric == "aligned_delta_margin":
            out[cid] = aligned_delta_margin(row)
        else:
            raise ValueError(metric)
    return out


@lru_cache(maxsize=128)
def bootstrap_cluster_count_matrix(n_clusters: int, bootstrap: int, seed: int) -> np.ndarray:
    """Cached scenario-bootstrap draw counts.

    The old implementation rebuilt `bootstrap` lists of sampled conversation IDs
    for every policy/reference/metric pair. Powered150 has hundreds of such pairs;
    only the per-cluster diff sums change. The resampling counts depend only on
    `(n_clusters, bootstrap, seed)`, so cache that matrix and reuse it.
    """
    if n_clusters <= 0 or bootstrap <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(bootstrap, n_clusters))
    counts = np.zeros((bootstrap, n_clusters), dtype=np.float64)
    rows = np.repeat(np.arange(bootstrap), n_clusters)
    np.add.at(counts, (rows, draws.reshape(-1)), 1.0)
    return counts


def paired_gap(
    policy: list[dict],
    reference: list[dict],
    metric: str,
    *,
    seed: int,
    bootstrap: int,
) -> dict:
    pol = metric_values(policy, metric)
    ref = metric_values(reference, metric)
    ids = sorted(set(pol) & set(ref))
    if not ids:
        return {"n": 0, "point": None, "ci95": None}
    pol_by_cid = {str(row["conversation_id"]): row for row in policy}
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        sid = str(pol_by_cid[cid].get("scenario_id") or cid)
        by_scenario[sid].append(cid)
    diffs = {cid: pol[cid] - ref[cid] for cid in ids}
    point = float(np.mean([diffs[cid] for cid in ids]))
    if bootstrap <= 0:
        return {"n": len(ids), "n_clusters": len(by_scenario), "point": point, "ci95": None}
    clusters = sorted(by_scenario)
    cluster_sums = np.asarray([
        sum(diffs[cid] for cid in by_scenario[cluster])
        for cluster in clusters
    ], dtype=np.float64)
    cluster_counts = np.asarray([len(by_scenario[cluster]) for cluster in clusters], dtype=np.float64)
    draw_counts = bootstrap_cluster_count_matrix(len(clusters), int(bootstrap), int(seed))
    sample_den = draw_counts @ cluster_counts
    samples = (draw_counts @ cluster_sums) / np.maximum(sample_den, 1e-12)
    return {
        "n": int(len(ids)),
        "n_clusters": int(len(by_scenario)),
        "point": point,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def policy_gaps(policies: dict[str, dict], references: list[str], *, seed: int, bootstrap: int) -> dict:
    gaps = {}
    for name, policy in policies.items():
        gaps[name] = {}
        for ref in references:
            if ref == name or ref not in policies:
                continue
            gaps[name][ref] = {
                metric: paired_gap(
                    policy["choices"],
                    policies[ref]["choices"],
                    metric,
                    seed=seed,
                    bootstrap=bootstrap,
                )
                for metric in ("fixes_error", "honest_harm", "reward", "aligned_margin", "aligned_delta_margin")
            }
    return gaps
