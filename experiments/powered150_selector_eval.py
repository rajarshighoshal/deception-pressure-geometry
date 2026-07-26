"""Powered150 selector evaluation over the completed action-response field.

This reuses the 4-bit MLX selector stack on the powered150 BF16
action-response artifact. It does not run generation. The output is a durable
JSON artifact consumed by `report_powered150_selector_eval.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.geometric_chart_atlas_selector import (  # noqa: E402
    ChartAtlasSelector,
    choose_chart,
    load_state_vectors,
)
from experiments.geometric_graph_control_selector import fold_diffusion_vectors  # noqa: E402
from experiments.learned_graph_control_model import (  # noqa: E402
    LearnedGraphSelector,
    choose_learned_graph,
)
from experiments.powered150_action_response_adapter import (  # noqa: E402
    adapt_row,
    adapted_method,
    validate_adapted,
)
from geoprobe.text.parse import parse_csv, parse_int_csv  # noqa: E402
from geoprobe.control.action_response import (  # noqa: E402
    TARGET_STATUSES,
    abstain_row,
    action_key,
    aligned_delta_margin,
    aligned_margin,
    best_key_for_target,
    choose_key,
    evaluate_fixed_route_grid,
    evaluate_abstain_policy,
    evaluate_oracle_policy,
    file_sha256,
    grouped_by_conversation,
    is_abstain,
    load_action_response,
    make_family_folds,
    paired_gap,
    safe_float,
    select_action_keys,
    slim_choice,
    summarize_choices,
    target_from_route,
    target_value,
)
from geoprobe.control.structural_candidates import structural_candidate_records  # noqa: E402
from experiments.report_powered150_selector_eval import (  # noqa: E402
    render_markdown,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402


ActionKey = tuple[str, str | None, int | None, float | None]


def parse_chart_counts(value: str) -> list[int | str]:
    out: list[int | str] = []
    for item in parse_csv(value):
        out.append("auto" if item == "auto" else int(item))
    return out


def parse_neighbors(value: str) -> int | str:
    value = value.strip()
    return "auto" if value == "auto" else int(value)


def deployable_feature_dict(row: dict, *, include_response: bool = False) -> dict[str, str | float]:
    """Features available before generation.

    Context mode deliberately excludes true labels, reward, final margin,
    correctness, and harm fields.  The optional response mode is a ceiling.
    """
    out: dict[str, str | float] = {
        "method": str(row.get("method") or ""),
        "target_status": str(row.get("target_status") or "NONE"),
        "route_action": str(row.get("route_action") or "unknown"),
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "base_margin": safe_float(row.get("base_margin")),
        "abs_base_margin": abs(safe_float(row.get("base_margin"))),
        "target_margin_sign": safe_float(row.get("target_margin_sign")),
        "gate_score_PASS_minus_FAIL": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
        "projection_fraction": safe_float(row.get("projection_fraction"), -1.0),
        "cos_to_raw": safe_float(row.get("cos_to_raw"), -1.0),
        "neighbor_distance_mean": safe_float(row.get("neighbor_distance_mean"), -1.0),
        "neighbor_distance_max": safe_float(row.get("neighbor_distance_max"), -1.0),
    }
    direction_info = row.get("direction_info") or {}
    if isinstance(direction_info, dict):
        out["direction_train_points"] = safe_float(direction_info.get("n_train_points"), -1.0)
        out["direction_mixed_pairs"] = safe_float(direction_info.get("n_mixed_scenario_level_pairs"), -1.0)
        out["direction_convention"] = str(direction_info.get("direction_convention") or "NONE")
    for key in sorted(row):
        if key.startswith("pc_"):
            out[key] = safe_float(row.get(key), 0.0)
    if include_response:
        out.update({
            "final_margin": safe_float(row.get("final_margin")),
            "delta_margin": safe_float(row.get("delta_margin")),
            "aligned_margin": aligned_margin(row),
            "aligned_delta_margin": aligned_delta_margin(row),
            "correct_after": float(bool(row.get("correct_after"))),
            "margin_source": str(row.get("margin_source") or "UNKNOWN"),
        })
    return out


def adapted_action_key(row: dict) -> ActionKey | None:
    if str(row.get("method")) == "baseline":
        return None
    layer = row.get("layer")
    if layer is None:
        return None
    return (
        str(row.get("method")),
        None if row.get("target_status") is None else str(row.get("target_status")),
        int(layer),
        float(row.get("alpha")),
    )


def canonical_selected_keys(selected: dict[str, list[ActionKey]]) -> set[ActionKey]:
    """Map raw selected action keys into the adapted method namespace.

    ``select_action_keys`` runs on raw action-response rows, so its keys carry raw
    method names (e.g. ``global_probe``). The structural selectors consume adapted
    rows where ``adapted_method`` renames the gated globals (``global_probe`` ->
    ``global_probe_gated`` etc.). Comparing raw keys against adapted rows silently
    drops every aliased global action and leaves only baseline, so canonicalize the
    method component here. Non-aliased methods pass through unchanged.
    """
    out: set[ActionKey] = set()
    for keys in selected.values():
        for method, target, layer, alpha in keys:
            out.add((adapted_method(method), target, layer, alpha))
    return out


def prune_adapted_to_selected_library(
    rows: list[dict],
    selected: dict[str, list[ActionKey]],
) -> list[dict]:
    selected_keys = canonical_selected_keys(selected)
    out = []
    for row in rows:
        if str(row.get("method")) == "baseline":
            out.append(row)
            continue
        key = adapted_action_key(row)
        if key in selected_keys:
            out.append(row)
    return out


def expected_heldout_cids(rows: list[dict], heldout: set[str]) -> set[str]:
    return {
        str(row["conversation_id"])
        for row in rows
        if str(row["family"]) in heldout and str(row.get("method")) == "baseline"
    }


def assert_complete_fold_choices(
    *,
    fold_idx: int,
    choices: list[dict],
    expected: set[str],
) -> None:
    got = {str(row["conversation_id"]) for row in choices}
    if got != expected:
        missing = sorted(expected - got)[:10]
        extra = sorted(got - expected)[:10]
        raise ValueError(
            f"fold {fold_idx} choice coverage mismatch: missing={len(expected - got)} "
            f"extra={len(got - expected)} missing_examples={missing} extra_examples={extra}"
        )


def evaluate_train_best_route(
    rows: list[dict],
    *,
    folds: list[list[str]],
    objective: str,
    selected_by_fold: list[dict[str, list[ActionKey]]] | None = None,
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
            "selected_library_restricted": selected_by_fold is not None,
            "best_by_target": {
                target: list(key) if key is not None else None
                for target, key in best_by_target.items()
            },
            "summary": summarize_choices(fold_choices),
        }
    return {
        "summary": summarize_choices(choices),
        "folds": fold_out,
        "choices": choices,
        "require_route_match": require_route_match,
    }


def route_class_spec(route_action: str) -> tuple[str, str, str] | None:
    if route_action == "steer_to_PASS":
        return ("PASS", "false_FAIL", "honest_FAIL")
    if route_action == "steer_to_FAIL":
        return ("FAIL", "false_PASS", "honest_PASS")
    return None


def best_selective_key_for_route(
    train_rows: list[dict],
    route_action: str,
    *,
    collateral_penalty: float,
) -> ActionKey | None:
    spec = route_class_spec(route_action)
    if spec is None:
        return None
    target_status, false_class, collateral_class = spec
    desired_sign = 1.0 if target_status == "PASS" else -1.0
    by_key: dict[ActionKey, list[dict]] = defaultdict(list)
    for row in train_rows:
        if is_abstain(row) or row.get("target_status") != target_status:
            continue
        if row.get("status_class") not in {false_class, collateral_class}:
            continue
        by_key[action_key(row)].append(row)
    scored = []
    for key, action_rows in by_key.items():
        target_rows = [row for row in action_rows if row.get("status_class") == false_class]
        collateral_rows = [row for row in action_rows if row.get("status_class") == collateral_class]
        if not target_rows or not collateral_rows:
            continue
        target_fix = float(np.mean([bool(row.get("fixes_error")) for row in target_rows]))
        collateral_harm = float(np.mean([bool(row.get("harms_honest")) for row in collateral_rows]))
        target_move = float(np.mean([desired_sign * safe_float(row.get("delta_margin")) for row in target_rows]))
        collateral_same_way = float(np.mean([desired_sign * safe_float(row.get("delta_margin")) for row in collateral_rows]))
        score = target_fix + 0.05 * target_move - collateral_penalty * (collateral_harm + 0.05 * collateral_same_way)
        method, _target, layer, alpha = key
        scored.append((score, target_fix, -collateral_harm, -float(alpha or 0.0), -float(layer or 0.0), method, key))
    return max(scored)[-1] if scored else None


def evaluate_selective_route(rows: list[dict], *, folds: list[list[str]], collateral_penalty: float) -> dict:
    grouped = grouped_by_conversation(rows)
    choices = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [row for row in rows if str(row["family"]) not in heldout_set]
        best_by_route = {
            route: best_selective_key_for_route(train, route, collateral_penalty=collateral_penalty)
            for route in ("steer_to_PASS", "steer_to_FAIL")
        }
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) not in heldout_set:
                continue
            route = str(candidates[0].get("route_action"))
            fold_choices.append(choose_key(candidates, best_by_route.get(route)))
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "best_by_route": {route: list(key) if key is not None else None for route, key in best_by_route.items()},
            "summary": summarize_choices(fold_choices),
        }
    return {"summary": summarize_choices(choices), "folds": fold_out, "choices": choices}


class RidgeRewardSelector:
    def __init__(self, *, objective: str, include_response: bool, alpha: float = 1.0) -> None:
        self.objective = objective
        self.include_response = include_response
        self.alpha = float(alpha)

    def fit(self, rows: list[dict]) -> "RidgeRewardSelector":
        self.vectorizer_ = DictVectorizer(sparse=False)
        x = self.vectorizer_.fit_transform([
            deployable_feature_dict(row, include_response=self.include_response)
            for row in rows
        ])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        y = np.asarray([target_value(row, self.objective) for row in rows], dtype=np.float64)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(xs, y)
        return self

    def score(self, rows: list[dict]) -> np.ndarray:
        x = self.vectorizer_.transform([
            deployable_feature_dict(row, include_response=self.include_response)
            for row in rows
        ])
        x = self.scaler_.transform(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        return np.asarray(self.model_.predict(x), dtype=np.float64)


class ImitationSelector:
    def __init__(self, *, include_response: bool, seed: int) -> None:
        self.include_response = include_response
        self.seed = int(seed)

    def fit(self, rows: list[dict]) -> "ImitationSelector":
        grouped = grouped_by_conversation(rows)
        x = []
        y = []
        for candidates in grouped.values():
            best_reward = max(target_value(row, "reward") for row in candidates)
            positive = {
                idx for idx, row in enumerate(candidates)
                if target_value(row, "reward") == best_reward and (best_reward > 0.0 or is_abstain(row))
            }
            for idx, row in enumerate(candidates):
                x.append(deployable_feature_dict(row, include_response=self.include_response))
                y.append(1 if idx in positive else 0)
        self.vectorizer_ = DictVectorizer(sparse=False)
        xv = self.vectorizer_.fit_transform(x)
        if len(set(y)) < 2:
            self.model_ = None
            return self
        self.model_ = RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.seed,
        )
        self.model_.fit(xv, y)
        return self

    def score(self, rows: list[dict]) -> np.ndarray:
        if self.model_ is None:
            return np.zeros(len(rows), dtype=np.float64)
        xv = self.vectorizer_.transform([
            deployable_feature_dict(row, include_response=self.include_response)
            for row in rows
        ])
        proba = self.model_.predict_proba(xv)
        pos_idx = list(self.model_.classes_).index(1)
        return np.asarray(proba[:, pos_idx], dtype=np.float64)


class GeometricKnnSelector:
    def __init__(self, *, objective: str, include_response: bool, k: int, metric: str) -> None:
        self.objective = objective
        self.include_response = include_response
        self.k = int(k)
        self.metric = str(metric)

    def fit(self, rows: list[dict]) -> "GeometricKnnSelector":
        self.vectorizer_ = DictVectorizer(sparse=False)
        raw = self.vectorizer_.fit_transform([
            deployable_feature_dict(row, include_response=self.include_response)
            for row in rows
        ])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        x = self.scaler_.fit_transform(raw)
        self.y_ = np.asarray([target_value(row, self.objective) for row in rows], dtype=np.float64)
        if self.metric == "supervised_diag":
            pos = self.y_ > 0.0
            neg = ~pos
            if pos.any() and neg.any():
                weights = (x[pos].mean(axis=0) - x[neg].mean(axis=0)) ** 2
            else:
                weights = np.ones(x.shape[1], dtype=np.float64)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = weights / max(float(np.mean(weights)), 1e-12)
            self.weights_ = np.clip(weights, 0.05, 20.0)
            x = x * np.sqrt(self.weights_[None, :])
        elif self.metric != "euclidean":
            raise ValueError(f"unknown metric {self.metric!r}")
        self.x_ = x
        self.nn_ = NearestNeighbors(
            n_neighbors=min(max(self.k, 1), len(self.x_)),
            algorithm="auto",
            metric="euclidean",
        ).fit(self.x_)
        return self

    def _transform(self, rows: list[dict]) -> np.ndarray:
        raw = self.vectorizer_.transform([
            deployable_feature_dict(row, include_response=self.include_response)
            for row in rows
        ])
        x = self.scaler_.transform(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
        if self.metric == "supervised_diag":
            x = x * np.sqrt(self.weights_[None, :])
        return x

    def score(self, rows: list[dict]) -> np.ndarray:
        q = self._transform(rows)
        if len(q) == 0:
            return np.asarray([], dtype=np.float64)
        kk = min(self.k, len(self.x_))
        distances, indices = self.nn_.kneighbors(q, n_neighbors=kk, return_distance=True)
        scores = np.empty(len(q), dtype=np.float64)
        for idx, (local_d, order) in enumerate(zip(distances, indices)):
            if np.all(local_d <= 1e-12):
                weights = np.ones_like(local_d)
            else:
                scale = float(np.median(local_d) + 1e-6)
                weights = np.exp(-local_d / scale)
            scores[idx] = float(np.sum(weights * self.y_[order]) / max(float(np.sum(weights)), 1e-12))
        return scores


def choose_scored(candidates: list[dict], scores: np.ndarray, *, threshold: float) -> dict:
    if len(scores) != len(candidates):
        chosen = dict(abstain_row(candidates))
        chosen["policy_score"] = -1e9
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score <= threshold:
        chosen = dict(abstain_row(candidates))
        chosen["policy_score"] = best_score
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    chosen = dict(candidates[best_idx])
    chosen["policy_score"] = best_score
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def evaluate_fold_selector(
    rows: list[dict],
    *,
    folds: list[list[str]],
    selector_factory: Callable[[int], Any],
    threshold: float,
) -> dict:
    grouped = grouped_by_conversation(rows)
    choices = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [row for row in rows if str(row["family"]) not in heldout_set]
        test = [row for row in rows if str(row["family"]) in heldout_set]
        selector = selector_factory(fold_idx).fit(train)
        test_scores = selector.score(test)
        scored_by_cid: dict[str, list[tuple[dict, float]]] = defaultdict(list)
        for row, score in zip(test, test_scores):
            scored_by_cid[str(row["conversation_id"])].append((row, float(score)))
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) not in heldout_set:
                continue
            scored = scored_by_cid.get(str(candidates[0]["conversation_id"]), [])
            scores = np.asarray([score for _row, score in scored], dtype=np.float64)
            fold_choices.append(choose_scored(candidates, scores, threshold=threshold))
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {"heldout_families": heldout, "summary": summarize_choices(fold_choices)}
    return {"summary": summarize_choices(choices), "folds": fold_out, "choices": choices}


def evaluate_chart_atlas_powered(
    rows: list[dict],
    *,
    folds: list[list[str]],
    selected_by_fold: list[dict[str, list[ActionKey]]],
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
    choices: list[dict] = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_full = [row for row in rows if str(row["family"]) not in heldout_set]
        test_full = [row for row in rows if str(row["family"]) in heldout_set]
        train = prune_adapted_to_selected_library(train_full, selected_by_fold[fold_idx])
        test = prune_adapted_to_selected_library(test_full, selected_by_fold[fold_idx])
        selector = ChartAtlasSelector(
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
            seed=seed + fold_idx,
        ).fit(train, state_vectors)
        fold_choices = [
            choose_chart(candidates, selector, state_vectors, threshold=threshold)
            for candidates in grouped_by_conversation(test).values()
        ]
        assert_complete_fold_choices(
            fold_idx=fold_idx,
            choices=fold_choices,
            expected=expected_heldout_cids(rows, heldout_set),
        )
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "n_train_rows_pruned": len(train),
            "n_test_rows_pruned": len(test),
            "chart_count": int(selector.chart_count_),
            "pca_dim": int(selector.pca_.n_components_),
            "chart_support": [float(x) for x in selector.chart_support_],
            "summary": summarize_choices(fold_choices),
        }
    return {"summary": summarize_choices(choices), "folds": fold_out, "choices": choices}


def evaluate_graph_atlas_powered(
    rows: list[dict],
    *,
    folds: list[list[str]],
    selected_by_fold: list[dict[str, list[ActionKey]]],
    raw_state_vectors: dict[tuple[str, int], np.ndarray],
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
    graph_neighbors: int | str,
    diffusion_dim: int,
    diffusion_time: float,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    seed: int,
) -> dict:
    choices: list[dict] = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_full = [row for row in rows if str(row["family"]) not in heldout_set]
        test_full = [row for row in rows if str(row["family"]) in heldout_set]
        train = prune_adapted_to_selected_library(train_full, selected_by_fold[fold_idx])
        test = prune_adapted_to_selected_library(test_full, selected_by_fold[fold_idx])
        state_vectors, graph_meta = fold_diffusion_vectors(
            train,
            test,
            raw_state_vectors,
            graph_neighbors=graph_neighbors,
            diffusion_dim=diffusion_dim,
            diffusion_time=diffusion_time,
            pre_pca_dim=pre_pca_dim,
            scenario_edge_weight=scenario_edge_weight,
            seed=seed + fold_idx,
        )
        selector = ChartAtlasSelector(
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
            seed=seed + fold_idx,
        ).fit(train, state_vectors)
        fold_choices = [
            choose_chart(candidates, selector, state_vectors, threshold=threshold)
            for candidates in grouped_by_conversation(test).values()
        ]
        assert_complete_fold_choices(
            fold_idx=fold_idx,
            choices=fold_choices,
            expected=expected_heldout_cids(rows, heldout_set),
        )
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "n_train_rows_pruned": len(train),
            "n_test_rows_pruned": len(test),
            "graph": graph_meta,
            "chart_count": int(selector.chart_count_),
            "pca_dim": int(selector.pca_.n_components_),
            "chart_support": [float(x) for x in selector.chart_support_],
            "summary": summarize_choices(fold_choices),
        }
    return {"summary": summarize_choices(choices), "folds": fold_out, "choices": choices}


def evaluate_learned_graph_powered(
    rows: list[dict],
    *,
    folds: list[list[str]],
    selected_by_fold: list[dict[str, list[ActionKey]]],
    state_vectors: dict[tuple[str, int], np.ndarray],
    include_response_margin: bool,
    threshold: float,
    graph_neighbors: int | str,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    hidden_dim: int,
    graph_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    extension_neighbors: int | str,
    extension_mix: float,
    seed: int,
) -> dict:
    choices: list[dict] = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_full = [row for row in rows if str(row["family"]) not in heldout_set]
        test_full = [row for row in rows if str(row["family"]) in heldout_set]
        train = prune_adapted_to_selected_library(train_full, selected_by_fold[fold_idx])
        test = prune_adapted_to_selected_library(test_full, selected_by_fold[fold_idx])
        selector = LearnedGraphSelector(
            include_response_margin=include_response_margin,
            graph_neighbors=graph_neighbors,
            pre_pca_dim=pre_pca_dim,
            scenario_edge_weight=scenario_edge_weight,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            extension_neighbors=extension_neighbors,
            extension_mix=extension_mix,
            seed=seed + fold_idx,
        ).fit(train, state_vectors)
        fold_choices = [
            choose_learned_graph(candidates, selector, state_vectors, threshold=threshold)
            for candidates in grouped_by_conversation(test).values()
        ]
        assert_complete_fold_choices(
            fold_idx=fold_idx,
            choices=fold_choices,
            expected=expected_heldout_cids(rows, heldout_set),
        )
        choices.extend(fold_choices)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "n_train_rows_pruned": len(train),
            "n_test_rows_pruned": len(test),
            "graph": getattr(selector, "graph_meta_", {}),
            "summary": summarize_choices(fold_choices),
        }
    return {"summary": summarize_choices(choices), "folds": fold_out, "choices": choices}


def import_typed_graph_policies(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for name, item in payload.get("policies", {}).items():
        if not name.startswith("typed_graph_"):
            continue
        choices = item.get("choices", [])
        out[name] = {
            "summary": summarize_choices(choices),
            "folds": item.get("folds"),
            "choices": choices,
            "imported_from": str(path),
        }
    return out


def slim_eval_choice(row: dict) -> dict:
    out = slim_choice(row)
    for key in (
        "decision_margin",
        "decision_forced_status",
        "strict_reward",
        "status_reward",
        "chart_top",
        "chart_prediction_std",
        "chart_count",
        "chart_pca_dim",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def rank_key(item: tuple[str, dict]) -> tuple[float, float, float, str]:
    name, policy = item
    s = policy["summary"]
    return (
        float(s["mean_reward"]),
        float(s["fixes_error"]) / max(float(s["deceptive_n"]), 1.0),
        -float(s["honest_harms"]) / max(float(s["honest_n"]), 1.0),
        name,
    )


def choose_reference_names(policies: dict[str, dict], *, top_fixed: int) -> list[str]:
    fixed = sorted(
        [(name, item) for name, item in policies.items() if name.startswith("fixed_route_")],
        key=rank_key,
        reverse=True,
    )[:top_fixed]
    refs = [
        "always_abstain",
        "train_best_route_full_reward",
        "selective_route_reward",
        "oracle_full_reward_ceiling",
        "oracle_full_aligned_margin_ceiling",
        "learned_context_ridge_reward",
        "learned_context_imitation_reward",
        "geom_context_euclidean_k11_reward",
        "geom_context_supervised_diag_k11_reward",
        "chart_mean_context_cauto_d12_strict",
        "chart_ridge_context_cauto_d12_strict",
        "graph_mean_context_cauto_gauto_d12_strict",
        "graph_ridge_context_cauto_gauto_d12_strict",
        "learned_graph_context_h32_l2_gauto_strict",
        "typed_graph_context_reward",
        "typed_graph_context_reward_no_library",
        "typed_graph_context_reward_state_masked",
        "typed_graph_response_reward",
    ]
    refs.extend(name for name, _ in fixed)
    return [name for name in refs if name in policies]


def compute_gaps(
    policies: dict[str, dict],
    *,
    references: list[str],
    seed: int,
    bootstrap: int,
) -> dict:
    out = {}
    for name, policy in policies.items():
        out[name] = {}
        for ref in references:
            if name == ref or ref not in policies:
                continue
            out[name][ref] = {
                metric: paired_gap(
                    policy["choices"],
                    policies[ref]["choices"],
                    metric,
                    seed=seed,
                    bootstrap=bootstrap,
                )
                for metric in ("fixes_error", "honest_harm", "reward", "aligned_margin", "aligned_delta_margin")
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-response", required=True)
    parser.add_argument("--typed-graph", default=None)
    parser.add_argument("--activations", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--top-actions-per-target", type=int, default=16)
    parser.add_argument("--learned-threshold", type=float, default=0.0)
    parser.add_argument("--geometric-threshold", type=float, default=0.0)
    parser.add_argument("--collateral-penalty", type=float, default=1.0)
    parser.add_argument("--k-values", default="5,11,21")
    parser.add_argument(
        "--structural-families",
        default="chart,graph,learned_graph",
        help="Comma list from chart,graph,learned_graph. Empty disables activation-state selectors.",
    )
    parser.add_argument("--layers", default="8,12,16,19,20,24,28,32")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--chart-counts", default="auto")
    parser.add_argument("--chart-heads", default="mean,ridge")
    parser.add_argument("--chart-pca-dim", type=int, default=12)
    parser.add_argument("--chart-top-charts", type=int, default=3)
    parser.add_argument("--chart-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--chart-threshold", type=float, default=0.0)
    parser.add_argument("--chart-min-support", type=float, default=4.0)
    parser.add_argument("--chart-min-action-support", type=float, default=2.0)
    parser.add_argument("--chart-fallbacks", default="full,method_layer,method,route")
    parser.add_argument("--graph-neighbors", default="auto")
    parser.add_argument("--graph-diffusion-dim", type=int, default=12)
    parser.add_argument("--graph-diffusion-time", type=float, default=1.0)
    parser.add_argument("--graph-pre-pca-dim", type=int, default=32)
    parser.add_argument("--graph-scenario-edge-weight", type=float, default=0.25)
    parser.add_argument("--learned-graph-hidden-dim", type=int, default=32)
    parser.add_argument("--learned-graph-layers", type=int, default=2)
    parser.add_argument("--learned-graph-dropout", type=float, default=0.1)
    parser.add_argument("--learned-graph-lr", type=float, default=0.003)
    parser.add_argument("--learned-graph-weight-decay", type=float, default=1e-4)
    parser.add_argument("--learned-graph-epochs", type=int, default=60)
    parser.add_argument("--learned-graph-extension-neighbors", default="auto")
    parser.add_argument("--learned-graph-extension-mix", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    action_response_path = Path(args.action_response)
    rows, input_meta = load_action_response(action_response_path)
    adapted_rows = [adapt_row(row) for row in rows]
    adapted_summary = validate_adapted(adapted_rows)
    folds = make_family_folds(rows, args.folds)
    print(
        f"loaded rows={len(rows)} conversations={len(grouped_by_conversation(rows))} folds={len(folds)}",
        flush=True,
    )
    selected_by_fold = [
        select_action_keys(
            [row for row in rows if str(row["family"]) not in set(heldout)],
            top_per_target=args.top_actions_per_target,
            objective="reward",
        )
        for heldout in folds
    ]
    structural_families = parse_csv(args.structural_families)
    if structural_families and not args.activations:
        raise SystemExit("--activations is required when --structural-families is non-empty")
    unknown_structural = sorted(set(structural_families) - {"chart", "graph", "learned_graph"})
    if unknown_structural:
        raise SystemExit(f"unknown --structural-families entries: {unknown_structural}")

    policies: dict[str, dict] = {}
    policy_types: dict[str, str] = {}

    def add(name: str, policy: dict, policy_type: str) -> None:
        policies[name] = policy
        policy_types[name] = policy_type

    add("always_abstain", evaluate_abstain_policy(rows), "baseline")
    add("oracle_full_reward_ceiling", evaluate_oracle_policy(rows, objective="reward"), "oracle")
    add("oracle_full_aligned_margin_ceiling", evaluate_oracle_policy(rows, objective="aligned_margin"), "oracle")
    fixed = evaluate_fixed_route_grid(rows)
    for name, policy in fixed.items():
        add(name, policy, "fixed_route_grid")
    add(
        "train_best_route_full_reward",
        evaluate_train_best_route(rows, folds=folds, objective="reward", selected_by_fold=None),
        "legacy_train_best_target_mismatched",
    )
    add(
        "train_best_route_matched_reward",
        evaluate_train_best_route(
            rows,
            folds=folds,
            objective="reward",
            selected_by_fold=None,
            require_route_match=True,
        ),
        "train_best_route_matched",
    )
    add(
        "train_best_route_selected_reward",
        evaluate_train_best_route(rows, folds=folds, objective="reward", selected_by_fold=selected_by_fold),
        "train_best",
    )
    add(
        "selective_route_reward",
        evaluate_selective_route(rows, folds=folds, collateral_penalty=args.collateral_penalty),
        "selective_route",
    )
    print(f"built fixed/train-best baselines policies={len(policies)}", flush=True)
    add(
        "learned_context_ridge_reward",
        evaluate_fold_selector(
            rows,
            folds=folds,
            selector_factory=lambda _fold: RidgeRewardSelector(objective="reward", include_response=False),
            threshold=args.learned_threshold,
        ),
        "learned_context",
    )
    add(
        "learned_response_ridge_reward",
        evaluate_fold_selector(
            rows,
            folds=folds,
            selector_factory=lambda _fold: RidgeRewardSelector(objective="reward", include_response=True),
            threshold=args.learned_threshold,
        ),
        "learned_response",
    )
    add(
        "learned_context_imitation_reward",
        evaluate_fold_selector(
            rows,
            folds=folds,
            selector_factory=lambda fold: ImitationSelector(include_response=False, seed=args.seed + fold),
            threshold=args.learned_threshold,
        ),
        "learned_context",
    )
    print(f"built learned feature baselines policies={len(policies)}", flush=True)
    for include_response, mode in [(False, "context"), (True, "response")]:
        for metric in ("euclidean", "supervised_diag"):
            for k in [int(item) for item in args.k_values.split(",") if item.strip()]:
                name = f"geom_{mode}_{metric}_k{k}_reward"
                add(
                    name,
                    evaluate_fold_selector(
                        rows,
                        folds=folds,
                        selector_factory=lambda _fold, k=k, metric=metric, include_response=include_response: GeometricKnnSelector(
                            objective="reward",
                            include_response=include_response,
                            k=k,
                            metric=metric,
                        ),
                        threshold=args.geometric_threshold,
                    ),
                    f"geom_{mode}",
                )
        print(f"built geometric kNN mode={mode} policies={len(policies)}", flush=True)
    for name, policy in import_typed_graph_policies(Path(args.typed_graph) if args.typed_graph else None).items():
        add(name, policy, "typed_graph")
    print(f"imported typed graph policies={len(policies)}", flush=True)

    activation_meta = None
    if structural_families:
        activation_path = Path(args.activations)
        layers = parse_int_csv(args.layers)
        state_vectors, activation_meta = load_state_vectors(
            activation_path,
            layers=layers,
            query_turn=args.query_turn,
            query_phase=args.query_phase,
        )
        print(f"loaded activation state vectors={len(state_vectors)}", flush=True)
        chart_counts = parse_chart_counts(args.chart_counts)
        chart_heads = parse_csv(args.chart_heads)
        fallbacks = parse_csv(args.chart_fallbacks)
        graph_neighbors = parse_neighbors(args.graph_neighbors)
        learned_extension_neighbors = parse_neighbors(args.learned_graph_extension_neighbors)

        if "chart" in structural_families:
            for head in chart_heads:
                for include_response, mode in [(False, "context"), (True, "response")]:
                    for chart_count in chart_counts:
                        count_name = str(chart_count)
                        name = f"chart_{head}_{mode}_c{count_name}_d{args.chart_pca_dim}_strict"
                        add(
                            name,
                            evaluate_chart_atlas_powered(
                                adapted_rows,
                                folds=folds,
                                selected_by_fold=selected_by_fold,
                                state_vectors=state_vectors,
                                chart_count=chart_count,
                                pca_dim=args.chart_pca_dim,
                                top_charts=args.chart_top_charts,
                                ridge_alpha=args.chart_ridge_alpha,
                                include_response_margin=include_response,
                                threshold=args.chart_threshold,
                                min_chart_support=args.chart_min_support,
                                min_action_support=args.chart_min_action_support,
                                fallbacks=fallbacks,
                                head=head,
                                seed=args.seed,
                            ),
                            f"chart_{mode}",
                        )
                print(f"built chart atlas head={head} policies={len(policies)}", flush=True)

        if "graph" in structural_families:
            for head in chart_heads:
                for include_response, mode in [(False, "context"), (True, "response")]:
                    for chart_count in chart_counts:
                        count_name = str(chart_count)
                        name = (
                            f"graph_{head}_{mode}_c{count_name}_g{args.graph_neighbors}_"
                            f"d{args.graph_diffusion_dim}_strict"
                        )
                        add(
                            name,
                            evaluate_graph_atlas_powered(
                                adapted_rows,
                                folds=folds,
                                selected_by_fold=selected_by_fold,
                                raw_state_vectors=state_vectors,
                                chart_count=chart_count,
                                pca_dim=args.chart_pca_dim,
                                top_charts=args.chart_top_charts,
                                ridge_alpha=args.chart_ridge_alpha,
                                include_response_margin=include_response,
                                threshold=args.chart_threshold,
                                min_chart_support=args.chart_min_support,
                                min_action_support=args.chart_min_action_support,
                                fallbacks=fallbacks,
                                head=head,
                                graph_neighbors=graph_neighbors,
                                diffusion_dim=args.graph_diffusion_dim,
                                diffusion_time=args.graph_diffusion_time,
                                pre_pca_dim=args.graph_pre_pca_dim,
                                scenario_edge_weight=args.graph_scenario_edge_weight,
                                seed=args.seed,
                            ),
                            f"graph_{mode}",
                        )
                print(f"built graph atlas head={head} policies={len(policies)}", flush=True)

        if "learned_graph" in structural_families:
            for include_response, mode in [(False, "context"), (True, "response")]:
                name = (
                    f"learned_graph_{mode}_h{args.learned_graph_hidden_dim}_"
                    f"l{args.learned_graph_layers}_g{args.graph_neighbors}_strict"
                )
                add(
                    name,
                    evaluate_learned_graph_powered(
                        adapted_rows,
                        folds=folds,
                        selected_by_fold=selected_by_fold,
                        state_vectors=state_vectors,
                        include_response_margin=include_response,
                        threshold=args.learned_threshold,
                        graph_neighbors=graph_neighbors,
                        pre_pca_dim=args.graph_pre_pca_dim,
                        scenario_edge_weight=args.graph_scenario_edge_weight,
                        hidden_dim=args.learned_graph_hidden_dim,
                        graph_layers=args.learned_graph_layers,
                        dropout=args.learned_graph_dropout,
                        lr=args.learned_graph_lr,
                        weight_decay=args.learned_graph_weight_decay,
                        epochs=args.learned_graph_epochs,
                        extension_neighbors=learned_extension_neighbors,
                        extension_mix=args.learned_graph_extension_mix,
                        seed=args.seed,
                    ),
                    f"learned_graph_{mode}",
                )
            print(f"built learned graph policies={len(policies)}", flush=True)

    references = choose_reference_names(policies, top_fixed=8)
    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "action_response": str(action_response_path.resolve()),
        "action_response_sha256": file_sha256(action_response_path),
        "activations": str(Path(args.activations).resolve()) if args.activations else None,
        "activations_sha256": file_sha256(Path(args.activations)) if args.activations and structural_families else None,
        "activation_meta": activation_meta,
        "typed_graph": str(Path(args.typed_graph).resolve()) if args.typed_graph else None,
        "typed_graph_sha256": file_sha256(Path(args.typed_graph)) if args.typed_graph else None,
        "input_meta": input_meta,
        "adapted_summary": adapted_summary,
        "provenance": git_provenance([
            Path(__file__),
            Path("experiments/powered150_action_response_adapter.py"),
            Path("src/geoprobe/control/atlas_action_selector.py"),
            Path("src/geoprobe/data/activation_bank.py"),
            action_response_path,
            *( [Path(args.typed_graph)] if args.typed_graph else [] ),
            *( [Path(args.activations)] if args.activations and structural_families else [] ),
        ]),
        "folds": folds,
        "top_actions_per_target": args.top_actions_per_target,
        "structural_families": structural_families,
        "structural_config": {
            "layers": parse_int_csv(args.layers),
            "query_turn": args.query_turn,
            "query_phase": args.query_phase,
            "chart_counts": parse_chart_counts(args.chart_counts),
            "chart_heads": parse_csv(args.chart_heads),
            "graph_neighbors": args.graph_neighbors,
            "graph_diffusion_dim": args.graph_diffusion_dim,
            "graph_pre_pca_dim": args.graph_pre_pca_dim,
            "learned_graph_epochs": args.learned_graph_epochs,
        },
        "next_structural_candidates_not_run": structural_candidate_records(),
        "n_rows": len(rows),
        "n_conversations": len(grouped_by_conversation(rows)),
        "family_balance": dict(Counter(row["family"] for row in rows if is_abstain(row))),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if is_abstain(row))),
        "policy_types": policy_types,
        "comparison_references": references,
        "selected_action_keys_by_fold": [
            {target: [list(key) for key in keys] for target, keys in selected.items()}
            for selected in selected_by_fold
        ],
        "policies": {
            name: {
                "type": policy_types[name],
                "summary": policy["summary"],
                "folds": policy.get("folds"),
                "choices": [slim_eval_choice(row) for row in policy["choices"]],
            }
            for name, policy in policies.items()
        },
        "paired_gaps_vs_references": compute_gaps(
            policies,
            references=references,
            seed=args.seed,
            bootstrap=args.bootstrap,
        ),
        "notes": [
            "Decision-token margin-response selector evaluation only; not full generation.",
            "Typed-graph policies are imported from the separate durable typed-graph run if provided.",
            "Chart/graph/learned-graph policies reuse the existing 4-bit selector classes through the powered150 adapter.",
            "Response-aware and oracle policies are ceilings/probes.",
            "Source sweep used oracle/feasibility routing.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {out_path}")
    if args.report:
        Path(args.report).write_text(render_markdown(out, top_n=40))
        print(f"report -> {args.report}")
    for name, item in sorted(policies.items(), key=rank_key, reverse=True)[:25]:
        s = item["summary"]
        print(
            f"{name:58s} {policy_types[name]:18s} "
            f"fixes={s['fixes_error']:3d}/{s['deceptive_n']} "
            f"harm={s['honest_harms']:3d}/{s['honest_n']} "
            f"reward={s['mean_reward']:.4f} aligned={s['mean_aligned_margin']:.4f}"
        )


if __name__ == "__main__":
    main()
