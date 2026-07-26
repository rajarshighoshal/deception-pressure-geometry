"""Guarded neural-over-atlas residual policy for powered150.

This is the focused repair after the fair-IID neural trial:

default to the fold-local chart atlas policy, and allow the neural controller to
override only when its own train-calibrated score margin over the chart action is
large enough.  The override threshold is chosen on train-fold rows only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometric_chart_atlas_selector import ChartAtlasSelector, load_state_vectors  # noqa: E402
from experiments.powered150_action_response_adapter import adapt_row  # noqa: E402
from experiments.powered150_selector_eval import (  # noqa: E402
    assert_complete_fold_choices,
    expected_heldout_cids,
    parse_int_csv,
    prune_adapted_to_selected_library,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402
from geoprobe.control import geometry_map as gm  # noqa: E402
from geoprobe.control.action_response import (  # noqa: E402
    file_sha256,
    load_action_response,
    make_family_folds,
    paired_gap,
    select_action_keys,
)
from geoprobe.control.chart_atlas import ChartAtlasConfig  # noqa: E402
from geoprobe.control.structural_neural import (  # noqa: E402
    EquivariantChartActionFieldSelector,
    StructuralNeuralConfig,
    resolve_device,
)


def action_sig(row: dict | None) -> str:
    if not row:
        return "NONE"
    return f"{row.get('method')}:{row.get('target_status')}@L{row.get('layer')}:a{row.get('alpha')}"


def compact_choice(row: dict) -> dict[str, Any]:
    out = gm.slim_choice(row)
    for key in (
        "policy_source",
        "override_tau",
        "feature_gate_tau",
        "feature_gate_score",
        "feature_gate_label",
        "neural_margin_over_chart",
        "neural_choice_sig",
        "chart_choice_sig",
        "neural_choice_score",
        "chart_choice_neural_score",
        "chart_choice_chart_score",
        "neural_choice_chart_score",
        "chart_prediction_std",
        "chart_count",
        "chart_pca_dim",
    ):
        if key in row:
            out[key] = row[key]
    return out


def chart_scores(
    candidates: list[dict],
    selector: ChartAtlasSelector,
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    scores = np.full(len(candidates), -1e9, dtype=np.float64)
    metas: list[dict] = [{} for _ in candidates]
    for idx, row in enumerate(candidates):
        if gm.is_baseline(row):
            continue
        score, meta = selector.score_one(row, state_vectors)
        scores[idx] = float(score)
        metas[idx] = dict(meta)
    return scores, metas


def choose_index(scores: np.ndarray, *, threshold: float) -> int | None:
    if scores.size == 0:
        return None
    idx = int(np.argmax(scores))
    if float(scores[idx]) <= threshold:
        return None
    return idx


def baseline_index(candidates: list[dict]) -> int:
    for idx, row in enumerate(candidates):
        if gm.is_baseline(row):
            return idx
    raise ValueError("candidate group has no baseline row")


def safe_num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def top_chart_stats(meta: dict, prefix: str) -> dict[str, float | str]:
    top = meta.get("chart_top") if isinstance(meta, dict) else None
    if not isinstance(top, list) or not top:
        return {
            f"{prefix}_chart_reason": str(meta.get("chart_reason") or "none") if isinstance(meta, dict) else "none",
            f"{prefix}_top_weight": 0.0,
            f"{prefix}_top_support": 0.0,
            f"{prefix}_top_action_support": 0.0,
            f"{prefix}_top_action_count": 0.0,
            f"{prefix}_top_prediction": -1.0,
            f"{prefix}_weighted_support": 0.0,
            f"{prefix}_weighted_action_support": 0.0,
            f"{prefix}_weighted_prediction": -1.0,
            f"{prefix}_prediction_std": safe_num(meta.get("chart_prediction_std")) if isinstance(meta, dict) else 0.0,
        }
    first = dict(top[0])
    weights = np.asarray([safe_num(item.get("weight")) for item in top], dtype=np.float64)
    supports = np.asarray([safe_num(item.get("support")) for item in top], dtype=np.float64)
    action_supports = np.asarray([safe_num(item.get("action_support")) for item in top], dtype=np.float64)
    predictions = np.asarray([safe_num(item.get("prediction"), -1.0) for item in top], dtype=np.float64)
    denom = max(float(weights.sum()), 1e-12)
    return {
        f"{prefix}_chart_reason": "ok",
        f"{prefix}_top_weight": safe_num(first.get("weight")),
        f"{prefix}_top_support": safe_num(first.get("support")),
        f"{prefix}_top_action_support": safe_num(first.get("action_support")),
        f"{prefix}_top_action_count": safe_num(first.get("action_count")),
        f"{prefix}_top_prediction": safe_num(first.get("prediction"), -1.0),
        f"{prefix}_weighted_support": float(np.sum(weights * supports) / denom),
        f"{prefix}_weighted_action_support": float(np.sum(weights * action_supports) / denom),
        f"{prefix}_weighted_prediction": float(np.sum(weights * predictions) / denom),
        f"{prefix}_prediction_std": safe_num(meta.get("chart_prediction_std")) if isinstance(meta, dict) else 0.0,
    }


def action_coord(row: dict, key: str, default: float = -1.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    return safe_num(value, default)


def gate_features(scored: dict) -> dict[str, float | str]:
    """Deployable features for a chart-vs-neural residual override decision."""
    candidates = scored["candidates"]
    chart_idx = int(scored["chart_idx"])
    neural_idx = int(scored["neural_idx"])
    baseline_idx = int(scored["baseline_idx"])
    chart = candidates[chart_idx]
    neural = candidates[neural_idx]
    chart_meta = scored["chart_metas"][chart_idx] if chart_idx != baseline_idx else {}
    neural_meta = scored["chart_metas"][neural_idx] if neural_idx != baseline_idx else {}
    chart_score = float(scored["chart_scores"][chart_idx])
    neural_chart_score = float(scored["chart_scores"][neural_idx])
    chart_neural_score = float(scored["neural_scores"][chart_idx])
    neural_score = float(scored["neural_scores"][neural_idx])
    chart_layer = action_coord(chart, "layer")
    neural_layer = action_coord(neural, "layer")
    chart_alpha = action_coord(chart, "alpha", 0.0)
    neural_alpha = action_coord(neural, "alpha", 0.0)
    out: dict[str, float | str] = {
        "route_action": str(chart.get("route_action") or "unknown"),
        "reported_status_before": str(chart.get("reported_status_before") or "UNKNOWN"),
        "chart_method": str(chart.get("method") or "NONE"),
        "neural_method": str(neural.get("method") or "NONE"),
        "chart_target": str(chart.get("target_status") or "NONE"),
        "neural_target": str(neural.get("target_status") or "NONE"),
        "chart_layer": chart_layer,
        "neural_layer": neural_layer,
        "chart_alpha": chart_alpha,
        "neural_alpha": neural_alpha,
        "target_agree": float(chart.get("target_status") == neural.get("target_status")),
        "method_agree": float(chart.get("method") == neural.get("method")),
        "layer_abs_diff": abs(neural_layer - chart_layer),
        "alpha_abs_diff": abs(neural_alpha - chart_alpha),
        "chart_score": chart_score,
        "neural_chart_score": neural_chart_score,
        "chart_neural_score": chart_neural_score,
        "neural_score": neural_score,
        "neural_minus_chart_neural_score": neural_score - chart_neural_score,
        "neural_minus_chart_chart_score": neural_chart_score - chart_score,
        "score_disagreement": (neural_score - chart_neural_score) - (chart_score - neural_chart_score),
        "chart_is_baseline": float(chart_idx == baseline_idx),
        "neural_is_baseline": float(neural_idx == baseline_idx),
        "gate_score_PASS_minus_FAIL": safe_num(chart.get("gate_score_PASS_minus_FAIL")),
        "abs_gate_score_PASS_minus_FAIL": abs(safe_num(chart.get("gate_score_PASS_minus_FAIL"))),
        "gate_proba_PASS": safe_num(chart.get("gate_proba_PASS"), 0.5),
        "base_margin": safe_num(chart.get("base_margin")),
        "abs_base_margin": abs(safe_num(chart.get("base_margin"))),
    }
    out.update(top_chart_stats(chart_meta, "chart_choice"))
    out.update(top_chart_stats(neural_meta, "neural_choice"))
    return out


def override_label(scored: dict) -> int:
    candidates = scored["candidates"]
    chart_reward = gm.target_value(candidates[int(scored["chart_idx"])], "reward")
    neural_reward = gm.target_value(candidates[int(scored["neural_idx"])], "reward")
    return int(neural_reward > chart_reward + 1e-12)


class FeatureOverrideGate:
    def __init__(self, *, c: float, max_iter: int) -> None:
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.constant_: float | None = None
        self.model_: Any | None = None

    def fit(self, features: list[dict[str, float | str]], labels: list[int]) -> "FeatureOverrideGate":
        if not features:
            self.constant_ = 0.0
            self.model_ = None
            return self
        unique = sorted(set(int(label) for label in labels))
        if len(unique) < 2:
            self.constant_ = float(unique[0])
            self.model_ = None
            return self
        self.constant_ = None
        self.model_ = make_pipeline(
            DictVectorizer(sparse=True),
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=self.c,
                class_weight="balanced",
                max_iter=self.max_iter,
                solver="liblinear",
                random_state=0,
            ),
        )
        self.model_.fit(features, labels)
        return self

    def predict_proba(self, features: list[dict[str, float | str]]) -> np.ndarray:
        if not features:
            return np.asarray([], dtype=np.float64)
        if self.model_ is None:
            value = 0.0 if self.constant_ is None else self.constant_
            return np.full(len(features), value, dtype=np.float64)
        return np.asarray(self.model_.predict_proba(features)[:, 1], dtype=np.float64)


def score_group(
    candidates: list[dict],
    *,
    chart_selector: ChartAtlasSelector,
    neural_selector: EquivariantChartActionFieldSelector,
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
    chart_threshold: float,
) -> dict:
    c_scores, c_metas = chart_scores(candidates, chart_selector, state_vectors)
    n_scores = neural_selector.score(candidates, state_vectors)
    bidx = baseline_index(candidates)
    chart_idx = choose_index(c_scores, threshold=chart_threshold)
    neural_idx = choose_index(n_scores, threshold=-1e9)
    if chart_idx is None:
        chart_idx = bidx
    if neural_idx is None:
        neural_idx = bidx
    chart_neural_score = float(n_scores[chart_idx]) if chart_idx != bidx else -1e9
    neural_score = float(n_scores[neural_idx]) if neural_idx != bidx else -1e9
    margin = neural_score - chart_neural_score
    return {
        "candidates": candidates,
        "chart_scores": c_scores,
        "chart_metas": c_metas,
        "neural_scores": np.asarray(n_scores, dtype=np.float64),
        "baseline_idx": bidx,
        "chart_idx": chart_idx,
        "neural_idx": neural_idx,
        "neural_margin_over_chart": float(margin),
    }


def materialize_choice(scored: dict, *, source: str, idx: int, tau: float | None = None) -> dict:
    candidates = scored["candidates"]
    chart_idx = int(scored["chart_idx"])
    neural_idx = int(scored["neural_idx"])
    row = dict(candidates[idx])
    row["policy_source"] = source
    row["override_tau"] = tau
    row["neural_margin_over_chart"] = float(scored["neural_margin_over_chart"])
    row["chart_choice_sig"] = action_sig(candidates[chart_idx])
    row["neural_choice_sig"] = action_sig(candidates[neural_idx])
    row["chart_choice_neural_score"] = float(scored["neural_scores"][chart_idx])
    row["neural_choice_score"] = float(scored["neural_scores"][neural_idx])
    row["chart_choice_chart_score"] = float(scored["chart_scores"][chart_idx])
    row["neural_choice_chart_score"] = float(scored["chart_scores"][neural_idx])
    if idx == chart_idx and scored["chart_metas"][idx]:
        row.update(scored["chart_metas"][idx])
    row["policy_score"] = row["neural_choice_score"] if source.startswith("neural") else row["chart_choice_chart_score"]
    row["policy_abstained_by_threshold"] = gm.is_baseline(row)
    return row


def feature_gate_choice(scored: dict, *, gate_score: float, tau: float) -> dict:
    use_neural = float(gate_score) > float(tau)
    if use_neural and int(scored["neural_idx"]) != int(scored["baseline_idx"]):
        row = materialize_choice(scored, source="neural_feature_gate", idx=int(scored["neural_idx"]), tau=tau)
    else:
        row = materialize_choice(scored, source="chart_feature_gate", idx=int(scored["chart_idx"]), tau=tau)
    row["feature_gate_tau"] = float(tau)
    row["feature_gate_score"] = float(gate_score)
    row["feature_gate_label"] = override_label(scored)
    return row


def guarded_choice(scored: dict, *, tau: float) -> dict:
    use_neural = float(scored["neural_margin_over_chart"]) > float(tau)
    if use_neural and int(scored["neural_idx"]) != int(scored["baseline_idx"]):
        return materialize_choice(scored, source="neural", idx=int(scored["neural_idx"]), tau=tau)
    return materialize_choice(scored, source="chart", idx=int(scored["chart_idx"]), tau=tau)


def candidate_dump(scored: dict, cid: str) -> list[dict]:
    c_scores = np.asarray(scored["chart_scores"], dtype=np.float64)
    n_scores = np.asarray(scored["neural_scores"], dtype=np.float64)
    chart_order = {int(idx): rank + 1 for rank, idx in enumerate(np.argsort(-c_scores))}
    neural_order = {int(idx): rank + 1 for rank, idx in enumerate(np.argsort(-n_scores))}
    out = []
    for idx, row in enumerate(scored["candidates"]):
        out.append({
            "conversation_id": cid,
            "method": row.get("method"),
            "target_status": row.get("target_status"),
            "layer": row.get("layer"),
            "alpha": row.get("alpha"),
            "status_class": row.get("status_class"),
            "reward": row.get("reward") if "reward" in row else row.get("strict_reward"),
            "fixes_error": bool(row.get("fixes_error")),
            "harms_honest": bool(row.get("harms_honest")),
            "chart_score": float(c_scores[idx]),
            "neural_score": float(n_scores[idx]),
            "chart_rank": int(chart_order[idx]),
            "neural_rank": int(neural_order[idx]),
            "is_chart_choice": idx == int(scored["chart_idx"]),
            "is_neural_choice": idx == int(scored["neural_idx"]),
        })
    return out


def choose_feature_tau(scored_groups: list[dict], gate_scores: np.ndarray) -> tuple[float, dict]:
    finite = gate_scores[np.isfinite(gate_scores)]
    if finite.size == 0:
        candidates = [1.1]
    else:
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 41))
        candidates = sorted(set([-0.1, 0.5, 1.1, *[float(x) for x in quantiles]]))
    best_tau = 1.1
    best_key = None
    best_summary = None
    best_sources = None
    labels = [override_label(group) for group in scored_groups]
    for tau in candidates:
        choices = [
            feature_gate_choice(group, gate_score=float(score), tau=float(tau))
            for group, score in zip(scored_groups, gate_scores)
        ]
        summary = gm.summarize_choices(choices)
        deceptive_n = max(float(summary["deceptive_n"]), 1.0)
        honest_n = max(float(summary["honest_n"]), 1.0)
        overrides = int(sum(row.get("policy_source") == "neural_feature_gate" for row in choices))
        override_rate = overrides / max(float(len(choices)), 1.0)
        key = (
            float(summary["mean_reward"]),
            -float(summary["honest_harms"]) / honest_n,
            float(summary["fixes_error"]) / deceptive_n,
            -override_rate,
            float(tau),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_tau = float(tau)
            best_summary = summary
            best_sources = {
                "chart_feature_gate": int(sum(row.get("policy_source") == "chart_feature_gate" for row in choices)),
                "neural_feature_gate": overrides,
            }
    return best_tau, {
        "tau": best_tau,
        "summary": best_summary,
        "sources": best_sources,
        "n_candidates": len(candidates),
        "positive_labels": int(sum(labels)),
        "negative_labels": int(len(labels) - sum(labels)),
        "score_min": float(np.min(finite)) if finite.size else None,
        "score_median": float(np.median(finite)) if finite.size else None,
        "score_max": float(np.max(finite)) if finite.size else None,
    }


def choose_tau(scored_groups: list[dict]) -> tuple[float, dict]:
    margins = np.asarray([group["neural_margin_over_chart"] for group in scored_groups], dtype=np.float64)
    finite = margins[np.isfinite(margins)]
    if finite.size == 0:
        candidates = [1e9]
    else:
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 41))
        candidates = sorted(set([-1e9, 0.0, 1e9, *[float(x) for x in quantiles]]))
    best_tau = 1e9
    best_key = None
    best_summary = None
    for tau in candidates:
        choices = [guarded_choice(group, tau=tau) for group in scored_groups]
        summary = gm.summarize_choices(choices)
        deceptive_n = max(float(summary["deceptive_n"]), 1.0)
        honest_n = max(float(summary["honest_n"]), 1.0)
        key = (
            float(summary["mean_reward"]),
            -float(summary["honest_harms"]) / honest_n,
            float(summary["fixes_error"]) / deceptive_n,
            -abs(float(tau)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_tau = float(tau)
            best_summary = summary
    return best_tau, {
        "tau": best_tau,
        "summary": best_summary,
        "n_candidates": len(candidates),
        "margin_min": float(np.min(finite)) if finite.size else None,
        "margin_median": float(np.median(finite)) if finite.size else None,
        "margin_max": float(np.max(finite)) if finite.size else None,
    }


def evaluate_guarded(
    rows: list[dict],
    *,
    folds: list[list[str]],
    selected_by_fold: list[dict],
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
    chart_config: dict,
    neural_config: StructuralNeuralConfig,
    chart_threshold: float,
    gate_c: float,
    gate_max_iter: int,
    dump_candidates: bool,
) -> dict[str, dict]:
    chart_choices: list[dict] = []
    neural_choices: list[dict] = []
    guarded_choices: list[dict] = []
    feature_guarded_choices: list[dict] = []
    candidate_scores: list[dict] = []
    fold_out: dict[str, dict] = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_full = [row for row in rows if gm.split_group(row) not in heldout_set]
        test_full = [row for row in rows if gm.split_group(row) in heldout_set]
        train_chart = prune_adapted_to_selected_library(train_full, selected_by_fold[fold_idx])
        train_eval = train_chart
        test = prune_adapted_to_selected_library(test_full, selected_by_fold[fold_idx])
        print(
            f"[fold {fold_idx}] fitting chart/neural | "
            f"train_full={len(train_full)} train_chart={len(train_chart)} test={len(test)}",
            flush=True,
        )
        chart_selector = ChartAtlasSelector(seed=neural_config.seed + fold_idx, **chart_config).fit(train_chart, state_vectors)
        neural_selector = EquivariantChartActionFieldSelector(
            StructuralNeuralConfig(
                chart=neural_config.chart,
                model=neural_config.model,
                include_response=neural_config.include_response,
                hidden_dim=neural_config.hidden_dim,
                latent_dim=neural_config.latent_dim,
                pre_pca_dim=neural_config.pre_pca_dim,
                graph_layers=neural_config.graph_layers,
                epochs=neural_config.epochs,
                batch_size=neural_config.batch_size,
                lr=neural_config.lr,
                weight_decay=neural_config.weight_decay,
                harm_penalty=neural_config.harm_penalty,
                ae_weight=neural_config.ae_weight,
                listwise_weight=neural_config.listwise_weight,
                listwise_temperature=neural_config.listwise_temperature,
                chart_prior_weight=neural_config.chart_prior_weight,
                chart_distill_weight=neural_config.chart_distill_weight,
                seed=neural_config.seed + fold_idx,
                device=neural_config.device,
            )
        ).fit(train_full, state_vectors)
        train_scored = [
            score_group(candidates, chart_selector=chart_selector, neural_selector=neural_selector, state_vectors=state_vectors, chart_threshold=chart_threshold)
            for candidates in gm.grouped_by_state(train_eval).values()
        ]
        tau, tau_meta = choose_tau(train_scored)
        gate = FeatureOverrideGate(c=gate_c, max_iter=gate_max_iter).fit(
            [gate_features(group) for group in train_scored],
            [override_label(group) for group in train_scored],
        )
        train_gate_scores = gate.predict_proba([gate_features(group) for group in train_scored])
        feature_tau, feature_tau_meta = choose_feature_tau(train_scored, train_gate_scores)
        print(
            f"[fold {fold_idx}] selected tau={tau:.6g} | train_guarded={fmt_summary(tau_meta['summary'])}",
            flush=True,
        )
        print(
            f"[fold {fold_idx}] selected feature_tau={feature_tau:.6g} | "
            f"train_feature_gate={fmt_summary(feature_tau_meta['summary'])}",
            flush=True,
        )
        fold_chart = []
        fold_neural = []
        fold_guarded = []
        fold_feature_guarded = []
        for cid, candidates in gm.grouped_by_state(test).items():
            scored = score_group(candidates, chart_selector=chart_selector, neural_selector=neural_selector, state_vectors=state_vectors, chart_threshold=chart_threshold)
            gate_score = float(gate.predict_proba([gate_features(scored)])[0])
            fold_chart.append(materialize_choice(scored, source="chart", idx=int(scored["chart_idx"]), tau=None))
            fold_neural.append(materialize_choice(scored, source="neural", idx=int(scored["neural_idx"]), tau=None))
            fold_guarded.append(guarded_choice(scored, tau=tau))
            fold_feature_guarded.append(feature_gate_choice(scored, gate_score=gate_score, tau=feature_tau))
            if dump_candidates:
                candidate_scores.extend(candidate_dump(scored, cid))
        expected = expected_heldout_cids(rows, heldout_set)
        assert_complete_fold_choices(fold_idx=fold_idx, choices=fold_chart, expected=expected)
        assert_complete_fold_choices(fold_idx=fold_idx, choices=fold_neural, expected=expected)
        assert_complete_fold_choices(fold_idx=fold_idx, choices=fold_guarded, expected=expected)
        assert_complete_fold_choices(fold_idx=fold_idx, choices=fold_feature_guarded, expected=expected)
        chart_choices.extend(fold_chart)
        neural_choices.extend(fold_neural)
        guarded_choices.extend(fold_guarded)
        feature_guarded_choices.extend(fold_feature_guarded)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "n_train_rows_pruned": len(train_chart),
            "n_train_rows_neural_full": len(train_full),
            "n_test_rows_pruned": len(test),
            "tau": tau,
            "tau_meta": tau_meta,
            "feature_tau": feature_tau,
            "feature_tau_meta": feature_tau_meta,
            "chart_summary": gm.summarize_choices(fold_chart),
            "neural_summary": gm.summarize_choices(fold_neural),
            "guarded_summary": gm.summarize_choices(fold_guarded),
            "feature_guarded_summary": gm.summarize_choices(fold_feature_guarded),
            "guarded_sources": {
                "chart": int(sum(row.get("policy_source") == "chart" for row in fold_guarded)),
                "neural": int(sum(row.get("policy_source") == "neural" for row in fold_guarded)),
            },
            "feature_guarded_sources": {
                "chart_feature_gate": int(sum(row.get("policy_source") == "chart_feature_gate" for row in fold_feature_guarded)),
                "neural_feature_gate": int(sum(row.get("policy_source") == "neural_feature_gate" for row in fold_feature_guarded)),
            },
        }
        print(
            f"[fold {fold_idx}] heldout guarded={fmt_summary(fold_out[str(fold_idx)]['guarded_summary'])} | "
            f"feature_gate={fmt_summary(fold_out[str(fold_idx)]['feature_guarded_summary'])}",
            flush=True,
        )
    return {
        "chart_base": {"summary": gm.summarize_choices(chart_choices), "folds": fold_out, "choices": chart_choices},
        "neural_residual": {"summary": gm.summarize_choices(neural_choices), "folds": fold_out, "choices": neural_choices},
        "chart_guarded_equivariant_neural_context": {
            "summary": gm.summarize_choices(guarded_choices),
            "folds": fold_out,
            "choices": guarded_choices,
        },
        "chart_feature_gate_equivariant_neural_context": {
            "summary": gm.summarize_choices(feature_guarded_choices),
            "folds": fold_out,
            "choices": feature_guarded_choices,
        },
        "candidate_scores": candidate_scores,
    }


def load_reference_policies(path: Path | None, names: list[str], *, keep_cids: set[str] | None = None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for name in names:
        item = payload.get("policies", {}).get(name)
        if not item:
            continue
        choices = item.get("choices", [])
        if keep_cids is not None:
            choices = [row for row in choices if str(row.get("conversation_id")) in keep_cids]
        out[name] = {
            "summary": gm.summarize_choices(choices),
            "choices": choices,
            "type": item.get("type", payload.get("policy_types", {}).get(name, "reference")),
            "imported_from": str(path),
        }
    return out


def compute_gaps(policies: dict[str, dict], references: list[str], *, seed: int, bootstrap: int) -> dict:
    out = {}
    for name, policy in policies.items():
        out[name] = {}
        for ref in references:
            if ref == name or ref not in policies:
                continue
            out[name][ref] = {
                metric: paired_gap(policy["choices"], policies[ref]["choices"], metric, seed=seed, bootstrap=bootstrap)
                for metric in ("fixes_error", "honest_harm", "reward", "aligned_margin")
            }
    return out


def fmt_summary(summary: dict) -> str:
    return (
        f"{summary['fixes_error']}/{summary['deceptive_n']} fixes, "
        f"{summary['honest_harms']}/{summary['honest_n']} harm, "
        f"reward {float(summary['mean_reward']):.4f}"
    )


def fmt_gap(gap: dict) -> str:
    lo, hi = gap["ci95"]
    return f"{float(gap['point']):+.4f} [{float(lo):+.4f}, {float(hi):+.4f}]"


def render_markdown(payload: dict) -> str:
    ranked = sorted(
        payload["policies"].items(),
        key=lambda item: (
            float(item[1]["summary"]["mean_reward"]),
            float(item[1]["summary"]["fixes_error"]),
            -float(item[1]["summary"]["honest_harms"]),
            item[0],
        ),
        reverse=True,
    )
    lines = [
        "# Powered150 Guarded Neural Residual",
        "",
        "Train-calibrated neural override over a fold-local chart atlas base policy.",
        "This is decision-token margin-field evidence, not full generation/audit.",
        "",
        "## Inputs",
        "",
        f"- Action response: `{payload['action_response']}`",
        f"- Activations: `{payload['activations']}`",
        f"- Device: `{payload['device']}`",
        f"- Rows: `{payload['n_rows']}`",
        f"- Conversations: `{payload['n_conversations']}`",
        "",
        "## Policies",
        "",
        "| policy | type | fixes | harm | reward | chosen methods |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, item in ranked:
        s = item["summary"]
        methods = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("chosen_methods", {}).items()))
        lines.append(
            f"| `{name}` | {item.get('type', 'unknown')} | "
            f"{s['fixes_error']}/{s['deceptive_n']} | {s['honest_harms']}/{s['honest_n']} | "
            f"{float(s['mean_reward']):.4f} | {methods} |"
        )
    lines.extend([
        "",
        "## Guarded Fold Thresholds",
        "",
        "| fold | scalar tau | feature tau | train scalar | train feature | heldout scalar | heldout feature | feature sources |",
        "|---:|---:|---:|---|---|---|---|---|",
    ])
    folds = payload["policies"]["chart_guarded_equivariant_neural_context"]["folds"]
    for fold_idx, fold in sorted(folds.items(), key=lambda item: int(item[0])):
        train_summary = fold["tau_meta"]["summary"]
        feature_train_summary = fold["feature_tau_meta"]["summary"]
        heldout_summary = fold["guarded_summary"]
        feature_heldout_summary = fold["feature_guarded_summary"]
        lines.append(
            f"| {fold_idx} | {float(fold['tau']):.4f} | {float(fold['feature_tau']):.4f} | "
            f"{fmt_summary(train_summary)} | {fmt_summary(feature_train_summary)} | "
            f"{fmt_summary(heldout_summary)} | {fmt_summary(feature_heldout_summary)} | "
            f"`{fold['feature_guarded_sources']}` |"
        )
    references = [
        "chart_base",
        "neural_residual",
        "chart_mean_context_cauto_d12_strict",
        "graph_mean_context_cauto_gauto_d12_strict",
        "train_best_route_full_reward",
    ]
    lines.extend([
        "",
        "## Key Paired Gaps",
        "",
        "| policy | reference | reward gap | fix gap | harm gap |",
        "|---|---|---:|---:|---:|",
    ])
    for policy_name in ("chart_feature_gate_equivariant_neural_context", "chart_guarded_equivariant_neural_context"):
        for ref in references:
            metrics = payload.get("paired_gaps", {}).get(policy_name, {}).get(ref)
            if not metrics:
                continue
            lines.append(
                f"| `{policy_name}` | `{ref}` | "
                f"{fmt_gap(metrics['reward'])} | {fmt_gap(metrics['fixes_error'])} | {fmt_gap(metrics['honest_harm'])} |"
            )
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- `tau` is selected on train-fold rows only.",
        "- The neural model trains on full train-family action-response rows; evaluation is restricted to the fold-selected deployable library.",
        "- Candidate score dumps are for review/debugging; no heldout reward is used by the guard decision.",
        "",
    ])
    return "\n".join(lines)


def balanced_debug_cids(rows: list[dict], max_cids: int) -> set[str]:
    by_family: dict[str, list[str]] = {}
    for cid, group in sorted(gm.grouped_by_state(rows).items()):
        family = str(group[0].get("family") or gm.split_group(group[0]))
        by_family.setdefault(family, []).append(cid)
    keep: list[str] = []
    families = sorted(by_family)
    cursor = 0
    while len(keep) < max_cids and families:
        progressed = False
        for family in families:
            cids = by_family[family]
            if cursor < len(cids):
                keep.append(cids[cursor])
                progressed = True
                if len(keep) >= max_cids:
                    break
        if not progressed:
            break
        cursor += 1
    return set(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-response", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--reference-selector-eval", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--layers", default="8,12,16,19,20,24,28,32")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--top-actions-per-target", type=int, default=16)
    parser.add_argument("--chart-count", default="auto")
    parser.add_argument("--chart-pca-dim", type=int, default=12)
    parser.add_argument("--chart-top-charts", type=int, default=3)
    parser.add_argument("--chart-min-support", type=float, default=4.0)
    parser.add_argument("--chart-min-action-support", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--pre-pca-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--harm-penalty", type=float, default=2.0)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--listwise-temperature", type=float, default=0.25)
    parser.add_argument("--chart-prior-weight", type=float, default=1.0)
    parser.add_argument("--chart-distill-weight", type=float, default=0.1)
    parser.add_argument("--chart-threshold", type=float, default=0.0)
    parser.add_argument("--feature-gate-c", type=float, default=0.25)
    parser.add_argument("--feature-gate-max-iter", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-cids", type=int, default=0)
    parser.add_argument("--dump-candidates", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    device = resolve_device(args.device)
    action_response_path = Path(args.action_response)
    raw_rows, input_meta = load_action_response(action_response_path)
    keep_cids = None
    if args.max_cids > 0:
        keep_cids = balanced_debug_cids(raw_rows, args.max_cids)
        raw_rows = [row for row in raw_rows if gm.state_id(row) in keep_cids]
    adapted_rows = [adapt_row(row) for row in raw_rows]
    folds = make_family_folds(raw_rows, args.folds)
    layers = parse_int_csv(args.layers)
    layer_set = set(layers)
    selection_rows = [
        row for row in raw_rows
        if row.get("layer") is None or int(row.get("layer")) in layer_set
    ]
    selected_by_fold = [
        select_action_keys(
            [row for row in selection_rows if str(row["family"]) not in set(heldout)],
            top_per_target=args.top_actions_per_target,
            objective="reward",
        )
        for heldout in folds
    ]
    activation_path = Path(args.activations)
    state_vectors, activation_meta = load_state_vectors(
        activation_path,
        layers=layers,
        query_turn=args.query_turn,
        query_phase=args.query_phase,
    )
    chart_config = {
        "chart_count": "auto" if args.chart_count == "auto" else int(args.chart_count),
        "pca_dim": args.chart_pca_dim,
        "top_charts": args.chart_top_charts,
        "ridge_alpha": 1.0,
        "include_response_margin": False,
        "objective": "strict_reward",
        "min_chart_support": args.chart_min_support,
        "min_action_support": args.chart_min_action_support,
        "fallbacks": ["full", "method_layer", "method", "route"],
        "head": "mean",
    }
    neural_chart = ChartAtlasConfig(
        chart_count="auto" if args.chart_count == "auto" else int(args.chart_count),
        pca_dim=args.chart_pca_dim,
        top_charts=args.chart_top_charts,
        head="mean",
        min_chart_support=args.chart_min_support,
        min_action_support=args.chart_min_action_support,
        include_response=False,
        objective="strict_reward",
        seed=args.seed,
    )
    neural_config = StructuralNeuralConfig(
        chart=neural_chart,
        model="equivariant_chart_action_field",
        include_response=False,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        pre_pca_dim=args.pre_pca_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        harm_penalty=args.harm_penalty,
        listwise_weight=args.listwise_weight,
        listwise_temperature=args.listwise_temperature,
        chart_prior_weight=args.chart_prior_weight,
        chart_distill_weight=args.chart_distill_weight,
        seed=args.seed,
        device=str(device),
    )
    evaluated = evaluate_guarded(
        adapted_rows,
        folds=folds,
        selected_by_fold=selected_by_fold,
        state_vectors=state_vectors,
        chart_config=chart_config,
        neural_config=neural_config,
        chart_threshold=args.chart_threshold,
        gate_c=args.feature_gate_c,
        gate_max_iter=args.feature_gate_max_iter,
        dump_candidates=args.dump_candidates,
    )
    policies = {
        "chart_base": {
            "type": "fold_local_chart_base",
            "summary": evaluated["chart_base"]["summary"],
            "folds": evaluated["chart_base"]["folds"],
            "choices": [compact_choice(row) for row in evaluated["chart_base"]["choices"]],
        },
        "neural_residual": {
            "type": "equivariant_chart_action_field_context",
            "summary": evaluated["neural_residual"]["summary"],
            "folds": evaluated["neural_residual"]["folds"],
            "choices": [compact_choice(row) for row in evaluated["neural_residual"]["choices"]],
        },
        "chart_guarded_equivariant_neural_context": {
            "type": "guarded_residual_context",
            "summary": evaluated["chart_guarded_equivariant_neural_context"]["summary"],
            "folds": evaluated["chart_guarded_equivariant_neural_context"]["folds"],
            "choices": [compact_choice(row) for row in evaluated["chart_guarded_equivariant_neural_context"]["choices"]],
        },
        "chart_feature_gate_equivariant_neural_context": {
            "type": "feature_gated_residual_context",
            "summary": evaluated["chart_feature_gate_equivariant_neural_context"]["summary"],
            "folds": evaluated["chart_feature_gate_equivariant_neural_context"]["folds"],
            "choices": [compact_choice(row) for row in evaluated["chart_feature_gate_equivariant_neural_context"]["choices"]],
        },
    }
    reference_names = [
        "chart_mean_context_cauto_d12_strict",
        "graph_mean_context_cauto_gauto_d12_strict",
        "typed_graph_context_reward",
        "train_best_route_full_reward",
        "fixed_route_bidir_linear_L16_a96",
    ]
    refs = load_reference_policies(
        Path(args.reference_selector_eval) if args.reference_selector_eval else None,
        reference_names,
        keep_cids=keep_cids,
    )
    policies.update(refs)
    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "action_response": str(action_response_path.resolve()),
        "action_response_sha256": file_sha256(action_response_path),
        "activations": str(activation_path.resolve()),
        "activations_sha256": file_sha256(activation_path),
        "reference_selector_eval": str(Path(args.reference_selector_eval).resolve()) if args.reference_selector_eval else None,
        "device": str(device),
        "input_meta": input_meta,
        "activation_meta": activation_meta,
        "provenance": git_provenance([
            Path(__file__),
            Path("src/geoprobe/control/structural_neural.py"),
            Path("src/geoprobe/control/chart_atlas.py"),
            Path("src/geoprobe/control/geometry_map.py"),
            Path("src/geoprobe/control/atlas_action_selector.py"),
            Path(_bank_mod.__file__),
            action_response_path,
            activation_path,
        ]),
        "folds": folds,
        "layers": layers,
        "n_rows": len(raw_rows),
        "n_conversations": len(gm.grouped_by_state(raw_rows)),
        "training_setup": {
            "base_policy": "fold_local_chart_mean_context",
            "override_policy": "equivariant_chart_action_field_context",
            "tau_selection": "train_fold_only",
            "feature_gate": "dictvectorizer_standardized_logistic_regression",
            "feature_gate_c": args.feature_gate_c,
            "feature_gate_max_iter": args.feature_gate_max_iter,
            "train_rows_neural": "full_train_family_action_response_field",
            "eval_rows": "fold_selected_deployable_library",
            "top_actions_per_target": args.top_actions_per_target,
            "dump_candidates": args.dump_candidates,
        },
        "policies": policies,
        "paired_gaps": compute_gaps(
            policies,
            ["chart_base", "neural_residual", *reference_names],
            seed=args.seed,
            bootstrap=args.bootstrap,
        ),
        "candidate_scores": evaluated["candidate_scores"] if args.dump_candidates else [],
        "notes": [
            "Guarded residual policy uses no heldout response labels for override decisions.",
            "Candidate score dumps are post-hoc audit records, not policy inputs.",
            "This is decision-token margin-field evaluation only.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    report_path = Path(args.report) if args.report else out_path.with_suffix(".md")
    report_path.write_text(render_markdown(out))
    print(f"saved -> {out_path}")
    print(f"report -> {report_path}")
    for name, item in sorted(
        policies.items(),
        key=lambda kv: (kv[1]["summary"]["mean_reward"], kv[1]["summary"]["fixes_error"], -kv[1]["summary"]["honest_harms"]),
        reverse=True,
    )[:20]:
        s = item["summary"]
        print(f"{name:48s} fixes={s['fixes_error']:3d}/{s['deceptive_n']} harm={s['honest_harms']:3d}/{s['honest_n']} reward={s['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
