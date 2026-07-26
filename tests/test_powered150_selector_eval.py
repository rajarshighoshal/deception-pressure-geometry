"""Regression tests for powered150_selector_eval structural integration.

These cover the seam between raw ``select_action_keys`` output and the adapted
row namespace used by the chart/graph/learned-graph selectors. The alias bug
(selected ``global_probe`` silently dropped because adapted rows call it
``global_probe_gated``) lived exactly here and was invisible to the per-module
tests, so exercise the eval-level helpers directly.
"""
from __future__ import annotations

import numpy as np

from experiments.powered150_action_response_adapter import adapt_row
from experiments.powered150_selector_eval import (
    canonical_selected_keys,
    evaluate_chart_atlas_powered,
    prune_adapted_to_selected_library,
)
from geoprobe.control.action_response import (
    best_key_for_target,
    evaluate_route_matched_fixed_coordinate,
    select_action_keys,
)


def _raw_row(
    cid: str,
    family: str,
    *,
    method: str,
    target_status: str | None,
    status_class: str,
    base_margin: float,
    final_margin: float,
    reward: float,
    layer: int | None = 8,
    alpha: float = 24.0,
) -> dict:
    true_status = "FAIL" if status_class in ("false_PASS", "honest_FAIL") else "PASS"
    route = "steer_to_FAIL" if true_status == "FAIL" else "steer_to_PASS"
    return {
        "conversation_id": cid,
        "scenario_id": cid,
        "family": family,
        "arm": "p5",
        "method": method,
        "target_status": target_status,
        "route_action": route,
        "layer": layer,
        "alpha": alpha,
        "status_class": status_class,
        "reported_status_before": "PASS" if base_margin > 0 else "FAIL",
        "true_status": true_status,
        "desired_status": true_status,
        "desired_margin_sign": 1 if true_status == "PASS" else -1,
        "target_margin_sign": (1 if target_status == "PASS" else -1) if target_status else 0,
        "base_margin": base_margin,
        "final_margin": final_margin,
        "delta_margin": final_margin - base_margin,
        "correct_after": reward > 0.0,
        "fixes_error": reward > 0.0 and status_class.startswith("false_"),
        "harms_honest": False,
        "reward": reward,
        "projection_fraction": 0.5,
        "cos_to_raw": 0.1,
        "neighbor_distance_mean": 0.2,
        "neighbor_distance_max": 0.4,
    }


def _gated_global_rows() -> list[dict]:
    """Two train families plus a held family whose winning actions are aliased globals."""
    rows: list[dict] = []
    for cid, family in [("fp_a", "famA"), ("fp_b", "famB"), ("fp_c", "held")]:
        rows.append(
            _raw_row(cid, family, method="abstain", target_status=None,
                     status_class="false_PASS", base_margin=2.0, final_margin=2.0,
                     reward=0.0, layer=None, alpha=0.0)
        )
        rows.append(
            _raw_row(cid, family, method="global_probe", target_status="FAIL",
                     status_class="false_PASS", base_margin=2.0, final_margin=-3.0, reward=1.0)
        )
    for cid, family in [("ff_a", "famA"), ("ff_b", "famB"), ("ff_c", "held")]:
        rows.append(
            _raw_row(cid, family, method="abstain", target_status=None,
                     status_class="false_FAIL", base_margin=-2.0, final_margin=-2.0,
                     reward=0.0, layer=None, alpha=0.0)
        )
        rows.append(
            _raw_row(cid, family, method="global_mean", target_status="PASS",
                     status_class="false_FAIL", base_margin=-2.0, final_margin=3.0, reward=1.0)
        )
    return rows


def test_canonical_selected_keys_maps_all_gated_aliases():
    raw = _gated_global_rows()
    selected = select_action_keys(raw, top_per_target=8, objective="reward")
    canonical = canonical_selected_keys(selected)
    methods = {key[0] for key in canonical}
    assert methods == {"global_probe_gated", "global_mean_gated"}
    passthrough = canonical_selected_keys({"FAIL": [("bidir_linear", "FAIL", 8, 24.0)]})
    assert passthrough == {("bidir_linear", "FAIL", 8, 24.0)}


def test_prune_keeps_selected_global_actions_after_aliasing():
    raw = _gated_global_rows()
    selected = select_action_keys(raw, top_per_target=8, objective="reward")
    adapted = [adapt_row(row) for row in raw]
    pruned = prune_adapted_to_selected_library(adapted, selected)
    kept = {row["method"] for row in pruned if row["method"] != "baseline"}
    assert kept == {"global_probe_gated", "global_mean_gated"}


def test_chart_atlas_powered_can_choose_gated_global_action():
    raw = _gated_global_rows()
    selected = select_action_keys(
        [row for row in raw if row["family"] != "held"],
        top_per_target=8,
        objective="reward",
    )
    state_vectors = {}
    for idx, cid in enumerate(sorted({row["conversation_id"] for row in raw})):
        offset = -3.0 if cid.startswith("fp_") else 3.0
        state_vectors[(cid, 8)] = np.asarray([offset + 0.01 * idx, 0.1, 0.0], dtype=np.float64)
    result = evaluate_chart_atlas_powered(
        [adapt_row(row) for row in raw],
        folds=[["held"]],
        selected_by_fold=[selected],
        state_vectors=state_vectors,
        chart_count=2,
        pca_dim=2,
        top_charts=2,
        ridge_alpha=1.0,
        include_response_margin=False,
        threshold=-1e9,
        min_chart_support=0.0,
        min_action_support=0.0,
        fallbacks=["route"],
        head="mean",
        seed=0,
    )
    chosen = result["summary"]["chosen_methods"]
    assert set(chosen) & {"global_probe_gated", "global_mean_gated"}, chosen


def test_route_matched_selection_excludes_counter_target_honest_actions():
    rows = [
        _raw_row(
            "false_good",
            "famA",
            method="good",
            target_status="PASS",
            status_class="false_FAIL",
            base_margin=-2.0,
            final_margin=3.0,
            reward=1.0,
        ),
        _raw_row(
            "honest_counter_good",
            "famA",
            method="good",
            target_status="PASS",
            status_class="honest_FAIL",
            base_margin=-2.0,
            final_margin=3.0,
            reward=-1.0,
        ),
        _raw_row(
            "false_weak",
            "famA",
            method="weak",
            target_status="PASS",
            status_class="false_FAIL",
            base_margin=-2.0,
            final_margin=1.0,
            reward=0.5,
        ),
        _raw_row(
            "honest_counter_weak",
            "famA",
            method="weak",
            target_status="PASS",
            status_class="honest_FAIL",
            base_margin=-2.0,
            final_margin=1.0,
            reward=0.5,
        ),
    ]

    legacy = best_key_for_target(rows, "PASS", objective="reward")
    matched = best_key_for_target(
        rows,
        "PASS",
        objective="reward",
        require_route_match=True,
    )
    selected = select_action_keys(
        rows,
        top_per_target=1,
        objective="reward",
        require_route_match=True,
    )

    assert legacy is not None and legacy[0] == "weak"
    assert matched is not None and matched[0] == "good"
    assert selected["PASS"][0][0] == "good"


def test_route_matched_fixed_coordinate_is_fit_on_train_families_only():
    rows: list[dict] = []
    for family in ("train", "held"):
        for suffix, status_class, target in (
            ("pass", "false_FAIL", "PASS"),
            ("fail", "false_PASS", "FAIL"),
        ):
            cid = f"{family}_{suffix}"
            rows.extend(
                [
                    _raw_row(
                        cid,
                        family,
                        method="abstain",
                        target_status=None,
                        status_class=status_class,
                        base_margin=-2.0 if target == "PASS" else 2.0,
                        final_margin=-2.0 if target == "PASS" else 2.0,
                        reward=0.0,
                        layer=None,
                        alpha=0.0,
                    ),
                    _raw_row(
                        cid,
                        family,
                        method="bidir_linear",
                        target_status=target,
                        status_class=status_class,
                        base_margin=-2.0 if target == "PASS" else 2.0,
                        final_margin=3.0 if target == "PASS" else -3.0,
                        reward=1.0,
                        layer=16,
                        alpha=96.0,
                    ),
                    _raw_row(
                        cid,
                        family,
                        method="global_mean",
                        target_status=target,
                        status_class=status_class,
                        base_margin=-2.0 if target == "PASS" else 2.0,
                        final_margin=1.0 if target == "PASS" else -1.0,
                        reward=0.5,
                        layer=8,
                        alpha=48.0,
                    ),
                ]
            )

    result = evaluate_route_matched_fixed_coordinate(
        rows,
        folds=[["held"]],
        objective="reward",
        methods={"bidir_linear", "global_mean"},
    )

    assert result["folds"]["0"]["coordinate"] == ["bidir_linear", 16, 96.0]
    assert result["summary"]["fixes_error"] == 2
    assert result["summary"]["honest_harms"] == 0
