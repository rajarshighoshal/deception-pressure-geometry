from __future__ import annotations

import numpy as np

from experiments.geometric_chart_atlas_selector import (
    build_chart_policies,
    evaluate_chart_atlas,
)
from tests.test_learned_generation_action_policy import make_row


METHODS = [
    "baseline",
    "bidir_tangent",
    "global_mean_gated",
    "global_probe_gated",
    "random_gated",
    "bidir_linear",
]


def chart_rows() -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
    rows = []
    states = {}
    for family in ["a", "b", "held"]:
        for region, winner, center in [
            ("left", "bidir_tangent", np.array([0.0, 0.0, 0.0])),
            ("right", "global_probe_gated", np.array([10.0, 0.0, 0.0])),
        ]:
            cid = f"{family}_{region}"
            states[(cid, 20)] = center
            for method in METHODS:
                strict = method == winner
                rows.append(
                    make_row(
                        cid,
                        family,
                        method,
                        "false_PASS",
                        status=strict,
                        strict=strict,
                        route="steer_to_FAIL",
                    )
                )
    return rows, states


def test_chart_atlas_selects_region_specific_actions_without_fixed_k():
    rows, states = chart_rows()
    result = evaluate_chart_atlas(
        rows,
        state_vectors=states,
        chart_count=2,
        pca_dim=2,
        top_charts=1,
        ridge_alpha=0.01,
        include_response_margin=False,
        threshold=0.0,
        min_chart_support=1.0,
        min_action_support=1.0,
        fallbacks=["full", "method_layer", "method", "route"],
        head="mean",
        seed=0,
    )
    assert result["summary"]["deceptive_strict_fixes"] == 6
    assert result["summary"]["chosen_methods"] == {
        "bidir_tangent": 3,
        "global_probe_gated": 3,
    }


def test_chart_policy_builder_includes_context_and_response_variants():
    rows, states = chart_rows()
    policies = build_chart_policies(
        rows,
        state_vectors=states,
        chart_counts=["auto"],
        pca_dim=2,
        top_charts=1,
        ridge_alpha=0.01,
        threshold=0.0,
        min_chart_support=1.0,
        min_action_support=1.0,
        fallbacks=["full", "method_layer", "method", "route"],
        heads=["mean", "ridge"],
        seed=0,
    )
    assert "chart_mean_context_cauto_d2_strict" in policies
    assert "chart_ridge_response_cauto_d2_strict" in policies
    assert policies["chart_mean_context_cauto_d2_strict"]["summary"]["deceptive_strict_fixes"] == 6


def test_atlas_action_selector_promotion_pins_scores():
    """Phase 4A: the promoted src AtlasActionSelector reproduces the pre-move ChartAtlasSelector
    scores exactly, and the old experiment path aliases to it. Guards the 'looks identical' trap
    that cancelled the ChartAtlas merge — do NOT alias/migrate this into chart_atlas.ChartAtlas."""
    from geoprobe.control.atlas_action_selector import AtlasActionSelector

    from experiments.geometric_chart_atlas_selector import ChartAtlasSelector

    assert ChartAtlasSelector is AtlasActionSelector  # compat alias for the old import path
    rows, states = chart_rows()
    sel = AtlasActionSelector(
        chart_count=2, pca_dim=2, top_charts=1, ridge_alpha=0.01, include_response_margin=False,
        objective="strict_reward", min_chart_support=1.0, min_action_support=1.0,
        fallbacks=["full", "method_layer", "method", "route"], head="mean", seed=0,
    )
    sel.fit(rows, states)
    # golden scores captured from the ORIGINAL ChartAtlasSelector before the promotion (seed=0)
    scores = {i: sel.score_one(rows[i], states)[0] for i in (1, 3, 7)}
    assert scores == {1: 1.0, 3: 0.0, 7: 0.25}
