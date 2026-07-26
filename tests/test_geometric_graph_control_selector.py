from __future__ import annotations

import numpy as np

from experiments.geometric_graph_control_selector import (
    build_graph_policies,
    evaluate_graph_atlas,
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


def graph_rows() -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
    rows = []
    states = {}
    for family_idx, family in enumerate(["a", "b", "c", "held"]):
        jitter = 0.05 * family_idx
        for region, winner, center in [
            ("left", "bidir_tangent", np.array([0.0, jitter, 0.0])),
            ("right", "global_probe_gated", np.array([10.0, jitter, 0.0])),
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


def test_graph_atlas_selects_region_actions_from_diffusion_coordinates():
    rows, states = graph_rows()
    result = evaluate_graph_atlas(
        rows,
        raw_state_vectors=states,
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
        graph_neighbors=2,
        diffusion_dim=2,
        diffusion_time=1.0,
        pre_pca_dim=2,
        scenario_edge_weight=0.0,
        seed=0,
    )
    assert result["summary"]["deceptive_strict_fixes"] == 8
    assert result["summary"]["chosen_methods"] == {
        "bidir_tangent": 4,
        "global_probe_gated": 4,
    }
    assert all(fold["graph"]["n_train_states"] == 6 for fold in result["folds"].values())


def test_graph_policy_builder_includes_context_and_response_variants():
    rows, states = graph_rows()
    policies = build_graph_policies(
        rows,
        raw_state_vectors=states,
        chart_counts=[2],
        pca_dim=2,
        top_charts=1,
        ridge_alpha=0.01,
        threshold=0.0,
        min_chart_support=1.0,
        min_action_support=1.0,
        fallbacks=["full", "method_layer", "method", "route"],
        heads=["mean", "ridge"],
        graph_neighbors=2,
        diffusion_dim=2,
        diffusion_time=1.0,
        pre_pca_dim=2,
        scenario_edge_weight=0.0,
        seed=0,
    )
    assert "graph_mean_context_c2_g2_d2_strict" in policies
    assert "graph_ridge_response_c2_g2_d2_strict" in policies
    assert policies["graph_mean_context_c2_g2_d2_strict"]["summary"]["deceptive_strict_fixes"] == 8
