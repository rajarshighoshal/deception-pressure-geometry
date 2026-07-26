from __future__ import annotations

import numpy as np

from experiments.learned_graph_control_model import (
    build_learned_graph_policies,
    evaluate_learned_graph,
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


def learned_graph_rows() -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
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


def test_learned_graph_context_can_learn_region_action_interaction():
    rows, states = learned_graph_rows()
    result = evaluate_learned_graph(
        rows,
        state_vectors=states,
        include_response_margin=False,
        threshold=0.0,
        graph_neighbors=2,
        pre_pca_dim=2,
        scenario_edge_weight=0.0,
        hidden_dim=16,
        graph_layers=1,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        epochs=220,
        extension_neighbors=2,
        extension_mix=0.5,
        seed=0,
    )
    assert result["summary"]["deceptive_strict_fixes"] >= 7


def test_learned_graph_policy_builder_has_context_and_response_modes():
    rows, states = learned_graph_rows()
    policies = build_learned_graph_policies(
        rows,
        state_vectors=states,
        modes=["context", "response"],
        threshold=0.0,
        graph_neighbors=2,
        pre_pca_dim=2,
        scenario_edge_weight=0.0,
        hidden_dim=16,
        graph_layers=1,
        dropout=0.0,
        lr=0.03,
        weight_decay=0.0,
        epochs=120,
        extension_neighbors=2,
        extension_mix=0.5,
        seed=0,
    )
    assert "learned_graph_context_h16_l1_g2_strict" in policies
    assert "learned_graph_response_h16_l1_g2_strict" in policies
