from __future__ import annotations

import copy
import math

import pytest

from src.geoprobe.geometry.relational_connection_path_distance import (
    _coerce_policy,
    _project_diagnostic,
    complete_path_view_distances,
    stable_common_rank_relation_sides,
    build_complete_path_signature,
)



def _selection_row(
    relation_name: str,
    view: str,
    fold: str,
    selected_rank: int,
    *,
    status: str = "admitted",
    admissible: bool = True,
) -> dict[str, object]:
    return {
        "relation_name": relation_name,
        "view": view,
        "selected_rank": selected_rank,
        "status": status,
        "admissible": admissible,
        "heldout_family_fold": fold,
    }


def _forward_diagnostic(*, cosines=(0.9,), status: str = "supported", projector: float = 0.2, polar: float = 0.1) -> dict[str, object]:
    if status in {"supported", "ill_conditioned"}:
        return {
            "status": status,
            "principal_angle_cosines": list(cosines),
            "normalized_transported_projector_discrepancy": projector,
            "normalized_polar_residual": polar,
        }
    return {"status": status}


def _relation_inventory_rows() -> list[dict[str, object]]:
    rows = []
    for fold in ["outer_1", "outer_2", "outer_3", "outer_4", "outer_5"]:
        rows.append(_selection_row("attention_left", "attention", fold, 2))
        rows.append(_selection_row("attention_right", "attention", fold, 3))
        rows.append(_selection_row("residual_core", "residual", fold, 1))
        rows.append(_selection_row("layer_core", "layer_transport", fold, 1))
        rows.append(_selection_row("invariant_bad", "attention", fold, 2, status="not_found"))
        rows.append(_selection_row("rank_shift", "residual", fold, 3 if fold != "outer_5" else 2))
    return rows


def test_stable_common_rank_relation_sides_expands_sides_and_filters_on_admitted_rows():
    stable = stable_common_rank_relation_sides(_relation_inventory_rows(), validate_counts=False)

    assert stable["relation_count"] == 4
    assert stable["relation_side_count"] == 6
    relation_names = {row["relation_name"] for row in stable["relation_sides"]}
    assert relation_names == {"attention_left", "attention_right", "residual_core", "layer_core"}

    attention_sides = [
        row["side"] for row in stable["relation_sides"] if row["view"] == "attention"
    ]
    assert sorted(attention_sides) == ["left", "left", "right", "right"]


def test_project_diagnostic_returns_rank_mismatch_status_for_low_rank_supported_diagnostic():
    mismatch = _project_diagnostic(
        _forward_diagnostic(cosines=(0.95,)),
        selected_rank=2,
    )
    assert mismatch.status == "rank_mismatch"
    assert mismatch.coordinates is None


def test_build_complete_path_signature_maps_missing_outgoing_branch_to_c_and_o_missing():
    policy_rows = [_selection_row("residual_core", "residual", fold, 1) for fold in ["outer_1", "outer_2", "outer_3", "outer_4", "outer_5"]]
    policy = _coerce_policy(policy_rows)

    incoming = [
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic()}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic()}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic()}],
    ]
    replay = [
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.8,))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.8,))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.8,))}],
    ]

    aa = {
        "residual_core|residual|symmetric": {
            "forward": _forward_diagnostic(cosines=(0.7,))
        }
    }
    ab = {"residual_core|residual|symmetric": {"status": "missing"}}

    signature = build_complete_path_signature(incoming, replay, aa, ab, policy)
    relation_side = signature["relation_sides"][0]

    assert relation_side["C"]["defined"] is False
    assert relation_side["O"]["defined"] is False
    assert "missing" in relation_side["C"]["status"]
    assert "missing" in relation_side["O"]["status"]


def test_build_complete_path_signature_promotes_rank_mismatch_status_for_i_component():
    policy_rows = [_selection_row("residual_core", "residual", fold, 2) for fold in ["outer_1", "outer_2", "outer_3", "outer_4", "outer_5"]]
    policy = _coerce_policy(policy_rows)

    incoming = [
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.95, 0.90))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.96, 0.88))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.91,))}],
    ]
    replay = [
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.40, 0.80))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.41, 0.79))}],
        [{"relation_name": "residual_core", "view": "residual", "side": "symmetric", "forward": _forward_diagnostic(cosines=(0.42, 0.78))}],
    ]
    aa = {"residual_core|residual|symmetric": {"status": "missing"}}
    ab = {"residual_core|residual|symmetric": {"status": "missing"}}

    signature = build_complete_path_signature(incoming, replay, aa, ab, policy)
    i_statuses = signature["relation_sides"][0]["I"]["status"]

    assert "rank_mismatch" in i_statuses


def test_complete_path_view_distances_uses_rms_equal_construction():
    policy_rows = [_selection_row("residual_core", "residual", fold, 1) for fold in ["outer_1", "outer_2", "outer_3", "outer_4", "outer_5"]]
    inventory = stable_common_rank_relation_sides(policy_rows, validate_counts=False)
    side = inventory["relation_sides"][0]

    base_side_row = {
        "relation_name": side["relation_name"],
        "view": side["view"],
        "side": side["side"],
        "selected_rank": side["selected_rank"],
        "I": {"defined": True, "status": ["supported"], "coordinates": [0.0]},
        "C": {"defined": True, "status": ["supported"], "coordinates": [0.0]},
        "O": {"defined": True, "status": ["supported"], "coordinates": [0.0]},
    }
    left = {
        "inventory_hash": inventory["inventory_hash"],
        "relation_sides": [copy.deepcopy(base_side_row)],
    }
    right = {
        "inventory_hash": inventory["inventory_hash"],
        "relation_sides": [
            {
                **base_side_row,
                "I": {"defined": True, "status": ["supported"], "coordinates": [2.0]},
            }
        ],
    }

    result = complete_path_view_distances(left, right, views=("residual",), components=("I", "C", "O"))

    expected = math.sqrt((0.5) / 3.0)
    assert result["full_distance"] == pytest.approx(expected)
    assert result["component_distances"]["I"] == pytest.approx(1.0 / math.sqrt(2.0))
    assert result["component_distances"]["C"] == pytest.approx(0.0)
    assert result["component_distances"]["O"] == pytest.approx(0.0)
    assert result["component_ablations"]["I"] == pytest.approx(0.0)
    assert result["component_ablations"]["C"] == pytest.approx(0.5)
    assert result["view_distances"]["residual"] == pytest.approx(expected)
    assert result["view_ablations"]["residual"] == pytest.approx(1.0)
