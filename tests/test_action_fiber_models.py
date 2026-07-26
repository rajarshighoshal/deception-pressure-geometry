from __future__ import annotations

import numpy as np

from geoprobe.control import geometry_map as gm
from geoprobe.control.action_fiber_models import (
    ActionFiberModelConfig,
    LocalActionAttentionSelector,
    TypedActionFiberControlNetSelector,
    _relations_between,
)


def make_row(
    cid: str,
    family: str,
    region: str,
    *,
    method: str,
    target_status: str | None = "PASS",
    layer: int | None = 16,
    alpha: float = 24.0,
    reward: float = 0.0,
    fixes: bool = False,
    harm: bool = False,
    status_class: str = "false_FAIL",
) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": f"scenario_{family}_{region}",
        "family": family,
        "method": method,
        "target_status": target_status,
        "route_action": "steer_to_PASS" if target_status == "PASS" else "steer_to_FAIL",
        "layer": layer,
        "alpha": alpha if layer is not None else 0.0,
        "status_class": status_class,
        "reported_status_before": "FAIL" if status_class == "false_FAIL" else "PASS",
        "true_status": "PASS" if status_class == "false_FAIL" else "FAIL",
        "desired_status": target_status or "NONE",
        "target_margin_sign": 1 if target_status == "PASS" else -1 if target_status == "FAIL" else 0,
        "base_margin": -2.0 if status_class == "false_FAIL" else 2.0,
        "gate_score_PASS_minus_FAIL": -2.0 if status_class == "false_FAIL" else 2.0,
        "gate_proba_PASS": 0.1 if status_class == "false_FAIL" else 0.9,
        "reward": reward,
        "strict_reward": reward,
        "fixes_error": fixes,
        "harms_honest": harm,
        "projection_fraction": 0.5,
        "cos_to_raw": 0.2,
        "neighbor_distance_mean": 0.3,
        "neighbor_distance_max": 0.6,
        "direction_info": {"direction_convention": "unit", "n_train_points": 4},
        "pc_axis_0": 1.0 if region == "left" else -1.0,
    }


def toy_rows_and_states() -> tuple[list[dict], dict[gm.StateLayerKey, np.ndarray]]:
    rows: list[dict] = []
    states: dict[gm.StateLayerKey, np.ndarray] = {}
    for family_idx, family in enumerate(["train_a", "train_b", "held"]):
        for region, offset in [("left", -3.0), ("right", 3.0)]:
            cid = f"{family}_{region}"
            for layer in (16, 24):
                states[(cid, layer)] = np.asarray(
                    [offset, 0.2 * family_idx, float(layer) / 32.0, 1.0],
                    dtype=np.float32,
                )
            rows.append(
                make_row(
                    cid,
                    family,
                    region,
                    method="baseline",
                    target_status=None,
                    layer=None,
                    reward=0.0,
                )
            )
            rows.append(
                make_row(
                    cid,
                    family,
                    region,
                    method="global_mean",
                    target_status="PASS",
                    layer=16,
                    alpha=24.0,
                    reward=1.0 if region == "left" else 0.0,
                    fixes=region == "left",
                )
            )
            rows.append(
                make_row(
                    cid,
                    family,
                    region,
                    method="bidir_linear",
                    target_status="PASS",
                    layer=24,
                    alpha=48.0,
                    reward=1.0 if region == "right" else 0.0,
                    fixes=region == "right",
                )
            )
            rows.append(
                make_row(
                    cid,
                    family,
                    region,
                    method="bidir_linear",
                    target_status="FAIL",
                    layer=24,
                    alpha=48.0,
                    reward=-1.0,
                    harm=True,
                    status_class="honest_PASS",
                )
            )
    return rows, states


def cfg(**overrides: object) -> ActionFiberModelConfig:
    values = {
        "hidden_dim": 12,
        "epochs": 20,
        "lr": 3e-3,
        "weight_decay": 0.0,
        "harm_penalty": 1.0,
        "top_k": 3,
        "neighbor_pool_multiplier": 3,
        "device": "cpu",
        "seed": 7,
    }
    values.update(overrides)
    return ActionFiberModelConfig(**values)


def test_local_action_attention_scores_heldout_candidates_and_explains_neighbors():
    rows, states = toy_rows_and_states()
    train = [row for row in rows if row["family"] != "held"]
    test = [row for row in rows if row["conversation_id"] == "held_left"]
    selector = LocalActionAttentionSelector(cfg()).fit(train, states)
    scores = selector.score(test, states)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores[1:]).all()
    assert scores[0] < -1e8
    choice = gm.choose_by_scores(test, scores, threshold=-1e9)
    explanation = selector.explain_choice(choice, states)
    assert explanation["model"] == "local_action_attention"
    assert explanation["top_neighbors"]


def test_typed_action_fiber_scores_heldout_candidates_and_explains_relations():
    rows, states = toy_rows_and_states()
    train = [row for row in rows if row["family"] != "held"]
    test = [row for row in rows if row["conversation_id"] == "held_right"]
    selector = TypedActionFiberControlNetSelector(cfg()).fit(train, states)
    scores = selector.score(test, states)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores[1:]).all()
    choice = gm.choose_by_scores(test, scores, threshold=-1e9)
    explanation = selector.explain_choice(choice, states)
    assert explanation["model"] == "typed_action_fiber_control_net"
    assert "same_full_action" in explanation["relations"]
    assert "soft_target_partner" in explanation["relations"]


def test_typed_action_fiber_no_relations_ablation_zeroes_relation_evidence():
    rows, states = toy_rows_and_states()
    train = [row for row in rows if row["family"] != "held"]
    test = [row for row in rows if row["conversation_id"] == "held_right"]
    selector = TypedActionFiberControlNetSelector(
        cfg(use_relation_features=False),
    ).fit(train, states)
    choice = gm.choose_by_scores(test, selector.score(test, states), threshold=-1e9)
    explanation = selector.explain_choice(choice, states)
    for values in explanation["relations"].values():
        assert values["support_fraction"] == 0.0
        assert values["mean_fix"] == 0.0
        assert values["mean_harm"] == 0.0
        assert values["mean_reward"] == 0.0


def test_action_fiber_scores_are_candidate_order_equivariant():
    rows, states = toy_rows_and_states()
    train = [row for row in rows if row["family"] != "held"]
    group = [row for row in rows if row["conversation_id"] == "held_left"]
    selector = TypedActionFiberControlNetSelector(cfg(epochs=5)).fit(train, states)
    forward = selector.score(group, states)
    backward = selector.score(list(reversed(group)), states)
    np.testing.assert_allclose(forward, backward[::-1], rtol=0.0, atol=1e-6)


def test_action_fiber_context_scores_ignore_mutated_query_response_fields():
    rows, states = toy_rows_and_states()
    train = [row for row in rows if row["family"] != "held"]
    group = [row for row in rows if row["conversation_id"] == "held_left"]
    mutated = []
    for row in group:
        copy = dict(row)
        copy.update({
            "final_margin": 999.0,
            "decision_margin": 999.0,
            "delta_margin": 999.0,
            "aligned_margin": 999.0,
            "aligned_delta_margin": 999.0,
            "reward": -123.0,
            "strict_reward": -123.0,
            "correct_after": not bool(row.get("correct_after")),
            "fixes_error": not bool(row.get("fixes_error")),
            "fixes_strict": not bool(row.get("fixes_strict")),
            "harms_honest": not bool(row.get("harms_honest")),
            "harms_honest_strict": not bool(row.get("harms_honest_strict")),
        })
        mutated.append(copy)
    selectors = [
        LocalActionAttentionSelector(cfg()).fit(train, states),
        TypedActionFiberControlNetSelector(cfg()).fit(train, states),
    ]
    for selector in selectors:
        before = selector.score(group, states)
        after = selector.score(mutated, states)
        np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-7)


def test_relation_buckets_capture_fiber_structure():
    row = make_row("a", "fam", "left", method="bidir_linear", target_status="PASS", layer=16, alpha=24)
    same = make_row("b", "fam", "left", method="bidir_linear", target_status="PASS", layer=16, alpha=24)
    layer_neighbor = make_row("c", "fam", "left", method="bidir_linear", target_status="PASS", layer=24, alpha=24)
    alpha_neighbor = make_row("d", "fam", "left", method="bidir_linear", target_status="PASS", layer=16, alpha=48)
    partner = make_row("e", "fam", "left", method="bidir_linear", target_status="FAIL", layer=16, alpha=24)
    assert "same_full_action" in _relations_between(row, same)
    assert "same_method_adjacent_layer" in _relations_between(row, layer_neighbor)
    assert "same_method_adjacent_alpha" in _relations_between(row, alpha_neighbor)
    assert "soft_target_partner" in _relations_between(row, partner)


def test_neighbor_state_projection_keeps_attention_memory_small():
    rows, states = toy_rows_and_states()
    expanded_states = {
        key: np.pad(value.astype(np.float32), (0, 60))
        for key, value in states.items()
    }
    train = [row for row in rows if row["family"] != "held"]
    selector = LocalActionAttentionSelector(cfg(state_dim=5, epochs=1)).fit(train, expanded_states)
    assert selector.neighbors_.x_.shape[1] == 5
