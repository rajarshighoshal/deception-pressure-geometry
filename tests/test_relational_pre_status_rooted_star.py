from __future__ import annotations

import numpy as np

from geoprobe.data.relational_pre_status_rooted_star import build_status_rooted_star_views
from geoprobe.data.relational_prefix_store import (
    CompactTensorLocators,
    RawRelationalPrefixCapture,
    RelationalPrefixReference,
)


def _reference() -> RelationalPrefixReference:
    return RelationalPrefixReference(
        reference_id="a" * 64, realization_sha256="b" * 64, canonical_realization_id="b" * 64,
        equality_status="verified_exact_capture", occurrence_id="occurrence", equivalent_occurrence_ids=("occurrence",),
        field_event_id="status-event", field_name="status", turn_index=2, conversation_id="conversation",
        family="family", family_fold="outer_1", split="train", scenario_id="scenario", orbit_id="orbit",
        sample_index=0, prefix_state_sha256="c" * 64, prefix_token_ids_sha256="d" * 64,
        prefix_token_stop=18, packed_attention_prefix_stop=171, source_row_sha256="e" * 64,
        source_tensor_object_path="rows/row.safetensors", source_tensor_sha256="f" * 64,
        capture_contract_sha256="0" * 64, layers=(12, 16, 19, 20), intervention_history=("A", "B"),
        pressure_exposed=True, intervention_token_indices=(3, 4),
        intervention_mask_scope="whole_intervention_bearing_user_message",
        compact=CompactTensorLocators("x", "1" * 64, "a", 0, "b", "c", "d", 0, "e", "f"),
    )


def _raw() -> RawRelationalPrefixCapture:
    # Messages are contiguous; turn-2's frozen status prefix occupies message 6 offsets 4 and 5.
    messages = np.array([0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6], dtype=np.int16)
    details = ["chat_source"] * len(messages)
    details[2] = "model_sampled_status_action"
    details[7] = "model_sampled_caveat_action"
    details[9] = "model_sampled_status_action"
    details[10] = "model_sampled_caveat_action"
    details[-2:] = ["environment_status_prefix", "environment_status_prefix"]
    origins = ["environment" if detail == "environment_status_prefix" else "model_sample" if "sampled" in detail else "chat_source" for detail in details]
    token_ids = np.arange(100, 118, dtype=np.int32)
    token_ids[-2:] = [2583, 25]
    residuals = {layer: (np.arange(18 * 3, dtype=np.float32).reshape(18, 3) + layer) for layer in (12, 16, 19, 20)}
    attentions: dict[int, np.ndarray] = {}
    for layer in residuals:
        packed = np.zeros((2, 171), dtype=np.float32)
        packed[:, 153:171] = np.arange(1, 19, dtype=np.float32)
        attentions[layer] = packed
    return RawRelationalPrefixCapture(
        object_id="object", residuals=residuals, attentions=attentions, token_ids=token_ids,
        position_ids=np.arange(18, dtype=np.int32), token_role_ids=np.array([0] * 16 + [2, 2], dtype=np.uint8),
        token_turn_ids=np.array([-1] * 11 + [2] * 7, dtype=np.int16), token_message_ids=messages,
        token_span_flags=np.array([0] * 16 + [16, 16], dtype=np.int16), token_origins=tuple(origins),
        token_origin_details=tuple(details), anchor_index=17,
    )


def test_rooted_stars_exclude_all_prior_actions_and_mask_intervention_messages() -> None:
    reference, raw = _reference(), _raw()
    full, primary = build_status_rooted_star_views(reference, raw)
    assert full.name == "action_free_full_context"
    assert primary.name == "intervention_masked_action_free"
    assert set(full.retained_indices.tolist()).isdisjoint({2, 7, 9, 10})
    assert {3, 4}.issubset(set(full.retained_indices.tolist()))
    assert set(primary.retained_indices.tolist()).isdisjoint({2, 3, 4, 7, 9, 10})
    assert full.root_residuals.shape == (4, 3)
    assert str(full.root_residuals.dtype) == "torch.bfloat16"
    assert full.root_to_context_residual_distances.shape[0] == 4
    assert primary.incoming_attention.shape[-1] == primary.retained_indices.numel()
    assert primary.removed_attention_mass.min().item() > 0
    assert all(len(view.star_sha256) == 64 for view in (full, primary))


def test_rooted_star_hashes_are_deterministic() -> None:
    first = build_status_rooted_star_views(_reference(), _raw())
    second = build_status_rooted_star_views(_reference(), _raw())
    assert [view.star_sha256 for view in first] == [view.star_sha256 for view in second]
    assert [view.tensor_hashes for view in first] == [view.tensor_hashes for view in second]
