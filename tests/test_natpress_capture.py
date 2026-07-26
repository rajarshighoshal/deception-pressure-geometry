"""Torch-free tests for natpress capture planning (validator, subset, bytes, cap)."""

from __future__ import annotations

import pytest

from geoprobe.models.natpress_capture import (
    ATTENTION_LAYERS,
    RESIDUAL_LAYERS,
    attention_bytes,
    build_natpress_capture_plan,
    canonical_json_sha256,
    derive_token_annotations,
    peak_attention_scores_bytes,
    residual_bytes,
    select_attention_subset,
    validate_natpress_capture_plan,
    validate_natpress_capture_row,
)

_BOT, _SH, _EH, _EOT = 128000, 128006, 128007, 128009
_SYS, _USR, _AST = 9125, 882, 78191


def _message(role_token: int, content: list[int], *, eot: bool = True) -> list[int]:
    return [_SH, role_token, _EH, *content, *([_EOT] if eot else [])]


def _token_ids(content_per_turn: int = 2) -> list[int]:
    ids = [_BOT] + _message(_SYS, [11, 12])
    token = 1000
    for _ in range(7):
        ids += _message(_USR, [token, token + 1][:content_per_turn])
        token += 10
        ids += _message(_AST, [token, token + 1][:content_per_turn])
        token += 10
    return ids


def _assistant_segments(ids: list[int]) -> list[list[int]]:
    segments, i = [], 0
    while i < len(ids):
        if ids[i] == _SH and ids[i + 1] == _AST:
            j = i + 3
            seg = []
            while j < len(ids) and ids[j] != _SH:
                seg.append(ids[j])
                j += 1
            segments.append(seg)
            i = j
        else:
            i += 1
    return segments


def _row(scenario: str, arm: str, sample: int, *, protocol: str = "natpress_v1",
         prefix: str = "natpress") -> dict:
    ids = _token_ids()
    return {
        "conversation_id": f"{prefix}:fam:{scenario}:{arm}:s{sample}",
        "protocol": protocol,
        "family": "fam",
        "scenario_id": scenario,
        "arm": arm,
        "sample_index": sample,
        "token_ids": ids,
        "turns": [
            {"turn_index": i, "assistant_token_ids": seg}
            for i, seg in enumerate(_assistant_segments(ids))
        ],
    }


def test_byte_formulas_hand_computed():
    assert residual_bytes(100) == 4 * 100 * 4096 * 2
    assert attention_bytes(100) == 2 * 32 * (100 * 101 // 2) * 2
    assert peak_attention_scores_bytes(2300) == 32 * 2300 * 2300 * 2


def test_annotations_roles_and_turns():
    ids = _token_ids()
    ann = derive_token_annotations(ids)
    assert len(ann["token_role_ids"]) == len(ids)
    assert ann["token_role_ids"][0] == 0 and ann["token_turn_ids"][0] == -1
    assert ann["token_role_ids"][1:4] == [1, 1, 1]  # system header
    assert max(ann["token_turn_ids"]) == 14  # system + 7 user + 7 assistant
    with pytest.raises(ValueError, match="begin_of_text"):
        derive_token_annotations(ids[1:])
    with pytest.raises(ValueError, match="messages"):
        derive_token_annotations(ids[: ids.index(_SH, 5)])  # system only


def test_row_validator_accepts_and_rejects():
    row = _row("scen_01", "smooth", 10)
    summary = validate_natpress_capture_row(row)
    assert summary["token_length"] == len(row["token_ids"])
    assert summary["arm"] == "smooth"

    bad = _row("scen_01", "smooth", 10)
    bad["turns"] = bad["turns"][:6]
    with pytest.raises(ValueError, match="expected 7 turns"):
        validate_natpress_capture_row(bad)

    bad = _row("scen_01", "smooth", 10)
    bad["turns"][3]["assistant_token_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="not length-capped"):
        validate_natpress_capture_row(bad)

    # length-capped turns are the ONLY sanctioned re-tokenization mismatch
    capped = _row("scen_01", "smooth", 10)
    capped["sampling"] = {"max_new_tokens_per_turn": 3}
    capped["turns"][3]["assistant_token_ids"] = [1, 2, 3]
    summary_capped = validate_natpress_capture_row(capped)
    assert summary_capped["n_boundary_retokenized_turns"] == 1

    bad = _row("scen_01", "smooth", 10)
    bad["protocol"] = "mystery_v9"
    with pytest.raises(ValueError, match="unsupported protocol"):
        validate_natpress_capture_row(bad)


def _banks(n_scenarios: int = 2):
    scripted, adaptive = [], []
    for s in range(n_scenarios):
        scenario = f"scen_{s:02d}"
        for arm, samples in (("smooth", [11, 10]), ("benign", [10]), ("step", [10]),
                             ("latedump", [10])):
            for sample in samples:
                scripted.append(_row(scenario, arm, sample))
                adaptive.append(_row(scenario, arm, sample + 10,
                                     protocol="natpress_v3_adaptive", prefix="natpressv3"))
    return scripted, adaptive


def test_subset_lowest_sample_index_per_bank():
    scripted, adaptive = _banks()
    subset = select_attention_subset(scripted, adaptive)
    assert len(subset) == 4  # 2 scenarios x 2 banks
    assert "natpress:fam:scen_00:smooth:s10" in subset
    assert "natpress:fam:scen_00:smooth:s11" not in subset
    assert "natpressv3:fam:scen_00:smooth:s20" in subset


def test_plan_build_totals_and_validation():
    scripted, adaptive = _banks()
    plan = build_natpress_capture_plan(scripted, adaptive)
    validate_natpress_capture_plan(plan)
    assert plan["n_rows"] == len(scripted) + len(adaptive)
    assert plan["n_attention_rows"] == 4
    T = len(scripted[0]["token_ids"])
    expected_residual = plan["n_rows"] * residual_bytes(T, n_layers=len(RESIDUAL_LAYERS))
    assert plan["byte_projection"]["total_residual_bytes"] == expected_residual
    expected_attention = 4 * attention_bytes(T, n_layers=len(ATTENTION_LAYERS))
    assert plan["byte_projection"]["total_attention_bytes"] == expected_attention
    lengths = [row["token_length"] for row in plan["rows"]]
    assert lengths == sorted(lengths)


def test_plan_cap_enforcement():
    scripted, adaptive = _banks()
    with pytest.raises(ValueError, match="exceed the cap"):
        build_natpress_capture_plan(scripted, adaptive, max_bytes=1000)


def test_plan_rejects_duplicate_conversation_ids():
    scripted, adaptive = _banks()
    with pytest.raises(ValueError, match="duplicate conversation_id"):
        build_natpress_capture_plan(scripted + [scripted[0]], adaptive)


def test_plan_sha_binds_content():
    scripted, adaptive = _banks()
    plan = build_natpress_capture_plan(scripted, adaptive)
    tampered = dict(plan)
    tampered["n_rows"] = 999
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_natpress_capture_plan(tampered)


def test_tokenization_fingerprint_deterministic():
    row = _row("scen_01", "smooth", 10)
    a = validate_natpress_capture_row(row)["token_ids_sha256"]
    b = validate_natpress_capture_row(dict(row))["token_ids_sha256"]
    assert a == b == canonical_json_sha256(row["token_ids"])


def test_emitted_artifact_with_provenance_still_validates():
    # cmd_plan adds git_provenance AFTER computing plan_sha256; the loaded artifact must
    # still self-validate (regression for the 20260721T112209Z pod failure).
    scripted, adaptive = _banks()
    plan = build_natpress_capture_plan(scripted, adaptive)
    artifact = dict(plan)
    artifact["git_provenance"] = {"git_hash": "deadbeef", "git_dirty": True}
    validate_natpress_capture_plan(artifact)  # must not raise
