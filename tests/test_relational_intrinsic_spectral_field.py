from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from geoprobe.eval.relational_intrinsic_spectral_field import (
    RelationalIntrinsicSpectralFieldError,
    build_intrinsic_spectral_fold_predictions,
    build_intrinsic_spectral_prediction_ledger,
    score_intrinsic_spectral_prediction_ledger,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _relation_names() -> list[tuple[str, str]]:
    values = [(f"residual.L{layer}", "residual") for layer in (12, 16, 19, 20)]
    values.extend(
        (f"attention.L{layer}.H{head}", "attention")
        for layer in (12, 16, 19, 20)
        for head in range(32)
    )
    values.extend(
        (name, "layer_transport")
        for name in (
            "transport.L12->L16",
            "transport.L16->L19",
            "transport.L19->L20",
        )
    )
    return values


def _profile(state_id: str, section: str, *, deceptive: bool) -> dict:
    leading = [2.0, 1.0] if deceptive else [3.0, 1.0]
    relations = []
    for name, view in _relation_names():
        signed = None
        if view == "layer_transport":
            signed = [-leading[0], leading[1]] if deceptive else [leading[0], -1.0]
        relations.append(
            {
                "relation_name": name,
                "view": view,
                "status": "valid",
                "leading_spectral_values": leading,
                "signed_leading_spectral_values": signed,
                "squared_energy_total": 10.0,
            }
        )
    return {
        "state_id": state_id,
        "compact_section_content_sha256": section,
        "relations": relations,
    }


def _counts(label: str) -> dict[str, int]:
    return {name: int(name == label) for name in OUTCOME_CLASSES}


def _quotients() -> list[dict]:
    rows = []
    for fold_index in range(1, 6):
        for deceptive in (False, True):
            label = "DECEPTIVE" if deceptive else "HONEST"
            identity = f"fold{fold_index}-{label}"
            state_id = _sha(f"state-{identity}")
            section = _sha(f"section-{identity}")
            rows.append(
                {
                    "state_id": state_id,
                    "family": f"family-{identity}",
                    "fold": f"outer_{fold_index}",
                    "scenario_id": f"scenario-{identity}",
                    "compact_section_content_sha256": section,
                    "prefix_state_sha256": _sha(f"prefix-{identity}"),
                    "event_ids": [f"event-{identity}"],
                    "class_counts": _counts(label),
                    "design_cell": {
                        "true_status": "PASS",
                        "desired_status": "FAIL",
                        "knowledge_correct": True,
                    },
                    "profile": _profile(state_id, section, deceptive=deceptive),
                }
            )
    return rows


def _selection() -> list[dict]:
    return [
        {
            "heldout_family_fold": f"outer_{fold_index}",
            "relation_name": name,
            "view": view,
            "selected_rank": 1,
            "admissible": True,
        }
        for fold_index in range(1, 6)
        for name, view in _relation_names()
    ]


def test_fold_predictions_do_not_read_heldout_counts() -> None:
    quotients = _quotients()
    policies = [
        row for row in _selection() if row["heldout_family_fold"] == "outer_1"
    ]
    original = build_intrinsic_spectral_fold_predictions(
        heldout_fold="outer_1",
        quotients=quotients,
        admitted_policies=policies,
    )
    changed = deepcopy(quotients)
    for row in changed:
        if row["fold"] == "outer_1":
            row["class_counts"] = _counts(
                "HONEST" if row["class_counts"]["DECEPTIVE"] else "DECEPTIVE"
            )

    repeated = build_intrinsic_spectral_fold_predictions(
        heldout_fold="outer_1",
        quotients=changed,
        admitted_policies=policies,
    )

    assert original == repeated


def test_intrinsic_field_scores_better_than_base_rate_on_separated_fixture() -> None:
    quotients = _quotients()
    ledger = build_intrinsic_spectral_prediction_ledger(
        quotients=quotients,
        calibration_selection=_selection(),
    )
    report = score_intrinsic_spectral_prediction_ledger(
        prediction_ledger=ledger,
        quotients=quotients,
    )

    assert len(ledger["predictions"]) == 10
    assert report["aggregate"]["equal_view"][
        "event_pooled_multiclass_log_loss"
    ] < report["aggregate"]["base_rate"]["event_pooled_multiclass_log_loss"]
    assert report["aggregate"]["equal_view"]["honest_deceptive_slice"][
        "auroc"
    ] == pytest.approx(1.0)
    assert report["aggregate"]["equal_view"]["honest_deceptive_slice"][
        "conditional_log_loss"
    ] < report["aggregate"]["base_rate"]["honest_deceptive_slice"][
        "conditional_log_loss"
    ]
    uncertainty = report["post_score_descriptive_uncertainty"]
    assert uncertainty["family_count"] == 10
    assert uncertainty["resamples"] == 10_000
    assert uncertainty["comparisons"]["base_rate"][
        "event_pooled_log_loss_gain"
    ]["point_estimate"] == pytest.approx(
        report["equal_view_gain_over_comparators"]["base_rate"][
            "event_pooled_log_loss_gain"
        ]
    )


def test_fold_local_selected_rank_changes_only_that_fold_contract() -> None:
    selection = _selection()
    for row in selection:
        if row["heldout_family_fold"] == "outer_3":
            row["selected_rank"] = 2

    ledger = build_intrinsic_spectral_prediction_ledger(
        quotients=_quotients(),
        calibration_selection=selection,
    )

    folds = {row["heldout_fold"]: row for row in ledger["folds"]}
    assert folds["outer_3"]["admitted_relation_counts"] == {
        "residual": 4,
        "attention": 128,
        "layer_transport": 3,
    }
    assert folds["outer_3"]["view_scales"] != folds["outer_2"]["view_scales"]


def test_rejects_identity_leakage_across_folds() -> None:
    quotients = _quotients()
    quotients[2]["family"] = quotients[0]["family"]

    with pytest.raises(RelationalIntrinsicSpectralFieldError, match="crosses"):
        build_intrinsic_spectral_prediction_ledger(
            quotients=quotients,
            calibration_selection=_selection(),
        )


def test_rejects_incomplete_fold_relation_policy_inventory() -> None:
    with pytest.raises(RelationalIntrinsicSpectralFieldError, match="135"):
        build_intrinsic_spectral_prediction_ledger(
            quotients=_quotients(),
            calibration_selection=_selection()[:-1],
        )
