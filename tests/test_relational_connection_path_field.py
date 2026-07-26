from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from geoprobe.eval.relational_connection_path_field import (
    RelationalConnectionPathFieldError,
    build_connection_path_prediction_ledger,
    canonical_sha256,
    score_connection_path_prediction_ledger,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.geometry.relational_connection_path_distance import (
    complete_path_view_distances,
)


def _signature(value: float, inventory_hash: str) -> dict[str, object]:
    rows = []
    for relation_name, view, side in (
        ("residual.L12", "residual", "symmetric"),
        ("attention.L12.H0", "attention", "left"),
        ("transport.L12-L16", "layer_transport", "symmetric"),
    ):
        rows.append(
            {
                "relation_name": relation_name,
                "view": view,
                "side": side,
                "selected_rank": 1,
                **{
                    component: {
                        "defined": True,
                        "status": ["supported"],
                        "coordinates": [value],
                        "missing_mask": [False],
                    }
                    for component in ("I", "C", "O")
                },
            }
        )
    return {"inventory_hash": inventory_hash, "relation_sides": rows}


def _bank() -> dict[str, object]:
    inventory_hash = "a" * 64
    paths: list[dict[str, object]] = []
    for family_index in range(20):
        family = f"family-{family_index:02d}"
        fold = f"outer_{family_index % 5 + 1}"
        jitter = family_index * 0.003
        for suffix, outcome, value, knowledge in (
            ("h", "HONEST", 0.05 + jitter, True),
            ("d", "DECEPTIVE", 0.95 + jitter, True),
            (
                "e",
                "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
                0.50 + jitter,
                False,
            ),
        ):
            event_id = f"event-{family_index:02d}-{suffix}"
            paths.append(
                {
                    "event_id": event_id,
                    "scenario_id": f"scenario-{family_index:02d}-{suffix}",
                    "family": family,
                    "fold": fold,
                    "prefix_state_sha256": f"prefix-{event_id}",
                    "source_reference_id": f"reference-{event_id}",
                    "source_section_sha256": f"section-{event_id}",
                    "design_cell": {
                        "true_status": "PASS",
                        "desired_status": "FAIL",
                        "baseline_knowledge_correct": knowledge,
                    },
                    "signature": _signature(value, inventory_hash),
                    "class_counts": {
                        label: int(label == outcome) for label in OUTCOME_CLASSES
                    },
                }
            )
    bank: dict[str, object] = {
        "schema_version": 1,
        "kind": "relational_complete_path_bank",
        "policy_contract": "artifact_only_cross_fitted",
        "confirmatory": False,
        "inventory_hash": inventory_hash,
        "paths": paths,
        "coverage": {"path_count": 60},
    }
    bank["bank_sha256"] = canonical_sha256(bank)
    return bank


def _spectral_ledger(bank: dict[str, object]) -> dict[str, object]:
    probabilities = {label: 1.0 / len(OUTCOME_CLASSES) for label in OUTCOME_CLASSES}
    ledger: dict[str, object] = {
        "schema_version": 1,
        "kind": "relational_intrinsic_spectral_prediction_ledger",
        "predictions": [
            {
                "event_ids": [path["event_id"]],
                "family": path["family"],
                "fold": path["fold"],
                "scenario_id": path["scenario_id"],
                "prefix_state_sha256": path["prefix_state_sha256"],
                "compact_section_content_sha256": path[
                    "source_section_sha256"
                ],
                "probabilities": {"equal_view": probabilities},
            }
            for path in bank["paths"]
        ],
    }
    ledger["prediction_ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _rehash_bank(bank: dict[str, object]) -> None:
    bank.pop("bank_sha256", None)
    bank["bank_sha256"] = canonical_sha256(bank)


def test_heldout_query_labels_do_not_change_their_fold_predictions() -> None:
    bank = _bank()
    spectral = _spectral_ledger(bank)
    original = build_connection_path_prediction_ledger(
        complete_path_bank=bank, spectral_prediction_ledger=spectral
    )
    mutated = deepcopy(bank)
    for path in mutated["paths"]:
        if path["fold"] == "outer_1":
            path["class_counts"] = {
                label: int(label == "SKIP") for label in OUTCOME_CLASSES
            }
    _rehash_bank(mutated)
    changed = build_connection_path_prediction_ledger(
        complete_path_bank=mutated, spectral_prediction_ledger=spectral
    )
    original_rows = {
        row["event_id"]: row["probabilities"]
        for row in original["predictions"]
        if row["fold"] == "outer_1"
    }
    changed_rows = {
        row["event_id"]: row["probabilities"]
        for row in changed["predictions"]
        if row["fold"] == "outer_1"
    }
    assert changed_rows == original_rows


def test_scores_zero_hd_cell_and_detects_synthetic_path_signal() -> None:
    bank = _bank()
    ledger = build_connection_path_prediction_ledger(
        complete_path_bank=bank,
        spectral_prediction_ledger=_spectral_ledger(bank),
    )
    score = score_connection_path_prediction_ledger(
        prediction_ledger=ledger, complete_path_bank=bank
    )
    primary = score["aggregate"][score["primary_model"]]
    assert primary["honest_deceptive_slice"]["auroc"] == pytest.approx(1.0)
    assert (
        score["full_path_gain_over_comparators"]["design_cell"][
            "honest_deceptive_conditional_log_loss_gain"
        ]
        > 0.0
    )
    error_cell = next(
        cell
        for cell in score["per_design_cell"].values()
        if not cell["design_cell"]["baseline_knowledge_correct"]
    )
    assert error_cell["metrics"][score["primary_model"]][
        "honest_deceptive_slice"
    ]["event_count"] == 0
    assert error_cell["full_path_gain_over_comparators"]["design_cell"][
        "honest_deceptive_conditional_log_loss_gain"
    ] is None
    for row in ledger["predictions"]:
        for probabilities in row["probabilities"].values():
            assert set(probabilities) == set(OUTCOME_CLASSES)
            assert sum(probabilities.values()) == pytest.approx(1.0)


def test_cached_primary_prediction_matches_direct_uncached_reference() -> None:
    bank = _bank()
    ledger = build_connection_path_prediction_ledger(
        complete_path_bank=bank,
        spectral_prediction_ledger=_spectral_ledger(bank),
    )
    paths = {path["event_id"]: path for path in bank["paths"]}
    query = next(row for row in ledger["predictions"] if row["fold"] == "outer_1")
    query_path = paths[query["event_id"]]
    query_cell = query_path["design_cell"]
    training = [
        path
        for path in paths.values()
        if path["fold"] != "outer_1" and path["design_cell"] == query_cell
    ]
    fold_training = [
        path for path in paths.values() if path["fold"] != "outer_1"
    ]
    pair_distances = [
        complete_path_view_distances(left["signature"], right["signature"])[
            "full_distance"
        ]
        for left_index, left in enumerate(fold_training)
        for right in fold_training[left_index + 1 :]
        if left["design_cell"] == right["design_cell"]
    ]
    bandwidth = float(np.median(np.asarray(pair_distances, dtype=np.float64)))
    scores = np.full(len(OUTCOME_CLASSES), 0.5, dtype=np.float64)
    for train_path in training:
        distance = complete_path_view_distances(
            query_path["signature"], train_path["signature"]
        )["full_distance"]
        weight = math.exp(-(distance**2) / (2.0 * bandwidth**2))
        scores += weight * np.asarray(
            [train_path["class_counts"][label] for label in OUTCOME_CLASSES],
            dtype=np.float64,
        )
    expected = scores / scores.sum()
    observed = query["probabilities"]["full_path_design_conditioned"]
    assert [observed[label] for label in OUTCOME_CLASSES] == pytest.approx(expected)


def test_rejects_confirmatory_or_different_bound_bank() -> None:
    bank = _bank()
    spectral = _spectral_ledger(bank)
    ledger = build_connection_path_prediction_ledger(
        complete_path_bank=bank, spectral_prediction_ledger=spectral
    )
    confirmatory = deepcopy(bank)
    confirmatory["confirmatory"] = True
    _rehash_bank(confirmatory)
    with pytest.raises(RelationalConnectionPathFieldError, match="confirmatory"):
        build_connection_path_prediction_ledger(
            complete_path_bank=confirmatory,
            spectral_prediction_ledger=spectral,
        )
    different = deepcopy(bank)
    different["coverage"] = {"path_count": 60, "note": "different binding"}
    _rehash_bank(different)
    with pytest.raises(RelationalConnectionPathFieldError, match="different"):
        score_connection_path_prediction_ledger(
            prediction_ledger=ledger, complete_path_bank=different
        )
