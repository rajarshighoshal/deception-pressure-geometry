from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoprobe.eval.relational_intrinsic_spectral_field_report import (
    render_intrinsic_spectral_field_markdown,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from experiments import report_relational_intrinsic_spectral_field as report_cli


def _coverage() -> dict[str, int]:
    return {
        "quotient_count": 4,
        "event_count": 4,
        "scenario_count": 1,
        "family_count": 1,
        "fold_count": 1,
        "class_counts": {label: 0 for label in OUTCOME_CLASSES},
    }


def _metric_block(*, conditional_hd_loss: float | None, hd_auroc: float) -> dict:
    return {
        "event_pooled_multiclass_log_loss": 0.51234,
        "event_pooled_multiclass_brier": 0.09876,
        "quotient_macro_cross_entropy": 0.44556,
        "honest_deceptive_slice": {
            "conditional_log_loss": conditional_hd_loss,
            "auroc": hd_auroc,
            "average_precision": 0.63333,
        },
    }


def _aggregate_block() -> dict:
    return {
        name: _metric_block(
            conditional_hd_loss=(None if name == "equal_view" else 0.21111),
            hd_auroc=0.55500,
        )
        for name in (
            "equal_view",
            "residual",
            "attention",
            "layer_transport",
            "design_cell",
            "base_rate",
        )
    }


def _score_fixture() -> dict:
    aggregate = _aggregate_block()
    aggregate["equal_view"]["class_support"] = {"NO_ACTION": {"event_count": 0}}
    return {
        "schema_version": 1,
        "kind": "relational_intrinsic_spectral_field_score",
        "prediction_ledger_sha256": "0" * 64,
        "aggregate": aggregate,
        "equal_view_gain_over_comparators": {
            "design_cell": {
                "event_pooled_log_loss_gain": 0.10000,
                "event_pooled_brier_gain": 0.20000,
                "quotient_macro_cross_entropy_gain": 0.30000,
                "honest_deceptive_conditional_log_loss_gain": 0.40000,
            },
            "base_rate": {
                "event_pooled_log_loss_gain": -0.05000,
                "event_pooled_brier_gain": -0.01000,
                "quotient_macro_cross_entropy_gain": 0.02000,
                "honest_deceptive_conditional_log_loss_gain": 0.01000,
            },
        },
        "post_score_descriptive_uncertainty": {
            "family_count": 10,
            "resamples": 200,
            "seed": 42,
            "comparisons": {
                "design_cell": {
                    "event_pooled_log_loss_gain": {
                        "point_estimate": 0.1,
                        "percentile_95": [0.02, 0.18],
                        "fraction_positive": 0.42,
                    },
                    "honest_deceptive_conditional_log_loss_gain": {
                        "point_estimate": 0.2,
                        "percentile_95": [0.12, 0.27],
                        "fraction_positive": 0.11,
                    },
                },
                "base_rate": {
                    "event_pooled_log_loss_gain": {
                        "point_estimate": -0.03,
                        "percentile_95": [-0.1, 0.04],
                        "fraction_positive": 0.52,
                    },
                    "honest_deceptive_conditional_log_loss_gain": {
                        "point_estimate": 0.05,
                        "percentile_95": [0.01, 0.09],
                        "fraction_positive": 0.71,
                    },
                },
            },
        },
        "per_fold": {
            "outer_1": {
                "equal_view": {
                    "event_count": 4,
                    "event_pooled_multiclass_log_loss": 0.51234,
                    "honest_deceptive_slice": {
                        "auroc": 0.60000,
                    },
                },
                "design_cell": {
                    "event_count": 4,
                    "event_pooled_multiclass_log_loss": 0.61111,
                },
            }
        },
    }


def _ledger_fixture() -> dict:
    return {
        "schema_version": 1,
        "kind": "relational_intrinsic_spectral_prediction_ledger",
        "prediction_ledger_sha256": "0" * 64,
        "folds": [
            {
                "heldout_fold": "outer_1",
                "admitted_relation_counts": {
                    "residual": 1,
                    "attention": 1,
                    "layer_transport": 1,
                },
                "bandwidths": {"equal_view": 0.33333},
            }
        ],
    }


def _selection_fixture(*, include_escaped: bool) -> list[dict[str, object]]:
    rows = []
    for index in range(675):
        rows.append(
            {
                "heldout_family_fold": f"outer_{(index % 5) + 1}",
                "relation_name": f"attention.L{12 + (index % 4)}.H{index % 2}",
                "view": "attention",
                "selected_rank": 1,
                "admissible": True,
                "status": "admitted",
                "fallback_reason": (
                    'literal "escaped" quote'
                    if include_escaped and index == 0
                    else "baseline"
                ),
            }
        )
    return rows


def test_markdown_includes_explicit_null_verdict_conditional_hd_interval_lines() -> None:
    bank = {"schema_version": 1, "kind": "relational_intrinsic_outcome_bank", "coverage": _coverage()}
    score = _score_fixture()
    ledger = _ledger_fixture()

    markdown = render_intrinsic_spectral_field_markdown(score=score, ledger=ledger, bank=bank)

    assert "NO_ACTION support is 0; its discrimination is not estimable in this cohort." in markdown
    assert "| equal_view | 0.51234 | 0.09876 | 0.44556 | not estimable | 0.55500 | 0.63333 |" in markdown
    assert "[0.02000, 0.18000]" in markdown


def test_calibration_selection_streams_escaped_strings_and_enforces_675_rows(tmp_path: Path) -> None:
    full_path = tmp_path / "calibration-full.json"
    full_payload = {"selection": _selection_fixture(include_escaped=True)}
    full_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    selected = report_cli._calibration_selection(full_path)
    assert len(selected) == 675
    assert selected[0]["fallback_reason"] == 'literal "escaped" quote'

    truncated_path = tmp_path / "calibration-truncated.json"
    truncated_payload = {"selection": _selection_fixture(include_escaped=False)[:-1]}
    truncated_path.write_text(json.dumps(truncated_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration selection must contain 675 fold-relation rows"):
        report_cli._calibration_selection(truncated_path)
