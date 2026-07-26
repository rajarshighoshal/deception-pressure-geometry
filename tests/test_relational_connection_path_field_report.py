"""Focused tests for complete-path connection-response field markdown and CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoprobe.eval.relational_connection_path_field import canonical_sha256
from geoprobe.eval.relational_connection_path_field_report import (
    RelationalConnectionPathFieldReportError,
    render_connection_path_field_markdown,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES

from experiments import report_relational_connection_path_field as report_cli


def _metric(*, event_count: int, hd_log_loss: float = 0.35, auroc: float = 0.51) -> dict[str, object]:
    return {
        "event_pooled_multiclass_log_loss": 0.85,
        "event_pooled_multiclass_brier": 0.45,
        "family_macro_multiclass_log_loss": 0.73,
        "family_macro_multiclass_brier": 0.41,
        "event_count": event_count,
        "honest_deceptive_slice": {
            "conditional_log_loss": hd_log_loss,
            "conditional_brier": 0.44,
            "auroc": auroc,
            "average_precision": 0.48,
            "event_count": event_count,
            "honest_count": int(event_count / 2),
            "deceptive_count": int(event_count / 2),
            "binary_refit": False,
        },
    }


def _models_with_primary(event_count: int) -> dict[str, object]:
    return {
        model: _metric(event_count=event_count)
        for model in (
            "full_path_design_conditioned",
            "full_path_unrestricted",
            "incoming_design_conditioned",
            "common_outgoing_design_conditioned",
            "asymmetry_design_conditioned",
            "residual_full_path_design_conditioned",
            "attention_full_path_design_conditioned",
            "layer_transport_full_path_design_conditioned",
            "identity_shuffled_path_design_conditioned",
            "design_cell",
            "base_rate",
            "one_state_spectral",
        )
    }


def _score() -> dict[str, object]:
    models = _models_with_primary(event_count=12)
    gain_payload = {
        comparator: {
            "event_pooled_log_loss_gain": 0.04,
            "event_pooled_brier_gain": -0.01,
            "honest_deceptive_conditional_log_loss_gain": 0.02,
        }
        for comparator in (
            "design_cell",
            "base_rate",
            "incoming_design_conditioned",
            "one_state_spectral",
            "identity_shuffled_path_design_conditioned",
        )
    }
    uncertainty = {
        comparator: {
            "event_pooled_log_loss_gain": {
                "point_estimate": 0.01,
                "percentile_95": [0.0, 0.02],
                "fraction_positive": 0.66,
            },
            "honest_deceptive_conditional_log_loss_gain": {
                "point_estimate": 0.02,
                "percentile_95": [0.01, 0.03],
                "fraction_positive": 0.75,
            },
        }
        for comparator in (
            "design_cell",
            "base_rate",
            "incoming_design_conditioned",
            "one_state_spectral",
            "identity_shuffled_path_design_conditioned",
        )
    }
    return {
        "schema_version": 1,
        "kind": "relational_connection_path_field_score",
        "prediction_ledger_sha256": "a" * 64,
        "complete_path_bank_sha256": "b" * 64,
        "score_sha256": "c" * 64,
        "primary_model": "full_path_design_conditioned",
        "aggregate": models,
        "per_fold": {
            "outer_1": models,
            "outer_2": models,
        },
        "per_design_cell": {
            "e" * 64: {
                "design_cell": {
                    "true_status": "PASS",
                    "desired_status": "FAIL",
                    "baseline_knowledge_correct": True,
                },
                "metrics": models,
                "full_path_gain_over_comparators": gain_payload,
            }
        },
        "full_path_gain_over_comparators": gain_payload,
        "post_score_descriptive_uncertainty": {
            "family_count": 2,
            "resamples": 100,
            "seed": 20260716,
            "comparisons": uncertainty,
        },
        "claim_boundary": {
            "exploratory_cross_fitted_only": True,
            "complete_sample0_path_only": True,
            "all_outcomes_retained": True,
            "global_flat_coordinates": "absent",
            "gauge_transport_or_holonomy": "not_available",
            "controller": "not_tested",
            "arbitrary_success_threshold": "absent",
        },
        "adjudication": {
            "status": "not_supported_under_frozen_checkpoint_path_instrument",
            "controller_admitted": False,
        },
    }


def _bank() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "relational_complete_path_bank",
        "policy_contract": "artifact_only_cross_fitted",
        "confirmatory": False,
        "bank_sha256": "b" * 64,
        "coverage": {
            "path_count": 2,
            "scenario_count": 2,
            "relation_side_count": 120,
            "class_counts": {label: 1 for label in OUTCOME_CLASSES},
        },
        "paths": [
            {"family": "F1", "fold": "outer_1", "scenario_id": "S1"},
            {"family": "F2", "fold": "outer_2", "scenario_id": "S2"},
        ],
    }


def _ledger() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "relational_connection_path_prediction_ledger",
        "policy_contract": "artifact_only_cross_fitted",
        "confirmatory": False,
        "complete_path_bank_sha256": "b" * 64,
        "spectral_prediction_ledger_sha256": "a" * 64,
        "prediction_ledger_sha256": "a" * 64,
        "predictions": [],
    }


def test_rendering_includes_primary_metrics_comparators_and_bound_hashes() -> None:
    score = _score()
    bank = _bank()
    ledger = _ledger()
    markdown = render_connection_path_field_markdown(
        score=score,
        ledger=ledger,
        bank=bank,
    )
    assert "# Complete-Path Connection-Response Field" in markdown
    assert "full_path_design_conditioned" in markdown
    assert "| full_path_design_conditioned | 0.85000" in markdown
    assert "design_cell | 0.04000 | -0.01000 | 0.02000" in markdown
    assert "Output bindings" in markdown
    assert f"- Complete-path bank SHA-256: `{bank['bank_sha256']}`" in markdown
    assert f"- Prediction ledger SHA-256: `{ledger['prediction_ledger_sha256']}`" in markdown


def test_rendering_rejects_bound_score_and_ledger_mismatch() -> None:
    score = _score()
    bank = _bank()
    ledger = _ledger()
    score["prediction_ledger_sha256"] = "d" * 64
    with pytest.raises(
        RelationalConnectionPathFieldReportError, match="score is not bound to the supplied prediction ledger"
    ):
        render_connection_path_field_markdown(score=score, ledger=ledger, bank=bank)


def test_report_cli_writes_bound_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _bank()
    spectral = {"schema_version": 1, "kind": "relational_intrinsic_spectral_prediction_ledger"}
    spectral["prediction_ledger_sha256"] = canonical_sha256(spectral)
    outcome_path = tmp_path / "outcome.json"
    evidence_path = tmp_path / "evidence.json"
    calibration_path = tmp_path / "calibration.json"
    spectral_path = tmp_path / "spectral-ledger.json"
    out_bank = tmp_path / "path-bank.json"
    out_ledger = tmp_path / "ledger.json"
    out_json = tmp_path / "score.json"
    out_md = tmp_path / "score.md"
    spectral_path.write_text(json.dumps(spectral, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    called: dict[str, str] = {}

    def fake_build_bank(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        called["bank"] = bank["bank_sha256"]
        return bank, {"fixture": {"file_sha256": "f" * 64}}

    def fake_build(*, complete_path_bank: dict[str, object], spectral_prediction_ledger: dict[str, object]) -> dict[str, object]:
        called["build"] = complete_path_bank["bank_sha256"] + ":" + spectral_prediction_ledger["prediction_ledger_sha256"]
        return {
            "schema_version": 1,
            "kind": "relational_connection_path_prediction_ledger",
            "policy_contract": "artifact_only_cross_fitted",
            "confirmatory": False,
            "complete_path_bank_sha256": complete_path_bank["bank_sha256"],
            "spectral_prediction_ledger_sha256": spectral_prediction_ledger["prediction_ledger_sha256"],
            "predictions": [],
        }

    def fake_score(*, prediction_ledger: dict[str, object], complete_path_bank: dict[str, object]) -> dict[str, object]:
        called["score"] = prediction_ledger["complete_path_bank_sha256"]
        aggregate = _models_with_primary(event_count=4)
        return {
            "schema_version": 1,
            "kind": "relational_connection_path_field_score",
            "prediction_ledger_sha256": prediction_ledger["prediction_ledger_sha256"],
            "complete_path_bank_sha256": complete_path_bank["bank_sha256"],
            "primary_model": "full_path_design_conditioned",
            "aggregate": aggregate,
            "per_fold": {"outer_1": aggregate, "outer_2": aggregate},
            "full_path_gain_over_comparators": {
                comparator: {
                    "event_pooled_log_loss_gain": 0.1,
                    "event_pooled_brier_gain": -0.2,
                    "honest_deceptive_conditional_log_loss_gain": 0.05,
                }
                for comparator in (
                    "design_cell",
                    "base_rate",
                    "incoming_design_conditioned",
                    "one_state_spectral",
                    "identity_shuffled_path_design_conditioned",
                )
            },
            "post_score_descriptive_uncertainty": {
                "family_count": 1,
                "resamples": 10,
                "seed": 20260716,
                "comparisons": {
                    comparator: {
                        "event_pooled_log_loss_gain": {
                            "point_estimate": 0.1,
                            "percentile_95": [0.0, 0.2],
                            "fraction_positive": 0.5,
                        },
                        "honest_deceptive_conditional_log_loss_gain": {
                            "point_estimate": 0.05,
                            "percentile_95": [0.01, 0.07],
                            "fraction_positive": 0.6,
                        },
                    }
                    for comparator in (
                        "design_cell",
                        "base_rate",
                        "incoming_design_conditioned",
                        "one_state_spectral",
                        "identity_shuffled_path_design_conditioned",
                    )
                },
            },
            "claim_boundary": {
                "exploratory_cross_fitted_only": True,
                "complete_sample0_path_only": True,
                "all_outcomes_retained": True,
                "global_flat_coordinates": "absent",
                "gauge_transport_or_holonomy": "not_available",
                "controller": "not_tested",
                "arbitrary_success_threshold": "absent",
            },
        }

    def fake_render(*, score: dict[str, object], ledger: dict[str, object], bank: dict[str, object]) -> str:
        return "# mocked report\n"

    monkeypatch.setattr(report_cli, "build_connection_path_prediction_ledger", fake_build)
    monkeypatch.setattr(report_cli, "score_connection_path_prediction_ledger", fake_score)
    monkeypatch.setattr(report_cli, "render_connection_path_field_markdown", fake_render)
    monkeypatch.setattr(report_cli, "_build_bank", fake_build_bank)
    monkeypatch.setattr(report_cli, "git_provenance", lambda *_args, **_kwargs: {"git_hash": "fixture"})

    assert (
        report_cli.main(
            [
                "--outcome-join",
                str(outcome_path),
                "--connection-evidence-report",
                str(evidence_path),
                "--calibration",
                str(calibration_path),
                "--spectral-prediction-ledger",
                str(spectral_path),
                "--out-bank",
                str(out_bank),
                "--out-ledger",
                str(out_ledger),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        == 0
    )
    assert called == {
        "bank": bank["bank_sha256"],
        "build": f"{bank['bank_sha256']}:{spectral['prediction_ledger_sha256']}",
        "score": bank["bank_sha256"],
    }
    assert out_bank.exists() and out_ledger.exists() and out_json.exists() and out_md.exists()
    score_payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert score_payload["artifact_identity"]["complete_path_bank_sha256"] == bank["bank_sha256"]
    assert out_md.read_text(encoding="utf-8") == "# mocked report\n"
