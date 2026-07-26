from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.report_public_detection_receipts import dict_require, main


REPO_ROOT = Path(__file__).resolve().parents[1]
C10_RECEIPT = REPO_ROOT / "paper_artifacts/c10_postcommitment_detection_receipt.json"
C11_RECEIPT = REPO_ROOT / "paper_artifacts/c11_precommitment_warning_receipt.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_shipped_c10_receipt_matches_registered_evidence() -> None:
    receipt = _load(C10_RECEIPT)
    assert receipt["claim_id"] == "C10"
    assert receipt["checks"]["primary_event_count_is_1283"] is True
    assert receipt["checks"]["pair_inventory_is_31"] is True
    assert receipt["checks"]["strict_pair_count_is_30"] is True
    assert receipt["checks"]["comparator_favors_linear_probe"] is True

    assert receipt["primary"]["population"]["event_count"] == 1283
    assert receipt["primary"]["population"]["class_counts"]["DECEPTIVE"] == 1036
    assert receipt["primary"]["models"]["local_joint_top8"][
        "family_macro_brier"
    ] == pytest.approx(0.020518345218457357)
    assert receipt["primary"]["models"]["local_joint_top8"][
        "event_pooled_auroc"
    ] == pytest.approx(0.9950526003157583)
    assert receipt["primary"]["models"]["exact_nuisance_family_balanced"][
        "family_macro_brier"
    ] == pytest.approx(0.09488401325865055)
    assert receipt["primary"]["exact_nuisance_gain"][
        "family_macro_brier_gain"
    ] == pytest.approx(0.07436566804019319)

    permutation = receipt["primary"]["exact_nuisance_gain"][
        "nuisance_preserving_permutation"
    ]
    assert permutation["all_event_count"] == 1680
    assert permutation["scored_event_count"] == 1283
    assert permutation["one_sided_randomization_p"] == pytest.approx(
        0.00009999000099990002
    )
    assert permutation["null_summary"]["mean"] == pytest.approx(0.0434722072921527)
    assert permutation["observed_excess_over_null_mean"] == pytest.approx(
        0.03089346074804048
    )

    probe = receipt["linear_probe_comparator"]["family_macro_brier"]
    assert probe["registered_probe"] == pytest.approx(0.0015015367445592714)
    assert probe["local_joint_top8"] == pytest.approx(0.020518345218457357)
    assert receipt["linear_probe_comparator"]["families_favouring_probe"] == 20

    lie_error = receipt["deception_versus_knowledge_error"]
    assert lie_error["event_count"] == 242
    assert lie_error["event_pooled_auroc"]["local_joint_top8"] == pytest.approx(
        0.949047256097561
    )
    assert lie_error["event_pooled_auroc"][
        "exact_nuisance_family_balanced"
    ] == pytest.approx(0.5416158536585366)

    assert receipt["bank_qualification"]["design"] == {
        "rows": 600,
        "scenarios": 60,
        "families": 20,
        "decision_turns": 4,
        "status_records": 2400,
        "unique_status_events": 1680,
    }
    assert receipt["bank_qualification"]["baseline_knowledge"] == {
        "source": "shared_unpressured_NN_sample0_turn0",
        "correct": 56,
        "denominator": 60,
        "required": 57,
        "passed": False,
    }
    assert receipt["checks"]["structured_action_knowledge_gate_failed_56_of_60"] is True


def test_shipped_c11_receipt_matches_registered_evidence() -> None:
    receipt = _load(C11_RECEIPT)
    assert receipt["claim_id"] == "C11"
    assert receipt["checks"]["spectral_h_d_slice_honest_deceptive_counts"] is True
    assert receipt["checks"]["spectral_not_gain_dominant"] is True
    assert receipt["checks"]["risk_gate_conclusion_matches_summary"] is True
    assert receipt["spectral_field"]["equal_view"]["honest_count"] == 32
    assert receipt["spectral_field"]["equal_view"]["deceptive_count"] == 74
    assert receipt["spectral_field"]["equal_view"]["auroc"] == pytest.approx(
        0.41680743243243246
    )
    assert (
        receipt["connection_path_field"]["status"]
        == "not_supported_under_frozen_checkpoint_path_instrument"
    )
    assert (
        receipt["risk_gate_repair"]["conclusion"]
        == "risk_gate_remains_unsolved_geometry_only_loses_to_design_prior"
    )
    assert receipt["risk_gate_repair"]["interpretation"][
        "primary_geometry_only_log_loss_gain_ci_lower_over_nuisance"
    ] == pytest.approx(-0.07332127645393847)
    assert receipt["risk_gate_repair"]["stochastic_floor"][
        "identical_prefix_mixed_outcome_group_count"
    ] == 20
    sealed = receipt["risk_gate_repair"]["secondary_comparisons"][
        "sealed_local_over_nuisance_prior"
    ]
    assert sealed["mean_log_loss_gain"] == pytest.approx(-0.0865453555859171)
    assert sealed["scenario_cluster_ci"]["interval"] == pytest.approx(
        [-0.1400037834468875, -0.03837311292011706]
    )
    assert receipt["checks"]["risk_sealed_local_over_nuisance_is_negative"] is True
    assert receipt["checks"]["risk_sealed_local_over_nuisance_ci_is_negative"] is True


def test_detection_cli_requires_sources_and_help_is_provider_neutral(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--c10-outcome-report",
        "--c10-comparator-report",
        "--structured-action-gate-report",
        "--c11-spectral-score",
        "--c11-connection-path-score",
        "--c11-risk-gate-repair-report",
        "--c11-stochastic-floor-report",
        "--c11-sealed-risk-report",
    ):
        assert option in help_text
    assert "runpod_results" not in help_text
    assert "/Users/" not in help_text
    assert "/workspace/" not in help_text

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


def test_shipped_detection_receipts_do_not_leak_source_paths() -> None:
    serialized = C10_RECEIPT.read_text() + C11_RECEIPT.read_text()
    assert "/Users/" not in serialized
    assert "/workspace/" not in serialized
    assert "runpod_results" not in serialized


def test_dict_require_rejects_expected_mismatch() -> None:
    with pytest.raises(ValueError, match="mapping mismatch"):
        dict_require({"a": 1}, "mismatch", expected={"a": 2})
