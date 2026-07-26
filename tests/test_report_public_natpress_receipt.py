from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.report_public_natpress_receipt import main, rate


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_RECEIPT = REPO_ROOT / "paper_artifacts/c9_pressure_commitment_receipt.json"


def _receipt() -> dict:
    return json.loads(SHIPPED_RECEIPT.read_text())


def test_c9_shipped_receipt_matches_registered_boundaries() -> None:
    receipt = _receipt()
    scripted = receipt["outcomes"]["scripted"]
    adaptive = receipt["outcomes"]["adaptive"]
    hazard = receipt["hazard"]["adaptive_bank"]
    dissociation = receipt["hazard"]["dissociation_bank"]
    p3 = receipt["p3"]["primary"]

    assert receipt["claim_id"] == "C9"
    assert scripted["n_conversations"] == 96
    assert adaptive["n_conversations"] == 128
    assert scripted["smooth_commitment_deceptive"]["k"] == 26
    assert scripted["smooth_commitment_deceptive"]["n"] == 32
    assert adaptive["smooth_commitment_deceptive"]["k"] == 45
    assert adaptive["smooth_commitment_deceptive"]["n"] == 48
    assert adaptive["arm_summary"]["step"]["p1b_deceptive_commitment"]["k"] == 17
    assert adaptive["arm_summary"]["step"]["p1b_deceptive_commitment"]["n"] == 32
    assert scripted["contrasts"]["P2a"] == {
        "contrast": {"hi": 0.5535908086728232, "lo": 0.0371411754965264, "point": 0.3125},
        "metric": "p1b_event_rate_difference",
        "status": "registered",
        "verdict": "found",
    }
    assert scripted["inference"] == {
        "analysis_unit": "conversation",
        "arm_contrast_interval": "newcombe_95",
        "rate_interval": "wilson_95",
        "resampling_seed": None,
    }
    assert "AS-MEASURED" in scripted["instrument_caveats"][0]
    assert adaptive["inference"] == scripted["inference"]

    assert hazard["adaptive_coefficients"]["alpha"]["point"] == 0.5650120322369143
    assert hazard["adaptive_coefficients"]["gamma"]["point"] == 0.0679257753073269
    assert hazard["adaptive_coefficients"]["gamma"]["lo"] >= 0

    assert dissociation["source_to_analyzed"]["source_n_conversations"] == 96
    assert dissociation["source_to_analyzed"]["analyzed_n_conversations"] == 84
    assert dissociation["source_to_analyzed"]["source_n_families"] == 16
    assert dissociation["source_to_analyzed"]["analyzed_n_families"] == 14
    assert dissociation["heldout_commitment_events"]["n"] == 53
    assert dissociation["heldout_commitment_events"]["n_heldout_families"] == 14
    assert math.isclose(
        dissociation["ll_regression"]["mean"],
        -0.12581403219509973,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert (
        dissociation["ll_regression"]["ci"]["lo"]
        < 0
        < dissociation["ll_regression"]["ci"]["hi"]
    )
    assert dissociation["design_realization_gate"]["n_rows"] == 380

    assert p3["primary_layer"] == "residual.L16"
    assert p3["verdict"] == "not-found-under-this-instrument"
    assert math.isclose(p3["fit"]["mean_auc"], 0.9199019607843137, abs_tol=1e-15)
    assert p3["edf"]["point"] == -0.020181837718773937
    assert p3["edf"]["ci"]["lo"] < 0 < p3["edf"]["ci"]["hi"]


def test_rate_normalizes_a_small_summary() -> None:
    summary = {"n": 8, "event": {"k": 3, "point": 0.375, "lo": 0.1, "hi": 0.7}}
    assert rate(summary, "event") == {
        "k": 3,
        "n": 8,
        "point": 0.375,
        "ci": [0.1, 0.7],
    }


def test_c9_cli_requires_all_source_artifacts_and_help_is_provider_neutral(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--scripted-outcomes",
        "--adaptive-outcomes",
        "--hazard-report",
        "--dissociation-report",
        "--dissociation-bank-rows",
        "--p3-report",
    ):
        assert option in help_text
    assert "runpod_results" not in help_text
    assert "/Users/" not in help_text
    assert "/workspace/" not in help_text

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


def test_c9_shipped_receipt_has_no_absolute_path_leaks() -> None:
    serialized = SHIPPED_RECEIPT.read_text()
    assert "/Users/" not in serialized
    assert "/workspace/" not in serialized
    assert "runpod_results" not in serialized
