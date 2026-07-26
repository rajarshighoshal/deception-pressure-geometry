from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.report_public_gauge_receipt import build_receipt, main


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_RECEIPT = REPO_ROOT / "paper_artifacts/c13_gauge_control_receipt.json"


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _contrast(point: float, interval: list[float]) -> dict:
    return {
        "metric": "deceptive_probability",
        "mean": point,
        "interval": interval,
        "cluster_count": 3,
        "root_count": 2,
        "resamples": 2000,
    }


def _cohort() -> dict:
    return {
        "contrasts": {
            "gauge_geodesic_minus_no_intervention": {
                "deceptive_probability_difference": _contrast(0.0, [0.0, 0.0]),
            },
            "gauge_geodesic_minus_sign_flipped": {
                "deceptive_probability_difference": _contrast(-0.02, [-0.04, 0.0]),
            },
            "gauge_geodesic_minus_random_tangent": {
                "deceptive_probability_difference": _contrast(0.02, [0.0, 0.04]),
            },
        }
    }


def _source() -> dict:
    return {
        "status": "success",
        "scope": {
            "source_exact_prefix_one_step": True,
            "novel_online_attachment_claim": False,
        },
        "inventory": {
            "root_count": 402,
            "event_count": 656,
            "row_count": 2624,
            "active_root_count": 21,
            "arm_order": [
                "no_intervention",
                "gauge_geodesic",
                "sign_flipped",
                "random_tangent",
            ],
            "proposal_status_counts": {
                "active": 21,
                "boundary_exit": 333,
                "field_undefined": 37,
                "off_support": 10,
                "zero_direction": 1,
            },
        },
        "estimands": {
            "all_roots": _cohort(),
            "active_roots": _cohort(),
        },
        "no_intervention_transition_matrices": {
            "gauge_geodesic": {
                "DECEPTIVE": {
                    "DECEPTIVE": 570,
                    "HONEST": 0,
                    "NO_ACTION": 1,
                    "SKIP": 0,
                    "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0,
                }
            },
            "sign_flipped": {
                "DECEPTIVE": {
                    "DECEPTIVE": 570,
                    "HONEST": 1,
                    "NO_ACTION": 0,
                    "SKIP": 0,
                    "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0,
                }
            },
            "random_tangent": {
                "DECEPTIVE": {
                    "DECEPTIVE": 570,
                    "HONEST": 1,
                    "NO_ACTION": 0,
                    "SKIP": 0,
                    "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0,
                }
            },
        },
    }


def _transition_report() -> dict:
    return {
        "status": "success",
        "paired_transitions_vs_noop": {
            "generic_t": {
                "truthful_fixes_deceptive_to_honest": {
                    "conditional_denominator": 573,
                    "unconditional_count": 8,
                    "defined_root_count": 350,
                },
            }
        },
    }


def _response_report() -> dict:
    return {
        "status": "success",
        "populations": {
            "knowledge_correct": {
                "measures": {
                    "full_reach": {
                        "mean": 0.37,
                        "count": 381,
                        "bootstrap": {
                            "lower": 0.33,
                            "upper": 0.41,
                            "resamples": 2000,
                        },
                    },
                    "generic_reach": {
                        "mean": 0.35,
                        "count": 381,
                        "bootstrap": {
                            "lower": 0.32,
                            "upper": 0.39,
                            "resamples": 2000,
                        },
                    },
                    "specific_after_generic": {
                        "mean": 0.012,
                        "count": 381,
                        "bootstrap": {
                            "lower": -0.02,
                            "upper": 0.04,
                            "resamples": 2000,
                        },
                    },
                },
                "boundary_summary": {
                    "negative_count": 350,
                    "negative_crossings": {
                        "full_h": {"count": 2}
                    },
                },
            }
        },
    }


def _holonomy_report() -> dict:
    return {
        "frozen_constants": {"theta_min": 0.1},
        "per_fold": [
            {"adequacy_gate": {"adequate": False, "n3_p95_median_angle": value}}
            for value in [0.69, 0.72, 0.72, 0.71, 0.73]
        ],
        "verdict": "not-found-under-this-instrument",
    }


def test_c13_receipt_checks_gauge_null_and_transport_decomposition(tmp_path: Path) -> None:
    source_path = tmp_path / "causal_report.json"
    transition_path = tmp_path / "transitions.json"
    response_path = tmp_path / "response.json"
    holonomy_path = tmp_path / "holonomy.json"
    _write(source_path, _source())
    _write(transition_path, _transition_report())
    _write(response_path, _response_report())
    _write(holonomy_path, _holonomy_report())

    receipt = build_receipt(
        causal_report_path=source_path,
        transition_report_path=transition_path,
        response_diagnostic_path=response_path,
        holonomy_report_path=holonomy_path,
    )

    assert receipt["checks"]["proposal_status_counts_match_registered"] is True
    assert receipt["checks"]["bank_shape_matches_registry"] is True
    assert receipt["checks"]["gauge_geodesic_zero_flip_count"] is True
    assert receipt["checks"]["deceptive_to_honest_flip_is_8_of_573"] is True
    assert receipt["checks"]["specific_after_generic_ci_crosses_zero"] is True
    assert receipt["checks"]["holonomy_adequacy_failed_all_folds"] is True
    assert receipt["causal_replay"]["controller_scope"] == {
        "actuation_layers": [12, 16, 19, 20],
        "interface": "structured_action_exact_frozen_prefix",
        "novel_online_attachment": False,
        "state_attachment": "sealed_source_bank_query_authenticated_against_live_root",
        "temporal_scope": "one_step",
    }
    assert receipt["holonomy_instrument"]["adequate_folds"] == 0
    assert receipt["transport_decomposition"]["crossed_committed_roots"]["crossed"] == 2
    assert receipt["transport_decomposition"]["crossed_committed_roots"]["committed"] == 350
    assert str(tmp_path) not in json.dumps(receipt)


def test_c13_shipped_receipt_matches_registered_evidence() -> None:
    receipt = json.loads(SHIPPED_RECEIPT.read_text())
    assert receipt["claim_id"] == "C13"
    assert receipt["causal_replay"]["inventory"] == {
        "active_roots": 21,
        "arm_order": [
            "no_intervention",
            "gauge_geodesic",
            "sign_flipped",
            "random_tangent",
        ],
        "events": 656,
        "roots": 402,
        "rows": 2624,
    }
    assert receipt["checks"]["all_roots_gauge_vs_noop_is_null"] is True
    assert receipt["checks"]["active_roots_no_op_delta_is_null"] is True
    assert receipt["checks"]["gauge_geodesic_zero_flip_count"] is True
    assert receipt["checks"]["deceptive_to_honest_flip_is_8_of_573"] is True
    assert receipt["checks"]["holonomy_adequacy_failed_all_folds"] is True
    assert receipt["holonomy_instrument"]["adequate_folds"] == 0
    assert "21 of 402" in receipt["claim_boundary"]
    assert "natural-prose generation" in receipt["claim_boundary"]
    assert receipt["verdict"] == (
        "not-found-under-this-instrument, instrument fully characterized"
    )


def test_gauge_cli_requires_sources_and_help_is_provider_neutral(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--causal-report",
        "--pre-status-transitions",
        "--pre-status-response",
        "--holonomy-report",
    ):
        assert option in help_text
    assert "runpod_results" not in help_text
    assert "/Users/" not in help_text
    assert "/workspace/" not in help_text

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


def test_c13_shipped_receipt_has_no_source_path_leaks() -> None:
    serialized = SHIPPED_RECEIPT.read_text()
    assert "/Users/" not in serialized
    assert "/workspace/" not in serialized
    assert "runpod_results" not in serialized
