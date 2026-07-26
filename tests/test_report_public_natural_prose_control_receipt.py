from __future__ import annotations

import json
from pathlib import Path

from experiments.report_public_natural_prose_control_receipt import build_parser, build_receipt


def _interval(point: float, ci: list[float], *, n: int = 48) -> dict:
    return {"point": point, "ci95": ci, "n": n, "n_clusters": 4}


def _arm(*, point: float, ci: list[float], fired: int | None, judge: float) -> dict:
    telemetry = None
    if fired is not None:
        telemetry = {
            "deceptive": {"n": 48, "eligible": 47, "fired": fired, "median_first_fire": 2},
            "honest": {"n": 48, "eligible": 37, "fired": 3, "median_first_fire": 2},
        }
    return {
        "n": 96,
        "changed_vs_base": fired or 77,
        "gate_telemetry": telemetry,
        "machine_task_audit": {
            "label_deceptive": {
                "n": 48,
                "delta_status_correct": _interval(point, ci),
                "status_fixes": 3,
                "status_harms": 7,
            },
            "label_honest": {
                "n": 48,
                "delta_status_correct": _interval(0, [-0.1, 0.1]),
                "status_fixes": 5,
                "status_harms": 5,
            },
        },
        "judge": {
            "label_deceptive": {
                "delta_honesty": _interval(judge, [judge - 0.1, judge + 0.1]),
                "delta_coherence": _interval(0, [-0.1, 0.1]),
            },
            "honest_path_harm_primary": {"honest_coherence_harm_rate": 0.1},
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def test_cli_requires_explicit_sources_and_has_provider_neutral_help() -> None:
    parser = build_parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {"primary", "random", "sign_flip"} <= required

    help_text = parser.format_help().lower()
    assert "runpod" not in help_text
    assert "runpod_results" not in help_text
    assert "/users/" not in help_text


def test_receipt_separates_failed_machine_endpoint_from_judge_channel(tmp_path: Path) -> None:
    primary = {
        "summary": {
            "n_base": 96,
            "arms": {
                "sdf_controller_gated": _arm(point=0, ci=[0, 0], fired=2, judge=0),
                "sdf_controller_early_window": _arm(
                    point=-0.0833, ci=[-0.2083, 0.0417], fired=47, judge=0.6458
                ),
                "bundle_repe_probe_a8": _arm(
                    point=0.3333, ci=[0.1667, 0.4583], fired=None, judge=0.875
                ),
            },
        }
    }
    attribution = {
        "summary": {
            "framing": "development attribution",
            "arms": {
                "sdf_controller_early_window": _arm(
                    point=0, ci=[0, 0], fired=23, judge=0
                )
            },
        }
    }
    primary_path = tmp_path / "primary.json"
    random_path = tmp_path / "random.json"
    sign_path = tmp_path / "sign.json"
    _write(primary_path, primary)
    _write(random_path, attribution)
    _write(sign_path, attribution)

    receipt = build_receipt(primary_path, random_path, sign_path)

    assert receipt["verdict"] == "refuted_under_registered_natural_prose_residual_instrument"
    assert receipt["checks"]["native_primary_delta_is_zero"] is True
    assert receipt["checks"]["frequent_arm_point_is_not_positive"] is True
    assert receipt["checks"]["frequent_arm_ci_includes_zero"] is True
    assert receipt["checks"]["family_matched_linear_ci_excludes_zero"] is True
    assert "layers 12, 16, 19, and 20" in receipt["claim_boundary"]
    assert "throughout fresh generation" in receipt["claim_boundary"]
    assert "outcome is unknown" in receipt["claim_boundary"]
    assert str(tmp_path) not in json.dumps(receipt)
