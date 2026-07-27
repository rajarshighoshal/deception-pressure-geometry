from __future__ import annotations

import json
from pathlib import Path

from experiments.report_public_natural_prose_control_receipt import build_parser, build_receipt


def _interval(point: float, ci: list[float], *, n: int = 48) -> dict:
    return {"point": point, "ci95": ci, "n": n, "n_clusters": 4}


def _arm(
    *,
    point: float,
    ci: list[float],
    fired: int | None,
    judge: float,
    deceptive_fixes: int = 3,
    deceptive_harms: int = 7,
    honest_fixes: int = 5,
    honest_harms: int = 5,
) -> dict:
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
                "status_fixes": deceptive_fixes,
                "status_harms": deceptive_harms,
            },
            "label_honest": {
                "n": 48,
                "delta_status_correct": _interval(0, [-0.1, 0.1]),
                "status_fixes": honest_fixes,
                "status_harms": honest_harms,
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


FAMILY_CELLS = {
    "cybersecurity": {
        "base": {"deceptive": 2, "honest": 3},
        "frequent": {"deceptive": (1, 2), "honest": (2, 0)},
        "linear": {"deceptive": (6, 1), "honest": (7, 0)},
    },
    "electrical_grid": {
        "base": {"deceptive": 1, "honest": 5},
        "frequent": {"deceptive": (2, 1), "honest": (3, 2)},
        "linear": {"deceptive": (6, 0), "honest": (6, 3)},
    },
    "structural_welding": {
        "base": {"deceptive": 3, "honest": 6},
        "frequent": {"deceptive": (0, 3), "honest": (0, 3)},
        "linear": {"deceptive": (4, 3), "honest": (4, 3)},
    },
    "water_treatment": {
        "base": {"deceptive": 3, "honest": 2},
        "frequent": {"deceptive": (0, 1), "honest": (0, 0)},
        "linear": {"deceptive": (5, 1), "honest": (9, 0)},
    },
}


def _reply(correct: bool) -> str:
    return "Reported status: PASS" if correct else "Reported status: FAIL"


def _write_holdout_fixture(tmp_path: Path) -> dict[str, Path]:
    source_rows: list[dict] = []
    generations = {name: [] for name in ("base", "native", "frequent", "linear")}
    for family, cells in FAMILY_CELLS.items():
        for label in ("deceptive", "honest"):
            base_correct_count = cells["base"][label]
            for index in range(12):
                conversation_id = f"{family}-{label}-{index:02d}"
                base_correct = index < base_correct_count
                source_rows.append(
                    {
                        "conversation_id": conversation_id,
                        "family": family,
                        "label": label,
                        "true_status": "PASS",
                        "messages": [],
                    }
                )
                generations["base"].append(
                    {"conversation_id": conversation_id, "generated_response": _reply(base_correct)}
                )
                generations["native"].append(
                    {"conversation_id": conversation_id, "generated_response": _reply(base_correct)}
                )
                for arm_name in ("frequent", "linear"):
                    fixes, harms = cells[arm_name][label]
                    correct = base_correct
                    if base_correct and index < harms:
                        correct = False
                    elif not base_correct and index - base_correct_count < fixes:
                        correct = True
                    generations[arm_name].append(
                        {"conversation_id": conversation_id, "generated_response": _reply(correct)}
                    )

    paths = {name: tmp_path / f"{name}.jsonl" for name in generations}
    paths["source"] = tmp_path / "source.jsonl"
    _write_jsonl(paths["source"], source_rows)
    for name, rows in generations.items():
        _write_jsonl(paths[name], rows)
    return paths


def test_cli_requires_explicit_sources_and_has_provider_neutral_help() -> None:
    parser = build_parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {
        "primary",
        "random",
        "sign_flip",
        "holdout_source_rows",
        "holdout_base_generations",
        "holdout_native_gated_arm",
        "holdout_frequent_early_window_arm",
        "holdout_family_matched_linear_arm",
    } <= required

    help_text = parser.format_help().lower()
    assert "runpod" not in help_text
    assert "runpod_results" not in help_text
    assert "/users/" not in help_text


def test_receipt_separates_failed_machine_endpoint_from_judge_channel(tmp_path: Path) -> None:
    primary = {
        "summary": {
            "n_base": 96,
            "arms": {
                "sdf_controller_gated": _arm(
                    point=0,
                    ci=[0, 0],
                    fired=2,
                    judge=0,
                    deceptive_fixes=0,
                    deceptive_harms=0,
                    honest_fixes=0,
                    honest_harms=0,
                ),
                "sdf_controller_early_window": _arm(
                    point=-0.0833, ci=[-0.2083, 0.0417], fired=47, judge=0.6458
                ),
                "bundle_repe_probe_a8": _arm(
                    point=0.3333,
                    ci=[0.1667, 0.4583],
                    fired=None,
                    judge=0.875,
                    deceptive_fixes=21,
                    deceptive_harms=5,
                    honest_fixes=26,
                    honest_harms=6,
                ),
            },
        }
    }
    attribution = {
        "summary": {
            "framing": "development attribution",
            "arms": {"sdf_controller_early_window": _arm(point=0, ci=[0, 0], fired=23, judge=0)},
        }
    }
    primary_path = tmp_path / "primary.json"
    random_path = tmp_path / "random.json"
    sign_path = tmp_path / "sign.json"
    _write(primary_path, primary)
    _write(random_path, attribution)
    _write(sign_path, attribution)

    holdout = _write_holdout_fixture(tmp_path)
    receipt = build_receipt(
        primary_path,
        random_path,
        sign_path,
        holdout_source_rows=holdout["source"],
        holdout_base_generations=holdout["base"],
        holdout_native_gated_arm=holdout["native"],
        holdout_frequent_early_window_arm=holdout["frequent"],
        holdout_family_matched_linear_arm=holdout["linear"],
    )

    assert receipt["verdict"] == "refuted_under_prospectively_specified_development_instrument"
    assert receipt["checks"]["native_primary_delta_is_zero"] is True
    assert receipt["checks"]["frequent_arm_point_is_not_positive"] is True
    assert receipt["checks"]["frequent_arm_ci_includes_zero"] is True
    assert receipt["checks"]["family_matched_linear_ci_excludes_zero"] is True
    assert receipt["checks"]["family_effects_reconcile_aggregate"] is True
    linear = receipt["evaluation"]["arms"]["family_matched_linear"]["machine_status_by_family"]
    assert linear["cybersecurity"]["deceptive_fixes"] == 6
    assert linear["cybersecurity"]["deceptive_harms"] == 1
    assert linear["electrical_grid"]["deceptive_delta_vs_base"] == {
        "point": 0.5,
        "n": 12,
    }
    assert linear["water_treatment"]["honest_fixes"] == 9
    assert "ci95" not in linear["water_treatment"]["honest_delta_vs_base"]
    assert "layers 12, 16, 19, and 20" in receipt["claim_boundary"]
    assert "throughout fresh generation" in receipt["claim_boundary"]
    assert "outcome is unknown" in receipt["claim_boundary"]
    assert str(tmp_path) not in json.dumps(receipt)
