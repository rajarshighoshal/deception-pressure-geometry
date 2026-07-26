"""Build a compact public receipt for C13 gauge-source causal control.

The receipt combines one source-level causal-replay report with two pre-status
diagnostic artifacts that anchor the transport lane decomposition language used
in the claim registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "paper_artifacts/c13_gauge_control_receipt.json"

EXPECTED_PROPOSAL_STATUS_COUNTS = {
    "active": 21,
    "boundary_exit": 333,
    "field_undefined": 37,
    "off_support": 10,
    "zero_direction": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "byte_size": path.stat().st_size}


def _metric_interval(value: dict[str, Any], metric: str) -> dict[str, Any]:
    return {
        "point": float(value["mean"]),
        "ci95": [float(value["interval"][0]), float(value["interval"][1])],
        "roots": int(value["root_count"]),
        "clusters": int(value["cluster_count"]),
        "resamples": int(value["resamples"]),
        "metric": metric,
    }


def _deceptive_transition_stats(matrix: dict[str, Any]) -> dict[str, Any]:
    deceptive_row = matrix.get("DECEPTIVE")
    if not isinstance(deceptive_row, dict):
        raise ValueError("transition matrix is missing a DECEPTIVE source row")
    totals = {target: int(deceptive_row.get(target, 0)) for target in (
        "DECEPTIVE",
        "HONEST",
        "NO_ACTION",
        "SKIP",
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
    )}
    deceptive_total = sum(totals.values())
    if deceptive_total <= 0:
        raise ValueError("deceptive transition row is empty")
    return {
        "deceptive_total": deceptive_total,
        "to_honest": totals["HONEST"],
        "to_deceptive": totals["DECEPTIVE"],
        "to_honest_rate": float(totals["HONEST"]) / deceptive_total,
    }


def _extract_deceptive_contrast(report: dict[str, Any], cohort: str, contrast: str) -> dict[str, float]:
    cohort_data = report["estimands"].get(cohort)
    if not isinstance(cohort_data, dict):
        raise ValueError(f"causal report has no cohort {cohort!r}")
    contrasts = cohort_data.get("contrasts")
    if not isinstance(contrasts, dict) or contrast not in contrasts:
        raise ValueError(f"causal report is missing contrast {contrast!r} in {cohort}")
    value = contrasts[contrast].get("deceptive_probability_difference")
    if not isinstance(value, dict):
        raise ValueError(f"contrast {contrast!r} in {cohort} has no deceptive metric")
    return _metric_interval(value, metric="deceptive_probability")


def _extract_push_measure(population: dict[str, Any], name: str) -> dict[str, Any]:
    measure = population["measures"].get(name)
    if not isinstance(measure, dict):
        raise ValueError(f"missing population measure {name!r}")
    bootstrap = measure.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError(f"measure {name!r} is missing bootstrap metadata")
    return {
        "point": float(measure["mean"]),
        "ci95": [float(bootstrap["lower"]), float(bootstrap["upper"])],
        "count": int(measure["count"]),
        "resamples": int(bootstrap["resamples"]),
    }


def _ci_opposite(interval: list[float]) -> list[float]:
    return [-float(interval[1]), -float(interval[0])]


def _extract_transport_fixes(effects: dict[str, Any], arm: str) -> dict[str, Any]:
    arm_payload = effects["paired_transitions_vs_noop"].get(arm)
    if not isinstance(arm_payload, dict):
        raise ValueError(f"paired transitions are missing {arm!r}")
    fixes = arm_payload.get("truthful_fixes_deceptive_to_honest")
    if not isinstance(fixes, dict):
        raise ValueError(f"arm {arm!r} is missing truthful_fixes_deceptive_to_honest")
    return {
        "denominator": int(fixes["conditional_denominator"]),
        "fixes": int(fixes["unconditional_count"]),
        "defined_root_count": int(fixes["defined_root_count"]),
    }


def _load_and_validate_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload.get("status"), str) or payload["status"] != "success":
        raise ValueError(f"{path}: report status is not success")
    return payload


def build_receipt(
    *,
    causal_report_path: Path,
    transition_report_path: Path,
    response_diagnostic_path: Path,
    holonomy_report_path: Path,
) -> dict[str, Any]:
    causal = _load_and_validate_report(causal_report_path)
    transition = _load_and_validate_report(transition_report_path)
    response = _load_and_validate_report(response_diagnostic_path)
    holonomy = _load_json(holonomy_report_path)

    inventory = causal.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("causal report has no inventory")
    proposal_status = inventory.get("proposal_status_counts")
    if not isinstance(proposal_status, dict):
        raise ValueError("causal report inventory is missing proposal_status_counts")
    scope = causal.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("causal report has no scope contract")
    if scope.get("source_exact_prefix_one_step") is not True:
        raise ValueError("causal report is not the registered exact-prefix one-step replay")
    if scope.get("novel_online_attachment_claim") is not False:
        raise ValueError("causal report overclaims novel online attachment")

    transitions = causal.get("no_intervention_transition_matrices")
    if not isinstance(transitions, dict):
        raise ValueError("causal report has no no_intervention_transition_matrices")
    transition_stats = {
        arm: _deceptive_transition_stats(matrix)
        for arm, matrix in transitions.items()
        if arm in {"gauge_geodesic", "sign_flipped", "random_tangent"}
    }

    all_no_int = _extract_deceptive_contrast(
        causal, "all_roots", "gauge_geodesic_minus_no_intervention"
    )
    all_sign = _extract_deceptive_contrast(
        causal, "all_roots", "gauge_geodesic_minus_sign_flipped"
    )
    all_rand = _extract_deceptive_contrast(
        causal, "all_roots", "gauge_geodesic_minus_random_tangent"
    )
    active_no_int = _extract_deceptive_contrast(
        causal, "active_roots", "gauge_geodesic_minus_no_intervention"
    )
    active_sign = _extract_deceptive_contrast(
        causal, "active_roots", "gauge_geodesic_minus_sign_flipped"
    )
    active_rand = _extract_deceptive_contrast(
        causal, "active_roots", "gauge_geodesic_minus_random_tangent"
    )

    transport_fixes = _extract_transport_fixes(transition, "generic_t")

    population = response["populations"].get("knowledge_correct")
    if not isinstance(population, dict):
        raise ValueError("response diagnostic lacks knowledge_correct population")
    measures = population.get("measures")
    if not isinstance(measures, dict):
        raise ValueError("response diagnostic knowledge_correct population lacks measures")
    full_reach = _extract_push_measure(population, "full_reach")
    generic_reach = _extract_push_measure(population, "generic_reach")
    specific_after_generic = _extract_push_measure(population, "specific_after_generic")

    boundary = population["boundary_summary"]
    if not isinstance(boundary, dict):
        raise ValueError("response diagnostic knowledge_correct population lacks boundary_summary")
    crossings = boundary["negative_crossings"].get("full_h")
    if not isinstance(crossings, dict):
        raise ValueError("response diagnostic is missing full_h negative crossing counts")

    fold_gates = [fold["adequacy_gate"] for fold in holonomy["per_fold"]]
    adequate_folds = sum(bool(gate["adequate"]) for gate in fold_gates)
    checks = {
        "proposal_status_counts_match_registered": proposal_status == EXPECTED_PROPOSAL_STATUS_COUNTS,
        "inventory_matches_four_arm_schema": inventory["arm_order"] == [
            "no_intervention",
            "gauge_geodesic",
            "sign_flipped",
            "random_tangent",
        ],
        "bank_shape_matches_registry": (
            inventory["root_count"] == 402
            and inventory["event_count"] == 656
            and inventory["row_count"] == 2624
        ),
        "all_roots_gauge_vs_noop_is_null": all_no_int["point"] == 0.0,
        "active_roots_no_op_delta_is_null": active_no_int["point"] == 0.0,
        "active_roots_flips_are_opposite": (
            active_sign["point"] == -active_rand["point"]
            and active_sign["ci95"] == _ci_opposite(active_rand["ci95"])
        ),
        "gauge_geodesic_zero_flip_count": transition_stats["gauge_geodesic"]["to_honest"] == 0,
        "paired_transitions_have_single_truthful_fix_denom": (
            transport_fixes["fixes"] <= transport_fixes["denominator"]
        ),
        "deceptive_to_honest_flip_is_8_of_573": (
            transport_fixes["fixes"] == 8 and transport_fixes["denominator"] == 573
        ),
        "generic_reach_not_greater_than_truthful_push": generic_reach["point"] <= full_reach["point"],
        "specific_after_generic_ci_crosses_zero": (
            specific_after_generic["ci95"][0] <= 0.0 <= specific_after_generic["ci95"][1]
        ),
        "specific_after_generic_ci_crosses_zero_not_zero": (  # guard against exact zero width
            specific_after_generic["ci95"][0] < specific_after_generic["ci95"][1]
        ),
        "deceptive_truthful_crossings_within_committed": int(crossings["count"]) <= int(boundary["negative_count"]),
        "holonomy_adequacy_failed_all_folds": adequate_folds == 0 and len(fold_gates) == 5,
    }

    return {
        "schema_version": 1,
        "kind": "c13_gauge_control_public_receipt",
        "claim_id": "C13",
        "producer": "experiments/report_public_gauge_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "causal_replay_report": {
                **_identity(causal_report_path),
                "status": causal["status"],
            },
            "pre_status_transition_replay": {
                **_identity(transition_report_path),
                "status": transition["status"],
            },
            "pre_status_response_diagnostic": {
                **_identity(response_diagnostic_path),
                "status": response["status"],
            },
            "holonomy_report": _identity(holonomy_report_path),
        },
        "causal_replay": {
            "controller_scope": {
                "actuation_layers": [12, 16, 19, 20],
                "interface": "structured_action_exact_frozen_prefix",
                "temporal_scope": "one_step",
                "state_attachment": "sealed_source_bank_query_authenticated_against_live_root",
                "novel_online_attachment": False,
            },
            "inventory": {
                "rows": int(inventory["row_count"]),
                "roots": int(inventory["root_count"]),
                "events": int(inventory["event_count"]),
                "active_roots": int(inventory["active_root_count"]),
                "arm_order": list(inventory["arm_order"]),
            },
            "proposal_status_counts": {str(k): int(v) for k, v in proposal_status.items()},
            "contrasts": {
                "all_roots": {
                    "gauge_geodesic_minus_no_intervention": all_no_int,
                    "gauge_geodesic_minus_sign_flipped": all_sign,
                    "gauge_geodesic_minus_random_tangent": all_rand,
                },
                "active_roots": {
                    "gauge_geodesic_minus_no_intervention": active_no_int,
                    "gauge_geodesic_minus_sign_flipped": active_sign,
                    "gauge_geodesic_minus_random_tangent": active_rand,
                },
            },
            "deceptive_outcome_transitions": transition_stats,
        },
        "transport_decomposition": {
            "sample_pool": {
                "flip_denominator": transport_fixes["denominator"],
                "flip_count": transport_fixes["fixes"],
                "defined_root_count": transport_fixes["defined_root_count"],
            },
            "truthful_push": {
                "full_reach": full_reach,
                "generic_reach": generic_reach,
                "specific_after_generic": {
                    **specific_after_generic,
                    "remainder": specific_after_generic["point"],
                    "remainder_ci": specific_after_generic["ci95"],
                },
            },
            "crossed_committed_roots": {
                "crossed": int(crossings["count"]),
                "committed": int(boundary["negative_count"]),
            },
        },
        "holonomy_instrument": {
            "folds": len(fold_gates),
            "adequate_folds": adequate_folds,
            "resolution_threshold_radians": float(holonomy["frozen_constants"]["theta_min"]),
            "residual_matched_null_p95_radians": [
                float(gate["n3_p95_median_angle"]) for gate in fold_gates
            ],
            "verdict": holonomy["verdict"],
            "interpretation": (
                "The residual-matched null exceeded the frozen resolution threshold in every "
                "fold, so curvature was not evaluated. Below-gate angles and correlations are "
                "diagnostics, not curvature evidence."
            ),
        },
        "verdict": "not-found-under-this-instrument, instrument fully characterized",
        "checks": checks,
        "claim_boundary": (
            "The retrospective one-step four-layer structured-action gauge controller is null "
            "on pooled and active samples; only 21 of 402 roots receive an active supported "
            "step. It authenticates live roots against sealed source-bank queries and does not "
            "test novel online attachment or updating throughout natural-prose generation. "
            "Transport effects are non-zero but not specific to deception under the cited "
            "metric support. The holonomy instrument failed its adequacy gate, so no curvature "
            "or flatness verdict is licensed."
        )
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-report", type=Path, required=True)
    parser.add_argument("--pre-status-transitions", type=Path, required=True)
    parser.add_argument("--pre-status-response", type=Path, required=True)
    parser.add_argument("--holonomy-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    write_json(
        args.out,
        build_receipt(
            causal_report_path=args.causal_report,
            transition_report_path=args.pre_status_transitions,
            response_diagnostic_path=args.pre_status_response,
            holonomy_report_path=args.holonomy_report,
        ),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
