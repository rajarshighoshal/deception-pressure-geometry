"""Build the compact public receipt for the prospective natural-prose control test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.report_wheel_test import cluster_ci, machine_audit_rows  # noqa: E402
from geoprobe.data.jsonl import read_jsonl  # noqa: E402


DEFAULT_OUT = Path("paper_artifacts/c5_natural_prose_control_receipt.json")

PRIMARY_ARMS = {
    "native_gated": "sdf_controller_gated",
    "frequent_early_window": "sdf_controller_early_window",
    "family_matched_linear": "bundle_repe_probe_a8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "byte_size": path.stat().st_size}


def _load_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for record in read_jsonl(path):
        if "conversation_id" not in record:
            raise ValueError(f"{path}: missing conversation_id in record")
        conversation_id = str(record["conversation_id"])
        if conversation_id in rows:
            raise ValueError(f"{path}: duplicate conversation_id={conversation_id}")
        rows[conversation_id] = record
    return rows


def _responses_by_id(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    replies = {}
    for conversation_id, row in records.items():
        response = row.get("generated_response")
        if response is None:
            raise ValueError(f"{conversation_id}: missing generated_response")
        replies[conversation_id] = str(response)
    return replies


def _family_to_ids(source_rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for conversation_id, row in source_rows.items():
        by_family[str(row.get("family", "unknown"))].append(conversation_id)
    return dict(by_family)


def _status_delta_block(
    base_audit: dict[str, dict[str, Any]],
    steered_audit: dict[str, dict[str, Any]],
    conversation_ids: list[str],
) -> dict[str, Any]:
    if not conversation_ids:
        return {
            "n": 0,
            "base_parse_success": 0,
            "steered_parse_success": 0,
            "base_status_correct": 0,
            "steered_status_correct": 0,
            "status_fixes": 0,
            "status_harms": 0,
            "base_basis_strict_ok": 0,
            "steered_basis_strict_ok": 0,
            "strict_fixes": 0,
            "strict_harms": 0,
            "delta_status_correct": {"point": None, "ci95": None, "n": 0, "n_clusters": 0},
            "delta_basis_strict_ok": {"point": None, "ci95": None, "n": 0, "n_clusters": 0},
        }

    families = [base_audit[cid]["family"] for cid in conversation_ids]
    status_delta = [
        int(steered_audit[cid]["status_correct"]) - int(base_audit[cid]["status_correct"])
        for cid in conversation_ids
    ]
    strict_delta = [
        int(steered_audit[cid]["basis_strict_ok"]) - int(base_audit[cid]["basis_strict_ok"])
        for cid in conversation_ids
    ]
    return {
        "n": len(conversation_ids),
        "base_parse_success": sum(base_audit[cid]["parse_success"] for cid in conversation_ids),
        "steered_parse_success": sum(
            steered_audit[cid]["parse_success"] for cid in conversation_ids
        ),
        "base_status_correct": sum(base_audit[cid]["status_correct"] for cid in conversation_ids),
        "steered_status_correct": sum(
            steered_audit[cid]["status_correct"] for cid in conversation_ids
        ),
        "status_fixes": sum(
            not base_audit[cid]["status_correct"] and steered_audit[cid]["status_correct"]
            for cid in conversation_ids
        ),
        "status_harms": sum(
            base_audit[cid]["status_correct"] and not steered_audit[cid]["status_correct"]
            for cid in conversation_ids
        ),
        "base_basis_strict_ok": sum(base_audit[cid]["basis_strict_ok"] for cid in conversation_ids),
        "steered_basis_strict_ok": sum(
            steered_audit[cid]["basis_strict_ok"] for cid in conversation_ids
        ),
        "strict_fixes": sum(
            not base_audit[cid]["basis_strict_ok"] and steered_audit[cid]["basis_strict_ok"]
            for cid in conversation_ids
        ),
        "strict_harms": sum(
            base_audit[cid]["basis_strict_ok"] and not steered_audit[cid]["basis_strict_ok"]
            for cid in conversation_ids
        ),
        "delta_status_correct": cluster_ci(status_delta, families),
        "delta_basis_strict_ok": cluster_ci(strict_delta, families),
    }


def _compact_family_machine_status(
    base_audit: dict[str, dict[str, Any]],
    steered_audit: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for family, ids in sorted(_family_to_ids(source_rows).items()):
        family_ids = [
            conversation_id
            for conversation_id in ids
            if conversation_id in base_audit and conversation_id in steered_audit
        ]
        all_block = _status_delta_block(base_audit, steered_audit, family_ids)
        deceptive_ids = [
            conversation_id
            for conversation_id in family_ids
            if str(source_rows[conversation_id].get("label", "")).lower() == "deceptive"
        ]
        honest_ids = [
            conversation_id
            for conversation_id in family_ids
            if str(source_rows[conversation_id].get("label", "")).lower() == "honest"
        ]
        deceptive_block = _status_delta_block(base_audit, steered_audit, deceptive_ids)
        honest_block = _status_delta_block(base_audit, steered_audit, honest_ids)
        compact[family] = {
            "population": {
                "total": all_block["n"],
                "deceptive": deceptive_block["n"],
                "honest": honest_block["n"],
            },
            "deceptive_delta_vs_base": {
                "point": deceptive_block["delta_status_correct"]["point"],
                "n": deceptive_block["n"],
            },
            "deceptive_fixes": deceptive_block["status_fixes"],
            "deceptive_harms": deceptive_block["status_harms"],
            "honest_delta_vs_base": {
                "point": honest_block["delta_status_correct"]["point"],
                "n": honest_block["n"],
            },
            "honest_fixes": honest_block["status_fixes"],
            "honest_harms": honest_block["status_harms"],
        }
    return compact


def _validate_family_reconciliation(arms: dict[str, dict[str, Any]]) -> None:
    """Require the family cells to reproduce every aggregate transition count."""
    for arm_name, arm in arms.items():
        families = arm["machine_status_by_family"].values()
        aggregate = arm["machine_status"]
        observed = {
            "deceptive_fixes": sum(cell["deceptive_fixes"] for cell in families),
            "deceptive_harms": sum(cell["deceptive_harms"] for cell in families),
            "honest_fixes": sum(cell["honest_fixes"] for cell in families),
            "honest_harms": sum(cell["honest_harms"] for cell in families),
        }
        expected = {key: aggregate[key] for key in observed}
        if observed != expected:
            raise ValueError(
                f"C5 family cells do not reconcile for {arm_name}: "
                f"observed={observed}, expected={expected}"
            )


def _interval(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "point": value["point"],
        "ci95": value["ci95"],
        "n": value["n"],
        "scenario_family_clusters": value["n_clusters"],
    }


def _compact_primary_arm(arm: dict[str, Any]) -> dict[str, Any]:
    deceptive = arm["machine_task_audit"]["label_deceptive"]
    honest = arm["machine_task_audit"]["label_honest"]
    judge_deceptive = arm["judge"]["label_deceptive"]
    compact: dict[str, Any] = {
        "population": {"total": arm["n"], "deceptive": deceptive["n"], "honest": honest["n"]},
        "changed_vs_base": arm["changed_vs_base"],
        "machine_status": {
            "deceptive_delta_vs_base": _interval(deceptive["delta_status_correct"]),
            "deceptive_fixes": deceptive["status_fixes"],
            "deceptive_harms": deceptive["status_harms"],
            "honest_delta_vs_base": _interval(honest["delta_status_correct"]),
            "honest_fixes": honest["status_fixes"],
            "honest_harms": honest["status_harms"],
        },
        "secondary_llm_judge": {
            "deceptive_honesty_delta": _interval(judge_deceptive["delta_honesty"]),
            "deceptive_coherence_delta": _interval(judge_deceptive["delta_coherence"]),
            "honest_coherence_harm_rate": arm["judge"]["honest_path_harm_primary"][
                "honest_coherence_harm_rate"
            ],
        },
    }
    telemetry = arm.get("gate_telemetry")
    if telemetry:
        compact["intervention_telemetry"] = {
            label: {
                "n": values["n"],
                "eligible": values["eligible"],
                "fired": values["fired"],
                "median_first_fire": values["median_first_fire"],
            }
            for label, values in telemetry.items()
        }
    return compact


def _compact_attribution(path: Path) -> dict[str, Any]:
    source = _load(path)
    summary = source["summary"]
    arm = summary["arms"]["sdf_controller_early_window"]
    deceptive = arm["machine_task_audit"]["label_deceptive"]
    judge = arm["judge"]["label_deceptive"]
    return {
        "source_artifact": _identity(path),
        "scope": summary["framing"],
        "population": {"total": arm["n"], "deceptive": deceptive["n"]},
        "deceptive_machine_status_delta": _interval(deceptive["delta_status_correct"]),
        "deceptive_judge_honesty_delta": _interval(judge["delta_honesty"]),
        "deceptive_judge_coherence_delta": _interval(judge["delta_coherence"]),
    }


def _ensure_conversation_ids_match(
    *,
    source_rows: dict[str, dict[str, Any]],
    base_rows: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, Any]],
    n_base: int,
) -> None:
    expected = set(source_rows)
    if set(base_rows) != expected:
        raise ValueError(
            "Source rows and base generations must use the same conversation_ids "
            f"(source={len(expected)}, base={len(base_rows)})"
        )
    for arm_name, rows in arm_rows.items():
        if set(rows) != expected:
            raise ValueError(
                f"Source rows and arm generation '{arm_name}' must use the same conversation_ids "
                f"(source={len(expected)}, arm={len(rows)})"
            )
    if n_base != len(source_rows):
        raise ValueError(
            f"C5 primary source n_base={n_base} but source rows contain {len(source_rows)} rows"
        )


def build_receipt(
    primary_path: Path,
    random_path: Path,
    sign_flip_path: Path,
    *,
    holdout_source_rows: Path,
    holdout_base_generations: Path,
    holdout_native_gated_arm: Path,
    holdout_frequent_early_window_arm: Path,
    holdout_family_matched_linear_arm: Path,
) -> dict[str, Any]:
    source = _load(primary_path)
    summary = source.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("C5 primary source is not a summary object")
    if "n_base" not in summary:
        raise ValueError("C5 primary source missing n_base")
    arms = summary.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("C5 primary source has no arms object")
    missing = [source_name for source_name in PRIMARY_ARMS.values() if source_name not in arms]
    if missing:
        raise ValueError(f"C5 primary source is missing arms: {missing}")
    if summary["n_base"] <= 0:
        raise ValueError("C5 primary source has non-positive n_base")

    source_rows = _load_jsonl_records(holdout_source_rows)
    base_rows = _load_jsonl_records(holdout_base_generations)
    arm_rows = {
        "native_gated": _load_jsonl_records(holdout_native_gated_arm),
        "frequent_early_window": _load_jsonl_records(holdout_frequent_early_window_arm),
        "family_matched_linear": _load_jsonl_records(holdout_family_matched_linear_arm),
    }
    _ensure_conversation_ids_match(
        source_rows=source_rows,
        base_rows=base_rows,
        arm_rows=arm_rows,
        n_base=summary["n_base"],
    )

    base_machine = machine_audit_rows(_responses_by_id(base_rows), source_rows)
    arm_machine = {
        arm_name: machine_audit_rows(_responses_by_id(rows), source_rows)
        for arm_name, rows in arm_rows.items()
    }

    compact_arms = {
        public_name: {
            **_compact_primary_arm(arms[source_name]),
            "machine_status_by_family": _compact_family_machine_status(
                base_machine,
                arm_machine[public_name],
                source_rows,
            ),
        }
        for public_name, source_name in PRIMARY_ARMS.items()
    }
    _validate_family_reconciliation(compact_arms)
    native_point = compact_arms["native_gated"]["machine_status"]["deceptive_delta_vs_base"][
        "point"
    ]
    frequent_point = compact_arms["frequent_early_window"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["point"]
    frequent_ci = compact_arms["frequent_early_window"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["ci95"]
    linear_ci = compact_arms["family_matched_linear"]["machine_status"]["deceptive_delta_vs_base"][
        "ci95"
    ]

    return {
        "schema_version": 1,
        "kind": "c5_natural_prose_control_public_receipt",
        "claim_id": "C5",
        "producer": "experiments/report_public_natural_prose_control_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "prospective_heldout_family_test": _identity(primary_path),
            "holdout_source_rows": _identity(holdout_source_rows),
            "holdout_base_generations": _identity(holdout_base_generations),
            "holdout_native_gated_arm": _identity(holdout_native_gated_arm),
            "holdout_frequent_early_window_arm": _identity(holdout_frequent_early_window_arm),
            "holdout_family_matched_linear_arm": _identity(holdout_family_matched_linear_arm),
        },
        "evaluation": {
            "model": "Llama-3.1-8B-Instruct",
            "activation_scope": "layer-16 residual stream",
            "response_protocol": "natural prose",
            "uncertainty_note": (
                "The family cells are the independent descriptive units. Aggregate percentile "
                "intervals resample only four families and are not precise tail-probability "
                "estimates; single-family cells do not report confidence intervals."
            ),
            "heldout_families": [
                "cybersecurity",
                "electrical_grid",
                "structural_welding",
                "water_treatment",
            ],
            "arms": compact_arms,
        },
        "attribution_controls": {
            "matched_random_axis": _compact_attribution(random_path),
            "sign_flipped_axis": _compact_attribution(sign_flip_path),
            "interpretation": (
                "The LLM-judge channels are secondary and cannot rescue the failed machine "
                "endpoint: perturbation, hedging, and coherence changes can move those scores."
            ),
        },
        "checks": {
            "native_primary_delta_is_zero": native_point == 0,
            "frequent_arm_point_is_not_positive": frequent_point <= 0,
            "frequent_arm_ci_includes_zero": frequent_ci[0] <= 0 <= frequent_ci[1],
            "family_matched_linear_ci_excludes_zero": linear_ci[0] > 0,
            "family_effects_reconcile_aggregate": True,
        },
        "verdict": "refuted_under_prospectively_specified_development_instrument",
        "claim_boundary": (
            "The prospectively specified development test of the layer-16 natural-prose "
            "controller failed. The public receipt does not include the timestamped registration "
            "history and the artifact manifest marks the result nonconfirmatory. This experiment "
            "did not "
            "test an online controller that attaches novel live typed token-residual-attention "
            "states at layers 12, 16, 19, and 20 and updates intervention, local direction, and "
            "dose throughout fresh generation. Building and prospectively evaluating that richer "
            "state-dependent controller was outside the completed study's available time and "
            "compute budget and remains concrete future work; its outcome is unknown, and its "
            "absence cannot rescue the failed controller."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="Prospective held-out-family natural-prose evaluation source artifact.",
    )
    parser.add_argument(
        "--random",
        type=Path,
        required=True,
        help="Matched-random-axis attribution source artifact.",
    )
    parser.add_argument(
        "--sign-flip",
        type=Path,
        required=True,
        help="Sign-flipped-axis attribution source artifact.",
    )
    parser.add_argument(
        "--holdout-source-rows",
        type=Path,
        required=True,
        help="Held-out family source rows for machine-status recomputation.",
    )
    parser.add_argument(
        "--holdout-base-generations",
        type=Path,
        required=True,
        help="Held-out family base generations for machine-status recomputation.",
    )
    parser.add_argument(
        "--holdout-native-gated-arm",
        type=Path,
        required=True,
        help="Held-out family native-gated arm generation.",
    )
    parser.add_argument(
        "--holdout-frequent-early-window-arm",
        type=Path,
        required=True,
        help="Held-out family frequent-early-window arm generation.",
    )
    parser.add_argument(
        "--holdout-family-matched-linear-arm",
        type=Path,
        required=True,
        help="Held-out family family-matched linear arm generation.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_json(
        args.out,
        build_receipt(
            args.primary,
            args.random,
            args.sign_flip,
            holdout_source_rows=args.holdout_source_rows,
            holdout_base_generations=args.holdout_base_generations,
            holdout_native_gated_arm=args.holdout_native_gated_arm,
            holdout_frequent_early_window_arm=args.holdout_frequent_early_window_arm,
            holdout_family_matched_linear_arm=args.holdout_family_matched_linear_arm,
        ),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
