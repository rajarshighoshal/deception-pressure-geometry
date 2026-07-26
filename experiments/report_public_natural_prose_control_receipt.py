"""Build the compact public receipt for the prospective natural-prose control test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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


def build_receipt(primary_path: Path, random_path: Path, sign_flip_path: Path) -> dict[str, Any]:
    source = _load(primary_path)
    summary = source.get("summary")
    if not isinstance(summary, dict) or summary.get("n_base") != 96:
        raise ValueError("C5 primary source must contain the 96-row held-out-family evaluation")
    arms = summary.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("C5 primary source has no arms object")
    missing = [source_name for source_name in PRIMARY_ARMS.values() if source_name not in arms]
    if missing:
        raise ValueError(f"C5 primary source is missing arms: {missing}")

    compact_arms = {
        public_name: _compact_primary_arm(arms[source_name])
        for public_name, source_name in PRIMARY_ARMS.items()
    }
    native_point = compact_arms["native_gated"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["point"]
    frequent_point = compact_arms["frequent_early_window"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["point"]
    frequent_ci = compact_arms["frequent_early_window"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["ci95"]
    linear_ci = compact_arms["family_matched_linear"]["machine_status"][
        "deceptive_delta_vs_base"
    ]["ci95"]

    return {
        "schema_version": 1,
        "kind": "c5_natural_prose_control_public_receipt",
        "claim_id": "C5",
        "producer": "experiments/report_public_natural_prose_control_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "prospective_heldout_family_test": _identity(primary_path),
        },
        "evaluation": {
            "model": "Llama-3.1-8B-Instruct",
            "activation_scope": "layer-16 residual stream",
            "response_protocol": "natural prose",
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
        },
        "verdict": "refuted_under_registered_natural_prose_residual_instrument",
        "claim_boundary": (
            "The prospective layer-16 natural-prose controller failed. This experiment did not "
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_json(args.out, build_receipt(args.primary, args.random, args.sign_flip))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
