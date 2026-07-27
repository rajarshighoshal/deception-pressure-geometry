"""Build compact, path-sanitized public receipts for the C2 and C12 steering results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_C2_OUT = Path("paper_artifacts/c2_dose_control_receipt.json")
DEFAULT_C12_OUT = Path("paper_artifacts/c12_steering_decomposition_receipt.json")

C2_POLICIES = (
    "local_control_flow_fixed_88",
    "local_control_flow_fixed_96",
    "local_control_flow_fixed_128",
    "local_control_flow_dense_alpha",
)
C12_PRIMARY_POLICIES = (
    "baseline",
    "bidir_linear",
    "bidir_tangent",
    "global_mean_gated",
    "global_probe_gated",
    "random_gated",
)
SUMMARY_FIELDS = (
    "n",
    "deceptive_n",
    "honest_n",
    "deceptive_status_fixes",
    "deceptive_strict_fixes",
    "honest_status_harms",
    "honest_strict_harms",
    "status_correct",
    "parse_success",
    "coherence_preserved",
    "mean_status_reward",
    "mean_strict_reward",
    "chosen_methods",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary[key] for key in SUMMARY_FIELDS if key in summary}


def source_identity(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "byte_size": path.stat().st_size}


def build_c2_receipt(source_path: Path) -> dict[str, Any]:
    source = load_json(source_path)
    policies = source.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("C2 source has no policies object")
    missing = [name for name in C2_POLICIES if name not in policies]
    if missing:
        raise ValueError(f"C2 source is missing policies: {missing}")

    compact = {
        name: {
            "summary": compact_summary(policies[name]["summary"]),
            "generation_artifact_sha256": policies[name].get("sha256"),
        }
        for name in C2_POLICIES
    }
    fixed_names = C2_POLICIES[:3]
    fixed_summaries = [compact[name]["summary"] for name in fixed_names]
    fixed_equal = all(summary == fixed_summaries[0] for summary in fixed_summaries[1:])
    dense = compact["local_control_flow_dense_alpha"]["summary"]
    fixed = fixed_summaries[0]

    provenance = source.get("provenance") or {}
    return {
        "schema_version": 1,
        "kind": "c2_dose_control_public_receipt",
        "claim_id": "C2",
        "producer": "experiments/report_public_steering_receipts.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifact": source_identity(source_path),
        "source_provenance": {
            "git_hash_recorded": provenance.get("git_hash"),
            "git_dirty_recorded": provenance.get("git_dirty"),
            "interpretation": "behavior-level targeted generation audit, not a margin-only result",
        },
        "policies": compact,
        "checks": {
            "fixed_88_96_128_summaries_identical": fixed_equal,
            "dense_minus_fixed_deceptive_status_fixes": (
                dense["deceptive_status_fixes"] - fixed["deceptive_status_fixes"]
            ),
            "dense_minus_fixed_honest_status_harms": (
                dense["honest_status_harms"] - fixed["honest_status_harms"]
            ),
            "dense_minus_fixed_mean_strict_reward": (
                dense["mean_strict_reward"] - fixed["mean_strict_reward"]
            ),
        },
        "interpretation": (
            "Learned dense dose ties fixed doses on deceptive status and strict fixes but adds "
            "honest harm. Dose learning is not solved under this instrument."
        ),
    }


def _paired_deceptive_differences(comparison_rows: list[dict[str, Any]]) -> np.ndarray:
    deceptive = [
        row
        for row in comparison_rows
        if str(row.get("status_class_before", "")).startswith("false_")
    ]
    if len({str(row.get("conversation_id")) for row in deceptive}) != len(deceptive):
        raise ValueError("C12 follow-up has duplicate deceptive conversation ids")
    return np.asarray(
        [
            float(bool(row["bidir_tangent_status_correct"]))
            - float(bool(row["bidir_off_tangent_status_correct"]))
            for row in deceptive
        ],
        dtype=np.float64,
    )


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    if differences.ndim != 1 or differences.size < 2:
        raise ValueError("paired bootstrap requires at least two one-dimensional differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    samples = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def build_c12_receipt(
    primary_path: Path,
    followup_path: Path,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    primary = load_json(primary_path)
    followup = load_json(followup_path)
    primary_summary = primary.get("summary")
    followup_summary = followup.get("summary")
    comparison_rows = followup.get("comparison_rows")
    if not isinstance(primary_summary, dict) or not isinstance(followup_summary, dict):
        raise ValueError("C12 source is missing a summary object")
    if not isinstance(comparison_rows, list):
        raise ValueError("C12 follow-up is missing comparison_rows")
    missing = [name for name in C12_PRIMARY_POLICIES if name not in primary_summary]
    if missing:
        raise ValueError(f"C12 primary source is missing policies: {missing}")

    primary_policies = {
        name: compact_summary(primary_summary[name]) for name in C12_PRIMARY_POLICIES
    }
    primary_total = {summary.get("n") for summary in primary_policies.values()}
    if primary_total != {160}:
        raise ValueError(f"C12 primary population changed: {primary_total}")

    differences = _paired_deceptive_differences(comparison_rows)
    deceptive_rows = [
        row
        for row in comparison_rows
        if str(row.get("status_class_before", "")).startswith("false_")
    ]
    tangent_fixes = sum(bool(row["bidir_tangent_status_correct"]) for row in deceptive_rows)
    off_tangent_fixes = sum(
        bool(row["bidir_off_tangent_status_correct"]) for row in deceptive_rows
    )

    return {
        "schema_version": 1,
        "kind": "c12_steering_decomposition_public_receipt",
        "claim_id": "C12",
        "producer": "experiments/report_public_steering_receipts.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "primary_six_arm_audit": {
                **source_identity(primary_path),
                "bound_results_sha256": primary.get("results_sha256"),
            },
            "off_tangent_followup_audit": {
                **source_identity(followup_path),
                "bound_results_sha256": followup.get("results_sha256"),
            },
        },
        "primary_six_arm_evaluation": {
            "population": {
                "total_rows": 160,
                "deceptive_rows": 80,
                "honest_rows": 80,
            },
            "policies": primary_policies,
            "routing_contract": {
                "shared_gate_scope": (
                    "all nonbaseline bidirectional and explicitly gated global arms use the "
                    "same heldout-family pre-action gate-file route"
                ),
                "gate_action": "steer toward the gate-predicted true status, or abstain",
                "bidir_linear_projection": "raw_unprojected",
                "bidir_linear_route_policy": "same gate route and abstention as bidir_tangent",
                "baseline_intervention": "none",
            },
            "interpretation": (
                "Tangent steering has the largest point correction, while matched gate-routed "
                "random and global directions recover much of it. Because direction, gate, dose, "
                "and token scope are not crossed factorially, this pilot does not identify a "
                "separate tangent-geometry contribution."
            ),
        },
        "off_tangent_followup": {
            "provenance_tier": "retrospective_recalculation",
            "historical_limitation": (
                "The historical report did not preserve its bootstrap seed or resample count. "
                "This receipt recomputes and newly labels the interval."
            ),
            "population": {
                "total_comparison_rows": len(comparison_rows),
                "deceptive_paired_rows": int(differences.size),
            },
            "tangent_fixes": int(tangent_fixes),
            "off_tangent_fixes": int(off_tangent_fixes),
            "paired_difference": float(differences.mean()),
            "paired_bootstrap": {
                "seed": bootstrap_seed,
                "resamples": bootstrap_resamples,
                "ci95": paired_bootstrap_interval(
                    differences,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
            },
        },
        "scope": (
            "Pilot 4-bit bank; 160 total rows in the primary comparison and 32 total rows in "
            "the follow-up. Counts such as 48/80 use the deceptive-row denominator."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c2-source",
        type=Path,
        required=True,
        help="Targeted-generation audit source artifact for C2.",
    )
    parser.add_argument(
        "--c12-primary",
        type=Path,
        required=True,
        help="Primary six-arm steering audit source artifact for C12.",
    )
    parser.add_argument(
        "--c12-followup",
        type=Path,
        required=True,
        help="Paired off-tangent follow-up source artifact for C12.",
    )
    parser.add_argument("--c2-out", type=Path, default=DEFAULT_C2_OUT)
    parser.add_argument("--c12-out", type=Path, default=DEFAULT_C12_OUT)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    write_json(args.c2_out, build_c2_receipt(args.c2_source))
    write_json(
        args.c12_out,
        build_c12_receipt(
            args.c12_primary,
            args.c12_followup,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        ),
    )
    print(f"wrote {args.c2_out}")
    print(f"wrote {args.c12_out}")


if __name__ == "__main__":
    main()
