"""Build a compact public evidence receipt for C9 from supplied natpress artifacts.

This is a registry-facing artifact: it summarizes registered outcomes, hazard-law,
and P3 diagnostics without carrying source paths or raw text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "paper_artifacts/c9_pressure_commitment_receipt.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "byte_size": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return payload


def rate(summary: dict[str, Any], key: str) -> dict[str, Any]:
    entry = summary[key]
    raw_n = summary.get("n")
    if raw_n is None:
        if "n" in entry:
            raw_n = entry["n"]
    return {
        "k": int(entry["k"]),
        "n": int(raw_n),
        "ci": [float(entry["lo"]), float(entry["hi"])],
        "point": float(entry["point"]),
    }


def rate_point(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "k": int(summary["k"]),
        "lo": float(summary["lo"]),
        "hi": float(summary["hi"]),
        "point": float(summary["point"]),
    }
    if "n" in summary:
        result["n"] = int(summary["n"])
    return result


def contrast(item: dict[str, Any] | None, *, default: Any = None) -> Any:
    if not item:
        return default
    out = {
        "contrast": item["contrast"],
        "status": item["status"],
        "verdict": item["verdict"],
        "metric": item["metric"],
    }
    if isinstance(out["contrast"], dict):
        out["contrast"] = {
            k: float(v)
            for k, v in out["contrast"].items()
            if k in {"point", "lo", "hi"}
        }
    return out


def parse_outcome_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    results = payload["results"]
    constants = payload["constants"]
    arms: dict[str, Any] = {}
    for arm, arm_data in results["per_arm"].items():
        arms[arm] = {
            "n": int(arm_data["n"]),
            "p1a_commitment": rate(arm_data, "p1a_event"),
            "p1b_deceptive_commitment": rate(arm_data, "p1b_event"),
            "corner_deceptive_committed": rate(arm_data, "corner_deceptive_committed"),
            "corner_deceptive_label": rate(arm_data, "corner_deceptive_label"),
            "corner_label_distribution": arm_data["corner_label_distribution"],
        }

    return {
        "n_conversations": int(payload["n_conversations"]),
        "inference": {
            "analysis_unit": "conversation",
            "rate_interval": str(constants["ci"]),
            "arm_contrast_interval": str(constants["contrast_ci"]),
            "resampling_seed": None,
        },
        "instrument_caveats": [str(item) for item in payload.get("caveats", [])],
        "verdicts": {
            "P1a": {
                "bar": results["P1a"]["bar"],
                "rate": rate_point(results["P1a"]["rate"]),
            },
            "P1b": {
                "bar": results["P1b"]["bar"],
                "rate": rate_point(results["P1b"]["rate"]),
            },
            "P2a": results["P2a"],
            "P2b": results["P2b"],
        },
        "arms": arms,
        "verdicts_clean": {
            "P1a": {
                "bar": float(results["P1a"]["bar"]),
                "rate": rate_point(results["P1a"]["rate"]),
                "verdict": results["P1a"]["verdict"],
            },
            "P1b": {
                "bar": float(results["P1b"]["bar"]),
                "rate": rate_point(results["P1b"]["rate"]),
                "verdict": results["P1b"]["verdict"],
            },
            "P2a": {
                "metric": results["P2a"].get("metric", ""),
                "status": results["P2a"].get("status", ""),
                "verdict": results["P2a"]["verdict"],
                "contrast": {
                    k: float(v)
                    for k, v in results["P2a"]["diff"].items()
                },
            },
            "P2b": {
                "metric": results["P2b"].get("metric", ""),
                "status": results["P2b"].get("status", ""),
                "verdict": results["P2b"]["verdict"],
                "contrast": {
                    k: float(v)
                    for k, v in results["P2b"].get("diff", {}).items()
                },
            },
        },
        "constants": payload.get("constants", {}),
    }


def _ci(obj: dict[str, Any], key: str) -> dict[str, float]:
    raw = obj[key]
    point = float(raw["point"])
    ci = raw.get("ci")
    if ci is not None:
        lo, hi = ci
        return {"point": point, "lo": float(lo), "hi": float(hi)}
    return {k: float(v) for k, v in raw.items() if k in {"point", "lo", "hi"}}


def _pooled_coeffs(pooled: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_conversations": int(pooled["n_conversations"]),
        "n_events": int(pooled["n_events"]),
        "delta": _ci(pooled, "delta"),
        "alpha": _ci(pooled, "alpha"),
        "gamma": _ci(pooled, "gamma"),
    }


def parse_hazard_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "scope": payload["scope"],
        "inference_status": payload["inference_status"],
        "constants": payload["constants"],
        "adaptive_coefficients": _pooled_coeffs(payload["pooled_adaptive"]),
        "adaptive_ll": {
            "mean_delta_ll_per_event": float(payload["H_gamma_primary_lofo"]["mean_delta_ll_per_event"]),
            "ci": {
                "lo": float(payload["H_gamma_primary_lofo"]["ci"][0]),
                "hi": float(payload["H_gamma_primary_lofo"]["ci"][1]),
            },
            "n_families_evaluated": int(payload["H_gamma_primary_lofo"]["n_families_evaluated"]),
            "zero_event_families_skipped": payload["H_gamma_primary_lofo"]["zero_event_families_skipped"],
            "verdict": payload["H_gamma_primary_lofo"]["verdict"],
            "heldout_commitment_events": int(sum(
                fam["n_events"] for fam in payload["H_gamma_primary_lofo"]["per_family"].values()
            )),
        },
        "mediation": payload.get("H_M_mediation"),
    }


def _jsonl_count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def _jsonl_families(path: Path) -> set[str]:
    families: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            families.add(str(row["family"]))
    return families


def parse_dissociation_report(path: Path, bank_rows: Path) -> dict[str, Any]:
    payload = load_json(path)
    source_families = _jsonl_families(bank_rows)
    return {
        "scope": payload["scope"],
        "constants": payload["constants"],
        "design_realization_gate": payload["design_realization_gate"],
        "frontload_vs_benign_t5t6": payload["secondary_frontload_vs_benign_t5t6"],
        "source_bank": {
            "source_n_conversations": _jsonl_count_lines(bank_rows),
            "source_n_families": len(source_families),
            "source_families": sorted(source_families),
            "source_artifact": source_identity(bank_rows),
        },
        "analyzed": {
            "n_conversations": int(payload["pooled_adaptive"]["n_conversations"]),
            "n_families": int(payload["H_gamma_primary_lofo"]["n_families_evaluated"]),
            "n_commitment_events": int(payload["pooled_adaptive"]["n_events"]),
            "excluded_families": payload["constants"].get("excluded_families", []),
            "verdict": payload["H_gamma_primary_lofo"]["verdict"],
        },
        "coefficients": _pooled_coeffs(payload["pooled_adaptive"]),
        "lofo_delta_ll_per_event": {
            "mean": float(payload["H_gamma_primary_lofo"]["mean_delta_ll_per_event"]),
            "ci": {
                "lo": float(payload["H_gamma_primary_lofo"]["ci"][0]),
                "hi": float(payload["H_gamma_primary_lofo"]["ci"][1]),
            },
        },
        "per_family_n_events": {
            fam: {
                "delta_ll_per_event": float(v["delta_ll_per_event"]),
                "n_events": int(v["n_events"]),
            }
            for fam, v in payload["H_gamma_primary_lofo"]["per_family"].items()
        },
        "zero_event_families_skipped": payload["H_gamma_primary_lofo"]["zero_event_families_skipped"],
    }


def parse_p3_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    primary_layer = payload["primary_layer"]
    primary = payload["results_by_layer"][primary_layer]
    return {
        "primary_layer": primary_layer,
        "verdict": payload["P3_primary_verdict"],
        "scope": payload["scope"],
        "scripted_corner_rows_used": int(payload["scripted_corner_rows_used"]),
        "fit": {
            "n_fit_deceptive": int(primary["n_fit_deceptive"]),
            "n_fit_honest": int(primary["n_fit_honest"]),
            "mean_auc": float(primary["mean_auc"]),
            "fold_aucs": [float(x) for x in primary["fold_aucs"]],
            "adequate": bool(primary["adequate"]),
        },
        "edf": {
            "point": float(primary["edf"]["contrast"]["point"]),
            "ci": {
                "lo": float(primary["edf"]["contrast"]["lo"]),
                "hi": float(primary["edf"]["contrast"]["hi"]),
            },
            "smooth_n": int(primary["edf"]["smooth_n"]),
            "step_n": int(primary["edf"]["step_n"]),
            "smooth_mean": float(primary["edf"]["smooth_mean"]),
            "step_mean": float(primary["edf"]["step_mean"]),
        },
        "spearman": {
            "point": float(primary["spearman_precorner"]["contrast"]["point"]),
            "ci": {
                "lo": float(primary["spearman_precorner"]["contrast"]["lo"]),
                "hi": float(primary["spearman_precorner"]["contrast"]["hi"]),
            },
            "smooth_mean": float(primary["spearman_precorner"]["smooth_mean"]),
            "step_mean": float(primary["spearman_precorner"]["step_mean"]),
        },
    }


def build_c9_receipt(
    *,
    scripted_outcomes: Path,
    adaptive_outcomes: Path,
    hazard_report: Path,
    dissociation_report: Path,
    dissociation_bank_rows: Path,
    p3_report: Path,
) -> dict[str, Any]:
    scripted = parse_outcome_report(scripted_outcomes)
    adaptive = parse_outcome_report(adaptive_outcomes)
    hazard = parse_hazard_report(hazard_report)
    dissociation = parse_dissociation_report(dissociation_report, dissociation_bank_rows)
    p3 = parse_p3_report(p3_report)

    return {
        "schema_version": 1,
        "kind": "c9_pressure_commitment_public_receipt",
        "claim_id": "C9",
        "scope": "development_bank_no_ood_claims",
        "producer": "experiments/report_public_natpress_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "outcomes": {
            "scripted": {
                "n_conversations": scripted["n_conversations"],
                "inference": scripted["inference"],
                "instrument_caveats": scripted["instrument_caveats"],
                "smooth_commitment_deceptive": scripted["arms"]["smooth"][
                    "p1b_deceptive_commitment"
                ],
                "arm_summary": {
                    arm: {
                        "n": data["n"],
                        "p1a_commitment": data["p1a_commitment"],
                        "p1b_deceptive_commitment": data["p1b_deceptive_commitment"],
                    }
                    for arm, data in scripted["arms"].items()
                },
                "contrasts": {
                    "P1a": scripted["verdicts_clean"]["P1a"],
                    "P1b": scripted["verdicts_clean"]["P1b"],
                    "P2a": scripted["verdicts_clean"]["P2a"],
                    "P2b": scripted["verdicts_clean"]["P2b"],
                },
            },
            "adaptive": {
                "n_conversations": adaptive["n_conversations"],
                "inference": adaptive["inference"],
                "instrument_caveats": adaptive["instrument_caveats"],
                "smooth_commitment_deceptive": adaptive["arms"]["smooth"][
                    "p1b_deceptive_commitment"
                ],
                "arm_summary": {
                    arm: {
                        "n": data["n"],
                        "p1a_commitment": data["p1a_commitment"],
                        "p1b_deceptive_commitment": data["p1b_deceptive_commitment"],
                    }
                    for arm, data in adaptive["arms"].items()
                },
                "contrasts": {
                    "P2a": adaptive["verdicts_clean"]["P2a"],
                    "P2b": adaptive["verdicts_clean"]["P2b"],
                },
            },
        },
        "hazard": {
            "adaptive_bank": {
                "scope": hazard["scope"],
                "inference_status": hazard["inference_status"],
                "adaptive_coefficients": hazard["adaptive_coefficients"],
                "ll_regression": hazard["adaptive_ll"],
                "mediation": hazard["mediation"],
            },
            "dissociation_bank": {
                "source_to_analyzed": {
                    "source_n_conversations": dissociation["source_bank"]["source_n_conversations"],
                    "analyzed_n_conversations": dissociation["analyzed"]["n_conversations"],
                    "source_n_families": dissociation["source_bank"]["source_n_families"],
                    "analyzed_n_families": dissociation["analyzed"]["n_families"],
                },
                "coefficients": dissociation["coefficients"],
                "heldout_commitment_events": {
                    "n": dissociation["analyzed"]["n_commitment_events"],
                    "n_heldout_families": dissociation["analyzed"]["n_families"],
                },
                "ll_regression": dissociation["lofo_delta_ll_per_event"],
                "design_realization_gate": dissociation["design_realization_gate"],
                "secondary_frontload_vs_benign_t5t6": dissociation[
                    "frontload_vs_benign_t5t6"
                ],
            },
        },
        "p3": {
            "primary": p3,
        },
        "source_artifacts": {
            "scripted_outcomes_report": source_identity(scripted_outcomes),
            "adaptive_outcomes_report": source_identity(adaptive_outcomes),
            "hazard_law_report": source_identity(hazard_report),
            "dissociation_hazard_report": source_identity(dissociation_report),
            "dissociation_source_bank_rows": source_identity(dissociation_bank_rows),
            "p3_report": source_identity(p3_report),
        },
        "sanity": {
            "scripted_population": scripted["n_conversations"],
            "adaptive_population": adaptive["n_conversations"],
            "dissociation_source_population": dissociation["source_bank"]["source_n_conversations"],
            "dissociation_analyzed_population": dissociation["analyzed"]["n_conversations"],
            "registered_exclusions": dissociation["analyzed"]["excluded_families"],
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted-outcomes", type=Path, required=True)
    parser.add_argument("--adaptive-outcomes", type=Path, required=True)
    parser.add_argument("--hazard-report", type=Path, required=True)
    parser.add_argument("--dissociation-report", type=Path, required=True)
    parser.add_argument("--dissociation-bank-rows", type=Path, required=True)
    parser.add_argument("--p3-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    receipt = build_c9_receipt(
        scripted_outcomes=args.scripted_outcomes,
        adaptive_outcomes=args.adaptive_outcomes,
        hazard_report=args.hazard_report,
        dissociation_report=args.dissociation_report,
        dissociation_bank_rows=args.dissociation_bank_rows,
        p3_report=args.p3_report,
    )
    write_json(args.out, receipt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
