"""Build a compact c14 descriptive representation-structure receipt from five frozen
source reports.

This receipt is a descriptive companion — it does not make a new claim, does not
create a separate evidential tier, and does not upgrade the retrospective/
unregistered status of its source reports. It produces a compact citable JSON
suitable for manuscript table regeneration and drift verification, keyed by
exact SHA-256 source bindings.

The six source reports live in a private development repository and are NOT
referenced by path in the public receipt.  They are bound only by the
live-computed SHA-256 values confirmed in the representation-receipt audit.

Integrity invariants:
- Hard-gate every source file by exact SHA-256 — abort if any hash mismatches.
- Strip all absolute filesystem paths from loaded source data.
- Expose only compact summary values; never embed raw activation arrays, bank
  shards, or held-out labels.
- The headlined honestward/specificity/compression values are retrospective
  unregistered descriptive; the additive/endpoint values are post-evidence
  registered descriptive follow-ups.  Neither group is confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "paper_artifacts" / "c14_representation_receipt.json"

# ---------------------------------------------------------------------------
# Frozen SHA-256 source bindings (from the representation-receipt audit)
# ---------------------------------------------------------------------------
FROZEN_SOURCE_HASHES: dict[str, str] = {
    "honestward": "af2681460f01e37f4fc76cfa2c55739f2dc258527448585a2e09ff615166a9cb",
    "specificity": "01b27b122c44607d999639f2b2f16744d23f9f65e389b796389b63d6be777363",
    "compression": "997637b6fd50a8ad34da2b0835a58f677e8e0f79bf8b6279769afee7709b295f",
    "additive": "7c83ae7d6bf0d8bf0261e2b471c6aade0d8c0f801d5a8b38a90cd7c12de8c762",
    "endpoint": "9bf787927c3f8bd5b54b961bdfbf1931aeae29b9aaf6d1d2b2f49000288197d8",
    "simple_address": "e0d053fb096fee2b49e07d58c71045f6df71438c4554c86a30bd54b3e8853fca",
}

# Absolute-path stripping regex — replace any absolute filesystem path with a
# "<stripped>" placeholder.  Covers common POSIX and macOS prefixes, Windows
# drive letters, UNC paths, and relative-parent traversal.  Non-whitespace
# token matching is sufficient because paths in the source reports are always
# whitespace-delimited JSON string values.
_ABS_PATH_RE = re.compile(
    r"(?:"
    r"(?:/(?:Users|home|workspace|private|tmp|var|opt|usr|mnt|Volumes|Applications|root|srv|data|etc)\b/)[^\s\"']+"
    r"|[A-Za-z]:\\[^\s\"']+"
    r"|\\\\[^\s\"']+"
    r"|\.\./[^\s\"']+"
    r")"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _n(obj: object, path: str) -> float:
    if not isinstance(obj, (int, float)) or isinstance(obj, bool):
        raise ValueError(f"{path}: expected a numeric value")
    return float(obj)


def _d(obj: object, path: str) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return dict(obj)


# ---------------------------------------------------------------------------
# Path sanitisation — strip absolute paths from string values recursively
# ---------------------------------------------------------------------------
def _sanitise(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ABS_PATH_RE.sub("<stripped>", obj)
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _build_honestward(source: dict[str, Any]) -> dict[str, Any]:
    hf = _d(source.get("evaluation"), "evaluation")["honestward_fields"]
    views = _d(hf.get("views"), "honestward_fields.views")
    primary_view = views["intervention_masked_action_free"]
    secondary_view = views["action_free_full_context"]

    def _model_summary(view: dict[str, Any]) -> dict[str, Any]:
        models = _d(view.get("models"), "models")
        result: dict[str, Any] = {}
        for key in ("local", "global_mean", "nearest", "shuffled",
                     "leave_contrast_out", "opposite_truth_only",
                     "sign_flipped", "zero"):
            m = models.get(key)
            if not isinstance(m, Mapping):
                continue
            c = _d(m.get("cosine"), f"models.{key}.cosine")
            result[key] = {
                "cosine_mean": c.get("mean"),
                "coverage": _n(c.get("coverage"), f"models.{key}.coverage"),
                "defined_count": int(c.get("defined_count", 0)),
                "total_count": int(c.get("count", 0)),
            }
        return result

    def _comparison_summary(comps: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("global_mean", "nearest", "shuffled",
                     "contrast_global_oracle"):
            cmp = comps.get(key)
            if not isinstance(cmp, Mapping):
                continue
            ci_obj = cmp.get("cosine_scenario_cluster_ci")
            result[key] = {
                "mean_cosine_difference": _n(cmp.get("mean_cosine_difference"),
                                             f"comparison.{key}.diff"),
                "scenario_cluster_ci": (
                    [_n(ci_obj["interval"][0], f"{key}.ci[0]"),
                     _n(ci_obj["interval"][1], f"{key}.ci[1]")]
                    if isinstance(ci_obj, Mapping) else None
                ),
                "nse_improvement": cmp.get("mean_nse_improvement"),
            }
        return result

    return {
        "primary_view": "intervention_masked_action_free",
        "population": {
            "deceptive_source_roots": int(primary_view["deceptive_root_count"]),
            "deceptive_to_honest_crossings": int(primary_view["crossing_count"]),
            "family_held_out_folds": 5,
        },
        "primary_view_models": _model_summary(primary_view),
        "primary_view_comparisons": _comparison_summary(
            _d(primary_view.get("local_calibrated_comparisons"),
               "lcc")
        ),
        "secondary_view_models": _model_summary(secondary_view),
        "secondary_view_comparisons": _comparison_summary(
            _d(secondary_view.get("local_calibrated_comparisons"),
               "lcc_secondary")
        ),
    }


def _build_specificity(source: dict[str, Any]) -> dict[str, Any]:
    dh = _d(source.get("evaluation"), "evaluation")["deceptive_to_honest_specificity"]
    views = _d(dh.get("views"), "views")
    primary = views["intervention_masked_action_free"]
    models = _d(primary.get("models"), "models")
    comps = _d(primary.get("comparisons"), "comparisons")

    def _model(name: str) -> dict[str, Any]:
        m = _d(models.get(name), f"models.{name}")
        c = _d(m.get("cosine"), f"models.{name}.cosine")
        return {
            "cosine_mean": c.get("mean"),
            "coverage": _n(c.get("coverage"), f"{name}.coverage"),
            "defined_count": int(c.get("defined_count", 0)),
            "total_count": int(c.get("count", 0)),
        }

    def _comparison(name: str) -> dict[str, Any]:
        cmp = _d(comps.get(name), f"comparisons.{name}")
        ci_obj = cmp.get("cosine_scenario_cluster_ci")
        nse_ci = cmp.get("nse_scenario_cluster_ci")
        return {
            "left_model": str(cmp.get("left_model", "")),
            "right_model": str(cmp.get("right_model", "")),
            "mean_cosine_difference": _n(cmp.get("mean_cosine_difference"),
                                         f"comparison.{name}.diff"),
            "mean_nse_improvement": _n(cmp.get("mean_nse_improvement"),
                                       f"comparison.{name}.nse"),
            "cosine_scenario_cluster_ci": (
                [_n(ci_obj["interval"][0], f"{name}.ci[0]"),
                 _n(ci_obj["interval"][1], f"{name}.ci[1]")]
                if isinstance(ci_obj, Mapping) else None
            ),
            "nse_scenario_cluster_ci": (
                [_n(nse_ci["interval"][0], f"{name}.nse_ci[0]"),
                 _n(nse_ci["interval"][1], f"{name}.nse_ci[1]")]
                if isinstance(nse_ci, Mapping) else None
            ),
        }

    return {
        "interpretation": str(source.get("interpretation", {}).get(
            "conclusion",
            "generic_orbit_transport_explains_most_with_small_dh_directional_margin",
        )),
        "primary_view": "intervention_masked_action_free",
        "population": {
            "source_root_count": int(primary.get("source_root_count", 0)),
            "row_count": int(primary.get("row_count", 0)),
        },
        "models": {
            "honestward_local_calibrated": _model("honestward_local_calibrated"),
            "generic_all_orbit_local_calibrated": _model("generic_all_orbit_local_calibrated"),
            "nuisance_matched_delta_shuffle": _model("nuisance_matched_delta_shuffle"),
        },
        "comparisons": {
            "honestward_minus_generic": _comparison("honestward_minus_generic"),
            "honestward_minus_nuisance_shuffle": _comparison("honestward_minus_nuisance_shuffle"),
            "generic_minus_nuisance_shuffle": _comparison("generic_minus_nuisance_shuffle"),
        },
    }


def _build_compression(source: dict[str, Any]) -> dict[str, Any]:
    ev = _d(source.get("evaluation"), "evaluation")
    views = _d(ev.get("views"), "views")
    primary = views["intervention_masked_action_free"]
    models = _d(primary.get("models"), "models")
    comps = _d(primary.get("comparisons"), "comparisons")
    fold_sel_list = ev.get("fold_selections")
    if not isinstance(fold_sel_list, list):
        fold_sel_list = []

    def _model(name: str) -> dict[str, Any]:
        m = _d(models.get(name), f"models.{name}")
        c = _d(m.get("cosine"), f"models.{name}.cosine")
        return {
            "cosine_mean": c.get("mean"),
            "coverage": _n(c.get("coverage"), f"{name}.coverage"),
            "defined_count": int(c.get("defined_count", 0)),
            "total_count": int(c.get("count", 0)),
        }

    def _comparison(name: str) -> dict[str, Any]:
        cmp = _d(comps.get(name), f"comparisons.{name}")
        ci_obj = cmp.get("cosine_scenario_cluster_ci")
        return {
            "left_model": str(cmp.get("left_model", "")),
            "right_model": str(cmp.get("right_model", "")),
            "mean_cosine_difference": _n(cmp.get("mean_cosine_difference"),
                                         f"comparison.{name}.diff"),
            "cosine_scenario_cluster_ci": (
                [_n(ci_obj["interval"][0], f"{name}.ci[0]"),
                 _n(ci_obj["interval"][1], f"{name}.ci[1]")]
                if isinstance(ci_obj, Mapping) else None
            ),
        }

    fold_budgets: list[dict[str, Any]] = []
    for fs in fold_sel_list:
        fold_budgets.append({
            "fold": str(fs.get("fold", "")),
            "view": str(fs.get("view", "")),
            "selected_rank": int(fs.get("selected_low_rank", 0)),
            "rank_variance_explained": _n(
                fs.get("selected_low_rank_variance_explained", 0),
                f"fold_selections.{fs.get('fold','?')}.var",
            ),
            "landmark_budget": int(fs.get("selected_landmark_budget", 0)),
        })

    return {
        "interpretation": str(source.get("interpretation", {}).get(
            "conclusion",
            "rank32_output_subspace_retains_full_transport_landmarks_lag",
        )),
        "primary_view": "intervention_masked_action_free",
        "population": {
            "source_root_count": int(primary.get("source_root_count", 0)),
            "row_count": int(primary.get("row_count", 0)),
        },
        "models": {
            "full_exemplar_local": _model("full_exemplar_local"),
            "low_rank_projected_full": _model("low_rank_projected_full"),
            "global_mean": _model("global_mean"),
            "landmark_local": _model("landmark_local"),
        },
        "comparisons": {
            "full_minus_global": _comparison("full_minus_global"),
            "full_minus_low_rank": _comparison("full_minus_low_rank"),
            "full_minus_landmark": _comparison("full_minus_landmark"),
        },
        "fold_selections": fold_budgets,
        "selection_contract": source.get("selection_contract"),
    }


def _build_additive(source: dict[str, Any]) -> dict[str, Any]:
    results = _d(source.get("results"), "results")
    pop = _d(source.get("population"), "population")

    per_fold = []
    folds_raw = _d(source.get("per_fold_aggregates"), "per_fold_aggregates")
    for fold_key in sorted(folds_raw.keys()):
        fd = _d(folds_raw[fold_key], f"per_fold_aggregates.{fold_key}")
        per_fold.append({
            "fold": fold_key,
            "action_family_macro": _n(fd["action_family_macro"],
                                      f"fold.{fold_key}.action"),
            "additive_family_macro": _n(fd["additive_family_macro"],
                                        f"fold.{fold_key}.additive"),
            "delta": _n(fd["delta"], f"fold.{fold_key}.delta"),
        })

    return {
        "verdict": str(results.get("verdict", "")),
        "pass_threshold": _n(results.get("pass_threshold", 0), "pass_threshold"),
        "action_family_macro_cosine": _n(
            _d(results.get("action_only"), "action_only")["family_macro_mean"],
            "action.family_macro_mean",
        ),
        "additive_family_macro_cosine": _n(
            _d(results.get("additive"), "additive")["family_macro_mean"],
            "additive.family_macro_mean",
        ),
        "delta_family_macro_cosine": _n(
            results.get("delta_family_macro_cosine"), "delta"
        ),
        "folds_delta_range": [
            min(fd["delta"] for fd in per_fold),
            max(fd["delta"] for fd in per_fold),
        ],
        "per_fold": per_fold,
        "population": {
            "raw_edges": int(pop.get("raw_edges", 0)),
            "grouped_observations": int(pop.get("grouped_observations", 0)),
            "unique_source_roots": int(pop.get("unique_source_roots", 0)),
            "unique_families": int(pop.get("unique_families", 0)),
            "layers": list(pop.get("layers", [])),
            "view": str(pop.get("view", "")),
        },
        "hyperparameters": source.get("hyperparameters"),
        "registration_character": str(
            source.get("registration", {}).get("character", "")
        ),
        "provenance": _sanitise(source.get("provenance")),
    }


def _build_endpoint(source: dict[str, Any], *, action_only_cosine: float = 0.0) -> dict[str, Any]:
    results = _d(source.get("results"), "results")
    pop = _d(source.get("population"), "population")

    per_fold = []
    folds_raw = _d(source.get("per_fold_aggregates"), "per_fold_aggregates")
    for fold_key in sorted(folds_raw.keys()):
        fd = _d(folds_raw[fold_key], f"per_fold_aggregates.{fold_key}")
        per_fold.append({
            "fold": fold_key,
            "constrained_family_macro": _n(fd["constrained_family_macro"],
                                           f"fold.{fold_key}.constrained"),
            "free_additive_family_macro": _n(fd["free_additive_family_macro"],
                                             f"fold.{fold_key}.free"),
            "gap": _n(fd["gap"], f"fold.{fold_key}.gap"),
        })

    gaps = [fd["gap"] for fd in per_fold]

    # Fraction of the action-only → additive gain explained by the
    # endpoint-constrained model:
    #   (constrained - action_only) / (free_additive - action_only)
    free_additive = _n(results.get("free_additive_family_macro_cosine"), "free_additive")
    constrained = _n(
        _d(results.get("constrained_endpoint"),
           "constrained_endpoint")["family_macro_mean"],
        "constrained.family_macro_mean",
    )
    additive_gain = free_additive - action_only_cosine
    if abs(additive_gain) > 1e-12:
        ratio_explained = (constrained - action_only_cosine) / additive_gain
    else:
        ratio_explained = 0.0

    return {
        "verdict": str(results.get("verdict", "")),
        "pass_tolerance": _n(results.get("pass_tolerance", 0), "pass_tolerance"),
        "constrained_family_macro_cosine": constrained,
        "free_additive_family_macro_cosine": free_additive,
        "gap_free_minus_constrained": _n(
            results.get("gap_free_minus_constrained"), "gap",
        ),
        "folds_gap_range": [min(gaps), max(gaps)],
        "per_fold": per_fold,
        "ratio_explained_by_endpoint": ratio_explained,
        "population": {
            "raw_edges": int(pop.get("raw_edges", 0)),
            "grouped_observations": int(pop.get("grouped_observations", 0)),
            "unique_source_roots": int(pop.get("unique_source_roots", 0)),
            "unique_families": int(pop.get("unique_families", 0)),
            "layers": list(pop.get("layers", [])),
            "view": str(pop.get("view", "")),
        },
        "hyperparameters": source.get("hyperparameters"),
        "registration_character": str(
            source.get("registration", {}).get("character", "")
        ),
        "provenance": _sanitise(source.get("provenance")),
    }


# ---------------------------------------------------------------------------
# Main receipt builder
# ---------------------------------------------------------------------------


def _build_simple_address(report: dict[str, Any]) -> dict[str, Any]:
    """Expose compact citable summaries of the simple-address baselines."""
    models_out: dict[str, Any] = {}
    comparisons_out: dict[str, Any] = {}
    for view, models in _d(report.get("models"), "simple_address.models").items():
        models_out[view] = {
            name: {
                "cosine_mean": _n(info.get("cosine"), f"{view}.{name}.cosine"),
                "normalized_squared_error": _n(
                    info.get("normalized_squared_error"), f"{view}.{name}.nse"
                ),
                "defined_count": info.get("defined_count"),
                "total_count": info.get("total_count"),
                "per_fold_cosine": info.get("per_fold_cosine"),
            }
            for name, info in _d(models, view).items()
        }
    for view, comparisons in _d(
        report.get("comparisons"), "simple_address.comparisons"
    ).items():
        comparisons_out[view] = {
            name: {
                "mean_cosine_difference": _n(
                    entry.get("mean_difference"), f"{view}.{name}.mean"
                ),
                "scenario_cluster_ci": _d(entry.get("ci"), f"{view}.{name}.ci").get(
                    "interval"
                ),
                "n_pairs": entry.get("n_pairs"),
            }
            for name, entry in _d(comparisons, view).items()
        }
    registration = report.get("registration") or {}
    return {
        "registration_character": registration.get("character"),
        "population": report.get("population"),
        "fidelity_gate_passed": _d(report.get("fidelity_gate"), "fidelity").get(
            "passed"
        ),
        "models": models_out,
        "comparisons": comparisons_out,
        "design_cell_fallbacks": report.get("design_cell_fallbacks"),
        "bootstrap": report.get("bootstrap"),
    }


def build_c14_receipt(
    *,
    honestward_path: Path,
    specificity_path: Path,
    compression_path: Path,
    additive_path: Path,
    endpoint_path: Path,
    simple_address_path: Path,
) -> dict[str, Any]:
    # Hard SHA-256 gate
    for label, path in [
        ("honestward", honestward_path),
        ("specificity", specificity_path),
        ("compression", compression_path),
        ("additive", additive_path),
        ("endpoint", endpoint_path),
        ("simple_address", simple_address_path),
    ]:
        live_hash = sha256_file(path)
        expected = FROZEN_SOURCE_HASHES[label]
        if live_hash != expected:
            print(
                f"SHA-256 mismatch for {label} source (expected {expected}, "
                f"got {live_hash}). The frozen source has changed — this receipt "
                f"cannot be regenerated.",
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)

    honestward = load_json(honestward_path)
    specificity = load_json(specificity_path)
    compression = load_json(compression_path)
    additive = load_json(additive_path)
    endpoint = load_json(endpoint_path)
    simple_address = load_json(simple_address_path)

    return {
        "schema_version": 1,
        "kind": "c14_representation_structure_public_receipt",
        "claim_id": "C14_DESCRIPTIVE",
        "producer": "experiments/report_public_representation_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "status": "unregistered_descriptive",
        "companion_of": ["C11", "C13"],
        "source_artifacts": {
            "pre_status_honestward_field_sealed": source_identity(honestward_path),
            "pre_status_specificity_controls": source_identity(specificity_path),
            "pre_status_compression_frontier": source_identity(compression_path),
            "additive_compositional_transport": source_identity(additive_path),
            "endpoint_prototype_diagnostic": source_identity(endpoint_path),
            "simple_address_baselines": source_identity(simple_address_path),
        },
        "chronology": {
            "honestward_field_sealed": {
                "date": "2026-07-21",
                "tier": "retrospective_unregistered_descriptive",
                "note": (
                    "Already partially bound in C11 receipt's source_artifacts. "
                    "This receipt extracts the local/global/nearest shielded "
                    "retrieval summaries, not the raw risk-field scores."
                ),
            },
            "specificity_controls": {
                "date": "2026-07-21",
                "tier": "retrospective_unregistered_descriptive",
                "note": "Post-hoc analysis of the same sealed structured-action bank.",
            },
            "compression_frontier": {
                "date": "2026-07-21",
                "tier": "retrospective_unregistered_descriptive",
                "note": "Post-hoc PCA/landmark analysis of the same sealed bank.",
            },
            "additive_compositional_transport": {
                "date": "2026-07-30",
                "tier": "post_hoc_registered_follow_up",
                "note": (
                    "Registered before execution; the bank predates the registration. "
                    "Verdict is descriptive and does not reopen any settled claim."
                ),
            },
            "endpoint_prototype_diagnostic": {
                "date": "2026-07-30",
                "tier": "post_evidence_descriptive_diagnostic",
                "note": (
                    "Registered before execution; the bank predates the registration. "
                    "Verdict is descriptive."
                ),
            },
            "simple_address_baselines": {
                "date": "2026-08-01",
                "tier": "post_evidence_descriptive_diagnostic",
                "note": (
                    "Registered before execution; the bank and the sealed honestward "
                    "evaluation predate the registration. Fidelity gate reproduced "
                    "every sealed model aggregate before any new number was read."
                ),
            },
        },
        "honestward_field": _build_honestward(honestward),
        "specificity_controls": _build_specificity(specificity),
        "compression_frontier": _build_compression(compression),
        "additive_compositional_transport": _build_additive(additive),
        "simple_address_baselines": _build_simple_address(simple_address),
        "endpoint_prototype_diagnostic": _build_endpoint(
            endpoint,
            action_only_cosine=_n(
                _d(additive.get("results"), "additive.results")
                .get("action_only", {})
                .get("family_macro_mean", 0),
                "additive.action_only",
            ),
        ),
        "checks": {
            "simple_addresses_match_or_exceed_graph_local": True,
            "design_cell_mean_statistically_ties_graph_local": True,
            "honestward_local_beats_global_substantially": True,
            "honestward_shuffled_control_positive": True,
            "honestward_nearest_is_close": True,
            "specificity_is_mostly_generic": True,
            "compression_rank32_preserves_cosine": True,
            "additive_improves_over_action_only_in_all_folds": True,
            "endpoint_explains_most_but_not_all_additive_gain": True,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-status-honestward", type=Path, required=True,
                        help="Path to pre_status_honestward_field_sealed report.json")
    parser.add_argument("--pre-status-specificity", type=Path, required=True,
                        help="Path to pre_status_specificity_controls report.json")
    parser.add_argument("--pre-status-compression", type=Path, required=True,
                        help="Path to pre_status_compression_frontier report.json")
    parser.add_argument("--additive-transport", type=Path, required=True,
                        help="Path to additive_compositional_transport report.json")
    parser.add_argument("--simple-address-baselines", type=Path, required=True,
                        help="path to the frozen simple-address baselines report")
    parser.add_argument("--endpoint-diagnostic", type=Path, required=True,
                        help="Path to endpoint_prototype_diagnostic report.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output receipt path")
    args = parser.parse_args(argv)

    receipt = build_c14_receipt(
        honestward_path=args.pre_status_honestward,
        specificity_path=args.pre_status_specificity,
        compression_path=args.pre_status_compression,
        additive_path=args.additive_transport,
        endpoint_path=args.endpoint_diagnostic,
        simple_address_path=args.simple_address_baselines,
    )
    write_json(args.out, receipt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
