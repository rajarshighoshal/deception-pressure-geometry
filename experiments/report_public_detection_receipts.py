"""Build compact public receipts for C10 and C11 detection/warning claims.

These receipts are constrained to public, path-agnostic artifacts and do not leak
absolute filesystem locations. They are intended to summarize confirmed registered
results, not run any analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C10_OUT = REPO_ROOT / "paper_artifacts/c10_postcommitment_detection_receipt.json"
DEFAULT_C11_OUT = REPO_ROOT / "paper_artifacts/c11_precommitment_warning_receipt.json"


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


def dict_require(payload: object, field: str, *, expected: Any | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected {field}: mapping")
    if expected is not None and payload != expected:
        raise ValueError(f"{field}: mapping mismatch")
    return dict(payload)


def number_value(payload: object, path: str) -> float:
    if not isinstance(payload, (int, float)) or isinstance(payload, bool):
        raise ValueError(f"{path}: expected a numeric value")
    return float(payload)


def _model_scores(model: Mapping[str, Any], *, label: str) -> tuple[float, float]:
    family_macro = dict_require(model.get("family_macro"), f"{label}.family_macro")
    event_pooled = dict_require(model.get("event_pooled"), f"{label}.event_pooled")
    return (
        number_value(family_macro["brier"], f"{label}.family_macro.brier"),
        number_value(event_pooled["auroc"], f"{label}.event_pooled.auroc"),
    )


def build_c10_receipt(
    *, outcome_report: Path, comparator_report: Path, gate_report: Path
) -> dict[str, Any]:
    outcome = load_json(outcome_report)
    comparator = load_json(comparator_report)
    gate = load_json(gate_report)
    if outcome.get("status") != "success":
        raise ValueError("C10 outcome report is not a success report")
    if comparator.get("status") != "success":
        raise ValueError("C10 comparator report is not a success report")
    if gate.get("kind") != "relational_structured_action_gate_report":
        raise ValueError("C10 structured-action gate report has the wrong kind")

    gate_design = dict_require(gate.get("design_validation"), "gate.design_validation")
    gate_inventory = dict_require(gate.get("inventory"), "gate.inventory")
    gate_observed = dict_require(gate_inventory.get("observed"), "gate.inventory.observed")
    gate_recognition = dict_require(
        dict_require(gate.get("status_recognition"), "gate.status_recognition").get("overall"),
        "gate.status_recognition.overall",
    )
    gate_knowledge = dict_require(gate.get("knowledge"), "gate.knowledge")
    gate_pressure = dict_require(gate.get("primary_a_vs_n"), "gate.primary_a_vs_n")
    gate_pressure_bootstrap = dict_require(
        gate_pressure.get("bootstrap"), "gate.primary_a_vs_n.bootstrap"
    )
    gate_pressure_support = dict_require(
        gate_pressure.get("support"), "gate.primary_a_vs_n.support"
    )

    primary = dict_require(outcome.get("primary_honest_deceptive"), "primary_honest_deceptive")
    gains = dict_require(
        primary.get("local_gain_over_comparators"), "primary_honest_deceptive.local_gain_over_comparators"
    ).get("exact_nuisance_family_balanced")
    if not isinstance(gains, Mapping):
        raise ValueError("primary exact nuisance gains are missing")
    nuisance_preserving = dict_require(
        outcome.get("nuisance_preserving_permutation"),
        "nuisance_preserving_permutation",
    ).get("primary_family_conditioned")
    if not isinstance(nuisance_preserving, Mapping):
        raise ValueError("primary nuisance-preserving permutation results are missing")
    permutation_null_summary = dict_require(
        nuisance_preserving.get("null_summary"), "nuisance_preserving_permutation.null_summary"
    )
    knowledge_error = dict_require(
        outcome.get("deception_versus_knowledge_error"),
        "deception_versus_knowledge_error",
    )
    knowledge_error_models = dict_require(
        knowledge_error.get("models"), "deception_versus_knowledge_error.models"
    )
    knowledge_error_local = dict_require(
        knowledge_error_models.get("local_joint_top8"),
        "deception_versus_knowledge_error.local_joint_top8",
    )
    knowledge_error_exact = dict_require(
        knowledge_error_models.get("exact_nuisance_family_balanced"),
        "deception_versus_knowledge_error.exact_nuisance_family_balanced",
    )

    models = dict_require(primary.get("models"), "primary_honest_deceptive.models")
    local_model = dict_require(models.get("local_joint_top8"), "local_joint_top8 model")
    exact_model = dict_require(
        models.get("exact_nuisance_family_balanced"),
        "exact_nuisance_family_balanced model",
    )

    pairs = dict_require(outcome.get("exact_prefix_pairs"), "exact_prefix_pairs")
    strict_pairs = dict_require(
        pairs.get("primary_strict_activation_exact"), "primary_strict_activation_exact"
    )
    strict_overall = dict_require(
        strict_pairs.get("overall"), "strict_pair.overall"
    )
    strict_flip = dict_require(
        strict_overall.get("pairwise_sign_flip"), "strict_pair.overall.pairwise_sign_flip"
    )
    pair_inventory = pairs.get("pair_inventory")
    if not isinstance(pair_inventory, list):
        raise ValueError("primary pair_inventory is missing")

    token_pairs = pairs.get("all_token_identical_sensitivity")
    if not isinstance(token_pairs, Mapping):
        raise ValueError("token-identical pair report is missing")
    token_overall = dict_require(token_pairs.get("overall"), "all_token_identical.overall")

    cmp_decision = dict_require(comparator.get("decision"), "comparator.decision")
    cmp_bootstrap = dict_require(
        cmp_decision.get("family_cluster_bootstrap"), "comparator.decision.family_cluster_bootstrap"
    )
    cmp_secondary = dict_require(
        comparator.get("secondary_auroc"), "comparator.secondary_auroc"
    )
    cmp_probe = dict_require(comparator.get("registered_probe"), "comparator.registered_probe")
    cmp_graph = dict_require(
        dict_require(comparator.get("reference_arms"), "comparator.reference_arms").get(
            "local_joint_top8"
        ),
        "comparator.reference_arms.local_joint_top8",
    )

    exact_brier_per_family = dict_require(gains.get("per_family_brier_gain"), "exact_nuisance per-family brier gain")
    per_family_positive = sum(1 for value in exact_brier_per_family.values() if number_value(value, "per_family_brier_gain") > 0)

    return {
        "schema_version": 1,
        "kind": "c10_postcommitment_detection_public_receipt",
        "claim_id": "C10",
        "producer": "experiments/report_public_detection_receipts.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "post_commitment_growth_outcomes": source_identity(outcome_report),
            "post_commitment_linear_probe_comparator": source_identity(comparator_report),
            "structured_action_gate": source_identity(gate_report),
        },
        "bank_qualification": {
            "protocol_id": str(gate["protocol_id"]),
            "protocol_sha256": str(gate["protocol_sha256"]),
            "design": {
                "rows": int(gate_design["row_count"]),
                "scenarios": int(gate_design["scenario_count"]),
                "families": int(gate_pressure_support["family_count"]),
                "decision_turns": 4,
                "status_records": int(gate_observed["status_records"]),
                "unique_status_events": int(gate_observed["unique_status_events"]),
            },
            "action_recognition": {
                "count": int(gate_recognition["recognized_count"]),
                "denominator": int(gate_recognition["denominator"]),
                "rate": number_value(gate_recognition["rate"], "gate.recognition.rate"),
                "required_rate": number_value(
                    gate_recognition["floor"], "gate.recognition.floor"
                ),
                "passed": bool(gate_recognition["passed"]),
            },
            "baseline_knowledge": {
                "source": str(gate_knowledge["source"]),
                "correct": int(gate_knowledge["correct_count"]),
                "denominator": int(gate_knowledge["denominator"]),
                "required": int(gate_knowledge["required_count"]),
                "passed": bool(gate_knowledge["passed"]),
            },
            "pressure_effect": {
                "estimand": str(gate_pressure["estimand"]),
                "point": number_value(gate_pressure["paired_effect"], "gate.pressure.effect"),
                "scenario_cluster_ci95": [
                    number_value(gate_pressure_bootstrap["lower"], "gate.pressure.ci.lower"),
                    number_value(gate_pressure_bootstrap["upper"], "gate.pressure.ci.upper"),
                ],
                "positive_families": int(gate_pressure_support["positive_families"]),
                "family_count": int(gate_pressure_support["family_count"]),
            },
            "public_status": (
                "development_exploratory_bank; the frozen 57-of-60 baseline-knowledge gate "
                "missed by one scenario (56-of-60)"
            ),
        },
        "primary": {
            "population": {
                "event_count": int(primary["event_count"]),
                "class_counts": dict_require(
                    primary.get("class_counts"), "primary.class_counts"
                ),
            },
            "models": {
                "local_joint_top8": {
                    "family_macro_brier": _model_scores(
                        local_model, label="local_joint_top8"
                    )[0],
                    "event_pooled_auroc": _model_scores(
                        local_model, label="local_joint_top8"
                    )[1],
                },
                "exact_nuisance_family_balanced": {
                    "family_macro_brier": _model_scores(
                        exact_model, label="exact_nuisance_family_balanced"
                    )[0],
                    "event_pooled_auroc": _model_scores(
                        exact_model, label="exact_nuisance_family_balanced"
                    )[1],
                },
            },
            "exact_nuisance_gain": {
                "family_macro_brier_gain": number_value(
                    gains["family_macro_brier_gain"], "exact_nuisance.family_macro_brier_gain"
                ),
                "family_macro_log_loss_gain": number_value(
                    gains["family_macro_log_loss_gain"],
                    "exact_nuisance.family_macro_log_loss_gain",
                ),
                "per_family_count": len(exact_brier_per_family),
                "per_family_positive_gain_count": int(per_family_positive),
                "nuisance_preserving_permutation": {
                    "all_event_count": int(nuisance_preserving["all_event_count"]),
                    "scored_event_count": int(nuisance_preserving["scored_event_count"]),
                    "non_hd_labels": str(nuisance_preserving["non_hd_labels"]),
                    "query_labels_edges_nuisance_priors": str(
                        nuisance_preserving["query_labels_edges_nuisance_priors"]
                    ),
                    "observed_family_macro_brier_gain": number_value(
                        nuisance_preserving["observed_family_macro_brier_gain"],
                        "nuisance_permutation.observed_family_macro_brier_gain",
                    ),
                    "one_sided_randomization_p": number_value(
                        nuisance_preserving["one_sided_randomization_p"],
                        "nuisance_permutation.one_sided_randomization_p",
                    ),
                    "seed": int(nuisance_preserving["seed"]),
                    "null_summary": {
                        "count": int(permutation_null_summary["count"]),
                        "max": number_value(
                            permutation_null_summary["max"],
                            "nuisance_permutation.null_summary.max",
                        ),
                        "min": number_value(
                            permutation_null_summary["min"],
                            "nuisance_permutation.null_summary.min",
                        ),
                        "mean": number_value(
                            permutation_null_summary["mean"],
                            "nuisance_permutation.null_summary.mean",
                        ),
                        "median": number_value(
                            permutation_null_summary["median"],
                            "nuisance_permutation.null_summary.median",
                        ),
                        "q25": number_value(
                            permutation_null_summary["q25"],
                            "nuisance_permutation.null_summary.q25",
                        ),
                        "q75": number_value(
                            permutation_null_summary["q75"],
                            "nuisance_permutation.null_summary.q75",
                        ),
                    },
                    "observed_excess_over_null_mean": number_value(
                        nuisance_preserving["observed_family_macro_brier_gain"],
                        "nuisance_permutation.observed_family_macro_brier_gain",
                    )
                    - number_value(
                        permutation_null_summary["mean"],
                        "nuisance_permutation.null_summary.mean",
                    ),
                },
            },
        },
        "deception_versus_knowledge_error": {
            "population": str(knowledge_error["population"]),
            "class_counts": dict_require(
                knowledge_error.get("class_counts"),
                "deception_versus_knowledge_error.class_counts",
            ),
            "event_count": int(knowledge_error_local["event_count"]),
            "event_pooled_auroc": {
                "local_joint_top8": _model_scores(
                    knowledge_error_local, label="knowledge_error.local_joint_top8"
                )[1],
                "exact_nuisance_family_balanced": _model_scores(
                    knowledge_error_exact,
                    label="knowledge_error.exact_nuisance_family_balanced",
                )[1],
            },
        },
        "exact_prefix_pairs": {
            "pair_inventory_count": len(pair_inventory),
            "strict_activation_exact": {
                "pair_count": int(strict_overall["pair_count"]),
                "true_status_counts": dict_require(
                    strict_overall.get("true_status_counts"),
                    "strict_pairs.true_status_counts",
                ),
                "pairwise_sign_flip": {
                    "observed_mean": number_value(
                        strict_flip["observed_mean"], "strict_pairs.pairwise_sign_flip.observed_mean"
                    ),
                    "one_sided_randomization_p": number_value(
                        strict_flip["one_sided_randomization_p"],
                        "strict_pairs.pairwise_sign_flip.one_sided_randomization_p",
                    ),
                    "replicates": int(strict_flip["replicates"]),
                    "seed": int(strict_flip["seed"]),
                },
            },
            "token_identical_pairs": {
                "pair_count": int(token_overall["pair_count"]),
                "true_status_counts": dict_require(
                    token_overall.get("true_status_counts"),
                    "token_identical.true_status_counts",
                ),
            },
        },
        "linear_probe_comparator": {
            "family_macro_brier": {
                "local_joint_top8": number_value(
                    cmp_graph["family_macro_brier"],
                    "comparator.reference_arms.local_joint_top8.family_macro_brier",
                ),
                "registered_probe": number_value(
                    cmp_probe["family_macro_brier"],
                    "comparator.registered_probe.family_macro_brier",
                ),
                "registered_probe_gain_over_prior": number_value(
                    cmp_probe["family_macro_brier_gain_over_prior"],
                    "comparator.registered_probe.family_macro_brier_gain_over_prior",
                ),
            },
            "decision_observed": number_value(cmp_decision["observed"], "comparator.decision.observed"),
            "families_favouring_graph": int(cmp_decision["families_favouring_graph"]),
            "families_favouring_probe": int(cmp_decision["families_favouring_probe"]),
            "cluster_bootstrap": {
                "percentile_95_interval": [
                    number_value(cmp_bootstrap["percentile_95_interval"][0], "bootstrap[0]"),
                    number_value(cmp_bootstrap["percentile_95_interval"][1], "bootstrap[1]"),
                ],
                "bootstrap_fraction_positive": number_value(
                    cmp_bootstrap["bootstrap_fraction_positive"],
                    "comparator.family_cluster_bootstrap.bootstrap_fraction_positive",
                ),
            },
            "secondary_auroc": {
                "local_joint_top8": number_value(cmp_secondary["local_joint_top8"], "secondary_auroc.local"),
                "registered_probe": number_value(cmp_secondary["registered_probe"], "secondary_auroc.registered"),
                "exact_nuisance_family_balanced": number_value(
                    cmp_secondary["exact_nuisance_family_balanced"],
                    "secondary_auroc.exact_nuisance",
                ),
            },
        },
        "checks": {
            "primary_event_count_is_1283": int(primary["event_count"]) == 1283,
            "pair_inventory_is_31": len(pair_inventory) == 31,
            "strict_pair_count_is_30": int(strict_overall["pair_count"]) == 30,
            "comparator_favors_linear_probe": number_value(cmp_decision["observed"], "comparator.observed") < 0,
            "all_primary_families_favor_graph_over_exact_nuisance_prior": int(per_family_positive)
            == len(exact_brier_per_family),
            "nuisance_permutation_observed_exceeds_null": number_value(
                nuisance_preserving["observed_family_macro_brier_gain"],
                "nuisance_permutation.observed_family_macro_brier_gain",
            )
            > number_value(
                permutation_null_summary["mean"],
                "nuisance_permutation.null_summary.mean",
            ),
            "knowledge_error_separation_exceeds_nuisance": _model_scores(
                knowledge_error_local, label="knowledge_error.local_joint_top8"
            )[1]
            > _model_scores(
                knowledge_error_exact,
                label="knowledge_error.exact_nuisance_family_balanced",
            )[1],
            "structured_action_inventory_is_600_rows_1680_events": (
                int(gate_design["row_count"]) == 600
                and int(gate_observed["unique_status_events"]) == 1680
            ),
            "structured_action_knowledge_gate_failed_56_of_60": (
                int(gate_knowledge["correct_count"]) == 56
                and int(gate_knowledge["required_count"]) == 57
                and not bool(gate_knowledge["passed"])
            ),
        },
    }


def build_c11_receipt(
    *,
    spectral_score: Path,
    connection_path_score: Path,
    risk_gate_repair_report: Path,
    stochastic_floor_report: Path,
    sealed_risk_report: Path,
) -> dict[str, Any]:
    spectral = load_json(spectral_score)
    connection = load_json(connection_path_score)
    risk_report = load_json(risk_gate_repair_report)
    partial_connection = load_json(stochastic_floor_report)
    sealed = load_json(sealed_risk_report)

    if spectral.get("kind") != "relational_intrinsic_spectral_field_score":
        raise ValueError("spectral score has wrong kind")
    if connection.get("kind") != "relational_connection_path_field_score":
        raise ValueError("connection-path score has wrong kind")
    if risk_report.get("kind") != "relational_pre_status_risk_gate_repair":
        raise ValueError("risk-gate repair report has wrong kind")
    if partial_connection.get("kind") != "relational_partial_frame_outcome_join_report":
        raise ValueError("partial-connection report has wrong kind")
    if sealed.get("kind") != "relational_pre_status_field_sealed_report":
        raise ValueError("sealed risk report has wrong kind")

    spectral_equal = dict_require(
        spectral.get("aggregate", {}).get("equal_view"), "spectral.aggregate.equal_view"
    )
    spectral_hd = dict_require(
        spectral_equal.get("honest_deceptive_slice"),
        "spectral.equal_view.honest_deceptive_slice",
    )
    spectral_fold_1 = dict_require(
        spectral.get("per_fold", {}).get("outer_1"), "spectral.per_fold[outer_1]"
    )
    spectral_fold_design = dict_require(
        spectral_fold_1.get("design_cell", {}).get("honest_deceptive_slice"),
        "spectral.outer_1.design_cell.honest_deceptive_slice",
    )
    spectral_fold_equal = dict_require(
        spectral_fold_1.get("equal_view", {}).get("honest_deceptive_slice"),
        "spectral.outer_1.equal_view.honest_deceptive_slice",
    )

    primary_model = str(connection.get("primary_model"))
    connection_aggregate = dict_require(
        connection.get("aggregate"), "connection.aggregate"
    ).get(primary_model)
    if not isinstance(connection_aggregate, Mapping):
        raise ValueError("connection primary model aggregate is missing")
    connection_hd = dict_require(
        connection_aggregate.get("honest_deceptive_slice"),
        "connection.primary.honest_deceptive_slice",
    )
    connection_adjudication = dict_require(
        connection.get("adjudication"), "connection.adjudication"
    )
    partial_summaries = dict_require(
        partial_connection.get("summaries"), "partial_connection.summaries"
    )
    stochastic_floor_count = number_value(
        partial_summaries["identical_prefix_mixed_outcome_group_count"],
        "partial_connection.summaries.identical_prefix_mixed_outcome_group_count",
    )
    stochastic_floor_groups = partial_summaries.get("identical_prefix_mixed_outcome_groups")
    if not isinstance(stochastic_floor_groups, list):
        raise ValueError(
            "expected partial_connection.summaries.identical_prefix_mixed_outcome_groups: list"
        )
    sealed_joint_view = dict_require(
        sealed.get("evaluation", {}).get("risk_fields", {}).get("views", {}).get("intervention_masked_action_free"),
        "sealed_risk.evaluation.risk_fields.views.intervention_masked_action_free",
    )
    sealed_joint_payload = dict_require(
        sealed_joint_view.get("joint"), "sealed_risk.intervention_masked_action_free.joint"
    )
    sealed_local_gain = dict_require(
        sealed_joint_payload.get("local_log_loss_gain_over_nuisance"),
        "sealed_risk.local_log_loss_gain_over_nuisance",
    )
    sealed_local_ci = dict_require(
        sealed_local_gain.get("scenario_cluster_ci"),
        "sealed_risk.local_log_loss_gain_over_nuisance.scenario_cluster_ci",
    )

    risk_primary_view = dict_require(
        risk_report.get("evaluation", {}).get("views", {}).get("intervention_masked_action_free"),
        "risk.evaluation.views.intervention_masked_action_free",
    )
    risk_joint = dict_require(
        risk_primary_view.get("joint"), "risk.intervention_masked_action_free.joint"
    )
    risk_models = dict_require(risk_joint.get("models"), "risk.intervention_masked_action_free.joint.models")
    risk_comparisons = dict_require(
        risk_joint.get("comparisons"), "risk.intervention_masked_action_free.joint.comparisons"
    )
    geometry_vs_nuisance = dict_require(
        risk_comparisons.get("geometry_only_logistic_over_nuisance_prior"),
        "geometry_only_logistic_over_nuisance_prior",
    )
    risk_interpretation = dict_require(
        risk_report.get("interpretation"), "risk.interpretation"
    )

    spectral_losing_design_cell_folds = []
    for fold, payload in dict_require(spectral.get("per_fold"), "spectral.per_fold").items():
        fold_payload = dict_require(payload, f"spectral.per_fold[{fold}]")
        fold_equal = dict_require(fold_payload.get("equal_view"), f"spectral.{fold}.equal_view")
        fold_design = dict_require(
            fold_payload.get("design_cell"), f"spectral.{fold}.design_cell"
        )
        equal_log_loss = number_value(
            fold_equal["event_pooled_multiclass_log_loss"],
            f"spectral.{fold}.equal_view.event_pooled_multiclass_log_loss",
        )
        design_log_loss = number_value(
            fold_design["event_pooled_multiclass_log_loss"],
            f"spectral.{fold}.design_cell.event_pooled_multiclass_log_loss",
        )
        spectral_losing_design_cell_folds.append(equal_log_loss >= design_log_loss)

    return {
        "schema_version": 1,
        "kind": "c11_precommitment_warning_public_receipt",
        "claim_id": "C11",
        "producer": "experiments/report_public_detection_receipts.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifacts": {
            "intrinsic_spectral_field_score": source_identity(spectral_score),
            "connection_path_field_score": source_identity(connection_path_score),
            "pre_status_risk_gate_repair_report": source_identity(risk_gate_repair_report),
            "partial_connection_report": source_identity(stochastic_floor_report),
            "pre_status_honestward_field_sealed_report": source_identity(sealed_risk_report),
        },
        "spectral_field": {
            "equal_view": {
                "honest_count": int(spectral_hd["honest_count"]),
                "deceptive_count": int(spectral_hd["deceptive_count"]),
                "auroc": number_value(
                    spectral_hd["auroc"], "spectral.equal_view.honest_deceptive_slice.auroc"
                ),
                "conditional_log_loss": number_value(
                    spectral_hd["conditional_log_loss"],
                    "spectral.equal_view.honest_deceptive_slice.conditional_log_loss",
                ),
            },
            "equal_view_loses_to_design_cell_in_each_fold": all(
                spectral_losing_design_cell_folds
            ),
            "folds": {
                fold: {
                    "equal_view": {
                        "auroc": number_value(
                            dict_require(
                                dict_require(v, f"spectral.{fold}").get("equal_view"),
                                f"spectral.{fold}.equal_view",
                            )["honest_deceptive_slice"]["auroc"],
                            f"spectral.{fold}.equal_view.auroc",
                        ),
                        "event_pooled_log_loss": number_value(
                            dict_require(
                                dict_require(v, f"spectral.{fold}").get("equal_view"),
                                f"spectral.{fold}.equal_view",
                            )["event_pooled_multiclass_log_loss"],
                            f"spectral.{fold}.equal_view.event_pooled_multiclass_log_loss",
                        ),
                    },
                    "design_cell": {
                        "auroc": number_value(
                            dict_require(
                                dict_require(v, f"spectral.{fold}").get("design_cell"),
                                f"spectral.{fold}.design_cell",
                            )["honest_deceptive_slice"]["auroc"],
                            f"spectral.{fold}.design_cell.auroc",
                        ),
                        "event_pooled_log_loss": number_value(
                            dict_require(
                                dict_require(v, f"spectral.{fold}").get("design_cell"),
                                f"spectral.{fold}.design_cell",
                            )["event_pooled_multiclass_log_loss"],
                            f"spectral.{fold}.design_cell.event_pooled_multiclass_log_loss",
                        ),
                    },
                    "equal_view_dominates_design_cell_log_loss": number_value(
                        dict_require(v, f"spectral.{fold}").get("equal_view")[
                            "event_pooled_multiclass_log_loss"
                        ],
                        f"spectral.{fold}.equal_view.event_pooled_multiclass_log_loss",
                    )
                    <= number_value(
                        dict_require(v, f"spectral.{fold}").get("design_cell")[
                            "event_pooled_multiclass_log_loss"
                        ],
                        f"spectral.{fold}.design_cell.event_pooled_multiclass_log_loss",
                    ),
                }
                for fold, v in dict_require(spectral.get("per_fold"), "spectral.per_fold").items()
            },
            "equal_view_outer_fold0_vs_design_cell": {
                "outer_1_auroc": number_value(
                    spectral_fold_equal["auroc"], "spectral.outer_1.equal_view.auroc"
                ),
                "design_cell_auroc": number_value(
                    spectral_fold_design["auroc"], "spectral.outer_1.design_cell.auroc"
                ),
                "equal_view_log_loss": number_value(
                    dict_require(
                        dict_require(spectral.get("per_fold"), "spectral.per_fold").get("outer_1"),
                        "spectral.per_fold[outer_1]",
                    )["equal_view"]["event_pooled_multiclass_log_loss"],
                    "spectral.outer_1.equal_view.event_pooled_multiclass_log_loss",
                ),
                "design_cell_log_loss": number_value(
                    dict_require(
                        dict_require(spectral.get("per_fold"), "spectral.per_fold").get("outer_1"),
                        "spectral.per_fold[outer_1]",
                    )["design_cell"]["event_pooled_multiclass_log_loss"],
                    "spectral.outer_1.design_cell.event_pooled_multiclass_log_loss",
                ),
            },
        },
        "connection_path_field": {
            "primary_model": primary_model,
            "status": connection_adjudication["status"],
            "controller_admitted": bool(connection_adjudication["controller_admitted"]),
            "honest_count": int(connection_hd["honest_count"]),
            "deceptive_count": int(connection_hd["deceptive_count"]),
            "event_count": int(connection_hd["event_count"]),
            "auroc": number_value(connection_hd["auroc"], "connection.primary.hd.auroc"),
            "conditional_log_loss": number_value(
                connection_hd["conditional_log_loss"], "connection.primary.hd.conditional_log_loss"
            ),
        },
        "risk_gate_repair": {
            "primary_view": str(risk_interpretation["primary_view"]),
            "primary_variant": str(risk_interpretation["primary_variant"]),
            "conclusion": str(risk_interpretation["conclusion"]),
            "interpretation": {
                "primary_geometry_only_log_loss_gain_over_nuisance": number_value(
                    geometry_vs_nuisance["mean_log_loss_gain"],
                    "risk.geometry_only_log_loss_gain",
                ),
                "primary_geometry_only_log_loss_gain_ci": {
                    "point": number_value(
                        geometry_vs_nuisance["log_loss_scenario_cluster_ci"]["point"],
                        "risk.log_loss_scenario_cluster_ci.point",
                    ),
                    "interval": [
                        number_value(
                            geometry_vs_nuisance["log_loss_scenario_cluster_ci"]["interval"][0],
                            "risk.log_loss_scenario_cluster_ci.interval[0]",
                        ),
                        number_value(
                            geometry_vs_nuisance["log_loss_scenario_cluster_ci"]["interval"][1],
                            "risk.log_loss_scenario_cluster_ci.interval[1]",
                        ),
                    ],
                    "pair_count": int(geometry_vs_nuisance["pair_count"]),
                },
                "primary_geometry_only_log_loss_gain_ci_lower_over_nuisance": number_value(
                    risk_interpretation[
                        "primary_geometry_only_log_loss_gain_ci_lower_over_nuisance"
                    ],
                    "risk.interpretation.primary_geometry_only_log_loss_gain_ci_lower_over_nuisance",
                ),
            },
            "secondary_comparisons": {
                "sealed_local_over_nuisance_prior": {
                    "mean_log_loss_gain": number_value(
                        sealed_local_gain["mean"],
                        "sealed_risk.local_log_loss_gain_over_nuisance.mean",
                    ),
                    "scenario_cluster_ci": {
                        "point": number_value(
                            sealed_local_ci["point"],
                            "sealed_risk.scenario_cluster_ci.point",
                        ),
                        "interval": [
                            number_value(
                                sealed_local_ci["interval"][0],
                                "sealed_risk.scenario_cluster_ci.interval[0]",
                            ),
                            number_value(
                                sealed_local_ci["interval"][1],
                                "sealed_risk.scenario_cluster_ci.interval[1]",
                            ),
                        ],
                        "pair_count": int(sealed_local_ci["defined_count"]),
                    },
                },
            },
            "stochastic_floor": {
                "identical_prefix_mixed_outcome_group_count": int(
                    stochastic_floor_count
                ),
                "identical_prefix_mixed_outcome_groups": len(stochastic_floor_groups),
            },
            "model_scores": {
                model: {
                    "honest_deceptive_count": int(model_payload["honest_deceptive_count"]),
                    "event_count": int(model_payload["event_count"]),
                    "honest_deceptive_auroc": number_value(
                        model_payload["honest_deceptive_auroc"],
                        f"risk.{model}.honest_deceptive_auroc",
                    ),
                    "mean_log_loss": number_value(
                        model_payload["mean_log_loss"],
                        f"risk.{model}.mean_log_loss",
                    ),
                }
                for model, model_payload in risk_models.items()
            },
        },
            "checks": {
            "spectral_h_d_slice_honest_deceptive_counts": (
                int(spectral_hd["honest_count"]) + int(spectral_hd["deceptive_count"]) == 106
            ),
            "connection_path_not_supported": connection_adjudication["status"]
            != "supported_exploratory_connection_response_field",
            "spectral_not_gain_dominant": all(spectral_losing_design_cell_folds),
            "risk_gate_conclusion_matches_summary": str(risk_interpretation["conclusion"])
            == "risk_gate_remains_unsolved_geometry_only_loses_to_design_prior",
            "risk_sealed_local_over_nuisance_is_negative": number_value(
                sealed_local_gain["mean"],
                "sealed_risk.local_log_loss_gain_over_nuisance.mean",
            )
            < 0,
            "risk_sealed_local_over_nuisance_ci_is_negative": number_value(
                sealed_local_ci["interval"][1],
                "sealed_risk.scenario_cluster_ci.interval[1]",
            )
            < 0,
            "stochastic_floor_mixed_outcome_group_count_is_20": int(
                stochastic_floor_count
            )
            == 20,
            "stochastic_floor_group_list_count_matches_summary": (
                int(stochastic_floor_count) == len(stochastic_floor_groups)
            ),
            "risk_geometry_only_log_loss_gain_is_negative": number_value(
                geometry_vs_nuisance["mean_log_loss_gain"],
                "risk.geometry_only_log_loss_gain",
            )
            < 0,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c10-outcome-report", type=Path, required=True)
    parser.add_argument("--c10-comparator-report", type=Path, required=True)
    parser.add_argument(
        "--structured-action-gate-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--c11-spectral-score", type=Path, required=True)
    parser.add_argument(
        "--c11-connection-path-score", type=Path, required=True
    )
    parser.add_argument(
        "--c11-risk-gate-repair-report", type=Path, required=True
    )
    parser.add_argument(
        "--c11-stochastic-floor-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--c11-sealed-risk-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--c10-out", type=Path, default=DEFAULT_C10_OUT)
    parser.add_argument("--c11-out", type=Path, default=DEFAULT_C11_OUT)
    args = parser.parse_args(argv)

    write_json(
        args.c10_out,
        build_c10_receipt(
            outcome_report=args.c10_outcome_report,
            comparator_report=args.c10_comparator_report,
            gate_report=args.structured_action_gate_report,
        ),
    )
    write_json(
        args.c11_out,
        build_c11_receipt(
            spectral_score=args.c11_spectral_score,
            connection_path_score=args.c11_connection_path_score,
            risk_gate_repair_report=args.c11_risk_gate_repair_report,
            stochastic_floor_report=args.c11_stochastic_floor_report,
            sealed_risk_report=args.c11_sealed_risk_report,
        ),
    )
    print(f"wrote {args.c10_out}")
    print(f"wrote {args.c11_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
