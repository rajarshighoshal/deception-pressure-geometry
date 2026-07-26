from __future__ import annotations

import numpy as np

from geoprobe.eval.relational_pre_status_risk_gate_repair import (
    GEOMETRY_ONLY_FEATURES,
    PRIMARY_VARIANT,
    PRIMARY_VIEW,
    REPORT_KIND,
    SCHEMA_VERSION,
    TASK_PRIVATE_EXCLUDED_FEATURES,
    _fit_linear_pool_alpha,
    _score_view_variant,
    GATE_CONCLUSION_INCONCLUSIVE,
    GATE_CONCLUSION_REPAIRED,
    GATE_CONCLUSION_UNSOLVED,
    geometry_only_gate_conclusion,
    validate_relational_pre_status_risk_gate_repair_report,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES


def _probabilities(deceptive: float) -> dict[str, float]:
    remaining = 1.0 - deceptive
    return {
        "HONEST": 0.9 * remaining,
        "DECEPTIVE": deceptive,
        "SKIP": 0.03 * remaining,
        "NO_ACTION": 0.02 * remaining,
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0.05 * remaining,
    }


def _row(
    *,
    fold: str,
    index: int,
    outcome: str,
    local_deceptive: float,
    nuisance_deceptive: float,
) -> dict[str, object]:
    return {
        "view": PRIMARY_VIEW,
        "variant": PRIMARY_VARIANT,
        "fold": fold,
        "event_id": f"{fold}:event:{index}",
        "root_id": f"root:{index}",
        "family": f"family:{fold}",
        "scenario_id": f"scenario:{index // 2}",
        "outcome_class": outcome,
        "support_count": 4 + index,
        "probabilities": {
            "local": _probabilities(local_deceptive),
            "nuisance": _probabilities(nuisance_deceptive),
            "base": {
                label: 1.0 / len(OUTCOME_CLASSES)
                for label in OUTCOME_CLASSES
            },
        },
    }


def test_risk_gate_report_self_hash_validator_accepts_minimal_report() -> None:
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "success",
        "scope": {"uses_model_or_gpu": False},
        "interpretation": {
            "primary_view": PRIMARY_VIEW,
            "primary_variant": PRIMARY_VARIANT,
            "conclusion": "risk_gate_remains_unsolved_geometry_only_loses_to_design_prior",
        },
    }
    from geoprobe.eval import relational_pre_status_risk_gate_repair as module

    report["report_sha256"] = module._self_hash(report)
    validate_relational_pre_status_risk_gate_repair_report(report)


def test_linear_pool_selection_uses_training_fold_only() -> None:
    local = np.asarray(
        [
            [0.05, 0.9, 0.02, 0.01, 0.02],
            [0.1, 0.8, 0.04, 0.02, 0.04],
            [0.1, 0.8, 0.04, 0.02, 0.04],
            [0.05, 0.9, 0.02, 0.01, 0.02],
        ],
        dtype=float,
    )
    nuisance = np.asarray(
        [
            [0.9, 0.05, 0.02, 0.01, 0.02],
            [0.8, 0.1, 0.04, 0.02, 0.04],
            [0.8, 0.1, 0.04, 0.02, 0.04],
            [0.9, 0.05, 0.02, 0.01, 0.02],
        ],
        dtype=float,
    )
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    assert _fit_linear_pool_alpha(nuisance[:2], local[:2], labels[:2]) == 1.0
    assert _fit_linear_pool_alpha(nuisance[2:], local[2:], labels[2:]) == 0.0


def test_score_view_variant_emits_fold_safe_geometry_only_rows() -> None:
    rows = [
        _row(
            fold="outer_1",
            index=0,
            outcome="DECEPTIVE",
            local_deceptive=0.8,
            nuisance_deceptive=0.7,
        ),
        _row(
            fold="outer_1",
            index=1,
            outcome="HONEST",
            local_deceptive=0.2,
            nuisance_deceptive=0.3,
        ),
        _row(
            fold="outer_2",
            index=2,
            outcome="DECEPTIVE",
            local_deceptive=0.7,
            nuisance_deceptive=0.6,
        ),
        _row(
            fold="outer_2",
            index=3,
            outcome="HONEST",
            local_deceptive=0.3,
            nuisance_deceptive=0.4,
        ),
    ]
    scored, selections, calibration = _score_view_variant(
        rows,
        folds=("outer_1", "outer_2"),
    )
    assert {row["fold"] for row in selections} == {"outer_1", "outer_2"}
    assert "geometry_only_logistic" in {row["model"] for row in scored}
    assert calibration["geometry_only_logistic"]["event_count"] == len(rows)
    assert "true_status" not in GEOMETRY_ONLY_FEATURES
    assert "desired_status" not in GEOMETRY_ONLY_FEATURES
    assert "true_status" in TASK_PRIVATE_EXCLUDED_FEATURES
    assert "desired_status" in TASK_PRIVATE_EXCLUDED_FEATURES


def _comparison(
    *,
    mean_log_loss_gain: float | None,
    ci_lower: float | None,
    ci_upper: float | None = 1.0,
) -> dict[str, object]:
    log_loss_ci: dict[str, object] | None
    if ci_lower is None:
        log_loss_ci = None
    else:
        log_loss_ci = {
            "point": mean_log_loss_gain,
            "interval": [ci_lower, ci_upper],
        }
    return {
        "candidate_model": "geometry_only_logistic",
        "comparator_model": "sealed_nuisance_prior",
        "mean_log_loss_gain": mean_log_loss_gain,
        "log_loss_scenario_cluster_ci": log_loss_ci,
    }


def test_gate_repaired_requires_positive_point_and_positive_ci_lower() -> None:
    comparison = _comparison(mean_log_loss_gain=0.05, ci_lower=0.01, ci_upper=0.09)
    assert geometry_only_gate_conclusion(comparison) == GATE_CONCLUSION_REPAIRED


def test_gate_positive_point_but_ci_crosses_zero_is_not_repaired() -> None:
    comparison = _comparison(mean_log_loss_gain=0.004, ci_lower=-0.02, ci_upper=0.03)
    conclusion = geometry_only_gate_conclusion(comparison)
    assert conclusion != GATE_CONCLUSION_REPAIRED
    assert conclusion == GATE_CONCLUSION_INCONCLUSIVE


def test_gate_positive_point_with_ci_lower_exactly_zero_is_not_repaired() -> None:
    comparison = _comparison(mean_log_loss_gain=0.01, ci_lower=0.0, ci_upper=0.05)
    assert geometry_only_gate_conclusion(comparison) != GATE_CONCLUSION_REPAIRED


def test_gate_positive_point_without_ci_is_inconclusive_not_repaired() -> None:
    comparison = _comparison(mean_log_loss_gain=0.02, ci_lower=None)
    assert geometry_only_gate_conclusion(comparison) == GATE_CONCLUSION_INCONCLUSIVE


def test_gate_nonpositive_point_gain_remains_unsolved() -> None:
    comparison = _comparison(mean_log_loss_gain=-0.0218, ci_lower=-0.0733, ci_upper=0.0233)
    assert geometry_only_gate_conclusion(comparison) == GATE_CONCLUSION_UNSOLVED


def test_gate_missing_point_gain_is_inconclusive() -> None:
    comparison = _comparison(mean_log_loss_gain=None, ci_lower=None)
    assert geometry_only_gate_conclusion(comparison) == GATE_CONCLUSION_INCONCLUSIVE
