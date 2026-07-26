"""Markdown renderer for the complete-path connection-response field."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES

_MODELS = (
    "full_path_design_conditioned",
    "full_path_unrestricted",
    "incoming_design_conditioned",
    "common_outgoing_design_conditioned",
    "asymmetry_design_conditioned",
    "residual_full_path_design_conditioned",
    "attention_full_path_design_conditioned",
    "layer_transport_full_path_design_conditioned",
    "identity_shuffled_path_design_conditioned",
    "design_cell",
    "base_rate",
    "one_state_spectral",
)

_COMPARATORS = (
    "design_cell",
    "base_rate",
    "incoming_design_conditioned",
    "one_state_spectral",
    "identity_shuffled_path_design_conditioned",
)


class RelationalConnectionPathFieldReportError(ValueError):
    """Raised when the complete-path field markdown cannot be rendered."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalConnectionPathFieldReportError(f"{name} must be an object")
    return value


def _rows(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise RelationalConnectionPathFieldReportError(f"{name} must be an array")
    return [_mapping(row, f"{name} row") for row in value]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalConnectionPathFieldReportError(f"{name} must be a non-empty string")
    return value


def _int(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RelationalConnectionPathFieldReportError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RelationalConnectionPathFieldReportError(f"{name} must be numeric")
    return float(value)


def _format(value: object, digits: int = 5) -> str:
    if value is None:
        return "not estimable"
    return f"{_number(value, 'metric value'):.{digits}f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        raise RelationalConnectionPathFieldReportError("interval must have two endpoints")
    return f"[{_format(value[0])}, {_format(value[1])}]"


def _sha256(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RelationalConnectionPathFieldReportError(f"{name} must be lowercase sha256")
    return text


def _coverage(bank: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(bank.get("coverage"), "bank coverage")
    return {
        "path_count": _int(raw.get("path_count"), "coverage.path_count", minimum=1),
        "scenario_count": _int(raw.get("scenario_count"), "coverage.scenario_count", minimum=1),
        "relation_side_count": _int(
            raw.get("relation_side_count"), "coverage.relation_side_count", minimum=1
        ),
        "class_counts": {label: _int(raw.get("class_counts", {}).get(label, 0), f"class_count.{label}") for label in OUTCOME_CLASSES},
    }


def _cohort_signature(bank: Mapping[str, Any]) -> tuple[int, int, int]:
    paths = _rows(bank.get("paths"), "bank paths")
    families = {str(row.get("family")) for row in paths}
    folds = {str(row.get("fold")) for row in paths}
    if not families or not folds:
        raise RelationalConnectionPathFieldReportError(
            "bank path metadata must contain family and fold"
        )
    return _coverage(bank)["path_count"], len(families), len(folds)


def _model_metrics(aggregate: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    payload = _mapping(aggregate.get(model), f"aggregate[{model}]")
    return payload


def _render_model_table(score: Mapping[str, Any]) -> list[str]:
    aggregate = _mapping(score.get("aggregate"), "aggregate metrics")
    lines = [
        "## Aggregate model comparison",
        "",
        "| Model | Event log-loss | Event Brier | Family-macro log-loss | Family-macro Brier | H/D conditional log-loss | H/D AUROC | H/D AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in _MODELS:
        metrics = _model_metrics(aggregate, model)
        hd = _mapping(metrics.get("honest_deceptive_slice"), "honest/deceptive slice")
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(
                (
                    _format(metrics.get("event_pooled_multiclass_log_loss")),
                    _format(metrics.get("event_pooled_multiclass_brier")),
                    _format(metrics.get("family_macro_multiclass_log_loss")),
                    _format(metrics.get("family_macro_multiclass_brier")),
                    _format(hd.get("conditional_log_loss")),
                    _format(hd.get("auroc")),
                    _format(hd.get("average_precision")),
                )
            )
            + " |"
        )
    return lines


def _render_design_cells(score: Mapping[str, Any], primary: str) -> list[str]:
    per_cell = _mapping(score.get("per_design_cell"), "per-design-cell metrics")
    lines = [
        "## Knowledge-correct design-cell directions",
        "",
        "Positive H/D log-loss gain favors the complete path over the design-cell prior.",
        "",
        "| True | Desired | H | D | Primary H/D log-loss | Gain over design prior |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cell_hash in sorted(per_cell):
        payload = _mapping(per_cell[cell_hash], f"design cell {cell_hash}")
        cell = _mapping(payload.get("design_cell"), "design-cell identity")
        if cell.get("baseline_knowledge_correct") is not True:
            continue
        metrics = _mapping(payload.get("metrics"), "design-cell metrics")
        primary_metrics = _mapping(metrics.get(primary), "design-cell primary")
        hd = _mapping(
            primary_metrics.get("honest_deceptive_slice"), "design-cell H/D"
        )
        gains = _mapping(
            payload.get("full_path_gain_over_comparators"), "design-cell gains"
        )
        design_gain = _mapping(gains.get("design_cell"), "design-cell prior gain")
        lines.append(
            f"| {cell.get('true_status')} | {cell.get('desired_status')} | "
            f"{hd.get('honest_count')} | {hd.get('deceptive_count')} | "
            f"{_format(hd.get('conditional_log_loss'))} | "
            f"{_format(design_gain.get('honest_deceptive_conditional_log_loss_gain'))} |"
        )
    return lines


def _render_gains(score: Mapping[str, Any]) -> list[str]:
    gains = _mapping(score.get("full_path_gain_over_comparators"), "full-path gains")
    lines = [
        "## Full-path gain over comparator baselines",
        "",
        "Positive values favour the full-path conditioned predictor.",
        "",
        "| Comparator | Event log-loss gain | Event Brier gain | H/D log-loss gain |",
        "|---|---:|---:|---:|",
    ]
    for comparator in _COMPARATORS:
        delta = _mapping(gains.get(comparator), f"gain[{comparator}]")
        lines.append(
            "| "
            + comparator
            + " | "
            + " | ".join(
                (
                    _format(delta.get("event_pooled_log_loss_gain")),
                    _format(delta.get("event_pooled_brier_gain")),
                    _format(delta.get("honest_deceptive_conditional_log_loss_gain")),
                )
            )
            + " |"
        )
    return lines


def _render_uncertainty(score: Mapping[str, Any]) -> list[str]:
    uncertainty = _mapping(
        score.get("post_score_descriptive_uncertainty"), "family-cluster uncertainty"
    )
    comparisons = _mapping(uncertainty.get("comparisons"), "uncertainty comparisons")
    lines = [
        "## Family-cluster uncertainty",
        "",
        f"Paired percentile bootstrap over {uncertainty.get('family_count')} families, "
        f"{uncertainty.get('resamples')} resamples, seed={uncertainty.get('seed')}.",
        "",
        "| Comparator | Metric | Point estimate | 95% interval | Positive fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for comparator in _COMPARATORS:
        payload = _mapping(comparisons.get(comparator), f"uncertainty comparator {comparator}")
        for metric_name in ("event_pooled_log_loss_gain", "honest_deceptive_conditional_log_loss_gain"):
            metric = _mapping(payload.get(metric_name), f"metric {metric_name}")
            lines.append(
                "| "
                + comparator
                + " | "
                + metric_name
                + " | "
                + _format(metric.get("point_estimate"))
                + " | "
                + _interval(metric.get("percentile_95"))
                + " | "
                + _format(metric.get("fraction_positive"), 3)
                + " |"
            )
    return lines


def _render_per_fold(score: Mapping[str, Any], primary: str) -> list[str]:
    per_fold = _mapping(score.get("per_fold"), "per-fold metrics")
    lines = [
        "## Outer-fold performance",
        "",
        "| Fold | Event count | Event log-loss | H/D log-loss | H/D AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold in sorted(per_fold):
        payload = _mapping(per_fold[fold], f"per-fold {fold}")
        metrics = _model_metrics(payload, primary)
        hd = _mapping(metrics.get("honest_deceptive_slice"), "fold H/D slice")
        lines.append(
            "| "
            + str(fold)
            + " | "
            + f"{_int(metrics.get('event_count'), f'{fold} event count')} | "
            + _format(metrics.get('event_pooled_multiclass_log_loss'))
            + " | "
            + _format(hd.get("conditional_log_loss"))
            + " | "
            + _format(hd.get("auroc"))
            + " |"
        )
    return lines


def render_connection_path_field_markdown(
    *,
    score: Mapping[str, Any],
    ledger: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> str:
    """Render a durable non-opinionated complete-path field report."""
    if score.get("kind") != "relational_connection_path_field_score":
        raise RelationalConnectionPathFieldReportError("score kind is invalid")
    if ledger.get("kind") != "relational_connection_path_prediction_ledger":
        raise RelationalConnectionPathFieldReportError("prediction-ledger kind is invalid")
    if bank.get("kind") != "relational_complete_path_bank":
        raise RelationalConnectionPathFieldReportError("complete path bank kind is invalid")
    if score.get("prediction_ledger_sha256") != ledger.get("prediction_ledger_sha256"):
        raise RelationalConnectionPathFieldReportError("score is not bound to the supplied prediction ledger")
    if score.get("complete_path_bank_sha256") != bank.get("bank_sha256"):
        raise RelationalConnectionPathFieldReportError("score is not bound to the supplied complete-path bank")
    if ledger.get("complete_path_bank_sha256") != bank.get("bank_sha256"):
        raise RelationalConnectionPathFieldReportError("prediction ledger is not bound to the supplied complete-path bank")

    path_count = _coverage(bank)["path_count"]
    scenario_count = _coverage(bank)["scenario_count"]
    relation_side_count = _coverage(bank)["relation_side_count"]
    _, family_count, fold_count = _cohort_signature(bank)
    counts = _coverage(bank)["class_counts"]
    coverage = [
        f"- Path count: {path_count}",
        f"- Scenario count: {scenario_count}",
        f"- Families / outer folds: {family_count} / {fold_count}",
        f"- Stable relation-side budget: {relation_side_count}",
        "- Outcome support: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in OUTCOME_CLASSES),
    ]

    primary_model = _string(score.get("primary_model"), "primary_model")
    aggregate = _mapping(score.get("aggregate"), "aggregate")
    primary = _model_metrics(aggregate, primary_model)
    boundary = _mapping(score.get("claim_boundary"), "claim_boundary")
    adjudication = _mapping(score.get("adjudication"), "adjudication")
    supported = adjudication.get("status") == "supported_exploratory_connection_response_field"
    lines = [
        "# Complete-Path Connection-Response Field",
        "",
        "## Cohort",
        "",
        *coverage,
        "",
        "## Result",
        "",
        (
            "**The frozen checkpoint-level connection-response field is supported as exploratory discovery evidence.**"
            if supported
            else "**The frozen checkpoint-level connection-response field is not supported.**"
        ),
        "",
        (
            "Its complete `I+C+O` path adds stable held-out honest-versus-deceptive information under the frozen criteria."
            if supported
            else "The complete `I+C+O` path does not improve held-out honest-versus-deceptive proper scores over the required controls in both knowledge-correct design cells. This is a clean negative for the persisted invariant checkpoint instrument, not a refutation of richer gauge transport or post-commitment geometry."
        ),
        "",
        f"Adjudication: `{adjudication.get('status')}`. Controller admitted: `{adjudication.get('controller_admitted')}`.",
        "",
        "## Primary model",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Primary model | " + primary_model + " |",
        "| Event pooled log-loss | "
        + _format(primary.get("event_pooled_multiclass_log_loss"))
        + " |",
        "| Event pooled Brier | "
        + _format(primary.get("event_pooled_multiclass_brier"))
        + " |",
        "| H/D conditional log-loss | "
        + _format(
            _mapping(
                primary.get("honest_deceptive_slice"),
                "primary honest/deceptive slice",
            ).get("conditional_log_loss")
        )
        + " |",
        "| Event count | " + str(_int(primary.get("event_count"), "primary event count")) + " |",
        "",
    ]
    lines.extend(_render_model_table(score))
    lines.extend(["", *_render_gains(score), ""])
    lines.extend(["", *_render_uncertainty(score), ""])
    lines.extend(["", *_render_per_fold(score, primary_model), ""])
    lines.extend(["", *_render_design_cells(score, primary_model), ""])
    lines.extend(
        [
            "## Claim boundary",
            "",
            f"- Exploratory cross-fitted protocol: `{boundary.get('exploratory_cross_fitted_only', True)}`",
            f"- Complete-sample0 design point only: `{boundary.get('complete_sample0_path_only', True)}`",
            f"- All outcomes retained: `{boundary.get('all_outcomes_retained', True)}`",
            f"- Global-flat PCA or coordinates: `{boundary.get('global_flat_coordinates', 'not_available')}`",
            f"- Gauge/holonomy claims: `{boundary.get('gauge_transport_or_holonomy', 'not_available')}`",
            f"- Controller tested: `{boundary.get('controller', 'not_tested')}`",
            f"- Arbitrary success threshold: `{boundary.get('arbitrary_success_threshold', 'absent')}`",
            "",
            "## Output bindings",
            "",
            f"- Complete-path bank SHA-256: `{_sha256(bank.get('bank_sha256'), 'bank SHA')}`",
            f"- Prediction ledger SHA-256: `{_sha256(ledger.get('prediction_ledger_sha256'), 'prediction ledger SHA')}`",
            f"- Field score SHA-256: `{_sha256(score.get('score_sha256'), 'score SHA')}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["RelationalConnectionPathFieldReportError", "render_connection_path_field_markdown"]
