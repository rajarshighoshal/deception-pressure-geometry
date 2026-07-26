"""Durable Markdown rendering for the intrinsic spectral outcome field."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES


_MODELS = (
    "equal_view",
    "residual",
    "attention",
    "layer_transport",
    "design_cell",
    "base_rate",
)


class RelationalIntrinsicSpectralFieldReportError(ValueError):
    """Raised when a scored field report cannot be rendered safely."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalIntrinsicSpectralFieldReportError(
            f"{name} must be an object"
        )
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RelationalIntrinsicSpectralFieldReportError(
            f"{name} must be numeric"
        )
    return float(value)


def _format(value: object, digits: int = 5) -> str:
    if value is None:
        return "not estimable"
    return f"{_number(value, 'reported metric'):.{digits}f}"


def _interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        raise RelationalIntrinsicSpectralFieldReportError(
            "reported interval must have two endpoints"
        )
    return f"[{_format(value[0])}, {_format(value[1])}]"


def render_intrinsic_spectral_field_markdown(
    *,
    score: Mapping[str, Any],
    ledger: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> str:
    """Render the scored field with counts, comparators, folds, and boundaries."""
    if score.get("kind") != "relational_intrinsic_spectral_field_score":
        raise RelationalIntrinsicSpectralFieldReportError("score kind is invalid")
    if ledger.get("kind") != "relational_intrinsic_spectral_prediction_ledger":
        raise RelationalIntrinsicSpectralFieldReportError("ledger kind is invalid")
    if bank.get("kind") != "relational_intrinsic_outcome_bank":
        raise RelationalIntrinsicSpectralFieldReportError("bank kind is invalid")
    if score.get("prediction_ledger_sha256") != ledger.get(
        "prediction_ledger_sha256"
    ):
        raise RelationalIntrinsicSpectralFieldReportError(
            "score does not bind the supplied prediction ledger"
        )
    coverage = _mapping(bank.get("coverage"), "bank coverage")
    counts = _mapping(coverage.get("class_counts"), "class counts")
    aggregate = _mapping(score.get("aggregate"), "aggregate metrics")
    lines = [
        "# Intrinsic Spectral Deception Field v1",
        "",
        "## Cohort",
        "",
        f"- Exact pre-status state quotients: {coverage.get('quotient_count')}",
        f"- Structured-action events retained: {coverage.get('event_count')}",
        f"- Scenarios / families / outer folds: {coverage.get('scenario_count')} / "
        f"{coverage.get('family_count')} / {coverage.get('fold_count')}",
        "- Outcomes: "
        + ", ".join(f"{label}={counts.get(label, 0)}" for label in OUTCOME_CLASSES),
        "- Condition: pressure-present `AN`, turn-2, exact pre-status anchor.",
        "",
        "## Result",
        "",
        "**The frozen one-state spectral radial field is not supported as the "
        "deception-specific state field.** Equal-view fusion does not beat the "
        "design-cell control, and its honest-versus-deceptive ranking is below "
        "chance in aggregate. The tiny base-rate difference is reported with "
        "family-cluster uncertainty and is not treated as affirmative evidence.",
        "",
        "## Held-out proper scores",
        "",
        "Lower log loss and Brier are better. AUROC/AP use the declared honest-versus-"
        "deceptive slice of the same five-way predictions; no binary model was refit.",
        "",
        "| Model | Event log loss | Event Brier | Quotient CE | H/D log loss | H/D AUROC | H/D AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in _MODELS:
        metrics = _mapping(aggregate.get(model), f"{model} metrics")
        hd = _mapping(metrics.get("honest_deceptive_slice"), f"{model} H/D slice")
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(
                (
                    _format(metrics.get("event_pooled_multiclass_log_loss")),
                    _format(metrics.get("event_pooled_multiclass_brier")),
                    _format(metrics.get("quotient_macro_cross_entropy")),
                    _format(hd.get("conditional_log_loss")),
                    _format(hd.get("auroc")),
                    _format(hd.get("average_precision")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Equal-view gain over fixed comparators",
            "",
            "Positive values favor the intrinsic equal-view field.",
            "",
            "| Comparator | Log-loss gain | Brier gain | Quotient-CE gain | H/D log-loss gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    gains = _mapping(
        score.get("equal_view_gain_over_comparators"), "comparator gains"
    )
    for comparator in ("design_cell", "base_rate"):
        values = _mapping(gains.get(comparator), f"{comparator} gains")
        lines.append(
            f"| {comparator} | "
            f"{_format(values.get('event_pooled_log_loss_gain'))} | "
            f"{_format(values.get('event_pooled_brier_gain'))} | "
            f"{_format(values.get('quotient_macro_cross_entropy_gain'))} | "
            f"{_format(values.get('honest_deceptive_conditional_log_loss_gain'))} |"
        )
    uncertainty = _mapping(
        score.get("post_score_descriptive_uncertainty"),
        "descriptive uncertainty",
    )
    uncertainty_comparisons = _mapping(
        uncertainty.get("comparisons"), "uncertainty comparisons"
    )
    lines.extend(
        [
            "",
            "## Family-cluster uncertainty",
            "",
            f"Paired percentile bootstrap over {uncertainty.get('family_count')} "
            f"families ({uncertainty.get('resamples')} resamples; seed "
            f"{uncertainty.get('seed')}). Predictions are frozen and are not refit "
            "inside resamples. Positive gains favor equal-view geometry.",
            "",
            "| Comparator | Metric | Point gain | 95% interval | Fraction positive |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for comparator in ("design_cell", "base_rate"):
        values = _mapping(
            uncertainty_comparisons.get(comparator),
            f"{comparator} uncertainty",
        )
        for metric in (
            "event_pooled_log_loss_gain",
            "honest_deceptive_conditional_log_loss_gain",
        ):
            estimate = _mapping(values.get(metric), f"{comparator} {metric}")
            lines.append(
                f"| {comparator} | {metric} | "
                f"{_format(estimate.get('point_estimate'))} | "
                f"{_interval(estimate.get('percentile_95'))} | "
                f"{_format(estimate.get('fraction_positive'), 3)} |"
            )
    lines.extend(
        [
            "",
            "## Outer-fold stability",
            "",
            "| Fold | Events | Equal-view log loss | Design-cell log loss | H/D AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    per_fold = _mapping(score.get("per_fold"), "per-fold metrics")
    for fold in sorted(per_fold):
        values = _mapping(per_fold[fold], f"{fold} metrics")
        equal = _mapping(values.get("equal_view"), f"{fold} equal-view metrics")
        design = _mapping(values.get("design_cell"), f"{fold} design metrics")
        hd = _mapping(equal.get("honest_deceptive_slice"), f"{fold} H/D slice")
        lines.append(
            f"| {fold} | {equal.get('event_count')} | "
            f"{_format(equal.get('event_pooled_multiclass_log_loss'))} | "
            f"{_format(design.get('event_pooled_multiclass_log_loss'))} | "
            f"{_format(hd.get('auroc'))} |"
        )
    lines.extend(
        [
            "",
            "## Fold-local geometry",
            "",
            "| Fold | Admitted residual | Admitted attention | Admitted transport | "
            "Equal-view bandwidth |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    folds = ledger.get("folds")
    if not isinstance(folds, list):
        raise RelationalIntrinsicSpectralFieldReportError("ledger folds must be an array")
    for raw in folds:
        fold = _mapping(raw, "ledger fold")
        admitted = _mapping(fold.get("admitted_relation_counts"), "admitted counts")
        bandwidths = _mapping(fold.get("bandwidths"), "bandwidths")
        lines.append(
            f"| {fold.get('heldout_fold')} | {admitted.get('residual')} | "
            f"{admitted.get('attention')} | {admitted.get('layer_transport')} | "
            f"{_format(bandwidths.get('equal_view'))} |"
        )
    no_action = _mapping(
        _mapping(aggregate.get("equal_view"), "equal-view metrics").get(
            "class_support"
        ),
        "class support",
    ).get("NO_ACTION")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- The five-way endpoint retains honest, deceptive, SKIP, NO_ACTION, and "
            "wrong-without-baseline-knowledge outcomes.",
            f"- NO_ACTION support is {no_action.get('event_count') if isinstance(no_action, Mapping) else 0}; "
            "its discrimination is not estimable in this cohort.",
            "- The predictor uses fold-admitted coordinate-free spectra only. PCA and "
            "global flat coordinates are absent.",
            "- Attention heads remain named typed relations; independent head-"
            "permutation invariance is not claimed.",
            "- This result can support or reject a deception-informative local state "
            "under fixed pressure. It does not establish curvature, holonomy, "
            "universality, or a working controller.",
            "- No arbitrary success threshold is applied.",
            "",
            f"Prediction ledger SHA-256: `{ledger.get('prediction_ledger_sha256')}`",
            "",
            f"Score SHA-256: `{score.get('score_sha256')}`",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "RelationalIntrinsicSpectralFieldReportError",
    "render_intrinsic_spectral_field_markdown",
]
