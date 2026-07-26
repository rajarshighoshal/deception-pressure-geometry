"""Durable, hash-bound reports for offline pre-status field evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from geoprobe.data import relational_pre_status_rooted_star_store as rooted_star_store
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
)
from geoprobe.eval import relational_pre_status_field_evaluation as field_evaluation
from geoprobe.eval import relational_pre_status_risk_diagnostics as risk_diagnostics
from geoprobe.eval import relational_pre_status_risk_field as risk_field
from geoprobe.eval import relational_pre_status_rooted_graph_artifact as graph_artifact
from geoprobe.eval import relational_pre_status_supervision as supervision_module
from geoprobe.geometry import relational_pre_status_honestward as honestward_geometry
from geoprobe.geometry import relational_pre_status_rooted_graph as rooted_graph_geometry
from geoprobe.geometry import relational_pre_status_rooted_metric as rooted_metric_geometry
from geoprobe.eval.relational_pre_status_field_evaluation import (
    evaluate_pre_status_honestward_fields,
    evaluate_pre_status_risk_fields,
)
from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (
    LoadedRelationalPreStatusRootedGraphs,
)
from geoprobe.eval.relational_pre_status_supervision import (
    RelationalPreStatusSupervision,
)
from geoprobe.io import file_sha256
from geoprobe.provenance import git_provenance


SCHEMA_VERSION = 1
REPORT_KIND = "relational_pre_status_field_report"


class RelationalPreStatusFieldReportError(ValueError):
    """Raised when a pre-status field report is not bound to its inputs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusFieldReportError(
            "report value is not canonical JSON"
        ) from error


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelationalPreStatusFieldReportError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _self_hash(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return sha256(_canonical(payload)).hexdigest()


def validate_relational_pre_status_field_report(report: Mapping[str, Any]) -> None:
    """Validate the terminal schema/status/kind/self-hash envelope."""
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("kind") != REPORT_KIND
        or report.get("status") != "success"
        or report.get("report_sha256") != _self_hash(report)
    ):
        raise RelationalPreStatusFieldReportError(
            "field report schema, status, kind, or self-hash is invalid"
        )


def _binding(path: Path, *, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RelationalPreStatusFieldReportError(f"input is absent: {resolved}")
    actual = file_sha256(resolved)
    if expected_sha256 is not None and actual != _sha(expected_sha256, "expected input SHA-256"):
        raise RelationalPreStatusFieldReportError(
            f"input differs from its expected physical SHA-256: {resolved}"
        )
    return {"path": str(resolved), "sha256": actual}


def _source_files(extra_paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    paths = {
        "field_report": Path(__file__).resolve(),
        "field_evaluation": Path(field_evaluation.__file__).resolve(),
        "risk_diagnostics": Path(risk_diagnostics.__file__).resolve(),
        "risk_field": Path(risk_field.__file__).resolve(),
        "rooted_graph_artifact": Path(graph_artifact.__file__).resolve(),
        "honestward_geometry": Path(honestward_geometry.__file__).resolve(),
        "rooted_graph_geometry": Path(rooted_graph_geometry.__file__).resolve(),
        "rooted_metric_geometry": Path(rooted_metric_geometry.__file__).resolve(),
        "supervision": Path(supervision_module.__file__).resolve(),
        "rooted_star_store": Path(rooted_star_store.__file__).resolve(),
    }
    for path in extra_paths:
        resolved = Path(path).resolve()
        paths.setdefault(resolved.stem, resolved)
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in sorted(paths.items())
    }


def _count_crossings(
    supervision: RelationalPreStatusSupervision,
) -> dict[str, int]:
    return {
        view: len(rows)
        for view, rows in sorted(supervision.honestward_observations_by_view.items())
    }


def _variant_neighbors(
    value: Mapping[str, Any], field: str
) -> Mapping[str, frozenset[str]]:
    grouped: dict[str, set[str]] = {}
    for row in value[field]:
        grouped.setdefault(str(row["source_id"]), set()).add(str(row["target_id"]))
    return {source: frozenset(targets) for source, targets in grouped.items()}


def _mean_neighbor_overlap(
    left: Mapping[str, frozenset[str]],
    right: Mapping[str, frozenset[str]],
) -> float | None:
    sources = sorted(set(left).intersection(right))
    values = [
        len(left[source].intersection(right[source]))
        / max(len(left[source].union(right[source])), 1)
        for source in sources
    ]
    return float(sum(values) / len(values)) if values else None


def summarize_relational_pre_status_rooted_graphs(
    graphs: LoadedRelationalPreStatusRootedGraphs,
) -> dict[str, Any]:
    """Summarize label-free graph fidelity, nuisance, and view ablations."""
    result: dict[str, Any] = {}
    for view, folds in sorted(graphs.folds_by_view.items()):
        view_rows: dict[str, Any] = {}
        for fold, artifact in sorted(folds.items()):
            graph = artifact["graph"]
            recalls = [
                float(row["candidate_recall_at_64"])
                for row in graph["candidate_recall_audit"]
            ]
            dispersion = list(graph["within_fibre_dispersion"].values())
            repeated = [
                row
                for row in dispersion
                if int(row["geometry_representative_count"]) > 1
            ]
            variants = graph["variants"]
            neighbors = {
                name: _variant_neighbors(value, "query_to_training")
                for name, value in variants.items()
            }
            view_rows[fold] = {
                "candidate_recall_at_64": {
                    "audit_count": len(recalls),
                    "mean": float(sum(recalls) / len(recalls)) if recalls else None,
                    "minimum": min(recalls) if recalls else None,
                },
                "within_fibre_replay_nuisance": {
                    "state_count": len(dispersion),
                    "repeated_state_count": len(repeated),
                    "mean_residual": (
                        float(
                            sum(float(row["mean_residual"]) for row in repeated)
                            / len(repeated)
                        )
                        if repeated
                        else None
                    ),
                    "mean_attention_head_set": (
                        float(
                            sum(
                                float(row["mean_attention_head_set"])
                                for row in repeated
                            )
                            / len(repeated)
                        )
                        if repeated
                        else None
                    ),
                },
                "query_neighborhood_jaccard": {
                    "joint_vs_residual": _mean_neighbor_overlap(
                        neighbors["joint"], neighbors["residual_only"]
                    ),
                    "joint_vs_attention": _mean_neighbor_overlap(
                        neighbors["joint"], neighbors["attention_only"]
                    ),
                    "residual_vs_attention": _mean_neighbor_overlap(
                        neighbors["residual_only"], neighbors["attention_only"]
                    ),
                },
                "metric_scaler": dict(graph["metric_scaler"]),
            }
        result[view] = view_rows
    return result


def build_relational_pre_status_field_report(
    index: RelationalPreStatusRootedStarIndex,
    graphs: LoadedRelationalPreStatusRootedGraphs,
    supervision: RelationalPreStatusSupervision,
    *,
    expected_rooted_star_manifest_sha256: str,
    rooted_graph_artifact_root: Path,
    outcome_report_path: Path,
    roster_path: Path,
    argv: Sequence[str] = (),
    extra_source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Evaluate bound offline fields without making controller-effect claims."""
    expected_bank_sha = _sha(
        expected_rooted_star_manifest_sha256,
        "expected rooted-star manifest SHA-256",
    )
    bank_manifest = Path(index.artifact_root).resolve() / "manifest.json"
    bank = _binding(bank_manifest, expected_sha256=expected_bank_sha)
    if index.manifest_sha256 != expected_bank_sha:
        raise RelationalPreStatusFieldReportError(
            "rooted-star index does not bind the expected manifest"
        )

    graph_manifest = Path(rooted_graph_artifact_root).resolve() / "manifest.json"
    graph_physical = _binding(graph_manifest)
    graph_binding = graphs.manifest.get("binding")
    if not isinstance(graph_binding, Mapping):
        raise RelationalPreStatusFieldReportError("graph manifest has no binding")
    if graph_binding.get("rooted_star_manifest_file_sha256") != expected_bank_sha:
        raise RelationalPreStatusFieldReportError(
            "graph manifest is not bound to the requested rooted-star bank"
        )
    outcome = _binding(
        outcome_report_path,
        expected_sha256=supervision.outcome_report_file_sha256,
    )
    roster = _binding(roster_path, expected_sha256=supervision.roster_file_sha256)
    inventory = graphs.evaluation_inventory()
    risk = evaluate_pre_status_risk_fields(supervision, inventory)
    honestward = evaluate_pre_status_honestward_fields(
        supervision,
        inventory,
        graph_variant="joint",
    )
    source_files = _source_files(extra_source_paths)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "success",
        "argv": [str(value) for value in argv],
        "scope": {
            "evidence_label": "offline source-model candidate evidence",
            "causal_controller_claim": False,
            "universal_controller_claim": False,
            "statement": (
                "These held-out source-model field scores are offline candidate "
                "evidence, not evidence of a causal or universal controller."
            ),
            "prediction_order": (
                "Every held-out risk event and every held-out quotient root receives "
                "a fold-safe prediction before outcome-conditioned scoring; the "
                "prediction inventory is retained with content hashes."
            ),
        },
        "inputs": {
            "rooted_star_manifest": bank,
            "rooted_graph_manifest": {
                **graph_physical,
                "content_sha256": _sha(
                    graphs.manifest.get("manifest_sha256"),
                    "graph manifest content SHA-256",
                ),
                "binding": dict(graph_binding),
            },
            "outcome_report": outcome,
            "frozen_orbit_roster": roster,
        },
        "source_files": source_files,
        "provenance": git_provenance(
            [Path(value["path"]) for value in source_files.values()]
            + [bank_manifest, graph_manifest, outcome_report_path, roster_path]
        ),
        "inventory": {
            "rooted_star_reference_count": len(index.references),
            "geometry_reference_count": len(index.geometry_references),
            "quotient_node_count": len(supervision.nodes),
            "outcome_event_count": len(supervision.outcomes_by_event_id),
            "honestward_crossing_count_by_view": _count_crossings(supervision),
            "edge_outcome_transition_counts": dict(
                supervision.edge_outcome_transition_counts
            ),
        },
        "evaluation": {
            "label_free_rooted_graphs": summarize_relational_pre_status_rooted_graphs(
                graphs
            ),
            "risk_fields": risk,
            "honestward_fields": honestward,
            "honestward_graph_variant": "joint_frozen_primary",
        },
    }
    report["report_sha256"] = _self_hash(report)
    validate_relational_pre_status_field_report(report)
    return report


def _number(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_relational_pre_status_field_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise companion Markdown summary for a validated report."""
    validate_relational_pre_status_field_report(report)
    lines = [
        "# Pre-status field evaluation",
        "",
        "**Scope:** Offline source-model candidate evidence only; this is neither "
        "causal evidence nor a universal-controller claim.",
        "",
        f"Report SHA-256: `{report['report_sha256']}`",
        "",
        "## Label-free rooted geometry",
        "",
        "| View | Fold | Candidate recall min | Joint/residual Jaccard | Joint/attention Jaccard |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    geometry_views = report["evaluation"]["label_free_rooted_graphs"]
    for view, folds in sorted(geometry_views.items()):
        for fold, summary in sorted(folds.items()):
            recall = summary["candidate_recall_at_64"]["minimum"]
            overlap = summary["query_neighborhood_jaccard"]
            lines.append(
                f"| `{view}` | `{fold}` | {_number(recall)} | "
                f"{_number(overlap['joint_vs_residual'])} | "
                f"{_number(overlap['joint_vs_attention'])} |"
            )
    lines.extend([
        "",
        "## Risk fields",
        "",
        "| View | Variant | Events | Local log loss | H/D AUROC | Gain over nuisance |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    risk_views = report["evaluation"]["risk_fields"]["views"]
    for view, variants in sorted(risk_views.items()):
        for variant, summary in sorted(variants.items()):
            gain = summary["local_log_loss_gain_over_nuisance"]["mean"]
            lines.append(
                f"| `{view}` | `{variant}` | {summary['event_count']} | "
                f"{_number(summary['local']['mean_log_loss'])} | "
                f"{_number(summary['local']['honest_deceptive_auroc'])} | "
                f"{_number(gain)} |"
            )
    lines.extend([
        "",
        "## Honestward field",
        "",
        "| View | Roots | Crossings | Local cosine | Global cosine | Shuffled cosine | L−G cosine | L−S cosine | Defined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    lift_views = report["evaluation"]["honestward_fields"]["views"]
    for view, summary in sorted(lift_views.items()):
        local = summary["models"]["local_calibrated"]
        global_mean = summary["models"]["global_mean"]
        shuffled = summary["models"]["shuffled"]
        comparisons = summary["local_calibrated_comparisons"]
        lines.append(
            f"| `{view}` | {summary['deceptive_root_count']} | "
            f"{summary['crossing_count']} | "
            f"{_number(local['cosine']['mean'])} | "
            f"{_number(global_mean['cosine']['mean'])} | "
            f"{_number(shuffled['cosine']['mean'])} | "
            f"{_number(comparisons['global_mean']['mean_cosine_difference'])} | "
            f"{_number(comparisons['shuffled']['mean_cosine_difference'])} | "
            f"{_number(local['defined_rate'])} |"
        )
    lines.extend([
        "",
        "### Reuse",
        "",
        "| View | Leave-contrast-out cosine | Opposite-truth-only cosine |",
        "| --- | ---: | ---: |",
    ])
    for view, summary in sorted(lift_views.items()):
        models = summary["models"]
        lines.append(
            f"| `{view}` | "
            f"{_number(models['leave_contrast_out']['cosine']['mean'])} | "
            f"{_number(models['opposite_truth_only']['cosine']['mean'])} |"
        )
    lines.extend([
        "",
        f"Honestward graph variant: `{report['evaluation']['honestward_graph_variant']}`.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "REPORT_KIND",
    "SCHEMA_VERSION",
    "RelationalPreStatusFieldReportError",
    "build_relational_pre_status_field_report",
    "render_relational_pre_status_field_markdown",
    "summarize_relational_pre_status_rooted_graphs",
    "validate_relational_pre_status_field_report",
]
