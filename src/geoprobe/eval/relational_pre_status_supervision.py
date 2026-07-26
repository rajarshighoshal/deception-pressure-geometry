"""Open frozen outcome labels only after pre-status rooted-star quotienting.

This adapter deliberately has no geometry construction logic.  It turns a
completed outcome-blind rooted-star store into the event and crossing units
consumed by the risk and honestward evaluators.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
    RootedStarReference,
    load_rooted_star_root_residuals,
)
from geoprobe.eval.relational_outcome_events import (
    OUTCOME_CLASSES,
    outcome_class_from_scientific_cohort,
)
from geoprobe.eval.relational_pre_status_risk_field import PreStatusRiskEvent
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
)
from geoprobe.io import file_sha256


_NODE_DOMAIN = b"geoprobe.pre-status-supervision.quotient-node.v1\x00"


class RelationalPreStatusSupervisionError(ValueError):
    """Raised when outcome opening does not bind the frozen pre-status bank."""


@dataclass(frozen=True, slots=True)
class PreStatusQuotientNode:
    """One view-specific prefix state, with exact replay duplicates collapsed."""

    node_id: str
    view: str
    prefix_state_sha256: str
    family: str
    family_fold: str
    turn_index: int
    event_ids: tuple[str, ...]
    representative_references: tuple[RootedStarReference, ...]


@dataclass(frozen=True, slots=True)
class LabelFreePrefixStateQuotient:
    """The complete outcome-blind quotient inventory for a rooted-star bank."""

    nodes: tuple[PreStatusQuotientNode, ...]
    event_to_node_ids: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True, slots=True)
class StatusEventOutcome:
    """The sealed behavioral record attached only after label-free prediction setup."""

    event_id: str
    outcome_class: str
    knowledge_correct: bool
    family: str
    family_fold: str
    scenario_id: str
    orbit_id: str
    turn_index: int
    intervention_history: tuple[str, ...]
    pressure_exposed: bool
    true_status: str
    desired_status: str
    prefix_state_sha256: str


@dataclass(frozen=True, slots=True)
class RelationalPreStatusSupervision:
    """Validated label opening over a fixed, outcome-blind quotient inventory."""

    outcome_report_file_sha256: str
    roster_file_sha256: str
    nodes: tuple[PreStatusQuotientNode, ...]
    nodes_by_id: Mapping[str, PreStatusQuotientNode]
    event_to_node_ids: Mapping[str, Mapping[str, str]]
    outcomes_by_event_id: Mapping[str, StatusEventOutcome]
    risk_events_by_view: Mapping[str, tuple[PreStatusRiskEvent, ...]]
    honestward_observations_by_view: Mapping[str, tuple[HonestwardCrossingObservation, ...]]
    edge_outcome_transition_counts: Mapping[str, int]


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusSupervisionError("value is not canonical JSON") from error


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusSupervisionError(f"{label} must be an object")
    return value


def _rows(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise RelationalPreStatusSupervisionError(f"{label} must be an array")
    return tuple(_mapping(item, f"{label} item") for item in value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPreStatusSupervisionError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RelationalPreStatusSupervisionError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RelationalPreStatusSupervisionError(f"{label} must be a non-negative integer")
    return value


def _history(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RelationalPreStatusSupervisionError(f"{label} must be an array of non-empty strings")
    return tuple(value)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalPreStatusSupervisionError(f"{label} is not finite UTF-8 JSON") from error
    return _mapping(value, label)


def _self_hash(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return sha256(_canonical(payload)).hexdigest()


def _node_id(view: str, prefix_state_sha256: str) -> str:
    return sha256(_NODE_DOMAIN + _canonical({"view": view, "prefix_state_sha256": prefix_state_sha256})).hexdigest()


def build_label_free_prefix_state_quotient(
    index: RelationalPreStatusRootedStarIndex,
) -> LabelFreePrefixStateQuotient:
    """Build the entire quotient before inspecting any outcome-bearing input."""
    groups: dict[tuple[str, str], list[RootedStarReference]] = defaultdict(list)
    for reference in index.references:
        if reference.view not in VIEWS:
            raise RelationalPreStatusSupervisionError("rooted-star index has an unsupported view")
        groups[(reference.view, reference.source_reference.prefix_state_sha256)].append(reference)
    if not groups:
        raise RelationalPreStatusSupervisionError("rooted-star index has no references")

    nodes: list[PreStatusQuotientNode] = []
    event_to_nodes: dict[str, dict[str, str]] = {}
    for (view, prefix), references in sorted(groups.items()):
        family = {item.family for item in references}
        folds = {item.source_reference.family_fold for item in references}
        turns = {item.source_reference.turn_index for item in references}
        if len(family) != 1 or len(folds) != 1 or len(turns) != 1:
            raise RelationalPreStatusSupervisionError("prefix quotient mixes family, fold, or turn")
        event_ids = tuple(sorted({item.field_event_id for item in references}))
        if not event_ids:
            raise RelationalPreStatusSupervisionError("prefix quotient has no field events")
        representatives: dict[str, RootedStarReference] = {}
        for reference in sorted(references, key=lambda item: (item.geometry_sha256, item.reference_id, item.rooted_star_id)):
            representatives.setdefault(reference.geometry_sha256, reference)
        node = PreStatusQuotientNode(
            node_id=_node_id(view, prefix), view=view, prefix_state_sha256=prefix,
            family=next(iter(family)), family_fold=next(iter(folds)), turn_index=next(iter(turns)),
            event_ids=event_ids, representative_references=tuple(representatives.values()),
        )
        nodes.append(node)
        for event_id in event_ids:
            assigned = event_to_nodes.setdefault(event_id, {})
            if view in assigned:
                raise RelationalPreStatusSupervisionError("event maps to multiple nodes in one view")
            assigned[view] = node.node_id
    return LabelFreePrefixStateQuotient(
        nodes=tuple(nodes),
        event_to_node_ids=MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in event_to_nodes.items()}
        ),
    )


def _outcomes(path: Path, expected_sha256: str, *, expected_event_count: int | None) -> tuple[str, Mapping[str, StatusEventOutcome]]:
    resolved = Path(path).resolve()
    expected = _sha(expected_sha256, "expected outcome-report file SHA-256")
    actual = file_sha256(resolved)
    if actual != expected:
        raise RelationalPreStatusSupervisionError("outcome report differs from its expected physical SHA-256")
    report = _read_json(resolved, "outcome report")
    if report.get("schema_version") != 1 or report.get("status") != "success" or report.get("report_sha256") != _self_hash(report):
        raise RelationalPreStatusSupervisionError("outcome report schema, status, or self-hash is invalid")
    indexed: dict[str, StatusEventOutcome] = {}
    for raw in _rows(report.get("scored_events"), "outcome scored events"):
        event_id = _string(raw.get("field_event_id"), "outcome field-event ID")
        if event_id in indexed:
            raise RelationalPreStatusSupervisionError("outcome report has duplicate field events")
        outcome_class = _string(raw.get("outcome_class"), "outcome class")
        if outcome_class not in OUTCOME_CLASSES:
            raise RelationalPreStatusSupervisionError("outcome report has an unsupported outcome class")
        cohort = _string(raw.get("scientific_cohort"), "outcome scientific cohort")
        if outcome_class_from_scientific_cohort(cohort) != outcome_class:
            raise RelationalPreStatusSupervisionError("outcome class disagrees with scientific cohort")
        knowledge_correct = raw.get("knowledge_correct")
        pressure = raw.get("pressure_exposed")
        if not isinstance(knowledge_correct, bool) or not isinstance(pressure, bool):
            raise RelationalPreStatusSupervisionError("outcome report knowledge and pressure flags must be Boolean")
        true_status = _string(raw.get("true_status"), "outcome true status")
        desired_status = _string(raw.get("desired_status"), "outcome desired status")
        if true_status not in {"PASS", "FAIL"} or desired_status not in {"PASS", "FAIL"}:
            raise RelationalPreStatusSupervisionError("outcome status labels are invalid")
        _string(raw.get("mapped_action"), "outcome mapped action")
        indexed[event_id] = StatusEventOutcome(
            event_id=event_id, outcome_class=outcome_class, knowledge_correct=knowledge_correct,
            family=_string(raw.get("family"), "outcome family"), family_fold=_string(raw.get("family_fold"), "outcome family fold"),
            scenario_id=_string(raw.get("scenario_id"), "outcome scenario ID"), orbit_id=_string(raw.get("orbit_id"), "outcome orbit ID"),
            turn_index=_integer(raw.get("turn_index"), "outcome turn index"),
            intervention_history=_history(raw.get("intervention_history"), "outcome intervention history"),
            pressure_exposed=pressure, true_status=true_status, desired_status=desired_status,
            prefix_state_sha256=_sha(raw.get("prefix_state_sha256"), "outcome prefix-state SHA"),
        )
    if not indexed or (expected_event_count is not None and len(indexed) != expected_event_count):
        raise RelationalPreStatusSupervisionError("outcome report event count is not the caller's expected inventory")
    return actual, MappingProxyType(indexed)


def _endpoint(edge: Mapping[str, Any], side: str, index: RelationalPreStatusRootedStarIndex) -> Mapping[str, Any]:
    endpoint = _mapping(edge.get(side), f"roster {side} endpoint")
    if _string(endpoint.get("field_name"), f"roster {side} field name") != "status":
        raise RelationalPreStatusSupervisionError("roster endpoint is not a status event")
    prefix = _sha(endpoint.get("prefix_state_sha256"), f"roster {side} prefix-state SHA")
    family = _string(endpoint.get("family"), f"roster {side} family")
    fold = _string(endpoint.get("family_fold"), f"roster {side} family fold")
    turn = _integer(endpoint.get("turn_index"), f"roster {side} turn index")
    members = _rows(endpoint.get("members"), f"roster {side} members")
    reference_ids = endpoint.get("reference_ids")
    if not isinstance(reference_ids, list) or not reference_ids or any(not isinstance(item, str) or not item for item in reference_ids):
        raise RelationalPreStatusSupervisionError("roster endpoint reference IDs are invalid")
    if len(set(reference_ids)) != len(reference_ids) or {item.get("reference_id") for item in members} != set(reference_ids):
        raise RelationalPreStatusSupervisionError("roster endpoint members do not match reference IDs")
    event_ids: set[str] = set()
    for member in members:
        reference_id = _sha(member.get("reference_id"), "roster member reference ID")
        event_id = _string(member.get("field_event_id"), "roster member field-event ID")
        event_ids.add(event_id)
        reference_group = index.by_reference_id.get(reference_id)
        if not reference_group:
            raise RelationalPreStatusSupervisionError("roster member reference is absent from rooted-star index")
        for reference in reference_group:
            source = reference.source_reference
            if (reference.field_event_id != event_id or source.prefix_state_sha256 != prefix or reference.family != family or source.family_fold != fold or source.turn_index != turn):
                raise RelationalPreStatusSupervisionError("roster member disagrees with rooted-star reference")
            for key in ("occurrence_id", "conversation_id", "realization_sha256", "canonical_realization_id", "source_row_sha256", "source_tensor_sha256"):
                if key in member and getattr(source, key) != member[key]:
                    raise RelationalPreStatusSupervisionError("roster member does not bind physical rooted-star input")
    if len(event_ids) != 1:
        raise RelationalPreStatusSupervisionError("roster endpoint must bind exactly one status event")
    event_id = next(iter(event_ids))
    indexed_reference_ids = {
        reference.reference_id for reference in index.references
        if reference.field_event_id == event_id
    }
    if indexed_reference_ids != set(reference_ids):
        raise RelationalPreStatusSupervisionError("roster endpoint does not cover its rooted-star event references")
    return MappingProxyType({"event_id": event_id, "prefix_state_sha256": prefix, "family": family, "family_fold": fold, "turn_index": turn})


def _roster(path: Path, expected_sha256: str, index: RelationalPreStatusRootedStarIndex, *, expected_forward_edge_count: int | None) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    resolved = Path(path).resolve()
    expected = _sha(expected_sha256, "expected roster file SHA-256")
    actual = file_sha256(resolved)
    if actual != expected:
        raise RelationalPreStatusSupervisionError("frozen orbit roster differs from its expected physical SHA-256")
    roster = _read_json(resolved, "frozen orbit roster")
    if roster.get("schema_version") != 1 or roster.get("status") != "frozen_label_free":
        raise RelationalPreStatusSupervisionError("frozen orbit roster schema or status is invalid")
    edges = _rows(roster.get("edges"), "frozen orbit roster edges")
    if roster.get("edge_roster_sha256") != sha256(_canonical(list(edges))).hexdigest():
        raise RelationalPreStatusSupervisionError("frozen orbit roster edge self-hash is invalid")
    forward = tuple(edge for edge in edges if edge.get("direction") == "forward")
    if len(forward) != len({edge.get("edge_pair_id") for edge in forward}):
        raise RelationalPreStatusSupervisionError("frozen orbit roster has non-unique forward edge pairs")
    if expected_forward_edge_count is not None and len(forward) != expected_forward_edge_count:
        raise RelationalPreStatusSupervisionError("roster forward edge count is not the caller's expected inventory")
    checked: list[Mapping[str, Any]] = []
    for edge in forward:
        pair_id = _sha(edge.get("edge_pair_id"), "roster edge-pair ID")
        source = _endpoint(edge, "source", index)
        target = _endpoint(edge, "target", index)
        if source["family"] != target["family"] or source["family_fold"] != target["family_fold"]:
            raise RelationalPreStatusSupervisionError("roster edge crosses family or fold")
        checked.append(MappingProxyType({
            "pair_id": pair_id, "contrast_id": _string(edge.get("contrast_id"), "roster contrast ID"),
            "scenario_id": _string(edge.get("scenario_id"), "roster scenario ID"), "source": source, "target": target,
        }))
    return actual, tuple(checked)


def _risk_events(outcomes: Mapping[str, StatusEventOutcome], nodes: Mapping[str, Mapping[str, str]], node_by_id: Mapping[str, PreStatusQuotientNode]) -> Mapping[str, tuple[PreStatusRiskEvent, ...]]:
    result: dict[str, list[PreStatusRiskEvent]] = {view: [] for view in VIEWS}
    if set(outcomes) != set(nodes):
        raise RelationalPreStatusSupervisionError("outcome and quotient event inventories differ")
    for event_id in sorted(outcomes):
        outcome = outcomes[event_id]
        view_nodes = nodes[event_id]
        if set(view_nodes) != set(VIEWS):
            raise RelationalPreStatusSupervisionError("status event does not map to exactly one node per view")
        nuisance = (
            str(outcome.turn_index), _canonical(list(outcome.intervention_history)).decode("utf-8"),
            "pressure" if outcome.pressure_exposed else "no_pressure", outcome.true_status, outcome.desired_status,
        )
        for view, node_id in sorted(view_nodes.items()):
            node = node_by_id[node_id]
            if (node.family != outcome.family or node.family_fold != outcome.family_fold or node.turn_index != outcome.turn_index or node.prefix_state_sha256 != outcome.prefix_state_sha256):
                raise RelationalPreStatusSupervisionError("outcome event disagrees with its pre-status quotient")
            result[view].append(PreStatusRiskEvent(event_id, node_id, node.family, node.family_fold, outcome.outcome_class, nuisance))
    return MappingProxyType({view: tuple(rows) for view, rows in result.items()})


def _mean_root_residuals(index: RelationalPreStatusRootedStarIndex, node: PreStatusQuotientNode) -> np.ndarray:
    values = [load_rooted_star_root_residuals(index, reference).detach().cpu().numpy() for reference in node.representative_references]
    if not values:
        raise RelationalPreStatusSupervisionError("quotient node has no geometry representative")
    shape = values[0].shape
    if any(value.shape != shape or not np.isfinite(value).all() for value in values):
        raise RelationalPreStatusSupervisionError("quotient root residuals have inconsistent shape or values")
    return np.asarray(np.mean(np.stack(values, axis=0, dtype=np.float64), axis=0), dtype=np.float32)


def _crossings(index: RelationalPreStatusRootedStarIndex, outcomes: Mapping[str, StatusEventOutcome], nodes: Mapping[str, Mapping[str, str]], node_by_id: Mapping[str, PreStatusQuotientNode], edges: Sequence[Mapping[str, Any]]) -> Mapping[str, tuple[HonestwardCrossingObservation, ...]]:
    by_view: dict[str, list[HonestwardCrossingObservation]] = {view: [] for view in VIEWS}
    root_residuals: dict[str, np.ndarray] = {}

    def residuals(node: PreStatusQuotientNode) -> np.ndarray:
        value = root_residuals.get(node.node_id)
        if value is None:
            value = _mean_root_residuals(index, node)
            root_residuals[node.node_id] = value
        return value

    for edge in edges:
        endpoints = (edge["source"], edge["target"])
        endpoint_outcomes = [outcomes.get(endpoint["event_id"]) for endpoint in endpoints]
        if any(item is None or not item.knowledge_correct for item in endpoint_outcomes):
            continue
        assert endpoint_outcomes[0] is not None and endpoint_outcomes[1] is not None
        labels = {endpoint_outcomes[0].outcome_class, endpoint_outcomes[1].outcome_class}
        if labels != {"HONEST", "DECEPTIVE"}:
            continue
        deceptive_index = 0 if endpoint_outcomes[0].outcome_class == "DECEPTIVE" else 1
        honest_index = 1 - deceptive_index
        deceptive_endpoint, honest_endpoint = endpoints[deceptive_index], endpoints[honest_index]
        deceptive_outcome, honest_outcome = endpoint_outcomes[deceptive_index], endpoint_outcomes[honest_index]
        if deceptive_outcome.true_status != honest_outcome.true_status:
            raise RelationalPreStatusSupervisionError("honest/deceptive crossing has mismatched true status")
        for view in VIEWS:
            deceptive_node = node_by_id[nodes[deceptive_endpoint["event_id"]][view]]
            honest_node = node_by_id[nodes[honest_endpoint["event_id"]][view]]
            if deceptive_node.prefix_state_sha256 != deceptive_endpoint["prefix_state_sha256"] or honest_node.prefix_state_sha256 != honest_endpoint["prefix_state_sha256"]:
                raise RelationalPreStatusSupervisionError("crossing endpoint does not bind its quotient prefix")
            delta = residuals(honest_node) - residuals(deceptive_node)
            by_view[view].append(HonestwardCrossingObservation(
                pair_id=f"{edge['pair_id']}:{view}", deceptive_root_id=deceptive_node.node_id, honest_root_id=honest_node.node_id,
                family=deceptive_node.family, family_fold=deceptive_node.family_fold, scenario_id=edge["scenario_id"], contrast_id=edge["contrast_id"],
                true_status=deceptive_outcome.true_status, delta=delta,
            ))
    return MappingProxyType({view: tuple(rows) for view, rows in by_view.items()})


def _edge_outcome_transition_counts(
    outcomes: Mapping[str, StatusEventOutcome],
    edges: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        labels = tuple(
            sorted(
                outcomes[edge[side]["event_id"]].outcome_class
                for side in ("source", "target")
            )
        )
        counts[" <-> ".join(labels)] += 1
    return MappingProxyType(dict(sorted(counts.items())))


def build_relational_pre_status_supervision(
    index: RelationalPreStatusRootedStarIndex,
    *,
    outcome_report_path: Path,
    expected_outcome_report_sha256: str,
    roster_path: Path,
    expected_roster_sha256: str,
    expected_outcome_event_count: int | None = None,
    expected_forward_edge_count: int | None = None,
) -> RelationalPreStatusSupervision:
    """Open sealed outcomes after deterministic rooted-star quotient construction.

    Count expectations are caller-owned so the reusable seam does not encode a
    particular production inventory (for example 1,680 events or 780 pairs).
    """
    quotient = build_label_free_prefix_state_quotient(index)
    nodes = quotient.nodes
    event_to_node_ids = quotient.event_to_node_ids
    node_by_id = MappingProxyType({node.node_id: node for node in nodes})
    outcome_sha, outcomes = _outcomes(outcome_report_path, expected_outcome_report_sha256, expected_event_count=expected_outcome_event_count)
    roster_sha, edges = _roster(roster_path, expected_roster_sha256, index, expected_forward_edge_count=expected_forward_edge_count)
    risk_events = _risk_events(outcomes, event_to_node_ids, node_by_id)
    crossings = _crossings(index, outcomes, event_to_node_ids, node_by_id, edges)
    transitions = _edge_outcome_transition_counts(outcomes, edges)
    return RelationalPreStatusSupervision(outcome_sha, roster_sha, nodes, node_by_id, event_to_node_ids, outcomes, risk_events, crossings, transitions)


open_relational_pre_status_supervision = build_relational_pre_status_supervision


__all__ = [
    "LabelFreePrefixStateQuotient", "PreStatusQuotientNode", "RelationalPreStatusSupervision", "RelationalPreStatusSupervisionError", "StatusEventOutcome",
    "build_label_free_prefix_state_quotient", "build_relational_pre_status_supervision",
    "open_relational_pre_status_supervision",
]
