"""Prepare exact source-bank rows for the capture-bound gauge controller."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from types import MappingProxyType
from typing import Any

from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
    RootedStarReference,
    load_rooted_star_observation_binding,
)
from geoprobe.eval.relational_gauge_bundle import FoldGaugeControllerBundle
from geoprobe.eval.relational_pre_status_supervision import (
    build_label_free_prefix_state_quotient,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FOLDS
from geoprobe.runtime.relational_gauge_replay_observation import (
    CaptureBoundPreStatusGaugeObservationBuilder,
    FrozenPreStatusGaugeReplayRow,
    RelationalGaugeReplayObservationError,
)


class RelationalGaugeSourceReplayError(ValueError):
    """Raised when frozen causal inputs cannot bind the source gauge artifact."""


def _canonical_history(value: Sequence[str]) -> str:
    return json.dumps(
        list(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _event_nuisance_key(
    index: RelationalPreStatusRootedStarIndex,
    event: Mapping[str, Any],
    *,
    view: str,
    prefix_state_sha256: str,
) -> tuple[str, ...]:
    event_id = str(event.get("event_id", ""))
    references = tuple(
        reference
        for reference in index.by_field_event_id.get(event_id, ())
        if reference.view == view
        and reference.source_reference.prefix_state_sha256 == prefix_state_sha256
    )
    if not references:
        raise RelationalGaugeSourceReplayError(
            f"event {event_id!r} has no rooted-star reference for its source prefix"
        )
    metadata = {
        (
            reference.source_reference.turn_index,
            tuple(reference.source_reference.intervention_history),
            reference.source_reference.pressure_exposed,
        )
        for reference in references
    }
    if len(metadata) != 1:
        raise RelationalGaugeSourceReplayError(
            "event rooted-star references disagree on nuisance metadata"
        )
    turn_index, history, pressure_exposed = next(iter(metadata))
    true_status = event.get("true_status")
    desired_status = event.get("desired_status")
    if (
        event.get("turn_index") != turn_index
        or not isinstance(pressure_exposed, bool)
        or true_status not in {"PASS", "FAIL"}
        or desired_status not in {"PASS", "FAIL"}
    ):
        raise RelationalGaugeSourceReplayError(
            "event and rooted-star nuisance metadata disagree"
        )
    return (
        str(turn_index),
        _canonical_history(history),
        "pressure" if pressure_exposed else "no_pressure",
        str(true_status),
        str(desired_status),
    )


def _root_references(
    node: object,
    *,
    prefix_token_ids_sha256: str,
) -> tuple[RootedStarReference, ...]:
    references = tuple(
        reference
        for reference in getattr(node, "representative_references", ())
        if reference.source_reference.prefix_token_ids_sha256
        == prefix_token_ids_sha256
    )
    if not references:
        raise RelationalGaugeSourceReplayError(
            "source root has no exact geometry representative for its replay tokens"
        )
    return references


def build_frozen_pre_status_gauge_replay_rows(
    index: RelationalPreStatusRootedStarIndex,
    works: Sequence[object],
    *,
    view: str = "intervention_masked_action_free",
) -> Mapping[str, tuple[FrozenPreStatusGaugeReplayRow, ...]]:
    """Join a validated causal replay plan to sealed quotient queries and tensors."""
    quotient = build_label_free_prefix_state_quotient(index)
    nodes = {node.node_id: node for node in quotient.nodes if node.view == view}
    rows_by_fold: dict[str, list[FrozenPreStatusGaugeReplayRow]] = defaultdict(list)
    seen_rows: set[str] = set()
    for work in works:
        root_id = str(getattr(work, "root_id", ""))
        fold = str(getattr(work, "family_fold", ""))
        prefix = tuple(int(value) for value in getattr(work, "prefix_token_ids", ()))
        prefix_sha = str(getattr(work, "prefix_token_ids_sha256", ""))
        events = tuple(getattr(work, "events", ()))
        try:
            node = nodes[root_id]
        except KeyError as error:
            raise RelationalGaugeSourceReplayError(
                f"replay root {root_id!r} is absent from the sealed {view} quotient"
            ) from error
        if (
            fold not in FOLDS
            or node.family_fold != fold
            or not prefix
            or not events
            or node.prefix_state_sha256
            not in {
                reference.source_reference.prefix_state_sha256
                for reference in node.representative_references
            }
        ):
            raise RelationalGaugeSourceReplayError(
                "replay work and sealed quotient node disagree"
            )
        root_references = _root_references(
            node, prefix_token_ids_sha256=prefix_sha
        )
        rooted_stars = tuple(
            load_rooted_star_observation_binding(index, reference)
            for reference in root_references
        )
        nuisance_keys = {
            _event_nuisance_key(
                index,
                event,
                view=view,
                prefix_state_sha256=node.prefix_state_sha256,
            )
            for event in events
        }
        if len(nuisance_keys) != 1:
            raise RelationalGaugeSourceReplayError(
                "events sharing one causal root disagree on field nuisance metadata"
            )
        nuisance_key = next(iter(nuisance_keys))
        if root_id in seen_rows:
            raise RelationalGaugeSourceReplayError(
                "replay root row IDs must be globally unique"
            )
        seen_rows.add(root_id)
        rows_by_fold[fold].append(
            FrozenPreStatusGaugeReplayRow(
                row_id=root_id,
                held_out_family_fold=fold,
                root_id=root_id,
                expected_token_ids=prefix,
                nuisance_key=nuisance_key,
                rooted_stars=rooted_stars,
            )
        )
    if not rows_by_fold:
        raise RelationalGaugeSourceReplayError("causal replay plan contains no works")
    return MappingProxyType(
        {
            fold: tuple(sorted(rows, key=lambda row: row.row_id))
            for fold, rows in sorted(rows_by_fold.items())
        }
    )


def build_capture_bound_source_replay_builders(
    index: RelationalPreStatusRootedStarIndex,
    works: Sequence[object],
    bundles: Mapping[str, FoldGaugeControllerBundle],
    *,
    controller_artifact_sha256: str,
    view: str = "intervention_masked_action_free",
    # Bind v2 (scale-free identity, 2026-07-23) — must stay in lockstep with
    # relational_gauge_replay_observation defaults and the runner's pins.
    residual_min_cosine: float = 0.98,
    attention_min_correlation: float = 0.98,
) -> Mapping[str, CaptureBoundPreStatusGaugeObservationBuilder]:
    """Construct one fold-specific production observation builder per input fold."""
    rows_by_fold = build_frozen_pre_status_gauge_replay_rows(
        index, works, view=view
    )
    if set(rows_by_fold) != set(bundles):
        raise RelationalGaugeSourceReplayError(
            "replay folds and loaded gauge bundles have different inventories"
        )
    try:
        built = {
            fold: CaptureBoundPreStatusGaugeObservationBuilder(
                bundle=bundles[fold],
                rows=rows,
                controller_artifact_sha256=controller_artifact_sha256,
                residual_min_cosine=residual_min_cosine,
                attention_min_correlation=attention_min_correlation,
            )
            for fold, rows in rows_by_fold.items()
        }
    except RelationalGaugeReplayObservationError as error:
        raise RelationalGaugeSourceReplayError(
            "source replay observation builder construction failed"
        ) from error
    return MappingProxyType(dict(sorted(built.items())))


__all__ = [
    "RelationalGaugeSourceReplayError",
    "build_capture_bound_source_replay_builders",
    "build_frozen_pre_status_gauge_replay_rows",
]
