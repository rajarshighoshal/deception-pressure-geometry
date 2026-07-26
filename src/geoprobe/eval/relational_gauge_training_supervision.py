"""Structurally sealed fold-training labels for gauge-controller construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
)
from geoprobe.eval.relational_pre_status_outcome_shards import (
    FOLDS,
    load_relational_pre_status_outcome_shard,
)
from geoprobe.eval.relational_pre_status_risk_field import PreStatusRiskEvent
from geoprobe.eval.relational_pre_status_supervision import (
    PreStatusQuotientNode,
    build_label_free_prefix_state_quotient,
)
from geoprobe.io import file_sha256


class RelationalGaugeTrainingSupervisionError(ValueError):
    """Fold training labels or their immutable shard bindings are invalid."""


@dataclass(frozen=True, slots=True)
class FoldGaugeTrainingSupervision:
    """The outcome-bearing state visible while one family fold stays sealed."""

    held_out_family_fold: str
    nodes_by_id: Mapping[str, PreStatusQuotientNode]
    risk_events_by_view: Mapping[str, tuple[PreStatusRiskEvent, ...]]
    honestward_observations_by_view: Mapping[str, tuple[object, ...]]
    opened_training_shards: tuple[Mapping[str, Any], ...]
    outcome_shard_manifest_file_sha256: str


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelationalGaugeTrainingSupervisionError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalGaugeTrainingSupervisionError(f"{name} must be non-empty")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RelationalGaugeTrainingSupervisionError(
            f"{name} must be a non-negative integer"
        )
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalGaugeTrainingSupervisionError(f"{name} must be an object")
    return value


def _read_bound_manifest(root: Path, expected_sha256: str) -> Mapping[str, Any]:
    path = root / "manifest.json"
    expected = _sha(expected_sha256, "outcome-shard manifest SHA-256")
    if not path.is_file() or file_sha256(path) != expected:
        raise RelationalGaugeTrainingSupervisionError(
            "outcome-shard manifest differs from its expected physical SHA-256"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalGaugeTrainingSupervisionError(
            "outcome-shard manifest is not finite UTF-8 JSON"
        ) from error
    manifest = _mapping(value, "outcome-shard manifest")
    folds = manifest.get("folds")
    shards = _mapping(manifest.get("shards"), "outcome-shard inventory")
    if folds != list(FOLDS) or set(shards) != set(FOLDS):
        raise RelationalGaugeTrainingSupervisionError(
            "outcome-shard manifest fold inventory is incomplete"
        )
    return manifest


def _nuisance_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    history = row.get("intervention_history")
    if not isinstance(history, list) or any(
        not isinstance(value, str) or not value for value in history
    ):
        raise RelationalGaugeTrainingSupervisionError(
            "training intervention history is invalid"
        )
    pressure = row.get("pressure_exposed")
    if not isinstance(pressure, bool):
        raise RelationalGaugeTrainingSupervisionError(
            "training pressure flag is invalid"
        )
    return (
        str(_integer(row.get("turn_index"), "training turn index")),
        json.dumps(history, separators=(",", ":")),
        "pressure" if pressure else "no_pressure",
        _string(row.get("true_status"), "training true status"),
        _string(row.get("desired_status"), "training desired status"),
    )


def build_fold_gauge_training_supervision(
    index: RelationalPreStatusRootedStarIndex,
    *,
    held_out_family_fold: str,
    outcome_shard_root: Path,
    expected_outcome_shard_manifest_file_sha256: str,
    expected_source_report_file_sha256: str,
) -> FoldGaugeTrainingSupervision:
    """Open exactly four legal training shards and never touch the held-out shard."""
    fold = _string(held_out_family_fold, "held-out family fold")
    if fold not in FOLDS:
        raise RelationalGaugeTrainingSupervisionError("held-out fold is unsupported")
    root = Path(outcome_shard_root).resolve()
    manifest_sha = _sha(
        expected_outcome_shard_manifest_file_sha256,
        "outcome-shard manifest SHA-256",
    )
    source_sha = _sha(
        expected_source_report_file_sha256,
        "source outcome-report SHA-256",
    )
    manifest = _read_bound_manifest(root, manifest_sha)
    source = _mapping(manifest.get("source_report"), "source report binding")
    if source.get("file_sha256") != source_sha:
        raise RelationalGaugeTrainingSupervisionError(
            "outcome-shard manifest source report binding changed"
        )

    quotient = build_label_free_prefix_state_quotient(index)
    nodes_by_id = MappingProxyType({node.node_id: node for node in quotient.nodes})
    expected_events = {
        event_id
        for event_id, view_nodes in quotient.event_to_node_ids.items()
        if nodes_by_id[view_nodes[VIEWS[0]]].family_fold != fold
    }
    risk: dict[str, list[PreStatusRiskEvent]] = {view: [] for view in VIEWS}
    opened: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    shards = _mapping(manifest["shards"], "outcome-shard inventory")
    for training_fold in FOLDS:
        if training_fold == fold:
            continue
        binding = _mapping(shards[training_fold], f"{training_fold} shard binding")
        loaded = load_relational_pre_status_outcome_shard(
            root,
            training_fold,
            expected_manifest_file_sha256=manifest_sha,
            expected_shard_file_sha256=_sha(
                binding.get("file_sha256"), f"{training_fold} shard file SHA-256"
            ),
            expected_content_sha256=_sha(
                binding.get("content_sha256"),
                f"{training_fold} shard content SHA-256",
            ),
            expected_source_report_file_sha256=source_sha,
        )
        for row in loaded.scored_events:
            event_id = _string(row.get("field_event_id"), "training event ID")
            if event_id in seen:
                raise RelationalGaugeTrainingSupervisionError(
                    "training outcome shards overlap"
                )
            seen.add(event_id)
            try:
                view_nodes = quotient.event_to_node_ids[event_id]
            except KeyError as error:
                raise RelationalGaugeTrainingSupervisionError(
                    "training event is absent from the label-free quotient"
                ) from error
            nuisance = _nuisance_key(row)
            for view in VIEWS:
                node = nodes_by_id[view_nodes[view]]
                if (
                    node.family_fold != training_fold
                    or node.family != row.get("family")
                    or node.turn_index != row.get("turn_index")
                    or node.prefix_state_sha256 != row.get("prefix_state_sha256")
                ):
                    raise RelationalGaugeTrainingSupervisionError(
                        "training outcome disagrees with its quotient node"
                    )
                risk[view].append(
                    PreStatusRiskEvent(
                        event_id=event_id,
                        root_id=node.node_id,
                        family=node.family,
                        family_fold=node.family_fold,
                        outcome_class=_string(
                            row.get("outcome_class"), "training outcome class"
                        ),
                        nuisance_key=nuisance,
                    )
                )
        opened.append(
            MappingProxyType(
                {
                    "family_fold": training_fold,
                    "event_count": len(loaded.scored_events),
                    "file_sha256": loaded.shard_file_sha256,
                    "content_sha256": loaded.content_sha256,
                    "shard_sha256": loaded.shard_sha256,
                }
            )
        )
    if seen != expected_events or len(opened) != 4:
        raise RelationalGaugeTrainingSupervisionError(
            "four training shards do not exactly cover non-heldout quotient events"
        )
    return FoldGaugeTrainingSupervision(
        held_out_family_fold=fold,
        nodes_by_id=nodes_by_id,
        risk_events_by_view=MappingProxyType(
            {
                view: tuple(sorted(rows, key=lambda row: row.event_id))
                for view, rows in risk.items()
            }
        ),
        honestward_observations_by_view=MappingProxyType(
            {view: tuple() for view in VIEWS}
        ),
        opened_training_shards=tuple(opened),
        outcome_shard_manifest_file_sha256=manifest_sha,
    )


__all__ = [
    "FoldGaugeTrainingSupervision",
    "RelationalGaugeTrainingSupervisionError",
    "build_fold_gauge_training_supervision",
]
