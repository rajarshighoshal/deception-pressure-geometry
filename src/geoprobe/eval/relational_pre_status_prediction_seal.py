"""Structurally sealed pre-status risk and lift prediction preparation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from safetensors import safe_open

from geoprobe.data.relational_pre_status_rooted_star import LAYERS, VIEWS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
)
from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.eval.relational_pre_status_outcome_shards import (
    FOLDS,
    load_relational_pre_status_outcome_shard,
)
from geoprobe.eval.relational_pre_status_risk_diagnostics import (
    shuffle_train_fold_outcome_labels,
    shuffle_train_fold_root_identities,
)
from geoprobe.eval.relational_pre_status_risk_field import (
    FoldSafePreStatusRiskField,
    PreStatusRiskEvent,
)
from geoprobe.eval.relational_pre_status_supervision import (
    StatusEventOutcome,
    _crossings,
    _roster,
    build_label_free_prefix_state_quotient,
)
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
    SharedPreStatusHonestwardField,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    FoldExactRootedGraph,
)
from geoprobe.io import file_sha256


PREDICTION_LEDGER_KIND: Final = "relational_pre_status_fold_prediction_seal"
STORED_MODELS: Final[tuple[str, ...]] = (
    "local",
    "local_calibrated",
    "global_mean",
    "nearest",
    "shuffled",
    "sign_flipped",
    "random_local_span",
    "leave_contrast_out",
    "training_true_pass_only",
    "training_true_fail_only",
    "contrast_global_oracle",
)
LEDGER_FILE_NAME: Final = "ledger.json"
TENSOR_FILE_NAME: Final = "predictions.safetensors"
_TENSOR_KIND: Final = "relational_pre_status_lift_prediction_tensor"
_RISK_MODELS: Final[tuple[str, ...]] = (
    "local",
    "base",
    "label_shuffle",
    "root_identity_shuffle",
)
_GRAPH_VARIANTS: Final[tuple[str, ...]] = (
    "joint",
    "residual_only",
    "attention_only",
)
_SEALED_NUISANCE_KEY: Final[tuple[str, ...]] = (
    "SEALED_HELDOUT_NUISANCE_UNAVAILABLE_AT_PREPARE",
)
_ARRAY_DOMAIN: Final = b"geoprobe.pre-status-prediction-seal.array.v1\x00"


class RelationalPreStatusPredictionSealError(ValueError):
    """A prediction build or stored artifact violates the temporal seal."""


@dataclass(frozen=True, slots=True)
class FoldPredictionArrays:
    """Dense lift predictions ordered exactly like the ledger lift queries."""

    values: np.ndarray
    defined: np.ndarray


@dataclass(frozen=True, slots=True)
class FoldPredictionBuild:
    """One fold-safe prediction build before physical persistence."""

    held_out_family_fold: str
    arrays: FoldPredictionArrays
    fit_payload: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if not isinstance(name, str) or not name or array.dtype.hasobject:
        raise RelationalPreStatusPredictionSealError("array identity is invalid")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise RelationalPreStatusPredictionSealError("array contains non-finite values")
    digest = sha256(_ARRAY_DOMAIN)
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusPredictionSealError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPreStatusPredictionSealError(
            f"{label} must be a non-empty string"
        )
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RelationalPreStatusPredictionSealError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return text


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RelationalPreStatusPredictionSealError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalPreStatusPredictionSealError(
            f"{label} is not finite UTF-8 JSON"
        ) from error
    return _mapping(value, label)


def _outcome(row: Mapping[str, Any]) -> StatusEventOutcome:
    history = row.get("intervention_history")
    if not isinstance(history, list) or any(
        not isinstance(item, str) or not item for item in history
    ):
        raise RelationalPreStatusPredictionSealError(
            "training outcome intervention history is invalid"
        )
    knowledge = row.get("knowledge_correct")
    pressure = row.get("pressure_exposed")
    if not isinstance(knowledge, bool) or not isinstance(pressure, bool):
        raise RelationalPreStatusPredictionSealError(
            "training outcome Boolean fields are invalid"
        )
    return StatusEventOutcome(
        event_id=_string(row.get("field_event_id"), "training field-event ID"),
        outcome_class=_string(row.get("outcome_class"), "training outcome class"),
        knowledge_correct=knowledge,
        family=_string(row.get("family"), "training family"),
        family_fold=_string(row.get("family_fold"), "training family fold"),
        scenario_id=_string(row.get("scenario_id"), "training scenario ID"),
        orbit_id=_string(row.get("orbit_id"), "training orbit ID"),
        turn_index=_integer(row.get("turn_index"), "training turn index"),
        intervention_history=tuple(history),
        pressure_exposed=pressure,
        true_status=_string(row.get("true_status"), "training true status"),
        desired_status=_string(row.get("desired_status"), "training desired status"),
        prefix_state_sha256=_sha(
            row.get("prefix_state_sha256"), "training prefix-state SHA-256"
        ),
    )


def _outcome_bindings(artifact_bindings: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    manifest_sha = _sha(
        artifact_bindings.get("outcome_shard_manifest_file_sha256"),
        "outcome-shard manifest physical SHA-256",
    )
    shards = _mapping(
        artifact_bindings.get("outcome_shards"), "outcome-shard bindings"
    )
    if set(shards) != set(FOLDS):
        raise RelationalPreStatusPredictionSealError(
            "outcome-shard bindings must contain exactly five folds"
        )
    return manifest_sha, shards


def _load_training_outcomes(
    root: Path,
    *,
    held_out_family_fold: str,
    expected_source_report_sha256: str,
    artifact_bindings: Mapping[str, Any],
) -> tuple[Mapping[str, StatusEventOutcome], tuple[Mapping[str, Any], ...]]:
    manifest_sha, shard_bindings = _outcome_bindings(artifact_bindings)
    outcomes: dict[str, StatusEventOutcome] = {}
    opened: list[Mapping[str, Any]] = []
    for fold in FOLDS:
        if fold == held_out_family_fold:
            continue
        binding = _mapping(shard_bindings[fold], f"{fold} outcome-shard binding")
        loaded = load_relational_pre_status_outcome_shard(
            root,
            fold,
            expected_manifest_file_sha256=manifest_sha,
            expected_shard_file_sha256=_sha(
                binding.get("file_sha256"), f"{fold} shard physical SHA-256"
            ),
            expected_content_sha256=_sha(
                binding.get("content_sha256"), f"{fold} shard content SHA-256"
            ),
            expected_source_report_file_sha256=expected_source_report_sha256,
        )
        for raw in loaded.scored_events:
            row = _outcome(raw)
            if row.family_fold != fold or row.event_id in outcomes:
                raise RelationalPreStatusPredictionSealError(
                    "training outcome shards overlap or cross folds"
                )
            outcomes[row.event_id] = row
        opened.append(
            MappingProxyType(
                {
                    "family_fold": fold,
                    "event_count": len(loaded.scored_events),
                    "file_sha256": loaded.shard_file_sha256,
                    "content_sha256": loaded.content_sha256,
                    "shard_sha256": loaded.shard_sha256,
                }
            )
        )
    if len(opened) != 4 or not outcomes:
        raise RelationalPreStatusPredictionSealError(
            "prepare must open exactly four non-heldout outcome shards"
        )
    return MappingProxyType(outcomes), tuple(opened)


def _risk_training_events(
    outcomes: Mapping[str, StatusEventOutcome],
    event_to_nodes: Mapping[str, Mapping[str, str]],
    nodes_by_id: Mapping[str, Any],
) -> Mapping[str, tuple[PreStatusRiskEvent, ...]]:
    result: dict[str, list[PreStatusRiskEvent]] = {view: [] for view in VIEWS}
    if not set(outcomes).issubset(event_to_nodes):
        raise RelationalPreStatusPredictionSealError(
            "training outcomes are absent from the label-free quotient"
        )
    for event_id in sorted(outcomes):
        outcome = outcomes[event_id]
        nuisance = (
            str(outcome.turn_index),
            json.dumps(list(outcome.intervention_history), separators=(",", ":")),
            "pressure" if outcome.pressure_exposed else "no_pressure",
            outcome.true_status,
            outcome.desired_status,
        )
        view_nodes = event_to_nodes[event_id]
        if set(view_nodes) != set(VIEWS):
            raise RelationalPreStatusPredictionSealError(
                "training event does not map to both frozen views"
            )
        for view in VIEWS:
            node = nodes_by_id[view_nodes[view]]
            if (
                node.family != outcome.family
                or node.family_fold != outcome.family_fold
                or node.turn_index != outcome.turn_index
                or node.prefix_state_sha256 != outcome.prefix_state_sha256
            ):
                raise RelationalPreStatusPredictionSealError(
                    "training outcome disagrees with its quotient node"
                )
            result[view].append(
                PreStatusRiskEvent(
                    event_id=event_id,
                    root_id=node.node_id,
                    family=node.family,
                    family_fold=node.family_fold,
                    outcome_class=outcome.outcome_class,
                    nuisance_key=nuisance,
                )
            )
    return MappingProxyType(
        {view: tuple(sorted(rows, key=lambda row: row.event_id)) for view, rows in result.items()}
    )


def _graph(
    graphs: Mapping[str, Mapping[str, Mapping[str, FoldExactRootedGraph]]],
    view: str,
    variant: str,
    fold: str,
) -> FoldExactRootedGraph:
    try:
        graph = graphs[view][variant][fold]
    except KeyError as error:
        raise RelationalPreStatusPredictionSealError(
            "frozen graph inventory is incomplete"
        ) from error
    if graph.held_out_family_fold != fold:
        raise RelationalPreStatusPredictionSealError(
            "graph held-out fold binding is invalid"
        )
    return graph


def _probability_row(probabilities: Mapping[str, float]) -> list[float]:
    if set(probabilities) != set(OUTCOME_CLASSES):
        raise RelationalPreStatusPredictionSealError(
            "risk probability classes are incomplete"
        )
    values = np.asarray(
        [float(probabilities[label]) for label in OUTCOME_CLASSES], dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise RelationalPreStatusPredictionSealError(
            "risk probability vector is invalid"
        )
    return [float(value) for value in values]


def _build_risk_rows(
    *,
    fold: str,
    quotient: Any,
    nodes_by_id: Mapping[str, Any],
    training_by_view: Mapping[str, tuple[PreStatusRiskEvent, ...]],
    graphs: Mapping[str, Mapping[str, Mapping[str, FoldExactRootedGraph]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in VIEWS:
        training = training_by_view[view]
        field = FoldSafePreStatusRiskField.fit(
            training, held_out_family_fold=fold
        )
        heldout_events = sorted(
            (
                event_id,
                nodes_by_id[view_nodes[view]],
            )
            for event_id, view_nodes in quotient.event_to_node_ids.items()
            if nodes_by_id[view_nodes[view]].family_fold == fold
        )
        if not heldout_events:
            raise RelationalPreStatusPredictionSealError(
                "held-out risk inventory is empty"
            )
        for variant in _GRAPH_VARIANTS:
            graph = _graph(graphs, view, variant, fold)
            seed = 20260721 + FOLDS.index(fold)
            label_field = FoldSafePreStatusRiskField.fit(
                shuffle_train_fold_outcome_labels(
                    training, held_out_family_fold=fold, seed=seed
                ).events,
                held_out_family_fold=fold,
            )
            root_field = FoldSafePreStatusRiskField.fit(
                shuffle_train_fold_root_identities(
                    training,
                    graph,
                    held_out_family_fold=fold,
                    seed=seed + 100,
                ).events,
                held_out_family_fold=fold,
            )
            for event_id, node in heldout_events:
                edges = graph.query_edges.get(node.node_id)
                if edges is None:
                    raise RelationalPreStatusPredictionSealError(
                        "held-out risk node is absent from query graph"
                    )
                primary = field.predict(
                    event_id=event_id,
                    root_id=node.node_id,
                    nuisance_key=_SEALED_NUISANCE_KEY,
                    edges=edges,
                )
                label = label_field.predict(
                    event_id=event_id,
                    root_id=node.node_id,
                    nuisance_key=_SEALED_NUISANCE_KEY,
                    edges=edges,
                )
                root = root_field.predict(
                    event_id=event_id,
                    root_id=node.node_id,
                    nuisance_key=_SEALED_NUISANCE_KEY,
                    edges=edges,
                )
                rows.append(
                    {
                        "view": view,
                        "graph_variant": variant,
                        "event_id": event_id,
                        "root_id": node.node_id,
                        "family": node.family,
                        "family_fold": node.family_fold,
                        "turn_index": node.turn_index,
                        "prefix_state_sha256": node.prefix_state_sha256,
                        "nuisance_key": list(_SEALED_NUISANCE_KEY),
                        "nuisance_semantics": (
                            "sealed_sentinel_not_exact_heldout_design_prior"
                        ),
                        "probabilities": {
                            "local": _probability_row(primary.local_probabilities),
                            "base": _probability_row(primary.base_probabilities),
                            "label_shuffle": _probability_row(
                                label.local_probabilities
                            ),
                            "root_identity_shuffle": _probability_row(
                                root.local_probabilities
                            ),
                        },
                        "support": {
                            "local": {
                                "event_ids": list(primary.support_event_ids),
                                "root_ids": list(primary.support_root_ids),
                                "count": primary.support_count,
                            },
                            "base": {
                                "event_ids": [
                                    event.event_id for event in training
                                ],
                                "root_ids": sorted(
                                    {event.root_id for event in training}
                                ),
                                "count": len(training),
                            },
                            "label_shuffle": {
                                "event_ids": list(label.support_event_ids),
                                "root_ids": list(label.support_root_ids),
                                "count": label.support_count,
                            },
                            "root_identity_shuffle": {
                                "event_ids": list(root.support_event_ids),
                                "root_ids": list(root.support_root_ids),
                                "count": root.support_count,
                            },
                        },
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["view"],
            row["graph_variant"],
            row["event_id"],
        ),
    )


def _root_balanced(
    observations: Sequence[HonestwardCrossingObservation],
) -> tuple[HonestwardCrossingObservation, ...]:
    grouped: dict[str, list[HonestwardCrossingObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.deceptive_root_id].append(observation)
    result: list[HonestwardCrossingObservation] = []
    for root_id, rows in sorted(grouped.items()):
        identities = {
            (row.family, row.family_fold, row.scenario_id, row.true_status)
            for row in rows
        }
        if len(identities) != 1:
            raise RelationalPreStatusPredictionSealError(
                "one training root spans incompatible identities"
            )
        targets: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            targets[row.honest_root_id].append(row.delta)
        deltas: list[np.ndarray] = []
        for values in targets.values():
            reference = values[0]
            if any(
                value.shape != reference.shape
                or not np.allclose(value, reference, atol=1e-6, rtol=1e-6)
                for value in values[1:]
            ):
                raise RelationalPreStatusPredictionSealError(
                    "duplicate training target has inconsistent deltas"
                )
            deltas.append(reference)
        family, family_fold, scenario_id, true_status = next(iter(identities))
        contrast_ids = tuple(
            sorted({item for row in rows for item in row.contrast_ids})
        )
        result.append(
            HonestwardCrossingObservation(
                pair_id=f"root-mean:{root_id}",
                deceptive_root_id=root_id,
                honest_root_id=f"target-mean:{root_id}",
                family=family,
                family_fold=family_fold,
                scenario_id=scenario_id,
                contrast_id=(
                    contrast_ids[0]
                    if len(contrast_ids) == 1
                    else "MULTI_CONTRAST"
                ),
                true_status=true_status,
                delta=np.asarray(
                    np.mean(np.stack(deltas).astype(np.float64), axis=0),
                    dtype=np.float32,
                ),
                contrast_ids=contrast_ids,
                source_pair_ids=tuple(
                    sorted({item for row in rows for item in row.source_pair_ids})
                ),
                honest_root_ids=tuple(sorted(targets)),
            )
        )
    return tuple(result)


def _fit_optional(
    rows: Sequence[HonestwardCrossingObservation],
    *,
    fold: str,
    training_edges: Mapping[str, Sequence[object]],
) -> SharedPreStatusHonestwardField | None:
    balanced = _root_balanced(rows)
    if not balanced:
        return None
    return SharedPreStatusHonestwardField.fit(
        balanced,
        held_out_family_fold=fold,
        training_edges=training_edges,
    )


def _contrast_global(
    rows: Sequence[HonestwardCrossingObservation],
    contrast_ids: Sequence[str],
    shape: tuple[int, int],
) -> np.ndarray:
    selected_contrasts = frozenset(contrast_ids)
    if not selected_contrasts:
        raise RelationalPreStatusPredictionSealError(
            "contrast-global query requires at least one contrast"
        )
    selected = _root_balanced(
        tuple(
            row
            for row in rows
            if selected_contrasts.intersection(row.contrast_ids)
        )
    )
    if not selected:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(
        np.mean(np.stack([row.delta for row in selected]).astype(np.float64), axis=0),
        dtype=np.float32,
    )


def _nonempty_subsets(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(sorted(set(values)))
    if not ordered:
        raise RelationalPreStatusPredictionSealError(
            "endpoint contrast inventory is empty"
        )
    return tuple(
        subset
        for size in range(1, len(ordered) + 1)
        for subset in combinations(ordered, size)
    )


def _prediction_arrays(
    *,
    field: SharedPreStatusHonestwardField,
    leave_field: SharedPreStatusHonestwardField | None,
    pass_field: SharedPreStatusHonestwardField | None,
    fail_field: SharedPreStatusHonestwardField | None,
    query_root_id: str,
    edges: Sequence[object],
    contrast_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    primary = field.predict(query_root_id, edges)
    zero = np.zeros(field.shape, dtype=np.float32)

    def optional(
        fitted: SharedPreStatusHonestwardField | None,
    ) -> tuple[np.ndarray, bool, int, tuple[str, ...], tuple[str, ...]]:
        if fitted is None:
            return zero, False, 0, (), ()
        prediction = fitted.predict(query_root_id, edges)
        return (
            prediction.dose_calibrated_local,
            prediction.defined,
            prediction.support_count,
            prediction.support_root_ids,
            prediction.support_pair_ids,
        )

    leave = optional(leave_field)
    pass_only = optional(pass_field)
    fail_only = optional(fail_field)
    values = np.stack(
        (
            primary.local,
            primary.dose_calibrated_local,
            primary.global_mean,
            primary.nearest,
            primary.shuffled,
            primary.sign_flipped,
            primary.random_local_span,
            leave[0],
            pass_only[0],
            fail_only[0],
            contrast_global,
        )
    ).astype(np.float32, copy=False)
    defined = np.asarray(
        (
            primary.defined,
            primary.defined,
            True,
            primary.defined,
            primary.defined,
            primary.defined,
            primary.defined,
            leave[1],
            pass_only[1],
            fail_only[1],
            True,
        ),
        dtype=np.bool_,
    )
    support = {
        "ordinary": {
            "root_ids": list(primary.support_root_ids),
            "pair_ids": list(primary.support_pair_ids),
            "count": primary.support_count,
        },
        "leave_contrast_out": {
            "root_ids": list(leave[3]),
            "pair_ids": list(leave[4]),
            "count": leave[2],
        },
        "training_true_pass_only": {
            "root_ids": list(pass_only[3]),
            "pair_ids": list(pass_only[4]),
            "count": pass_only[2],
        },
        "training_true_fail_only": {
            "root_ids": list(fail_only[3]),
            "pair_ids": list(fail_only[4]),
            "count": fail_only[2],
        },
    }
    return values, defined, support


def _stable_artifact_bindings(
    artifact_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    excluded = {
        "outcome_shard_manifest_file_sha256",
        "outcome_shards",
        "source_report_file_sha256",
    }
    return MappingProxyType(
        {
            str(key): _jsonable(value)
            for key, value in sorted(artifact_bindings.items(), key=lambda item: str(item[0]))
            if key not in excluded
        }
    )


def build_pre_status_fold_predictions(
    index: RelationalPreStatusRootedStarIndex,
    graphs: Mapping[str, Mapping[str, Mapping[str, FoldExactRootedGraph]]],
    *,
    held_out_family_fold: str,
    outcome_shard_root: Path,
    expected_source_report_sha256: str,
    roster_path: Path,
    expected_roster_sha256: str,
    artifact_bindings: Mapping[str, Any],
    argv: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> FoldPredictionBuild:
    """Prepare one fold while touching only its four legal training shards/tensors."""
    fold = _string(held_out_family_fold, "held-out family fold")
    if fold not in FOLDS:
        raise RelationalPreStatusPredictionSealError(
            "held-out family fold is unsupported"
        )
    source_sha = _sha(
        expected_source_report_sha256, "expected source-report physical SHA-256"
    )
    quotient = build_label_free_prefix_state_quotient(index)
    nodes_by_id = MappingProxyType({node.node_id: node for node in quotient.nodes})
    outcomes, opened = _load_training_outcomes(
        Path(outcome_shard_root),
        held_out_family_fold=fold,
        expected_source_report_sha256=source_sha,
        artifact_bindings=artifact_bindings,
    )
    expected_training_events = {
        event_id
        for event_id, view_nodes in quotient.event_to_node_ids.items()
        if nodes_by_id[view_nodes[VIEWS[0]]].family_fold != fold
    }
    if set(outcomes) != expected_training_events:
        raise RelationalPreStatusPredictionSealError(
            "four training shards do not exactly cover non-heldout quotient events"
        )
    roster_sha, roster_edges = _roster(
        roster_path,
        expected_roster_sha256,
        index,
        expected_forward_edge_count=None,
    )
    training_edges = tuple(
        edge
        for edge in roster_edges
        if edge["source"]["family_fold"] != fold
    )
    heldout_edges = tuple(
        edge
        for edge in roster_edges
        if edge["source"]["family_fold"] == fold
    )
    if (
        not training_edges
        or not heldout_edges
        or any(edge["target"]["family_fold"] == fold for edge in training_edges)
        or any(edge["target"]["family_fold"] != fold for edge in heldout_edges)
    ):
        raise RelationalPreStatusPredictionSealError(
            "roster fold partition is invalid"
        )
    risk_training = _risk_training_events(
        outcomes, quotient.event_to_node_ids, nodes_by_id
    )
    risk_rows = _build_risk_rows(
        fold=fold,
        quotient=quotient,
        nodes_by_id=nodes_by_id,
        training_by_view=risk_training,
        graphs=graphs,
    )
    crossings_by_view = _crossings(
        index,
        outcomes,
        quotient.event_to_node_ids,
        nodes_by_id,
        training_edges,
    )

    lift_rows: list[dict[str, Any]] = []
    lift_values: list[np.ndarray] = []
    lift_defined: list[np.ndarray] = []
    for view in VIEWS:
        graph = _graph(graphs, view, "joint", fold)
        raw_training = tuple(crossings_by_view[view])
        field = _fit_optional(
            raw_training, fold=fold, training_edges=graph.training_edges
        )
        if field is None:
            raise RelationalPreStatusPredictionSealError(
                "training fold has no honestward crossings"
            )
        pass_field = _fit_optional(
            tuple(row for row in raw_training if row.true_status == "PASS"),
            fold=fold,
            training_edges=graph.training_edges,
        )
        fail_field = _fit_optional(
            tuple(row for row in raw_training if row.true_status == "FAIL"),
            fold=fold,
            training_edges=graph.training_edges,
        )
        endpoint_contrasts: dict[str, set[str]] = defaultdict(set)
        for edge in heldout_edges:
            for side in ("source", "target"):
                endpoint_contrasts[edge[side]["event_id"]].add(edge["contrast_id"])
        endpoint_contrast_subsets = {
            event_id: _nonempty_subsets(tuple(contrasts))
            for event_id, contrasts in endpoint_contrasts.items()
        }
        contrast_sets = {
            subset
            for subsets in endpoint_contrast_subsets.values()
            for subset in subsets
        }
        leave_fields = {
            contrasts: _fit_optional(
                tuple(
                    row
                    for row in raw_training
                    if not set(row.contrast_ids).intersection(contrasts)
                ),
                fold=fold,
                training_edges=graph.training_edges,
            )
            for contrasts in sorted(contrast_sets)
        }
        for edge in sorted(heldout_edges, key=lambda row: row["pair_id"]):
            for side in ("source", "target"):
                endpoint = edge[side]
                event_id = endpoint["event_id"]
                node_id = quotient.event_to_node_ids[event_id][view]
                node = nodes_by_id[node_id]
                if (
                    node.family_fold != fold
                    or node.prefix_state_sha256 != endpoint["prefix_state_sha256"]
                ):
                    raise RelationalPreStatusPredictionSealError(
                        "held-out roster endpoint disagrees with label-free quotient"
                    )
                query_edges = graph.query_edges.get(node_id)
                if query_edges is None:
                    raise RelationalPreStatusPredictionSealError(
                        "held-out lift endpoint is absent from query graph"
                    )
                contrast_id = edge["contrast_id"]
                for query_contrast_ids in endpoint_contrast_subsets[event_id]:
                    values, defined, support = _prediction_arrays(
                        field=field,
                        leave_field=leave_fields[query_contrast_ids],
                        pass_field=pass_field,
                        fail_field=fail_field,
                        query_root_id=node_id,
                        edges=query_edges,
                        contrast_global=_contrast_global(
                            raw_training, query_contrast_ids, field.shape
                        ),
                    )
                    row_index = len(lift_rows)
                    lift_values.append(values)
                    lift_defined.append(defined)
                    lift_rows.append(
                        {
                            "tensor_row_index": row_index,
                            "view": view,
                            "pair_id": edge["pair_id"],
                            "side": side,
                            "event_id": event_id,
                            "root_id": node_id,
                            "contrast_id": contrast_id,
                            "query_contrast_ids": list(query_contrast_ids),
                            "scenario_id": edge["scenario_id"],
                            "family": node.family,
                            "family_fold": node.family_fold,
                            "turn_index": node.turn_index,
                            "prefix_state_sha256": node.prefix_state_sha256,
                            "support": support,
                        }
                    )
    keys = {
        (
            row["view"],
            row["pair_id"],
            row["side"],
            tuple(row["query_contrast_ids"]),
        )
        for row in lift_rows
    }
    if len(keys) != len(lift_rows):
        raise RelationalPreStatusPredictionSealError(
            "held-out lift endpoints are not uniquely and completely represented"
        )
    values = np.ascontiguousarray(np.stack(lift_values), dtype=np.float32)
    defined = np.ascontiguousarray(np.stack(lift_defined), dtype=np.bool_)
    if (
        values.ndim != 4
        or values.shape[1] != len(STORED_MODELS)
        or values.shape[2] != len(LAYERS)
        or defined.shape != values.shape[:2]
        or not np.isfinite(values).all()
    ):
        raise RelationalPreStatusPredictionSealError(
            "built lift prediction arrays are invalid"
        )
    arrays = FoldPredictionArrays(values=values, defined=defined)
    fit_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PREDICTION_LEDGER_KIND,
        "held_out_family_fold": fold,
        "artifact_bindings": dict(_stable_artifact_bindings(artifact_bindings)),
        "roster_file_sha256": roster_sha,
        "prediction_contract": {
            "outcome_manifest_access": "opened_for_bindings",
            "training_outcome_shard_access": "exactly_four_nonheldout_family_folds",
            "heldout_outcome_shard_access": "not_opened",
            "full_outcome_report_access": "not_opened",
            "training_root_residual_access": "training_fold_nodes_only",
            "heldout_target_root_residual_access": "not_loaded",
            "heldout_lift_orientation": "not_inferred_at_prepare",
            "heldout_nuisance_prior": (
                "sealed_sentinel_only_not_exact_heldout_nuisance_prior"
            ),
        },
        "models": list(STORED_MODELS),
        "risk_models": list(_RISK_MODELS),
        "outcome_classes": list(OUTCOME_CLASSES),
        "layers": list(LAYERS),
        "views": list(VIEWS),
        "graph_variants": list(_GRAPH_VARIANTS),
        "argv": list(argv),
        "provenance": dict(provenance or {}),
        "opened_training_outcome_shards": [dict(row) for row in opened],
        "training_outcome_event_count": len(outcomes),
        "training_roster_pair_count": len(training_edges),
        "heldout_roster_pair_count": len(heldout_edges),
        "risk_prediction_count": len(risk_rows),
        "lift_query_count": len(lift_rows),
        "prediction_array_sha256_by_model": {
            model: _array_sha256(model, values[:, index])
            for index, model in enumerate(STORED_MODELS)
        },
        "defined_array_sha256": _array_sha256(
            "defined", defined.astype(np.uint8)
        ),
        "risk_predictions": risk_rows,
        "lift_queries": lift_rows,
    }
    fit_payload = _jsonable(fit_payload)
    fit_payload["fit_payload_sha256"] = _canonical_sha256(fit_payload)
    return FoldPredictionBuild(
        held_out_family_fold=fold,
        arrays=arrays,
        fit_payload=MappingProxyType(fit_payload),
    )


def _ledger_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("prediction_ledger_sha256", None)
    return _canonical_sha256(payload)


def _fit_payload_without_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("tensor_artifact", None)
    payload.pop("prediction_ledger_sha256", None)
    return payload


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(
                json.dumps(
                    _jsonable(value),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_deterministic_safetensors(
    path: Path,
    *,
    values: np.ndarray,
    defined: np.ndarray,
    metadata: Mapping[str, str],
) -> None:
    defined_bytes = np.ascontiguousarray(defined, dtype=np.uint8).tobytes(order="C")
    value_array = np.ascontiguousarray(values, dtype="<f4")
    value_bytes = value_array.tobytes(order="C")
    header = {
        "__metadata__": dict(sorted(metadata.items())),
        "defined": {
            "dtype": "U8",
            "shape": list(defined.shape),
            "data_offsets": [0, len(defined_bytes)],
        },
        "lift_predictions": {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [
                len(defined_bytes),
                len(defined_bytes) + len(value_bytes),
            ],
        },
    }
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("xb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(defined_bytes)
        handle.write(value_bytes)
        handle.flush()
        os.fsync(handle.fileno())


def save_pre_status_fold_prediction_artifact(
    build: FoldPredictionBuild,
    out_dir: Path,
) -> Mapping[str, Any]:
    """Persist one immutable tensor/ledger pair without replacing existing files."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / TENSOR_FILE_NAME
    ledger_path = root / LEDGER_FILE_NAME
    if tensor_path.exists() or ledger_path.exists():
        raise RelationalPreStatusPredictionSealError(
            "prediction artifact destination already contains output files"
        )
    metadata = {
        "schema_version": "1",
        "kind": _TENSOR_KIND,
        "held_out_family_fold": build.held_out_family_fold,
        "model_order": json.dumps(list(STORED_MODELS), separators=(",", ":")),
        "layers": json.dumps(list(LAYERS), separators=(",", ":")),
        "query_count": str(build.arrays.values.shape[0]),
    }
    with tempfile.NamedTemporaryFile(
        dir=str(root), suffix=".safetensors.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    try:
        _write_deterministic_safetensors(
            temporary,
            values=build.arrays.values,
            defined=build.arrays.defined,
            metadata=metadata,
        )
        os.link(temporary, tensor_path)
    finally:
        temporary.unlink(missing_ok=True)
    ledger = dict(build.fit_payload)
    ledger["tensor_artifact"] = {
        "path": TENSOR_FILE_NAME,
        "bytes": tensor_path.stat().st_size,
        "sha256": file_sha256(tensor_path),
        "prediction_shape": list(build.arrays.values.shape),
        "defined_shape": list(build.arrays.defined.shape),
    }
    ledger["prediction_ledger_sha256"] = _ledger_sha256(ledger)
    try:
        _write_json_new(ledger_path, ledger)
    except BaseException:
        tensor_path.unlink(missing_ok=True)
        raise
    return MappingProxyType(ledger)


def load_pre_status_fold_prediction_artifact(
    root: Path,
) -> tuple[Mapping[str, Any], FoldPredictionArrays]:
    """Validate a sealed ledger and every physical/semantic tensor binding."""
    artifact_root = Path(root).resolve()
    ledger = _read_json(
        artifact_root / LEDGER_FILE_NAME, "pre-status prediction ledger"
    )
    if (
        ledger.get("schema_version") != 1
        or ledger.get("kind") != PREDICTION_LEDGER_KIND
        or ledger.get("prediction_ledger_sha256") != _ledger_sha256(ledger)
    ):
        raise RelationalPreStatusPredictionSealError(
            "prediction ledger schema or self-hash is invalid"
        )
    fit_payload = _fit_payload_without_artifact(ledger)
    recorded_fit_sha = fit_payload.pop("fit_payload_sha256", None)
    if recorded_fit_sha != _canonical_sha256(fit_payload):
        raise RelationalPreStatusPredictionSealError(
            "prediction fit-payload hash is invalid"
        )
    if (
        ledger.get("models") != list(STORED_MODELS)
        or ledger.get("layers") != list(LAYERS)
        or ledger.get("risk_models") != list(_RISK_MODELS)
        or ledger.get("outcome_classes") != list(OUTCOME_CLASSES)
    ):
        raise RelationalPreStatusPredictionSealError(
            "prediction ledger frozen orders are invalid"
        )
    tensor_entry = _mapping(ledger.get("tensor_artifact"), "tensor artifact")
    relative = _string(tensor_entry.get("path"), "tensor path")
    if relative != TENSOR_FILE_NAME:
        raise RelationalPreStatusPredictionSealError(
            "tensor path is not the frozen filename"
        )
    tensor_path = (artifact_root / relative).resolve()
    if not tensor_path.is_relative_to(artifact_root) or not tensor_path.is_file():
        raise RelationalPreStatusPredictionSealError(
            "tensor path escapes or is absent"
        )
    if (
        tensor_path.stat().st_size
        != _integer(tensor_entry.get("bytes"), "tensor bytes", minimum=1)
        or file_sha256(tensor_path)
        != _sha(tensor_entry.get("sha256"), "tensor physical SHA-256")
    ):
        raise RelationalPreStatusPredictionSealError(
            "tensor byte/SHA binding is invalid"
        )
    with safe_open(tensor_path, framework="numpy") as handle:
        expected_metadata = {
            "schema_version": "1",
            "kind": _TENSOR_KIND,
            "held_out_family_fold": ledger.get("held_out_family_fold"),
            "model_order": json.dumps(list(STORED_MODELS), separators=(",", ":")),
            "layers": json.dumps(list(LAYERS), separators=(",", ":")),
            "query_count": str(ledger.get("lift_query_count")),
        }
        if handle.metadata() != expected_metadata or set(handle.keys()) != {
            "lift_predictions",
            "defined",
        }:
            raise RelationalPreStatusPredictionSealError(
                "tensor metadata or key inventory is invalid"
            )
        values = np.asarray(handle.get_tensor("lift_predictions"))
        defined_raw = np.asarray(handle.get_tensor("defined"))
    if (
        tensor_entry.get("prediction_shape") != list(values.shape)
        or tensor_entry.get("defined_shape") != list(defined_raw.shape)
        or values.ndim != 4
        or values.shape[0] != ledger.get("lift_query_count")
        or values.shape[1] != len(STORED_MODELS)
        or values.shape[2] != len(LAYERS)
        or values.shape[3] < 1
        or values.dtype != np.float32
        or defined_raw.shape != values.shape[:2]
        or not np.isin(defined_raw, (0, 1)).all()
        or not np.isfinite(values).all()
    ):
        raise RelationalPreStatusPredictionSealError(
            "tensor shapes, dtypes, or values are invalid"
        )
    defined = defined_raw.astype(np.bool_)
    expected_hashes = {
        model: _array_sha256(model, values[:, index])
        for index, model in enumerate(STORED_MODELS)
    }
    if dict(
        _mapping(
            ledger.get("prediction_array_sha256_by_model"),
            "prediction array hashes",
        )
    ) != expected_hashes or ledger.get("defined_array_sha256") != _array_sha256(
        "defined", defined.astype(np.uint8)
    ):
        raise RelationalPreStatusPredictionSealError(
            "tensor semantic array hashes are invalid"
        )
    lift_queries = ledger.get("lift_queries")
    risk_predictions = ledger.get("risk_predictions")
    if (
        not isinstance(lift_queries, list)
        or len(lift_queries) != values.shape[0]
        or [row.get("tensor_row_index") for row in lift_queries]
        != list(range(values.shape[0]))
        or not isinstance(risk_predictions, list)
        or len(risk_predictions) != ledger.get("risk_prediction_count")
    ):
        raise RelationalPreStatusPredictionSealError(
            "ledger query inventories are invalid"
        )
    return MappingProxyType(dict(ledger)), FoldPredictionArrays(
        values=np.ascontiguousarray(values, dtype=np.float32),
        defined=np.ascontiguousarray(defined),
    )


def validate_rebuilt_pre_status_fold_prediction(
    submitted_root: Path,
    rebuilt: FoldPredictionBuild,
) -> Mapping[str, Any]:
    """Require exact fit payload and dense array equality before score opens labels."""
    ledger, arrays = load_pre_status_fold_prediction_artifact(submitted_root)
    if _fit_payload_without_artifact(ledger) != dict(rebuilt.fit_payload):
        raise RelationalPreStatusPredictionSealError(
            "submitted prediction ledger differs from exact fold-safe rebuild"
        )
    if not np.array_equal(
        arrays.values, rebuilt.arrays.values
    ) or not np.array_equal(arrays.defined, rebuilt.arrays.defined):
        raise RelationalPreStatusPredictionSealError(
            "submitted prediction arrays differ from exact fold-safe rebuild"
        )
    return ledger


__all__ = [
    "LEDGER_FILE_NAME",
    "PREDICTION_LEDGER_KIND",
    "STORED_MODELS",
    "TENSOR_FILE_NAME",
    "FoldPredictionArrays",
    "FoldPredictionBuild",
    "RelationalPreStatusPredictionSealError",
    "build_pre_status_fold_predictions",
    "load_pre_status_fold_prediction_artifact",
    "save_pre_status_fold_prediction_artifact",
    "validate_rebuilt_pre_status_fold_prediction",
]
