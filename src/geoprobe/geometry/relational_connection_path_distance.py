"""Pure geometry for complete-path connection-response signatures.

This module intentionally depends on no CLI, model-loading, or outcome logic.
It exposes deterministic helpers for:

* selecting invariant relation-side inventory from calibration rows,
* projecting relation diagnostics to normalized principal-angle / polar coordinates,
* and computing bounded per-component / per-view path distances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Final
import hashlib
import json
import math

import numpy as np


DEFAULT_REQUIRED_FOLDS: Final = ("outer_1", "outer_2", "outer_3", "outer_4", "outer_5")

_VIEW_ORDER: Final = ("residual", "attention", "layer_transport")
_VIEW_SET: Final = frozenset(_VIEW_ORDER)
_SIDES_BY_VIEW: Final = {
    "attention": ("left", "right"),
    "residual": ("symmetric",),
    "layer_transport": ("symmetric",),
}

_COMPONENT_ORDER: Final = ("I", "C", "O")
_COMPONENT_RANGES: Final = {"I": 2.0, "C": 1.0, "O": 2.0}

_COSINE_EPS: Final = 1e-12
_HASH_PREFIX: Final = b"geoprobe.connection-path-distance.v1\x00"

_POLICY_STATUSES: Final = frozenset({"admitted"})

_COORDINATE_DIAGNOSTIC_STATUSES: Final = frozenset({"supported", "ill_conditioned"})
_KNOWN_DIAGNOSTIC_STATUSES: Final = frozenset(
    {
        "supported",
        "ill_conditioned",
        "missing",
        "unsupported_frame",
        "insufficient_rank",
        "insufficient_energy",
        "boundary_degenerate",
        "low_attention_support",
        "not_symmetric",
        "invalid_relation",
        "rank_mismatch",
        "binding_mismatch",
    }
)


class RelationalConnectionPathDistanceError(ValueError):
    """Raised when geometric path-distance inputs are invalid."""


@dataclass(frozen=True, slots=True)
class _ProjectedDiagnostic:
    status: str
    coordinates: tuple[float, ...] | None


def _coerce_nonempty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalConnectionPathDistanceError(f"{name} must be a non-empty string")
    return value


def _coerce_int(value: Any, *, name: str, positive: bool = True) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RelationalConnectionPathDistanceError(f"{name} must be an integer")
    if positive and value < 1:
        raise RelationalConnectionPathDistanceError(f"{name} must be positive")
    return int(value)


def _coerce_float(value: Any, *, name: str, min_value: float | None = None, max_value: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RelationalConnectionPathDistanceError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise RelationalConnectionPathDistanceError(f"{name} must be finite")
    if min_value is not None and number < min_value:
        raise RelationalConnectionPathDistanceError(f"{name} is below minimum allowed value")
    if max_value is not None and number > max_value:
        raise RelationalConnectionPathDistanceError(f"{name} exceeds maximum allowed value")
    return number


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(_HASH_PREFIX + encoded).hexdigest()


def _side_id(relation_name: str, view: str, side: str) -> str:
    return f"{relation_name}|{view}|{side}"


def _parse_side_id(value: Any, *, name: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise RelationalConnectionPathDistanceError(f"{name} must be a relation-side key")
    parts = value.split("|")
    if len(parts) != 3 or any(part == "" for part in parts):
        raise RelationalConnectionPathDistanceError(f"{name} must be in relation|view|side format")
    relation_name, view, side = parts
    _coerce_nonempty_str(view, name=f"{name} view")
    _coerce_nonempty_str(side, name=f"{name} side")
    if view not in _VIEW_SET:
        raise RelationalConnectionPathDistanceError(f"{name} view is invalid: {view}")
    return relation_name, view, side


def _coerce_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalConnectionPathDistanceError(f"{name} must be a mapping")
    return value


def _coerce_sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise RelationalConnectionPathDistanceError(f"{name} must be a list")


def _coerce_non_negative_float(value: Any, *, name: str, allow_one: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RelationalConnectionPathDistanceError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise RelationalConnectionPathDistanceError(f"{name} must be finite")
    if number < 0:
        raise RelationalConnectionPathDistanceError(f"{name} must be non-negative")
    if (not allow_one and number >= 1) or number > 1:
        raise RelationalConnectionPathDistanceError(f"{name} must be <= 1")
    return number


def _coerce_relation_side_payload(
    value: Any,
    *,
    context: str,
    require_selected_rank: bool,
) -> tuple[str, str, str, int | None]:
    if not isinstance(value, Mapping):
        raise RelationalConnectionPathDistanceError(f"{context} must be a relation-side mapping")
    relation_name = _coerce_nonempty_str(value.get("relation_name"), name=f"{context}.relation_name")
    view = _coerce_nonempty_str(value.get("view"), name=f"{context}.view")
    if view not in _VIEW_SET:
        raise RelationalConnectionPathDistanceError(f"{context}.view is invalid: {view}")
    side = _coerce_nonempty_str(value.get("side"), name=f"{context}.side")
    if require_selected_rank:
        selected_rank = _coerce_int(value.get("selected_rank"), name=f"{context}.selected_rank")
    else:
        selected_rank = None
    return relation_name, view, side, selected_rank


def stable_common_rank_relation_sides(
    selection: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    validate_counts: bool = False,
    expected_relation_count: int = 63,
    expected_relation_side_count: int = 120,
    required_folds: tuple[str, ...] = DEFAULT_REQUIRED_FOLDS,
) -> dict[str, Any]:
    """Select relation sides with a constant admissible rank across heldout folds.

    The function keeps only rows admitted in all required folds and with identical
    selected rank across those folds, then expands each relation to view-appropriate
    sides (left/right for attention; symmetric for residual/layer_transport).
    """

    if isinstance(selection, Mapping):
        raw_rows = _coerce_sequence(
            selection.get("selection_rows", selection.get("selection", tuple())),
            name="selection.selection_rows",
        )
        if not raw_rows:
            raise RelationalConnectionPathDistanceError("selection has no selection rows")
    else:
        raw_rows = _coerce_sequence(selection, name="selection")

    required = tuple(_coerce_nonempty_str(value, name="required_fold") for value in required_folds)
    if len(required) != len(set(required)):
        raise RelationalConnectionPathDistanceError("required_folds must be unique")

    observed: dict[tuple[str, str], dict[str, tuple[int, str]]] = {}

    for row_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise RelationalConnectionPathDistanceError(f"selection row {row_index} must be a mapping")
        relation_name = _coerce_nonempty_str(raw_row.get("relation_name"), name=f"selection[{row_index}].relation_name")
        view = _coerce_nonempty_str(raw_row.get("view"), name=f"selection[{row_index}].view")
        if view not in _VIEW_SET:
            raise RelationalConnectionPathDistanceError(f"selection[{row_index}].view is invalid: {view}")
        selected_rank = _coerce_int(raw_row.get("selected_rank"), name=f"selection[{row_index}].selected_rank")

        fold = _coerce_nonempty_str(
            raw_row.get("heldout_family_fold")
            or raw_row.get("family_fold")
            or raw_row.get("fold"),
            name=f"selection[{row_index}].fold",
        )
        status = _coerce_nonempty_str(
            raw_row.get("status")
            or raw_row.get("policy_status")
            or raw_row.get("selection_status"),
            name=f"selection[{row_index}].status",
        ).lower()
        admissible = raw_row.get("admissible", True)
        if not isinstance(admissible, bool):
            raise RelationalConnectionPathDistanceError(
                f"selection[{row_index}].admissible must be a boolean"
            )
        if status not in {s.lower() for s in {"admitted", "not_found"}}:
            raise RelationalConnectionPathDistanceError(f"selection[{row_index}].status is invalid")
        if not admissible:
            continue
        if status != "admitted":
            continue

        identifier = (relation_name, view)
        fold_rows = observed.setdefault(identifier, {})
        if fold in fold_rows:
            raise RelationalConnectionPathDistanceError(f"duplicate fold for relation {identifier}")
        fold_rows[fold] = (selected_rank, status)

    if not observed:
        raise RelationalConnectionPathDistanceError("selection has no admissible relation rows")

    selected_sides: list[dict[str, Any]] = []
    for relation_name, view in sorted(observed):
        fold_rows = observed[(relation_name, view)]
        if set(fold_rows) != set(required):
            continue
        if any(status.lower() not in _POLICY_STATUSES for status in (s for _, s in fold_rows.values())):
            continue

        ranks = {rank for rank, _ in fold_rows.values()}
        if len(ranks) != 1:
            continue
        rank = next(iter(ranks))
        for side in _SIDES_BY_VIEW[view]:
            selected_sides.append(
                {
                    "relation_name": relation_name,
                    "view": view,
                    "side": side,
                    "selected_rank": rank,
                }
            )

    payload = {
        "relation_sides": selected_sides,
        "relation_side_count": len(selected_sides),
        "relation_count": len({side["relation_name"] for side in selected_sides}),
    }
    payload["inventory_hash"] = _canonical_hash(payload)

    if validate_counts and payload["relation_count"] != expected_relation_count:
        raise RelationalConnectionPathDistanceError(
            f"relation count {payload['relation_count']} does not match required {expected_relation_count}"
        )
    if validate_counts and payload["relation_side_count"] != expected_relation_side_count:
        raise RelationalConnectionPathDistanceError(
            "relation side count "
            f"{payload['relation_side_count']} does not match required {expected_relation_side_count}"
        )
    return payload


def _coerce_policy(value: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw_rows = value.get("relation_sides")
        if raw_rows is None:
            raw_rows = value.get("relation_side_inventory")
        if raw_rows is None:
            raise RelationalConnectionPathDistanceError(
                "policy must expose relation_sides or relation_side_inventory"
            )
        expected_hash = value.get("inventory_hash")
    else:
        raw_rows = value
        expected_hash = None

    rows = _coerce_sequence(raw_rows, name="policy.relation_sides")
    if not rows:
        raise RelationalConnectionPathDistanceError("policy has no relation sides")
    if any(not isinstance(raw_row, Mapping) for raw_row in rows):
        raise RelationalConnectionPathDistanceError("policy.relation_sides must contain mapping rows")

    has_side = any("side" in raw_row for raw_row in rows)
    if has_side and not all("side" in raw_row for raw_row in rows):
        raise RelationalConnectionPathDistanceError("policy rows must either all include side or all omit side")
    if has_side:
        relation_sides: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows):
            relation_name, view, side, selected_rank = _coerce_relation_side_payload(
                raw_row,
                context=f"policy[{index}]",
                require_selected_rank=True,
            )
            selected_rank = _coerce_int(selected_rank, name=f"policy[{index}].selected_rank")
            identifier = _side_id(relation_name, view, side)
            if any(_side_id(row["relation_name"], row["view"], row["side"]) == identifier for row in relation_sides):
                raise RelationalConnectionPathDistanceError(f"duplicate relation side in policy: {identifier}")
            relation_sides.append(
                {
                    "relation_name": relation_name,
                    "view": view,
                    "side": side,
                    "selected_rank": selected_rank,
                }
            )
        stable = {
            "relation_sides": sorted(
                relation_sides,
                key=lambda row: (row["relation_name"], row["view"], row["side"]),
            ),
            "relation_side_count": len(relation_sides),
            "relation_count": len({row["relation_name"] for row in relation_sides}),
        }
        stable["inventory_hash"] = _canonical_hash(stable)

    else:
        stable = stable_common_rank_relation_sides(
            [
                {
                    "relation_name": _coerce_nonempty_str(raw_row.get("relation_name"), name=f"policy[{index}].relation_name"),
                    "view": _coerce_nonempty_str(raw_row.get("view"), name=f"policy[{index}].view"),
                    "selected_rank": _coerce_int(raw_row.get("selected_rank"), name=f"policy[{index}].selected_rank"),
                    "status": _coerce_nonempty_str(
                        raw_row.get("status") or raw_row.get("policy_status") or raw_row.get("selection_status"),
                        name=f"policy[{index}].status",
                    ).lower(),
                    "admissible": raw_row.get("admissible", True),
                    "heldout_family_fold": raw_row.get("heldout_family_fold")
                    or raw_row.get("family_fold")
                    or raw_row.get("fold"),
                }
                for index, raw_row in enumerate(rows)
            ],
            validate_counts=False,
            required_folds=DEFAULT_REQUIRED_FOLDS,
        )

    if expected_hash is not None and _coerce_nonempty_str(expected_hash, name="policy.inventory_hash") != stable["inventory_hash"]:
        raise RelationalConnectionPathDistanceError("policy inventory hash does not match policy payload")

    return {
        "relation_sides": stable["relation_sides"],
        "inventory_hash": stable["inventory_hash"],
    }


def _coerce_event_records(value: Any, *, context: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        # Either a single row with explicit identity fields
        if {"relation_name", "view", "side"}.issubset(set(value.keys())):
            relation_name, view, side, _ = _coerce_relation_side_payload(
                value,
                context=f"{context}[0]",
                require_selected_rank=False,
            )
            return {_side_id(relation_name, view, side): value}

        records: dict[str, Mapping[str, Any]] = {}
        for key, item in value.items():
            if not isinstance(item, Mapping):
                raise RelationalConnectionPathDistanceError(f"{context}[{key}] must be a mapping")
            if {"relation_name", "view", "side"}.issubset(set(item.keys())):
                relation_name, view, side, _ = _coerce_relation_side_payload(
                    item,
                    context=f"{context}[{key}]",
                    require_selected_rank=False,
                )
                identifier = _side_id(relation_name, view, side)
            else:
                identifier = _parse_side_id(key, name=f"{context}[{key}]")
            records[identifier] = item
        if not records:
            raise RelationalConnectionPathDistanceError(f"{context} is empty")
        return records

    if isinstance(value, (list, tuple)):
        records = {}
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise RelationalConnectionPathDistanceError(f"{context}[{index}] must be a mapping")
            relation_name, view, side, _ = _coerce_relation_side_payload(
                item,
                context=f"{context}[{index}]",
                require_selected_rank=False,
            )
            records[_side_id(relation_name, view, side)] = item
        if not records:
            raise RelationalConnectionPathDistanceError(f"{context} is empty")
        return records

    raise RelationalConnectionPathDistanceError(f"{context} must be mapping or sequence")


def _project_to_coordinate(value: float, *, context: str) -> float:
    # Principal-angle cosine to normalized coordinate in [0, 1] using the
    # 2/π arccos transform and 1e-12 clamping near cosines close to one.
    cosine = _coerce_float(value, name=f"{context} cosine")
    if cosine > 1 + _COSINE_EPS:
        raise RelationalConnectionPathDistanceError(f"{context} cosine exceeds one beyond tolerance")
    if cosine < -_COSINE_EPS:
        raise RelationalConnectionPathDistanceError(f"{context} cosine is below zero beyond tolerance")
    if cosine > 1:
        cosine = 1.0
    if cosine < 0:
        cosine = 0.0
    return 2.0 * math.acos(cosine) / math.pi


def _project_diagnostic(
    diagnostic: Mapping[str, Any], *, selected_rank: int
) -> _ProjectedDiagnostic:
    status = _coerce_nonempty_str(
        diagnostic.get("status"),
        name="diagnostic.status",
    ).lower()
    if status not in _KNOWN_DIAGNOSTIC_STATUSES:
        raise RelationalConnectionPathDistanceError(f"diagnostic.status is unknown: {status}")

    if status in _COORDINATE_DIAGNOSTIC_STATUSES:
        raw_cosines = diagnostic.get("principal_angle_cosines")
        if not isinstance(raw_cosines, (list, tuple)):
            raise RelationalConnectionPathDistanceError("diagnostic principal_angle_cosines must be a list")
        cosines = tuple(raw_cosines)
        if len(cosines) != selected_rank:
            return _ProjectedDiagnostic(status="rank_mismatch", coordinates=None)

        principal = tuple(
            _project_to_coordinate(
                cosines[index],
                context=f"diagnostic principal_angle_cosines[{index}]",
            )
            for index in range(selected_rank)
        )

        projected = (_coerce_non_negative_float(
            diagnostic.get("normalized_transported_projector_discrepancy"),
            name=f"diagnostic for {status} projector discrepancy",
        ),
                     _coerce_non_negative_float(
                         diagnostic.get("normalized_polar_residual"),
                         name=f"diagnostic for {status} polar residual",
                     ))
        return _ProjectedDiagnostic(status=status, coordinates=(*principal, *projected))

    return _ProjectedDiagnostic(status=status, coordinates=None)


def _combine_coordinates(
    diagnostics: tuple[Mapping[str, Any], ...],
    selected_rank: int,
) -> list[_ProjectedDiagnostic]:
    return tuple(_project_diagnostic(diagnostic, selected_rank=selected_rank) for diagnostic in diagnostics)


def _extract_forward_diagnostic(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if "forward" in value and isinstance(value.get("forward"), Mapping):
        forward = value.get("forward")
        if isinstance(forward, Mapping):
            return forward
    return value


def _aggregate_coordinates(
    values: Sequence[_ProjectedDiagnostic],
    *,
    method: Literal["mean", "median"],
) -> tuple[float, ...] | None:
    if not values:
        return None

    arrays = [
        np.asarray(item.coordinates, dtype=np.float64)
        for item in values
        if item.coordinates is not None
    ]
    if not arrays:
        return None

    dims = arrays[0].size
    for index, array in enumerate(arrays, start=1):
        if array.shape != (dims,):
            raise RelationalConnectionPathDistanceError(
                f"incompatible diagnostic coordinate dimension at index {index}"
            )

    stacked = np.stack(arrays, axis=0)
    if method == "mean":
        aggregate = np.mean(stacked, axis=0)
    else:
        aggregate = np.median(stacked, axis=0)
    return tuple(float(value) for value in aggregate.tolist())


def _vector_for_component(
    rows: Sequence[_ProjectedDiagnostic],
    *,
    method: Literal["mean", "median"],
    fallback_zero: bool,
) -> tuple[float, ...] | None:
    values = _aggregate_coordinates(rows, method=method)
    if values is None and not fallback_zero:
        return None
    if values is None and fallback_zero:
        return None
    return values


def _project_component(
    contribution_rows: Sequence[_ProjectedDiagnostic],
    *,
    method: Literal["mean", "median"],
) -> dict[str, Any]:
    statuses = tuple(sorted(row.status for row in contribution_rows))
    vector = _vector_for_component(contribution_rows, method=method, fallback_zero=False)
    return {"defined": vector is not None, "status": list(statuses), "coordinates": list(vector) if vector else None}


def build_complete_path_signature(
    incoming_primaries: Sequence[Any],
    replay_pairs: Sequence[Any],
    outgoing_aa: Any,
    outgoing_ab: Any,
    policies: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    policy = _coerce_policy(policies)
    if len(incoming_primaries) != 3:
        raise RelationalConnectionPathDistanceError("incoming_primaries must provide three realizations")
    if len(replay_pairs) != 3:
        raise RelationalConnectionPathDistanceError("replay_pairs must provide three realizations")

    incoming_records = tuple(
        _coerce_event_records(item, context=f"incoming_primaries[{index}]") for index, item in enumerate(incoming_primaries)
    )
    replay_records = tuple(
        _coerce_event_records(item, context=f"replay_pairs[{index}]") for index, item in enumerate(replay_pairs)
    )
    aa_records = _coerce_event_records(outgoing_aa, context="outgoing_aa")
    ab_records = _coerce_event_records(outgoing_ab, context="outgoing_ab")

    payload_sides: list[dict[str, Any]] = []
    for index, side in enumerate(policy["relation_sides"]):
        relation_name = _coerce_nonempty_str(side.get("relation_name"), name=f"policy.relation_sides[{index}].relation_name")
        view = _coerce_nonempty_str(side.get("view"), name=f"policy.relation_sides[{index}].view")
        if view not in _VIEW_SET:
            raise RelationalConnectionPathDistanceError(f"policy.relation_sides[{index}].view is invalid")
        side_id = side.get("side")
        if not isinstance(side_id, str) or not side_id:
            raise RelationalConnectionPathDistanceError(f"policy.relation_sides[{index}].side must be non-empty")
        selected_rank = _coerce_int(side.get("selected_rank"), name=f"policy.relation_sides[{index}].selected_rank")
        identifier = _side_id(relation_name, view, side_id)

        primary_diagnostics = [
            _project_diagnostic(
                _extract_forward_diagnostic(records.get(identifier, {"status": "missing"})),
                selected_rank=selected_rank,
            )
            for records in incoming_records
        ]
        replay_diagnostics = [
            _project_diagnostic(
                _extract_forward_diagnostic(records.get(identifier, {"status": "missing"})),
                selected_rank=selected_rank,
            )
            for records in replay_records
        ]

        aa_diagnostic = _project_diagnostic(
            _extract_forward_diagnostic(aa_records.get(identifier, {"status": "missing"})),
            selected_rank=selected_rank,
        )
        ab_diagnostic = _project_diagnostic(
            _extract_forward_diagnostic(ab_records.get(identifier, {"status": "missing"})),
            selected_rank=selected_rank,
        )

        primary_vector = _aggregate_coordinates(tuple(d for d in primary_diagnostics if d.coordinates is not None), method="median")
        replay_vector = _aggregate_coordinates(tuple(d for d in replay_diagnostics if d.coordinates is not None), method="median")
        aa_vector = _aggregate_coordinates((aa_diagnostic,), method="mean")
        ab_vector = _aggregate_coordinates((ab_diagnostic,), method="mean")

        i_statuses = tuple(sorted(item.status for item in primary_diagnostics + replay_diagnostics))
        c_statuses = tuple(sorted(item.status for item in (aa_diagnostic, ab_diagnostic)))
        o_statuses = tuple(sorted(item.status for item in (aa_diagnostic, ab_diagnostic)))

        if primary_vector is None or replay_vector is None:
            i_coordinates = None
        else:
            if len(primary_vector) != len(replay_vector):
                raise RelationalConnectionPathDistanceError(f"I component dimensions mismatch for {identifier}")
            i_coordinates = [p - r for p, r in zip(primary_vector, replay_vector, strict=True)]

        if aa_vector is None or ab_vector is None:
            c_coordinates = None
            o_coordinates = None
        else:
            if len(aa_vector) != len(ab_vector):
                raise RelationalConnectionPathDistanceError(f"C/O component dimensions mismatch for {identifier}")
            c_coordinates = [(aa + ab) * 0.5 for aa, ab in zip(aa_vector, ab_vector, strict=True)]
            o_coordinates = [ab - aa for ab, aa in zip(ab_vector, aa_vector, strict=True)]

        payload_sides.append(
            {
                "relation_name": relation_name,
                "view": view,
                "side": side_id,
                "selected_rank": selected_rank,
                "I": {
                    "defined": i_coordinates is not None,
                    "status": list(i_statuses),
                    "coordinates": i_coordinates,
                },
                "C": {
                    "defined": c_coordinates is not None,
                    "status": list(c_statuses),
                    "coordinates": c_coordinates,
                },
                "O": {
                    "defined": o_coordinates is not None,
                    "status": list(o_statuses),
                    "coordinates": o_coordinates,
                },
            }
        )

    if not payload_sides:
        raise RelationalConnectionPathDistanceError("no admissible policy relation-sides were available")

    payload = {
        "inventory_hash": policy["inventory_hash"],
        "relation_sides": payload_sides,
    }
    payload["signature_hash"] = _canonical_hash(payload)
    return payload


def _coerce_signature(value: Any) -> tuple[str, dict[str, Any]]:
    signature = _coerce_mapping(value, name="signature")
    signature_hash = _coerce_nonempty_str(signature.get("inventory_hash"), name="signature.inventory_hash")
    relation_sides = _coerce_sequence(signature.get("relation_sides"), name="signature.relation_sides")
    side_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(relation_sides):
        if not isinstance(row, Mapping):
            raise RelationalConnectionPathDistanceError(f"signature.relation_sides[{index}] must be a mapping")
        relation_name = _coerce_nonempty_str(row.get("relation_name"), name=f"signature.relation_sides[{index}].relation_name")
        view = _coerce_nonempty_str(row.get("view"), name=f"signature.relation_sides[{index}].view")
        if view not in _VIEW_SET:
            raise RelationalConnectionPathDistanceError(f"signature.relation_sides[{index}].view is invalid")
        side = _coerce_nonempty_str(row.get("side"), name=f"signature.relation_sides[{index}].side")
        _coerce_int(row.get("selected_rank"), name=f"signature.relation_sides[{index}].selected_rank")
        identifier = _side_id(relation_name, view, side)
        if identifier in side_rows:
            raise RelationalConnectionPathDistanceError(f"duplicate relation side in signature: {identifier}")

        components = {key: _coerce_mapping(row.get(key), name=f"signature.relation_sides[{index}].{key}") for key in _COMPONENT_ORDER}
        side_rows[identifier] = {
            "view": view,
            "relation_name": relation_name,
            "side": side,
            "selected_rank": _coerce_int(row.get("selected_rank"), name=f"signature.relation_sides[{index}].selected_rank"),
            "components": components,
        }

        for component_name in _COMPONENT_ORDER:
            if "status" not in components[component_name] or not isinstance(components[component_name].get("status"), (list, tuple)):
                raise RelationalConnectionPathDistanceError(
                    f"signature.relation_sides[{index}].{component_name}.status must be a list"
                )
            if components[component_name]["status"] is None:
                raise RelationalConnectionPathDistanceError(
                    f"signature.relation_sides[{index}].{component_name}.status may not be null"
                )
            status = tuple(components[component_name]["status"])
            if any(not isinstance(item, str) for item in status):
                raise RelationalConnectionPathDistanceError(f"signature.relation_sides[{index}].{component_name}.status contains non-strings")

            coordinates = components[component_name].get("coordinates")
            if coordinates is not None and not isinstance(coordinates, (list, tuple)):
                raise RelationalConnectionPathDistanceError(
                    f"signature.relation_sides[{index}].{component_name}.coordinates must be list-like or null"
                )
            if isinstance(coordinates, tuple):
                components[component_name]["coordinates"] = list(coordinates)
            if coordinates is not None:
                for value in components[component_name]["coordinates"]:
                    _coerce_float(value, name=f"signature.relation_sides[{index}].{component_name}.coordinate")

    return signature_hash, side_rows


def _relation_side_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    component: str,
) -> float:
    left_status = tuple(str(item) for item in left["status"])
    right_status = tuple(str(item) for item in right["status"])
    left_support = (
        left_status,
        bool(left.get("defined")),
        tuple(bool(item) for item in left.get("missing_mask", ())),
    )
    right_support = (
        right_status,
        bool(right.get("defined")),
        tuple(bool(item) for item in right.get("missing_mask", ())),
    )
    status_distance = 0.0 if left_support == right_support else 1.0

    left_coordinates = left.get("coordinates")
    right_coordinates = right.get("coordinates")
    if left_coordinates is None and right_coordinates is None:
        numeric_distance = 0.0
    elif left_coordinates is None or right_coordinates is None:
        numeric_distance = 1.0
    else:
        if len(left_coordinates) != len(right_coordinates):
            raise RelationalConnectionPathDistanceError(
                f"coordinate dimension mismatch for component {component}"
            )
        left_vector = np.asarray(left_coordinates, dtype=np.float64)
        right_vector = np.asarray(right_coordinates, dtype=np.float64)
        distance = float(np.sqrt(np.mean(np.square(left_vector - right_vector))))
        numeric_distance = min(
            1.0,
            max(
                0.0,
                distance / _COMPONENT_RANGES[component],
            ),
        )

    return float(math.sqrt((status_distance**2 + numeric_distance**2) / 2.0))


def complete_path_view_distances(
    left_signature: Mapping[str, Any],
    right_signature: Mapping[str, Any],
    components: Sequence[str] = _COMPONENT_ORDER,
    views: Sequence[str] | None = None,
) -> dict[str, Any]:
    left_hash, left_rows = _coerce_signature(left_signature)
    right_hash, right_rows = _coerce_signature(right_signature)
    if left_hash != right_hash:
        raise RelationalConnectionPathDistanceError("signature inventories must match")

    selected_components = tuple(_coerce_nonempty_str(component, name=f"components[{index}]") for index, component in enumerate(components))
    for component in selected_components:
        if component not in _COMPONENT_ORDER:
            raise RelationalConnectionPathDistanceError(f"unknown component {component}")

    selected_views = tuple(_coerce_nonempty_str(view, name=f"views[{index}]") for index, view in enumerate(views or _VIEW_ORDER))
    invalid = [view for view in selected_views if view not in _VIEW_SET]
    if invalid:
        raise RelationalConnectionPathDistanceError(f"invalid views: {invalid}")

    left_identifiers = set(left_rows)
    right_identifiers = set(right_rows)
    if left_identifiers != right_identifiers:
        raise RelationalConnectionPathDistanceError("signatures do not expose the same relation-side inventory")

    relation_side_ids_by_view: dict[str, list[str]] = {}
    for side_id in left_rows:
        side_view = left_rows[side_id]["view"]
        relation_side_ids_by_view.setdefault(side_view, []).append(side_id)

    for view in selected_views:
        if view not in relation_side_ids_by_view:
            raise RelationalConnectionPathDistanceError(f"signature has no relation sides for view {view}")

    component_view_distances: dict[str, dict[str, float]] = {view: {} for view in selected_views}

    def _rms(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        arr = np.asarray(values, dtype=np.float64)
        return float(math.sqrt(np.mean(np.square(arr))))

    view_distances: dict[str, float] = {}
    for view in selected_views:
        side_ids = sorted(relation_side_ids_by_view[view])
        for component in selected_components:
            per_side_distances = [
                _relation_side_distance(
                    left_rows[side_id]["components"][component],
                    right_rows[side_id]["components"][component],
                    component=component,
                )
                for side_id in side_ids
            ]
            component_view_distances[view][component] = _rms(per_side_distances)
        view_distances[view] = _rms([component_view_distances[view][component] for component in selected_components])

    component_distances: dict[str, float] = {
        component: _rms([component_view_distances[view][component] for view in selected_views])
        for component in selected_components
    }
    full_distance = _rms([view_distances[view] for view in selected_views])

    component_ablations: dict[str, float] = {}
    for component in selected_components:
        keep = tuple(other for other in selected_components if other != component)
        if not keep:
            component_ablations[component] = 1.0
            continue
        distances = [
            _rms([component_view_distances[view][other] for other in keep])
            for view in selected_views
        ]
        component_ablations[component] = _rms(distances)

    view_ablations: dict[str, float] = {}
    for view in selected_views:
        keep = tuple(other for other in selected_views if other != view)
        if not keep:
            view_ablations[view] = 1.0
            continue
        view_ablations[view] = _rms([view_distances[item] for item in keep])

    return {
        "components": selected_components,
        "views": selected_views,
        "relation_side_counts": {
            view: len(relation_side_ids_by_view[view]) for view in selected_views
        },
        "component_view_distances": component_view_distances,
        "view_distances": view_distances,
        "component_distances": component_distances,
        "full_distance": full_distance,
        "component_ablations": component_ablations,
        "view_ablations": view_ablations,
    }
