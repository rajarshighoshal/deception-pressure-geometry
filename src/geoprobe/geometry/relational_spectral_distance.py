"""Pure NumPy utilities for intrinsic relational spectral distances."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from collections.abc import Mapping, Sequence

import numpy as np


_VIEWS = ("residual", "attention", "layer_transport")
_VALID_VIEWS = frozenset(_VIEWS)


class RelationalSpectralDistanceError(ValueError):
    """Raised when profile/policy inputs are invalid for distance computation."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RelationalSpectralDistanceError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise RelationalSpectralDistanceError(f"{name} must be a sequence")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalSpectralDistanceError(f"{name} must be a non-empty string")
    return value


def _finite_scalar(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise RelationalSpectralDistanceError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise RelationalSpectralDistanceError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise RelationalSpectralDistanceError(f"{name} must be at least {minimum}")
    return result


def _view(value: object, name: str) -> str:
    result = _string(value, name)
    if result not in _VALID_VIEWS:
        raise RelationalSpectralDistanceError(f"{name} must be one of {_VIEWS}")
    return result


def _vector(value: object, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1:
        raise RelationalSpectralDistanceError(f"{name} must be one-dimensional")
    if not np.isfinite(values).all():
        raise RelationalSpectralDistanceError(f"{name} must be finite")
    return values


def _ensure_nonnegative(values: np.ndarray, name: str) -> None:
    if np.any(values < 0):
        raise RelationalSpectralDistanceError(f"{name} may not be negative")


@dataclass(frozen=True, slots=True)
class _FrozenRelationProfile:
    relation_name: str
    view: str
    status: str
    leading: np.ndarray
    squared_energy_total: float
    signed_leading: np.ndarray | None


def _validate_relation_profile(
    profile: Mapping[str, object],
    selected_rank: int,
) -> _FrozenRelationProfile:
    relation_name = _string(profile.get("relation_name"), "relation_name")
    view = _view(profile.get("view"), "relation view")
    status = _string(profile.get("status"), "relation status")
    if status != "valid":
        raise RelationalSpectralDistanceError("relation status must be valid")

    leading = _vector(profile.get("leading_spectral_values"), "leading_spectral_values")
    _ensure_nonnegative(leading, "leading_spectral_values")
    if leading.size == 0:
        raise RelationalSpectralDistanceError("leading_spectral_values must be non-empty")

    rank = _positive_int(selected_rank, "selected_rank")
    if rank > leading.size:
        raise RelationalSpectralDistanceError("selected_rank exceeds leading_spectral_values length")

    total = _finite_scalar(profile.get("squared_energy_total"), "squared_energy_total")
    if total <= 0:
        raise RelationalSpectralDistanceError("squared_energy_total must be positive")

    retained = leading[:rank]
    retained_energy = float(np.square(retained).sum())
    tail = total - retained_energy
    if tail < -1e-12:
        raise RelationalSpectralDistanceError("squared_energy_total is inconsistent with retained leading spectrum")

    signed = None
    raw_signed = profile.get("signed_leading_spectral_values")
    if view == "layer_transport":
        if raw_signed is None:
            raise RelationalSpectralDistanceError("layer_transport profiles require signed_leading_spectral_values")
        signed = _vector(raw_signed, "signed_leading_spectral_values")
        if signed.shape != leading.shape:
            raise RelationalSpectralDistanceError("signed_leading_spectral_values must match leading_spectral_values length")
        if not np.allclose(np.abs(signed), leading, atol=1e-12, rtol=1e-12):
            raise RelationalSpectralDistanceError("signed values must bind leading magnitudes")
    elif raw_signed is not None:
        raise RelationalSpectralDistanceError("non-transport profiles may not carry signed_leading_spectral_values")

    return _FrozenRelationProfile(
        relation_name=relation_name,
        view=view,
        status=status,
        leading=leading,
        squared_energy_total=total,
        signed_leading=signed,
    )


def validate_frozen_relation_profile(profile: Mapping[str, object], *, selected_rank: int) -> None:
    """Validate a frozen relation profile and selected rank."""

    _validate_relation_profile(profile, selected_rank)


def validate_frozen_relation_profile_and_selected_rank(
    profile: Mapping[str, object],
    selected_rank: int,
) -> None:
    """Compatibility wrapper with an explicit positional signature."""

    _validate_relation_profile(profile, selected_rank)


def build_relation_mass_vector(profile: Mapping[str, object], selected_rank: int) -> np.ndarray:
    """Build normalized retained-spectrum masses and a single tail mass."""

    parsed = _validate_relation_profile(profile, selected_rank)
    leading = parsed.leading
    rank = selected_rank
    retained = np.square(leading[:rank])
    tail = float(parsed.squared_energy_total) - float(retained.sum())
    if tail < 0.0:
        tail = 0.0
    masses = np.concatenate((retained, np.asarray([tail], dtype=np.float64)))
    return masses / float(parsed.squared_energy_total)


def build_relation_rank_mass(profile: Mapping[str, object], selected_rank: int) -> np.ndarray:
    """Compatibility alias for rank-mass construction."""

    return build_relation_mass_vector(profile, selected_rank)


def build_relation_mass(profile: Mapping[str, object], selected_rank: int) -> np.ndarray:
    """Another compatibility alias for rank-mass construction."""

    return build_relation_mass_vector(profile, selected_rank)


def _build_layer_transport_sign_mass(profile: Mapping[str, object], selected_rank: int) -> np.ndarray:
    parsed = _validate_relation_profile(profile, selected_rank)
    if parsed.view != "layer_transport":
        raise RelationalSpectralDistanceError("signed mass requires a layer_transport profile")

    signed = parsed.signed_leading[:selected_rank]
    retained = np.square(parsed.leading[:selected_rank])
    tail = float(parsed.squared_energy_total) - float(retained.sum())
    if tail < 0.0:
        tail = 0.0
    positive = np.square(np.maximum(signed, 0.0)).sum()
    negative = np.square(np.minimum(signed, 0.0)).sum()
    masses = np.asarray([positive, negative, tail], dtype=np.float64)
    return masses / float(parsed.squared_energy_total)


def hellinger_distance(left_mass: Sequence[float] | np.ndarray, right_mass: Sequence[float] | np.ndarray) -> float:
    """Compute the Hellinger distance between two finite mass vectors."""

    left = np.asarray(left_mass, dtype=np.float64)
    right = np.asarray(right_mass, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise RelationalSpectralDistanceError("mass vectors must have matching one-dimensional shape")
    if left.size == 0:
        raise RelationalSpectralDistanceError("mass vectors must be non-empty")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RelationalSpectralDistanceError("mass vectors must be finite")
    if np.any(left < 0) or np.any(right < 0):
        raise RelationalSpectralDistanceError("mass vectors must be non-negative")
    return float(np.linalg.norm(np.sqrt(left) - np.sqrt(right)) / np.sqrt(2.0))


def relation_profile_distance(
    left_profile: Mapping[str, object],
    right_profile: Mapping[str, object],
    selected_rank: int,
) -> float:
    """Distance for a single relation using rank (and sign for layer_transport)."""

    left = _validate_relation_profile(left_profile, selected_rank)
    right = _validate_relation_profile(right_profile, selected_rank)
    if left.relation_name != right.relation_name:
        raise RelationalSpectralDistanceError("relation names must match")
    if left.view != right.view:
        raise RelationalSpectralDistanceError("relation views must match")

    rank_distance = hellinger_distance(
        build_relation_mass_vector(left_profile, selected_rank),
        build_relation_mass_vector(right_profile, selected_rank),
    )
    if left.view != "layer_transport":
        return rank_distance

    sign_distance = hellinger_distance(
        _build_layer_transport_sign_mass(left_profile, selected_rank),
        _build_layer_transport_sign_mass(right_profile, selected_rank),
    )
    return float(np.sqrt(0.5 * (rank_distance * rank_distance + sign_distance * sign_distance)))


def relation_distance(
    left_profile: Mapping[str, object],
    right_profile: Mapping[str, object],
    selected_rank: int,
) -> float:
    """Compatibility alias for single-relation distance."""

    return relation_profile_distance(left_profile, right_profile, selected_rank)


def compute_layer_transport_relation_distance(
    left_profile: Mapping[str, object],
    right_profile: Mapping[str, object],
    selected_rank: int,
) -> float:
    """Alias that always applies signed mass logic for layer transport."""

    return relation_profile_distance(left_profile, right_profile, selected_rank)


def _index_state_relations(
    state_profile: Mapping[str, object],
    *,
    name: str,
) -> dict[str, Mapping[str, object]]:
    relations = _sequence(state_profile.get("relations"), f"{name} relations")
    indexed: dict[str, Mapping[str, object]] = {}
    for raw_relation in relations:
        relation = _mapping(raw_relation, f"{name} relation")
        relation_name = _string(relation.get("relation_name"), f"{name} relation name")
        if relation_name in indexed:
            raise RelationalSpectralDistanceError(f"{name} relation {relation_name} is duplicated")
        indexed[relation_name] = relation
    return indexed


def _validate_policy_records(policies: Sequence[Mapping[str, object]]) -> list[tuple[str, str, int]]:
    if not policies:
        raise RelationalSpectralDistanceError("admitted relation policy sequence may not be empty")
    validated: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for policy in policies:
        record = _mapping(policy, "admitted policy")
        relation_name = _string(record.get("relation_name"), "policy relation_name")
        if relation_name in seen:
            raise RelationalSpectralDistanceError("policy relation names must be unique")
        seen.add(relation_name)
        view = _view(record.get("view"), "policy view")
        rank = _positive_int(record.get("selected_rank"), "selected_rank")
        validated.append((relation_name, view, rank))
    return validated


def state_profile_view_rms_distances(
    source_state_profile: Mapping[str, object],
    target_state_profile: Mapping[str, object],
    admitted_relation_policies: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute per-view RMS distances for residual, attention and layer_transport."""

    source_relations = _index_state_relations(
        _mapping(source_state_profile, "source state profile"),
        name="source",
    )
    target_relations = _index_state_relations(
        _mapping(target_state_profile, "target state profile"),
        name="target",
    )
    if source_relations.keys() != target_relations.keys():
        raise RelationalSpectralDistanceError("state profiles have different relation inventories")

    policies = _validate_policy_records(_sequence(admitted_relation_policies, "admitted relation policies"))
    policy_names = [relation_name for relation_name, _, _ in policies]
    if not set(policy_names) <= set(source_relations):
        raise RelationalSpectralDistanceError(
            "admitted policy relation inventory is not a subset of the state profile inventory"
        )

    by_view: dict[str, list[float]] = {view: [] for view in _VIEWS}
    policy_by_name = {relation_name: (view, rank) for relation_name, view, rank in policies}
    for relation_name, (policy_view, rank) in policy_by_name.items():
        source_relation = source_relations[relation_name]
        target_relation = target_relations[relation_name]
        if _string(source_relation.get("view"), "source relation view") != policy_view:
            raise RelationalSpectralDistanceError("relation view does not match policy view")
        if _string(target_relation.get("view"), "target relation view") != policy_view:
            raise RelationalSpectralDistanceError("relation view does not match policy view")
        distance = relation_profile_distance(
            source_relation,
            target_relation,
            rank,
        )
        by_view[policy_view].append(distance)

    for view in _VIEWS:
        if not by_view[view]:
            raise RelationalSpectralDistanceError(f"policy sequence has no admitted {view} relations")

    return {
        view: float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))
        for view, values in by_view.items()
    }


def state_profile_view_distances(
    source_state_profile: Mapping[str, object],
    target_state_profile: Mapping[str, object],
    admitted_relation_policies: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compatibility alias for per-view RMS distance computation."""

    return state_profile_view_rms_distances(
        source_state_profile,
        target_state_profile,
        admitted_relation_policies,
    )


def compute_fold_state_profile_view_rms_distances(
    source_state_profile: Mapping[str, object],
    target_state_profile: Mapping[str, object],
    admitted_relation_policies: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compatibility wrapper for fold-level state profile distance construction."""

    return state_profile_view_rms_distances(
        source_state_profile,
        target_state_profile,
        admitted_relation_policies,
    )


def per_view_state_profile_distance(
    source_state_profile: Mapping[str, object],
    target_state_profile: Mapping[str, object],
    admitted_relation_policies: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compatibility alias for fold-level state-profile RMS distances."""

    return state_profile_view_rms_distances(
        source_state_profile,
        target_state_profile,
        admitted_relation_policies,
    )


__all__ = [
    "RelationalSpectralDistanceError",
    "validate_frozen_relation_profile",
    "validate_frozen_relation_profile_and_selected_rank",
    "build_relation_mass_vector",
    "build_relation_rank_mass",
    "hellinger_distance",
    "relation_distance",
    "relation_profile_distance",
    "state_profile_view_distances",
    "compute_layer_transport_relation_distance",
    "state_profile_view_rms_distances",
    "per_view_state_profile_distance",
    "compute_fold_state_profile_view_rms_distances",
    "build_relation_mass",
]
