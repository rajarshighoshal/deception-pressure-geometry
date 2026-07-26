"""Sparse partial-isometry connections between treatment-blind token cores.

The connection in this module is a known token correspondence, not an inferred
transport plan.  It is therefore represented only by integer index pairs.  The
module intentionally contains no distance, coordinate, PCA, or global-chart API.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np


class PartialConnectionError(ValueError):
    """Raised when a canonical core or sparse connection is invalid."""


def _integer_tuple(
    values: Sequence[int],
    *,
    name: str,
    unique: bool = False,
) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise PartialConnectionError(f"{name} must be a sequence of integers") from error
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in raw):
        raise PartialConnectionError(f"{name} must contain only integers")
    result = tuple(int(value) for value in raw)
    if unique and len(set(result)) != len(result):
        raise PartialConnectionError(f"{name} must be unique")
    return result


def _hashable_tuple(
    values: Sequence[Hashable],
    *,
    name: str,
    non_empty: bool = False,
    unique: bool = False,
) -> tuple[Hashable, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise PartialConnectionError(f"{name} must be a sequence") from error
    if non_empty and not result:
        raise PartialConnectionError(f"{name} must be non-empty")
    for value in result:
        if value is None:
            raise PartialConnectionError(f"{name} may not contain None placeholders")
        try:
            hash(value)
        except TypeError as error:
            raise PartialConnectionError(f"{name} must contain hashable exact keys") from error
    if unique and len(set(result)) != len(result):
        raise PartialConnectionError(f"{name} must be unique")
    return result


def _metadata_tuple(
    values: Sequence[tuple[str, Hashable]],
) -> tuple[tuple[str, Hashable], ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise PartialConnectionError("correspondence metadata must contain key/value pairs") from error
    if not raw:
        raise PartialConnectionError("correspondence metadata must be non-empty")
    result: list[tuple[str, Hashable]] = []
    for item in raw:
        try:
            key, value = item
        except (TypeError, ValueError) as error:
            raise PartialConnectionError(
                "correspondence metadata must contain key/value pairs"
            ) from error
        if not isinstance(key, str) or not key:
            raise PartialConnectionError(
                "correspondence metadata keys must be non-empty strings"
            )
        try:
            hash(value)
        except TypeError as error:
            raise PartialConnectionError(
                "correspondence metadata values must be exact hashable values"
            ) from error
        result.append((key, value))
    if len({key for key, _ in result}) != len(result):
        raise PartialConnectionError("correspondence metadata keys must be unique")
    return tuple(sorted(result, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class CanonicalTokenCore:
    """One state's injection of explicit canonical keys into raw token indices.

    ``canonical_keys[i]`` names the retained token at ``token_indices[i]``.
    Keys are caller-supplied and are never derived from treatment-bearing text.
    ``token_type_keys`` covers the complete raw token sequence so permutation
    covariance can be checked without assuming a particular annotation schema.
    """

    object_id: str
    token_ids: tuple[int, ...]
    token_type_keys: tuple[Hashable, ...]
    canonical_keys: tuple[Hashable, ...]
    token_indices: tuple[int, ...]
    anchor_index: int
    correspondence_metadata: tuple[tuple[str, Hashable], ...]
    excluded_treatment_indices: tuple[int, ...] = ()
    excluded_action_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise PartialConnectionError("object_id must be a non-empty string")
        token_ids = _integer_tuple(self.token_ids, name="token_ids")
        if not token_ids or any(value < 0 for value in token_ids):
            raise PartialConnectionError("token_ids must be non-empty and non-negative")
        token_type_keys = _hashable_tuple(
            self.token_type_keys,
            name="token_type_keys",
            non_empty=True,
        )
        if len(token_type_keys) != len(token_ids):
            raise PartialConnectionError(
                "token_type_keys must cover the complete raw token sequence"
            )
        canonical_keys = _hashable_tuple(
            self.canonical_keys,
            name="canonical_keys",
            non_empty=True,
            unique=True,
        )
        token_indices = _integer_tuple(
            self.token_indices,
            name="token_indices",
            unique=True,
        )
        if len(canonical_keys) != len(token_indices):
            raise PartialConnectionError(
                "canonical_keys and token_indices must define one bijection"
            )
        if any(index < 0 or index >= len(token_ids) for index in token_indices):
            raise PartialConnectionError("token_indices contain an out-of-range index")
        if not isinstance(self.anchor_index, Integral) or isinstance(self.anchor_index, bool):
            raise PartialConnectionError("anchor_index must be an integer")
        anchor_index = int(self.anchor_index)
        if anchor_index < 0 or anchor_index >= len(token_ids):
            raise PartialConnectionError("anchor_index is outside the raw token sequence")
        if anchor_index not in set(token_indices):
            raise PartialConnectionError("the exact action anchor must be retained")
        treatment = _integer_tuple(
            self.excluded_treatment_indices,
            name="excluded_treatment_indices",
            unique=True,
        )
        action = _integer_tuple(
            self.excluded_action_indices,
            name="excluded_action_indices",
            unique=True,
        )
        for name, excluded in (
            ("excluded_treatment_indices", treatment),
            ("excluded_action_indices", action),
        ):
            if any(index < 0 or index >= len(token_ids) for index in excluded):
                raise PartialConnectionError(f"{name} contain an out-of-range index")
            overlap = set(excluded) & set(token_indices)
            if overlap:
                raise PartialConnectionError(f"{name} must be absent from the canonical core")
        if set(treatment) & set(action):
            raise PartialConnectionError(
                "treatment and sampled-action exclusion sets must be disjoint"
            )
        metadata = _metadata_tuple(self.correspondence_metadata)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "token_type_keys", token_type_keys)
        object.__setattr__(self, "canonical_keys", canonical_keys)
        object.__setattr__(self, "token_indices", token_indices)
        object.__setattr__(self, "anchor_index", anchor_index)
        object.__setattr__(self, "correspondence_metadata", metadata)
        object.__setattr__(self, "excluded_treatment_indices", tuple(sorted(treatment)))
        object.__setattr__(self, "excluded_action_indices", tuple(sorted(action)))

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def rank(self) -> int:
        return len(self.canonical_keys)

    @property
    def key_to_index(self) -> dict[Hashable, int]:
        return dict(zip(self.canonical_keys, self.token_indices, strict=True))

    @property
    def anchor_key(self) -> Hashable:
        return self.canonical_keys[self.token_indices.index(self.anchor_index)]


@dataclass(frozen=True, slots=True)
class PartialIsometryEdge:
    """A rectangular partial permutation stored as exact integer index pairs."""

    source: CanonicalTokenCore
    target: CanonicalTokenCore
    canonical_keys: tuple[Hashable, ...]
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, CanonicalTokenCore) or not isinstance(
            self.target, CanonicalTokenCore
        ):
            raise PartialConnectionError(
                "source and target must be validated CanonicalTokenCore objects"
            )
        if self.source.correspondence_metadata != self.target.correspondence_metadata:
            raise PartialConnectionError("wrong-orbit correspondence metadata mismatch")
        keys = _hashable_tuple(
            self.canonical_keys,
            name="edge canonical_keys",
            non_empty=True,
            unique=True,
        )
        source_indices = _integer_tuple(
            self.source_indices,
            name="source_indices",
            unique=True,
        )
        target_indices = _integer_tuple(
            self.target_indices,
            name="target_indices",
            unique=True,
        )
        if len(keys) != len(source_indices) or len(keys) != len(target_indices):
            raise PartialConnectionError(
                "edge keys and index lists must define one rectangular bijection"
            )
        source_by_key = self.source.key_to_index
        target_by_key = self.target.key_to_index
        for key, source_index, target_index in zip(
            keys, source_indices, target_indices, strict=True
        ):
            if key not in source_by_key or key not in target_by_key:
                raise PartialConnectionError("edge key is absent from an endpoint core")
            if source_by_key[key] != source_index or target_by_key[key] != target_index:
                raise PartialConnectionError(
                    "edge indices do not equal the endpoint canonical injections"
                )
            if self.source.token_ids[source_index] != self.target.token_ids[target_index]:
                raise PartialConnectionError(
                    "matched canonical keys have different exact token IDs"
                )
            if (
                self.source.token_type_keys[source_index]
                != self.target.token_type_keys[target_index]
            ):
                raise PartialConnectionError(
                    "matched canonical keys have different token types"
                )
        if self.source.anchor_key != self.target.anchor_key:
            raise PartialConnectionError("endpoint action anchors have different canonical keys")
        if self.source.anchor_key not in set(keys):
            raise PartialConnectionError("the action anchor must survive every connection edge")
        object.__setattr__(self, "canonical_keys", keys)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "target_indices", target_indices)

    @property
    def rank(self) -> int:
        return len(self.canonical_keys)

    @property
    def index_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.source_indices, self.target_indices, strict=True))

    @property
    def source_to_target(self) -> dict[int, int]:
        return dict(self.index_pairs)

    @property
    def source_projector_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.source_indices))

    @property
    def target_projector_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.target_indices))


def build_canonical_token_core(
    *,
    object_id: str,
    token_ids: Sequence[int],
    token_type_keys: Sequence[Hashable],
    canonical_keys: Sequence[Hashable],
    token_indices: Sequence[int],
    anchor_index: int,
    correspondence_metadata: Mapping[str, Hashable],
    excluded_treatment_indices: Sequence[int] = (),
    excluded_action_indices: Sequence[int] = (),
) -> CanonicalTokenCore:
    """Build a core from caller-supplied exact keys and treatment-blind metadata."""

    if not isinstance(correspondence_metadata, Mapping):
        raise PartialConnectionError("correspondence_metadata must be a mapping")
    return CanonicalTokenCore(
        object_id=object_id,
        token_ids=tuple(token_ids),
        token_type_keys=tuple(token_type_keys),
        canonical_keys=tuple(canonical_keys),
        token_indices=tuple(token_indices),
        anchor_index=anchor_index,
        correspondence_metadata=tuple(correspondence_metadata.items()),
        excluded_treatment_indices=tuple(excluded_treatment_indices),
        excluded_action_indices=tuple(excluded_action_indices),
    )


def build_partial_isometry_edge(
    source: CanonicalTokenCore,
    target: CanonicalTokenCore,
) -> PartialIsometryEdge:
    """Match the complete exact-key intersection of two compatible cores."""

    if source.correspondence_metadata != target.correspondence_metadata:
        raise PartialConnectionError("wrong-orbit correspondence metadata mismatch")
    target_keys = set(target.canonical_keys)
    keys = tuple(key for key in source.canonical_keys if key in target_keys)
    if not keys:
        raise PartialConnectionError("endpoint cores have no surviving canonical keys")
    source_by_key = source.key_to_index
    target_by_key = target.key_to_index
    return PartialIsometryEdge(
        source=source,
        target=target,
        canonical_keys=keys,
        source_indices=tuple(source_by_key[key] for key in keys),
        target_indices=tuple(target_by_key[key] for key in keys),
    )


def identity_partial_isometry(core: CanonicalTokenCore) -> PartialIsometryEdge:
    """Return the identity on a canonical core's retained support."""

    return PartialIsometryEdge(
        source=core,
        target=core,
        canonical_keys=core.canonical_keys,
        source_indices=core.token_indices,
        target_indices=core.token_indices,
    )


def reverse_partial_isometry(edge: PartialIsometryEdge) -> PartialIsometryEdge:
    """Return the exact inverse partial isometry on the same canonical keys."""

    return PartialIsometryEdge(
        source=edge.target,
        target=edge.source,
        canonical_keys=edge.canonical_keys,
        source_indices=edge.target_indices,
        target_indices=edge.source_indices,
    )


def compose_partial_isometries(
    first: PartialIsometryEdge,
    second: PartialIsometryEdge,
) -> PartialIsometryEdge:
    """Compose ``source -> middle -> target`` on the surviving key intersection."""

    if first.target != second.source:
        raise PartialConnectionError(
            "connection composition requires the exact same validated middle core"
        )
    second_keys = set(second.canonical_keys)
    keys = tuple(key for key in first.canonical_keys if key in second_keys)
    if not keys:
        raise PartialConnectionError("connection composition has empty surviving support")
    first_source = dict(zip(first.canonical_keys, first.source_indices, strict=True))
    second_target = dict(zip(second.canonical_keys, second.target_indices, strict=True))
    return PartialIsometryEdge(
        source=first.source,
        target=second.target,
        canonical_keys=keys,
        source_indices=tuple(first_source[key] for key in keys),
        target_indices=tuple(second_target[key] for key in keys),
    )


def validate_partial_isometry_identities(edge: PartialIsometryEdge) -> None:
    """Validate ``Gamma* Gamma`` and ``Gamma Gamma*`` as sparse projectors.

    The validation composes integer maps only.  It never materializes a dense
    partial-permutation or projector matrix.
    """

    forward = edge.source_to_target
    backward = reverse_partial_isometry(edge).source_to_target
    source_projection = {index: backward[target] for index, target in forward.items()}
    target_projection = {index: forward[source] for index, source in backward.items()}
    expected_source = {index: index for index in edge.source_projector_indices}
    expected_target = {index: index for index in edge.target_projector_indices}
    if source_projection != expected_source:
        raise PartialConnectionError("Gamma* Gamma is not the source-support projector")
    if target_projection != expected_target:
        raise PartialConnectionError("Gamma Gamma* is not the target-support projector")


def _validated_type_preserving_permutation(
    core: CanonicalTokenCore,
    permutation: Sequence[int],
) -> tuple[int, ...]:
    result = _integer_tuple(permutation, name="permutation", unique=True)
    if len(result) != core.n_tokens or set(result) != set(range(core.n_tokens)):
        raise PartialConnectionError(
            "permutation must be an old-index to new-index bijection"
        )
    if any(
        core.token_type_keys[old_index] != core.token_type_keys[new_index]
        for old_index, new_index in enumerate(result)
    ):
        raise PartialConnectionError("permutation must preserve every token type")
    return result


def permute_canonical_token_core(
    core: CanonicalTokenCore,
    permutation: Sequence[int],
) -> CanonicalTokenCore:
    """Reindex a core by a checked type-preserving ``old -> new`` permutation."""

    old_to_new = _validated_type_preserving_permutation(core, permutation)
    token_ids = [0] * core.n_tokens
    token_types: list[Hashable | None] = [None] * core.n_tokens
    for old_index, new_index in enumerate(old_to_new):
        token_ids[new_index] = core.token_ids[old_index]
        token_types[new_index] = core.token_type_keys[old_index]
    return CanonicalTokenCore(
        object_id=core.object_id,
        token_ids=tuple(token_ids),
        token_type_keys=tuple(token_types),
        canonical_keys=core.canonical_keys,
        token_indices=tuple(old_to_new[index] for index in core.token_indices),
        anchor_index=old_to_new[core.anchor_index],
        correspondence_metadata=core.correspondence_metadata,
        excluded_treatment_indices=tuple(
            old_to_new[index] for index in core.excluded_treatment_indices
        ),
        excluded_action_indices=tuple(
            old_to_new[index] for index in core.excluded_action_indices
        ),
    )


def permute_partial_isometry_edge(
    edge: PartialIsometryEdge,
    *,
    source_permutation: Sequence[int],
    target_permutation: Sequence[int],
) -> PartialIsometryEdge:
    """Apply endpoint reindexings and return the covariant sparse edge."""

    source_old_to_new = _validated_type_preserving_permutation(
        edge.source, source_permutation
    )
    target_old_to_new = _validated_type_preserving_permutation(
        edge.target, target_permutation
    )
    source = permute_canonical_token_core(edge.source, source_old_to_new)
    target = permute_canonical_token_core(edge.target, target_old_to_new)
    return PartialIsometryEdge(
        source=source,
        target=target,
        canonical_keys=edge.canonical_keys,
        source_indices=tuple(source_old_to_new[index] for index in edge.source_indices),
        target_indices=tuple(target_old_to_new[index] for index in edge.target_indices),
    )


def restrict_residual_section(
    residual: np.ndarray,
    core: CanonicalTokenCore,
) -> np.ndarray:
    """Slice a raw ``(token, hidden)`` residual section in canonical-key order."""

    values = np.asarray(residual)
    if values.ndim != 2 or values.shape[0] != core.n_tokens or values.shape[1] < 1:
        raise PartialConnectionError(
            "residual must have shape (complete_raw_token_count, hidden_size)"
        )
    try:
        finite = bool(np.isfinite(values).all())
    except TypeError as error:
        raise PartialConnectionError("residual must be numeric") from error
    if not finite:
        raise PartialConnectionError("residual contains non-finite values")
    return np.take(values, core.token_indices, axis=0).copy()


def _dense_causal_attention(
    attention: np.ndarray,
    *,
    n_tokens: int,
) -> np.ndarray:
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim >= 2 and values.shape[-2:] == (n_tokens, n_tokens):
        return values.copy()
    packed_size = n_tokens * (n_tokens + 1) // 2
    if values.ndim >= 1 and values.shape[-1] == packed_size:
        dense = np.zeros((*values.shape[:-1], n_tokens, n_tokens), dtype=np.float64)
        rows, columns = np.tril_indices(n_tokens)
        dense[..., rows, columns] = values
        return dense
    raise PartialConnectionError(
        "attention must be dense (..., token, token) or packed lower-triangular"
    )


def restrict_causal_attention_section(
    attention: np.ndarray,
    core: CanonicalTokenCore,
    *,
    row_sum_tolerance: float = 0.02,
    causal_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict and renormalize causal attention on a canonical token core.

    Returns ``(renormalized, retained_mass, removed_mass)``.  The two mass arrays
    have shape ``attention_leading_dimensions + (core.rank,)`` and describe the
    original full-row mass before renormalization.
    """

    if not np.isfinite(row_sum_tolerance) or row_sum_tolerance <= 0:
        raise PartialConnectionError("row_sum_tolerance must be positive and finite")
    if not np.isfinite(causal_tolerance) or causal_tolerance < 0:
        raise PartialConnectionError("causal_tolerance must be non-negative and finite")
    dense = _dense_causal_attention(attention, n_tokens=core.n_tokens)
    if not np.isfinite(dense).all() or np.any(dense < 0):
        raise PartialConnectionError("attention must contain finite non-negative weights")
    rows, columns = np.triu_indices(core.n_tokens, k=1)
    if rows.size and np.any(dense[..., rows, columns] > causal_tolerance):
        raise PartialConnectionError("attention contains non-causal upper-triangular mass")
    row_mass = np.sum(dense, axis=-1)
    if np.max(np.abs(row_mass - 1.0)) > row_sum_tolerance:
        raise PartialConnectionError("attention violates the row-stochastic contract")
    indices = np.asarray(core.token_indices, dtype=np.int64)
    restricted = np.take(np.take(dense, indices, axis=-2), indices, axis=-1)
    retained_mass = np.sum(restricted, axis=-1)
    selected_full_mass = np.take(row_mass, indices, axis=-1)
    removed_mass = selected_full_mass - retained_mass
    scale = max(1.0, float(np.max(np.abs(selected_full_mass))))
    if np.any(removed_mass < -1e-12 * scale):
        raise PartialConnectionError("restricted attention mass exceeds the full-row mass")
    removed_mass = np.maximum(removed_mass, 0.0)
    if np.any(retained_mass <= 0):
        raise PartialConnectionError(
            "a retained attention row has no mass on retained causal keys"
        )
    normalized = restricted / retained_mass[..., None]
    return normalized, retained_mass, removed_mass


def restrict_layer_transport_section(
    layer_residuals: Mapping[int, np.ndarray],
    core: CanonicalTokenCore,
) -> dict[int, np.ndarray]:
    """Restrict every raw residual layer on the identical canonical indices."""

    if not isinstance(layer_residuals, Mapping) or len(layer_residuals) < 2:
        raise PartialConnectionError(
            "layer transport requires at least two raw residual layers"
        )
    normalized: dict[int, np.ndarray] = {}
    raw_shape: tuple[int, int] | None = None
    for raw_layer, residual in layer_residuals.items():
        if not isinstance(raw_layer, Integral) or isinstance(raw_layer, bool):
            raise PartialConnectionError("layer identities must be integers")
        layer = int(raw_layer)
        if layer in normalized:
            raise PartialConnectionError("layer identities must be unique")
        values = np.asarray(residual)
        if raw_shape is None:
            raw_shape = values.shape
        elif values.shape != raw_shape:
            raise PartialConnectionError(
                "all layer residuals must share one raw token/hidden shape"
            )
        normalized[layer] = restrict_residual_section(values, core)
    return {layer: normalized[layer] for layer in sorted(normalized)}


__all__ = [
    "CanonicalTokenCore",
    "PartialConnectionError",
    "PartialIsometryEdge",
    "build_canonical_token_core",
    "build_partial_isometry_edge",
    "compose_partial_isometries",
    "identity_partial_isometry",
    "permute_canonical_token_core",
    "permute_partial_isometry_edge",
    "restrict_causal_attention_section",
    "restrict_layer_transport_section",
    "restrict_residual_section",
    "reverse_partial_isometry",
    "validate_partial_isometry_identities",
]
