"""Typed fused Gromov--Wasserstein distances for relational capture banks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np
import ot


class RelationalMetricError(ValueError):
    """Raised when a relational distance cannot be certified under the frozen solver contract."""


@dataclass(frozen=True)
class RelationalMetricConfig:
    layers: tuple[int, ...] = (12, 16, 19, 20)
    relation_weight: float = 0.8
    typed_weight: float = 0.2
    epsilon: float = 0.05
    outer_max_iter: int = 1000
    outer_tolerance: float = 1e-9
    inner_max_iter: int = 10000
    inner_tolerance: float = 1e-7
    marginal_tolerance: float = 1e-7
    attention_row_sum_tolerance: float = 0.02
    hidden_relation_kind: str = "centered_euclidean"

    def validate(self) -> None:
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise RelationalMetricError("layers must be non-empty and unique")
        if any(layer < 1 for layer in self.layers):
            raise RelationalMetricError("layers must be positive")
        if not np.isclose(self.relation_weight + self.typed_weight, 1.0):
            raise RelationalMetricError("relation and typed weights must sum to one")
        if self.relation_weight <= 0 or self.typed_weight <= 0:
            raise RelationalMetricError("relation and typed weights must both be positive")
        if self.epsilon <= 0:
            raise RelationalMetricError("epsilon must be positive")
        if self.hidden_relation_kind not in {"centered_euclidean", "cosine"}:
            raise RelationalMetricError("hidden relation kind is unknown")
        if self.outer_max_iter < 1 or self.inner_max_iter < 1:
            raise RelationalMetricError("solver iteration limits must be positive")
        if min(
            self.outer_tolerance,
            self.inner_tolerance,
            self.marginal_tolerance,
            self.attention_row_sum_tolerance,
        ) <= 0:
            raise RelationalMetricError("solver and validation tolerances must be positive")


@dataclass(frozen=True)
class TypedRelationalObject:
    object_id: str
    residuals: Mapping[int, np.ndarray]
    attentions: Mapping[int, np.ndarray]
    role_ids: np.ndarray
    turn_ids: np.ndarray
    span_flags: np.ndarray
    normalized_positions: np.ndarray | None = None


@dataclass(frozen=True)
class RelationalStructure:
    object_id: str
    relations: Mapping[int, np.ndarray]
    attentions: Mapping[int, np.ndarray]
    role_ids: np.ndarray
    turn_ids: np.ndarray
    span_flags: np.ndarray
    normalized_positions: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(self.role_ids.shape[0])


@dataclass(frozen=True)
class LayerRelationalDistance:
    layer: int
    relation_loss: float
    typed_loss: float
    hidden_fused_loss: float
    attention_loss: float
    solver_error: float
    row_marginal_error: float
    column_marginal_error: float
    inner_warning_count: int
    symmetry_error: float


@dataclass(frozen=True)
class RelationalComponentDistance:
    hidden: float
    attention: float
    relation: float
    typed: float
    layers: tuple[LayerRelationalDistance, ...]


@dataclass(frozen=True)
class RawLayerRelationalDistance:
    layer: int
    relation_loss: float
    typed_loss: float
    hidden_fused_loss: float
    attention_loss: float
    solver_error: float
    row_marginal_error: float
    column_marginal_error: float
    inner_warning_count: int


@dataclass(frozen=True)
class RelationalSelfDistance:
    object_id: str
    config: RelationalMetricConfig
    layers: tuple[RawLayerRelationalDistance, ...]


@dataclass(frozen=True)
class RelationalDistanceScaler:
    hidden_median_positive: float
    attention_median_positive: float
    hidden_weight: float = 0.5
    attention_weight: float = 0.5

    @classmethod
    def fit(
        cls,
        distances: Sequence[RelationalComponentDistance],
        *,
        hidden_weight: float = 0.5,
        attention_weight: float = 0.5,
    ) -> RelationalDistanceScaler:
        if not distances:
            raise RelationalMetricError("cannot fit component scales without training distances")
        if not np.isclose(hidden_weight + attention_weight, 1.0):
            raise RelationalMetricError("component fusion weights must sum to one")
        hidden = _median_positive([distance.hidden for distance in distances], "hidden")
        attention = _median_positive(
            [distance.attention for distance in distances], "attention"
        )
        return cls(
            hidden_median_positive=hidden,
            attention_median_positive=attention,
            hidden_weight=hidden_weight,
            attention_weight=attention_weight,
        )

    def transform(self, distance: RelationalComponentDistance) -> float:
        values = np.asarray([distance.hidden, distance.attention], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RelationalMetricError("component distances must be finite and non-negative")
        return float(
            self.hidden_weight * distance.hidden / self.hidden_median_positive
            + self.attention_weight
            * distance.attention
            / self.attention_median_positive
        )


def _median_positive(values: Sequence[float], name: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    positive = array[np.isfinite(array) & (array > 0)]
    if positive.size == 0:
        raise RelationalMetricError(f"training-fold {name} distances have no positive scale")
    return float(np.median(positive))


def _as_vector(values: np.ndarray, *, name: str, n_tokens: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (n_tokens is not None and array.shape[0] != n_tokens):
        expected = "a one-dimensional token vector"
        if n_tokens is not None:
            expected += f" of length {n_tokens}"
        raise RelationalMetricError(f"{name} must be {expected}")
    return array


def _unpack_attention(values: np.ndarray, n_tokens: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        if array.shape[1:] != (n_tokens, n_tokens):
            raise RelationalMetricError("dense attention shape differs from token count")
        return np.asarray(array, dtype=np.float64)
    expected_edges = n_tokens * (n_tokens + 1) // 2
    if array.ndim != 2 or array.shape[1] != expected_edges:
        raise RelationalMetricError("packed attention shape differs from token count")
    dense = np.zeros((array.shape[0], n_tokens, n_tokens), dtype=np.float64)
    rows, columns = np.tril_indices(n_tokens)
    dense[:, rows, columns] = np.asarray(array, dtype=np.float64)
    return dense


def _median_scale_relation(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    cost = 0.5 * (cost + cost.T)
    np.fill_diagonal(cost, 0.0)
    positive = cost[np.triu_indices(cost.shape[0], k=1)]
    positive = positive[positive > 0]
    if positive.size == 0:
        raise RelationalMetricError("residual relation has no positive off-diagonal scale")
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale < 1e-6:
        raise RelationalMetricError("residual relation has an invalid median-positive scale")
    return cost / scale


def _scaled_centered_euclidean_relation(residual: np.ndarray) -> np.ndarray:
    vectors = np.asarray(residual, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 1:
        raise RelationalMetricError("residuals must contain at least two token vectors")
    if not np.isfinite(vectors).all():
        raise RelationalMetricError("residuals contain non-finite values")
    centered = np.asarray(vectors, dtype=np.float64)
    centered -= np.mean(centered, axis=0, dtype=np.float64, keepdims=True)
    gram = centered @ centered.T
    squared = np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(squared))))
    if np.any(squared < -tolerance) or not np.isfinite(squared).all():
        raise RelationalMetricError("centered residual distances are invalid")
    return _median_scale_relation(np.sqrt(np.maximum(squared, 0.0)))


def _scaled_cosine_relation(residual: np.ndarray) -> np.ndarray:
    vectors = np.asarray(residual, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] < 2 or vectors.shape[1] < 1:
        raise RelationalMetricError("residuals must contain at least two token vectors")
    if not np.isfinite(vectors).all():
        raise RelationalMetricError("residuals contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise RelationalMetricError("residuals contain a zero-norm token vector")
    normalized = vectors / norms[:, None]
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return _median_scale_relation(np.asarray(1.0 - cosine, dtype=np.float64))


def build_relational_structure(
    value: TypedRelationalObject,
    config: RelationalMetricConfig = RelationalMetricConfig(),
) -> RelationalStructure:
    config.validate()
    if not value.object_id:
        raise RelationalMetricError("object_id must be non-empty")
    role_ids = _as_vector(value.role_ids, name="role_ids").astype(np.int16, copy=True)
    n_tokens = int(role_ids.shape[0])
    if n_tokens < 2:
        raise RelationalMetricError("relational objects require at least two tokens")
    turn_ids = _as_vector(value.turn_ids, name="turn_ids", n_tokens=n_tokens).astype(
        np.int16, copy=True
    )
    span_flags = _as_vector(
        value.span_flags, name="span_flags", n_tokens=n_tokens
    ).astype(np.int16, copy=True)
    if np.any((role_ids < 0) | (role_ids > 2)):
        raise RelationalMetricError("role_ids contain an unknown role")
    if np.any((turn_ids < -1) | (turn_ids > 3)):
        raise RelationalMetricError("turn_ids contain an unknown turn")
    if np.any((span_flags < 0) | (span_flags > 63)):
        raise RelationalMetricError("span_flags contain an unknown six-bit span mask")
    if set(value.residuals) != set(config.layers):
        raise RelationalMetricError("residual layer set differs from the metric contract")
    if set(value.attentions) != set(config.layers):
        raise RelationalMetricError("attention layer set differs from the metric contract")

    relations: dict[int, np.ndarray] = {}
    attentions: dict[int, np.ndarray] = {}
    expected_heads: int | None = None
    for layer in config.layers:
        residual = np.asarray(value.residuals[layer])
        if residual.ndim != 2 or residual.shape[0] != n_tokens:
            raise RelationalMetricError(f"layer {layer} residual shape differs from token count")
        if config.hidden_relation_kind == "centered_euclidean":
            relations[layer] = _scaled_centered_euclidean_relation(residual)
        else:
            relations[layer] = _scaled_cosine_relation(residual)
        dense_attention = _unpack_attention(value.attentions[layer], n_tokens)
        if expected_heads is None:
            expected_heads = int(dense_attention.shape[0])
        if dense_attention.shape[0] != expected_heads or expected_heads < 1:
            raise RelationalMetricError("attention head count differs across layers")
        if not np.isfinite(dense_attention).all() or np.any(dense_attention < 0):
            raise RelationalMetricError("attention contains negative or non-finite values")
        if np.any(np.triu(dense_attention, k=1) != 0):
            raise RelationalMetricError("attention contains future-token mass")
        row_sums = dense_attention.sum(axis=-1)
        if not np.allclose(
            row_sums,
            np.ones_like(row_sums),
            atol=config.attention_row_sum_tolerance,
            rtol=0.0,
        ):
            raise RelationalMetricError("attention row sums exceed the metric tolerance")
        attentions[layer] = dense_attention
    if value.normalized_positions is None:
        normalized_positions = np.arange(n_tokens, dtype=np.float64) / max(
            n_tokens - 1, 1
        )
    else:
        normalized_positions = _as_vector(
            value.normalized_positions,
            name="normalized_positions",
            n_tokens=n_tokens,
        ).astype(np.float64, copy=True)
        if (
            not np.isfinite(normalized_positions).all()
            or np.any(normalized_positions < 0)
            or np.any(normalized_positions > 1)
            or np.any(np.diff(normalized_positions) <= 0)
        ):
            raise RelationalMetricError(
                "normalized_positions must be finite, strictly increasing, and within [0, 1]"
            )
    return RelationalStructure(
        object_id=value.object_id,
        relations=relations,
        attentions=attentions,
        role_ids=role_ids,
        turn_ids=turn_ids,
        span_flags=span_flags,
        normalized_positions=normalized_positions,
    )


def typed_feature_cost(left: RelationalStructure, right: RelationalStructure) -> np.ndarray:
    role_mismatch = left.role_ids[:, None] != right.role_ids[None, :]
    turn_mismatch = left.turn_ids[:, None] != right.turn_ids[None, :]
    xor = np.bitwise_xor(left.span_flags[:, None], right.span_flags[None, :]).astype(
        np.uint8
    )
    bit_counts = np.unpackbits(xor[..., None], axis=-1, bitorder="little")[..., :6].sum(
        axis=-1
    )
    span_fraction = bit_counts.astype(np.float64) / 6.0
    position_difference = np.abs(
        left.normalized_positions[:, None] - right.normalized_positions[None, :]
    )
    return (
        2.0 * role_mismatch.astype(np.float64)
        + turn_mismatch.astype(np.float64)
        + 2.0 * span_fraction
        + position_difference
    ) / 6.0


def quadratic_relation_loss(
    left: np.ndarray,
    right: np.ndarray,
    coupling: np.ndarray,
    left_mass: np.ndarray,
    right_mass: np.ndarray,
) -> float:
    left_term = float(np.sum(np.square(left) * np.outer(left_mass, left_mass)))
    right_term = float(np.sum(np.square(right) * np.outer(right_mass, right_mass)))
    cross = float(np.sum(coupling * (left @ coupling @ right.T)))
    value = left_term + right_term - 2.0 * cross
    tolerance = 1e-10 * max(1.0, abs(left_term) + abs(right_term))
    if value < -tolerance or not np.isfinite(value):
        raise RelationalMetricError("quadratic relational loss is negative or non-finite")
    return max(0.0, value)


def _solve_layer_coupling(
    typed_cost: np.ndarray,
    left_relation: np.ndarray,
    right_relation: np.ndarray,
    config: RelationalMetricConfig,
) -> tuple[np.ndarray, float, float, float, int]:
    left_mass = np.full(left_relation.shape[0], 1.0 / left_relation.shape[0])
    right_mass = np.full(right_relation.shape[0], 1.0 / right_relation.shape[0])
    initial = np.outer(left_mass, right_mass)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coupling, log = ot.gromov.entropic_fused_gromov_wasserstein(
            typed_cost,
            left_relation,
            right_relation,
            p=left_mass,
            q=right_mass,
            loss_fun="square_loss",
            epsilon=config.epsilon,
            symmetric=True,
            alpha=config.relation_weight,
            G0=initial,
            max_iter=config.outer_max_iter,
            tol=config.outer_tolerance,
            solver="PGD",
            warmstart=False,
            verbose=False,
            log=True,
            numItermax=config.inner_max_iter,
            stopThr=config.inner_tolerance,
            warn=True,
        )
    warning_messages = [str(item.message) for item in caught]
    fatal_warnings = [
        message
        for message in warning_messages
        if "Sinkhorn did not converge" not in message
    ]
    if fatal_warnings:
        raise RelationalMetricError(
            "fused GW solver warning: " + "; ".join(fatal_warnings)
        )
    coupling = np.asarray(coupling, dtype=np.float64)
    errors = np.asarray(log.get("err", []), dtype=np.float64)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise RelationalMetricError("fused GW solver did not return a finite convergence trace")
    solver_error = float(errors[-1])
    if solver_error > config.outer_tolerance:
        raise RelationalMetricError(
            f"fused GW solver did not converge: final error {solver_error}"
        )
    if (
        coupling.shape != (left_relation.shape[0], right_relation.shape[0])
        or not np.isfinite(coupling).all()
        or np.any(coupling < 0)
    ):
        raise RelationalMetricError("fused GW solver returned an invalid coupling")
    row_error = float(np.max(np.abs(coupling.sum(axis=1) - left_mass)))
    column_error = float(np.max(np.abs(coupling.sum(axis=0) - right_mass)))
    if max(row_error, column_error) > config.marginal_tolerance:
        raise RelationalMetricError("fused GW coupling violates its token marginals")
    return coupling, solver_error, row_error, column_error, len(warning_messages)


def relational_component_distance(
    left: RelationalStructure,
    right: RelationalStructure,
    config: RelationalMetricConfig = RelationalMetricConfig(),
    *,
    left_self: RelationalSelfDistance | None = None,
    right_self: RelationalSelfDistance | None = None,
) -> RelationalComponentDistance:
    config.validate()
    if left.object_id == right.object_id:
        if left is not right:
            raise RelationalMetricError("distinct structures may not share an object_id")
        zero_layers = tuple(
            LayerRelationalDistance(
                layer, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
            )
            for layer in config.layers
        )
        return RelationalComponentDistance(0.0, 0.0, 0.0, 0.0, zero_layers)
    if set(left.relations) != set(config.layers) or set(right.relations) != set(
        config.layers
    ):
        raise RelationalMetricError("structure layer set differs from the metric contract")
    if left.object_id > right.object_id:
        left, right = right, left
        left_self, right_self = right_self, left_self
    left_self = left_self or relational_self_distance(left, config)
    right_self = right_self or relational_self_distance(right, config)
    _validate_self_distance(left_self, left, config)
    _validate_self_distance(right_self, right, config)
    left_self_layers = {value.layer: value for value in left_self.layers}
    right_self_layers = {value.layer: value for value in right_self.layers}
    layer_distances: list[LayerRelationalDistance] = []
    for layer in config.layers:
        forward = _raw_layer_distance(left, right, layer=layer, config=config)
        reverse = _raw_layer_distance(right, left, layer=layer, config=config)
        cross_relation = 0.5 * (forward.relation_loss + reverse.relation_loss)
        cross_typed = 0.5 * (forward.typed_loss + reverse.typed_loss)
        cross_hidden = 0.5 * (forward.hidden_fused_loss + reverse.hidden_fused_loss)
        cross_attention = 0.5 * (forward.attention_loss + reverse.attention_loss)
        self_left_layer = left_self_layers[layer]
        self_right_layer = right_self_layers[layer]
        relation_loss = _debiased_diagnostic(
            cross_relation,
            self_left_layer.relation_loss,
            self_right_layer.relation_loss,
        )
        typed_loss = _debiased_diagnostic(
            cross_typed,
            self_left_layer.typed_loss,
            self_right_layer.typed_loss,
        )
        hidden_fused_loss = _debiased_cost(
            cross_hidden,
            self_left_layer.hidden_fused_loss,
            self_right_layer.hidden_fused_loss,
            "hidden fused",
        )
        attention_loss = _debiased_cost(
            cross_attention,
            self_left_layer.attention_loss,
            self_right_layer.attention_loss,
            "attention relation",
        )
        diagnostics = (forward, reverse, self_left_layer, self_right_layer)
        layer_distances.append(
            LayerRelationalDistance(
                layer=layer,
                relation_loss=relation_loss,
                typed_loss=typed_loss,
                hidden_fused_loss=hidden_fused_loss,
                attention_loss=attention_loss,
                solver_error=max(value.solver_error for value in diagnostics),
                row_marginal_error=max(value.row_marginal_error for value in diagnostics),
                column_marginal_error=max(
                    value.column_marginal_error for value in diagnostics
                ),
                inner_warning_count=sum(value.inner_warning_count for value in diagnostics),
                symmetry_error=max(
                    abs(forward.hidden_fused_loss - reverse.hidden_fused_loss),
                    abs(forward.attention_loss - reverse.attention_loss),
                ),
            )
        )
    return RelationalComponentDistance(
        hidden=float(np.mean([value.hidden_fused_loss for value in layer_distances])),
        attention=float(np.mean([value.attention_loss for value in layer_distances])),
        relation=float(np.mean([value.relation_loss for value in layer_distances])),
        typed=float(np.mean([value.typed_loss for value in layer_distances])),
        layers=tuple(layer_distances),
    )


def _raw_layer_distance(
    left: RelationalStructure,
    right: RelationalStructure,
    *,
    layer: int,
    config: RelationalMetricConfig,
) -> RawLayerRelationalDistance:
    typed_cost = typed_feature_cost(left, right)
    coupling, solver_error, row_error, column_error, warning_count = (
        _solve_layer_coupling(
            typed_cost, left.relations[layer], right.relations[layer], config
        )
    )
    left_mass = np.full(left.n_tokens, 1.0 / left.n_tokens)
    right_mass = np.full(right.n_tokens, 1.0 / right.n_tokens)
    relation_loss = quadratic_relation_loss(
        left.relations[layer],
        right.relations[layer],
        coupling,
        left_mass,
        right_mass,
    )
    typed_loss = float(np.sum(typed_cost * coupling))
    left_attention = left.attentions[layer]
    right_attention = right.attentions[layer]
    if left_attention.shape[0] != right_attention.shape[0]:
        raise RelationalMetricError("attention head count differs between objects")
    attention_losses = [
        quadratic_relation_loss(
            left_attention[head],
            right_attention[head],
            coupling,
            left_mass,
            right_mass,
        )
        for head in range(left_attention.shape[0])
    ]
    return RawLayerRelationalDistance(
        layer=layer,
        relation_loss=relation_loss,
        typed_loss=typed_loss,
        hidden_fused_loss=(
            config.relation_weight * relation_loss + config.typed_weight * typed_loss
        ),
        attention_loss=float(np.mean(attention_losses)),
        solver_error=solver_error,
        row_marginal_error=row_error,
        column_marginal_error=column_error,
        inner_warning_count=warning_count,
    )


def relational_self_distance(
    structure: RelationalStructure,
    config: RelationalMetricConfig = RelationalMetricConfig(),
) -> RelationalSelfDistance:
    config.validate()
    return RelationalSelfDistance(
        object_id=structure.object_id,
        config=config,
        layers=tuple(
            _raw_layer_distance(structure, structure, layer=layer, config=config)
            for layer in config.layers
        ),
    )


def _validate_self_distance(
    value: RelationalSelfDistance,
    structure: RelationalStructure,
    config: RelationalMetricConfig,
) -> None:
    if value.object_id != structure.object_id or value.config != config:
        raise RelationalMetricError("cached self-distance identity differs from the metric call")
    if tuple(layer.layer for layer in value.layers) != config.layers:
        raise RelationalMetricError("cached self-distance layers differ from the metric contract")


def _debiased_cost(cross: float, left_self: float, right_self: float, name: str) -> float:
    value = cross - 0.5 * left_self - 0.5 * right_self
    scale = max(1.0, abs(cross), abs(left_self), abs(right_self))
    if value < -1e-10 * scale or not np.isfinite(value):
        raise RelationalMetricError(f"debiased {name} cost is negative or non-finite")
    return max(0.0, value)


def _debiased_diagnostic(cross: float, left_self: float, right_self: float) -> float:
    value = cross - 0.5 * left_self - 0.5 * right_self
    if not np.isfinite(value):
        raise RelationalMetricError("debiased diagnostic component is non-finite")
    return value


def _token_type_keys(structure: RelationalStructure) -> list[tuple[int, int, int]]:
    return [
        (int(role), int(turn), int(flags))
        for role, turn, flags in zip(
            structure.role_ids, structure.turn_ids, structure.span_flags
        )
    ]


def shuffle_relational_structure(
    structure: RelationalStructure,
    *,
    seed: int,
) -> RelationalStructure:
    rng = np.random.default_rng(seed)
    token_types = _token_type_keys(structure)
    relations: dict[int, np.ndarray] = {}
    attentions: dict[int, np.ndarray] = {}
    for layer, source in structure.relations.items():
        target = np.asarray(source, dtype=np.float64).copy()
        blocks: dict[tuple[tuple[int, int, int], tuple[int, int, int]], list[tuple[int, int]]] = {}
        for row in range(structure.n_tokens):
            for column in range(row + 1, structure.n_tokens):
                endpoint_types = tuple(sorted((token_types[row], token_types[column])))
                blocks.setdefault(endpoint_types, []).append((row, column))
        for indices in blocks.values():
            values = np.asarray([target[row, column] for row, column in indices])
            values = rng.permutation(values)
            for (row, column), value in zip(indices, values):
                target[row, column] = target[column, row] = value
        np.fill_diagonal(target, 0.0)
        relations[layer] = target
    for layer, source in structure.attentions.items():
        target = np.asarray(source, dtype=np.float64).copy()
        for head in range(target.shape[0]):
            for query in range(structure.n_tokens):
                key_blocks: dict[tuple[int, int, int], list[int]] = {}
                for key in range(query + 1):
                    key_blocks.setdefault(token_types[key], []).append(key)
                for keys in key_blocks.values():
                    target[head, query, keys] = rng.permutation(
                        target[head, query, keys]
                    )
        attentions[layer] = target
    return RelationalStructure(
        object_id=f"{structure.object_id}:shuffle:{seed}",
        relations=relations,
        attentions=attentions,
        role_ids=structure.role_ids.copy(),
        turn_ids=structure.turn_ids.copy(),
        span_flags=structure.span_flags.copy(),
        normalized_positions=structure.normalized_positions.copy(),
    )
