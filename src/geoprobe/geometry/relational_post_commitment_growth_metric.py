"""Outcome-blind distances for post-commitment relational growth edges.

This module deliberately consumes relation tensors, not residual coordinates or
tokens.  The small public adapter is also the boundary that prevents labels and
future-caveat material from reaching a metric fit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import ot
import torch
from scipy.optimize import linear_sum_assignment

from geoprobe.geometry.relational_post_commitment_growth import (
    BRIDGE_LENGTH,
    LAYERS,
    PostCommitmentGrowthEdge,
)


class GrowthMetricError(ValueError):
    """Raised when a growth edge is not a safe, well-formed metric input."""


_OUTCOME_TERMS = ("outcome", "label", "target", "reward", "success", "correct", "true")
_FORBIDDEN_TERMS = (
    "token_id",
    "anchor_residual",
    "caveat_future",
    "future_caveat",
)


@dataclass(frozen=True, slots=True)
class GrowthMetricInput:
    """Only the typed relational payload used by the outcome-blind metric."""

    residual_bridge_to_base: np.ndarray
    residual_within_bridge: np.ndarray
    attention_values: np.ndarray
    attention_offsets: np.ndarray
    attention_lengths: np.ndarray
    attention_query_indices: np.ndarray
    base_annotations: np.ndarray
    bridge_annotations: np.ndarray


@dataclass(frozen=True, slots=True)
class UnpackedGrowthAttention:
    """All seven packed rows, separated into always-causal base and bridge keys."""

    base_keys: np.ndarray  # [layer, head, 7, base]
    bridge_keys: np.ndarray  # [layer, head, 6, 6], NaN above the causal diagonal


@dataclass(frozen=True, slots=True)
class GrowthMetricDistance:
    bridge_to_base: float
    within_bridge: float
    attention_head_set: float
    attention_indexed: float


@dataclass(frozen=True, slots=True)
class GrowthMetricFusion:
    """Nested primary fusion, retaining one residual and one attention vote."""

    residual: float
    attention: float
    score: float


@dataclass(frozen=True, slots=True)
class GrowthMetricScaler:
    """Fold-local robust scales; the indexed-head value remains diagnostic only."""

    bridge_to_base_scale: float
    within_bridge_scale: float
    attention_scale: float

    @classmethod
    def fit(cls, training_distances: Sequence[GrowthMetricDistance]) -> GrowthMetricScaler:
        if not training_distances:
            raise GrowthMetricError("fit requires at least one training distance")

        def scale(name: str) -> float:
            values = np.asarray([getattr(item, name) for item in training_distances], dtype=float)
            positives = values[np.isfinite(values) & (values > 0)]
            if positives.size == 0:
                raise GrowthMetricError(f"{name} training distances have no positive scale")
            return float(np.median(positives))

        return cls(scale("bridge_to_base"), scale("within_bridge"), scale("attention_head_set"))

    def components(self, distance: GrowthMetricDistance) -> GrowthMetricFusion:
        bridge = distance.bridge_to_base / self.bridge_to_base_scale
        within = distance.within_bridge / self.within_bridge_scale
        attention = distance.attention_head_set / self.attention_scale
        values = np.asarray([bridge, within, attention], dtype=float)
        if not np.isfinite(values).all():
            raise GrowthMetricError("distance is non-finite")
        residual = float(np.sqrt(np.mean(values[:2] * values[:2])))
        score = float(np.sqrt(np.mean(np.asarray([residual, attention]) ** 2)))
        return GrowthMetricFusion(residual, float(attention), score)

    def transform(self, distance: GrowthMetricDistance) -> float:
        return self.components(distance).score


def _as_numpy(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        # NumPy has no bfloat16 scalar type; attention is safely promoted for
        # CPU-side comparisons without recovering any omitted coordinates.
        array = tensor.to(torch.float32).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
    else:
        array = np.asarray(value)
    if not np.isfinite(array).all():
        raise GrowthMetricError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _annotation_matrix(edge: PostCommitmentGrowthEdge) -> np.ndarray:
    a = edge.prefix_annotations
    # Token IDs are intentionally absent.  These are stable typed structure only.
    values = (
        a.position_ids,
        a.normalized_positions,
        a.role_ids,
        a.turn_ids,
        a.message_ids,
        a.span_flags,
        a.origin_ids,
        a.origin_detail_ids,
        a.intervention_mask,
    )
    return np.column_stack([_as_numpy(value, "annotation") for value in values])


def _rejected_keys(wrapper: Mapping[str, Any]) -> None:
    for key in wrapper:
        normalized = str(key).lower()
        if any(term in normalized for term in (*_OUTCOME_TERMS, *_FORBIDDEN_TERMS)):
            raise GrowthMetricError(f"metric payload may not contain {key!r}")


def adapt_growth_edge(
    edge: PostCommitmentGrowthEdge | Mapping[str, Any] | None = None,
    *,
    residual_bridge_to_base: np.ndarray | torch.Tensor | None = None,
    residual_within_bridge: np.ndarray | torch.Tensor | None = None,
    attention_values: np.ndarray | torch.Tensor | None = None,
    attention_offsets: np.ndarray | torch.Tensor | None = None,
    attention_lengths: np.ndarray | torch.Tensor | None = None,
    attention_query_indices: np.ndarray | torch.Tensor | None = None,
    base_annotations: np.ndarray | torch.Tensor | None = None,
    bridge_annotations: np.ndarray | torch.Tensor | None = None,
) -> GrowthMetricInput:
    """Adapt one edge, or explicit label-free tensors, into the metric payload.

    A mapping is accepted only as a wrapper containing ``edge``; this catches
    accidental outcome joins at the boundary instead of silently ignoring them.
    """
    if isinstance(edge, Mapping):
        _rejected_keys(edge)
        if set(edge) != {"edge"} or not isinstance(edge["edge"], PostCommitmentGrowthEdge):
            raise GrowthMetricError("mapping metric input must contain only an edge")
        edge = edge["edge"]
    if edge is not None:
        if any(value is not None for value in (
            residual_bridge_to_base, residual_within_bridge, attention_values,
            attention_offsets, attention_lengths, attention_query_indices,
            base_annotations, bridge_annotations,
        )):
            raise GrowthMetricError("provide an edge or explicit tensors, not both")
        if not isinstance(edge, PostCommitmentGrowthEdge):
            raise GrowthMetricError("edge must be a PostCommitmentGrowthEdge")
        s = edge.identity.status_anchor_index
        annotations = _annotation_matrix(edge)
        result = GrowthMetricInput(
            _as_numpy(edge.residual_bridge_to_base, "residual_bridge_to_base"),
            _as_numpy(edge.residual_within_bridge, "residual_within_bridge"),
            _as_numpy(edge.attention_values, "attention_values"),
            _as_numpy(edge.attention_offsets, "attention_offsets"),
            _as_numpy(edge.attention_lengths, "attention_lengths"),
            _as_numpy(edge.attention_query_indices, "attention_query_indices"),
            annotations[: s + 1],
            annotations[s + 1 : s + 1 + BRIDGE_LENGTH],
        )
    else:
        values = (
            residual_bridge_to_base, residual_within_bridge, attention_values,
            attention_offsets, attention_lengths, attention_query_indices,
            base_annotations, bridge_annotations,
        )
        if any(value is None for value in values):
            raise GrowthMetricError("explicit metric input requires every label-free tensor")
        result = GrowthMetricInput(*(_as_numpy(value, "explicit metric tensor") for value in values))
    _validate_input(result)
    return result


def _validate_input(item: GrowthMetricInput) -> None:
    base_count = item.base_annotations.shape[0]
    if base_count < 1 or item.base_annotations.ndim != 2 or item.bridge_annotations.shape != (6, item.base_annotations.shape[1]):
        raise GrowthMetricError("typed base/bridge annotations have invalid shape")
    if item.residual_bridge_to_base.shape != (len(LAYERS), BRIDGE_LENGTH, base_count):
        raise GrowthMetricError("bridge-to-base residual relations must be [4, 6, base]")
    if item.residual_within_bridge.shape != (len(LAYERS), BRIDGE_LENGTH, BRIDGE_LENGTH):
        raise GrowthMetricError("within-bridge residual relations must be [4, 6, 6]")
    if not np.allclose(item.residual_within_bridge, item.residual_within_bridge.swapaxes(-1, -2)):
        raise GrowthMetricError("within-bridge residual relations must be symmetric")
    heads = item.attention_offsets.shape[1] if item.attention_offsets.ndim == 3 else 0
    if heads < 1 or item.attention_offsets.shape != (len(LAYERS), heads, BRIDGE_LENGTH + 1) or item.attention_lengths.shape != (len(LAYERS), heads, BRIDGE_LENGTH + 1):
        raise GrowthMetricError("packed attention offsets/lengths must be [4, heads, 7]")
    if item.attention_query_indices.shape != (7,) or not np.array_equal(
        item.attention_query_indices, np.arange(base_count - 1, base_count + 6)
    ):
        raise GrowthMetricError("attention queries must be the anchor plus six bridge tokens")
    cursor = 0
    for layer in range(4):
        for query_offset, query in enumerate(item.attention_query_indices):
            for head in range(heads):
                length = int(item.attention_lengths[layer, head, query_offset])
                offset = int(item.attention_offsets[layer, head, query_offset])
                if length != int(query) + 1 or offset != cursor:
                    raise GrowthMetricError("packed attention is not complete causal rows")
                cursor += length
    if cursor != item.attention_values.size:
        raise GrowthMetricError("packed attention payload has trailing or missing values")


def unpack_growth_attention(item: GrowthMetricInput) -> UnpackedGrowthAttention:
    """Losslessly split every packed query row into base and causal bridge keys."""
    _validate_input(item)
    base_count, heads = item.base_annotations.shape[0], item.attention_offsets.shape[1]
    base = np.empty((4, heads, 7, base_count), dtype=item.attention_values.dtype)
    bridge = np.full((4, heads, 6, 6), np.nan, dtype=item.attention_values.dtype)
    for layer in range(4):
        for head in range(heads):
            for row, query in enumerate(item.attention_query_indices):
                offset, length = int(item.attention_offsets[layer, head, row]), int(item.attention_lengths[layer, head, row])
                values = item.attention_values[offset : offset + length]
                base[layer, head, row] = values[:base_count]
                if row:
                    bridge[layer, head, row - 1, :row] = values[base_count:]
    return UnpackedGrowthAttention(base, bridge)


def pack_growth_attention(item: GrowthMetricInput, unpacked: UnpackedGrowthAttention) -> np.ndarray:
    """Inverse of :func:`unpack_growth_attention`, useful for lossless audits."""
    base_count = item.base_annotations.shape[0]
    packed = np.empty_like(item.attention_values)
    for layer in range(4):
        for head in range(item.attention_offsets.shape[1]):
            for row in range(7):
                offset, length = int(item.attention_offsets[layer, head, row]), int(item.attention_lengths[layer, head, row])
                packed[offset : offset + base_count] = unpacked.base_keys[layer, head, row]
                if row:
                    packed[offset + base_count : offset + length] = unpacked.bridge_keys[layer, head, row - 1, :row]
    return packed


def _typed_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Normalized causal position is comparable across unequal-length prefixes;
    # raw absolute indices would otherwise dominate every typed relation.
    normalized = np.abs(left[:, None, 1] - right[None, :, 1])
    categorical = (left[:, None, 2:] != right[None, :, 2:]).mean(axis=-1)
    return (normalized + categorical) / 2.0


def _pairwise_profile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """RMS profile distance for [features, tokens] arrays without cubic tensors."""
    count = left.shape[0]
    squared = (
        np.mean(left * left, axis=0)[:, None]
        + np.mean(right * right, axis=0)[None, :]
        - 2 * (left.T @ right) / count
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _coupled_mse(left: np.ndarray, right: np.ndarray, coupling: np.ndarray) -> float:
    """Mean squared difference under a base-token coupling."""
    if left.shape == right.shape:
        indices = np.argmax(coupling, axis=1)
        hard = (
            coupling.shape[0] == coupling.shape[1]
            and np.allclose(coupling[np.arange(coupling.shape[0]), indices], 1.0 / coupling.shape[0])
            and np.count_nonzero(coupling > 0) == coupling.shape[0]
        )
        if hard:
            return float(np.mean((left - right[:, indices]) ** 2))
    source, target = coupling.sum(axis=1), coupling.sum(axis=0)
    left_squared = (left * left) @ source
    right_squared = (right * right) @ target
    cross = np.sum((left @ coupling) * right, axis=1)
    return float(np.mean(np.maximum(left_squared + right_squared - 2 * cross, 0.0)))


def _exact_coupling(cost: np.ndarray, *, require_unique: bool = False) -> np.ndarray | None:
    """Preserve an exact typed bijection without entropy-induced blur."""
    rows, columns = linear_sum_assignment(cost)
    positive = cost[cost > 0]
    tolerance = max(1e-10, (float(np.median(positive)) if positive.size else 1.0) * 1e-6)
    if cost.shape[0] != cost.shape[1] or not np.all(cost[rows, columns] <= tolerance):
        return None
    if require_unique and (
        not np.all(np.sum(cost <= tolerance, axis=1) == 1)
        or not np.all(np.sum(cost <= tolerance, axis=0) == 1)
    ):
        return None
    coupling = np.zeros_like(cost, dtype=float)
    coupling[rows, columns] = 1.0 / cost.shape[0]
    return coupling


def _balanced_coupling(cost: np.ndarray) -> np.ndarray:
    """Return an exact balanced coupling with uniform token mass.

    Exact transport avoids an entropy hyperparameter and the slow near-bijection
    convergence that occurs when causal prefixes differ by only one token.
    """
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.isfinite(cost).all() or np.any(cost < 0):
        raise GrowthMetricError("base coupling cost must be finite and non-negative")
    transpose = cost.shape[0] > cost.shape[1]
    if cost.shape[0] == cost.shape[1]:
        forward = sha256(np.ascontiguousarray(cost).view(np.uint8)).digest()
        reverse = sha256(np.ascontiguousarray(cost.T).view(np.uint8)).digest()
        transpose = reverse < forward
    if transpose:
        return _balanced_coupling(cost.T).T
    exact = _exact_coupling(cost)
    if exact is not None:
        return exact
    source = np.full(cost.shape[0], 1.0 / cost.shape[0])
    target = np.full(cost.shape[1], 1.0 / cost.shape[1])
    coupling = np.asarray(ot.emd(source, target, cost, numItermax=100_000), dtype=float)
    residual = max(
        float(np.max(np.abs(coupling.sum(axis=1) - source))),
        float(np.max(np.abs(coupling.sum(axis=0) - target))),
    )
    if coupling.shape != cost.shape or not np.isfinite(coupling).all() or np.any(coupling < 0) or residual > 1e-10:
        raise GrowthMetricError("exact base coupling failed marginal validation")
    return coupling


def _residual_base_coupling(left: GrowthMetricInput, right: GrowthMetricInput) -> np.ndarray:
    if left.base_annotations.shape == right.base_annotations.shape and np.array_equal(left.base_annotations, right.base_annotations):
        return np.eye(left.base_annotations.shape[0], dtype=float) / left.base_annotations.shape[0]
    typed = _typed_cost(left.base_annotations, right.base_annotations)
    exact = _exact_coupling(typed, require_unique=True)
    if exact is not None:
        return exact
    profile = _pairwise_profile(
        left.residual_bridge_to_base.reshape(-1, left.base_annotations.shape[0]),
        right.residual_bridge_to_base.reshape(-1, right.base_annotations.shape[0]),
    )
    return _balanced_coupling(typed + profile)


def _attention_base_coupling(
    left: GrowthMetricInput,
    right: GrowthMetricInput,
    a: UnpackedGrowthAttention,
    b: UnpackedGrowthAttention,
) -> np.ndarray:
    if left.base_annotations.shape == right.base_annotations.shape and np.array_equal(left.base_annotations, right.base_annotations):
        return np.eye(left.base_annotations.shape[0], dtype=float) / left.base_annotations.shape[0]
    typed = _typed_cost(left.base_annotations, right.base_annotations)
    exact = _exact_coupling(typed, require_unique=True)
    if exact is not None:
        return exact
    # Sorting head values gives a candidate-key profile invariant to head labels.
    aa, bb = np.sort(a.base_keys, axis=1), np.sort(b.base_keys, axis=1)
    profiles = _pairwise_profile(
        aa.reshape(-1, left.base_annotations.shape[0]),
        bb.reshape(-1, right.base_annotations.shape[0]),
    )
    return _balanced_coupling(typed + profiles)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=float) ** 2)))


def _attention_head_distances(
    a: UnpackedGrowthAttention,
    b: UnpackedGrowthAttention,
    coupling: np.ndarray,
) -> tuple[float, float]:
    set_squares: list[float] = []
    indexed_squares: list[float] = []
    bridge_mask = np.tril(np.ones((6, 6), dtype=bool))
    hard = (
        coupling.ndim == 2
        and coupling.shape[0] == coupling.shape[1]
        and np.count_nonzero(coupling > 0) == coupling.shape[0]
        and np.allclose(
            coupling[np.arange(coupling.shape[0]), np.argmax(coupling, axis=1)],
            1.0 / coupling.shape[0],
        )
    )
    source = coupling.sum(axis=1)
    target = coupling.sum(axis=0)
    for layer in range(4):
        a_base = a.base_keys[layer]
        b_base = b.base_keys[layer]
        a_bridge = a.bridge_keys[layer][:, bridge_mask]
        b_bridge = b.bridge_keys[layer][:, bridge_mask]
        if hard:
            indices = np.argmax(coupling, axis=1)
            base_mse = np.mean((a_base[:, None, :, :] - b_base[:, :, indices][None, :, :, :]) ** 2, axis=(2, 3))
        else:
            left = a_base @ coupling
            base_sumsq = (a_base * a_base) @ source
            right_sumsq = (b_base * b_base) @ target
            base_mse = np.mean(
                np.maximum(
                    base_sumsq[:, None, :] + right_sumsq[None, :, :] - 2 * np.einsum("iqr,jqr->ijq", left, b_base),
                    0.0,
                ),
                axis=2,
            )
        bridge_sumsq = (a_bridge[:, None, :] - b_bridge[None, :, :]) ** 2
        bridge_mse = np.mean(bridge_sumsq, axis=2)
        # The normalized base measure and the finite bridge subgraph
        # each receive one vote, independent of prefix token count.
        cost = (base_mse + bridge_mse) / 2.0
        rows, columns = linear_sum_assignment(cost)
        set_squares.extend(cost[rows, columns].tolist())
        indexed_squares.extend(np.diag(cost).tolist())
    return float(np.sqrt(np.mean(set_squares))), float(np.sqrt(np.mean(indexed_squares)))


def growth_metric_distance(left: GrowthMetricInput, right: GrowthMetricInput) -> GrowthMetricDistance:
    """Compare two typed edges without labels, raw residual coordinates, or tokens."""
    _validate_input(left)
    _validate_input(right)
    if left.attention_offsets.shape[1] != right.attention_offsets.shape[1]:
        raise GrowthMetricError("attention comparison requires equal head counts")
    residual_coupling = _residual_base_coupling(left, right)
    bridge_to_base = float(np.sqrt(_coupled_mse(
        left.residual_bridge_to_base.reshape(-1, left.base_annotations.shape[0]),
        right.residual_bridge_to_base.reshape(-1, right.base_annotations.shape[0]),
        residual_coupling,
    )))
    within_bridge = _rms(left.residual_within_bridge - right.residual_within_bridge)
    a, b = unpack_growth_attention(left), unpack_growth_attention(right)
    attention_coupling = _attention_base_coupling(left, right, a, b)
    head_set, indexed = _attention_head_distances(a, b, attention_coupling)
    return GrowthMetricDistance(bridge_to_base, within_bridge, head_set, indexed)


def fit_growth_metric_scaler(training_pairs: Sequence[tuple[GrowthMetricInput, GrowthMetricInput]]) -> GrowthMetricScaler:
    """Fit only from the supplied training pairs; callers own fold membership."""
    return GrowthMetricScaler.fit([growth_metric_distance(left, right) for left, right in training_pairs])


__all__ = [
    "GrowthMetricDistance", "GrowthMetricError", "GrowthMetricFusion", "GrowthMetricInput", "GrowthMetricScaler",
    "UnpackedGrowthAttention", "adapt_growth_edge", "fit_growth_metric_scaler",
    "growth_metric_distance", "pack_growth_attention", "unpack_growth_attention",
]
