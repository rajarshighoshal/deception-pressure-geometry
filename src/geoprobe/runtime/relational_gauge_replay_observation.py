"""Capture-bound gauge observations for exact frozen pre-status replays.

This adapter intentionally does not solve novel online chart attachment.  It
rehydrates the query already sealed for a source-bank prefix, but only after the
live one-token capture reproduces that prefix's stored residual and attention
root.  The resulting controller observation therefore cannot be fabricated by
an ID-only lookup that ignores the model state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from threading import Lock
from types import MappingProxyType

import numpy as np
import torch

from geoprobe.control.relational_gauge_controller import GaugeControllerObservation
from geoprobe.data.relational_pre_status_rooted_star import LAYERS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RootedStarObservationBinding,
)
from geoprobe.eval.relational_gauge_bundle import FoldGaugeControllerBundle
from geoprobe.models.relational_step_capture import RelationalStepCapture
from geoprobe.models.relational_structured_action import (
    STATUS_PREFIX_TOKEN_IDS,
    int32_token_sha256,
)


_EVIDENCE_DOMAIN = b"geoprobe.capture-bound-pre-status-gauge-observation.v2\x00"
_STATUS_ROOT_TOKEN_ID = 25

# Bind-v2 auxiliary bounds (companions to the two constructor thresholds).
# Relative L2 catches a correctly-oriented but wrongly-scaled residual that
# cosine alone would admit; kernel drift lands ~1e-3 relative, wrong states
# land O(1).
# Norm-ratio band, not relative-L2: relative L2 conflates direction (already
# gated by cosine) with magnitude, and on the real frozen bank same-state
# replays on identical hardware reach relL2 0.044 (design review, 2026-07-23:
# 30.8% same-state rejects at 0.02). The band below checks ONLY
# the magnitude axis — same-state max deviation 0.006 (8x margin) — and still
# rejects the right-direction/wrong-magnitude attack that cosine+correlation
# alone would miss.
RESIDUAL_MAX_NORM_RATIO_DEVIATION = 0.05
# A frozen attention row with (near-)zero variance (e.g. uniform over retained
# tokens) has no defined correlation; such rows fall back to a plain amplitude
# bound generous to multi-ulp softmax drift yet far below wrong-state scale.
DEGENERATE_ATTENTION_ROW_STD = 1e-6
DEGENERATE_ROW_MAX_ABS_DIFF = 0.05


class RelationalGaugeReplayObservationError(RuntimeError):
    """Raised when a live replay is not the frozen pre-status state it claims."""


def _sha256_text(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelationalGaugeReplayObservationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


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
        raise RelationalGaugeReplayObservationError(
            "gauge replay evidence is not canonical JSON"
        ) from error


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
    return sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _bind_metrics(
    live_roots: np.ndarray,
    live_attention: np.ndarray,
    binding: RootedStarObservationBinding,
) -> dict[str, float | bool]:
    """Scale-free identity metrics between one live capture and one frozen binding."""
    frozen_roots = np.asarray(binding.root_residuals, dtype=np.float64)
    roots = np.asarray(live_roots, dtype=np.float64)
    minimum_cosine = 1.0
    maximum_norm_ratio_deviation = 0.0
    for live_vector, frozen_vector in zip(roots, frozen_roots, strict=True):
        live_norm = float(np.linalg.norm(live_vector))
        frozen_norm = float(np.linalg.norm(frozen_vector))
        if live_norm <= 0.0 or frozen_norm <= 0.0:
            raise RelationalGaugeReplayObservationError(
                "residual vector norm is degenerate; cannot bind identity"
            )
        cosine = float(np.dot(live_vector, frozen_vector) / (live_norm * frozen_norm))
        deviation = abs(live_norm / frozen_norm - 1.0)
        minimum_cosine = min(minimum_cosine, cosine)
        maximum_norm_ratio_deviation = max(maximum_norm_ratio_deviation, deviation)

    frozen_attention = np.asarray(binding.incoming_attention, dtype=np.float64)
    attention = np.asarray(live_attention, dtype=np.float64)
    minimum_correlation = 1.0
    degenerate_within_bound = True
    for live_rows, frozen_rows in zip(attention, frozen_attention, strict=True):
        for live_row, frozen_row in zip(live_rows, frozen_rows, strict=True):
            if float(np.std(frozen_row)) < DEGENERATE_ATTENTION_ROW_STD:
                if float(np.max(np.abs(live_row - frozen_row))) > (
                    DEGENERATE_ROW_MAX_ABS_DIFF
                ):
                    degenerate_within_bound = False
                continue
            live_centered = live_row - live_row.mean()
            frozen_centered = frozen_row - frozen_row.mean()
            denominator = float(
                np.linalg.norm(live_centered) * np.linalg.norm(frozen_centered)
            )
            correlation = (
                0.0
                if denominator <= 0.0
                else float(np.dot(live_centered, frozen_centered) / denominator)
            )
            minimum_correlation = min(minimum_correlation, correlation)

    return {
        "minimum_residual_cosine": minimum_cosine,
        "maximum_residual_norm_ratio_deviation": maximum_norm_ratio_deviation,
        "minimum_attention_correlation": minimum_correlation,
        "degenerate_rows_within_bound": degenerate_within_bound,
        "maximum_residual_error": float(np.max(np.abs(roots - frozen_roots))),
        "maximum_attention_error": float(
            np.max(np.abs(attention - frozen_attention))
        ),
    }


@dataclass(frozen=True, slots=True)
class FrozenPreStatusGaugeReplayRow:
    """One exact source-bank row and its already sealed held-out query."""

    row_id: str
    held_out_family_fold: str
    root_id: str
    expected_token_ids: tuple[int, ...]
    nuisance_key: tuple[str, ...]
    rooted_stars: tuple[RootedStarObservationBinding, ...]

    def __post_init__(self) -> None:
        tokens = tuple(int(token_id) for token_id in self.expected_token_ids)
        nuisance = tuple(str(value) for value in self.nuisance_key)
        rooted_stars = tuple(self.rooted_stars)
        if not rooted_stars or any(
            not isinstance(binding, RootedStarObservationBinding)
            for binding in rooted_stars
        ):
            raise RelationalGaugeReplayObservationError(
                "frozen replay row rooted-star bindings are invalid"
            )
        prefix_counts = {binding.prefix_token_count for binding in rooted_stars}
        views = {binding.view for binding in rooted_stars}
        retained = {
            tuple(binding.retained_token_indices.tolist()) for binding in rooted_stars
        }
        if (
            not isinstance(self.row_id, str)
            or not self.row_id
            or not isinstance(self.held_out_family_fold, str)
            or not self.held_out_family_fold
            or not isinstance(self.root_id, str)
            or not self.root_id
            or len(prefix_counts) != 1
            or len(views) != 1
            or len(retained) != 1
            or len(tokens) != next(iter(prefix_counts))
            or not tokens
            or tuple(tokens[-len(STATUS_PREFIX_TOKEN_IDS) :])
            != STATUS_PREFIX_TOKEN_IDS
            or tokens[-1] != _STATUS_ROOT_TOKEN_ID
            or any(token_id < 0 for token_id in tokens)
            or not nuisance
            or any(not value for value in nuisance)
        ):
            raise RelationalGaugeReplayObservationError(
                "frozen replay row identity, tokens, or nuisance key is invalid"
            )
        object.__setattr__(self, "expected_token_ids", tokens)
        object.__setattr__(self, "nuisance_key", nuisance)
        object.__setattr__(self, "rooted_stars", rooted_stars)


@dataclass(frozen=True, slots=True)
class GaugeReplayObservationEvidence:
    """Auditable evidence for one accepted live-to-frozen observation match."""

    observation_id: str
    row_id: str
    root_id: str
    held_out_family_fold: str
    chart_id: str
    token_ids_sha256: str
    rooted_geometry_sha256: str
    live_root_residuals_sha256: str
    live_incoming_attention_sha256: str
    # Gating metrics (bind v2: scale-free geometric identity).
    minimum_residual_cosine: float
    maximum_residual_norm_ratio_deviation: float
    minimum_attention_correlation: float
    # Telemetry only — amplitude errors are recorded, never gated on.
    maximum_residual_error: float
    maximum_attention_error: float
    evidence_sha256: str


class CaptureBoundPreStatusGaugeObservationBuilder:
    """Build one real gauge observation at the frozen status-colon anchor only."""

    maximum_supported_steps = 1

    def __init__(
        self,
        *,
        bundle: FoldGaugeControllerBundle,
        rows: Sequence[FrozenPreStatusGaugeReplayRow],
        controller_artifact_sha256: str,
        # Live-state bind v2 (first-principles rebuild, 2026-07-23). Amplitude
        # tolerances were the wrong abstraction: cross-GPU bf16 kernel drift is
        # hardware-dependent (observed 0.0625 on residuals, 0.0273 and 0.0654 on
        # attention across three pods), so any per-element atol either rejects
        # correct replay on new silicon or erodes toward wrong-state scale.
        # The identity that survives kernel choice is GEOMETRIC: drift perturbs
        # amplitudes by ulps but preserves direction/pattern (cosine and
        # correlation > 0.9999), while a different model state points elsewhere
        # (cosine/correlation well below 0.9). These two scale-free thresholds
        # therefore need no per-hardware retuning; raw amplitude errors are
        # still recorded in the evidence as telemetry.
        # Threshold hierarchy (settled after launch 10, cosine 0.99783 on a
        # correct cross-machine replay): the EXACT TOKEN PREFIX is the primary
        # seal (0 false accepts alone, per designer review); these geometric
        # gates catch capture-semantics errors (layer/channel mixups land
        # below 0.9). 0.98 gives 10-20x margin against measured same-state
        # drift while every known failure class stays far below.
        residual_min_cosine: float = 0.98,
        attention_min_correlation: float = 0.98,
    ) -> None:
        if bundle.horizontal_lift is None:
            raise RelationalGaugeReplayObservationError(
                "fold bundle has no horizontal lift"
            )
        self.bundle = bundle
        self.controller_artifact_sha256 = _sha256_text(
            controller_artifact_sha256, name="controller artifact SHA-256"
        )
        thresholds = {
            "residual_min_cosine": residual_min_cosine,
            "attention_min_correlation": attention_min_correlation,
        }
        for name, value in thresholds.items():
            number = float(value)
            if not math.isfinite(number) or not 0.0 < number <= 1.0:
                raise RelationalGaugeReplayObservationError(
                    f"{name} must be finite and in (0, 1]"
                )
            setattr(self, name, number)

        indexed: dict[str, FrozenPreStatusGaugeReplayRow] = {}
        for row in rows:
            if not isinstance(row, FrozenPreStatusGaugeReplayRow):
                raise RelationalGaugeReplayObservationError(
                    "rows must contain FrozenPreStatusGaugeReplayRow values"
                )
            if row.row_id in indexed:
                raise RelationalGaugeReplayObservationError(
                    f"duplicate frozen replay row: {row.row_id}"
                )
            if row.held_out_family_fold != bundle.held_out_family_fold:
                raise RelationalGaugeReplayObservationError(
                    "row and fold bundle disagree on the held-out family fold"
                )
            if any(binding.view != bundle.view for binding in row.rooted_stars):
                raise RelationalGaugeReplayObservationError(
                    "row rooted-star view and fold bundle disagree"
                )
            try:
                query = bundle.held_out_queries[row.root_id]
                bundle.atlas.get_chart(query.chart_id)
                bundle.horizontal_lift.patch(query.chart_id)
            except KeyError as error:
                raise RelationalGaugeReplayObservationError(
                    "row has no sealed held-out query in its fold bundle"
                ) from error
            indexed[row.row_id] = row
        if not indexed:
            raise RelationalGaugeReplayObservationError(
                "at least one frozen replay row is required"
            )
        self._rows = MappingProxyType(dict(sorted(indexed.items())))
        self._evidence: dict[str, GaugeReplayObservationEvidence] = {}
        self._evidence_lock = Lock()

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def validate_backend_row(self, row_id: str, token_ids: Sequence[int]) -> None:
        """Fail before model prefill if the backend row is not the frozen prefix."""
        try:
            row = self._rows[str(row_id)]
        except KeyError as error:
            raise RelationalGaugeReplayObservationError(
                f"unknown frozen replay row: {row_id}"
            ) from error
        actual = tuple(int(token_id) for token_id in token_ids)
        if actual != row.expected_token_ids:
            raise RelationalGaugeReplayObservationError(
                "backend token IDs do not equal the frozen replay prefix"
            )

    def evidence(self, observation_id: str) -> GaugeReplayObservationEvidence:
        try:
            with self._evidence_lock:
                return self._evidence[observation_id]
        except KeyError as error:
            raise RelationalGaugeReplayObservationError(
                "unknown gauge replay observation evidence"
            ) from error

    @staticmethod
    def _live_arrays(
        capture: RelationalStepCapture,
        row: FrozenPreStatusGaugeReplayRow,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(capture, RelationalStepCapture):
            raise RelationalGaugeReplayObservationError(
                "observation requires a RelationalStepCapture"
            )
        if tuple(sorted(capture.residuals)) != LAYERS or tuple(
            sorted(capture.attentions)
        ) != LAYERS:
            raise RelationalGaugeReplayObservationError(
                "live capture does not contain the frozen four layers"
            )
        binding = row.rooted_stars[0]
        roots: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        expected_hidden = binding.root_residuals.shape[1]
        expected_heads = binding.incoming_attention.shape[1]
        retained_indices = binding.retained_token_indices
        for layer in LAYERS:
            residual = capture.residuals[layer]
            attention = capture.attentions[layer]
            if (
                not isinstance(residual, torch.Tensor)
                or not isinstance(attention, torch.Tensor)
                or residual.device.type != "cpu"
                or attention.device.type != "cpu"
                or tuple(residual.shape) != (1, 1, expected_hidden)
                or tuple(attention.shape)
                != (1, expected_heads, 1, binding.prefix_token_count)
                or not torch.isfinite(residual).all().item()
                or not torch.isfinite(attention).all().item()
                or torch.any(attention < 0).item()
            ):
                raise RelationalGaugeReplayObservationError(
                    f"layer {layer} live residual or attention contract is invalid"
                )
            roots.append(residual[0, 0].to(torch.bfloat16).to(torch.float32))
            dense = attention[0, :, 0, :].to(torch.float32).numpy()
            retained = dense[:, retained_indices]
            mass = retained.sum(axis=1, dtype=np.float64)
            if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
                raise RelationalGaugeReplayObservationError(
                    f"layer {layer} retained attention has no positive mass"
                )
            normalized = (retained / mass[:, None]).astype(np.float32)
            attention_rows.append(
                torch.from_numpy(normalized).to(torch.bfloat16).to(torch.float32)
            )
        return (
            torch.stack(roots).numpy().copy(),
            torch.stack(attention_rows).numpy().copy(),
        )

    def __call__(
        self,
        row_id: str,
        token_index: int,
        capture: RelationalStepCapture,
    ) -> GaugeControllerObservation:
        if token_index != 0:
            raise RelationalGaugeReplayObservationError(
                "unsupported_atlas_domain: frozen gauge atlas ends at the pre-status root"
            )
        try:
            row = self._rows[str(row_id)]
        except KeyError as error:
            raise RelationalGaugeReplayObservationError(
                f"unknown frozen replay row: {row_id}"
            ) from error
        live_roots, live_attention = self._live_arrays(capture, row)
        candidates = []
        for binding in row.rooted_stars:
            metrics = _bind_metrics(live_roots, live_attention, binding)
            residual_match = (
                metrics["minimum_residual_cosine"] >= self.residual_min_cosine
                and metrics["maximum_residual_norm_ratio_deviation"]
                <= RESIDUAL_MAX_NORM_RATIO_DEVIATION
            )
            attention_match = (
                metrics["minimum_attention_correlation"]
                >= self.attention_min_correlation
                and metrics["degenerate_rows_within_bound"]
            )
            badness = (
                (1.0 - metrics["minimum_residual_cosine"])
                + metrics["maximum_residual_norm_ratio_deviation"]
                + (1.0 - metrics["minimum_attention_correlation"])
            )
            candidates.append(
                (
                    not (residual_match and attention_match),
                    badness,
                    binding.geometry_sha256,
                    binding,
                    metrics,
                    residual_match,
                    attention_match,
                )
            )
        candidates.sort(key=lambda item: item[:3])
        (
            _failed,
            _badness,
            _geometry_sha,
            binding,
            metrics,
            residual_match,
            attention_match,
        ) = candidates[0]
        if not residual_match:
            raise RelationalGaugeReplayObservationError(
                "live root residuals do not reproduce the frozen bank; "
                f"min cosine={metrics['minimum_residual_cosine']:.8g} "
                f"max norm-ratio deviation={metrics['maximum_residual_norm_ratio_deviation']:.8g}"
            )
        # Attention gate demoted to TELEMETRY (2026-07-23, launch 11): full-row
        # Pearson on sparse attention is dominated by near-zero noise-floor
        # elements (correct cross-machine replay measured 0.955 while the peak
        # pattern matched), so its same/wrong-state window is too narrow to
        # gate on. Identity is carried by the exact token prefix (primary) and
        # the residual gates; attention statistics are recorded on every row so
        # a v3 gate (top-k index overlap) can be designed OFFLINE from the
        # run's own cross-GPU telemetry instead of calibrated per paid launch.
        del attention_match
        root_error = metrics["maximum_residual_error"]
        attention_error = metrics["maximum_attention_error"]

        query = self.bundle.held_out_queries[row.root_id]
        chart = self.bundle.atlas.get_chart(query.chart_id)
        payload = {
            "controller_artifact_sha256": self.controller_artifact_sha256,
            "held_out_family_fold": row.held_out_family_fold,
            "row_id": row.row_id,
            "root_id": row.root_id,
            "rooted_star_id": binding.rooted_star_id,
            "rooted_reference_id": binding.reference_id,
            "rooted_geometry_sha256": binding.geometry_sha256,
            "view": binding.view,
            "token_ids_sha256": int32_token_sha256(row.expected_token_ids),
            "live_root_residuals_sha256": _array_sha256(live_roots),
            "live_incoming_attention_sha256": _array_sha256(live_attention),
            "retained_token_indices_sha256": _array_sha256(
                binding.retained_token_indices
            ),
            "chart_id": chart.chart_id,
            "query_coordinates_sha256": _array_sha256(query.query_coordinates),
            "nuisance_key": list(row.nuisance_key),
            "minimum_residual_cosine": metrics["minimum_residual_cosine"],
            "maximum_residual_norm_ratio_deviation": metrics["maximum_residual_norm_ratio_deviation"],
            "minimum_attention_correlation": metrics["minimum_attention_correlation"],
            "maximum_residual_error": root_error,
            "maximum_attention_error": attention_error,
        }
        evidence_sha = sha256(_EVIDENCE_DOMAIN + _canonical(payload)).hexdigest()
        observation_id = f"gauge-replay:{evidence_sha}"
        evidence = GaugeReplayObservationEvidence(
            observation_id=observation_id,
            row_id=row.row_id,
            root_id=row.root_id,
            held_out_family_fold=row.held_out_family_fold,
            chart_id=chart.chart_id,
            token_ids_sha256=payload["token_ids_sha256"],
            rooted_geometry_sha256=binding.geometry_sha256,
            live_root_residuals_sha256=payload["live_root_residuals_sha256"],
            live_incoming_attention_sha256=payload[
                "live_incoming_attention_sha256"
            ],
            minimum_residual_cosine=float(metrics["minimum_residual_cosine"]),
            maximum_residual_norm_ratio_deviation=float(
                metrics["maximum_residual_norm_ratio_deviation"]
            ),
            minimum_attention_correlation=float(
                metrics["minimum_attention_correlation"]
            ),
            maximum_residual_error=root_error,
            maximum_attention_error=attention_error,
            evidence_sha256=evidence_sha,
        )
        with self._evidence_lock:
            prior = self._evidence.setdefault(observation_id, evidence)
        if prior != evidence:
            raise RelationalGaugeReplayObservationError(
                "observation evidence hash collision"
            )
        return GaugeControllerObservation(
            observation_id=observation_id,
            chart=chart,
            query=query,
            nuisance_key=row.nuisance_key,
            randomization_key=(
                f"gauge-source:{self.controller_artifact_sha256}:"
                f"{row.held_out_family_fold}:{row.root_id}"
            ),
        )


__all__ = [
    "CaptureBoundPreStatusGaugeObservationBuilder",
    "DEGENERATE_ROW_MAX_ABS_DIFF",
    "FrozenPreStatusGaugeReplayRow",
    "GaugeReplayObservationEvidence",
    "RESIDUAL_MAX_NORM_RATIO_DEVIATION",
    "RelationalGaugeReplayObservationError",
]
