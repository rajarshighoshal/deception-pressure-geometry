"""Support-preserving gauge controller over an intrinsic deception field."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np

from geoprobe.control.relational_horizontal_lift import (
    HorizontalLift,
    RelationalHorizontalLiftError,
)
from geoprobe.control.relational_intrinsic_risk_field import (
    FIELD_KINDS,
    PressureMatchedFieldEvaluation,
    PressureMatchedRiskField,
)
from geoprobe.geometry.relational_gauge_atlas import GaugeChart, GaugeQueryState


CONTROL_ARMS = (
    "gauge_geodesic",
    "sign_flipped",
    "random_tangent",
    "no_intervention",
)
STOP_REASONS = (
    "active",
    "off_support",
    "field_undefined",
    "boundary_exit",
    "lift_undefined",
    "zero_direction",
    "step_cap_exhausted",
    "max_steps",
    "finished_eos",
    "nonfinite",
    "no_intervention",
)


class RelationalGaugeControllerError(ValueError):
    """A gauge-controller configuration or state is invalid."""


def _readonly(value: object, *, name: str, ndim: int, dtype: object = np.float64) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    except (TypeError, ValueError) as error:
        raise RelationalGaugeControllerError(f"{name} is not numeric") from error
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise RelationalGaugeControllerError(
            f"{name} must be a finite non-empty {ndim}-dimensional array"
        )
    array = array.copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class GaugeControllerConfig:
    field_kind: str = "pressure_residual_deception"
    control_arm: str = "gauge_geodesic"
    boundary_depth: float = 0.0
    metric_regularization: float = 1.0
    trust_fraction: float = 0.5
    path_budget_multiplier: float = 2.0
    minimum_gradient_norm: float = 1e-8
    maximum_query_stress: float = 0.5
    maximum_steps: int = 8

    def __post_init__(self) -> None:
        if self.field_kind not in FIELD_KINDS:
            raise RelationalGaugeControllerError("field_kind is invalid")
        if self.control_arm not in CONTROL_ARMS:
            raise RelationalGaugeControllerError("control_arm is invalid")
        for name in (
            "boundary_depth",
            "metric_regularization",
            "trust_fraction",
            "path_budget_multiplier",
            "minimum_gradient_norm",
            "maximum_query_stress",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RelationalGaugeControllerError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.metric_regularization <= 0.0:
            raise RelationalGaugeControllerError(
                "metric_regularization must be positive"
            )
        if not 0.0 < self.trust_fraction <= 1.0:
            raise RelationalGaugeControllerError("trust_fraction must lie in (0, 1]")
        if self.path_budget_multiplier <= 0.0 or self.minimum_gradient_norm <= 0.0:
            raise RelationalGaugeControllerError(
                "path budget and minimum gradient must be positive"
            )
        if self.maximum_query_stress < 0.0:
            raise RelationalGaugeControllerError(
                "maximum_query_stress must be non-negative"
            )
        if not isinstance(self.maximum_steps, int) or self.maximum_steps < 1:
            raise RelationalGaugeControllerError("maximum_steps must be positive")


@dataclass(frozen=True, slots=True)
class GaugeControllerObservation:
    observation_id: str
    chart: GaugeChart
    query: GaugeQueryState
    nuisance_key: tuple[str, ...]
    finished: bool = False
    randomization_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise RelationalGaugeControllerError("observation_id must be non-empty")
        key = self.observation_id if self.randomization_key is None else self.randomization_key
        if not isinstance(key, str) or not key:
            raise RelationalGaugeControllerError("randomization_key must be non-empty")
        if self.query.chart_id != self.chart.chart_id:
            raise RelationalGaugeControllerError("query and chart IDs disagree")
        if self.query.query_coordinates.shape != (self.chart.rank,):
            raise RelationalGaugeControllerError("query coordinate rank disagrees with chart")
        if not self.nuisance_key or any(
            not isinstance(value, str) or not value for value in self.nuisance_key
        ):
            raise RelationalGaugeControllerError("nuisance_key is invalid")
        object.__setattr__(self, "randomization_key", key)


@dataclass(frozen=True, slots=True)
class GaugeControllerState:
    step_count: int = 0
    cumulative_intrinsic_length: float = 0.0
    stopped: bool = False
    stop_reason: str = "active"

    def __post_init__(self) -> None:
        if not isinstance(self.step_count, int) or self.step_count < 0:
            raise RelationalGaugeControllerError("step_count must be non-negative")
        if (
            not math.isfinite(float(self.cumulative_intrinsic_length))
            or self.cumulative_intrinsic_length < 0.0
        ):
            raise RelationalGaugeControllerError(
                "cumulative_intrinsic_length must be finite and non-negative"
            )
        if self.stop_reason not in STOP_REASONS:
            raise RelationalGaugeControllerError("stop_reason is invalid")
        if self.stopped != (self.stop_reason != "active"):
            raise RelationalGaugeControllerError("stopped and stop_reason disagree")


@dataclass(frozen=True, slots=True)
class GaugeControlProposal:
    observation_id: str
    arm: str
    field: PressureMatchedFieldEvaluation
    status: str
    stop: bool
    intrinsic_direction: np.ndarray
    intrinsic_step: np.ndarray
    intrinsic_step_norm: float
    distance_to_boundary: float
    remaining_path_budget: float
    fiber_step: np.ndarray
    fiber_scale: float
    layer_fiber_norms: np.ndarray
    layer_fiber_norm_caps: np.ndarray
    next_state: GaugeControllerState

    def __post_init__(self) -> None:
        if self.arm not in CONTROL_ARMS or self.status not in STOP_REASONS:
            raise RelationalGaugeControllerError("proposal arm or status is invalid")
        direction = _readonly(
            self.intrinsic_direction, name="intrinsic_direction", ndim=1
        )
        step = _readonly(self.intrinsic_step, name="intrinsic_step", ndim=1)
        if step.shape != direction.shape:
            raise RelationalGaugeControllerError("direction and step ranks disagree")
        fiber = _readonly(
            self.fiber_step, name="fiber_step", ndim=2, dtype=np.float32
        )
        norms = _readonly(
            self.layer_fiber_norms, name="layer_fiber_norms", ndim=1
        )
        caps = _readonly(
            self.layer_fiber_norm_caps, name="layer_fiber_norm_caps", ndim=1
        )
        if norms.shape != caps.shape or norms.shape != (fiber.shape[0],):
            raise RelationalGaugeControllerError("fiber diagnostics disagree on layers")
        for name in (
            "intrinsic_step_norm",
            "distance_to_boundary",
            "remaining_path_budget",
            "fiber_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise RelationalGaugeControllerError(
                    f"{name} must be finite and non-negative"
                )
        object.__setattr__(self, "intrinsic_direction", direction)
        object.__setattr__(self, "intrinsic_step", step)
        object.__setattr__(self, "fiber_step", fiber)
        object.__setattr__(self, "layer_fiber_norms", norms)
        object.__setattr__(self, "layer_fiber_norm_caps", caps)


class RelationalGaugeController:
    """Closed-loop local risk descent with a model-specific horizontal lift."""

    def __init__(
        self,
        *,
        risk_field: PressureMatchedRiskField,
        horizontal_lift: HorizontalLift,
        config: GaugeControllerConfig | None = None,
    ) -> None:
        self.risk_field = risk_field
        self.horizontal_lift = horizontal_lift
        self.config = config or GaugeControllerConfig()

    @staticmethod
    def initial_state() -> GaugeControllerState:
        return GaugeControllerState()

    def _empty_proposal(
        self,
        observation: GaugeControllerObservation,
        state: GaugeControllerState,
        field: PressureMatchedFieldEvaluation,
        status: str,
        *,
        stop: bool = True,
    ) -> GaugeControlProposal:
        rank = observation.chart.rank
        try:
            patch = self.horizontal_lift.patch(observation.chart.chart_id)
            fiber_shape = (patch.layer_count, patch.hidden_size)
            caps = patch.layer_fiber_norm_caps
        except RelationalHorizontalLiftError:
            fiber_shape = (self.horizontal_lift.layer_count, self.horizontal_lift.hidden_size)
            caps = np.ones(self.horizontal_lift.layer_count)
        next_state = replace(state, stopped=stop, stop_reason=status)
        return GaugeControlProposal(
            observation_id=observation.observation_id,
            arm=self.config.control_arm,
            field=field,
            status=status,
            stop=stop,
            intrinsic_direction=np.zeros(rank),
            intrinsic_step=np.zeros(rank),
            intrinsic_step_norm=0.0,
            distance_to_boundary=0.0,
            remaining_path_budget=0.0,
            fiber_step=np.zeros(fiber_shape, dtype=np.float32),
            fiber_scale=0.0,
            layer_fiber_norms=np.zeros(fiber_shape[0]),
            layer_fiber_norm_caps=caps,
            next_state=next_state,
        )

    def _arm_direction(
        self,
        unit_descent: np.ndarray,
        metric: np.ndarray,
        randomization_key: str,
        step_count: int,
    ) -> np.ndarray:
        arm = self.config.control_arm
        if arm == "gauge_geodesic":
            return unit_descent
        if arm == "sign_flipped":
            return -unit_descent
        if arm == "no_intervention":
            return np.zeros_like(unit_descent)
        seed = int.from_bytes(
            sha256(
                f"gauge-random-tangent-v1\x00{randomization_key}\x00{step_count}".encode()
            ).digest()[:8],
            "little",
        )
        vector = np.random.default_rng(seed).normal(size=unit_descent.size)
        projection = float(vector @ metric @ unit_descent)
        vector = vector - projection * unit_descent
        norm = math.sqrt(max(float(vector @ metric @ vector), 0.0))
        if norm <= self.config.minimum_gradient_norm:
            vector = -unit_descent
            norm = 1.0
        return np.asarray(vector / norm, dtype=np.float64)

    def propose(
        self,
        observation: GaugeControllerObservation,
        state: GaugeControllerState,
    ) -> GaugeControlProposal:
        if state.stopped:
            field = self.risk_field.evaluate(
                observation.chart,
                observation.query.query_coordinates,
                nuisance_key=observation.nuisance_key,
            )
            return self._empty_proposal(
                observation, state, field, state.stop_reason, stop=True
            )
        field = self.risk_field.evaluate(
            observation.chart,
            observation.query.query_coordinates,
            nuisance_key=observation.nuisance_key,
        )
        if observation.finished:
            return self._empty_proposal(
                observation, state, field, "finished_eos", stop=True
            )
        if (
            not observation.query.support_status
            or observation.query.stress > self.config.maximum_query_stress
        ):
            return self._empty_proposal(
                observation, state, field, "off_support", stop=True
            )
        if not field.defined:
            return self._empty_proposal(
                observation, state, field, "field_undefined", stop=True
            )
        if state.step_count >= self.config.maximum_steps:
            return self._empty_proposal(
                observation, state, field, "max_steps", stop=True
            )
        depth = field.depth(self.config.field_kind)
        if depth <= self.config.boundary_depth:
            return self._empty_proposal(
                observation, state, field, "boundary_exit", stop=True
            )
        try:
            patch = self.horizontal_lift.patch(observation.chart.chart_id)
        except RelationalHorizontalLiftError:
            return self._empty_proposal(
                observation, state, field, "lift_undefined", stop=True
            )

        gradient = field.gradient(self.config.field_kind)
        metric = patch.tangent_metric
        try:
            natural_gradient = np.linalg.solve(metric, gradient)
        except np.linalg.LinAlgError:
            return self._empty_proposal(
                observation, state, field, "nonfinite", stop=True
            )
        slope = math.sqrt(max(float(gradient @ natural_gradient), 0.0))
        if not math.isfinite(slope) or slope <= self.config.minimum_gradient_norm:
            return self._empty_proposal(
                observation, state, field, "zero_direction", stop=True
            )
        unit_descent = -natural_gradient / slope
        unit_direction = self._arm_direction(
            unit_descent,
            metric,
            observation.randomization_key,
            state.step_count,
        )
        if self.config.control_arm == "no_intervention":
            return self._empty_proposal(
                observation, state, field, "no_intervention", stop=True
            )

        distance_to_boundary = max(
            (depth - self.config.boundary_depth) / slope, 0.0
        )
        unconstrained_length = slope / (2.0 * self.config.metric_regularization)
        per_step_cap = patch.intrinsic_trust_radius * self.config.trust_fraction
        path_budget = patch.intrinsic_trust_radius * self.config.path_budget_multiplier
        remaining = max(path_budget - state.cumulative_intrinsic_length, 0.0)
        step_length = min(
            distance_to_boundary,
            unconstrained_length,
            per_step_cap,
            remaining,
        )
        if step_length <= self.config.minimum_gradient_norm:
            return self._empty_proposal(
                observation, state, field, "step_cap_exhausted", stop=True
            )
        intrinsic_step = unit_direction * step_length
        try:
            fiber = patch.lift(intrinsic_step, require_in_support=True).astype(
                np.float64
            )
        except RelationalHorizontalLiftError:
            return self._empty_proposal(
                observation, state, field, "lift_undefined", stop=True
            )
        layer_norms = np.linalg.norm(fiber, axis=1)
        ratios = np.divide(
            patch.layer_fiber_norm_caps,
            layer_norms,
            out=np.full_like(layer_norms, np.inf),
            where=layer_norms > 1e-12,
        )
        fiber_scale = min(1.0, float(np.min(ratios)))
        fiber *= fiber_scale
        layer_norms = np.linalg.norm(fiber, axis=1)
        applied_intrinsic_length = step_length * fiber_scale
        intrinsic_step *= fiber_scale
        next_state = GaugeControllerState(
            step_count=state.step_count + 1,
            cumulative_intrinsic_length=(
                state.cumulative_intrinsic_length + applied_intrinsic_length
            ),
            stopped=False,
            stop_reason="active",
        )
        return GaugeControlProposal(
            observation_id=observation.observation_id,
            arm=self.config.control_arm,
            field=field,
            status="active",
            stop=False,
            intrinsic_direction=unit_direction,
            intrinsic_step=intrinsic_step,
            intrinsic_step_norm=applied_intrinsic_length,
            distance_to_boundary=distance_to_boundary,
            remaining_path_budget=remaining,
            fiber_step=fiber.astype(np.float32),
            fiber_scale=fiber_scale,
            layer_fiber_norms=layer_norms,
            layer_fiber_norm_caps=patch.layer_fiber_norm_caps,
            next_state=next_state,
        )


__all__ = [
    "CONTROL_ARMS",
    "STOP_REASONS",
    "GaugeControlProposal",
    "GaugeControllerConfig",
    "GaugeControllerObservation",
    "GaugeControllerState",
    "RelationalGaugeController",
    "RelationalGaugeControllerError",
]
