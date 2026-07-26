"""Efficient one-root causal execution for the source gauge controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from geoprobe.control.relational_gauge_controller import (
    GaugeControlProposal,
    GaugeControllerConfig,
    GaugeControllerObservation,
    RelationalGaugeController,
)
from geoprobe.data.relational_pre_status_rooted_star import LAYERS
from geoprobe.eval.relational_gauge_bundle import FoldGaugeControllerBundle
from geoprobe.models.hf_capture import (
    build_prefix_cache,
    forward_logits_batch_with_steering_cached,
)
from geoprobe.models.interface import ResidualSteeringSpec, SteeringSpec
from geoprobe.models.relational_step_capture import capture_relational_step_with_cache
from geoprobe.models.relational_structured_action import STATUS_PREFIX_TOKEN_IDS
from geoprobe.runtime.relational_gauge_replay_observation import (
    CaptureBoundPreStatusGaugeObservationBuilder,
)


SOURCE_GAUGE_ARM_ORDER = (
    "no_intervention",
    "gauge_geodesic",
    "sign_flipped",
    "random_tangent",
)


class RelationalGaugeReplayError(RuntimeError):
    """Raised when one-root gauge replay violates its causal execution contract."""


@dataclass(frozen=True, slots=True)
class GaugeRootArmForward:
    """One verified observation and all same-prefix intervention-arm logits."""

    row_id: str
    observation: GaugeControllerObservation
    proposals: Mapping[str, GaugeControlProposal]
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not self.row_id or tuple(self.proposals) != SOURCE_GAUGE_ARM_ORDER:
            raise RelationalGaugeReplayError("gauge root arm result identity is invalid")
        if (
            not isinstance(self.logits, torch.Tensor)
            or self.logits.ndim != 2
            or self.logits.shape[0] != len(SOURCE_GAUGE_ARM_ORDER)
            or not torch.isfinite(self.logits).all().item()
        ):
            raise RelationalGaugeReplayError(
                "gauge root arm logits must be finite [arm, vocabulary]"
            )


def build_source_gauge_controllers(
    bundle: FoldGaugeControllerBundle,
    *,
    base_config: GaugeControllerConfig | None = None,
) -> Mapping[str, RelationalGaugeController]:
    """Instantiate every intrinsic arm over one identical field and lift."""
    if bundle.horizontal_lift is None:
        raise RelationalGaugeReplayError("source gauge bundle has no horizontal lift")
    base = base_config or GaugeControllerConfig(maximum_steps=1)
    if base.maximum_steps != 1:
        raise RelationalGaugeReplayError(
            "source pre-status gauge replay is defined for exactly one controller step"
        )
    return MappingProxyType(
        {
            arm: RelationalGaugeController(
                risk_field=bundle.risk_field,
                horizontal_lift=bundle.horizontal_lift,
                config=replace(base, control_arm=arm),
            )
            for arm in SOURCE_GAUGE_ARM_ORDER
        }
    )


def build_gauge_steering_batch(
    proposals: Mapping[str, GaugeControlProposal],
    *,
    layers: Sequence[int] = LAYERS,
) -> tuple[SteeringSpec | None, ...]:
    """Convert four-layer horizontal lifts to one same-prefix HF arm batch."""
    selected = tuple(int(layer) for layer in layers)
    if selected != LAYERS or tuple(proposals) != SOURCE_GAUGE_ARM_ORDER:
        raise RelationalGaugeReplayError(
            "gauge steering batch must use the frozen arm and layer order"
        )
    result: list[SteeringSpec | None] = []
    shape: tuple[int, int] | None = None
    observation_ids: set[str] = set()
    for arm in SOURCE_GAUGE_ARM_ORDER:
        proposal = proposals[arm]
        if not isinstance(proposal, GaugeControlProposal) or proposal.arm != arm:
            raise RelationalGaugeReplayError("proposal and gauge arm disagree")
        observation_ids.add(proposal.observation_id)
        if proposal.stop != proposal.next_state.stopped:
            raise RelationalGaugeReplayError(
                "proposal stop flag and next controller state disagree"
            )
        fiber = np.asarray(proposal.fiber_step, dtype=np.float32)
        if shape is None:
            shape = tuple(fiber.shape)
        if tuple(fiber.shape) != shape or fiber.shape[0] != len(LAYERS):
            raise RelationalGaugeReplayError(
                "gauge proposals have inconsistent four-layer fibers"
            )
        if proposal.stop:
            if np.any(fiber != 0.0):
                raise RelationalGaugeReplayError(
                    "stopped gauge proposal contains a nonzero intervention"
                )
            result.append(None)
            continue
        if not np.any(fiber != 0.0):
            raise RelationalGaugeReplayError(
                "active gauge proposal contains no fiber intervention"
            )
        result.append(
            tuple(
                ResidualSteeringSpec(
                    layer=layer,
                    direction=torch.from_numpy(fiber[index].copy()),
                    alpha=1.0,
                )
                for index, layer in enumerate(LAYERS)
                if np.any(fiber[index] != 0.0)
            )
        )
    if len(observation_ids) != 1:
        raise RelationalGaugeReplayError(
            "gauge proposals do not bind one shared observation"
        )
    return tuple(result)


def forward_source_gauge_arms_hf(
    model: Any,
    token_ids: Sequence[int],
    *,
    row_id: str,
    observation_builder: CaptureBoundPreStatusGaugeObservationBuilder,
    controllers: Mapping[str, RelationalGaugeController],
) -> GaugeRootArmForward:
    """Capture once, verify/locate once, then score every arm from one KV prefill."""
    tokens = [int(token_id) for token_id in token_ids]
    if tuple(controllers) != SOURCE_GAUGE_ARM_ORDER:
        raise RelationalGaugeReplayError("source gauge controller arm order is invalid")
    if tuple(tokens[-len(STATUS_PREFIX_TOKEN_IDS) :]) != STATUS_PREFIX_TOKEN_IDS:
        raise RelationalGaugeReplayError(
            "source gauge replay prefix does not end at the exact Status anchor"
        )
    observation_builder.validate_backend_row(row_id, tokens)
    prefix_cache = build_prefix_cache(model, tokens, batch_size=1)
    parameter = next(model.parameters())
    pending = torch.tensor(
        [[tokens[-1]]], dtype=torch.long, device=parameter.device
    )
    try:
        provisional = capture_relational_step_with_cache(
            model,
            prefix_cache,
            input_ids=pending,
            layers=LAYERS,
            rollback=True,
        )
        observation = observation_builder(row_id, 0, provisional.capture)
        proposals = MappingProxyType(
            {
                arm: controllers[arm].propose(
                    observation, controllers[arm].initial_state()
                )
                for arm in SOURCE_GAUGE_ARM_ORDER
            }
        )
        if any(
            proposal.observation_id != observation.observation_id
            for proposal in proposals.values()
        ):
            raise RelationalGaugeReplayError(
                "gauge proposal is stale or bound to a different observation"
            )
        steering_batch = build_gauge_steering_batch(proposals)
        if len(steering_batch) > 1:
            repeat = getattr(
                prefix_cache.past_key_values, "batch_repeat_interleave", None
            )
            if repeat is None:
                raise RelationalGaugeReplayError(
                    "source gauge replay requires a repeatable HF cache"
                )
            repeat(len(steering_batch))
            prefix_cache.batch_size = len(steering_batch)
        logits = forward_logits_batch_with_steering_cached(
            model,
            prefix_cache,
            steering_batch=steering_batch,
        )
    finally:
        del prefix_cache
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise RelationalGaugeReplayError(
            "source gauge cached forward returned an invalid token axis"
        )
    return GaugeRootArmForward(
        row_id=row_id,
        observation=observation,
        proposals=proposals,
        logits=logits[:, -1, :],
    )


__all__ = [
    "GaugeRootArmForward",
    "RelationalGaugeReplayError",
    "SOURCE_GAUGE_ARM_ORDER",
    "build_gauge_steering_batch",
    "build_source_gauge_controllers",
    "forward_source_gauge_arms_hf",
]
