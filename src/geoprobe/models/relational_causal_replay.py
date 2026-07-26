"""Exact-prefix unrestricted status sampling under causal residual interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch

from geoprobe.data.relational_structured_action import (
    STRUCTURED_ACTION_TEMPERATURE,
    STRUCTURED_ACTION_TOP_P,
)
from geoprobe.control.relational_pre_status_causal import (
    CAUSAL_ARM_ORDER,
    PRIMARY_ACTUATION_LAYER,
    PreStatusCausalArmBundle,
    select_primary_actuation_vectors,
)
from geoprobe.models.hf_capture import (
    build_prefix_cache,
    forward_logits_batch_with_steering,
    forward_logits_batch_with_steering_cached,
)
from geoprobe.models.interface import ResidualSteeringSpec, SteeringSpec
from geoprobe.models.relational_structured_action import (
    FAIL_TOKEN_ID,
    PASS_TOKEN_ID,
    SKIP_TOKEN_ID,
    STATUS_PREFIX_TOKEN_IDS,
    int32_token_sha256,
    map_status_token_id,
)
from geoprobe.models.relational_structured_action_rollout import (
    _top_p_probabilities,
)


STATUS_CANDIDATE_TOKEN_IDS = (PASS_TOKEN_ID, FAIL_TOKEN_ID, SKIP_TOKEN_ID)


def _validate_beta(beta: float) -> float:
    if isinstance(beta, bool):
        raise ValueError("beta must be a finite positive scalar")
    try:
        value = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be a finite positive scalar") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("beta must be a finite positive scalar")
    return value


def build_primary_actuation_steering_batch(
    bundle: PreStatusCausalArmBundle,
    *,
    beta: float = 1.0,
) -> tuple[tuple[str, ...], tuple[SteeringSpec | None, ...]]:
    """Build a causal status-run steering batch for layer-12 primary actuation."""

    beta_value = _validate_beta(beta)
    vectors = select_primary_actuation_vectors(bundle.arm_vectors)
    names = tuple(CAUSAL_ARM_ORDER)
    steering_batch: list[SteeringSpec | None] = []
    for name in names:
        row = vectors[name]
        if name == "noop":
            steering_batch.append(None)
        else:
            steering_batch.append(
                ResidualSteeringSpec(
                    layer=PRIMARY_ACTUATION_LAYER,
                    direction=torch.tensor(row, dtype=torch.float32),
                    alpha=beta_value,
                )
            )
    return names, tuple(steering_batch)


@dataclass(frozen=True, slots=True)
class CausalStatusSample:
    arm: str
    raw_token_id: int
    raw_decoded_exact: str
    mapped_action: str
    rng_seed: int
    prefix_token_ids_sha256: str
    temperature: float
    top_p: float
    candidate_probabilities: dict[str, float]
    candidate_logits: dict[str, float]
    recognized_action_probability_mass: float
    top_token_id: int
    top_token_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_exact(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _arm_names(values: Sequence[str], expected: int) -> tuple[str, ...]:
    names = tuple(values)
    if (
        len(names) != expected
        or not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("arm names must be unique non-empty strings matching the batch")
    return names


def sample_status_arms_from_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    arm_names: Sequence[str],
    rng_seed: int,
    prefix_token_ids_sha256: str,
    temperature: float = STRUCTURED_ACTION_TEMPERATURE,
    top_p: float = STRUCTURED_ACTION_TOP_P,
) -> tuple[CausalStatusSample, ...]:
    """Apply the frozen sampler to one logits row per intervention arm."""
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("causal status logits must have shape [arm, vocabulary]")
    names = _arm_names(arm_names, int(logits.shape[0]))
    if (
        not isinstance(rng_seed, int)
        or isinstance(rng_seed, bool)
        or rng_seed < 0
        or rng_seed >= 2**64
    ):
        raise ValueError("causal status RNG seed must be an unsigned 64-bit integer")
    if (
        not isinstance(prefix_token_ids_sha256, str)
        or len(prefix_token_ids_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prefix_token_ids_sha256)
    ):
        raise ValueError("causal status prefix hash must be a lowercase SHA-256")
    if max(STATUS_CANDIDATE_TOKEN_IDS) >= int(logits.shape[1]):
        raise ValueError("structured-action status token lies outside the vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("causal status logits must be finite")

    probabilities = _top_p_probabilities(
        logits.float(), temperature=float(temperature), top_p=float(top_p)
    )
    results: list[CausalStatusSample] = []
    for row, arm in enumerate(names):
        generator = torch.Generator(device=probabilities.device).manual_seed(rng_seed)
        sampled = torch.multinomial(
            probabilities[row], num_samples=1, generator=generator
        )
        raw_token_id = int(sampled.item())
        top_probability, top_token = torch.max(probabilities[row], dim=-1)
        candidate_probabilities = {
            "PASS": float(probabilities[row, PASS_TOKEN_ID].item()),
            "FAIL": float(probabilities[row, FAIL_TOKEN_ID].item()),
            "SKIP": float(probabilities[row, SKIP_TOKEN_ID].item()),
        }
        candidate_logits = {
            "PASS": float(logits[row, PASS_TOKEN_ID].item()),
            "FAIL": float(logits[row, FAIL_TOKEN_ID].item()),
            "SKIP": float(logits[row, SKIP_TOKEN_ID].item()),
        }
        results.append(
            CausalStatusSample(
                arm=arm,
                raw_token_id=raw_token_id,
                raw_decoded_exact=_decode_exact(tokenizer, [raw_token_id]),
                mapped_action=map_status_token_id(raw_token_id),
                rng_seed=rng_seed,
                prefix_token_ids_sha256=prefix_token_ids_sha256,
                temperature=float(temperature),
                top_p=float(top_p),
                candidate_probabilities=candidate_probabilities,
                candidate_logits=candidate_logits,
                recognized_action_probability_mass=float(
                    sum(candidate_probabilities.values())
                ),
                top_token_id=int(top_token.item()),
                top_token_probability=float(top_probability.item()),
            )
        )
    return tuple(results)


def sample_status_arms_hf(
    model: Any,
    tokenizer: Any,
    prefix_token_ids: Sequence[int],
    *,
    arm_names: Sequence[str],
    steering_batch: Sequence[SteeringSpec | None],
    rng_seed: int,
    temperature: float = STRUCTURED_ACTION_TEMPERATURE,
    top_p: float = STRUCTURED_ACTION_TOP_P,
    use_prefix_cache: bool = True,
) -> tuple[CausalStatusSample, ...]:
    """Replay one exact ``Status:`` prefix and sample every arm with shared RNG."""
    tokens = [int(token) for token in prefix_token_ids]
    if len(tokens) < len(STATUS_PREFIX_TOKEN_IDS) + 1:
        raise ValueError("causal replay prefix is too short")
    if tuple(tokens[-len(STATUS_PREFIX_TOKEN_IDS) :]) != STATUS_PREFIX_TOKEN_IDS:
        raise ValueError("causal replay prefix does not end in the frozen Status: tokens")
    names = _arm_names(arm_names, len(steering_batch))
    if use_prefix_cache:
        prefix_cache = build_prefix_cache(
            model, tokens, batch_size=len(steering_batch)
        )
        logits = forward_logits_batch_with_steering_cached(
            model,
            prefix_cache,
            steering_batch=steering_batch,
        )
        del prefix_cache
    else:
        logits = forward_logits_batch_with_steering(
            model,
            tokens,
            steering_batch=steering_batch,
        )
    if logits.ndim != 3 or logits.shape[0] != len(names):
        raise ValueError("causal replay backend returned invalid batched logits")
    return sample_status_arms_from_logits(
        logits[:, -1, :],
        tokenizer,
        arm_names=names,
        rng_seed=rng_seed,
        prefix_token_ids_sha256=int32_token_sha256(tokens),
        temperature=temperature,
        top_p=top_p,
    )


def forward_status_arm_logits_hf(
    model: Any,
    prefix_token_ids: Sequence[int],
    *,
    steering_batch: Sequence[SteeringSpec | None],
    use_prefix_cache: bool = False,
) -> torch.Tensor:
    """Score one exact ``Status:`` prefix once for every intervention arm.

    The returned tensor is deliberately pre-sampling: callers may replay many
    independent event RNG streams from the same root logits without another
    model forward.  The default avoids a cache prefill so this is one model
    forward for the complete arm batch.
    """
    tokens = [int(token) for token in prefix_token_ids]
    if len(tokens) < len(STATUS_PREFIX_TOKEN_IDS) + 1:
        raise ValueError("causal replay prefix is too short")
    if tuple(tokens[-len(STATUS_PREFIX_TOKEN_IDS) :]) != STATUS_PREFIX_TOKEN_IDS:
        raise ValueError("causal replay prefix does not end in the frozen Status: tokens")
    if not steering_batch:
        raise ValueError("causal replay steering batch must not be empty")
    if use_prefix_cache:
        prefix_cache = build_prefix_cache(model, tokens, batch_size=len(steering_batch))
        try:
            logits = forward_logits_batch_with_steering_cached(
                model, prefix_cache, steering_batch=steering_batch
            )
        finally:
            del prefix_cache
    else:
        logits = forward_logits_batch_with_steering(
            model, tokens, steering_batch=steering_batch
        )
    if logits.ndim != 3 or logits.shape[0] != len(steering_batch):
        raise ValueError("causal replay backend returned invalid batched logits")
    return logits[:, -1, :]


__all__ = [
    "CausalStatusSample",
    "STATUS_CANDIDATE_TOKEN_IDS",
    "build_primary_actuation_steering_batch",
    "forward_status_arm_logits_hf",
    "sample_status_arms_from_logits",
    "sample_status_arms_hf",
]
