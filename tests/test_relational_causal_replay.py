from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import pytest
import torch
from geoprobe.control.relational_pre_status_causal import (
    CAUSAL_ARM_ORDER,
    PRIMARY_ACTUATION_LAYER,
    build_relational_pre_status_causal_arm_bundle,
    PreStatusCausalArmBundle,
    select_primary_actuation_vectors,
)

from geoprobe.models.interface import ResidualSteeringSpec
from geoprobe.models.relational_causal_replay import (
    build_primary_actuation_steering_batch,
    sample_status_arms_from_logits,
    sample_status_arms_hf,
)
from geoprobe.models.relational_structured_action import (
    FAIL_TOKEN_ID,
    PASS_TOKEN_ID,
    SKIP_TOKEN_ID,
    STATUS_PREFIX_TOKEN_IDS,
)


class _Tokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return f" token-{token_ids[0]}"


def _logits(rows: int) -> torch.Tensor:
    logits = torch.full((rows, SKIP_TOKEN_ID + 2), -30.0)
    logits[:, PASS_TOKEN_ID] = 3.0
    logits[:, FAIL_TOKEN_ID] = 2.0
    logits[:, SKIP_TOKEN_ID] = 1.0
    return logits


def test_logits_sampler_is_deterministic_and_uses_common_random_numbers() -> None:
    first = sample_status_arms_from_logits(
        _logits(3),
        _Tokenizer(),
        arm_names=("a", "b", "c"),
        rng_seed=17,
        prefix_token_ids_sha256="a" * 64,
    )
    again = sample_status_arms_from_logits(
        _logits(3),
        _Tokenizer(),
        arm_names=("a", "b", "c"),
        rng_seed=17,
        prefix_token_ids_sha256="a" * 64,
    )

    assert first == again
    assert len({row.raw_token_id for row in first}) == 1
    assert {row.arm for row in first} == {"a", "b", "c"}
    assert all(row.recognized_action_probability_mass > 0.99 for row in first)


def test_logits_sampler_keeps_full_vocabulary_no_action() -> None:
    logits = _logits(1)
    unrelated = 7
    logits[0, unrelated] = 100.0
    sample = sample_status_arms_from_logits(
        logits,
        _Tokenizer(),
        arm_names=("noop",),
        rng_seed=1,
        prefix_token_ids_sha256="b" * 64,
    )[0]

    assert sample.raw_token_id == unrelated
    assert sample.mapped_action == "NO_ACTION"
    assert sample.top_token_id == unrelated


def test_logits_sampler_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="unique"):
        sample_status_arms_from_logits(
            _logits(2),
            _Tokenizer(),
            arm_names=("same", "same"),
            rng_seed=1,
            prefix_token_ids_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="unsigned"):
        sample_status_arms_from_logits(
            _logits(1),
            _Tokenizer(),
            arm_names=("a",),
            rng_seed=-1,
            prefix_token_ids_sha256="c" * 64,
        )


def test_hf_replay_uses_exact_prefix_and_steering_batch(monkeypatch) -> None:
    import geoprobe.models.relational_causal_replay as replay

    seen: dict[str, object] = {}
    fake_cache = SimpleNamespace()

    def fake_build(model, tokens, *, batch_size):
        seen["build"] = (model, tokens, batch_size)
        return fake_cache

    def fake_forward(model, cache, *, steering_batch):
        seen["forward"] = (model, cache, steering_batch)
        return _logits(len(steering_batch))[:, None, :]

    monkeypatch.setattr(replay, "build_prefix_cache", fake_build)
    monkeypatch.setattr(
        replay, "forward_logits_batch_with_steering_cached", fake_forward
    )
    model = object()
    steering = (
        None,
        [
            ResidualSteeringSpec(12, torch.ones(2), 1.0),
            ResidualSteeringSpec(20, -torch.ones(2), 1.0),
        ],
    )
    prefix = [99, *STATUS_PREFIX_TOKEN_IDS]
    results = sample_status_arms_hf(
        model,
        _Tokenizer(),
        prefix,
        arm_names=("noop", "full_h"),
        steering_batch=steering,
        rng_seed=23,
    )

    assert seen["build"] == (model, prefix, 2)
    assert seen["forward"] == (model, fake_cache, steering)
    assert [row.arm for row in results] == ["noop", "full_h"]
    assert len({row.prefix_token_ids_sha256 for row in results}) == 1


def test_hf_replay_rejects_non_status_prefix() -> None:
    with pytest.raises(ValueError, match="frozen Status"):
        sample_status_arms_hf(
            object(),
            _Tokenizer(),
            [1, 2, 3],
            arm_names=("noop",),
            steering_batch=(None,),
            rng_seed=1,
        )


def _sample_event_bundle() -> PreStatusCausalArmBundle:
    h = np.array(
        [
            [1.2, 0.9, 1.4, -0.2, 0.3],
            [0.5, -1.0, 0.2, 2.1, -0.8],
            [1.0, 0.8, 0.5, -0.4, 1.7],
            [2.3, -0.6, 1.9, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    t = np.array(
        [
            [1.0, 0.4, 0.8, -0.5, 0.2],
            [0.1, -0.6, -0.5, 1.0, -0.3],
            [0.8, 0.2, 0.1, -1.1, 0.7],
            [1.8, -1.1, 1.5, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    g = np.array(
        [
            [0.2, 0.1, 0.2, -0.2, 0.2],
            [0.3, 0.7, 0.1, 0.9, 0.2],
            [0.4, 0.0, 0.6, 0.6, 0.9],
            [0.1, 0.4, 0.2, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    return build_relational_pre_status_causal_arm_bundle("event", "fold", h, t, g)


def test_primary_actuation_steering_batch_returns_noop_none_and_layer12_specs() -> None:
    bundle = _sample_event_bundle()
    selected = select_primary_actuation_vectors(bundle.arm_vectors)
    beta = 1.5
    arm_names, steering_batch = build_primary_actuation_steering_batch(
        bundle, beta=beta
    )

    assert tuple(arm_names) == CAUSAL_ARM_ORDER
    assert len(steering_batch) == len(CAUSAL_ARM_ORDER)
    assert sum(spec is not None for spec in steering_batch) == len(CAUSAL_ARM_ORDER) - 1

    for name, spec in zip(arm_names, steering_batch, strict=True):
        if name == "noop":
            assert spec is None
            continue
        assert isinstance(spec, ResidualSteeringSpec)
        assert spec.layer == PRIMARY_ACTUATION_LAYER
        assert spec.alpha == pytest.approx(beta)
        expected_direction = torch.tensor(selected[name], dtype=torch.float32)
        assert torch.equal(spec.direction, expected_direction)
        assert not np.may_share_memory(selected[name], spec.direction.numpy())


def test_primary_actuation_steering_batch_preserves_readonly_source_bundle() -> None:
    bundle = _sample_event_bundle()
    snapshot = {
        key: value.copy()
        for key, value in {
            **{f"arm_{name}": bundle.arm_vectors[name] for name in CAUSAL_ARM_ORDER},
            "h": bundle.h,
            "t": bundle.t,
            "g": bundle.g,
            "s": bundle.s,
            "random_s": bundle.random_s,
        }.items()
    }

    build_primary_actuation_steering_batch(bundle)

    assert tuple(bundle.arm_vectors) == CAUSAL_ARM_ORDER
    assert not bundle.h.flags.writeable
    assert not bundle.t.flags.writeable
    assert not bundle.g.flags.writeable
    assert not bundle.s.flags.writeable
    assert not bundle.random_s.flags.writeable
    for key, value in snapshot.items():
        if key.startswith("arm_"):
            arm_name = key.removeprefix("arm_")
            np.testing.assert_array_equal(bundle.arm_vectors[arm_name], value)
            assert not bundle.arm_vectors[arm_name].flags.writeable
            continue
        np.testing.assert_array_equal(
            getattr(bundle, key), value
        )


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_primary_actuation_steering_batch_rejects_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        build_primary_actuation_steering_batch(_sample_event_bundle(), beta=beta)
