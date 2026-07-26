from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from geoprobe.models.interface import ResidualSteeringSpec


def test_hf_residual_steering_hook_changes_requested_layer_last_token_only():
    import torch.nn as nn

    from geoprobe.models.hf_capture import _residual_steering_hook

    class AddBlock(nn.Module):
        def forward(self, hidden):
            return hidden + 1.0

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([AddBlock() for _ in range(4)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.param = nn.Parameter(torch.zeros(()))

    model = FakeModel()
    hidden = torch.zeros(1, 3, 2)
    steering = ResidualSteeringSpec(
        layer=2,
        direction=torch.tensor([10.0, -2.0]),
        alpha=0.5,
    )

    with _residual_steering_hook(model, steering):
        for block in model.model.layers:
            hidden = block(hidden)

    # Four blocks add +4 everywhere; the hook after block 2 adds alpha * direction to the final
    # token only, and later blocks preserve that offset.
    assert torch.allclose(hidden[0, 0], torch.tensor([4.0, 4.0]))
    assert torch.allclose(hidden[0, 1], torch.tensor([4.0, 4.0]))
    assert torch.allclose(hidden[0, 2], torch.tensor([9.0, 3.0]))


def test_hf_residual_steering_hook_composes_distinct_layers_on_last_token() -> None:
    import torch.nn as nn

    from geoprobe.models.hf_capture import _residual_steering_hook

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([IdentityBlock() for _ in range(20)])
            self.param = nn.Parameter(torch.zeros(()))

    model = FakeModel()
    hidden = torch.zeros(1, 2, 3)
    steering = [
        ResidualSteeringSpec(layer, torch.tensor([float(layer), 1.0, -1.0]), 0.1)
        for layer in (12, 16, 19, 20)
    ]

    with _residual_steering_hook(model, steering):
        for block in model.model.layers:
            hidden = block(hidden)

    assert torch.equal(hidden[0, 0], torch.zeros(3))
    assert torch.allclose(hidden[0, 1], torch.tensor([6.7, 0.4, -0.4]))


def test_hf_residual_steering_hook_rejects_layer_zero_for_generation():
    import torch.nn as nn

    from geoprobe.models.hf_capture import _residual_steering_hook

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Identity()])
            self.param = nn.Parameter(torch.zeros(()))

    with pytest.raises(ValueError, match="HF residual steering layer"):
        with _residual_steering_hook(
            FakeModel(),
            ResidualSteeringSpec(layer=0, direction=torch.ones(1), alpha=1.0),
        ):
            pass


def test_hf_forward_logits_with_steering_scores_forced_margin():
    import torch.nn as nn

    from geoprobe.models.hf_capture import forward_logits_with_steering

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 5, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = FakeModel()
    base = forward_logits_with_steering(model, [9, 9])
    steered = forward_logits_with_steering(
        model,
        [9, 9],
        steering=ResidualSteeringSpec(layer=1, direction=torch.tensor([3.0, 0.0]), alpha=1.0),
    )

    assert float(base[0, -1, 1] - base[0, -1, 2]) == pytest.approx(0.0)
    assert float(steered[0, -1, 1] - steered[0, -1, 2]) == pytest.approx(6.0)


def test_hf_batched_forward_logits_with_steering_matches_scalar():
    import torch.nn as nn

    from geoprobe.models.hf_capture import forward_logits_batch_with_steering, forward_logits_with_steering

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 5, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = FakeModel()
    batch = [
        None,
        ResidualSteeringSpec(layer=1, direction=torch.tensor([3.0, 0.0]), alpha=1.0),
        [ResidualSteeringSpec(layer=1, direction=torch.tensor([1.0, 0.0]), alpha=2.0)],
    ]
    batched = forward_logits_batch_with_steering(model, [9, 9], steering_batch=batch)
    scalar = [
        forward_logits_with_steering(model, [9, 9], steering=steering)
        for steering in batch
    ]

    assert batched.shape[0] == 3
    for idx, expected in enumerate(scalar):
        assert torch.allclose(batched[idx], expected[0])


def test_decision_margins_batches_hf_prompt():
    import torch.nn as nn

    from experiments.decision_token_backend import decision_margin, decision_margins

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 5, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = FakeModel()
    steering_batch = [
        None,
        ResidualSteeringSpec(layer=1, direction=torch.tensor([3.0, 0.0]), alpha=1.0),
        ResidualSteeringSpec(layer=1, direction=torch.tensor([-2.0, 0.0]), alpha=1.0),
    ]
    batched = decision_margins(model, [9, 9], 1, 2, steering_batch=steering_batch, backend="hf")
    scalar = [
        decision_margin(model, [9, 9], 1, 2, steering=steering, backend="hf")
        for steering in steering_batch
    ]

    assert batched == pytest.approx(scalar)


def test_hf_left_pad_batch_positions_last_real_token():
    from geoprobe.models.hf_capture import _left_pad_batch

    input_ids, attention_mask, position_ids = _left_pad_batch(
        [[5, 6, 7], [8]],
        pad_id=0,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert attention_mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert position_ids.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_hf_decision_margins_batch_rows_matches_scalar_different_prompts():
    import torch.nn as nn

    from experiments.decision_token_backend import decision_margin
    from geoprobe.models.hf_capture import decision_margins_batch_rows

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.param = nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask=None, use_cache=False, position_ids=None):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            hidden[:, :, 0] = input_ids.float()
            for block in self.model.layers:
                hidden = block(hidden)
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 5)
            logits[:, :, 1] = hidden[:, :, 0]
            logits[:, :, 2] = -hidden[:, :, 0]
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    prompts = [[4, 5], [9]]
    steering_batch = [
        ResidualSteeringSpec(layer=1, direction=torch.tensor([1.0, 0.0]), alpha=3.0),
        ResidualSteeringSpec(layer=1, direction=torch.tensor([-2.0, 0.0]), alpha=1.0),
    ]

    batched = decision_margins_batch_rows(
        model,
        prompts,
        1,
        2,
        steering_batch=steering_batch,
        pad_id=0,
    )
    scalar = [
        decision_margin(model, prompt, 1, 2, steering=steering, backend="hf")
        for prompt, steering in zip(prompts, steering_batch)
    ]

    assert batched == pytest.approx(scalar)


def test_hf_generate_greedy_batch_matches_scalar_fake_cache():
    import torch.nn as nn

    from geoprobe.models.hf_capture import generate_greedy_batch, generate_greedy_with_steering

    class FakeCache:
        pass

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 5

        def decode(self, token_ids, skip_special_tokens=True):
            table = {3: " A", 4: " B"}
            return "".join(table.get(int(token_id), "") for token_id in token_ids)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, position_ids=None):
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 6)
            for row_idx in range(input_ids.shape[0]):
                last = int(input_ids[row_idx, -1])
                next_token = {1: 3, 2: 4, 3: 0, 4: 0}.get(last, 0)
                logits[row_idx, -1, next_token] = 10.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=FakeCache() if use_cache else past_key_values,
            )

    model = FakeModel()
    tokenizer = FakeTokenizer()
    prompts = [[10, 1], [2]]

    batched = generate_greedy_batch(model, tokenizer, prompts, max_new_tokens=4)
    scalar = [
        generate_greedy_with_steering(
            model,
            tokenizer,
            prompt,
            max_new_tokens=4,
            steering=None,
        )
        for prompt in prompts
    ]

    assert batched == scalar == ["A", "B"]


def test_generation_batch_config_requires_smoke_guard():
    from experiments.decision_token_backend import (
        GENERATION_PROTOCOL_HF_BATCHED,
        generation_protocol_name,
        validate_generation_batch_config,
    )

    with pytest.raises(ValueError, match="requires --verify-generation-smoke"):
        validate_generation_batch_config(
            backend="hf",
            generation_batch_size=8,
            verify_generation_smoke=False,
        )

    validate_generation_batch_config(
        backend="hf",
        generation_batch_size=8,
        verify_generation_smoke=True,
    )
    assert generation_protocol_name(backend="hf", generation_batch_size=8) == GENERATION_PROTOCOL_HF_BATCHED


def test_cached_decision_margins_match_full_fake_model():
    import torch.nn as nn

    from experiments.decision_token_backend import (
        build_decision_prefix_cache,
        decision_margin,
        decision_margins,
    )

    class FakeCache:
        def __init__(self, seq_length, batch_size):
            self.seq_length = seq_length
            self.batch_size = batch_size

        def get_seq_length(self):
            return self.seq_length

        def batch_repeat_interleave(self, repeats):
            self.batch_size *= int(repeats)

        def crop(self, max_length):
            self.seq_length = int(max_length)

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 5, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(
                logits=self.lm_head(hidden),
                past_key_values=FakeCache(input_ids.shape[1], input_ids.shape[0]) if use_cache else past_key_values,
            )

    model = FakeModel()
    steering_batch = [
        ResidualSteeringSpec(layer=1, direction=torch.tensor([3.0, 0.0]), alpha=1.0),
        ResidualSteeringSpec(layer=1, direction=torch.tensor([-2.0, 0.0]), alpha=1.0),
    ]
    cache = build_decision_prefix_cache(model, [9, 9, 9], backend="hf", batch_size=2)
    cached = decision_margins(
        model,
        [9, 9, 9],
        1,
        2,
        steering_batch=steering_batch,
        backend="hf",
        prefix_cache=cache,
    )
    full = [
        decision_margin(model, [9, 9, 9], 1, 2, steering=steering, backend="hf")
        for steering in steering_batch
    ]

    assert cached == pytest.approx(full)
    assert cache.past_key_values.seq_length == 2


def test_decision_token_hf_forces_status_then_completes_unsteered():
    import torch.nn as nn

    from experiments.decision_token_backend import choose_status_then_complete

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            return [10, 11]

        def encode(self, text, add_special_tokens=False):
            return [20]

        def decode(self, token_ids, skip_special_tokens=True):
            table = {7: " done"}
            return "".join(table.get(int(item), "") for item in token_ids)

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 8, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])
            self.seen_status_prefix = False

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            last_id = int(input_ids[0, -1])
            if last_id == 1:
                self.seen_status_prefix = True
                logits[0, -1, 7] = 9.0
                return SimpleNamespace(logits=logits)
            if last_id == 7:
                logits[0, -1, 0] = 9.0
                return SimpleNamespace(logits=logits)
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = FakeModel()
    reply, decision = choose_status_then_complete(
        model,
        FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        pass_id=1,
        fail_id=2,
        steering=ResidualSteeringSpec(layer=1, direction=torch.tensor([2.0, 0.0]), alpha=1.0),
        max_new_tokens=4,
        max_generation_seconds=None,
        backend="hf",
    )

    assert decision["forced_status"] == "PASS"
    assert decision["backend"] == "hf"
    assert decision["margin"] > 0
    assert reply == '{\n  "reported_status": "PASSdone'
    assert model.seen_status_prefix


def test_decision_token_hf_passes_no_steering_to_continuation(monkeypatch):
    import torch.nn as nn

    # decision_token_backend moved to geoprobe.models.decision_backend (Phase 2); the internal
    # helper below is monkeypatched where the function under test resolves it (the src module).
    import geoprobe.models.decision_backend as backend
    from experiments.decision_token_backend import choose_status_then_complete  # via re-export shim

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [10, 11]

        def encode(self, text, add_special_tokens=False):
            return [20]

    class IdentityBlock(nn.Module):
        def forward(self, hidden):
            return hidden

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([IdentityBlock()])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(2, 3, bias=False)
            self.lm_head.weight.data.zero_()
            self.lm_head.weight.data[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.data[2] = torch.tensor([-1.0, 0.0])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
            for block in self.model.layers:
                hidden = block(hidden)
            return SimpleNamespace(logits=self.lm_head(hidden))

    def fake_generate(model, tokenizer, token_ids, *, max_new_tokens, steering, max_generation_seconds):
        assert steering is None
        assert token_ids[-1] == 1
        return " continuation"

    monkeypatch.setattr(backend, "hf_generate_greedy_with_steering", fake_generate)
    reply, decision = choose_status_then_complete(
        FakeModel(),
        FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        pass_id=1,
        fail_id=2,
        steering=ResidualSteeringSpec(layer=1, direction=torch.tensor([2.0, 0.0]), alpha=1.0),
        max_new_tokens=4,
        max_generation_seconds=None,
        backend="hf",
    )

    assert decision["forced_status"] == "PASS"
    assert reply == '{\n  "reported_status": "PASS continuation'


def test_mlx_forward_steering_changes_requested_layer_last_token_only():
    mx = pytest.importorskip("mlx.core")

    from geoprobe.models.mlx_capture import _forward_logits_with_steering

    class FakeLayer:
        def __call__(self, h, mask, cache=None):
            return h + 1.0

    class Inner:
        def __init__(self):
            self.layers = [FakeLayer(), FakeLayer()]

        def embed_tokens(self, inputs):
            return mx.zeros((1, inputs.shape[1], 2), dtype=mx.float32)

        def norm(self, h):
            return h

    class FakeModel:
        def __init__(self):
            self.model = Inner()

        def lm_head(self, h):
            return h

    model = FakeModel()
    steering = ResidualSteeringSpec(
        layer=1,
        direction=torch.tensor([3.0, -1.0]),
        alpha=2.0,
    )
    try:
        logits = _forward_logits_with_steering(model, [1, 2, 3], steering=steering)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("MLX installed but Metal is unavailable in this sandbox")
        raise

    arr = torch.from_numpy(__import__("numpy").array(logits, dtype="float32"))
    assert torch.allclose(arr[0, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(arr[0, 1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(arr[0, 2], torch.tensor([8.0, 0.0]))
