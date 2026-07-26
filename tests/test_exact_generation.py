from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from geoprobe.models import exact_generation
from geoprobe.models.exact_generation import (
    ExactGeneration,
    GenerationCapError,
    generate_exact_greedy_batch,
    generate_exact_sample_batch,
)


class _Tokenizer:
    eos_token_id = 7
    pad_token_id = 0

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids)

    @staticmethod
    def convert_tokens_to_ids(_token):
        return 7


def test_exact_generation_retains_eos(monkeypatch) -> None:
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval()

    class Result:
        sequences = torch.tensor([[1, 2, 5, 7, 0, 0], [3, 4, 6, 7, 0, 0]])

    monkeypatch.setattr(model, "generate", lambda **_kwargs: Result())
    output = generate_exact_greedy_batch(
        model, _Tokenizer(), [[1, 2], [3, 4]], max_new_tokens=4,
    )
    assert output[0].token_ids == (5, 7)
    assert output[0].text == "5"
    assert output[1].token_ids == (6, 7)
    assert output[1].stop_token_id == 7


def test_exact_generation_rejects_missing_terminal(monkeypatch) -> None:
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval()

    class Result:
        sequences = torch.tensor([[1, 2, 5, 6]])

    monkeypatch.setattr(model, "generate", lambda **_kwargs: Result())
    try:
        generate_exact_greedy_batch(model, _Tokenizer(), [[1, 2]], max_new_tokens=2)
    except GenerationCapError as error:
        assert "without EOT" in str(error)
        assert error.rows[0]["row_index"] == 0
        assert error.rows[0]["token_ids"] == (5, 6)
        assert error.rows[0]["partial_text"] == "5 6"
    else:
        raise AssertionError("missing terminal token was accepted")


def test_per_row_sampling_stream_is_batch_composition_invariant() -> None:
    torch.manual_seed(11)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
    )).eval()
    batched = generate_exact_sample_batch(
        model, _Tokenizer(), [[1, 2], [3, 4, 5]], seeds=[17, 29],
        temperature=0.7, top_p=0.95, max_new_tokens=64,
    )
    scalar = generate_exact_sample_batch(
        model, _Tokenizer(), [[1, 2]], seeds=[17], temperature=0.7, top_p=0.95,
        max_new_tokens=64,
    )[0]
    assert scalar.token_ids == batched[0].token_ids


def test_independent_sampling_never_co_batches_unrelated_prompts(monkeypatch) -> None:
    calls = []

    def fake_batch(_model, _tokenizer, token_ids_batch, *, seeds, **_kwargs):
        calls.append((token_ids_batch, seeds))
        return [ExactGeneration((5, 7), "5", 7, "eot_token", "5")]

    monkeypatch.setattr(exact_generation, "generate_exact_sample_batch", fake_batch)
    output = exact_generation.generate_exact_sample_independent(
        object(), _Tokenizer(), [[1, 2], [3, 4]], seeds=[17, 29],
        temperature=0.7, top_p=0.95, max_new_tokens=8,
    )
    assert [item.token_ids for item in output] == [(5, 7), (5, 7)]
    assert calls == [([ [1, 2] ], [17]), ([ [3, 4] ], [29])]
