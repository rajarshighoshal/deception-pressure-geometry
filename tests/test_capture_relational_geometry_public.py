from __future__ import annotations

import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.capture_relational_geometry import (
    capture_batches,
    capture_contract,
    establish_capture_contract,
    selected_rows,
)
from geoprobe.models.relational_capture import (
    capture_relational_batch,
    set_eager_attention,
)


def _row(conversation_id: str, length: int) -> dict:
    return {
        "conversation_id": conversation_id,
        "token_ids": list(range(1, length + 1)),
    }


def test_selection_and_batches_preserve_exact_equal_length_groups(tmp_path) -> None:
    rows = [_row("long", 5), _row("short-b", 3), _row("short-a", 3)]
    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("short-b\nlong\nshort-a\n", encoding="utf-8")
    selected = selected_rows(rows, ids_path, None)
    assert [row["conversation_id"] for row in selected] == [
        "short-a",
        "short-b",
        "long",
    ]
    assert [len(batch) for batch in capture_batches(selected, 2)] == [2, 1]
    assert all(len({len(row["token_ids"]) for row in batch}) == 1 for batch in capture_batches(selected, 2))

    ids_path.write_text("short-a\nshort-a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        selected_rows(rows, ids_path, None)


def test_contract_is_source_bound_and_resume_is_fail_closed(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    protocol_path = tmp_path / "protocol.json"
    rows_path.write_text('{"conversation_id":"one"}\n', encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    model_artifact = {
        "artifact_sha256": "1" * 64,
        "weights_sha256": "2" * 64,
        "model_config_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
    }
    target_rows = [_row("one", 3)]
    contract = capture_contract(
        rows_path=rows_path,
        protocol_path=protocol_path,
        model_artifact=model_artifact,
        target_rows=target_rows,
        layers=[1, 3],
        dtype="bfloat16",
        device="cuda",
        batch_size=1,
        parity_rows=1,
        residual_parity_atol=0.02,
        attention_parity_atol=0.002,
        attention_row_sum_atol=0.02,
    )
    assert contract["capture"]["attention_representation"] == (
        "lossless_causal_lower_triangle_per_head"
    )
    path = tmp_path / "capture" / "capture_contract.json"
    establish_capture_contract(path, contract, resume=True)
    establish_capture_contract(path, contract, resume=True)
    changed = json.loads(json.dumps(contract))
    changed["capture"]["batch_size"] = 2
    with pytest.raises(ValueError, match="source identity changed"):
        establish_capture_contract(path, changed, resume=True)


def test_capture_uses_every_head_and_requested_residual_layer() -> None:
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_implementation="sdpa",
        )
    ).to(dtype=torch.bfloat16).eval()
    set_eager_attention(model)
    captured = capture_relational_batch(
        model,
        [[1, 2, 3, 4]],
        layers=[1, 3],
        pad_token_id=0,
    )[0]
    assert captured.residuals[1].shape == (4, 32)
    assert captured.residuals[3].shape == (4, 32)
    assert captured.attentions[1].shape == (4, 10)
    assert captured.attentions[3].shape == (4, 10)
