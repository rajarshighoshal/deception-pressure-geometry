from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from geoprobe.io import file_sha256
from geoprobe.models.relational_capture import (
    CapturedRelationalRow,
    canonical_json_sha256,
    validate_relational_row_artifact,
    write_relational_row,
)


def _row() -> dict:
    token_ids = [10, 11, 12, 13]
    fields = [
        {
            "field_name": "status",
            "field_event_id": "turn-0:status",
            "anchor_token_index": 0,
            "sampled_token_index": 1,
            "raw_token_id": 11,
            "raw_token_origin": "model_sample",
            "rng_stream_id": "status-stream",
            "rng_seed": 17,
        },
        {
            "field_name": "caveat",
            "field_event_id": "turn-0:caveat",
            "anchor_token_index": 1,
            "sampled_token_index": 2,
            "raw_token_id": 12,
            "raw_token_origin": "model_sample",
            "rng_stream_id": "caveat-stream",
            "rng_seed": 19,
        },
    ]
    return {
        "schema_version": 2,
        "kind": "relational_structured_action_row",
        "conversation_id": "structured:s0:nn",
        "family": "structured",
        "scenario_id": "s0",
        "orbit_id": "s0:nn",
        "token_ids": token_ids,
        "stop_token_id": 13,
        "stop_token_index": 3,
        "token_role_ids": [0, 2, 2, 2],
        "token_turn_ids": [-1, 0, 0, 0],
        "token_message_ids": [0, 2, 2, 2],
        "token_span_flags": [0, 16, 32, 0],
        "token_origins": ["chat_source", "model_sample", "model_sample", "environment"],
        "token_origin_details": [
            "chat_source",
            "model_sampled_status_action",
            "model_sampled_caveat_action",
            "environment_eot",
        ],
        "status_outcome": "honest",
        "intervention_program": "NN",
        "assistant_action_turn_records": [{
            "turn_index": 0,
            "turn_event_id": "turn-0",
            "fields": fields,
            "environment_eot": {
                "token_id": 13,
                "token_index": 3,
                "token_origin": "environment",
            },
        }],
        "action_anchors": [
            {
                "turn_index": 0,
                "field_name": "status",
                "field_event_id": "turn-0:status",
                "anchor_token_index": 0,
                "anchor_token_id": 10,
                "sampled_token_index": 1,
                "anchor_semantic_position": "immediately_before_status_sample",
            },
            {
                "turn_index": 0,
                "field_name": "caveat",
                "field_event_id": "turn-0:caveat",
                "anchor_token_index": 1,
                "anchor_token_id": 11,
                "sampled_token_index": 2,
                "anchor_semantic_position": (
                    "immediately_before_caveat_sample_status_visible"
                ),
            },
        ],
        "action_token_spans": [
            {
                "turn_index": 0,
                "field_name": "status",
                "token_start": 1,
                "token_end": 2,
                "token_origin": "model_sample",
            },
            {
                "turn_index": 0,
                "field_name": "caveat",
                "token_start": 2,
                "token_end": 3,
                "token_origin": "model_sample",
            },
        ],
    }


def _captured() -> CapturedRelationalRow:
    return CapturedRelationalRow(
        token_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        residuals={1: torch.arange(8, dtype=torch.float32).reshape(4, 2)},
        attentions={1: torch.ones((1, 10), dtype=torch.float32)},
    )


def _kwargs() -> dict:
    return {
        "capture_contract_sha256": "a" * 64,
        "layers": [1],
        "hidden_size": 2,
        "n_attention_heads": 1,
        "residual_dtype": "float32",
        "attention_dtype": "float32",
    }


def test_structured_capture_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    row = _row()
    record = write_relational_row(tmp_path, row, _captured(), resume=False, **_kwargs())
    record_path = next((tmp_path / "rows").glob("*.json"))
    assert record["schema_version"] == 2
    assert record["source_schema_version"] == 2
    assert record["source_kind"] == "relational_structured_action_row"
    assert validate_relational_row_artifact(record_path, expected_row=row) == record

    tensor_path = tmp_path / record["tensor_path"]
    tensors = load_file(tensor_path)
    assert tensors["token_origin_ids"].tolist() == [0, 2, 2, 1]
    assert tensors["token_origin_detail_ids"].tolist() == [0, 2, 4, 5]

    tampered = json.loads(record_path.read_text())
    tampered["structured_metadata"]["token_origin_id_mapping"]["model"] = 9
    tampered["structured_metadata_sha256"] = canonical_json_sha256(
        tampered["structured_metadata"]
    )
    record_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="mapping"):
        validate_relational_row_artifact(record_path, expected_row=row)


def test_structured_capture_rejects_tampered_numeric_origin_array(tmp_path) -> None:
    row = _row()
    record = write_relational_row(tmp_path, row, _captured(), resume=False, **_kwargs())
    record_path = next((tmp_path / "rows").glob("*.json"))
    tensor_path = tmp_path / record["tensor_path"]
    tensors = load_file(tensor_path)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    tensors["token_origin_ids"][1] = 1
    save_file(tensors, tensor_path, metadata=metadata)
    record["tensor_bytes"] = tensor_path.stat().st_size
    record["tensor_sha256"] = file_sha256(tensor_path)
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="token-origin arrays"):
        validate_relational_row_artifact(record_path, expected_row=row)
