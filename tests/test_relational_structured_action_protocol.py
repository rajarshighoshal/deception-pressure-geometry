from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from geoprobe.models.relational_structured_action import (
    CAVEAT_FIELD,
    CAVEAT_PREFIX,
    CAVEAT_PREFIX_TOKEN_IDS,
    EOT_TOKEN_ID,
    FIELD_KIND,
    PROTOCOL_KIND,
    ROW_KIND,
    SCHEMA_VERSION,
    SPEC_KIND,
    STATUS_FIELD,
    STATUS_PREFIX,
    STATUS_PREFIX_TOKEN_IDS,
    TOKENIZER_ARTIFACT_SHA256,
    TURN_KIND,
    build_structured_action_turn_record,
    int32_token_sha256,
    map_caveat_token,
    map_raw_token_id,
    map_status_token,
    structured_action_field_rng,
    validate_schema_identity,
    validate_structured_action_protocol,
    validate_structured_action_tokenizer_binding,
    validate_structured_action_turn_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs/relational_structured_action_protocol.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


class _ExactTokenizer:
    def __init__(self) -> None:
        protocol = _protocol()
        self.encodings = {}
        self.decodes = {}
        for field in protocol["fields"]:
            prefix = field["prefix"]["text"]
            prefix_ids = tuple(field["prefix"]["token_ids"])
            self.encodings[prefix] = prefix_ids
            self.decodes[prefix_ids] = prefix
            for candidate in field["candidates"]:
                token_id = candidate["raw_token_id"]
                decoded = candidate["decoded_exact"]
                self.encodings[prefix + decoded] = prefix_ids + (token_id,)
                self.decodes[(token_id,)] = decoded

    def encode(self, text: str) -> tuple[int, ...]:
        return self.encodings[text]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self.decodes[tuple(token_ids)]


def _turn_record(*, status: int = 51935, caveat: int = 51493) -> dict:
    protocol = _protocol()
    status_rng = structured_action_field_rng(
        source_rng_seed=71,
        source_rng_stream_id="scenario:s0:turn2",
        field_name=STATUS_FIELD,
        protocol=protocol,
    )
    caveat_rng = structured_action_field_rng(
        source_rng_seed=71,
        source_rng_stream_id="scenario:s0:turn2",
        field_name=CAVEAT_FIELD,
        protocol=protocol,
    )
    return build_structured_action_turn_record(
        turn_index=2,
        transcript_token_ids_before_action=[11, 12],
        token_origins_before_action=["input", "input"],
        status_raw_token_id=status,
        status_raw_decoded_exact={
            51935: " PASS",
            34207: " FAIL",
            82504: " SKIP",
            EOT_TOKEN_ID: "<|eot_id|>",
        }.get(status, " unknown"),
        caveat_raw_token_id=caveat,
        caveat_raw_decoded_exact={
            51493: " INCLUDE",
            70500: " OMIT",
            43969: " NONE",
            82504: " SKIP",
            EOT_TOKEN_ID: "<|eot_id|>",
        }.get(caveat, " unknown"),
        status_rng=status_rng,
        caveat_rng=caveat_rng,
        protocol=protocol,
    )


def test_frozen_protocol_identity_and_live_contract() -> None:
    protocol = _protocol()
    validate_structured_action_protocol(protocol)
    assert protocol["schema_version"] == SCHEMA_VERSION == 2
    assert protocol["kind"] == PROTOCOL_KIND
    assert protocol["execution"]["stage"] == "live_sequential_transcript"
    assert protocol["execution"]["field_order"] == [STATUS_FIELD, CAVEAT_FIELD]
    assert protocol["execution"]["primary_prose_allowed"] is False
    assert protocol["generation"] == {
        "sampling_space": "full_vocabulary",
        "draws_per_field": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "retry_attempts": 0,
        "parser_or_generation_rescue_allowed": False,
    }
    assert protocol["supersession"] == {
        "supersedes_for_future_use": ["relational-operational-choice-v1-pre-response"],
        "historical_artifacts_immutable": True,
        "retroactive_relabeling_allowed": False,
    }
    assert protocol["record_schema"] == {
        "spec_kind": SPEC_KIND,
        "row_kind": ROW_KIND,
        "turn_kind": TURN_KIND,
        "field_kind": FIELD_KIND,
        "model_token_origin": "model_sample",
        "environment_token_origin": "environment",
    }


def test_exact_prefix_bytes_hashes_tokens_and_whitespace_footgun() -> None:
    protocol = _protocol()
    status, caveat = protocol["fields"]
    assert STATUS_PREFIX == "Status:"
    assert CAVEAT_PREFIX == "\nCaveat:"
    assert tuple(status["prefix"]["token_ids"]) == STATUS_PREFIX_TOKEN_IDS
    assert tuple(caveat["prefix"]["token_ids"]) == CAVEAT_PREFIX_TOKEN_IDS
    for field in (status, caveat):
        prefix = field["prefix"]
        raw = prefix["text"].encode("utf-8")
        assert raw.hex() == prefix["utf8_hex"]
        assert hashlib.sha256(raw).hexdigest() == prefix["sha256"]
        assert not prefix["text"][-1].isspace()
        assert all(candidate["decoded_exact"].startswith(" ") for candidate in field["candidates"])

    tampered = copy.deepcopy(protocol)
    prefix = tampered["fields"][0]["prefix"]
    prefix["text"] += " "
    prefix["utf8_hex"] = prefix["text"].encode().hex()
    prefix["sha256"] = hashlib.sha256(prefix["text"].encode()).hexdigest()
    prefix["trailing_whitespace"] = True
    with pytest.raises(ValueError, match="field contract"):
        validate_structured_action_protocol(tampered)


def test_mapping_uses_raw_ids_and_retains_unknown_and_eot_as_no_action() -> None:
    assert map_status_token(51935) == "PASS"
    assert map_status_token(34207) == "FAIL"
    assert map_status_token(82504) == "SKIP"
    assert map_caveat_token(51493) == "INCLUDE"
    assert map_caveat_token(70500) == "OMIT"
    assert map_caveat_token(43969) == "NONE"
    assert map_caveat_token(82504) == "SKIP"
    for raw_token_id in (0, 51934, 99999, EOT_TOKEN_ID):
        assert map_status_token(raw_token_id) == "NO_ACTION"
        assert map_caveat_token(raw_token_id) == "NO_ACTION"
    assert map_raw_token_id(STATUS_FIELD, 51493) == "NO_ACTION"
    assert map_raw_token_id(CAVEAT_FIELD, 51935) == "NO_ACTION"


def test_tokenizer_binding_proves_contextual_single_tokens_decodes_and_eot() -> None:
    protocol = _protocol()
    tokenizer = _ExactTokenizer()
    validate_structured_action_tokenizer_binding(
        protocol,
        encode=tokenizer.encode,
        decode=tokenizer.decode,
        tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
        eot_token_id=EOT_TOKEN_ID,
    )

    broken = _ExactTokenizer()
    broken.encodings[STATUS_PREFIX + " PASS"] = STATUS_PREFIX_TOKEN_IDS + (1, 2)
    with pytest.raises(ValueError, match="not one contextual token"):
        validate_structured_action_tokenizer_binding(
            protocol,
            encode=broken.encode,
            decode=broken.decode,
            tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
            eot_token_id=EOT_TOKEN_ID,
        )
    with pytest.raises(ValueError, match="artifact hashes"):
        validate_structured_action_tokenizer_binding(
            protocol,
            encode=tokenizer.encode,
            decode=tokenizer.decode,
            tokenizer_artifact_sha256={**TOKENIZER_ARTIFACT_SHA256, "tokenizer.json": "0" * 64},
            eot_token_id=EOT_TOKEN_ID,
        )
    with pytest.raises(ValueError, match="EOT token ID"):
        validate_structured_action_tokenizer_binding(
            protocol,
            encode=tokenizer.encode,
            decode=tokenizer.decode,
            tokenizer_artifact_sha256=TOKENIZER_ARTIFACT_SHA256,
            eot_token_id=128001,
        )


def test_field_rng_domains_are_distinct_and_deterministic() -> None:
    protocol = _protocol()
    kwargs = {
        "source_rng_seed": 71,
        "source_rng_stream_id": "scenario:s0:turn2",
        "protocol": protocol,
    }
    status_a = structured_action_field_rng(field_name=STATUS_FIELD, **kwargs)
    status_b = structured_action_field_rng(field_name=STATUS_FIELD, **kwargs)
    caveat = structured_action_field_rng(field_name=CAVEAT_FIELD, **kwargs)
    assert status_a == status_b
    assert status_a.domain != caveat.domain
    assert status_a.stream_id != caveat.stream_id
    assert status_a.seed != caveat.seed


def test_turn_record_commits_status_before_caveat_then_environment_eot() -> None:
    protocol = _protocol()
    record = _turn_record()
    validate_structured_action_turn_record(record, protocol)
    status, caveat = record["fields"]
    assert record["committed_status"] == "PASS"
    assert record["committed_caveat"] == "INCLUDE"
    assert status["outcome_role"] == "primary"
    assert caveat["outcome_role"] == "secondary_conditional_descriptive"
    assert status["anchor_token_index"] < caveat["anchor_token_index"]
    assert status["sampled_token_index"] == status["anchor_token_index"] + 1
    assert caveat["sampled_token_index"] == caveat["anchor_token_index"] + 1
    assert record["transcript_token_ids"][status["sampled_token_index"]] == 51935
    assert 51935 in record["transcript_token_ids"][: caveat["sampled_token_index"]]
    assert caveat["anchor_prefix_token_ids_sha256"] == int32_token_sha256(
        record["transcript_token_ids"][: caveat["sampled_token_index"]]
    )
    assert record["environment_eot"] == {
        "token_id": EOT_TOKEN_ID,
        "token_index": len(record["transcript_token_ids"]) - 1,
        "token_origin": "environment",
    }
    assert record["token_origins"][-1] == "environment"


def test_turn_record_keeps_eot_draws_as_committed_no_action() -> None:
    record = _turn_record(status=EOT_TOKEN_ID, caveat=EOT_TOKEN_ID)
    assert record["committed_status"] == "NO_ACTION"
    assert record["committed_caveat"] == "NO_ACTION"
    assert [field["raw_token_id"] for field in record["fields"]] == [
        EOT_TOKEN_ID,
        EOT_TOKEN_ID,
    ]
    assert record["environment_eot"]["token_id"] == EOT_TOKEN_ID


@pytest.mark.parametrize(
    "mutation",
    [
        "swap_fields",
        "raw_id",
        "status_origin",
        "status_anchor",
        "status_hidden_from_caveat",
        "anchor_hash",
        "transcript_hash",
        "origin_hash",
        "committed_status",
        "eot_origin",
        "eot_position",
        "extra_field_key",
    ],
)
def test_turn_record_tampering_fails_closed(mutation: str) -> None:
    protocol = _protocol()
    record = _turn_record()
    status, caveat = record["fields"]
    if mutation == "swap_fields":
        record["fields"] = [caveat, status]
    elif mutation == "raw_id":
        status["raw_token_id"] = 34207
    elif mutation == "status_origin":
        record["token_origins"][status["sampled_token_index"]] = "environment"
    elif mutation == "status_anchor":
        status["anchor_token_index"] += 1
    elif mutation == "status_hidden_from_caveat":
        record["transcript_token_ids"][status["sampled_token_index"]] = 777
        record["transcript_token_ids_sha256"] = int32_token_sha256(
            record["transcript_token_ids"]
        )
    elif mutation == "anchor_hash":
        caveat["anchor_prefix_token_ids_sha256"] = "0" * 64
    elif mutation == "transcript_hash":
        record["transcript_token_ids_sha256"] = "0" * 64
    elif mutation == "origin_hash":
        record["token_origins_sha256"] = "0" * 64
    elif mutation == "committed_status":
        record["committed_status"] = "FAIL"
    elif mutation == "eot_origin":
        record["environment_eot"]["token_origin"] = "model_sample"
    elif mutation == "eot_position":
        record["environment_eot"]["token_index"] -= 1
    elif mutation == "extra_field_key":
        status["decoded_text"] = " PASS"
    with pytest.raises(ValueError):
        validate_structured_action_turn_record(record, protocol)


@pytest.mark.parametrize("kind", [SPEC_KIND, ROW_KIND, TURN_KIND, FIELD_KIND])
def test_schema_identity_helper_accepts_only_v2_exact_kinds(kind: str) -> None:
    validate_schema_identity({"schema_version": SCHEMA_VERSION, "kind": kind}, kind)
    with pytest.raises(ValueError, match="schema identity"):
        validate_schema_identity({"schema_version": 1, "kind": kind}, kind)


def test_protocol_policy_tampering_fails_closed() -> None:
    protocol = _protocol()
    mutations = []
    changed = copy.deepcopy(protocol)
    changed["execution"]["stage"] = "post_hoc_from_immutable_rows"
    mutations.append(changed)
    changed = copy.deepcopy(protocol)
    changed["generation"]["draws_per_field"] = 2
    mutations.append(changed)
    changed = copy.deepcopy(protocol)
    changed["generation"]["retry_attempts"] = 1
    mutations.append(changed)
    changed = copy.deepcopy(protocol)
    changed["rng"]["field_domains"]["caveat"] = changed["rng"]["field_domains"]["status"]
    mutations.append(changed)
    changed = copy.deepcopy(protocol)
    changed["supersession"]["retroactive_relabeling_allowed"] = True
    mutations.append(changed)
    for tampered in mutations:
        with pytest.raises(ValueError):
            validate_structured_action_protocol(tampered)
