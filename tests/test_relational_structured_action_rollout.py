from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import pytest
import torch

from geoprobe.data.relational_structured_action import (
    STRUCTURED_ACTION_SPEC_KIND,
    expected_structured_action_counts,
    select_relational_structured_action_specs,
    structured_action_token_lengths,
    validate_relational_structured_action_specs,
)
from geoprobe.models.relational_structured_action import (
    CAVEAT_PREFIX_TOKEN_IDS,
    EOT_TOKEN_ID,
    FAIL_TOKEN_ID,
    INCLUDE_TOKEN_ID,
    OMIT_TOKEN_ID,
    PASS_TOKEN_ID,
    STATUS_PREFIX_TOKEN_IDS,
    canonical_json_sha256,
)
from geoprobe.models.relational_structured_action_rollout import (
    next_token_distribution,
    rollout_relational_structured_action_scenario,
    sample_one_token_independent,
    validate_relational_structured_action_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "data/raw/deception_intent/graded_ks_150fam_l7_scenarios.jsonl"
PROTOCOL = REPO_ROOT / "configs/relational_structured_action_protocol.json"


class _Tokenizer:
    start_header_id = 120_001
    end_header_id = 120_002
    bos_token_id = 120_003
    pad_token_id = 0
    _char_base = 200_000
    _special = {
        "<|start_header_id|>": start_header_id,
        "<|end_header_id|>": end_header_id,
        "<|eot_id|>": EOT_TOKEN_ID,
    }
    _piece_decode = {
        2583: "Status",
        25: ":",
        198: "\n",
        34: "C",
        525: "ave",
        266: "at",
        PASS_TOKEN_ID: " PASS",
        FAIL_TOKEN_ID: " FAIL",
        82504: " SKIP",
        INCLUDE_TOKEN_ID: " INCLUDE",
        OMIT_TOKEN_ID: " OMIT",
        43969: " NONE",
    }

    def convert_tokens_to_ids(self, value: str) -> int | None:
        return self._special.get(value)

    def _encode_rendered(self, text: str) -> list[int]:
        output: list[int] = []
        cursor = 0
        markers = sorted(self._special, key=len, reverse=True)
        while cursor < len(text):
            marker = next(
                (value for value in markers if text.startswith(value, cursor)), None
            )
            if marker is None:
                output.append(self._char_base + ord(text[cursor]))
                cursor += 1
            else:
                output.append(self._special[marker])
                cursor += len(marker)
        return output

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text == "Status:":
            return list(STATUS_PREFIX_TOKEN_IDS)
        if text == "\nCaveat:":
            return list(CAVEAT_PREFIX_TOKEN_IDS)
        for prefix, prefix_ids, candidates in (
            (
                "Status:",
                STATUS_PREFIX_TOKEN_IDS,
                {" PASS": PASS_TOKEN_ID, " FAIL": FAIL_TOKEN_ID, " SKIP": 82504},
            ),
            (
                "\nCaveat:",
                CAVEAT_PREFIX_TOKEN_IDS,
                {
                    " INCLUDE": INCLUDE_TOKEN_ID,
                    " OMIT": OMIT_TOKEN_ID,
                    " NONE": 43969,
                    " SKIP": 82504,
                },
            ),
        ):
            for decoded, token_id in candidates.items():
                if text == prefix + decoded:
                    return [*prefix_ids, token_id]
        return self._encode_rendered(text)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        reverse_special = {value: key for key, value in self._special.items()}
        pieces: list[str] = []
        for token in token_ids:
            value = int(token)
            if value in self._piece_decode:
                pieces.append(self._piece_decode[value])
            elif value in reverse_special:
                pieces.append(reverse_special[value])
            elif value >= self._char_base:
                pieces.append(chr(value - self._char_base))
            else:
                pieces.append(f"<{value}>")
        return "".join(pieces)

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        tokens = [self.bos_token_id]
        for message in messages:
            tokens.extend(
                [
                    self.start_header_id,
                    *self._encode_rendered(str(message["role"])),
                    self.end_header_id,
                    *self._encode_rendered("\n\n" + str(message["content"])),
                    EOT_TOKEN_ID,
                ]
            )
        tokens.extend(
            [
                self.start_header_id,
                *self._encode_rendered("assistant"),
                self.end_header_id,
                *self._encode_rendered("\n\n"),
            ]
        )
        return tokens


class _Sampler:
    def __init__(self) -> None:
        self.calls: list[tuple[list[list[int]], list[int]]] = []

    def __call__(
        self,
        token_ids_batch: Sequence[Sequence[int]],
        *,
        seeds: Sequence[int],
        temperature: float,
        top_p: float,
    ) -> list[int]:
        assert temperature == 0.7
        assert top_p == 0.95
        prompts = [[int(token) for token in values] for values in token_ids_batch]
        self.calls.append((prompts, [int(seed) for seed in seeds]))
        output: list[int] = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            if tuple(prompt[-len(STATUS_PREFIX_TOKEN_IDS):]) == STATUS_PREFIX_TOKEN_IDS:
                output.append(PASS_TOKEN_ID if (sum(prompt) + seed) % 2 else FAIL_TOKEN_ID)
            elif tuple(prompt[-len(CAVEAT_PREFIX_TOKEN_IDS):]) == CAVEAT_PREFIX_TOKEN_IDS:
                output.append(INCLUDE_TOKEN_ID if (sum(prompt) + seed) % 2 else OMIT_TOKEN_ID)
            else:
                raise AssertionError("sampler received an unfrozen field prefix")
        return output


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 32, device=input_ids.device)
        logits[:, -1, 7] = 5.0
        logits[:, -1, 11] = 4.0
        return SimpleNamespace(logits=logits)


@pytest.fixture(scope="module")
def specs() -> list[dict]:
    if not SCENARIOS.exists():
        pytest.skip(
            "requires local data/raw/deception_intent scenarios file (gitignored)"
        )
    scenarios = [
        json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()
    ]
    return select_relational_structured_action_specs(scenarios)


@pytest.fixture(scope="module")
def scenario_specs(specs: list[dict]) -> list[dict]:
    scenario_id = str(specs[0]["scenario_id"])
    return [spec for spec in specs if spec["scenario_id"] == scenario_id]


@pytest.fixture(scope="module")
def rollout(
    scenario_specs: list[dict],
) -> tuple[list[dict], _Sampler]:
    sampler = _Sampler()
    rows = rollout_relational_structured_action_scenario(
        None,
        _Tokenizer(),
        scenario_specs,
        protocol=json.loads(PROTOCOL.read_text()),
        one_token_sampler=sampler,
    )
    return rows, sampler


def _field_records(rows: Sequence[dict]) -> list[dict]:
    return [
        field
        for row in rows
        for turn in row["assistant_action_turn_records"]
        for field in turn["fields"]
    ]


def test_full_spec_bank_reuses_selection_and_declares_exact_record_counts(
    specs: list[dict],
) -> None:
    validate_relational_structured_action_specs(specs)
    assert len(specs) == 600
    assert expected_structured_action_counts() == {
        "rows": 600,
        "status_records": 2400,
        "caveat_records": 2400,
        "unique_status_events": 1680,
        "unique_caveat_events": 1680,
    }
    assert {spec["kind"] for spec in specs} == {STRUCTURED_ACTION_SPEC_KIND}
    assert {spec["schema_version"] for spec in specs} == {2}
    assert not any(
        key in spec
        for spec in specs
        for key in ("slot_actions", "slot_payloads", "rng_seed_schedule")
    )
    assert Counter(spec["intervention_program"] for spec in specs) == {
        "AN": 120,
        "AA": 120,
        "AB": 120,
        "BA": 120,
        "NN": 60,
        "D2N": 60,
    }


def test_exact_lengths_are_outcome_independent_and_match_rollout(
    scenario_specs: list[dict], rollout: tuple[list[dict], _Sampler]
) -> None:
    rows, _sampler = rollout
    lengths = structured_action_token_lengths(_Tokenizer(), scenario_specs)
    assert {
        row["conversation_id"]: len(row["token_ids"]) for row in rows
    } == lengths


def test_scenario_rollout_has_80_records_56_unique_events_and_exact_clones(
    rollout: tuple[list[dict], _Sampler],
) -> None:
    rows, sampler = rollout
    assert len(rows) == 10
    records = _field_records(rows)
    assert len(records) == 80
    assert Counter(record["field_name"] for record in records) == {
        "status": 40,
        "caveat": 40,
    }
    assert len({record["field_event_id"] for record in records}) == 56
    assert len(sampler.calls) == 8
    assert sum(len(prompts) for prompts, _seeds in sampler.calls) == 56

    by_program = {
        (row["sample_index"], row["intervention_program"]): row for row in rows
    }
    for sample_index in (0, 1):
        first_a = [by_program[(sample_index, program)] for program in ("AN", "AA", "AB")]
        for field_index in (0, 1):
            signatures = {
                (
                    row["assistant_action_turn_records"][1]["fields"][field_index][
                        "field_event_id"
                    ],
                    row["assistant_action_turn_records"][1]["fields"][field_index][
                        "raw_token_id"
                    ],
                    row["assistant_action_turn_records"][1]["fields"][field_index][
                        "anchor_prefix_token_ids_sha256"
                    ],
                )
                for row in first_a
            }
            assert len(signatures) == 1


def test_actions_enter_later_context_and_anchors_are_strictly_causal(
    rollout: tuple[list[dict], _Sampler],
) -> None:
    rows, _sampler = rollout
    for row in rows:
        turns = row["assistant_action_turn_records"]
        for turn_index, turn in enumerate(turns):
            status, caveat = turn["fields"]
            assert status["sampled_token_index"] == status["anchor_token_index"] + 1
            assert caveat["sampled_token_index"] == caveat["anchor_token_index"] + 1
            assert row["token_ids"][status["anchor_token_index"]] == 25
            assert row["token_ids"][caveat["anchor_token_index"]] == 25
            assert (
                status["sampled_token_index"]
                < caveat["anchor_token_index"]
                < caveat["sampled_token_index"]
            )
            assert (
                row["token_ids"][status["sampled_token_index"]]
                == status["raw_token_id"]
            )
            if turn_index:
                previous_status = turns[turn_index - 1]["fields"][0]
                before = int(turn["transcript_before_action_token_count"])
                assert previous_status["sampled_token_index"] < before
                assert (
                    row["token_ids"][previous_status["sampled_token_index"]]
                    == previous_status["raw_token_id"]
                )


def test_environment_eot_and_detailed_origins_are_aligned(
    rollout: tuple[list[dict], _Sampler],
) -> None:
    rows, _sampler = rollout
    for row in rows:
        assert len(row["token_origin_details"]) == len(row["token_ids"])
        for turn in row["assistant_action_turn_records"]:
            eot_index = turn["environment_eot"]["token_index"]
            assert row["token_ids"][eot_index] == EOT_TOKEN_ID
            assert row["token_origin_details"][eot_index] == "environment_eot"
            status, caveat = turn["fields"]
            assert (
                row["token_origin_details"][status["sampled_token_index"]]
                == "model_sampled_status_action"
            )
            assert (
                row["token_origin_details"][caveat["sampled_token_index"]]
                == "model_sampled_caveat_action"
            )


def test_messages_include_exact_sampled_assistant_records(
    rollout: tuple[list[dict], _Sampler],
) -> None:
    rows, _sampler = rollout
    for row in rows:
        assert [message["role"] for message in row["messages"]] == [
            "system",
            "user", "assistant",
            "user", "assistant",
            "user", "assistant",
            "user", "assistant",
        ]
        assert max(row["token_message_ids"]) == 8
        for turn_index, turn in enumerate(row["assistant_action_turn_records"]):
            status, caveat = turn["fields"]
            assert row["messages"][2 * turn_index + 2] == {
                "role": "assistant",
                "content": (
                    "Status:"
                    + status["raw_decoded_exact"]
                    + "\nCaveat:"
                    + caveat["raw_decoded_exact"]
                ),
            }


def test_rollout_is_deterministic_and_schema_tampering_fails_closed(
    scenario_specs: list[dict], rollout: tuple[list[dict], _Sampler]
) -> None:
    rows, _sampler = rollout
    repeated = rollout_relational_structured_action_scenario(
        None,
        _Tokenizer(),
        scenario_specs,
        protocol=json.loads(PROTOCOL.read_text()),
        one_token_sampler=_Sampler(),
    )
    assert [row["token_sha256"] for row in repeated] == [
        row["token_sha256"] for row in rows
    ]
    tampered = copy.deepcopy(rows[0])
    tampered["assistant_action_turn_records"][0]["fields"][0]["raw_token_id"] = 7
    with pytest.raises(ValueError, match="row hash mismatch"):
        validate_relational_structured_action_row(
            tampered, json.loads(PROTOCOL.read_text())
        )

    tampered["row_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="mapped action|raw token"):
        validate_relational_structured_action_row(
            tampered, json.loads(PROTOCOL.read_text())
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row["token_role_ids"].__setitem__(0, 999), "role/turn"),
        (
            lambda row: row["token_origin_details"].__setitem__(0, "foreign"),
            "detailed token origins",
        ),
        (
            lambda row: row["token_span_flags"].__setitem__(
                row["assistant_action_turn_records"][0]["fields"][0][
                    "sampled_token_index"
                ],
                0,
            ),
            "turn annotations",
        ),
        (
            lambda row: row["assistant_action_turn_records"][0]["fields"][0].__setitem__(
                "rng_seed", 7
            ),
            "field RNG",
        ),
        (
            lambda row: row["assistant_action_turn_records"][0].__setitem__(
                "turn_event_id", "foreign"
            ),
            "turn event ID",
        ),
    ],
)
def test_rehashed_typed_metadata_tampering_fails_closed(
    rollout: tuple[list[dict], _Sampler],
    mutate: Callable[[dict], None],
    match: str,
) -> None:
    rows, _sampler = rollout
    tampered = copy.deepcopy(rows[0])
    mutate(tampered)
    tampered["row_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match=match):
        validate_relational_structured_action_row(
            tampered, json.loads(PROTOCOL.read_text())
        )


def test_missing_source_spec_hashes_fail_closed(
    rollout: tuple[list[dict], _Sampler],
) -> None:
    rows, _sampler = rollout
    tampered = copy.deepcopy(rows[0])
    del tampered["source_spec_sha256"]
    del tampered["spec_sha256"]
    tampered["row_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "row_sha256"}
    )
    with pytest.raises(ValueError, match="bound to its source spec"):
        validate_relational_structured_action_row(
            tampered, json.loads(PROTOCOL.read_text())
        )


def test_candidate_mass_diagnostic_and_scalar_sampler_share_distribution() -> None:
    model = _TinyModel()
    tokenizer = _Tokenizer()
    prompts = [[1, 2, 3], [4, 5]]
    diagnostics = next_token_distribution(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=[7, 11],
        temperature=0.7,
        top_p=0.95,
    )
    assert len(diagnostics) == 2
    assert all(result["candidate_probability_mass"] > 0.8 for result in diagnostics)
    assert all(result["top_token_id"] == 7 for result in diagnostics)
    first = sample_one_token_independent(
        model, tokenizer, prompts, seeds=[17, 23]
    )
    again = sample_one_token_independent(
        model, tokenizer, prompts, seeds=[17, 23]
    )
    assert first == again
