"""Run the frozen schema-v2 structured-action 600 bank in one resumable job."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.io import atomic_text  # noqa: E402
from geoprobe.data.jsonl import read_jsonl  # noqa: E402
from geoprobe.data.relational_structured_action import (  # noqa: E402
    expected_structured_action_counts,
    structured_action_token_lengths,
    validate_relational_structured_action_specs,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.models.artifact_identity import (  # noqa: E402
    fingerprint_local_hf_artifact,
    validate_local_hf_artifact_identity,
)
from geoprobe.models.loader import load_hf_model  # noqa: E402
from geoprobe.models.relational_structured_action import (  # noqa: E402
    FIELD_SPECS,
    TOKENIZER_ARTIFACT_SHA256,
    canonical_json_sha256,
    validate_structured_action_protocol,
    validate_tokenizer_object_binding,
)
from geoprobe.models.relational_structured_action_rollout import (  # noqa: E402
    build_structured_action_sentinel_prompts,
    next_token_distribution,
    rollout_relational_structured_action_scenario,
    validate_relational_structured_action_row,
    validate_relational_structured_action_scenario_rows,
)

MANIFEST_KIND = "relational_structured_action_rollout_manifest"
CONTRACT_KIND = "relational_structured_action_rollout_contract"
SELECTION_KIND = "relational_structured_action_selection"
BUDGET_KIND = "relational_structured_action_capture_budget"
GENERATION_MODE = "two_field_full_vocabulary_one_token_live_sequential"
# The registered structured-action plan that was hash-bound alongside these
# sources is part of the privately retained run ledgers of the program; the
# public tree freezes only the in-repo sources.
REQUIRED_FROZEN_SOURCE_PATHS = (
    "src/geoprobe/data/relational_structured_action.py",
    "src/geoprobe/eval/relational_structured_action_gate.py",
    "src/geoprobe/models/relational_structured_action.py",
    "src/geoprobe/models/relational_structured_action_rollout.py",
    "experiments/report_relational_structured_action_gate.py",
    "experiments/rollout_relational_structured_action.py",
)

ModelLoader = Callable[..., tuple[Any, Any, dict[str, Any]]]
DistributionFunction = Callable[..., list[dict[str, Any]]]
ScenarioRolloutFunction = Callable[..., list[dict[str, Any]]]


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _tokenizer_artifact_hashes(model_dir: Path) -> dict[str, str]:
    return {name: file_sha256(model_dir / name) for name in TOKENIZER_ARTIFACT_SHA256}


def _source_hashes() -> dict[str, str]:
    return {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in REQUIRED_FROZEN_SOURCE_PATHS
    }


def _recorded_hash_by_suffix(provenance: Mapping[str, Any], relative: str) -> str:
    hashes = provenance.get("file_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("frozen provenance lacks file hashes")
    normalized = relative.replace("\\", "/")
    matches = [
        str(value)
        for path, value in hashes.items()
        if str(path).replace("\\", "/").endswith(normalized)
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen provenance does not bind exactly one {relative}")
    return matches[0]


def _validate_frozen_source_hashes(
    selection: Mapping[str, Any], budget: Mapping[str, Any]
) -> dict[str, str]:
    current = _source_hashes()
    for relative, digest in current.items():
        if _recorded_hash_by_suffix(selection.get("provenance", {}), relative) != digest:
            raise ValueError(f"selection provenance is stale for {relative}")
        if _recorded_hash_by_suffix(budget.get("provenance", {}), relative) != digest:
            raise ValueError(f"budget provenance is stale for {relative}")
    return current


def _validate_neutral_sentinel(
    sentinel: Any, specs_by_id: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(sentinel, Mapping):
        raise ValueError("selection lacks the frozen neutral sentinel")
    records = sentinel.get("records")
    if (
        sentinel.get("selection_stage") != "pre_outcome"
        or not isinstance(sentinel.get("selection_seed"), int)
        or sentinel.get("strata")
        != "two_per_true_status_x_caveat_presence_with_distinct_families"
        or not isinstance(records, list)
        or len(records) != 8
    ):
        raise ValueError("neutral sentinel selection is not frozen")
    expected_strata: Counter[tuple[str, bool]] = Counter()
    families: set[str] = set()
    conversation_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "scenario_id", "conversation_id", "family", "expected_status", "expected_caveat",
        }:
            raise ValueError("neutral sentinel record schema differs")
        conversation_id = str(record["conversation_id"])
        if conversation_id in conversation_ids or conversation_id not in specs_by_id:
            raise ValueError("neutral sentinel conversation IDs are invalid")
        conversation_ids.add(conversation_id)
        spec = specs_by_id[conversation_id]
        if spec["intervention_program"] != "NN" or spec["sample_index"] != 0:
            raise ValueError("neutral sentinel is not an NN sample-0 record")
        expected_caveat = "INCLUDE" if spec["caveat_present"] else "NONE"
        if record != {
            "scenario_id": str(spec["scenario_id"]),
            "conversation_id": conversation_id,
            "family": str(spec["family"]),
            "expected_status": str(spec["true_status"]),
            "expected_caveat": expected_caveat,
        }:
            raise ValueError("neutral sentinel metadata differs from its frozen spec")
        expected_strata[(str(spec["true_status"]), bool(spec["caveat_present"]))] += 1
        families.add(str(spec["family"]))
    if expected_strata != Counter({
        ("PASS", True): 2,
        ("PASS", False): 2,
        ("FAIL", True): 2,
        ("FAIL", False): 2,
    }) or len(families) != 8:
        raise ValueError("neutral sentinel balance or family diversity differs")
    if sentinel.get("prompt_contract") != {
        "turn_index": 0,
        "status": "exact_neutral_prefix_plus_status_prefix",
        "caveat": "same_prefix_plus_known_truthful_status_token_plus_caveat_prefix",
        "sampled_tokens_entered": False,
    }:
        raise ValueError("neutral sentinel prompt contract differs")
    if sentinel.get("thresholds") != {
        "candidate_probability_mass_median_min": 0.9,
        "candidate_probability_mass_each_min": 0.8,
        "expected_action_rank_first_min_count": 7,
    }:
        raise ValueError("neutral sentinel thresholds differ")
    if sentinel.get("failure_policy") != {
        "status": "stop_before_generation",
        "caveat": "disable_caveat_claims_and_continue_primary_status",
    }:
        raise ValueError("neutral sentinel failure policy differs")
    return dict(sentinel)


def _validate_engineering_checkpoint(
    checkpoint: Any,
    specs_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selection lacks the frozen engineering checkpoint")
    scenario_ids = checkpoint.get("scenario_ids")
    conversation_ids = checkpoint.get("conversation_ids")
    if (
        checkpoint.get("selection_stage") != "pre_outcome"
        or checkpoint.get("scenario_count") != 2
        or checkpoint.get("row_count") != 20
        or not isinstance(scenario_ids, list)
        or len(scenario_ids) != 2
        or len(set(map(str, scenario_ids))) != 2
        or not isinstance(conversation_ids, list)
        or len(conversation_ids) != 20
        or len(set(map(str, conversation_ids))) != 20
        or checkpoint.get("complete_scenario_blocks_required") is not True
        or checkpoint.get("treatment_effect_inspection_allowed") is not False
        or checkpoint.get("passing_action") != "continue_same_process_to_full600"
    ):
        raise ValueError("engineering checkpoint is not the frozen two-scenario block")
    expected = [
        conversation_id
        for conversation_id, spec in specs_by_id.items()
        if str(spec["scenario_id"]) in set(map(str, scenario_ids))
    ]
    if expected != list(map(str, conversation_ids)):
        raise ValueError("engineering checkpoint IDs differ from complete frozen blocks")
    return dict(checkpoint)


def _validate_selection_and_budget(
    *,
    specs_path: Path,
    selection_path: Path,
    budget_path: Path,
    protocol_path: Path,
    model_dir: Path,
    model_artifact_identity_path: Path | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    specs = read_jsonl(specs_path)
    validate_relational_structured_action_specs(specs, require_full_bank=True)
    protocol = json.loads(protocol_path.read_text())
    validate_structured_action_protocol(protocol)
    selection = json.loads(selection_path.read_text())
    budget = json.loads(budget_path.read_text())
    if (
        selection.get("schema_version") != 2
        or selection.get("kind") != SELECTION_KIND
        or selection.get("status") != "full600_candidate_frozen_not_authorized"
    ):
        raise ValueError("invalid structured-action selection manifest")
    if budget.get("schema_version") != 2 or budget.get("kind") != BUDGET_KIND:
        raise ValueError("invalid structured-action capture budget")
    specs_sha256 = file_sha256(specs_path)
    protocol_sha256 = file_sha256(protocol_path)
    budget_sha256 = file_sha256(budget_path)
    if selection.get("outputs") != {
        "specs": selection["outputs"].get("specs"),
        "specs_sha256": specs_sha256,
        "budget": selection["outputs"].get("budget"),
        "budget_sha256": budget_sha256,
    }:
        raise ValueError("selection outputs differ from the frozen specs or budget")
    inputs = selection.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("protocol_sha256") != protocol_sha256:
        raise ValueError("selection protocol hash differs from the frozen protocol")
    generation = selection.get("generation")
    if generation != {
        "mode": GENERATION_MODE,
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 0.95,
        "retry_attempts": 0,
        "primary_prose_allowed": False,
    }:
        raise ValueError("selection generation contract differs")
    selection_body = selection.get("selection")
    if (
        not isinstance(selection_body, Mapping)
        or selection_body.get("expected_counts") != expected_structured_action_counts()
        or selection_body.get("n_scenarios") != 60
        or selection_body.get("selection_stage") != "pre_outcome"
    ):
        raise ValueError("selection does not bind the complete pre-outcome 600 bank")
    if selection.get("authorization") != {
        "full600_candidate": True,
        "paid_run_allowed_by_this_artifact": False,
        "capture_allowed_by_this_artifact": False,
    }:
        raise ValueError("selection authorization boundary differs")
    specs_by_id = {str(spec["conversation_id"]): spec for spec in specs}
    sentinel = _validate_neutral_sentinel(selection.get("neutral_sentinel"), specs_by_id)
    checkpoint = _validate_engineering_checkpoint(
        selection.get("engineering_checkpoint"), specs_by_id
    )

    model_artifact = inputs.get("model_artifact_identity")
    if not isinstance(model_artifact, dict):
        raise ValueError("selection lacks the exact model artifact identity")
    validate_local_hf_artifact_identity(model_artifact)
    if model_artifact_identity_path is not None:
        supplied = json.loads(model_artifact_identity_path.read_text())
        validate_local_hf_artifact_identity(supplied)
        if supplied != model_artifact:
            raise ValueError("supplied model identity differs from the frozen selection")
    observed_model_artifact = fingerprint_local_hf_artifact(model_dir)
    if observed_model_artifact != model_artifact:
        raise ValueError("local model files differ from the frozen artifact hashes")
    tokenizer_hashes = _tokenizer_artifact_hashes(model_dir)
    if (
        tokenizer_hashes != TOKENIZER_ARTIFACT_SHA256
        or inputs.get("tokenizer_artifact_sha256") != tokenizer_hashes
    ):
        raise ValueError("local tokenizer hashes differ from the frozen protocol/selection")

    budget_inputs = budget.get("inputs")
    if not isinstance(budget_inputs, Mapping) or any((
        budget_inputs.get("specs_sha256") != specs_sha256,
        budget_inputs.get("protocol_sha256") != protocol_sha256,
        budget_inputs.get("model_artifact_sha256") != model_artifact["artifact_sha256"],
        budget_inputs.get("tokenizer_artifact_sha256") != tokenizer_hashes,
    )):
        raise ValueError("capture budget inputs differ from frozen artifacts")
    token_lengths = budget.get("token_lengths")
    capture_plan = budget.get("capture_plan")
    if (
        not isinstance(token_lengths, Mapping)
        or token_lengths.get("outcome_independent") is not True
        or token_lengths.get("count") != 600
        or not isinstance(token_lengths.get("values"), list)
        or len(token_lengths["values"]) != 600
        or any(not isinstance(value, int) or value < 1 for value in token_lengths["values"])
        or token_lengths.get("minimum") != min(token_lengths["values"])
        or token_lengths.get("maximum") != max(token_lengths["values"])
        or not isinstance(capture_plan, Mapping)
        or not isinstance(capture_plan.get("plan"), Mapping)
        or capture_plan["plan"].get("token_lengths") != token_lengths["values"]
        or capture_plan["plan"].get("residual_dtype") != "bfloat16"
        or capture_plan["plan"].get("attention_dtype") != "bfloat16"
        or not isinstance(capture_plan.get("bytes"), Mapping)
        or capture_plan["bytes"].get("projected_plus_reserve", 2**63)
        > capture_plan["plan"].get("max_bytes", 0)
    ):
        raise ValueError("capture budget lacks a feasible exact 600-row BF16 plan")
    if budget.get("authorization") != {"paid_capture_allowed_by_this_artifact": False}:
        raise ValueError("capture budget authorization boundary differs")
    source_hashes = _validate_frozen_source_hashes(selection, budget)
    return specs, selection, budget, protocol, {
        "neutral_sentinel": sentinel,
        "engineering_checkpoint": checkpoint,
        "model_artifact": model_artifact,
    }, source_hashes


def _scenario_order(
    specs: Sequence[dict[str, Any]], engineering_scenario_ids: Sequence[str]
) -> list[str]:
    frozen_order: list[str] = []
    for spec in specs:
        scenario_id = str(spec["scenario_id"])
        if scenario_id not in frozen_order:
            frozen_order.append(scenario_id)
    engineering = list(map(str, engineering_scenario_ids))
    return [*engineering, *(value for value in frozen_order if value not in set(engineering))]


def _validate_resume_rows(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[dict[str, Any]],
    protocol: Mapping[str, Any],
    planned_scenarios: Sequence[str],
) -> list[str]:
    if len(rows) % 10:
        raise ValueError("resume rows contain a partial ten-row scenario block")
    grouped_specs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        grouped_specs[str(spec["scenario_id"])].append(spec)
    completed: list[str] = []
    for offset in range(0, len(rows), 10):
        block = list(rows[offset:offset + 10])
        scenario_ids = {str(row.get("scenario_id")) for row in block}
        if len(scenario_ids) != 1:
            raise ValueError("resume rows mix scenarios inside a ten-row block")
        scenario_id = scenario_ids.pop()
        expected_position = len(completed)
        if expected_position >= len(planned_scenarios) or scenario_id != planned_scenarios[expected_position]:
            raise ValueError("resume scenario blocks are not a prefix of the frozen run order")
        expected_specs = grouped_specs.get(scenario_id)
        if expected_specs is None:
            raise ValueError("resume contains a foreign scenario")
        expected_ids = [str(spec["conversation_id"]) for spec in expected_specs]
        if [str(row.get("conversation_id")) for row in block] != expected_ids:
            raise ValueError("resume scenario rows differ from the frozen spec order")
        for row, spec in zip(block, expected_specs, strict=True):
            validate_relational_structured_action_row(row, protocol)
            for key, value in spec.items():
                if key != "kind" and row.get(key) != value:
                    raise ValueError(
                        f"resume row {row.get('conversation_id')} differs from spec field {key}"
                    )
            if row.get("source_spec_sha256") != spec["spec_sha256"]:
                raise ValueError("resume row source-spec hash differs")
        validate_relational_structured_action_scenario_rows(block, protocol)
        completed.append(scenario_id)
    return completed


def _evaluate_sentinel_channel(
    diagnostics: Sequence[dict[str, Any]],
    expected_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if len(diagnostics) != len(expected_token_ids):
        raise ValueError("sentinel diagnostic count differs from its frozen records")
    masses: list[float] = []
    rank_first: list[bool] = []
    records: list[dict[str, Any]] = []
    candidate_keys = {str(int(token)) for token in candidate_token_ids}
    for diagnostic, expected_token_id in zip(diagnostics, expected_token_ids, strict=True):
        probabilities = diagnostic.get("candidate_probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != candidate_keys:
            raise ValueError("sentinel diagnostic candidate set differs")
        mass = float(diagnostic.get("candidate_probability_mass", -1.0))
        expected_probability = float(probabilities[str(expected_token_id)])
        first = expected_probability == max(float(value) for value in probabilities.values())
        masses.append(mass)
        rank_first.append(first)
        records.append({
            "candidate_probability_mass": mass,
            "expected_token_id": int(expected_token_id),
            "expected_candidate_probability": expected_probability,
            "expected_action_rank_first_among_candidates": first,
            "candidate_probabilities": dict(probabilities),
        })
    median_mass = statistics.median(masses)
    each_minimum = float(thresholds["candidate_probability_mass_each_min"])
    median_minimum = float(thresholds["candidate_probability_mass_median_min"])
    rank_first_minimum = int(thresholds["expected_action_rank_first_min_count"])
    if not 1 <= rank_first_minimum <= len(rank_first):
        raise ValueError("sentinel expected-action rank threshold is invalid")
    passed = (
        median_mass >= median_minimum
        and min(masses) >= each_minimum
        and sum(rank_first) >= rank_first_minimum
    )
    return {
        "passed": passed,
        "record_count": len(records),
        "candidate_probability_mass_median": median_mass,
        "candidate_probability_mass_minimum": min(masses),
        "expected_action_rank_first_count": sum(rank_first),
        "expected_action_rank_first_required": rank_first_minimum,
        "records": records,
    }


def run_neutral_sentinel(
    model: Any,
    tokenizer: Any,
    specs_by_id: Mapping[str, dict[str, Any]],
    sentinel: Mapping[str, Any],
    *,
    distribution_fn: DistributionFunction = next_token_distribution,
) -> dict[str, Any]:
    prompt_records = [
        build_structured_action_sentinel_prompts(
            tokenizer, specs_by_id[str(record["conversation_id"])]
        )
        for record in sentinel["records"]
    ]
    for declared, built in zip(sentinel["records"], prompt_records, strict=True):
        if (
            built["scenario_id"] != declared["scenario_id"]
            or built["conversation_id"] != declared["conversation_id"]
            or built["status"]["expected_action"] != declared["expected_status"]
            or built["caveat"]["expected_action"] != declared["expected_caveat"]
        ):
            raise ValueError("sentinel prompt differs from its preselected expected action")
    status_candidates = [
        candidate.raw_token_id for candidate in FIELD_SPECS["status"].candidates
    ]
    caveat_candidates = [
        candidate.raw_token_id for candidate in FIELD_SPECS["caveat"].candidates
    ]
    status_diagnostics = distribution_fn(
        model,
        tokenizer,
        [record["status"]["prompt_token_ids"] for record in prompt_records],
        candidate_token_ids=status_candidates,
        temperature=0.7,
        top_p=0.95,
    )
    caveat_diagnostics = distribution_fn(
        model,
        tokenizer,
        [record["caveat"]["prompt_token_ids"] for record in prompt_records],
        candidate_token_ids=caveat_candidates,
        temperature=0.7,
        top_p=0.95,
    )
    thresholds = sentinel["thresholds"]
    status = _evaluate_sentinel_channel(
        status_diagnostics,
        [record["status"]["expected_token_id"] for record in prompt_records],
        status_candidates,
        thresholds,
    )
    caveat = _evaluate_sentinel_channel(
        caveat_diagnostics,
        [record["caveat"]["expected_token_id"] for record in prompt_records],
        caveat_candidates,
        thresholds,
    )
    return {
        "kind": "relational_structured_action_neutral_sentinel_report",
        "selection_stage": "pre_outcome",
        "treatment_effects_inspected": False,
        "thresholds": dict(thresholds),
        "status": status,
        "caveat": caveat,
        "primary_status_generation_allowed": bool(status["passed"]),
        "caveat_claims_enabled": bool(caveat["passed"]),
        "prompt_hashes": [
            {
                "conversation_id": record["conversation_id"],
                "status": record["status"]["prompt_token_ids_sha256"],
                "caveat": record["caveat"]["prompt_token_ids_sha256"],
            }
            for record in prompt_records
        ],
    }


def _engineering_report(
    rows: Sequence[dict[str, Any]],
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    budget_lengths: Mapping[str, int],
) -> dict[str, Any]:
    by_id = {str(row["conversation_id"]): row for row in rows}
    declared_ids = list(map(str, checkpoint["conversation_ids"]))
    if any(conversation_id not in by_id for conversation_id in declared_ids):
        return {
            "status": "pending",
            "completed_rows": sum(conversation_id in by_id for conversation_id in declared_ids),
            "required_rows": 20,
            "treatment_effects_inspected": False,
        }
    selected = [by_id[conversation_id] for conversation_id in declared_ids]
    if [str(row["conversation_id"]) for row in selected] != declared_ids:
        raise ValueError("engineering checkpoint row order differs")
    for scenario_id in checkpoint["scenario_ids"]:
        block = [row for row in selected if str(row["scenario_id"]) == str(scenario_id)]
        validate_relational_structured_action_scenario_rows(block, protocol)
    fields = [
        field
        for row in selected
        for turn in row["assistant_action_turn_records"]
        for field in turn["fields"]
    ]
    if len(fields) != 160 or len({field["field_event_id"] for field in fields}) != 112:
        raise ValueError("engineering checkpoint field/event counts differ")
    if any(len(row["token_ids"]) != budget_lengths[str(row["conversation_id"])] for row in selected):
        raise ValueError("engineering checkpoint serialization differs from the frozen budget")
    status_fields = [field for field in fields if field["field_name"] == "status"]
    caveat_fields = [field for field in fields if field["field_name"] == "caveat"]
    return {
        "status": "passed",
        "row_count": 20,
        "scenario_count": 2,
        "logical_field_record_count": 160,
        "unique_field_event_count": 112,
        "status_action_recognition_rate": sum(
            field["mapped_action"] != "NO_ACTION" for field in status_fields
        ) / len(status_fields),
        "caveat_action_recognition_rate": sum(
            field["mapped_action"] != "NO_ACTION" for field in caveat_fields
        ) / len(caveat_fields),
        "exact_prefix_anchor_clone_crn_validation": "passed",
        "serialization_and_budget_validation": "passed",
        "treatment_effects_inspected": False,
        "passing_action": "continue_same_process_to_full600",
    }


def _manifest_payload(
    *,
    state: str,
    contract: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    specs_path: Path,
    selection_path: Path,
    budget_path: Path,
    protocol_path: Path,
    model_meta: Mapping[str, Any] | None,
    sentinel_report: Mapping[str, Any] | None,
    engineering_report: Mapping[str, Any],
    started: float,
    failure: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": MANIFEST_KIND,
        "argv": sys.argv,
        "status": state,
        "rollout_contract": dict(contract),
        "inputs": {
            "specs": str(specs_path.resolve()),
            "specs_sha256": file_sha256(specs_path),
            "selection": str(selection_path.resolve()),
            "selection_sha256": file_sha256(selection_path),
            "budget": str(budget_path.resolve()),
            "budget_sha256": file_sha256(budget_path),
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": file_sha256(protocol_path),
        },
        "model": dict(model_meta) if model_meta is not None else None,
        "neutral_sentinel": dict(sentinel_report) if sentinel_report is not None else None,
        "caveat_claims_enabled": (
            bool(sentinel_report["caveat_claims_enabled"])
            if sentinel_report is not None
            else None
        ),
        "engineering_checkpoint": dict(engineering_report),
        "completed_scenarios": len({str(row["scenario_id"]) for row in rows}),
        "completed_rows": len(rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    if rows:
        payload["working_output"] = {
            "rows_sha256": canonical_json_sha256(
                [str(row["row_sha256"]) for row in rows]
            ),
        }
    if failure is not None:
        payload["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure)[:4096],
        }
    return payload


def run_rollout(
    args: argparse.Namespace,
    *,
    model_loader: ModelLoader = load_hf_model,
    distribution_fn: DistributionFunction = next_token_distribution,
    scenario_rollout_fn: ScenarioRolloutFunction = (
        rollout_relational_structured_action_scenario
    ),
) -> int:
    if args.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    if args.dtype != "bfloat16":
        raise ValueError("structured-action rollout requires BF16")
    started = time.monotonic()
    status_path = args.status_out or args.manifest_out.parent / "STATUS"
    specs, selection, budget, protocol, frozen, source_hashes = (
        _validate_selection_and_budget(
            specs_path=args.specs,
            selection_path=args.selection_manifest,
            budget_path=args.budget,
            protocol_path=args.protocol,
            model_dir=args.model,
            model_artifact_identity_path=args.model_artifact_identity,
        )
    )
    specs_by_id = {str(spec["conversation_id"]): spec for spec in specs}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        grouped[str(spec["scenario_id"])].append(spec)
    checkpoint = frozen["engineering_checkpoint"]
    planned_scenarios = _scenario_order(specs, checkpoint["scenario_ids"])
    contract_body = {
        "specs_sha256": file_sha256(args.specs),
        "selection_sha256": file_sha256(args.selection_manifest),
        "budget_sha256": file_sha256(args.budget),
        "protocol_sha256": file_sha256(args.protocol),
        "model_artifact_sha256": frozen["model_artifact"]["artifact_sha256"],
        "tokenizer_artifact_sha256": dict(TOKENIZER_ARTIFACT_SHA256),
        "source_sha256": source_hashes,
        "scenario_order": planned_scenarios,
        "generation": dict(selection["generation"]),
        "neutral_sentinel": frozen["neutral_sentinel"],
        "engineering_checkpoint": checkpoint,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    contract = {
        "schema_version": 2,
        "kind": CONTRACT_KIND,
        **contract_body,
        "contract_sha256": canonical_json_sha256(contract_body),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    previous_manifest: dict[str, Any] | None = None
    if args.resume:
        if args.out.exists() and not args.manifest_out.exists():
            raise ValueError("resume rows exist without the rollout manifest")
        if args.out.exists():
            rows = read_jsonl(args.out)
        if args.manifest_out.exists():
            previous_manifest = json.loads(args.manifest_out.read_text())
            if (
                previous_manifest.get("schema_version") != 2
                or previous_manifest.get("kind") != MANIFEST_KIND
                or previous_manifest.get("rollout_contract") != contract
            ):
                raise ValueError("resume manifest differs from the exact frozen contract")
    elif args.out.exists() or args.manifest_out.exists() or status_path.exists():
        raise FileExistsError("rollout outputs exist; use --resume with the exact contract")
    completed = _validate_resume_rows(rows, specs, protocol, planned_scenarios)
    budget_lengths = {
        str(spec["conversation_id"]): int(length)
        for spec, length in zip(specs, budget["token_lengths"]["values"], strict=True)
    }
    engineering_report = _engineering_report(
        rows, checkpoint, protocol, budget_lengths
    )
    if previous_manifest is not None and previous_manifest.get("status") == "success":
        output = previous_manifest.get("output")
        if (
            len(rows) != 600
            or completed != planned_scenarios
            or not isinstance(output, Mapping)
            or output.get("rows_sha256") != file_sha256(args.out)
        ):
            raise ValueError("successful resume manifest is incomplete or tampered")
        atomic_text(status_path, "success\n")
        print(json.dumps({"status": "success", "already_complete": True}, indent=2))
        return 0
    if len(rows) == 600:
        sentinel_report = (
            previous_manifest.get("neutral_sentinel") if previous_manifest else None
        )
        if (
            not isinstance(sentinel_report, Mapping)
            or sentinel_report.get("status", {}).get("passed") is not True
            or engineering_report.get("status") != "passed"
        ):
            raise ValueError("complete resume rows lack passed frozen admission records")
        final = _manifest_payload(
            state="success",
            contract=contract,
            rows=rows,
            specs_path=args.specs,
            selection_path=args.selection_manifest,
            budget_path=args.budget,
            protocol_path=args.protocol,
            model_meta=previous_manifest.get("model") if previous_manifest else None,
            sentinel_report=sentinel_report,
            engineering_report=engineering_report,
            started=started,
        )
        final["output"] = {
            "rows": str(args.out.resolve()),
            "rows_sha256": file_sha256(args.out),
        }
        atomic_json(args.manifest_out, final)
        atomic_text(status_path, "success\n")
        return 0

    sentinel_report: dict[str, Any] | None = None
    model_meta: dict[str, Any] | None = None

    def write_state(state: str, failure: BaseException | None = None) -> None:
        atomic_json(
            args.manifest_out,
            _manifest_payload(
                state=state,
                contract=contract,
                rows=rows,
                specs_path=args.specs,
                selection_path=args.selection_manifest,
                budget_path=args.budget,
                protocol_path=args.protocol,
                model_meta=model_meta,
                sentinel_report=sentinel_report,
                engineering_report=engineering_report,
                started=started,
                failure=failure,
            ),
        )
        atomic_text(status_path, state + "\n")

    write_state("in_progress")
    try:
        model, tokenizer, loaded_meta = model_loader(
            str(args.model), device=args.device, dtype=torch.bfloat16
        )
        model_meta = dict(loaded_meta)
        parameter = next(model.parameters())
        if parameter.dtype is not torch.bfloat16:
            raise ValueError(f"loaded model dtype {parameter.dtype} != torch.bfloat16")
        validate_tokenizer_object_binding(
            protocol,
            tokenizer,
            tokenizer_artifact_sha256=_tokenizer_artifact_hashes(args.model),
        )
        observed_lengths = structured_action_token_lengths(tokenizer, specs)
        if [observed_lengths[str(spec["conversation_id"])] for spec in specs] != budget[
            "token_lengths"
        ]["values"]:
            raise ValueError("loaded tokenizer transcript lengths differ from frozen budget")
        if time.monotonic() - started >= args.max_wall_seconds:
            raise TimeoutError("rollout wall-time cap reached after model validation")

        sentinel_report = run_neutral_sentinel(
            model,
            tokenizer,
            specs_by_id,
            frozen["neutral_sentinel"],
            distribution_fn=distribution_fn,
        )
        write_state("in_progress")
        if sentinel_report["status"]["passed"] is not True:
            raise ValueError("neutral status sentinel failed before generation")

        for scenario_id in planned_scenarios[len(completed):]:
            if time.monotonic() - started >= args.max_wall_seconds:
                raise TimeoutError("rollout wall-time cap reached between scenarios")
            scenario_rows = scenario_rollout_fn(
                model,
                tokenizer,
                grouped[scenario_id],
                protocol=protocol,
            )
            expected_ids = [
                str(spec["conversation_id"]) for spec in grouped[scenario_id]
            ]
            if [str(row.get("conversation_id")) for row in scenario_rows] != expected_ids:
                raise ValueError("scenario rollout differs from the frozen row order")
            validate_relational_structured_action_scenario_rows(
                scenario_rows, protocol
            )
            rows.extend(scenario_rows)
            completed.append(scenario_id)
            atomic_jsonl(args.out, rows)
            engineering_report = _engineering_report(
                rows, checkpoint, protocol, budget_lengths
            )
            write_state("in_progress")
            if (
                len(completed) == int(checkpoint["scenario_count"])
                and engineering_report.get("status") != "passed"
            ):
                raise ValueError("engineering checkpoint failed before full600 continuation")

        if len(rows) != 600 or completed != planned_scenarios:
            raise ValueError("structured-action rollout ended before the complete 600 bank")
        _validate_resume_rows(rows, specs, protocol, planned_scenarios)
        final = _manifest_payload(
            state="success",
            contract=contract,
            rows=rows,
            specs_path=args.specs,
            selection_path=args.selection_manifest,
            budget_path=args.budget,
            protocol_path=args.protocol,
            model_meta=model_meta,
            sentinel_report=sentinel_report,
            engineering_report=engineering_report,
            started=started,
        )
        final["output"] = {
            "rows": str(args.out.resolve()),
            "rows_sha256": file_sha256(args.out),
        }
        atomic_json(args.manifest_out, final)
        atomic_text(status_path, "success\n")
        print(json.dumps({
            "status": "success",
            "scenarios": 60,
            "rows": 600,
            "caveat_claims_enabled": sentinel_report["caveat_claims_enabled"],
            "elapsed_seconds": final["elapsed_seconds"],
        }, indent=2))
        return 0
    except BaseException as error:
        write_state("failed", error)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-artifact-identity", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--status-out", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    return run_rollout(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
