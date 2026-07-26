"""Freeze the schema-v2 structured-action 600-cell bank and exact capture budget."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.data.relational_capture import RelationalCapturePlan  # noqa: E402
from geoprobe.data.relational_structured_action import (  # noqa: E402
    expected_structured_action_counts,
    select_relational_structured_action_specs,
    structured_action_token_lengths,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.models.artifact_identity import (  # noqa: E402
    fingerprint_local_hf_artifact,
    validate_local_hf_artifact_identity,
    validate_local_hf_support_identity,
)
from geoprobe.models.relational_structured_action import (  # noqa: E402
    TOKENIZER_ARTIFACT_SHA256,
    validate_structured_action_protocol,
    validate_tokenizer_object_binding,
)
from geoprobe.provenance import git_provenance  # noqa: E402


DEFAULT_SCENARIOS = (
    REPO_ROOT / "data/raw/deception_intent/graded_ks_150fam_l7_scenarios.jsonl"
)
DEFAULT_PROTOCOL = REPO_ROOT / "configs/relational_structured_action_protocol.json"
CAPTURE_LAYERS = (12, 16, 19, 20)
CAPTURE_MAX_BYTES = 38 * 1024**3
CAPTURE_RESERVE_BYTES = 2 * 1024**3
SENTINEL_SELECTION_SEED = 20260714
SENTINEL_PER_STATUS_CAVEAT_STRATUM = 2
SENTINEL_MEDIAN_CANDIDATE_MASS_MIN = 0.90
SENTINEL_EACH_CANDIDATE_MASS_MIN = 0.80
SENTINEL_EXPECTED_RANK_FIRST_MIN_COUNT = 7
ENGINEERING_SCENARIO_COUNT = 2


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _artifact_hashes(model_dir: Path) -> dict[str, str]:
    return {name: file_sha256(model_dir / name) for name in TOKENIZER_ARTIFACT_SHA256}


def _seeded_order_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _neutral_rows(specs: list[dict]) -> list[dict]:
    rows = [
        spec
        for spec in specs
        if spec["intervention_program"] == "NN" and spec["sample_index"] == 0
    ]
    if len(rows) != 60 or len({str(row["scenario_id"]) for row in rows}) != 60:
        raise ValueError("structured-action bank lacks one shared neutral row per scenario")
    return rows


def _select_neutral_sentinels(specs: list[dict]) -> list[dict]:
    neutral = _neutral_rows(specs)
    strata = (("PASS", True), ("PASS", False), ("FAIL", True), ("FAIL", False))
    selected: list[dict] = []
    used_families: set[str] = set()
    for true_status, caveat_present in strata:
        candidates = sorted(
            (
                row
                for row in neutral
                if row["true_status"] == true_status
                and row["caveat_present"] is caveat_present
            ),
            key=lambda row: _seeded_order_key(
                SENTINEL_SELECTION_SEED, str(row["scenario_id"])
            ),
        )
        stratum: list[dict] = []
        for row in candidates:
            family = str(row["family"])
            if family in used_families or any(
                str(chosen["family"]) == family for chosen in stratum
            ):
                continue
            stratum.append(row)
            if len(stratum) == SENTINEL_PER_STATUS_CAVEAT_STRATUM:
                break
        if len(stratum) != SENTINEL_PER_STATUS_CAVEAT_STRATUM:
            raise ValueError("neutral sentinel selection cannot preserve family diversity")
        selected.extend(stratum)
        used_families.update(str(row["family"]) for row in stratum)
    return selected


def _select_engineering_scenarios(specs: list[dict], selection_seed: int) -> list[str]:
    scenario_ids = sorted(
        {str(row["scenario_id"]) for row in _neutral_rows(specs)},
        key=lambda value: _seeded_order_key(selection_seed, value),
    )
    return scenario_ids[:ENGINEERING_SCENARIO_COUNT]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model-artifact-dir", type=Path, required=True)
    parser.add_argument("--model-artifact-identity", type=Path)
    parser.add_argument("--selection-seed", type=int, default=20260711)
    parser.add_argument("--generation-base-seed", type=int, default=731_000_000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--budget-out", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    validate_structured_action_protocol(protocol)
    if args.model_artifact_identity is None:
        model_artifact = fingerprint_local_hf_artifact(args.model_artifact_dir)
    else:
        model_artifact = json.loads(args.model_artifact_identity.read_text())
        validate_local_hf_artifact_identity(model_artifact)
        validate_local_hf_support_identity(args.model_artifact_dir, model_artifact)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_artifact_dir, local_files_only=True
    )
    tokenizer_hashes = _artifact_hashes(args.model_artifact_dir)
    validate_tokenizer_object_binding(
        protocol,
        tokenizer,
        tokenizer_artifact_sha256=tokenizer_hashes,
    )
    specs = select_relational_structured_action_specs(
        _read_jsonl(args.scenarios),
        selection_seed=args.selection_seed,
        generation_base_seed=args.generation_base_seed,
    )
    _write_jsonl(args.out, specs)

    token_lengths_by_id = structured_action_token_lengths(tokenizer, specs)
    token_lengths = [token_lengths_by_id[str(spec["conversation_id"])] for spec in specs]
    model_config = json.loads((args.model_artifact_dir / "config.json").read_text())
    capture_plan = RelationalCapturePlan(
        token_lengths=tuple(token_lengths),
        hidden_size=int(model_config["hidden_size"]),
        layers=CAPTURE_LAYERS,
        n_attention_heads=int(model_config["num_attention_heads"]),
        n_model_layers=int(model_config["num_hidden_layers"]),
        max_bytes=CAPTURE_MAX_BYTES,
        safety_reserve_bytes=CAPTURE_RESERVE_BYTES,
    ).validate()
    expected_counts = expected_structured_action_counts()
    sentinels = _select_neutral_sentinels(specs)
    engineering_scenario_ids = _select_engineering_scenarios(specs, args.selection_seed)
    engineering_conversation_ids = [
        str(spec["conversation_id"])
        for spec in specs
        if str(spec["scenario_id"]) in set(engineering_scenario_ids)
    ]
    if len(engineering_conversation_ids) != 20:
        raise ValueError("engineering checkpoint must contain two complete ten-row blocks")
    program_counts = Counter(str(spec["intervention_program"]) for spec in specs)
    family_counts = Counter(str(spec["family"]) for spec in specs)
    status_counts = Counter(str(spec["true_status"]) for spec in specs)
    # The registered structured-action plan is part of the registered protocol and run
    # artifacts; only in-repo sources are hash-bound here.
    provenance_paths = [
        Path(__file__),
        REPO_ROOT / "experiments/rollout_relational_structured_action.py",
        args.scenarios,
        args.protocol,
        REPO_ROOT / "src/geoprobe/data/relational_structured_action.py",
        REPO_ROOT / "src/geoprobe/eval/relational_structured_action_gate.py",
        REPO_ROOT / "src/geoprobe/models/relational_structured_action.py",
        REPO_ROOT / "src/geoprobe/models/relational_structured_action_rollout.py",
    ]
    if args.model_artifact_identity is not None:
        provenance_paths.append(args.model_artifact_identity)
    provenance = git_provenance(provenance_paths)
    budget_payload = {
        "schema_version": 2,
        "kind": "relational_structured_action_capture_budget",
        "argv": sys.argv,
        "provenance": provenance,
        "inputs": {
            "specs": str(args.out.resolve()),
            "specs_sha256": file_sha256(args.out),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": file_sha256(args.protocol),
            "model_artifact_sha256": model_artifact["artifact_sha256"],
            "tokenizer_artifact_sha256": tokenizer_hashes,
        },
        "token_lengths": {
            "outcome_independent": True,
            "count": len(token_lengths),
            "minimum": min(token_lengths),
            "maximum": max(token_lengths),
            "values": token_lengths,
        },
        "capture_plan": capture_plan,
        "authorization": {
            "paid_capture_allowed_by_this_artifact": False,
        },
    }
    _write_json(args.budget_out, budget_payload)
    manifest = {
        "schema_version": 2,
        "kind": "relational_structured_action_selection",
        "argv": sys.argv,
        "provenance": provenance,
        "status": "full600_candidate_frozen_not_authorized",
        "inputs": {
            "scenarios": str(args.scenarios.resolve()),
            "scenarios_sha256": file_sha256(args.scenarios),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": file_sha256(args.protocol),
            "model_artifact_support": str(args.model_artifact_dir.resolve()),
            "model_artifact_identity_file": (
                str(args.model_artifact_identity.resolve())
                if args.model_artifact_identity is not None
                else None
            ),
            "model_artifact_identity": model_artifact,
            "tokenizer_artifact_sha256": tokenizer_hashes,
        },
        "outputs": {
            "specs": str(args.out.resolve()),
            "specs_sha256": file_sha256(args.out),
            "budget": str(args.budget_out.resolve()),
            "budget_sha256": file_sha256(args.budget_out),
        },
        "generation": {
            "mode": "two_field_full_vocabulary_one_token_live_sequential",
            "dtype": "bfloat16",
            "temperature": 0.7,
            "top_p": 0.95,
            "retry_attempts": 0,
            "primary_prose_allowed": False,
        },
        "selection": {
            "selection_seed": args.selection_seed,
            "generation_base_seed": args.generation_base_seed,
            "expected_counts": expected_counts,
            "n_scenarios": len({str(spec["scenario_id"]) for spec in specs}),
            "program_counts": dict(sorted(program_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "true_status_counts": dict(sorted(status_counts.items())),
            "selection_stage": "pre_outcome",
        },
        "neutral_sentinel": {
            "selection_stage": "pre_outcome",
            "selection_seed": SENTINEL_SELECTION_SEED,
            "strata": "two_per_true_status_x_caveat_presence_with_distinct_families",
            "records": [
                {
                    "scenario_id": str(spec["scenario_id"]),
                    "conversation_id": str(spec["conversation_id"]),
                    "family": str(spec["family"]),
                    "expected_status": str(spec["true_status"]),
                    "expected_caveat": (
                        "INCLUDE" if spec["caveat_present"] else "NONE"
                    ),
                }
                for spec in sentinels
            ],
            "prompt_contract": {
                "turn_index": 0,
                "status": "exact_neutral_prefix_plus_status_prefix",
                "caveat": "same_prefix_plus_known_truthful_status_token_plus_caveat_prefix",
                "sampled_tokens_entered": False,
            },
            "thresholds": {
                "candidate_probability_mass_median_min": (
                    SENTINEL_MEDIAN_CANDIDATE_MASS_MIN
                ),
                "candidate_probability_mass_each_min": (
                    SENTINEL_EACH_CANDIDATE_MASS_MIN
                ),
                "expected_action_rank_first_min_count": (
                    SENTINEL_EXPECTED_RANK_FIRST_MIN_COUNT
                ),
            },
            "failure_policy": {
                "status": "stop_before_generation",
                "caveat": "disable_caveat_claims_and_continue_primary_status",
            },
        },
        "engineering_checkpoint": {
            "selection_stage": "pre_outcome",
            "scenario_count": ENGINEERING_SCENARIO_COUNT,
            "row_count": len(engineering_conversation_ids),
            "scenario_ids": engineering_scenario_ids,
            "conversation_ids": engineering_conversation_ids,
            "complete_scenario_blocks_required": True,
            "treatment_effect_inspection_allowed": False,
            "passing_action": "continue_same_process_to_full600",
        },
        "authorization": {
            "full600_candidate": True,
            "paid_run_allowed_by_this_artifact": False,
            "capture_allowed_by_this_artifact": False,
        },
    }
    _write_json(args.manifest_out, manifest)
    print(json.dumps({
        "specs_sha256": manifest["outputs"]["specs_sha256"],
        "budget_sha256": manifest["outputs"]["budget_sha256"],
        "expected_counts": expected_counts,
        "capture_projected_bytes": capture_plan["bytes"]["projected_plus_reserve"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
