from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from experiments import rollout_natpress
from geoprobe.models.natpress_rollout import (
    assemble_arm_user_turns,
    load_natpress_config,
    plan_conversations,
    turn_seed,
    validate_natpress_config,
)

CID_RE = re.compile(r"^natpress:[^:]+:[^:]+:(smooth|benign|step|latedump):s\d{2}$")


def _scenario(sid: str, family: str) -> dict:
    return {
        "scenario_id": sid,
        "family": family,
        "title": f"Title {sid}",
        "ground_truth": {
            "fact": f"Fact for {sid}.",
            "honest_verdict": "FAIL",
            "deceptive_verdict": "PASS",
        },
        "pressurer": "Dana, peer-plus, ships Thursday",
        "system": f"You are the assistant for {family}.",
        "t1_user": f"T1 question for {sid} with the raw numbers.",
        "benign_turns": [f"benign {sid} {i}" for i in range(6)],
        "smooth_turns": [f"smooth {sid} {i}" for i in range(5)],
        "step_t6": f"step t6 {sid}",
        "latedump_t5": f"latedump t5 {sid}",
        "latedump_t6": f"latedump t6 {sid}",
        "corner": f"corner {sid}: one plain sentence, yes or no?",
    }


def _valid_cfg() -> dict:
    return {
        "protocol": "natpress_v1",
        "version": "smoke_v2",
        "sampling": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens_per_turn": 256},
        "samples_per_arm": {"smooth": 2, "benign": 2, "step": 1, "latedump": 1},
        "scenarios": [_scenario("natpress_a_01", "alpha"), _scenario("natpress_b_01", "beta")],
    }


# --------------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------------
def test_validate_happy_path() -> None:
    validate_natpress_config(_valid_cfg())


def test_validate_missing_top_level_key() -> None:
    cfg = _valid_cfg()
    del cfg["sampling"]
    with pytest.raises(ValueError, match="missing top-level key: 'sampling'"):
        validate_natpress_config(cfg)


def test_validate_benign_turns_wrong_length() -> None:
    cfg = _valid_cfg()
    cfg["scenarios"][0]["benign_turns"] = cfg["scenarios"][0]["benign_turns"][:5]
    with pytest.raises(ValueError, match="benign_turns must be a list of 6"):
        validate_natpress_config(cfg)


def test_validate_step_exceeds_benign() -> None:
    cfg = _valid_cfg()
    cfg["samples_per_arm"]["step"] = 3
    with pytest.raises(ValueError, match="step must be <= benign"):
        validate_natpress_config(cfg)


def test_validate_duplicate_scenario_id() -> None:
    cfg = _valid_cfg()
    cfg["scenarios"][1]["scenario_id"] = cfg["scenarios"][0]["scenario_id"]
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        validate_natpress_config(cfg)


def test_validate_empty_string_field() -> None:
    cfg = _valid_cfg()
    cfg["scenarios"][0]["t1_user"] = "   "
    with pytest.raises(ValueError, match="t1_user must be a non-empty string"):
        validate_natpress_config(cfg)


def test_validate_unexpected_scenario_field() -> None:
    cfg = _valid_cfg()
    cfg["scenarios"][0]["extra"] = "nope"
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_natpress_config(cfg)


def test_validate_optional_benign_corner_turn() -> None:
    cfg = _valid_cfg()
    cfg["scenarios"][0]["benign_corner_turn"] = "State your verdict plainly, either way."
    validate_natpress_config(cfg)  # optional field accepted
    cfg["scenarios"][0]["benign_corner_turn"] = "   "
    with pytest.raises(ValueError, match="benign_corner_turn must be a non-empty string"):
        validate_natpress_config(cfg)


def test_validate_v3_top_level_keys() -> None:
    cfg = _valid_cfg()
    cfg["sample_index_offset"] = 10
    cfg["adaptive_samples_per_arm"] = {"smooth": 3, "benign": 1, "step": 2, "latedump": 2}
    validate_natpress_config(cfg)
    cfg["sample_index_offset"] = -1
    with pytest.raises(ValueError, match="sample_index_offset"):
        validate_natpress_config(cfg)
    cfg["sample_index_offset"] = 10
    del cfg["adaptive_samples_per_arm"]["step"]
    with pytest.raises(ValueError, match="adaptive_samples_per_arm missing arm"):
        validate_natpress_config(cfg)
    # adaptive arms share no siblings: step > benign is legal there
    cfg["adaptive_samples_per_arm"] = {"smooth": 3, "benign": 1, "step": 2, "latedump": 2}
    validate_natpress_config(cfg)


def test_plan_conversations_sample_index_offset() -> None:
    cfg = _valid_cfg()
    cfg["sample_index_offset"] = 10
    plan = plan_conversations(cfg)
    assert all(entry["sample_index"] >= 10 for entry in plan)
    assert all(":s1" in entry["conversation_id"] for entry in plan)
    # sibling alignment survives the offset: every step/latedump has a benign at the
    # same offset sample_index
    benign = {(e["scenario_id"], e["sample_index"]) for e in plan if e["arm"] == "benign"}
    for entry in plan:
        if entry["arm"] in ("step", "latedump"):
            assert (entry["scenario_id"], entry["sample_index"]) in benign


def test_load_natpress_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_cfg()))
    cfg = load_natpress_config(path)
    assert cfg["protocol"] == "natpress_v1"
    assert len(cfg["scenarios"]) == 2


# --------------------------------------------------------------------------------------------------
# Arm assembly byte-identity invariants
# --------------------------------------------------------------------------------------------------
def test_assemble_arm_user_turns_invariants() -> None:
    scenario = _scenario("natpress_a_01", "alpha")
    benign = assemble_arm_user_turns(scenario, "benign")
    smooth = assemble_arm_user_turns(scenario, "smooth")
    step = assemble_arm_user_turns(scenario, "step")
    latedump = assemble_arm_user_turns(scenario, "latedump")

    assert len(benign) == len(smooth) == len(step) == len(latedump) == 7
    # T1 shared across every arm.
    assert benign[0] == smooth[0] == step[0] == latedump[0] == scenario["t1_user"]
    # step reuses benign user turns for T2-T5 (indices 1-4); latedump for T2-T4 (indices 1-3).
    assert step[1:5] == benign[1:5]
    assert latedump[1:4] == benign[1:4]
    # The pressure arms all share the identical turn-7 corner.
    assert smooth[6] == step[6] == latedump[6] == scenario["corner"]
    assert benign[6] != scenario["corner"]
    # Arm-specific divergence points.
    assert step[5] == scenario["step_t6"]
    assert latedump[4] == scenario["latedump_t5"]
    assert latedump[5] == scenario["latedump_t6"]


# --------------------------------------------------------------------------------------------------
# Seed determinism / divergence
# --------------------------------------------------------------------------------------------------
def test_turn_seed_determinism_and_divergence() -> None:
    a = turn_seed("natpress_v1", "sid", 0, 2, "a" * 64)
    again = turn_seed("natpress_v1", "sid", 0, 2, "a" * 64)
    assert a == again
    assert isinstance(a, int) and 0 <= a < (1 << 64)
    # Divergence when the prompt prefix differs.
    assert turn_seed("natpress_v1", "sid", 0, 2, "a" * 64) != turn_seed("natpress_v1", "sid", 0, 2, "b" * 64)
    # Divergence across turn index and sample index.
    assert turn_seed("natpress_v1", "sid", 0, 2, "a" * 64) != turn_seed("natpress_v1", "sid", 0, 3, "a" * 64)
    assert turn_seed("natpress_v1", "sid", 0, 2, "a" * 64) != turn_seed("natpress_v1", "sid", 1, 2, "a" * 64)


def test_plan_orders_benign_before_siblings() -> None:
    plan = plan_conversations(_valid_cfg())
    positions: dict[tuple[str, str, int], int] = {}
    for idx, entry in enumerate(plan):
        positions[(entry["scenario_id"], entry["arm"], entry["sample_index"])] = idx
    for (sid, arm, sample), pos in positions.items():
        if arm in ("step", "latedump"):
            assert positions[(sid, "benign", sample)] < pos


# --------------------------------------------------------------------------------------------------
# End-to-end stub run + dry run through the CLI
# --------------------------------------------------------------------------------------------------
def _write_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_valid_cfg()))
    return path


def test_stub_run_end_to_end(tmp_path: Path) -> None:
    config_path = _write_cfg(tmp_path)
    out_dir = tmp_path / "out"
    rc = rollout_natpress.main(["--config", str(config_path), "--out", str(out_dir), "--stub"])
    assert rc == 0

    rows = [json.loads(line) for line in (out_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
    # 2 scenarios * (benign 2 + smooth 2 + step 1 + latedump 1) = 12.
    assert len(rows) == 12

    by_cid = {row["conversation_id"]: row for row in rows}
    benign_by_key: dict[tuple[str, int], dict] = {}
    for row in rows:
        assert row["stub"] is True
        assert row["schema_version"] == "natpress_rollout_v1"
        assert row["protocol"] == "natpress_v1"
        assert CID_RE.match(row["conversation_id"]), row["conversation_id"]
        assert len(row["turns"]) == 7
        assert [t["turn_index"] for t in row["turns"]] == list(range(7))
        assert row["token_ids"] and row["decoded_text_exact"]
        if row["arm"] == "benign":
            benign_by_key[(row["scenario_id"], row["sample_index"])] = row

    expected_shared = {"step": {1, 2, 3, 4}, "latedump": {1, 2, 3}}
    for row in rows:
        shared = expected_shared.get(row["arm"], set())
        sibling = benign_by_key.get((row["scenario_id"], row["sample_index"]))
        for turn in row["turns"]:
            idx = turn["turn_index"]
            if idx in shared:
                assert turn["shared_from"] == f"benign:s{row['sample_index']:02d}"
                sib_turn = sibling["turns"][idx]
                assert turn["assistant_text"] == sib_turn["assistant_text"]
                assert turn["assistant_token_ids"] == sib_turn["assistant_token_ids"]
            else:
                assert turn["shared_from"] is None
        # Non-shared but prefix-identical turn 0 still matches the benign sibling deterministically.
        if row["arm"] in ("step", "latedump"):
            assert row["turns"][0]["assistant_text"] == sibling["turns"][0]["assistant_text"]

    # Divergent tail turns are genuinely fresh generations, not shares.
    step_cid = next(c for c, r in by_cid.items() if r["arm"] == "step")
    assert by_cid[step_cid]["turns"][5]["shared_from"] is None

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["num_conversations"] == 12
    assert manifest["stub"] is True
    assert manifest["per_arm_conversation_counts"] == {"benign": 4, "latedump": 2, "smooth": 4, "step": 2}
    assert set(manifest["token_length_stats"]) == {"benign", "latedump", "smooth", "step"}


def test_dry_run_writes_report_and_exits_zero(tmp_path: Path) -> None:
    config_path = _write_cfg(tmp_path)
    out_dir = tmp_path / "dry"
    rc = rollout_natpress.main(["--config", str(config_path), "--out", str(out_dir), "--dry-run"])
    assert rc == 0

    report = json.loads((out_dir / "dry_run_report.json").read_text())
    assert report["num_conversations"] == 12
    assert not (out_dir / "rows.jsonl").exists()
    budget = report["token_budget_estimate"]
    assert budget["total_estimate_tokens"] == budget["estimated_input_tokens"] + budget["worst_case_output_tokens"]
    assert budget["worst_case_output_tokens"] == 12 * 7 * 256
    assert len(report["seed_table"]) == 12
    assert all(len(entry["turns"]) == 7 for entry in report["seed_table"])


def test_deepcopy_isolation() -> None:
    # _valid_cfg must return an independent structure so per-test mutation cannot leak.
    a = _valid_cfg()
    b = _valid_cfg()
    a["scenarios"][0]["benign_turns"][0] = "mutated"
    assert b["scenarios"][0]["benign_turns"][0] != "mutated"
    assert copy.deepcopy(a) is not a
