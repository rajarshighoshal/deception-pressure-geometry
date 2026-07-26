import json

import torch

from experiments.resident_decision_token_bank_worker import (
    assert_resume_identity,
    canonical_methods,
    load_eval_rows,
    seen_generation_keys,
)


def transcript(cid, family, arm="p3", true_status="PASS", reported_status="FAIL"):
    return {
        "conversation_id": cid,
        "scenario_id": f"s_{cid}",
        "family": family,
        "arm": arm,
        "sample_seed": 0,
        "valid_outcome": True,
        "true_status": true_status,
        "reported_status": reported_status,
        "desired_status": true_status,
        "deceptive": true_status != reported_status,
        "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_activation_bank(path, rows):
    n = len(rows)
    torch.save(
        {
            "activations": {20: torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)},
            "conversation_id": [row["conversation_id"] for row in rows],
            "scenario_id": [row["scenario_id"] for row in rows],
            "family": [row["family"] for row in rows],
            "arm": [row["arm"] for row in rows],
            "sample_seed": [row["sample_seed"] for row in rows],
            "turn_index": torch.tensor([3 for _ in rows], dtype=torch.long),
            "phase": ["pre_response" for _ in rows],
            "deceptive": torch.tensor([int(row["deceptive"]) for row in rows], dtype=torch.long),
            "true_status": [row["true_status"] for row in rows],
            "desired_status": [row["desired_status"] for row in rows],
            "layers": [20],
            "model_name": "m",
            "backend": "hf",
            "device": "cuda",
            "capture": {},
        },
        path,
    )


def test_canonical_methods_includes_real_action_library():
    assert canonical_methods("tangent,probe,random") == [
        "bidir_tangent",
        "global_probe",
        "random_global",
    ]


def test_load_eval_rows_shards_by_family_after_selection(tmp_path):
    rows = [
        transcript("a1", "alpha"),
        transcript("a2", "alpha", true_status="FAIL", reported_status="PASS"),
        transcript("b1", "beta"),
        transcript("c1", "charlie"),
    ]
    tx = tmp_path / "tx.jsonl"
    act = tmp_path / "turns.pt"
    write_jsonl(tx, rows)
    write_activation_bank(act, rows)

    shard0, _, _ = load_eval_rows(
        transcript_paths=[tx],
        activation_path=act,
        layer=20,
        query_turn=3,
        query_phase="pre_response",
        eval_levels={"p3"},
        limit=None,
        limit_per_status_class=None,
        limit_strategy="family_round_robin",
        seed=0,
        scenario_include=None,
        scenario_exclude=None,
        num_shards=2,
        shard_index=0,
    )
    shard1, _, _ = load_eval_rows(
        transcript_paths=[tx],
        activation_path=act,
        layer=20,
        query_turn=3,
        query_phase="pre_response",
        eval_levels={"p3"},
        limit=None,
        limit_per_status_class=None,
        limit_strategy="family_round_robin",
        seed=0,
        scenario_include=None,
        scenario_exclude=None,
        num_shards=2,
        shard_index=1,
    )

    assert {row["family"] for row in shard0}.isdisjoint({row["family"] for row in shard1})
    assert {row["conversation_id"] for row in shard0 + shard1} == {"a1", "a2", "b1", "c1"}


def test_generation_resume_keys_include_target_status():
    rows = [
        {"conversation_id": "c1", "method": "global_mean", "target_status": "PASS", "layer": 20, "alpha": 24},
        {"conversation_id": "c1", "method": "global_mean", "target_status": "FAIL", "layer": 20, "alpha": 24},
    ]

    assert len(seen_generation_keys(rows)) == 2


def test_assert_resume_identity_rejects_mismatch():
    prior = {"routing": "oracle_true_status", "alphas": [24.0]}
    current = {"routing": "gate_file", "alphas": [24.0]}

    try:
        assert_resume_identity(prior, current)
    except ValueError as exc:
        assert "routing" in str(exc)
    else:
        raise AssertionError("expected resume identity mismatch")
