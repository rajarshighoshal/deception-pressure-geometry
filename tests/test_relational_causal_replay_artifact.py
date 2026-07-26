from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

import geoprobe.data.relational_causal_replay_artifact as artifact
from geoprobe.control.relational_pre_status_vector_bank import CausalVectorBankArrays
from geoprobe.data.relational_causal_replay import (
    RelationalCausalReplayEvent,
    RelationalCausalReplayPlanEvent,
)
from geoprobe.models.relational_structured_action import int32_token_sha256


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _event(identifier: str, seed: int) -> RelationalCausalReplayEvent:
    return RelationalCausalReplayEvent(
        event_id=identifier, scenario_id="scenario", family="family", turn_index=1,
        true_status="PASS", desired_status="FAIL", prefix_token_ids=(1, 2, 3),
        knowledge_correct=True, knowledge_status="PASS",
        prefix_token_ids_sha256=int32_token_sha256((1, 2, 3)), rng_seed=seed,
        historical_raw_token_id=7, historical_raw_decoded_exact=" PASS",
        historical_mapped_action="PASS", source_conversation_ids=("conversation",),
        source_row_sha256s=(_sha("row"),),
    )


def _binding(tmp_path: Path) -> tuple[dict, dict]:
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "rollout.json"
    rows_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "relational_structured_action_rollout_manifest",
                "status": "success",
                "output": {"rows_sha256": artifact.file_sha256(rows_path)},
            }
        ),
        encoding="utf-8",
    )
    rollout = {
        "rows": {"path": str(rows_path), "sha256": artifact.file_sha256(rows_path), "bytes": rows_path.stat().st_size},
        "rollout_manifest": {"path": str(manifest_path), "sha256": artifact.file_sha256(manifest_path), "bytes": manifest_path.stat().st_size},
    }
    vectors = {}
    for fold in artifact.FOLDS:
        root = tmp_path / "inputs" / fold
        root.mkdir(parents=True)
        ledger, tensor = root / "ledger.json", root / "vectors.safetensors"
        ledger.write_text(fold, encoding="utf-8")
        tensor.write_bytes(f"tensor:{fold}".encode())
        vectors[fold] = {
            "undefined_root_count": 0, "undefined_event_count": 0,
            "ledger": {"path": str(ledger), "sha256": artifact.file_sha256(ledger), "bytes": ledger.stat().st_size},
            "tensor": {"path": str(tensor), "sha256": artifact.file_sha256(tensor), "bytes": tensor.stat().st_size},
        }
    return rollout, vectors


def _build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> artifact.RelationalCausalReplayArtifactBuild:
    first, second = _event("event-a", 11), _event("event-b", 12)
    plan = (
        RelationalCausalReplayPlanEvent(first, _sha("root-a"), "outer_1", 0),
        RelationalCausalReplayPlanEvent(second, _sha("root-a"), "outer_1", 0),
    )
    monkeypatch.setattr(artifact, "build_relational_causal_replay_inventory", lambda _rows: (first, second))
    monkeypatch.setattr(artifact, "join_relational_causal_replay_plan", lambda _inventory, _vectors: plan)
    rollout, vectors = _binding(tmp_path)
    vectors["outer_1"]["undefined_root_count"] = 2
    vectors["outer_1"]["undefined_event_count"] = 3
    return artifact.build_relational_causal_replay_artifact(
        [{"unused": True}], [{"unused": True}],
        rollout_binding=rollout, vector_bank_bindings=vectors,
        argv=("--out", "plan"), provenance={"git": {"git_hash": "test"}},
    )


def test_save_load_is_compact_and_immutable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build = _build(monkeypatch, tmp_path)
    ledger = artifact.save_relational_causal_replay_artifact(build, tmp_path / "plan")
    loaded, rows, arrays = artifact.load_relational_causal_replay_artifact(tmp_path / "plan")

    assert loaded["selection_did_not_use_historical_action"] is True
    assert ledger["excluded_undefined_root_count"] == 2
    assert [row["event_id"] for row in rows] == ["event-a", "event-b"]
    assert all(row["knowledge_correct"] is True for row in rows)
    assert all(row["knowledge_status"] == "PASS" for row in rows)
    assert arrays.flat_token_ids.dtype == np.int32
    assert arrays.offsets.tolist() == [0, 3]
    assert arrays.counts.tolist() == [3, 3]
    with pytest.raises(artifact.RelationalCausalReplayArtifactError, match="overwrite"):
        artifact.save_relational_causal_replay_artifact(build, tmp_path / "plan")


def test_load_rejects_event_tampering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    build = _build(monkeypatch, tmp_path)
    root = tmp_path / "plan"
    artifact.save_relational_causal_replay_artifact(build, root)
    (root / artifact.EVENTS_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(artifact.RelationalCausalReplayArtifactError, match="physical bindings"):
        artifact.load_relational_causal_replay_artifact(root)


def test_execution_load_is_portable_after_build_sources_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(monkeypatch, tmp_path)
    root = tmp_path / "plan"
    artifact.save_relational_causal_replay_artifact(build, root)
    rollout_rows = Path(build.ledger_payload["rollout_binding"]["rows"]["path"])
    rollout_rows.unlink()

    with pytest.raises(
        artifact.RelationalCausalReplayArtifactError,
        match="physical binding",
    ):
        artifact.load_relational_causal_replay_artifact(root)
    ledger, rows, _arrays = artifact.load_relational_causal_replay_artifact(
        root,
        validate_source_bindings=False,
    )
    assert ledger["ledger_sha256"]
    assert len(rows) == 2


def test_load_rejects_incoherent_knowledge_even_when_rehashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _build(monkeypatch, tmp_path)
    root = tmp_path / "plan"
    artifact.save_relational_causal_replay_artifact(build, root)
    events_path = root / artifact.EVENTS_NAME
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["knowledge_correct"] = False
    content = artifact._jsonl_bytes(rows)
    events_path.write_bytes(content)
    ledger_path = root / artifact.LEDGER_NAME
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["events"].update(
        {
            "bytes": len(content),
            "sha256": artifact.file_sha256(events_path),
            "content_sha256": sha256(content).hexdigest(),
        }
    )
    ledger["ledger_sha256"] = artifact._self_hash(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(artifact.RelationalCausalReplayArtifactError, match="knowledge semantics"):
        artifact.load_relational_causal_replay_artifact(root)


def test_only_all_h_t_s_g_defined_roots_are_selected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "banks"
    selected_id, excluded_id = _sha("selected"), _sha("excluded")
    for fold in artifact.FOLDS:
        fold_root = root / fold
        fold_root.mkdir(parents=True)
        (fold_root / "ledger.json").write_text("ledger", encoding="utf-8")
        (fold_root / "vectors.safetensors").write_bytes(b"tensor")

    def fake_load(fold_root: Path):
        fold = fold_root.name
        rows = []
        flags = []
        if fold == artifact.FOLDS[0]:
            rows = [
                {"root_id": selected_id, "tensor_row_index": 0, "event_ids": ["event-a"]},
                {"root_id": excluded_id, "tensor_row_index": 1, "event_ids": ["event-b", "event-c"]},
            ]
            flags = [[True, True, True, True], [True, True, False, True]]
        values = np.zeros((len(rows), 4, 1), dtype=np.float32)
        return {"held_out_family_fold": fold, "rows": rows, "ledger_sha256": _sha(fold)}, CausalVectorBankArrays(values, values, values, values, np.asarray(flags, dtype=np.bool_))

    monkeypatch.setattr(artifact, "load_pre_status_causal_vector_bank", fake_load)
    rows, bindings = artifact.load_all_defined_pre_status_causal_vector_rows(root)
    assert [row["root_id"] for row in rows] == [selected_id]
    assert bindings[artifact.FOLDS[0]]["undefined_root_count"] == 1
    assert bindings[artifact.FOLDS[0]]["undefined_event_count"] == 2
