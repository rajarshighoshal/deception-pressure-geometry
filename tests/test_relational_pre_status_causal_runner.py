from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from geoprobe.control.relational_pre_status_causal import (
    CAUSAL_ARM_ORDER,
    CAUSAL_LAYERS,
    PRIMARY_ACTUATION_LAYER,
    _array_sha256,
    build_relational_pre_status_causal_arm_bundle,
)
from geoprobe.models.relational_causal_replay import forward_status_arm_logits_hf
from geoprobe.models.relational_structured_action import SKIP_TOKEN_ID, STATUS_PREFIX_TOKEN_IDS, int32_token_sha256
import geoprobe.runtime.relational_pre_status_causal_runner as runner
from geoprobe.runtime.relational_pre_status_causal_runner import (
    CausalRootWork,
    _sample_outcome_class,
)


class _Tokenizer:
    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return f" token-{token_ids[0]}"


def _work(root: str, seeds: tuple[int, ...]) -> CausalRootWork:
    h = np.asarray([[2.0, 1.0, -1.0]] * 4, dtype=np.float32)
    t = np.asarray([[1.0, -0.5, 0.25]] * 4, dtype=np.float32)
    g = np.asarray([[0.5, 0.25, 0.75]] * 4, dtype=np.float32)
    prefix = (91, *STATUS_PREFIX_TOKEN_IDS)
    prefix_sha = int32_token_sha256(prefix)
    root_id = sha256(root.encode()).hexdigest()
    events = tuple({
        "event_id": f"{root}-event-{index}", "scenario_id": f"scenario-{root}",
        "family": "family-a",
        "turn_index": 1, "true_status": "PASS", "desired_status": "FAIL",
        "knowledge_status": "PASS", "knowledge_correct": True, "rng_seed": seed,
    } for index, seed in enumerate(seeds))
    return CausalRootWork(
        root_id, "outer_1", 0, prefix, prefix_sha, events,
        build_relational_pre_status_causal_arm_bundle(root_id, "outer_1", h, t, g),
    )


def _logits() -> torch.Tensor:
    logits = torch.full((len(CAUSAL_ARM_ORDER), SKIP_TOKEN_ID + 2), -30.0)
    logits[:, 7] = 4.0
    return logits


def _run(
    tmp_path, works, forward, *, resume: bool = False, runtime_provenance: dict[str, str] | None = None
):
    runtime_provenance = {"runtime": "fake"} if runtime_provenance is None else runtime_provenance
    return runner.run_exact_prefix_causal_replay(
        works, model=object(), tokenizer=_Tokenizer(), out=tmp_path / "rows.jsonl",
        manifest_out=tmp_path / "manifest.json", status_out=tmp_path / "STATUS",
        beta=1.0, expected_root_count=len(works),
        expected_event_count=sum(len(work.events) for work in works),
        input_hashes={"plan": "frozen"}, model_provenance={"model": "fake"},
        runtime_provenance=runtime_provenance, resume=resume, forward_fn=forward,
    )


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _actuation_vector_layer_hash(bundle, arm: str) -> str:
    layer_index = CAUSAL_LAYERS.index(PRIMARY_ACTUATION_LAYER)
    return _array_sha256(bundle.arm_vectors[arm][layer_index])


def test_row_includes_actuation_vector_hash_and_hook_layers(tmp_path) -> None:
    work = _work("one", (17, 19))
    manifest = _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    rows = [json.loads(row) for row in (tmp_path / "rows.jsonl").read_text().splitlines()]

    assert manifest["completed_rows"] == 14
    for row in rows:
        expected_hook_layers = [] if row["arm"] == "noop" else [PRIMARY_ACTUATION_LAYER]
        assert row["hook_layers"] == expected_hook_layers
        assert row["actuation_vector_sha256"] == _actuation_vector_layer_hash(work.bundle, row["arm"])
    assert {row["bundle_hash"] for row in rows} == {work.bundle.bundle_hash}


def test_run_rejects_aliased_output_paths(tmp_path) -> None:
    work = _work("one", (17,))
    out = tmp_path / "rows.jsonl"
    status = tmp_path / "STATUS"

    with pytest.raises(ValueError, match="runner output paths must be distinct"):
        runner.run_exact_prefix_causal_replay(
            (work,), model=object(), tokenizer=_Tokenizer(), out=out, manifest_out=out, status_out=status, beta=1.0,
            expected_root_count=1, expected_event_count=len(work.events), input_hashes={"plan": "frozen"},
            model_provenance={"model": "fake"}, runtime_provenance={"runtime": "fake"}, forward_fn=lambda _model, _prefix, _steering: _logits(),
        )


def test_run_rejects_symlink_aliased_output_paths(tmp_path) -> None:
    work = _work("one", (17,))
    out = tmp_path / "rows.jsonl"
    out.write_text("")
    manifest_alias = tmp_path / "manifest-alias.json"
    manifest_alias.symlink_to(out)
    status = tmp_path / "STATUS"
    status.write_text("running\n")

    with pytest.raises(ValueError, match="runner output paths must be distinct"):
        runner.run_exact_prefix_causal_replay(
            (work,), model=object(), tokenizer=_Tokenizer(), out=out, manifest_out=manifest_alias, status_out=status, beta=1.0,
            expected_root_count=1, expected_event_count=len(work.events), input_hashes={"plan": "frozen"},
            model_provenance={"model": "fake"}, runtime_provenance={"runtime": "fake"}, resume=True,
            forward_fn=lambda _model, _prefix, _steering: _logits(),
        )



def test_one_root_forward_serves_multiple_event_rng_streams_with_crn(tmp_path) -> None:
    work = _work("one", (17, 19))
    calls: list[tuple[tuple[int, ...], int]] = []

    def forward(_model, prefix, steering):
        calls.append((tuple(prefix), len(steering)))
        assert all(spec is None or spec.layer == 12 for spec in steering)
        return _logits()

    manifest = _run(tmp_path, (work,), forward)
    rows = (tmp_path / "rows.jsonl").read_text().splitlines()
    decoded = [json.loads(row) for row in rows]

    assert calls == [(work.prefix_token_ids, 7)]
    assert manifest["completed_rows"] == 14
    assert {row["actuation_layer"] for row in decoded} == {12}
    assert {row["capture_hooks_enabled"] for row in decoded} == {False}
    assert {row["knowledge_status"] for row in decoded} == {"PASS"}
    assert {row["knowledge_correct"] for row in decoded} == {True}
    for event in work.events:
        event_rows = [row for row in decoded if row["event_id"] == event["event_id"]]
        assert len({row["raw_token_id"] for row in event_rows}) == 1
        assert {row["mapped_action"] for row in event_rows} == {"NO_ACTION"}
        assert {row["behavioral_outcome_class"] for row in event_rows} == {"NO_ACTION"}


def test_resume_accepts_only_complete_roots_and_does_not_duplicate(tmp_path) -> None:
    works = (_work("one", (17, 19)), _work("two", (23,)))
    _run(tmp_path, works, lambda _model, _prefix, _steering: _logits())
    before = (tmp_path / "rows.jsonl").read_text()

    def no_forward(*_args):
        raise AssertionError("completed roots must not run again")

    manifest = _run(tmp_path, works, no_forward, resume=True)
    assert manifest["status"] == "success"
    assert (tmp_path / "rows.jsonl").read_text() == before
    assert len(before.splitlines()) == 21


def test_resume_accepts_only_float32_probability_boundary_roundoff(tmp_path) -> None:
    work = _work("one", (17,))
    _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    row = json.loads((tmp_path / "rows.jsonl").read_text().splitlines()[0])
    contract = json.loads((tmp_path / "manifest.json").read_text())["contract"]
    probabilities = {
        "PASS": 0.051515281200408936,
        "FAIL": 0.8969695568084717,
        "SKIP": 0.051515281200408936,
    }
    row["status_probabilities"] = probabilities
    for action in ("PASS", "FAIL", "SKIP"):
        row[f"{action.lower()}_probability"] = probabilities[action]
    row["recognized_action_probability_mass"] = sum(probabilities.values())
    row["row_sha256"] = _canonical_sha(
        {name: row[name] for name in row if name != "row_sha256"}
    )

    runner._validate_row_against_plan(
        row,
        contract=contract,
        expected_root=work.root_id,
        expected_event_id=str(work.events[0]["event_id"]),
        expected_arm=str(row["arm"]),
        expected_event=work.events[0],
        work=work,
    )

    probabilities["SKIP"] = 0.00001
    probabilities["PASS"] = 0.5
    probabilities["FAIL"] = 0.5
    for action in ("PASS", "FAIL", "SKIP"):
        row[f"{action.lower()}_probability"] = probabilities[action]
    row["recognized_action_probability_mass"] = sum(probabilities.values())
    row["row_sha256"] = _canonical_sha(
        {name: row[name] for name in row if name != "row_sha256"}
    )
    with pytest.raises(ValueError, match="recognized_action_probability_mass"):
        runner._validate_row_against_plan(
            row,
            contract=contract,
            expected_root=work.root_id,
            expected_event_id=str(work.events[0]["event_id"]),
            expected_arm=str(row["arm"]),
            expected_event=work.events[0],
            work=work,
        )


def test_resume_rejects_corrupt_row_after_row_hash_update(tmp_path) -> None:
    work = _work("one", (17,))
    _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["knowledge_correct"] = not rows[0]["knowledge_correct"]
    rows[0]["row_sha256"] = _canonical_sha({name: rows[0][name] for name in rows[0] if name != "row_sha256"})
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    rows_path.write_text(payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_content_sha256"] = _canonical_sha(rows)
    manifest["rows_sha256"] = sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="resume row knowledge metadata differs from frozen plan"):
        _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits(), resume=True)


def test_resume_rejects_stale_rows_content_sha256(tmp_path) -> None:
    work = _work("one", (17,))
    _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_content_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="resume rows_content_sha256 differs from manifest rows"):
        _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits(), resume=True)


def test_resume_rejects_runtime_change_on_partial_manifest(tmp_path) -> None:
    works = (_work("one", (17, 19)), _work("two", (23,)))
    _run(tmp_path, works, lambda _model, _prefix, _steering: _logits())
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    status_path = tmp_path / "STATUS"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    partial = rows[: len(CAUSAL_ARM_ORDER) * len(works[0].events)]
    payload = "\n".join(json.dumps(row) for row in partial) + "\n"
    rows_path.write_text(payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest.pop("failure", None)
    manifest["completed_rows"] = len(partial)
    manifest["completed_root_count"] = len({row["root_id"] for row in partial})
    manifest["rows_content_sha256"] = _canonical_sha(partial)
    manifest["rows_sha256"] = sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    status_path.write_text("running\n")

    with pytest.raises(ValueError, match="resume manifest differs from exact frozen causal contract"):
        _run(
            tmp_path, works, lambda _model, _prefix, _steering: _logits(),
            runtime_provenance={"runtime": "changed"}, resume=True,
        )


def test_resume_rejects_rows_from_later_root(tmp_path) -> None:
    works = (_work("one", (17,)), _work("two", (23,)))
    _run(tmp_path, works, lambda _model, _prefix, _steering: _logits())
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    first_root = rows[0]["root_id"]
    later_rows = [row for row in rows if row["root_id"] != first_root]
    payload = "\n".join(json.dumps(row) for row in later_rows) + "\n"
    rows_path.write_text(payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest.pop("failure", None)
    manifest["completed_rows"] = len(later_rows)
    manifest["completed_root_count"] = len({row["root_id"] for row in later_rows})
    manifest["rows_content_sha256"] = _canonical_sha(later_rows)
    manifest["rows_sha256"] = sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "STATUS").write_text("running\n")

    with pytest.raises(ValueError, match="resume row root metadata is invalid"):
        _run(tmp_path, works, lambda _model, _prefix, _steering: _logits(), resume=True)


def test_resume_row_hash_is_verified(tmp_path) -> None:
    work = _work("one", (17,))
    _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["top_token_probability"] = 0.0
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    rows_path.write_text(payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_content_sha256"] = _canonical_sha(rows)
    manifest["rows_sha256"] = sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="resume row self-hash is invalid"):
        _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits(), resume=True)


def test_resume_replays_only_unfinished_roots_without_duplicateing(tmp_path) -> None:
    works = (_work("one", (17,)), _work("two", (23,)))
    _run(tmp_path, works, lambda _model, _prefix, _steering: _logits())
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    status_path = tmp_path / "STATUS"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    completed_root_row_count = len(CAUSAL_ARM_ORDER) * len(works[0].events)
    partial = rows[:completed_root_row_count]
    payload = "\n".join(json.dumps(row) for row in partial) + "\n"
    rows_path.write_text(payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest.pop("failure", None)
    manifest["completed_rows"] = len(partial)
    manifest["completed_root_count"] = len({row["root_id"] for row in partial})
    manifest["rows_content_sha256"] = _canonical_sha(partial)
    manifest["rows_sha256"] = sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    status_path.write_text("running\n")
    calls: list[tuple[tuple[int, ...], int]] = []

    def forward(_model, prefix, steering):
        calls.append((tuple(prefix), len(steering)))
        return _logits()

    manifest = _run(tmp_path, works, forward, resume=True)
    assert calls == [(works[1].prefix_token_ids, len(CAUSAL_ARM_ORDER))]
    assert manifest["status"] == "success"
    assert manifest["completed_rows"] == 14
    assert len([json.loads(line) for line in rows_path.read_text().splitlines()]) == 14
    assert manifest["rows_content_sha256"] == _canonical_sha([json.loads(row) for row in rows_path.read_text().splitlines()])


def test_runner_uses_eight_root_checkpoint_interval(tmp_path, monkeypatch) -> None:
    works = tuple(_work(f"root-{index}", (index,)) for index in range(9))
    calls: dict[str, int] = {"rows": 0, "manifest": 0, "status": 0}
    original_atomic_text = runner._atomic_text

    def _record(path, text: str) -> None:
        name = path.name
        if name == "rows.jsonl":
            calls["rows"] += 1
        if name == "manifest.json":
            calls["manifest"] += 1
        if name == "STATUS":
            calls["status"] += 1
        original_atomic_text(path, text)

    monkeypatch.setattr(runner, "_atomic_text", _record)

    manifest = _run(tmp_path, works, lambda _model, _prefix, _steering: _logits())
    assert manifest["completed_rows"] == 63
    assert calls["rows"] == 3
    assert calls["manifest"] == 3
    assert calls["status"] == 2


def test_resume_rejects_a_partial_root(tmp_path) -> None:
    work = _work("one", (17, 19))
    _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits())
    rows_path, manifest_path = tmp_path / "rows.jsonl", tmp_path / "manifest.json"
    truncated = "\n".join(rows_path.read_text().splitlines()[:-1]) + "\n"
    rows_path.write_text(truncated)
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_content_sha256"] = _canonical_sha([json.loads(row) for row in truncated.splitlines() if row])
    manifest["status"] = "running"
    manifest["completed_rows"] = len(truncated.splitlines())
    manifest["rows_sha256"] = sha256(truncated.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "STATUS").write_text("running\n")

    with pytest.raises(ValueError, match="partial root"):
        _run(tmp_path, (work,), lambda _model, _prefix, _steering: _logits(), resume=True)


def test_outcome_class_uses_canonical_no_baseline_knowledge_label() -> None:
    assert _sample_outcome_class("PASS", "PASS", False) == "HONEST"
    assert _sample_outcome_class("FAIL", "PASS", True) == "DECEPTIVE"
    assert _sample_outcome_class("FAIL", "PASS", False) == "WRONG_WITHOUT_BASELINE_KNOWLEDGE"


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1


class _HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Block() for _ in range(12))

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        hidden = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        for block in self.model.layers:
            hidden = block(hidden)
        logits = torch.zeros((*input_ids.shape, SKIP_TOKEN_ID + 2), device=input_ids.device)
        logits[..., 7] = hidden.sum(dim=-1)
        return SimpleNamespace(logits=logits)


def test_logits_once_helper_cleans_steering_hooks() -> None:
    work = _work("hooks", (17,))
    from geoprobe.models.relational_causal_replay import build_primary_actuation_steering_batch

    _names, steering = build_primary_actuation_steering_batch(work.bundle, beta=1.0)
    model = _HookModel()
    logits = forward_status_arm_logits_hf(model, work.prefix_token_ids, steering_batch=steering)

    assert logits.shape == (7, SKIP_TOKEN_ID + 2)
    assert all(not block._forward_hooks for block in model.model.layers)


def test_live_cli_rejects_quantization_or_device_map_overrides(monkeypatch) -> None:
    import experiments.run_relational_pre_status_causal as cli

    monkeypatch.setenv("GEOPROBE_HF_QUANTIZATION", "4bit")
    with pytest.raises(ValueError, match="forbids quantization"):
        cli._reject_runtime_overrides()
    monkeypatch.delenv("GEOPROBE_HF_QUANTIZATION")
    monkeypatch.setenv("GEOPROBE_HF_DEVICE_MAP", "auto")
    with pytest.raises(ValueError, match="device-map"):
        cli._reject_runtime_overrides()


def test_live_cli_rejects_effective_cpu_or_non_bf16_model() -> None:
    import experiments.run_relational_pre_status_causal as cli

    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="CUDA with BF16"):
        cli._assert_effective_cuda_bf16(model)


def test_cuda_runtime_identity_binds_hardware_and_determinism(monkeypatch) -> None:
    import experiments.run_relational_pre_status_causal as cli

    properties = SimpleNamespace(
        uuid="GPU-1234",
        name="Synthetic L40",
        major=8,
        minor=9,
        total_memory=48_000,
        multi_processor_count=142,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    identity = cli._cuda_runtime_identity(0)

    assert identity["device_uuid"] == "GPU-1234"
    assert identity["device_name"] == "Synthetic L40"
    assert identity["compute_capability"] == [8, 9]
    assert identity["total_memory_bytes"] == 48_000
    assert "torch_compiled_cuda" in identity
    assert "deterministic_algorithms_enabled" in identity

    properties.uuid = None
    with pytest.raises(RuntimeError, match="UUID is unavailable"):
        cli._cuda_runtime_identity(0)


def test_live_cli_frozen_trust_roots_are_exact() -> None:
    import experiments.run_relational_pre_status_causal as cli

    assert cli.FROZEN_REPLAY_PLAN_LEDGER_SHA256 == (
        "1970535a607bf650982aaa829e8ca9dedcae6824ce0b8d6c2fe7c44c3fd43732"
    )
    assert cli.FROZEN_MODEL_ARTIFACT_SHA256 == (
        "8071d53a4509c0404328b791800ba79657556490b276b8383e1e8b2f0f63e104"
    )
    assert cli.FROZEN_TOKENIZER_SHA256 == (
        "e901ac02d4fd9ca87689c4399c57897d03f261b9671f01808f9127614dc50b4c"
    )
    assert (cli.FROZEN_ROOT_COUNT, cli.FROZEN_EVENT_COUNT) == (402, 656)
