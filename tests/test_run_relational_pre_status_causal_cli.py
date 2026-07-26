from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import run_relational_pre_status_causal as cli
from geoprobe.io import file_sha256


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _contract(
    *,
    model_sha: str = cli.FROZEN_MODEL_ARTIFACT_SHA256,
    tokenizer_map: dict[str, str] | None = None,
    contract_sha: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 2,
        "kind": cli.FROZEN_SOURCE_CONTRACT_KIND,
        "model_artifact_sha256": model_sha,
        "tokenizer_artifact_sha256": dict(tokenizer_map or cli.FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256),
        "source_sha256": {},
        "scenario_order": [],
        "generation": {"mode": "test"},
    }
    body["contract_sha256"] = (
        contract_sha if contract_sha is not None else cli.canonical_json_sha256(cli._canonical_contract_body(body))
    )
    return body


def _manifest(path: Path, rows_sha: str, *, contract: dict[str, Any] | None = None) -> None:
    payload = {
        "schema_version": cli.FROZEN_SOURCE_MANIFEST_SCHEMA_VERSION,
        "kind": cli.FROZEN_SOURCE_MANIFEST_KIND,
        "status": "success",
        "output": {"rows_sha256": rows_sha},
        "rollout_contract": contract or _contract(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _binding(tmp_path: Path, rows_sha: str, manifest_path: Path) -> dict[str, Any]:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    return {
        "rows": {
            "path": str(rows_path),
            "sha256": rows_sha,
            "bytes": rows_path.stat().st_size,
        },
        "rollout_manifest": {
            "path": str(tmp_path / "bound_manifest.json"),
            "sha256": file_sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
    }


def _identity(tokenizer_map: dict[str, str] | None = None) -> dict[str, Any]:
    files = [
        {"path": name, "bytes": 8, "sha256": digest}
        for name, digest in (tokenizer_map or cli.FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256).items()
    ]
    return {
        "schema_version": 1,
        "kind": "local_hf_artifact_identity",
        "artifact_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "files": files,
        "file_count": len(files),
        "total_bytes": 64,
    }


def test_relocated_source_rollout_manifest_passes_physical_validation(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "actual" / "manifest.json"
    manifest_path.parent.mkdir()
    _manifest(manifest_path, rows_sha)
    source_binding = _binding(tmp_path, rows_sha, manifest_path)

    validated = cli._validate_source_rollout_manifest(source_binding, manifest_path)
    assert "path" not in validated
    assert validated["rows_sha256"] == rows_sha
    assert validated["rollout_contract"]["model_artifact_sha256"] == cli.FROZEN_MODEL_ARTIFACT_SHA256


def test_source_rollout_manifest_hash_mismatch_rejected(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, rows_sha)
    source_binding = _binding(tmp_path, rows_sha, manifest_path)
    source_binding["rollout_manifest"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="physical binding"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path)


def test_source_rollout_rows_hash_mismatch_rejected(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, rows_sha=_sha("different"))
    source_binding = _binding(tmp_path, rows_sha, manifest_path)

    with pytest.raises(ValueError, match="rows hash is not bound"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path)


def test_source_rollout_contract_hash_rejected(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    contract = _contract(contract_sha="0" * 64)
    _manifest(manifest_path, rows_sha, contract=contract)
    source_binding = _binding(tmp_path, rows_sha, manifest_path)

    with pytest.raises(ValueError, match="canonical hash is invalid"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path)


def test_source_rollout_model_sha_rejected(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, rows_sha, contract=_contract(model_sha="0" * 64))
    source_binding = _binding(tmp_path, rows_sha, manifest_path)

    with pytest.raises(ValueError, match="does not reference the frozen source model"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path)


def test_source_rollout_tokenizer_mapping_mismatch_is_rejected_live(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path, rows_sha)
    source_binding = _binding(tmp_path, rows_sha, manifest_path)
    bad_identity = _identity({**cli.FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256, "tokenizer.json": "0" * 64})

    with pytest.raises(ValueError, match="do not match validated model identity"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path, model_identity=bad_identity)


def test_live_main_includes_compact_source_rollout_manifest_in_input_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "source_manifest.json"
    _manifest(manifest_path, rows_sha)
    source_binding = _binding(tmp_path, rows_sha, manifest_path)
    replay_plan = tmp_path / "plan"
    replay_plan.mkdir()
    (tmp_path / "ledger_hash").write_text("plan")
    captured: dict[str, Any] = {}
    ledger_sha = "a" * 64
    ledger = {
        "ledger_sha256": ledger_sha,
        "replay_plan": {},
        "rollout_binding": source_binding,
    }
    work = SimpleNamespace(events=[{}])
    monkeypatch.setattr(
        cli,
        "load_causal_replay_inputs",
        lambda *_values, **_kwargs: (ledger, (work,), {"replay_plan": {}}),
    )
    monkeypatch.setattr(cli, "_read_identity", lambda *_args, **_kwargs: _identity())
    monkeypatch.setattr(cli, "_model_provenance", lambda *_args, **_kwargs: {"stub": True})
    monkeypatch.setattr(cli, "_reject_runtime_overrides", lambda: None)
    monkeypatch.setattr(cli, "load_hf_model", lambda * _values, **_kwargs: (SimpleNamespace(parameters=tuple), SimpleNamespace(encode=lambda *_args, **_kwargs: (), decode=lambda *_args, **_kwargs: ""), {"loaded": True}))
    monkeypatch.setattr(
        cli,
        "_assert_effective_cuda_bf16",
        lambda *_args, **_kwargs: {
            "quantized": False,
            "parameter_cuda_device_indices": [0],
        },
    )
    monkeypatch.setattr(
        cli,
        "_cuda_runtime_identity",
        lambda *_args, **_kwargs: {"device_uuid": "GPU-test"},
    )
    monkeypatch.setattr(cli, "_validate_status_tokenizer", lambda *_args, **_kwargs: None)

    def _run_exact(*_values: Any, **kwargs: Any) -> dict[str, Any]:
        captured["input_hashes"] = dict(kwargs["input_hashes"])
        return {"status": "success", "completed_rows": 1, "expected_row_count": 1}

    monkeypatch.setattr(cli, "run_exact_prefix_causal_replay", _run_exact)

    monkeypatch.setattr(cli, "FROZEN_ROOT_COUNT", 1)
    monkeypatch.setattr(cli, "FROZEN_EVENT_COUNT", 1)
    monkeypatch.setattr(cli, "FROZEN_REPLAY_PLAN_LEDGER_SHA256", ledger_sha)

    args = [
        "--replay-plan", str(replay_plan),
        "--vector-bank-root", str(tmp_path),
        "--source-rollout-manifest", str(manifest_path),
        "--expected-root-count", "1",
        "--expected-event-count", "1",
        "--expected-replay-plan-ledger-sha256", ledger_sha,
        "--model", str(tmp_path / "model"),
        "--model-artifact-identity", str(tmp_path / "identity.json"),
        "--expected-model-artifact-sha256", cli.FROZEN_MODEL_ARTIFACT_SHA256,
        "--expected-tokenizer-sha256", cli.FROZEN_TOKENIZER_SHA256,
        "--out", str(tmp_path / "causal_rows.jsonl"),
    ]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "identity.json").write_text(json.dumps(_identity()), encoding="utf-8")

    cli.main(args)

    assert not (tmp_path / "GPU_WORK_STARTED").exists()
    assert "path" not in captured["input_hashes"]["source_rollout_manifest"]
    assert captured["input_hashes"]["source_rollout_manifest"]["rollout_contract"]["tokenizer_artifact_sha256"] == dict(sorted(cli.FROZEN_SOURCE_TOKENIZER_ARTIFACT_SHA256.items()))


def test_source_rollout_manifest_nonfinite_rejected(tmp_path: Path) -> None:
    rows_sha = _sha("rows")
    manifest_path = tmp_path / "manifest.json"
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        "{\"schema_version\":2,\"kind\":\"" + cli.FROZEN_SOURCE_MANIFEST_KIND + "\",\"status\":\"success\",\"output\":{\"rows_sha256\":\"" + rows_sha + "\"},\"rollout_contract\":{}NaN}",
        encoding="utf-8",
    )
    source_binding = {
        "rows": {"path": str(rows_path), "sha256": rows_sha, "bytes": rows_path.stat().st_size},
        "rollout_manifest": {"path": str(tmp_path / "manifest.json"), "sha256": file_sha256(manifest_path), "bytes": manifest_path.stat().st_size},
    }

    with pytest.raises(ValueError, match="not finite UTF-8 JSON"):
        cli._validate_source_rollout_manifest(source_binding, manifest_path)
