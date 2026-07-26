from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_relational_gauge_source as cli
from geoprobe.io import file_sha256


def _argv(*extra: str) -> list[str]:
    digest = "a" * 64
    return [
        "--replay-plan",
        "plan",
        "--vector-bank-root",
        "vectors",
        "--source-rollout-manifest",
        "source.json",
        "--rooted-star-bank",
        "stars",
        "--rooted-graph-artifact-root",
        "graphs",
        "--controller-artifact-root",
        "controller",
        "--expected-replay-plan-ledger-sha256",
        digest,
        "--expected-rooted-star-manifest-sha256",
        digest,
        "--expected-rooted-graph-manifest-sha256",
        digest,
        "--expected-controller-manifest-sha256",
        digest,
        "--expected-controller-internal-sha256",
        digest,
        "--expected-root-count",
        "1",
        "--expected-event-count",
        "1",
        *extra,
    ]


def _patch_static(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "FROZEN_ROOT_COUNT", 1)
    monkeypatch.setattr(cli, "FROZEN_EVENT_COUNT", 1)
    work = SimpleNamespace(causal=SimpleNamespace(events=({"event_id": "e"},)))
    monkeypatch.setattr(
        cli,
        "_validate_static_inputs",
        lambda *_args, **_kwargs: (
            {"ledger_sha256": "b" * 64},
            (),
            {},
            object(),
            object(),
            {"outer_1": ()},
            (work,),
            {},
            "c" * 64,
        ),
    )
    monkeypatch.setattr(cli, "_resource_loader", lambda **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "_validate_controller_inventory",
        lambda *_args, **_kwargs: {"active": 1},
    )


def test_validate_only_reports_inventory_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_static(monkeypatch)
    assert cli.main(_argv("--validate-only")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated"
    assert result["root_count"] == 1
    assert result["event_count"] == 1
    assert result["expected_row_count"] == 4
    assert result["proposal_status_counts"] == {"active": 1}


def test_validate_only_rejects_output_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_static(monkeypatch)
    with pytest.raises(ValueError, match="creates no output"):
        cli.main(_argv("--validate-only", "--out", "rows.jsonl"))


def test_fold_loader_binds_bundle_and_tensor_to_top_level_manifest(
    tmp_path: Path,
) -> None:
    fold_root = tmp_path / "outer_1"
    fold_root.mkdir()
    bundle = fold_root / "bundle.json"
    tensor = fold_root / "controller.safetensors"
    bundle.write_text("{}\n")
    tensor.write_bytes(b"tensor")
    manifest = {
        "fold_artifacts": {
            "outer_1": {
                "path": "outer_1/bundle.json",
                "sha256": file_sha256(bundle),
                "tensor_path": "outer_1/controller.safetensors",
                "tensor_sha256": file_sha256(tensor),
            }
        }
    }
    assert cli._validated_fold_artifact_root(
        tmp_path, manifest, "outer_1"
    ) == fold_root
    bundle.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="top-level manifest"):
        cli._validated_fold_artifact_root(tmp_path, manifest, "outer_1")
