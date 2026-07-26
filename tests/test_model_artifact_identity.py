from __future__ import annotations

import json
import shutil

import pytest

from geoprobe.models.artifact_identity import (
    fingerprint_local_hf_artifact,
    validate_local_hf_artifact_identity,
    validate_local_hf_support_identity,
)


def _model_artifact(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"llama"}\n')
    (tmp_path / "generation_config.json").write_text('{"eos_token_id":2}\n')
    (tmp_path / "tokenizer.json").write_text('{"version":"1.0"}\n')
    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length":128}\n')
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"weight-one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"weight-two")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }))
    return tmp_path


def test_model_artifact_identity_binds_weights_and_tokenizer(tmp_path) -> None:
    model_dir = _model_artifact(tmp_path)
    identity = fingerprint_local_hf_artifact(model_dir)
    validate_local_hf_artifact_identity(identity)
    assert identity["file_count"] == 7

    (model_dir / "tokenizer.json").write_text('{"version":"changed"}\n')
    tokenizer_changed = fingerprint_local_hf_artifact(model_dir)
    assert tokenizer_changed["tokenizer_sha256"] != identity["tokenizer_sha256"]
    assert tokenizer_changed["artifact_sha256"] != identity["artifact_sha256"]

    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"changed-weight")
    weights_changed = fingerprint_local_hf_artifact(model_dir)
    assert weights_changed["weights_sha256"] != tokenizer_changed["weights_sha256"]
    assert weights_changed["artifact_sha256"] != tokenizer_changed["artifact_sha256"]


def test_model_artifact_identity_rejects_tampered_inventory(tmp_path) -> None:
    identity = fingerprint_local_hf_artifact(_model_artifact(tmp_path))
    identity["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate digest"):
        validate_local_hf_artifact_identity(identity)


def test_model_artifact_identity_requires_local_complete_hf_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="local model directory"):
        fingerprint_local_hf_artifact(tmp_path / "missing")
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "tokenizer.json").write_text("{}\n")
    with pytest.raises(ValueError, match="weight artifact"):
        fingerprint_local_hf_artifact(tmp_path)


def test_model_support_can_bind_verified_remote_weight_identity(tmp_path) -> None:
    full = tmp_path / "full"
    full.mkdir()
    model_dir = _model_artifact(full)
    identity = fingerprint_local_hf_artifact(model_dir)
    support = tmp_path / "support"
    support.mkdir()
    for name in (
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    ):
        shutil.copy2(model_dir / name, support / name)
    validate_local_hf_support_identity(support, identity)

    (support / "tokenizer_config.json").write_text('{"model_max_length":256}\n')
    with pytest.raises(ValueError, match="support differs"):
        validate_local_hf_support_identity(support, identity)
