"""Clean-clone contract for the public paper artifact manifest."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.verify_paper_artifacts import (
    EXACT_IDENTITY_REQUIRED_CLAIMS,
    REGISTRATION_TIERS,
    _archive_regular_paths,
    main,
    validate_manifest,
)

CLAIM_IDS = ("C1", "C2", "C5", "C9", "C10", "C11", "C12", "C13")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return json.loads((REPO_ROOT / "paper_artifacts" / "manifest.json").read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry() -> dict:
    return {
        "results": [
            {"id": f"{claim_id.lower()}_result", "claim": claim_id}
            for claim_id in CLAIM_IDS
        ],
        "claims": [{"id": claim_id} for claim_id in CLAIM_IDS],
    }


def _provenance(claim_id: str) -> dict:
    return {
        "registration_tier": "prospective" if claim_id == "C5" else "retrospective_synthesis",
        "confirmatory": claim_id == "C5",
        "lifecycle_status": "completed",
        "source_commit": "a" * 40,
        "source_tree_dirty": False,
        "note": "",
    }


def _model_identity(claim_id: str) -> dict:
    if claim_id in EXACT_IDENTITY_REQUIRED_CLAIMS:
        return {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "weights_sha256": hashlib.sha256(f"{claim_id}:weights".encode()).hexdigest(),
            "config_sha256": hashlib.sha256(f"{claim_id}:config".encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(f"{claim_id}:tokenizer".encode()).hexdigest(),
            "note": "",
        }
    return {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": None,
        "weights_sha256": None,
        "config_sha256": None,
        "tokenizer_sha256": None,
        "note": "The legacy run did not retain immutable model component hashes.",
    }


def _entry(claim_id: str) -> dict:
    slug = claim_id.lower()
    producer = f"experiments/produce_{slug}.py"
    return {
        "artifact_id": f"receipt-{slug}",
        "result_id": f"{slug}_result",
        "claim_id": claim_id,
        "location": {"kind": "tracked_path", "path": f"paper_artifacts/{slug}.json"},
        "sha256": "0" * 64,
        "byte_size": 1,
        "producer": producer,
        "producer_command": f".venv/bin/python {producer} --out paper_artifacts/{slug}.json",
        "input_hashes": {"frozen_input": hashlib.sha256(claim_id.encode()).hexdigest()},
        "model_dependent": True,
        "model_identity": _model_identity(claim_id),
        "provenance": _provenance(claim_id),
    }


def _write_receipt(repo_root: Path, entry: dict, payload_updates: dict | None = None) -> None:
    producer_path = repo_root / entry["producer"]
    producer_path.parent.mkdir(parents=True, exist_ok=True)
    if not producer_path.exists():
        producer_path.write_text(f"# producer for {entry['claim_id']}\n")
    payload = {
        "claim_id": entry["claim_id"],
        "kind": "public_evidence_receipt",
        "producer": entry["producer"],
        "producer_sha256": hashlib.sha256(producer_path.read_bytes()).hexdigest(),
    }
    if payload_updates:
        payload.update(payload_updates)
    receipt = repo_root / entry["location"]["path"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    receipt.write_bytes(content)
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    entry["byte_size"] = len(content)


def _fixture(tmp_path: Path) -> tuple[dict, dict, set[str]]:
    entries = [_entry(claim_id) for claim_id in CLAIM_IDS]
    tracked: set[str] = set()
    for entry in entries:
        _write_receipt(tmp_path, entry)
        tracked.update((entry["location"]["path"], entry["producer"]))
    manifest = {
        "schema_version": 2,
        "registry_path": "docs/results_registry.yaml",
        "artifacts": entries,
    }
    return manifest, _registry(), tracked


def _errors(tmp_path: Path, manifest: dict, registry: dict, tracked: set[str]) -> list[str]:
    return validate_manifest(
        manifest,
        repo_root=tmp_path,
        registry=registry,
        tracked_paths=tracked,
    )


def _by_claim(manifest: dict, claim_id: str) -> dict:
    return next(entry for entry in manifest["artifacts"] if entry["claim_id"] == claim_id)


@pytest.mark.parametrize(
    ("claim_id", "hash_key", "config_path"),
    [
        ("C9", "scripted_protocol", "configs/natpress_arcs_full_v3.json"),
        ("C9", "dissociation_protocol", "configs/natpress_dissociation_v1.json"),
        ("C10", "structured_action_protocol", "configs/relational_structured_action_protocol.json"),
        ("C11", "structured_action_protocol", "configs/relational_structured_action_protocol.json"),
        ("C13", "structured_action_protocol", "configs/relational_structured_action_protocol.json"),
    ],
)
def test_shipped_manifest_inputs_match_frozen_protocol_configs(claim_id: str, hash_key: str, config_path: str) -> None:
    manifest = _manifest()
    entry = _by_claim(manifest, claim_id)
    expected = entry["input_hashes"][hash_key]
    path = REPO_ROOT / config_path
    assert path.is_file()
    assert expected == _file_sha256(path)


def test_schema_v2_accepts_exact_eight_entry_registry_closure(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    assert len(manifest["artifacts"]) == 8
    assert _errors(tmp_path, manifest, registry, tracked) == []


def test_empty_manifest_no_longer_fabricates_a_valid_evidence_set(tmp_path):
    manifest = {
        "schema_version": 2,
        "registry_path": "docs/results_registry.yaml",
        "artifacts": [],
    }
    errors = _errors(tmp_path, manifest, _registry(), set())
    assert sum("missing registry result_id" in error for error in errors) == 8
    assert sum("missing registry claim_id" in error for error in errors) == 8


def test_missing_and_duplicate_result_and_claim_links_fail(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    manifest["artifacts"].pop()
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("missing registry result_id 'c13_result'" in error for error in errors)
    assert any("missing registry claim_id 'C13'" in error for error in errors)

    manifest, registry, tracked = _fixture(tmp_path)
    duplicate = copy.deepcopy(manifest["artifacts"][0])
    duplicate["artifact_id"] = "receipt-c1-duplicate"
    manifest["artifacts"].append(duplicate)
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("duplicate result_id 'c1_result'" in error for error in errors)
    assert any("duplicate claim_id 'C1'" in error for error in errors)


def test_result_to_claim_mismatch_fails_closed(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    _by_claim(manifest, "C1")["claim_id"] = "C2"
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("result 'c1_result' belongs to claim 'C1', not 'C2'" in error for error in errors)
    assert any("duplicate claim_id 'C2'" in error for error in errors)
    assert any("missing registry claim_id 'C1'" in error for error in errors)


def test_tracked_json_receipt_binds_claim_producer_and_current_producer_hash(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    producer_path = tmp_path / entry["producer"]
    producer_path.write_text("# changed producer bytes\n")
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("embedded producer_sha256" in error for error in errors)

    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    _write_receipt(tmp_path, entry, {"producer": "experiments/not_the_producer.py"})
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("embedded producer" in error and "does not match" in error for error in errors)

    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    entry["producer_command"] = ".venv/bin/python experiments/produce_c1.py.backup"
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("must reference declared producer" in error for error in errors)


def test_tracked_json_receipt_embedded_claim_must_match(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    _write_receipt(tmp_path, entry, {"claim_id": "C2"})
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("embedded claim_id 'C2'" in error for error in errors)


def test_legacy_identity_can_be_incomplete_only_with_explicit_note(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    assert _errors(tmp_path, manifest, registry, tracked) == []

    identity = _by_claim(manifest, "C1")["model_identity"]
    identity["note"] = ""
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("note must explain missing identity fields" in error for error in errors)

    identity["note"] = "Legacy identity is incomplete."
    identity["config_sha256"] = "unknown"
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("config_sha256 must be a lowercase SHA-256" in error for error in errors)


@pytest.mark.parametrize("claim_id", sorted(EXACT_IDENTITY_REQUIRED_CLAIMS))
def test_exact_identity_claims_reject_missing_components_even_with_note(tmp_path, claim_id):
    manifest, registry, tracked = _fixture(tmp_path)
    identity = _by_claim(manifest, claim_id)["model_identity"]
    identity["revision"] = None
    identity["tokenizer_sha256"] = None
    identity["note"] = "The legacy run did not preserve these values."
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any(f"claim {claim_id} requires exact model identity" in error for error in errors)


def test_c10_cannot_be_confirmatory(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    provenance = _by_claim(manifest, "C10")["provenance"]
    provenance["registration_tier"] = "prospective"
    provenance["confirmatory"] = True
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("claim 'C10' cannot be marked confirmatory" in error for error in errors)


def test_confirmatory_is_reserved_for_prospective_c5(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    provenance = _by_claim(manifest, "C1")["provenance"]
    provenance["confirmatory"] = True
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("must use the prospective registration tier" in error for error in errors)
    assert any("claim 'C1' cannot be marked confirmatory" in error for error in errors)


@pytest.mark.parametrize("tier", sorted(REGISTRATION_TIERS))
def test_all_declared_registration_tiers_are_accepted_when_nonconfirmatory(tmp_path, tier):
    manifest, registry, tracked = _fixture(tmp_path)
    provenance = _by_claim(manifest, "C1")["provenance"]
    provenance["registration_tier"] = tier
    provenance["confirmatory"] = False
    assert _errors(tmp_path, manifest, registry, tracked) == []


def test_all_public_evidence_entries_are_model_dependent(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    entry["model_dependent"] = False
    entry["model_identity"] = None
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("model_dependent must be true" in error for error in errors)


def test_untracked_artifact_and_producer_fail(tmp_path):
    manifest, registry, _ = _fixture(tmp_path)
    errors = _errors(tmp_path, manifest, registry, set())
    assert any("artifact is not Git-tracked" in error for error in errors)
    assert any("producer is not Git-tracked" in error for error in errors)


def test_archive_mode_uses_present_regular_files_without_changing_default(
    tmp_path, capsys
):
    manifest, registry, _ = _fixture(tmp_path)
    manifest_path = tmp_path / "paper_artifacts" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    registry_path = tmp_path / "docs" / "results_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry))

    assert main([
        "--manifest",
        str(manifest_path),
        "--repo-root",
        str(tmp_path),
    ]) == 1
    assert "could not enumerate Git-tracked files" in capsys.readouterr().err

    assert main([
        "--manifest",
        str(manifest_path),
        "--repo-root",
        str(tmp_path),
        "--archive-mode",
    ]) == 0
    output = capsys.readouterr().out
    assert "Git provenance was not asserted" in output


def test_archive_regular_paths_exclude_symlinks_and_cache_directories(tmp_path):
    (tmp_path / "kept.txt").write_text("kept")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"cache")
    (tmp_path / "link.txt").symlink_to(tmp_path / "kept.txt")

    assert _archive_regular_paths(tmp_path) == {"kept.txt"}


@pytest.mark.parametrize(
    "path",
    (
        ".private/c1.json",
        "paper_artifacts/../.private/c1.json",
        "paper_artifacts/runpod_results/c1.json",
        "/tmp/c1.json",
    ),
)
def test_private_operational_and_escaping_paths_fail(tmp_path, path):
    manifest, registry, tracked = _fixture(tmp_path)
    _by_claim(manifest, "C1")["location"]["path"] = path
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("location.path" in error for error in errors)


def test_remote_receipt_metadata_is_checked_without_network(tmp_path):
    manifest, registry, tracked = _fixture(tmp_path)
    entry = _by_claim(manifest, "C1")
    entry["location"] = {
        "kind": "immutable_url",
        "url": "https://zenodo.org/records/123456/files/c1.json",
        "immutable_id": "zenodo-record:123456",
    }
    assert _errors(tmp_path, manifest, registry, tracked) == []

    entry["location"]["url"] = "https://example.org/latest/c1.json?token=secret"
    entry["location"].pop("immutable_id")
    errors = _errors(tmp_path, manifest, registry, tracked)
    assert any("query string" in error for error in errors)
    assert any("immutable_id" in error for error in errors)
