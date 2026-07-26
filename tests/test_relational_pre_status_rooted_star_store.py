from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from geoprobe.data.relational_pre_status_rooted_star import build_status_rooted_star_views
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RootedStarStoreError,
    _canonical_annotation_codes,
    build_relational_pre_status_rooted_star_index,
    load_rooted_star_metric_input,
    load_rooted_star_observation_binding,
    load_rooted_star_root_residuals,
)
from geoprobe.io import file_sha256
from tests.test_relational_pre_status_rooted_star import _raw, _reference


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "rooted-stars"
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    tensors: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    template_views = build_status_rooted_star_views(_reference(), _raw())
    for row_index, conversation in enumerate(("conversation-a", "conversation-b")):
        stars: list[dict[str, object]] = []
        row_keys: list[str] = []
        for turn in range(4):
            reference = replace(
                _reference(), reference_id=f"{row_index + 1:x}{turn:x}" * 32,
                occurrence_id=f"occurrence-{row_index}-{turn}", field_event_id=f"event-{row_index}-{turn}",
                turn_index=turn, conversation_id=conversation,
            )
            encoded_views: list[dict[str, object]] = []
            for view in template_views:
                prefix = f"row_{row_index:06d}__star_{turn:02d}__{view.name}__"
                keyed = {name: prefix + name for name in view.tensors()}
                tensors.update({keyed[name]: value.clone() for name, value in view.tensors().items()})
                row_keys.extend(keyed.values())
                payload = {
                    "view": view.name, "reference_id": reference.reference_id,
                    "tensor_hashes": dict(view.tensor_hashes),
                    "annotation_schema": {"origin_id_mapping": dict(view.origin_id_mapping), "origin_detail_id_mapping": dict(view.origin_detail_id_mapping)},
                }
                encoded_views.append({
                    "view": view.name, "star_sha256": hashlib.sha256(b"geoprobe.pre-status-rooted-star.payload.v1\x00" + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                    "tensor_hashes": dict(view.tensor_hashes), "tensor_keys": keyed,
                    "annotation_schema": payload["annotation_schema"],
                })
            stars.append({"reference": reference.to_dict(), "views": encoded_views})
        row = {
            "schema_version": 1, "kind": "relational_pre_status_rooted_star_materialization_row",
            "row_index": row_index, "conversation_id": conversation, "family": "family",
            "source_row_sha256": "e" * 64, "source_tensor_sha256": "f" * 64,
            "source_tensor_bytes": 17, "reference_ids": [star["reference"]["reference_id"] for star in stars],
            "stars": stars, "tensor_keys": sorted(row_keys),
        }
        row["star_row_sha256"] = _sha({
            "conversation_id": conversation, "source_tensor_sha256": "f" * 64,
            "stars": [{"reference_id": star["reference"]["reference_id"], "view_hashes": [view["star_sha256"] for view in star["views"]]} for star in stars],
        })
        rows.append(row)
    tensor_path = shard_dir / "family.safetensors"
    input_identity = "a" * 64
    save_file(tensors, tensor_path, metadata={
        "schema_version": "1", "kind": "relational_pre_status_rooted_star_materialization_family_shard",
        "family": "family", "input_identity_sha256": input_identity, "payload_scope": "outcome_blind_rooted_stars",
    })
    jsonl_path = shard_dir / "family.stars.jsonl"
    jsonl_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    entry = {
        "expected_row_count": 2, "completed_row_count": 2, "occurrence_count": 8, "finalized": True,
        "tensor_path": "shards/family.safetensors", "tensor_bytes": tensor_path.stat().st_size,
        "tensor_sha256": file_sha256(tensor_path), "tensor_key_count": len(tensors),
        "stars_jsonl_path": "shards/family.stars.jsonl", "stars_jsonl_bytes": jsonl_path.stat().st_size,
        "stars_jsonl_sha256": file_sha256(jsonl_path),
        "star_rows_sha256": _sha(sorted([{"conversation_id": row["conversation_id"], "sha256": row["star_row_sha256"]} for row in rows], key=lambda value: value["conversation_id"])),
    }
    manifest = {
        "schema_version": 1, "kind": "relational_pre_status_rooted_star_materialization_manifest", "status": "success",
        "input_identity_sha256": input_identity, "row_count": 2, "occurrence_count": 8, "view_count": 16,
        "families": {"family": entry},
        "source": {
            "state_graph_manifest_sha256": "1" * 64, "state_graph_geometry_sha256": "2" * 64,
            "state_graph_activation_sha256": "3" * 64, "capture_manifest_sha256": "4" * 64,
            "capture_inventory_sha256": "5" * 64, "action_exclusion_protocol_sha256": "6" * 64,
            "tensor_source": {"kind": "synthetic"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root


def test_store_loads_metric_only_payload_and_collapses_exact_geometry(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    index = build_relational_pre_status_rooted_star_index(root)
    assert len(index.references) == 16
    assert len(index.geometry_references) == 2
    assert {len(group) for group in index.exact_view_groups.values()} == {8}
    reference = index.references[0]
    metric = load_rooted_star_metric_input(index, reference)
    assert metric.residual_root_distances.shape[0] == 4
    assert metric.incoming_attention.shape[:2] == (4, 2)
    assert metric.typed_annotations.shape[1] == 7
    roots = load_rooted_star_root_residuals(index, reference)
    assert roots.shape == (4, 3)
    assert str(roots.dtype) == "torch.float32"
    observation = load_rooted_star_observation_binding(index, reference)
    assert observation.rooted_star_id == reference.rooted_star_id
    assert observation.geometry_sha256 == reference.geometry_sha256
    assert observation.prefix_token_count == reference.source_reference.prefix_token_stop
    assert observation.root_residuals.shape == (4, 3)
    assert observation.incoming_attention.shape[:2] == (4, 2)
    assert observation.retained_token_indices[-1] == observation.prefix_token_count - 1
    assert not observation.root_residuals.flags.writeable
    assert len(index.by_reference_id[reference.reference_id]) == 2
    assert len(index.by_field_event_id[reference.field_event_id]) == 2
    assert len(index.by_occurrence_id[reference.occurrence_id]) == 2


def test_local_annotation_codes_are_remapped_by_semantic_identity() -> None:
    first = _canonical_annotation_codes(
        torch.tensor([0, 1], dtype=torch.uint8),
        {"chat_source": 0, "environment_status_prefix": 1},
        {"chat_source": 0, "environment_status_prefix": 1, "environment_eot": 5},
    )
    second = _canonical_annotation_codes(
        torch.tensor([1, 2], dtype=torch.uint8),
        {"environment_eot": 0, "chat_source": 1, "environment_status_prefix": 2},
        {"chat_source": 0, "environment_status_prefix": 1, "environment_eot": 5},
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize("target", ("jsonl", "tensor", "tensor_hash", "outcome"))
def test_store_rejects_corrupt_or_unsealed_artifacts(tmp_path: Path, target: str) -> None:
    root = _artifact(tmp_path)
    if target == "jsonl":
        path = root / "shards" / "family.stars.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif target == "tensor":
        path = root / "shards" / "family.safetensors"
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif target == "tensor_hash":
        path = root / "shards" / "family.safetensors"
        tensors = load_file(path, device="cpu")
        key = next(name for name in tensors if name.endswith("root_residuals"))
        tensors[key] = tensors[key] + 1
        save_file(tensors, path, metadata={
            "schema_version": "1", "kind": "relational_pre_status_rooted_star_materialization_family_shard",
            "family": "family", "input_identity_sha256": "a" * 64, "payload_scope": "outcome_blind_rooted_stars",
        })
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["families"]["family"]["tensor_bytes"] = path.stat().st_size
        manifest["families"]["family"]["tensor_sha256"] = file_sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["outcome_label"] = "forbidden"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RootedStarStoreError):
        build_relational_pre_status_rooted_star_index(root)
