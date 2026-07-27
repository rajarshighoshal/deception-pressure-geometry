from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoprobe.control.residual_transfer_prep import label_of, row_id
from geoprobe.models import ResidualSteeringSpec


EXTERNAL_DIRECTION_KIND = "external_soft_control_direction"
EXTERNAL_DIRECTION_BUNDLE_KIND = "external_soft_control_direction_bundle"
EXTERNAL_ROW_DIRECTION_POLICY_KIND = "external_soft_control_row_direction_policy"
LEGACY_APOLLO_DIRECTION_KIND = "apollo_soft_control_direction"
LEGACY_APOLLO_DIRECTION_BUNDLE_KIND = "apollo_soft_control_direction_bundle"


@dataclass(frozen=True)
class DirectionArtifact:
    name: str
    vector: np.ndarray
    layer: int
    turn: int
    phase: str
    source: str
    n_honest: int
    n_deceptive: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RowDirectionPolicy:
    name: str
    vectors: dict[str, np.ndarray]
    layer: int
    turn: int
    phase: str
    source: str
    metadata: dict[str, Any]
    alphas: dict[str, float] | None = None


def arr(values: Any) -> np.ndarray:
    return values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize zero/non-finite direction")
    return (vec / norm).astype(np.float32)


def direction_summary(vector: np.ndarray) -> dict[str, Any]:
    vec = np.asarray(vector, dtype=np.float32)
    return {
        "dim": int(vec.shape[0]),
        "norm": float(np.linalg.norm(vec)),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
    }


def artifact_payload(artifact: DirectionArtifact) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": EXTERNAL_DIRECTION_KIND,
        "name": artifact.name,
        "layer": int(artifact.layer),
        "turn": int(artifact.turn),
        "phase": artifact.phase,
        "source": artifact.source,
        "n_honest": int(artifact.n_honest),
        "n_deceptive": int(artifact.n_deceptive),
        "summary": direction_summary(artifact.vector),
        "metadata": artifact.metadata,
    }


def save_direction_artifacts(path: Path, artifacts: list[DirectionArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": EXTERNAL_DIRECTION_BUNDLE_KIND,
        "legacy_kind": LEGACY_APOLLO_DIRECTION_BUNDLE_KIND,
        "directions": {artifact.name: torch.from_numpy(artifact.vector.astype(np.float32)) for artifact in artifacts},
        "metadata": {artifact.name: artifact_payload(artifact) for artifact in artifacts},
    }
    torch.save(payload, path)


def save_row_direction_policy(path: Path, policy: RowDirectionPolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": EXTERNAL_ROW_DIRECTION_POLICY_KIND,
        "name": policy.name,
        "directions": {
            cid: torch.from_numpy(unit_vector(vector).astype(np.float32))
            for cid, vector in sorted(policy.vectors.items())
        },
        "alphas": dict(sorted((policy.alphas or {}).items())),
        "metadata": {
            "schema_version": 1,
            "kind": EXTERNAL_ROW_DIRECTION_POLICY_KIND,
            "name": policy.name,
            "layer": int(policy.layer),
            "turn": int(policy.turn),
            "phase": policy.phase,
            "source": policy.source,
            **policy.metadata,
        },
    }
    torch.save(payload, path)


def load_direction_artifact(path: Path, name: str) -> DirectionArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("kind") not in {
        EXTERNAL_DIRECTION_BUNDLE_KIND,
        LEGACY_APOLLO_DIRECTION_BUNDLE_KIND,
    }:
        raise ValueError(f"{path}: expected external_soft_control_direction_bundle")
    directions = payload.get("directions")
    metadata = payload.get("metadata")
    if not isinstance(directions, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing directions/metadata")
    if name not in directions or name not in metadata:
        raise ValueError(f"{path}: missing direction {name!r}; available={sorted(directions)}")
    meta = metadata[name]
    vector = arr(directions[name]).astype(np.float32)
    return DirectionArtifact(
        name=str(meta["name"]),
        vector=unit_vector(vector),
        layer=int(meta["layer"]),
        turn=int(meta["turn"]),
        phase=str(meta["phase"]),
        source=str(meta["source"]),
        n_honest=int(meta["n_honest"]),
        n_deceptive=int(meta["n_deceptive"]),
        metadata=dict(meta.get("metadata") or {}),
    )


def load_row_direction_policy(path: Path) -> RowDirectionPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("kind") != EXTERNAL_ROW_DIRECTION_POLICY_KIND:
        raise ValueError(f"{path}: expected {EXTERNAL_ROW_DIRECTION_POLICY_KIND}")
    directions = payload.get("directions")
    metadata = payload.get("metadata")
    if not isinstance(directions, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing directions/metadata")
    vectors = {str(cid): unit_vector(arr(vector).astype(np.float32)) for cid, vector in directions.items()}
    raw_alphas = payload.get("alphas") or {}
    alphas = {str(cid): float(alpha) for cid, alpha in raw_alphas.items()} if isinstance(raw_alphas, dict) else {}
    return RowDirectionPolicy(
        name=str(metadata["name"]),
        vectors=vectors,
        layer=int(metadata["layer"]),
        turn=int(metadata["turn"]),
        phase=str(metadata["phase"]),
        source=str(metadata["source"]),
        metadata=dict(metadata),
        alphas=alphas,
    )


def direction_artifact_for_row(row: dict[str, Any], policy: RowDirectionPolicy) -> DirectionArtifact | None:
    cid = row_id(row)
    vector = policy.vectors.get(cid)
    if vector is None:
        return None
    return DirectionArtifact(
        name=policy.name,
        vector=vector,
        layer=policy.layer,
        turn=policy.turn,
        phase=policy.phase,
        source=policy.source,
        n_honest=int(policy.metadata.get("n_honest", 0) or 0),
        n_deceptive=int(policy.metadata.get("n_deceptive", 0) or 0),
        metadata={**policy.metadata, "row_direction": True},
    )


def fit_ecb_honesty_direction(
    bank: dict[str, Any],
    *,
    name: str,
    layer: int,
    turn: int,
    phase: str,
    train_split: str = "train",
    source_name: str = "ecb",
) -> DirectionArtifact:
    cids = np.asarray(bank["conversation_id"]).astype(str)
    turns = arr(bank["turn_index"]).astype(int)
    phases = np.asarray(bank["phase"]).astype(str)
    splits = np.asarray(bank["split"]).astype(str)
    deceptive = arr(bank["deceptive"]).astype(int)
    x = bank["activations"][int(layer)].detach().float().cpu().numpy()

    mask = (turns == int(turn)) & (phases == phase) & (splits == train_split)
    idxs = np.where(mask)[0]
    if not len(idxs):
        raise ValueError("no ECB rows selected for direction fit")
    honest = idxs[deceptive[idxs] == 0]
    deceptive_idxs = idxs[deceptive[idxs] == 1]
    if not len(honest) or not len(deceptive_idxs):
        raise ValueError("ECB direction fit requires both honest and deceptive train rows")
    direction = unit_vector(x[honest].mean(axis=0) - x[deceptive_idxs].mean(axis=0))
    return DirectionArtifact(
        name=name,
        vector=direction,
        layer=int(layer),
        turn=int(turn),
        phase=phase,
        source=f"{source_name}_train_mean_honest_minus_deceptive",
        n_honest=int(len(honest)),
        n_deceptive=int(len(deceptive_idxs)),
        metadata={
            "source_name": source_name,
            "train_split": train_split,
            "fit_split": train_split,
            "split_provenance": "fit on ECB rows with split == train_split",
            "selected_conversation_ids": int(len(set(cids[idx] for idx in idxs))),
        },
    )


def fit_powered150_honesty_direction(
    bank: dict[str, Any],
    transcripts: dict[str, dict[str, Any]],
    *,
    name: str,
    layer: int,
    turn: int,
    phase: str,
    levels: set[str],
) -> DirectionArtifact:
    cids = np.asarray(bank["conversation_id"]).astype(str)
    turns = arr(bank["turn_index"]).astype(int)
    phases = np.asarray(bank["phase"]).astype(str)
    arms = np.asarray(bank["arm"]).astype(str)
    x = bank["activations"][int(layer)].detach().float().cpu().numpy()

    honest: list[int] = []
    deceptive: list[int] = []
    for idx, cid in enumerate(cids):
        if turns[idx] != int(turn) or phases[idx] != phase or arms[idx] not in levels:
            continue
        row = transcripts.get(str(cid))
        if not row or not row.get("valid_outcome"):
            continue
        label = row.get("deceptive")
        if label is None:
            continue
        if bool(label):
            deceptive.append(idx)
        else:
            honest.append(idx)
    if not honest or not deceptive:
        raise ValueError("powered150 direction fit requires both honest and deceptive rows")
    direction = unit_vector(x[honest].mean(axis=0) - x[deceptive].mean(axis=0))
    return DirectionArtifact(
        name=name,
        vector=direction,
        layer=int(layer),
        turn=int(turn),
        phase=phase,
        source="powered150_mean_honest_minus_deceptive",
        n_honest=int(len(honest)),
        n_deceptive=int(len(deceptive)),
        metadata={
            "levels": sorted(levels),
            "fit_split": "train+eval/all",
            "split_provenance": "powered150 mean direction fit from all valid selected levels, not split-held-out",
        },
    )


def select_generation_rows(
    rows: list[dict[str, Any]],
    *,
    split: str | None,
    labels: set[str] | None,
    families: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and str(row.get("split", "")) != split:
            continue
        if labels is not None and label_of(row) not in labels:
            continue
        if families is not None and str(row.get("family", "")) not in families:
            continue
        selected.append(row)
    selected = sorted(selected, key=row_id)
    return selected[:limit] if limit is not None else selected


def should_steer_row(row: dict[str, Any], *, routing: str) -> bool:
    if routing == "all":
        return True
    if routing == "oracle_deceptive_only":
        return label_of(row) == "deceptive"
    if routing == "none":
        return False
    raise ValueError(f"unknown routing {routing!r}")


def steering_spec_for_row(
    row: dict[str, Any],
    *,
    direction: DirectionArtifact,
    alpha: float,
    routing: str,
) -> ResidualSteeringSpec | None:
    if not should_steer_row(row, routing=routing):
        return None
    return ResidualSteeringSpec(
        layer=int(direction.layer),
        direction=torch.from_numpy(direction.vector.astype(np.float32)),
        alpha=float(alpha),
    )


def generation_record(
    row: dict[str, Any],
    *,
    response: str,
    policy_name: str,
    steering_applied: bool,
    direction: DirectionArtifact | None,
    alpha: float | None,
    routing: str,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "external_soft_control_generation",
        "conversation_id": row_id(row),
        "policy_name": policy_name,
        "family": str(row.get("family", "")),
        "dataset": str(row.get("dataset", "")),
        "split": str(row.get("split", "")),
        "label": label_of(row),
        "subset_role": str(row.get("subset_role", "")),
        "source_file": str(row.get("source_file", "")),
        "source_index": int(row.get("source_index", -1)),
        "routing": routing,
        "steering_applied": bool(steering_applied),
        "direction_name": direction.name if direction else None,
        "direction_source": direction.source if direction else None,
        "layer": int(direction.layer) if direction else None,
        "alpha": float(alpha) if alpha is not None else None,
        "seed": seed,
        "generated_response": response,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    tmp.replace(path)


__all__ = [
    "DirectionArtifact",
    "EXTERNAL_ROW_DIRECTION_POLICY_KIND",
    "EXTERNAL_DIRECTION_BUNDLE_KIND",
    "EXTERNAL_DIRECTION_KIND",
    "LEGACY_APOLLO_DIRECTION_BUNDLE_KIND",
    "LEGACY_APOLLO_DIRECTION_KIND",
    "RowDirectionPolicy",
    "arr",
    "artifact_payload",
    "direction_artifact_for_row",
    "direction_summary",
    "fit_ecb_honesty_direction",
    "fit_powered150_honesty_direction",
    "generation_record",
    "load_direction_artifact",
    "load_row_direction_policy",
    "save_direction_artifacts",
    "save_row_direction_policy",
    "select_generation_rows",
    "should_steer_row",
    "steering_spec_for_row",
    "unit_vector",
    "write_json",
]
