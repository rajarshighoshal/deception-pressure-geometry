"""External Control Bank row and activation validation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from geoprobe.data.jsonl import read_jsonl as _read_jsonl
from geoprobe.io import file_sha256


VALID_LABELS = {"honest", "deceptive", "ambiguous"}
VALID_CONTROL_TARGETS = {"honest", "deceptive", "abstain"}
DEFAULT_REQUIRED_SPLITS = {"train", "eval"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    for line_no, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object row")
    return rows


def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()
    aliases = {
        "0": "honest",
        "false": "honest",
        "nondeceptive": "honest",
        "non_deceptive": "honest",
        "truthful": "honest",
        "truth": "honest",
        "1": "deceptive",
        "true": "deceptive",
        "deception": "deceptive",
        "lie": "deceptive",
        "lying": "deceptive",
    }
    label = aliases.get(label, label)
    if label not in VALID_LABELS:
        raise ValueError(f"bad ECB label {value!r}; expected one of {sorted(VALID_LABELS)}")
    return label


def normalize_control_target(value: Any) -> str:
    target = "honest" if value is None else str(value).strip().lower()
    aliases = {
        "truthful": "honest",
        "truth": "honest",
        "nondeceptive": "honest",
        "non_deceptive": "honest",
        "none": "abstain",
    }
    target = aliases.get(target, target)
    if target not in VALID_CONTROL_TARGETS:
        raise ValueError(f"bad ECB control target {value!r}; expected one of {sorted(VALID_CONTROL_TARGETS)}")
    return target


def row_id(row: dict[str, Any]) -> str:
    cid = str(row.get("conversation_id") or row.get("id") or "").strip()
    if not cid:
        raise ValueError(f"row missing conversation_id/id: {row}")
    return cid


def has_prompt(row: dict[str, Any]) -> bool:
    prompt = row.get("prompt")
    messages = row.get("messages")
    return bool(prompt) or (isinstance(messages, list) and len(messages) > 0)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for raw in rows:
        cid = row_id(raw)
        dataset = str(raw.get("dataset") or raw.get("source_dataset") or "").strip()
        split = str(raw.get("split") or "").strip()
        if not dataset:
            raise ValueError(f"{cid}: missing dataset/source_dataset")
        if not split:
            raise ValueError(f"{cid}: missing split")
        label_source = next(
            (raw[key] for key in ("label", "ecb_label", "apollo_label", "deceptive") if raw.get(key) is not None),
            None,
        )
        if label_source is None:
            raise ValueError(f"{cid}: missing label/ecb_label/apollo_label/deceptive")
        label = normalize_label(label_source)
        out.append({
            **raw,
            "conversation_id": cid,
            "dataset": dataset,
            "family": str(raw.get("family") or dataset),
            "split": split,
            "label": label,
            "ecb_label": label,
            "control_target": normalize_control_target(raw.get("control_target")),
            "has_prompt": has_prompt(raw),
        })
    return out


def duplicate_ids(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(row["conversation_id"]) for row in rows)
    return sorted(cid for cid, count in counts.items() if count > 1)


def validate_rows(rows: list[dict[str, Any]], *, required_splits: set[str], allow_ambiguous: bool) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("no rows")
        return errors
    dupes = duplicate_ids(rows)
    if dupes:
        errors.append(f"duplicate conversation_id values: {dupes[:5]}")
    labels = Counter(str(row["ecb_label"]) for row in rows)
    if labels.get("ambiguous", 0) and not allow_ambiguous:
        errors.append("ambiguous labels present but --allow-ambiguous was not set")
    splits = {str(row["split"]) for row in rows}
    missing_splits = sorted(required_splits - splits)
    if missing_splits:
        errors.append(f"missing required splits: {missing_splits}")
    if labels.get("honest", 0) == 0 or labels.get("deceptive", 0) == 0:
        errors.append("need at least one honest and one deceptive row")
    missing_prompt = [str(row["conversation_id"]) for row in rows if not bool(row["has_prompt"])]
    if missing_prompt:
        errors.append(f"{len(missing_prompt)} rows missing prompt/messages, e.g. {missing_prompt[:5]}")
    targets = Counter(str(row["control_target"]) for row in rows)
    if targets.get("honest", 0) == 0:
        errors.append("need at least one row with control_target=honest")
    return errors


def load_activation_bank(path: Path) -> dict[str, Any]:
    bank = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(bank, dict):
        raise ValueError(f"{path}: expected dict activation bank")
    if "conversation_id" not in bank:
        raise ValueError(f"{path}: missing conversation_id")
    if "activations" not in bank or not isinstance(bank["activations"], dict):
        raise ValueError(f"{path}: missing activations dict")
    return bank


def activation_summary(path: Path, row_ids: set[str], *, allow_extra_activations: bool) -> tuple[dict[str, Any], list[str]]:
    bank = load_activation_bank(path)
    ids = [str(cid) for cid in bank["conversation_id"]]
    activation_ids = set(ids)
    missing = sorted(row_ids - activation_ids)
    extra = sorted(activation_ids - row_ids)
    errors = []
    if missing:
        errors.append(f"activation bank missing {len(missing)} row ids, e.g. {missing[:5]}")
    if extra and not allow_extra_activations:
        errors.append(f"activation bank has {len(extra)} extra ids, e.g. {extra[:5]}")
    layer_shapes = {}
    for layer, tensor in bank["activations"].items():
        if int(tensor.shape[0]) != len(ids):
            errors.append(f"layer {layer} has {int(tensor.shape[0])} rows but conversation_id has {len(ids)}")
        layer_shapes[str(layer)] = list(tensor.shape)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "activation_rows": len(ids),
        "unique_activation_ids": len(activation_ids),
        "activation_layers": sorted(layer_shapes, key=lambda item: int(item) if item.lstrip("-").isdigit() else item),
        "layer_shapes": layer_shapes,
        "phase_values": dict(Counter(map(str, bank.get("phase", [])))) if "phase" in bank else {},
        "turn_index_values": dict(Counter(map(str, bank.get("turn_index", [])))) if "turn_index" in bank else {},
        "extra_activation_ids": extra[:50],
        "missing_activation_ids": missing[:50],
    }, errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_conversation_ids": len({str(row["conversation_id"]) for row in rows}),
        "labels": dict(Counter(str(row["ecb_label"]) for row in rows)),
        "control_targets": dict(Counter(str(row["control_target"]) for row in rows)),
        "splits": dict(Counter(str(row["split"]) for row in rows)),
        "datasets": dict(Counter(str(row["dataset"]) for row in rows)),
        "families": dict(Counter(str(row["family"]) for row in rows)),
        "rows_with_prompt": int(sum(bool(row["has_prompt"]) for row in rows)),
        "by_dataset_label": {
            dataset: dict(Counter(str(row["ecb_label"]) for row in rows if str(row["dataset"]) == dataset))
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
    }


def control_structure_contract(*, source_name: str = "external dataset") -> dict[str, Any]:
    return {
        "contract": "External Control Bank (ECB)",
        "source": source_name,
        "base_space": "ECB prompt/activation rows grouped by dataset/family/split",
        "implemented": {
            "row_activation_alignment": "labels/control_target/splits/prompts aligned by conversation_id",
            "default_control_target": "honest",
        },
        "fiber": {
            "status": "planned",
            "description": "candidate steering/control actions attached to each ECB state",
        },
        "control_target_routing": {
            "status": "planned",
            "description": "generation adapters may not consume control_target until explicitly wired",
        },
        "symmetry": {
            "status": "planned_if_fitted",
            "description": "honest/deceptive partner relation if both directions are fitted",
        },
        "required_before_result": [
            "explicit labels",
            "prompt/messages present",
            "train/eval split present",
            "activation IDs aligned with row IDs",
        ],
    }


def validation_notes(*, source_name: str = "external dataset") -> list[str]:
    return [
        f"This validates {source_name} rows against the ECB contract; it is not a detector/control result.",
        "Labels, control targets, prompts, and row/activation conversation IDs must align before control claims.",
    ]


__all__ = [
    "DEFAULT_REQUIRED_SPLITS",
    "VALID_CONTROL_TARGETS",
    "VALID_LABELS",
    "activation_summary",
    "control_structure_contract",
    "duplicate_ids",
    "file_sha256",
    "has_prompt",
    "load_activation_bank",
    "normalize_control_target",
    "normalize_label",
    "normalize_rows",
    "read_jsonl",
    "row_id",
    "summarize_rows",
    "validate_rows",
    "validation_notes",
]
