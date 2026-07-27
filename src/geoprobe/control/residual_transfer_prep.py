from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from geoprobe.io import file_sha256  # re-exported; canonical impl moved to geoprobe.io


def stable_hash(value: Any, *, seed: int = 0) -> str:
    payload = {"seed": seed, "value": value}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def label_of(row: dict[str, Any]) -> str:
    value = next(
        (row[key] for key in ("ecb_label", "apollo_label", "label", "deceptive") if row.get(key) is not None),
        "",
    )
    if isinstance(value, bool):
        return "deceptive" if value else "honest"
    return str(value).strip().lower()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("conversation_id") or row.get("id") or "").strip()


def count_by(rows: list[dict[str, Any]], *keys: str) -> dict[str, int]:
    counts = Counter(
        "|".join(str(row.get(key, "")) for key in keys)
        for row in rows
    )
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class ApolloSubsetConfig:
    seed: int = 0
    primary_keys: tuple[str, ...] = ("dataset", "split")
    fallback_keys: tuple[str, ...] = ("family", "split")
    exclude_ambiguous: bool = True
    exclude_alpaca_as_match: bool = True


def build_apollo_balanced_subset(
    rows: list[dict[str, Any]],
    config: ApolloSubsetConfig = ApolloSubsetConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deceptive = [row for row in rows if label_of(row) == "deceptive"]
    ambiguous = [row for row in rows if label_of(row) == "ambiguous"]
    honest = [row for row in rows if label_of(row) == "honest"]
    if config.exclude_alpaca_as_match:
        honest_pool = [row for row in honest if str(row.get("family")) != "alpaca"]
    else:
        honest_pool = honest

    selected: dict[str, dict[str, Any]] = {}
    group_reports: list[dict[str, Any]] = []
    honest_by_primary = group_rows(honest_pool, config.primary_keys)
    honest_by_fallback = group_rows(honest_pool, config.fallback_keys)
    used_honest: set[str] = set()

    for row in deceptive:
        rid = row_id(row)
        selected[rid] = {**row, "subset_role": "deceptive_all"}

    deceptive_by_primary = group_rows(deceptive, config.primary_keys)
    for primary_key, group_deceptive in sorted(deceptive_by_primary.items()):
        needed = len(group_deceptive)
        primary_candidates = [
            row for row in honest_by_primary.get(primary_key, [])
            if row_id(row) not in used_honest
        ]
        chosen = deterministic_sample(primary_candidates, needed, seed=config.seed)
        for row in chosen:
            used_honest.add(row_id(row))
            selected[row_id(row)] = {
                **row,
                "subset_role": "matched_honest_primary",
                "subset_match_key": key_to_string(config.primary_keys, primary_key),
            }

        remaining = needed - len(chosen)
        fallback_chosen: list[dict[str, Any]] = []
        fallback_key = project_key(group_deceptive[0], config.fallback_keys)
        if remaining > 0:
            fallback_candidates = [
                row for row in honest_by_fallback.get(fallback_key, [])
                if row_id(row) not in used_honest
            ]
            fallback_chosen = deterministic_sample(
                fallback_candidates,
                remaining,
                seed=config.seed + 1,
            )
            for row in fallback_chosen:
                used_honest.add(row_id(row))
                selected[row_id(row)] = {
                    **row,
                    "subset_role": "matched_honest_family_fallback",
                    "subset_match_key": key_to_string(config.fallback_keys, fallback_key),
                }
        group_reports.append({
            "primary_key": key_to_string(config.primary_keys, primary_key),
            "fallback_key": key_to_string(config.fallback_keys, fallback_key),
            "deceptive": needed,
            "matched_honest_primary": len(chosen),
            "matched_honest_family_fallback": len(fallback_chosen),
            "honest_deficit": max(0, needed - len(chosen) - len(fallback_chosen)),
        })

    subset = sorted(selected.values(), key=lambda row: row_id(row))
    summary = {
        "schema_version": 1,
        "selection": {
            "deceptive_policy": "include_all_non_ambiguous_deceptive_rows",
            "honest_policy": (
                "match honest rows within dataset+split first, then family+split; "
                "never use alpaca as a non-alpaca negative by default"
            ),
            "seed": config.seed,
            "primary_keys": list(config.primary_keys),
            "fallback_keys": list(config.fallback_keys),
            "exclude_ambiguous": config.exclude_ambiguous,
            "exclude_alpaca_as_match": config.exclude_alpaca_as_match,
        },
        "source": summarize_subset_rows(rows),
        "subset": summarize_subset_rows(subset),
        "excluded": {
            "ambiguous": len(ambiguous) if config.exclude_ambiguous else 0,
            "alpaca_honest_not_used_as_match": (
                sum(1 for row in honest if str(row.get("family")) == "alpaca")
                if config.exclude_alpaca_as_match else 0
            ),
        },
        "group_reports": group_reports,
        "total_honest_deficit": int(sum(item["honest_deficit"] for item in group_reports)),
        "notes": [
            "This is a capture/evaluation subset manifest, not a detection/control result.",
            "Rows are selected only from labels and pre-existing dataset/family/split metadata.",
            "Deficits are explicit instead of filled from alpaca, avoiding an alpaca-vs-deception shortcut.",
        ],
    }
    return subset, summary


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[project_key(row, keys)].append(row)
    return grouped


def project_key(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in keys)


def key_to_string(keys: tuple[str, ...], values: tuple[str, ...]) -> str:
    return ",".join(f"{key}={value}" for key, value in zip(keys, values, strict=True))


def deterministic_sample(rows: list[dict[str, Any]], n: int, *, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_hash(row_id(row), seed=seed))
    return ordered[: max(0, min(n, len(ordered)))]


def summarize_subset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_conversation_ids": len({row_id(row) for row in rows}),
        "labels": count_by_label(rows),
        "splits": count_by(rows, "split"),
        "families": count_by(rows, "family"),
        "datasets": count_by(rows, "dataset"),
        "by_family_label": nested_counts(rows, "family", "label"),
        "by_dataset_label": nested_counts(rows, "dataset", "label"),
        "by_split_label": nested_counts(rows, "split", "label"),
        "subset_roles": count_by(rows, "subset_role") if rows and "subset_role" in rows[0] else {},
    }


def count_by_label(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(label_of(row) for row in rows).items()))


def nested_counts(rows: list[dict[str, Any]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    values = sorted({str(row.get(outer, "")) for row in rows})
    out = {}
    for value in values:
        items = [
            label_of(row) if inner in {"label", "apollo_label"} else str(row.get(inner, ""))
            for row in rows
            if str(row.get(outer, "")) == value
        ]
        out[value] = dict(sorted(Counter(items).items()))
    return out


def rollout_turn_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["conversation_id"]), int(row["turn_index"])


def judged_turn_labels(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    judged = data.get("judged", [])
    turn_counts = Counter()
    stance_counts = Counter()
    ids = []
    for item in judged:
        cid = str(item.get("conversation_id"))
        ids.append(cid)
        turns = item.get("turns") or []
        turn_counts[len(turns)] += 1
        for turn in turns:
            stance_counts[str(turn.get("stance"))] += 1
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "judge_backend": data.get("judge_backend"),
        "judge_model": data.get("judge_model"),
        "judge_variant": data.get("judge_variant"),
        "n_conversations": len(ids),
        "unique_conversation_ids": len(set(ids)),
        "turn_count_distribution": dict(sorted(turn_counts.items())),
        "stance_counts": dict(sorted(stance_counts.items())),
        "conversation_ids": sorted(set(ids)),
    }


def assistant_message_indices(messages: list[dict[str, Any]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def final_assistant_capture_messages(row: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{row_id(row)}: missing non-empty messages list")
    assistant_indices = assistant_message_indices(messages)
    if not assistant_indices:
        raise ValueError(f"{row_id(row)}: no assistant messages to capture")
    index = assistant_indices[-1]
    pre_messages = messages[:index]
    post_messages = messages[: index + 1]
    if not pre_messages:
        raise ValueError(f"{row_id(row)}: final assistant has empty prefix")
    return index, pre_messages, post_messages


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


def activation_bank_inventory(path: Path, *, check_finite: bool = True) -> dict[str, Any]:
    bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, dict):
        raise ValueError(f"{path}: expected dict activation bank")
    cids = [str(item) for item in bank.get("conversation_id", [])]
    phases = [str(item) for item in bank.get("phase", [])]
    turns = [int(item) for item in bank.get("turn_index", [])]
    if not cids:
        raise ValueError(f"{path}: missing conversation_id")
    if not phases:
        raise ValueError(f"{path}: missing phase")
    if not turns:
        raise ValueError(f"{path}: missing turn_index")
    if not (len(cids) == len(phases) == len(turns)):
        raise ValueError(
            f"{path}: conversation_id/phase/turn_index length mismatch "
            f"{len(cids)}/{len(phases)}/{len(turns)}"
        )
    activations = bank.get("activations")
    if not isinstance(activations, dict) or not activations:
        raise ValueError(f"{path}: missing activations dict")

    row_keys = [(cid, turn, phase) for cid, turn, phase in zip(cids, turns, phases, strict=True)]
    turn_to_phases: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cid, turn, phase in row_keys:
        turn_to_phases[(cid, turn)].add(phase)

    layer_shapes: dict[str, list[int]] = {}
    layer_dtypes: dict[str, str] = {}
    layer_finite: dict[str, bool] = {}
    for layer, tensor in sorted(activations.items(), key=lambda item: int(item[0])):
        layer_name = str(layer)
        layer_shapes[layer_name] = [int(dim) for dim in tensor.shape]
        layer_dtypes[layer_name] = str(tensor.dtype)
        layer_finite[layer_name] = bool(torch.isfinite(tensor).all().item()) if check_finite else True

    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "backend": bank.get("backend"),
        "config_name": bank.get("config_name"),
        "model_key": bank.get("model_key"),
        "model_name": bank.get("model_name"),
        "device": bank.get("device"),
        "source_run": bank.get("source_run"),
        "source_shards": bank.get("source_shards"),
        "activation_rows": len(cids),
        "unique_conversation_ids": len(set(cids)),
        "unique_turn_keys": len(turn_to_phases),
        "phase_counts": dict(sorted(Counter(phases).items())),
        "turn_phase_sets": dict(sorted(Counter(",".join(sorted(value)) for value in turn_to_phases.values()).items())),
        "duplicate_row_keys": duplicate_items(row_keys)[:50],
        "layers": sorted(layer_shapes, key=lambda value: int(value)),
        "layer_shapes": layer_shapes,
        "layer_dtypes": layer_dtypes,
        "layer_finite": layer_finite,
        "all_layers_finite": all(layer_finite.values()),
        "turn_keys": sorted([list(key) for key in turn_to_phases]),
    }


def duplicate_items(items: list[Any]) -> list[Any]:
    counts = Counter(items)
    return sorted([item for item, count in counts.items() if count > 1])


def build_70b_residual_inventory(
    *,
    rollout_path: Path,
    activation_path: Path,
    audit_path: Path | None = None,
    judged_paths: list[Path] | None = None,
    shard_paths: list[Path] | None = None,
    check_finite: bool = True,
) -> dict[str, Any]:
    rollout_rows = load_jsonl(rollout_path)
    rollout_key_rows = [rollout_turn_key(row) for row in rollout_rows]
    rollout_keys = set(rollout_key_rows)
    rollout_cids = {str(row["conversation_id"]) for row in rollout_rows}
    activation = activation_bank_inventory(activation_path, check_finite=check_finite)
    activation_turn_keys = {
        (str(cid), int(turn))
        for cid, turn in ((item[0], item[1]) for item in activation["turn_keys"])
    }

    audit_rows: list[dict[str, Any]] = []
    if audit_path is not None and audit_path.exists():
        audit_rows = load_jsonl(audit_path)

    judged = [judged_turn_labels(path) for path in judged_paths or [] if path.exists()]
    shard_summaries = [
        activation_bank_inventory(path, check_finite=False)
        for path in shard_paths or []
        if path.exists()
    ]
    audit_cids = {str(row.get("conversation_id")) for row in audit_rows}
    judged_cids = set().union(*(set(item["conversation_ids"]) for item in judged)) if judged else set()
    errors = []
    missing_activation_turns = sorted(rollout_keys - activation_turn_keys)
    extra_activation_turns = sorted(activation_turn_keys - rollout_keys)
    if missing_activation_turns:
        errors.append(f"activation bank missing {len(missing_activation_turns)} rollout turn keys")
    if extra_activation_turns:
        errors.append(f"activation bank has {len(extra_activation_turns)} extra turn keys")
    if activation["duplicate_row_keys"]:
        errors.append(f"activation bank has duplicate row keys: {activation['duplicate_row_keys'][:5]}")
    if set(activation["turn_phase_sets"]) != {"post_response,pre_response"}:
        errors.append(f"unexpected turn phase sets: {activation['turn_phase_sets']}")
    if not activation["all_layers_finite"]:
        errors.append("activation bank has non-finite values")

    return {
        "schema_version": 1,
        "task": "70b_residual_inventory",
        "status": "failed" if errors else "ok",
        "paths": {
            "rollout": str(rollout_path),
            "activations": str(activation_path),
            "audit": str(audit_path) if audit_path else None,
            "judged": [str(path) for path in judged_paths or []],
            "shards": [str(path) for path in shard_paths or []],
        },
        "sha256": {
            "rollout": file_sha256(rollout_path),
            "activations": file_sha256(activation_path),
            "audit": file_sha256(audit_path) if audit_path and audit_path.exists() else None,
            "judged": {str(path): file_sha256(path) for path in judged_paths or [] if path.exists()},
            "shards": {str(path): file_sha256(path) for path in shard_paths or [] if path.exists()},
        },
        "rollout": {
            "n_rows": len(rollout_rows),
            "unique_conversation_ids": len(rollout_cids),
            "turn_keys": len(rollout_keys),
            "conditions": count_by(rollout_rows, "condition"),
            "domains": count_by(rollout_rows, "domain"),
            "scenarios": count_by(rollout_rows, "scenario"),
            "duplicate_turn_keys": duplicate_items(rollout_key_rows)[:50],
        },
        "audit": {
            "n_rows": len(audit_rows),
            "unique_conversation_ids": len(audit_cids),
            "missing_from_rollout": sorted(audit_cids - rollout_cids)[:50],
        },
        "judged": judged,
        "judged_alignment": {
            "unique_judged_conversation_ids": len(judged_cids),
            "judged_missing_from_rollout": sorted(judged_cids - rollout_cids)[:50],
            "rollout_missing_from_judged": sorted(rollout_cids - judged_cids)[:50],
        },
        "activation_bank": {k: v for k, v in activation.items() if k != "turn_keys"},
        "turn_alignment": {
            "missing_activation_turns": [list(item) for item in missing_activation_turns[:50]],
            "extra_activation_turns": [list(item) for item in extra_activation_turns[:50]],
        },
        "shards": [
            {k: v for k, v in summary.items() if k != "turn_keys"}
            for summary in shard_summaries
        ],
        "interpretation": {
            "current_framing": (
                "This existing 70B bank is sycophancy/pressure data, not the direct powered150 deception-control bank. "
                "It is still hypothesis-relevant if sycophancy and deception share an internal honesty/misalignment geometry, "
                "but direct 70B deception transfer needs a separate powered150-on-70B capture/evaluation."
            ),
            "recommended_next_use": (
                "Use this artifact for CPU-only cross-scale honesty-transfer diagnostics first; plan a separate same-powered150 "
                "70B residual capture for the direct deception-transfer test. No 70B control/generation yet."
            ),
        },
        "validation_errors": errors,
    }
