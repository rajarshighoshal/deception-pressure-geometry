"""Immutable family-fold shards for the frozen pre-status outcome report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any

from geoprobe.eval.relational_outcome_events import (
    OUTCOME_CLASSES,
    outcome_class_from_scientific_cohort,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FOLDS
from geoprobe.io import file_sha256


SCHEMA_VERSION = 1
SHARD_KIND = "relational_pre_status_outcome_family_fold_shard"
MANIFEST_KIND = "relational_pre_status_outcome_family_fold_manifest"
MANIFEST_NAME = "manifest.json"
SHARD_SUBDIR = "shards"
SOURCE_REPORT_KIND = "relational_post_commitment_growth_outcome_score"


class RelationalPreStatusOutcomeShardError(ValueError):
    """Raised when an outcome shard violates its immutable split contract."""


@dataclass(frozen=True, slots=True)
class LoadedRelationalPreStatusOutcomeShard:
    """One validated fold shard loaded without touching other fold files."""

    family_fold: str
    scored_events: tuple[Mapping[str, Any], ...]
    source_report_file_sha256: str
    source_report_internal_sha256: str
    manifest_file_sha256: str
    manifest_sha256: str
    shard_file_sha256: str
    content_sha256: str
    shard_sha256: str


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusOutcomeShardError(
            "value is not canonical JSON"
        ) from error


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256(_canonical(payload)).hexdigest()


def _content_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256(_canonical(list(rows))).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusOutcomeShardError(f"{label} must be an object")
    return value


def _rows(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise RelationalPreStatusOutcomeShardError(
            f"{label} must be a non-empty array"
        )
    return tuple(_mapping(item, f"{label} item") for item in value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPreStatusOutcomeShardError(
            f"{label} must be a non-empty string"
        )
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RelationalPreStatusOutcomeShardError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return text


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RelationalPreStatusOutcomeShardError(
            f"{label} must be a non-negative integer"
        )
    return value


def _history(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RelationalPreStatusOutcomeShardError(
            f"{label} must be an array of non-empty strings"
        )
    return tuple(value)


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalPreStatusOutcomeShardError(
            f"{label} is not finite UTF-8 JSON"
        ) from error
    return _mapping(value, label)


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete file atomically without any overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        except FileExistsError as error:
            raise RelationalPreStatusOutcomeShardError(
                f"refusing to overwrite existing destination: {path}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
    ):
        raise RelationalPreStatusOutcomeShardError(
            f"{label} is not a safe relative POSIX path"
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RelationalPreStatusOutcomeShardError(
            f"{label} escapes the artifact root"
        )
    return path


def _shard_relative_path(fold: str) -> str:
    return f"{SHARD_SUBDIR}/{fold}.json"


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
    *,
    expected_fold: str | None = None,
) -> None:
    event_ids: set[str] = set()
    for row in rows:
        event_id = _string(row.get("field_event_id"), f"{label} field-event ID")
        if event_id in event_ids:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} has duplicate field events"
            )
        event_ids.add(event_id)

        fold = _string(row.get("family_fold"), f"{label} family fold")
        if fold not in FOLDS:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} has an unsupported family fold"
            )
        if expected_fold is not None and fold != expected_fold:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} crosses its family-fold shard"
            )

        outcome_class = _string(
            row.get("outcome_class"), f"{label} outcome class"
        )
        if outcome_class not in OUTCOME_CLASSES:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} has an unsupported outcome class"
            )
        try:
            cohort_class = outcome_class_from_scientific_cohort(
                row.get("scientific_cohort")
            )
        except ValueError as error:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} has an unsupported scientific cohort"
            ) from error
        if cohort_class != outcome_class:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} outcome class disagrees with scientific cohort"
            )

        knowledge = row.get("knowledge_correct")
        pressure = row.get("pressure_exposed")
        if not isinstance(knowledge, bool) or not isinstance(pressure, bool):
            raise RelationalPreStatusOutcomeShardError(
                f"{label} knowledge and pressure flags must be Boolean"
            )
        true_status = _string(
            row.get("true_status"), f"{label} true status"
        )
        desired_status = _string(
            row.get("desired_status"), f"{label} desired status"
        )
        if true_status not in {"PASS", "FAIL"} or desired_status not in {
            "PASS",
            "FAIL",
        }:
            raise RelationalPreStatusOutcomeShardError(
                f"{label} has invalid status labels"
            )

        _string(row.get("mapped_action"), f"{label} mapped action")
        _string(row.get("family"), f"{label} family")
        _string(row.get("scenario_id"), f"{label} scenario ID")
        _string(row.get("orbit_id"), f"{label} orbit ID")
        _integer(row.get("turn_index"), f"{label} turn index")
        _history(
            row.get("intervention_history"),
            f"{label} intervention history",
        )
        _sha(
            row.get("prefix_state_sha256"),
            f"{label} prefix-state SHA-256",
        )


def _validate_source_report(
    report: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("kind") != SOURCE_REPORT_KIND
        or report.get("status") != "success"
    ):
        raise RelationalPreStatusOutcomeShardError(
            "source outcome report schema, kind, or status is invalid"
        )
    internal_sha = _sha(
        report.get("report_sha256"), "source report internal SHA-256"
    )
    if internal_sha != _self_hash(report, "report_sha256"):
        raise RelationalPreStatusOutcomeShardError(
            "source outcome report self-hash is invalid"
        )
    rows = _rows(report.get("scored_events"), "source scored events")
    _validate_rows(rows, "source scored event")
    return rows, internal_sha


def _build_shard(
    fold: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    copied_rows = [deepcopy(dict(row)) for row in rows]
    shard: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SHARD_KIND,
        "status": "success",
        "family_fold": fold,
        "event_count": len(copied_rows),
        "content_sha256": _content_hash(copied_rows),
        "scored_events": copied_rows,
    }
    shard["shard_sha256"] = _self_hash(shard, "shard_sha256")
    return shard


def build_relational_pre_status_outcome_shards(
    *,
    source_report_path: Path,
    expected_source_report_sha256: str,
    out_dir: Path,
    argv: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
    expected_event_count: int | None = None,
) -> Mapping[str, Any]:
    """Split one frozen report into exactly five immutable family-fold shards."""
    source_path = Path(source_report_path).resolve()
    expected_source_sha = _sha(
        expected_source_report_sha256,
        "expected source-report file SHA-256",
    )
    if not source_path.is_file():
        raise RelationalPreStatusOutcomeShardError(
            "source outcome report is not a file"
        )
    source_file_sha = file_sha256(source_path)
    if source_file_sha != expected_source_sha:
        raise RelationalPreStatusOutcomeShardError(
            "source outcome report differs from its expected physical SHA-256"
        )
    rows, source_internal_sha = _validate_source_report(
        _read_json(source_path, "source outcome report")
    )
    if expected_event_count is not None:
        expected_count = _integer(
            expected_event_count, "expected source event count"
        )
        if len(rows) != expected_count:
            raise RelationalPreStatusOutcomeShardError(
                "source event count is not the caller's expected inventory"
            )

    rows_by_fold: dict[str, list[Mapping[str, Any]]] = {
        fold: [] for fold in FOLDS
    }
    for row in rows:
        rows_by_fold[str(row["family_fold"])].append(row)
    if any(not fold_rows for fold_rows in rows_by_fold.values()):
        raise RelationalPreStatusOutcomeShardError(
            "source report does not contain every family fold"
        )
    if sum(map(len, rows_by_fold.values())) != len(rows):
        raise RelationalPreStatusOutcomeShardError(
            "shard counts do not sum to the source row count"
        )

    root = Path(out_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    shard_paths = {
        fold: _safe_child(
            root,
            _shard_relative_path(fold),
            f"{fold} shard path",
        )
        for fold in FOLDS
    }
    for path in (manifest_path, *shard_paths.values()):
        if path.exists():
            raise RelationalPreStatusOutcomeShardError(
                f"refusing to overwrite existing destination: {path}"
            )

    built_shards: dict[str, Mapping[str, Any]] = {}
    for fold in FOLDS:
        _validate_rows(
            rows_by_fold[fold],
            f"{fold} scored event",
            expected_fold=fold,
        )
        shard = _build_shard(fold, rows_by_fold[fold])
        _atomic_json_new(shard_paths[fold], shard)
        built_shards[fold] = shard

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "status": "success",
        "argv": list(argv),
        "provenance": dict(provenance) if provenance is not None else {},
        "source_report": {
            "file_sha256": source_file_sha,
            "internal_sha256": source_internal_sha,
        },
        "folds": list(FOLDS),
        "total_event_count": len(rows),
        "shards": {
            fold: {
                "path": _shard_relative_path(fold),
                "event_count": int(built_shards[fold]["event_count"]),
                "file_sha256": file_sha256(shard_paths[fold]),
                "content_sha256": str(
                    built_shards[fold]["content_sha256"]
                ),
                "shard_sha256": str(built_shards[fold]["shard_sha256"]),
            }
            for fold in FOLDS
        },
    }
    manifest["manifest_sha256"] = _self_hash(
        manifest, "manifest_sha256"
    )
    _atomic_json_new(manifest_path, manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("status") != "success"
        or manifest.get("manifest_sha256")
        != _self_hash(manifest, "manifest_sha256")
    ):
        raise RelationalPreStatusOutcomeShardError(
            "shard manifest schema, status, or self-hash is invalid"
        )
    folds = manifest.get("folds")
    if not isinstance(folds, list) or folds != list(FOLDS):
        raise RelationalPreStatusOutcomeShardError(
            "shard manifest fold inventory is invalid"
        )
    shards = _mapping(manifest.get("shards"), "manifest shards")
    if set(shards) != set(FOLDS):
        raise RelationalPreStatusOutcomeShardError(
            "shard manifest does not contain exactly the five folds"
        )
    counts = [
        _integer(
            _mapping(shards[fold], f"{fold} manifest entry").get(
                "event_count"
            ),
            f"{fold} manifest event count",
        )
        for fold in FOLDS
    ]
    if any(count < 1 for count in counts):
        raise RelationalPreStatusOutcomeShardError(
            "every manifest shard must contain events"
        )
    if sum(counts) != _integer(
        manifest.get("total_event_count"), "manifest total event count"
    ):
        raise RelationalPreStatusOutcomeShardError(
            "manifest shard counts do not sum to the total"
        )


def load_relational_pre_status_outcome_shard(
    root: Path,
    fold: str,
    *,
    expected_manifest_file_sha256: str,
    expected_shard_file_sha256: str,
    expected_content_sha256: str,
    expected_source_report_file_sha256: str,
) -> LoadedRelationalPreStatusOutcomeShard:
    """Load exactly one fold and validate pinned manifest, file, and content hashes."""
    if fold not in FOLDS:
        raise RelationalPreStatusOutcomeShardError(
            "requested fold is not a supported family fold"
        )
    manifest_file_expected = _sha(
        expected_manifest_file_sha256,
        "expected manifest file SHA-256",
    )
    shard_file_expected = _sha(
        expected_shard_file_sha256,
        "expected shard file SHA-256",
    )
    content_expected = _sha(
        expected_content_sha256,
        "expected shard content SHA-256",
    )
    source_expected = _sha(
        expected_source_report_file_sha256,
        "expected source-report file SHA-256",
    )

    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / MANIFEST_NAME
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != manifest_file_expected
    ):
        raise RelationalPreStatusOutcomeShardError(
            "manifest differs from its expected physical SHA-256"
        )
    manifest = _read_json(manifest_path, "outcome shard manifest")
    _validate_manifest(manifest)

    source = _mapping(
        manifest.get("source_report"), "manifest source report"
    )
    source_file_sha = _sha(
        source.get("file_sha256"),
        "manifest source-report file SHA-256",
    )
    source_internal_sha = _sha(
        source.get("internal_sha256"),
        "manifest source-report internal SHA-256",
    )
    if source_file_sha != source_expected:
        raise RelationalPreStatusOutcomeShardError(
            "manifest source binding differs from its expected physical SHA-256"
        )

    entry = _mapping(
        _mapping(manifest["shards"], "manifest shards").get(fold),
        f"{fold} manifest entry",
    )
    manifest_file_binding = _sha(
        entry.get("file_sha256"),
        f"{fold} manifest file SHA-256",
    )
    manifest_content_binding = _sha(
        entry.get("content_sha256"),
        f"{fold} manifest content SHA-256",
    )
    manifest_shard_binding = _sha(
        entry.get("shard_sha256"),
        f"{fold} manifest shard SHA-256",
    )
    if manifest_file_binding != shard_file_expected:
        raise RelationalPreStatusOutcomeShardError(
            "requested shard file hash differs from the manifest"
        )
    if manifest_content_binding != content_expected:
        raise RelationalPreStatusOutcomeShardError(
            "requested shard content hash differs from the manifest"
        )

    shard_path = _safe_child(
        artifact_root,
        entry.get("path"),
        f"{fold} shard path",
    )
    if (
        not shard_path.is_file()
        or file_sha256(shard_path) != shard_file_expected
    ):
        raise RelationalPreStatusOutcomeShardError(
            "requested shard differs from its expected physical SHA-256"
        )
    shard = _read_json(shard_path, f"{fold} outcome shard")
    if (
        shard.get("schema_version") != SCHEMA_VERSION
        or shard.get("kind") != SHARD_KIND
        or shard.get("status") != "success"
        or shard.get("family_fold") != fold
    ):
        raise RelationalPreStatusOutcomeShardError(
            "requested shard schema, status, or fold is invalid"
        )
    if (
        shard.get("shard_sha256") != manifest_shard_binding
        or _self_hash(shard, "shard_sha256") != manifest_shard_binding
    ):
        raise RelationalPreStatusOutcomeShardError(
            "requested shard self-hash is invalid"
        )

    rows = _rows(shard.get("scored_events"), f"{fold} scored events")
    _validate_rows(rows, f"{fold} scored event", expected_fold=fold)
    actual_content_sha = _content_hash(rows)
    if (
        shard.get("content_sha256") != content_expected
        or actual_content_sha != content_expected
    ):
        raise RelationalPreStatusOutcomeShardError(
            "requested shard content hash is invalid"
        )
    expected_count = _integer(
        entry.get("event_count"), f"{fold} manifest event count"
    )
    if (
        len(rows) != expected_count
        or shard.get("event_count") != expected_count
    ):
        raise RelationalPreStatusOutcomeShardError(
            "requested shard event count differs from the manifest"
        )

    return LoadedRelationalPreStatusOutcomeShard(
        family_fold=fold,
        scored_events=tuple(
            MappingProxyType(deepcopy(dict(row))) for row in rows
        ),
        source_report_file_sha256=source_file_sha,
        source_report_internal_sha256=source_internal_sha,
        manifest_file_sha256=manifest_file_expected,
        manifest_sha256=str(manifest["manifest_sha256"]),
        shard_file_sha256=shard_file_expected,
        content_sha256=content_expected,
        shard_sha256=manifest_shard_binding,
    )


__all__ = [
    "FOLDS",
    "LoadedRelationalPreStatusOutcomeShard",
    "MANIFEST_KIND",
    "MANIFEST_NAME",
    "RelationalPreStatusOutcomeShardError",
    "SCHEMA_VERSION",
    "SHARD_KIND",
    "SHARD_SUBDIR",
    "SOURCE_REPORT_KIND",
    "build_relational_pre_status_outcome_shards",
    "load_relational_pre_status_outcome_shard",
]
