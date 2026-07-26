"""Build, predict, and score the fixed-pressure intrinsic spectral field."""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import gc
from hashlib import sha256
import json
import mmap
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.eval.relational_intrinsic_outcome_bank import (  # noqa: E402
    build_an_turn2_intrinsic_quotients,
    project_alignment_checkpoint,
    project_profile_checkpoint,
    validate_frozen_an_turn2_intrinsic_contract,
)
from geoprobe.eval.relational_intrinsic_spectral_field import (  # noqa: E402
    build_intrinsic_spectral_prediction_ledger,
    canonical_sha256,
    score_intrinsic_spectral_prediction_ledger,
)
from geoprobe.eval.relational_intrinsic_spectral_field_report import (  # noqa: E402
    render_intrinsic_spectral_field_markdown,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.provenance import git_provenance  # noqa: E402


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"input must be a JSON object: {path}")
    return value, sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, name: str
) -> None:
    declared = value.get(field)
    if not isinstance(declared, str):
        raise ValueError(f"{name} has no {field}")
    payload = dict(value)
    payload.pop(field, None)
    if canonical_sha256(payload) != declared:
        raise ValueError(f"{name} self-hash is invalid")


def _iter_top_level_object_array(path: Path, key: str) -> Iterator[dict[str, Any]]:
    """Stream one top-level object array without materializing a large JSON file."""
    marker = json.dumps(key).encode("utf-8")
    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        marker_index = mapped.find(marker)
        if marker_index < 0:
            raise ValueError(f"top-level key is missing: {key}")
        colon = mapped.find(b":", marker_index + len(marker))
        array_start = mapped.find(b"[", colon + 1)
        if colon < 0 or array_start < 0:
            raise ValueError(f"top-level key is not an array: {key}")
        position = array_start + 1
        length = len(mapped)
        while True:
            while position < length and mapped[position] in b" \t\r\n,":
                position += 1
            if position >= length:
                raise ValueError(f"unterminated top-level array: {key}")
            if mapped[position] == ord("]"):
                return
            if mapped[position] != ord("{"):
                raise ValueError(f"{key} array must contain only objects")
            start = position
            depth = 0
            in_string = False
            escaped = False
            while position < length:
                byte = mapped[position]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                elif byte == ord('"'):
                    in_string = True
                elif byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        position += 1
                        value = json.loads(mapped[start:position])
                        if not isinstance(value, dict):
                            raise ValueError(f"{key} item is not an object")
                        yield value
                        break
                position += 1
            else:
                raise ValueError(f"unterminated object in {key}")


def _calibration_selection(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in _iter_top_level_object_array(path, "selection"):
        rows.append(
            {
                key: raw.get(key)
                for key in (
                    "heldout_family_fold",
                    "relation_name",
                    "view",
                    "selected_rank",
                    "admissible",
                    "status",
                    "fallback_reason",
                )
            }
        )
    if len(rows) != 675:
        raise ValueError("calibration selection must contain 675 fold-relation rows")
    return rows


def _resolve_bound_path(raw: object, *, relative_to: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is invalid")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _input_binding(path: Path, digest: str | None = None) -> dict[str, str]:
    return {"path": str(path), "sha256": digest or file_sha256(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-join", type=Path, required=True)
    parser.add_argument("--connection-evidence-report", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--out-bank", type=Path, required=True)
    parser.add_argument("--out-ledger", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    effective_argv = list(sys.argv) if argv is None else [parser.prog, *argv]

    outcome_path = args.outcome_join.resolve()
    evidence_path = args.connection_evidence_report.resolve()
    manifest_path = args.calibration_manifest.resolve()
    calibration_report_path = args.calibration_report.resolve()
    calibration_path = args.calibration.resolve()
    outputs = {
        "bank": args.out_bank.resolve(),
        "ledger": args.out_ledger.resolve(),
        "score": args.out_json.resolve(),
        "markdown": args.out_md.resolve(),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("all output paths must differ")

    outcome_join, outcome_file_sha = _read_json_bytes(outcome_path)
    evidence, evidence_file_sha = _read_json_bytes(evidence_path)
    manifest, manifest_file_sha = _read_json_bytes(manifest_path)
    calibration_report, calibration_report_file_sha = _read_json_bytes(
        calibration_report_path
    )
    _validate_self_hash(outcome_join, field="report_sha256", name="outcome join")
    _validate_self_hash(manifest, field="manifest_sha256", name="calibration manifest")
    _validate_self_hash(
        calibration_report,
        field="report_sha256",
        name="calibration report",
    )
    if calibration_report.get("kind") != "relational_partial_frame_calibration_report":
        raise ValueError("calibration report kind is invalid")
    report_inputs = calibration_report.get("inputs")
    if not isinstance(report_inputs, Mapping):
        raise ValueError("calibration report inputs are invalid")
    report_calibration = report_inputs.get("calibration")
    report_manifest = report_inputs.get("manifest")
    if not isinstance(report_calibration, Mapping) or not isinstance(
        report_manifest, Mapping
    ):
        raise ValueError("calibration report bindings are invalid")
    if report_manifest.get("file_sha256") != manifest_file_sha:
        raise ValueError("calibration report does not bind the supplied manifest")
    calibration_file_sha = file_sha256(calibration_path)
    if report_calibration.get("file_sha256") != calibration_file_sha:
        raise ValueError("calibration report does not bind the supplied calibration")
    if report_calibration.get("calibration_sha256") != manifest.get(
        "calibration_sha256"
    ):
        raise ValueError("manifest and calibration report identities differ")
    bound_evidence = outcome_join.get("artifact_identity", {}).get(
        "connection_evidence_report", {}
    )
    if bound_evidence.get("file_sha256") != evidence_file_sha:
        raise ValueError("outcome join does not bind the supplied evidence bytes")

    alignment: dict[str, dict[str, Any]] = {}
    alignment_files: list[dict[str, str]] = []
    sections_by_scenario: dict[str, set[str]] = {}
    evidence_inputs = evidence.get("inputs")
    if not isinstance(evidence_inputs, Mapping) or not isinstance(
        evidence_inputs.get("checkpoints"), list
    ):
        raise ValueError("connection evidence checkpoint inventory is invalid")
    for binding in evidence_inputs["checkpoints"]:
        if not isinstance(binding, Mapping):
            raise ValueError("connection evidence checkpoint binding is invalid")
        path = _resolve_bound_path(
            binding.get("path"), relative_to=evidence_path.parent, name="checkpoint"
        )
        raw, digest = _read_json_bytes(path)
        if digest != binding.get("file_sha256"):
            raise ValueError(f"alignment checkpoint file hash differs: {path}")
        projected = project_alignment_checkpoint(raw)
        scenario_id = projected.get("scenario_id")
        if not isinstance(scenario_id, str) or scenario_id in alignment:
            raise ValueError("alignment scenario inventory is duplicated or invalid")
        if scenario_id != binding.get("scenario_id"):
            raise ValueError("alignment checkpoint scenario binding differs")
        alignment[scenario_id] = projected
        sections_by_scenario[scenario_id] = {
            attempt["source_section_sha256"] for attempt in projected["attempts"]
        }
        if not sections_by_scenario[scenario_id]:
            raise ValueError("alignment scenario has no AN turn-two source section")
        alignment_files.append(_input_binding(path, digest))
        del raw

    profile_namespace = manifest.get("profile_namespace")
    if args.profiles_dir is not None:
        profiles_dir = args.profiles_dir.resolve()
    else:
        profiles_dir = _resolve_bound_path(
            profile_namespace,
            relative_to=manifest_path.parent,
            name="profile namespace",
        )
    profile_paths = sorted(profiles_dir.glob("*.json"))
    selected_scenarios = manifest.get("selected_scenario_ids")
    if not isinstance(selected_scenarios, list) or len(profile_paths) != len(
        selected_scenarios
    ):
        raise ValueError("profile file inventory differs from selected scenarios")
    profiles: dict[str, dict[str, Any]] = {}
    profile_files: list[dict[str, str]] = []
    for path in profile_paths:
        raw, digest = _read_json_bytes(path)
        scenario_id = raw.get("scenario_id")
        if (
            not isinstance(scenario_id, str)
            or scenario_id not in sections_by_scenario
            or scenario_id in profiles
        ):
            raise ValueError("profile scenario inventory is invalid")
        profiles[scenario_id] = project_profile_checkpoint(
            raw,
            section_sha256s=sections_by_scenario[scenario_id],
        )
        profile_files.append(_input_binding(path, digest))
        del raw
        gc.collect()
    if set(profiles) != set(selected_scenarios):
        raise ValueError("profile scenarios differ from the frozen manifest")

    expected_counts = {
        "HONEST": 32,
        "DECEPTIVE": 74,
        "SKIP": 7,
        "NO_ACTION": 0,
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 7,
    }
    bank = build_an_turn2_intrinsic_quotients(
        outcome_join=outcome_join,
        connection_evidence_report=evidence,
        alignment_checkpoints_by_scenario=alignment,
        calibration_manifest=manifest,
        profile_checkpoints_by_scenario=profiles,
        expected_event_count=120,
        expected_class_counts=expected_counts,
        expected_scenario_count=60,
        expected_family_count=20,
        expected_fold_count=5,
    )
    validate_frozen_an_turn2_intrinsic_contract(bank)
    source_files = {
        "outcome_join": _input_binding(outcome_path, outcome_file_sha),
        "connection_evidence_report": _input_binding(
            evidence_path, evidence_file_sha
        ),
        "calibration_manifest": _input_binding(manifest_path, manifest_file_sha),
        "calibration_report": _input_binding(
            calibration_report_path, calibration_report_file_sha
        ),
        "calibration": _input_binding(calibration_path, calibration_file_sha),
        "alignment_checkpoints": alignment_files,
        "profile_checkpoints": profile_files,
    }
    bank["argv"] = effective_argv
    bank["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_intrinsic_outcome_bank.py",
            ]
        ),
        "inputs": source_files,
    }
    bank["bank_sha256"] = canonical_sha256(bank)
    _atomic_json(outputs["bank"], bank)

    selection = _calibration_selection(calibration_path)
    ledger = build_intrinsic_spectral_prediction_ledger(
        quotients=bank["quotients"],
        calibration_selection=selection,
    )
    ledger.pop("prediction_ledger_sha256", None)
    ledger["argv"] = effective_argv
    ledger["artifact_identity"] = {
        "outcome_bank_sha256": bank["bank_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
        "calibration_file_sha256": calibration_file_sha,
    }
    ledger["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_intrinsic_spectral_field.py",
                REPO_ROOT
                / "src/geoprobe/geometry/relational_spectral_distance.py",
            ]
        ),
        "inputs": source_files,
    }
    ledger["prediction_ledger_sha256"] = canonical_sha256(ledger)
    _atomic_json(outputs["ledger"], ledger)

    score = score_intrinsic_spectral_prediction_ledger(
        prediction_ledger=ledger,
        quotients=bank["quotients"],
    )
    score.pop("score_sha256", None)
    score["argv"] = effective_argv
    score["artifact_identity"] = {
        "outcome_bank_sha256": bank["bank_sha256"],
        "prediction_ledger_file_sha256": file_sha256(outputs["ledger"]),
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
    }
    score["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_intrinsic_spectral_field.py",
                REPO_ROOT
                / "src/geoprobe/eval/relational_intrinsic_spectral_field_report.py",
            ]
        )
    }
    score["score_sha256"] = canonical_sha256(score)
    _atomic_json(outputs["score"], score)
    _atomic_text(
        outputs["markdown"],
        render_intrinsic_spectral_field_markdown(
            score=score,
            ledger=ledger,
            bank=bank,
        ),
    )
    print(
        json.dumps(
            {
                "bank_sha256": bank["bank_sha256"],
                "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
                "score_sha256": score["score_sha256"],
                "outputs": {key: str(path) for key, path in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
