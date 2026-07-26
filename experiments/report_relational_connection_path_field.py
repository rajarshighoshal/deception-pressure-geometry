"""Build, predict, score, and render the complete-path connection field."""
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

from geoprobe.eval.relational_connection_path_bank import (  # noqa: E402
    build_complete_path_bank,
)
from geoprobe.eval.relational_connection_path_field import (  # noqa: E402
    build_connection_path_prediction_ledger,
    canonical_sha256,
    score_connection_path_prediction_ledger,
)
from geoprobe.eval.relational_connection_path_field_report import (  # noqa: E402
    render_connection_path_field_markdown,
)
from geoprobe.geometry.relational_connection_path_distance import (  # noqa: E402
    stable_common_rank_relation_sides,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.provenance import git_provenance  # noqa: E402


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError(f"input must be an object: {path}")
    return value, sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
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
        raise ValueError(f"{name} self-hash mismatch for {field}")


def _iter_top_level_object_array(path: Path, key: str) -> Iterator[dict[str, Any]]:
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
        while True:
            while position < len(mapped) and mapped[position] in b" \t\r\n,":
                position += 1
            if position >= len(mapped):
                raise ValueError(f"unterminated top-level array: {key}")
            if mapped[position] == ord("]"):
                return
            if mapped[position] != ord("{"):
                raise ValueError(f"{key} array must contain objects")
            start = position
            depth = 0
            in_string = False
            escaped = False
            while position < len(mapped):
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
    fields = (
        "heldout_family_fold",
        "relation_name",
        "view",
        "selected_rank",
        "admissible",
        "status",
    )
    rows = [
        {field: raw.get(field) for field in fields}
        for raw in _iter_top_level_object_array(path, "selection")
    ]
    if len(rows) != 675:
        raise ValueError("calibration selection must contain 675 rows")
    return rows


def _resolve_bound_path(raw: object, *, relative_to: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is invalid")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def _checkpoint_stream(
    *, evidence: Mapping[str, Any], evidence_path: Path
) -> tuple[Iterator[dict[str, Any]], list[dict[str, str]]]:
    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(
        inputs.get("checkpoints"), list
    ):
        raise ValueError("connection evidence checkpoint inventory is invalid")
    bindings: list[tuple[Path, Mapping[str, Any]]] = []
    source_bindings: list[dict[str, str]] = []
    for raw in inputs["checkpoints"]:
        if not isinstance(raw, Mapping):
            raise ValueError("checkpoint binding is invalid")
        path = _resolve_bound_path(
            raw.get("path"), relative_to=evidence_path.parent, name="checkpoint"
        )
        bindings.append((path, raw))
        source_bindings.append(
            {
                "path": str(path),
                "file_sha256": str(raw.get("file_sha256")),
                "scenario_id": str(raw.get("scenario_id")),
                "scenario_checkpoint_sha256": str(
                    raw.get("scenario_checkpoint_sha256")
                ),
            }
        )
    if len(bindings) != 60:
        raise ValueError("connection evidence must bind exactly 60 checkpoints")

    def iterator() -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        for path, binding in bindings:
            checkpoint, digest = _read_json_bytes(path)
            if digest != binding.get("file_sha256"):
                raise ValueError(f"checkpoint file hash differs: {path}")
            scenario_id = checkpoint.get("scenario_id")
            if (
                not isinstance(scenario_id, str)
                or scenario_id != binding.get("scenario_id")
                or scenario_id in seen
            ):
                raise ValueError("checkpoint scenario binding is invalid")
            seen.add(scenario_id)
            _validate_self_hash(
                checkpoint,
                field="scenario_checkpoint_sha256",
                name=f"checkpoint {scenario_id}",
            )
            if checkpoint["scenario_checkpoint_sha256"] != binding.get(
                "scenario_checkpoint_sha256"
            ):
                raise ValueError("checkpoint internal hash differs from evidence")
            yield checkpoint
            del checkpoint
            gc.collect()

    return iterator(), source_bindings


def _input_binding(path: Path, digest: str | None = None) -> dict[str, str]:
    return {"path": str(path), "file_sha256": digest or file_sha256(path)}


def _build_bank(
    *,
    outcome_path: Path,
    evidence_path: Path,
    calibration_path: Path,
    effective_argv: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome, outcome_file_sha = _read_json_bytes(outcome_path)
    evidence, evidence_file_sha = _read_json_bytes(evidence_path)
    _validate_self_hash(outcome, field="report_sha256", name="outcome join")
    _validate_self_hash(
        evidence, field="report_sha256", name="connection evidence"
    )
    if evidence.get("kind") != "relational_partial_frame_connection_evidence_report":
        raise ValueError("connection evidence kind is invalid")
    bound_evidence = outcome.get("artifact_identity", {}).get(
        "connection_evidence_report", {}
    )
    if not isinstance(bound_evidence, Mapping) or bound_evidence.get(
        "file_sha256"
    ) != evidence_file_sha:
        raise ValueError("outcome join does not bind the connection evidence")

    calibration_file_sha = file_sha256(calibration_path)
    physical_calibration = evidence.get("inputs", {}).get(
        "physical_calibration", {}
    )
    bound_calibration = (
        physical_calibration.get("calibration", {})
        if isinstance(physical_calibration, Mapping)
        else {}
    )
    if not isinstance(bound_calibration, Mapping) or bound_calibration.get(
        "file_sha256"
    ) != calibration_file_sha:
        raise ValueError("connection evidence does not bind the calibration")
    stable_inventory = stable_common_rank_relation_sides(
        {"selection": _calibration_selection(calibration_path)},
        validate_counts=True,
    )
    checkpoints, checkpoint_bindings = _checkpoint_stream(
        evidence=evidence, evidence_path=evidence_path
    )
    bank = build_complete_path_bank(
        outcome_join=outcome,
        checkpoints_by_scenario=checkpoints,
        stable_relation_sides=stable_inventory,
    )
    bank.pop("bank_sha256", None)
    source_files: dict[str, Any] = {
        "outcome_join": _input_binding(outcome_path, outcome_file_sha),
        "connection_evidence_report": _input_binding(
            evidence_path, evidence_file_sha
        ),
        "calibration": _input_binding(calibration_path, calibration_file_sha),
        "checkpoints": checkpoint_bindings,
    }
    bank["argv"] = effective_argv
    bank["artifact_identity"] = source_files
    bank["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_connection_path_bank.py",
                REPO_ROOT
                / "src/geoprobe/geometry/relational_connection_path_distance.py",
            ]
        ),
        "resource_contract": {
            "model_loaded": False,
            "gpu_used": False,
            "checkpoint_loading": "one_file_at_a_time",
        },
    }
    bank["bank_sha256"] = canonical_sha256(bank)
    return bank, source_files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-join", type=Path, required=True)
    parser.add_argument("--connection-evidence-report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--spectral-prediction-ledger", type=Path, required=True)
    parser.add_argument("--out-bank", type=Path, required=True)
    parser.add_argument("--out-ledger", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    effective_argv = list(sys.argv) if argv is None else [parser.prog, *argv]
    inputs = {
        "outcome_join": args.outcome_join.resolve(),
        "connection_evidence_report": args.connection_evidence_report.resolve(),
        "calibration": args.calibration.resolve(),
        "spectral_prediction_ledger": args.spectral_prediction_ledger.resolve(),
    }
    outputs = {
        "bank": args.out_bank.resolve(),
        "prediction_ledger": args.out_ledger.resolve(),
        "score": args.out_json.resolve(),
        "markdown": args.out_md.resolve(),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("all output paths must be distinct")
    if set(outputs.values()) & set(inputs.values()):
        raise ValueError("an output path must not overwrite an input")
    if any(outputs[key].suffix != ".json" for key in ("bank", "prediction_ledger", "score")):
        raise ValueError("bank, prediction ledger, and score outputs must be JSON")

    bank, source_files = _build_bank(
        outcome_path=inputs["outcome_join"],
        evidence_path=inputs["connection_evidence_report"],
        calibration_path=inputs["calibration"],
        effective_argv=effective_argv,
    )
    _atomic_json(outputs["bank"], bank)
    bank_file_sha = file_sha256(outputs["bank"])

    spectral, spectral_file_sha = _read_json_bytes(
        inputs["spectral_prediction_ledger"]
    )
    _validate_self_hash(
        spectral,
        field="prediction_ledger_sha256",
        name="spectral prediction ledger",
    )
    prediction_ledger = build_connection_path_prediction_ledger(
        complete_path_bank=bank,
        spectral_prediction_ledger=spectral,
    )
    prediction_ledger.pop("prediction_ledger_sha256", None)
    prediction_ledger["argv"] = effective_argv
    prediction_ledger["artifact_identity"] = {
        "complete_path_bank_sha256": bank["bank_sha256"],
        "complete_path_bank_file_sha256": bank_file_sha,
        "spectral_prediction_ledger_sha256": spectral[
            "prediction_ledger_sha256"
        ],
        "spectral_prediction_ledger_file_sha256": spectral_file_sha,
    }
    prediction_ledger["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_connection_path_field.py",
            ]
        )
    }
    prediction_ledger["prediction_ledger_sha256"] = canonical_sha256(
        prediction_ledger
    )
    _atomic_json(outputs["prediction_ledger"], prediction_ledger)
    prediction_file_sha = file_sha256(outputs["prediction_ledger"])

    score = score_connection_path_prediction_ledger(
        prediction_ledger=prediction_ledger,
        complete_path_bank=bank,
    )
    score.pop("score_sha256", None)
    score["argv"] = effective_argv
    score["artifact_identity"] = {
        "complete_path_bank_sha256": bank["bank_sha256"],
        "complete_path_bank_file_sha256": bank_file_sha,
        "prediction_ledger_sha256": prediction_ledger[
            "prediction_ledger_sha256"
        ],
        "prediction_ledger_file_sha256": prediction_file_sha,
        "spectral_prediction_ledger_sha256": spectral[
            "prediction_ledger_sha256"
        ],
        "spectral_prediction_ledger_file_sha256": spectral_file_sha,
        "source_files": source_files,
    }
    score["provenance"] = {
        "git": git_provenance(
            [
                Path(__file__).resolve(),
                REPO_ROOT
                / "src/geoprobe/eval/relational_connection_path_field.py",
                REPO_ROOT
                / "src/geoprobe/eval/relational_connection_path_field_report.py",
            ]
        )
    }
    score["score_sha256"] = canonical_sha256(score)
    _atomic_json(outputs["score"], score)
    _atomic_text(
        outputs["markdown"],
        render_connection_path_field_markdown(
            score=score, ledger=prediction_ledger, bank=bank
        ),
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "outputs": {key: str(path) for key, path in outputs.items()},
                "bank_sha256": bank["bank_sha256"],
                "prediction_ledger_sha256": prediction_ledger[
                    "prediction_ledger_sha256"
                ],
                "score_sha256": score["score_sha256"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
