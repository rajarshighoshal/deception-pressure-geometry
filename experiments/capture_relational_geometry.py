"""Capture full-token residuals and causal attention for the structured-action bank.

This is the public scientific acquisition path: source binding, tensor capture,
parity checks, and checksummed row artifacts only.
"""

from __future__ import annotations

import argparse
from itertools import groupby
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from geoprobe.data.relational_structured_capture import (
    validate_structured_action_capture_source,
)
from geoprobe.io import file_sha256
from geoprobe.models.artifact_identity import fingerprint_local_hf_artifact
from geoprobe.models.interface import resolve_torch_dtype
from geoprobe.models.loader import load_hf_model
from geoprobe.models.relational_capture import (
    MATCHED_SHAPE_CLONE_PARITY,
    canonical_json_sha256,
    capture_relational_batch,
    compare_captured_rows,
    matched_shape_clone_batch,
    matched_shape_parity_roster,
    set_eager_attention,
    write_relational_row,
)
from geoprobe.provenance import git_provenance


DEFAULT_LAYERS = (12, 16, 19, 20)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def selected_rows(
    rows: Sequence[dict[str, Any]],
    conversation_ids_file: Path | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = list(rows)
    if conversation_ids_file is not None:
        requested = [
            line.strip()
            for line in conversation_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        wanted = set(requested)
        if len(requested) != len(wanted):
            raise ValueError("conversation ID selection contains duplicates")
        selected = [row for row in selected if str(row["conversation_id"]) in wanted]
        found = {str(row["conversation_id"]) for row in selected}
        if found != wanted:
            raise ValueError(f"conversation ID selection missing {sorted(wanted - found)[:5]}")
    selected.sort(key=lambda row: (len(row["token_ids"]), str(row["conversation_id"])))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("capture selection is empty")
    return selected


def capture_batches(
    rows: Sequence[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    """Batch only equal-length prefixes so padding cannot alter captured states."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    ordered = list(rows)
    lengths = [len(row["token_ids"]) for row in ordered]
    if lengths != sorted(lengths):
        raise ValueError("capture rows must be sorted by token length")
    batches: list[list[dict[str, Any]]] = []
    for _, group in groupby(ordered, key=lambda row: len(row["token_ids"])):
        same_length = list(group)
        batches.extend(
            same_length[index : index + batch_size]
            for index in range(0, len(same_length), batch_size)
        )
    return batches


def capture_contract(
    *,
    rows_path: Path,
    protocol_path: Path,
    model_artifact: dict[str, Any],
    target_rows: Sequence[dict[str, Any]],
    layers: Sequence[int],
    dtype: str,
    device: str,
    batch_size: int,
    parity_rows: int,
    residual_parity_atol: float,
    attention_parity_atol: float,
    attention_row_sum_atol: float,
) -> dict[str, Any]:
    selected_layers = [int(layer) for layer in layers]
    if not selected_layers or len(selected_layers) != len(set(selected_layers)):
        raise ValueError("capture layers must be non-empty and unique")
    if parity_rows < 1 or parity_rows > len(target_rows):
        raise ValueError("parity row count is outside the capture selection")
    body = {
        "source": {
            "rows_sha256": file_sha256(rows_path),
            "protocol_sha256": file_sha256(protocol_path),
            "model_artifact_sha256": model_artifact["artifact_sha256"],
            "model_weights_sha256": model_artifact["weights_sha256"],
            "model_config_sha256": model_artifact["model_config_sha256"],
            "tokenizer_sha256": model_artifact["tokenizer_sha256"],
        },
        "selection": [
            {
                "conversation_id": str(row["conversation_id"]),
                "row_sha256": canonical_json_sha256(row),
                "token_ids_sha256": canonical_json_sha256(row["token_ids"]),
            }
            for row in target_rows
        ],
        "capture": {
            "layers": selected_layers,
            "dtype": dtype,
            "device": device,
            "batch_size": int(batch_size),
            "attention_representation": "lossless_causal_lower_triangle_per_head",
            "parity_reference_mode": MATCHED_SHAPE_CLONE_PARITY,
            "parity_rows": int(parity_rows),
            "residual_parity_atol": float(residual_parity_atol),
            "attention_parity_atol": float(attention_parity_atol),
            "attention_row_sum_atol": float(attention_row_sum_atol),
        },
    }
    return {
        "schema_version": 1,
        "kind": "public_relational_capture_contract",
        **body,
        "contract_sha256": canonical_json_sha256(body),
    }


def establish_capture_contract(
    path: Path, contract: dict[str, Any], *, resume: bool
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"capture contract already exists: {path}")
        if existing != contract:
            raise ValueError("resume capture arguments or source identity changed")
        return
    if path.parent.joinpath("rows").exists() or path.parent.joinpath("manifest.json").exists():
        raise ValueError("capture artifacts exist without a matching contract")
    atomic_json(path, contract)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parity-rows", type=int, default=2)
    parser.add_argument("--residual-parity-atol", type=float, default=0.02)
    parser.add_argument("--attention-parity-atol", type=float, default=0.002)
    parser.add_argument("--attention-row-sum-atol", type=float, default=0.02)
    parser.add_argument("--conversation-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = load_jsonl(args.rows)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    source_binding = validate_structured_action_capture_source(rows, protocol)
    targets = selected_rows(rows, args.conversation_ids_file, args.limit)
    batches = capture_batches(targets, args.batch_size)
    parity_rows = min(args.parity_rows, len(targets))

    model_artifact = fingerprint_local_hf_artifact(args.model_dir)
    contract = capture_contract(
        rows_path=args.rows,
        protocol_path=args.protocol,
        model_artifact=model_artifact,
        target_rows=targets,
        layers=args.layers,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        parity_rows=parity_rows,
        residual_parity_atol=args.residual_parity_atol,
        attention_parity_atol=args.attention_parity_atol,
        attention_row_sum_atol=args.attention_row_sum_atol,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    establish_capture_contract(
        args.out_dir / "capture_contract.json", contract, resume=args.resume
    )

    model, tokenizer, _ = load_hf_model(
        str(args.model_dir),
        device=args.device,
        dtype=resolve_torch_dtype(args.dtype),
    )
    set_eager_attention(model)
    hidden_size = int(model.config.hidden_size)
    attention_heads = int(model.config.num_attention_heads)
    effective_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    if effective_dtype != args.dtype:
        raise ValueError(
            f"loaded model dtype {effective_dtype!r} differs from requested {args.dtype!r}"
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no pad token ID")

    roster = matched_shape_parity_roster(
        targets, batch_size=args.batch_size, count=parity_rows
    )
    parity_by_id = {entry["conversation_id"]: entry for entry in roster}
    parity: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches:
            token_batch = [row["token_ids"] for row in batch]
            captured = capture_relational_batch(
                model,
                token_batch,
                layers=args.layers,
                pad_token_id=pad_token_id,
                row_sum_atol=args.attention_row_sum_atol,
            )
            for batch_index, (row, state) in enumerate(zip(batch, captured, strict=True)):
                conversation_id = str(row["conversation_id"])
                if conversation_id in parity_by_id:
                    clone_batch = matched_shape_clone_batch(
                        token_batch, batch_index=batch_index
                    )
                    reference = capture_relational_batch(
                        model,
                        clone_batch,
                        layers=args.layers,
                        pad_token_id=pad_token_id,
                        row_sum_atol=args.attention_row_sum_atol,
                    )[batch_index]
                    parity.append({
                        "conversation_id": conversation_id,
                        **compare_captured_rows(
                            reference,
                            state,
                            residual_atol=args.residual_parity_atol,
                            attention_atol=args.attention_parity_atol,
                        ),
                    })
                records.append(
                    write_relational_row(
                        args.out_dir,
                        row,
                        state,
                        resume=args.resume,
                        capture_contract_sha256=contract["contract_sha256"],
                        layers=args.layers,
                        hidden_size=hidden_size,
                        n_attention_heads=attention_heads,
                        residual_dtype=args.dtype,
                        attention_dtype=args.dtype,
                    )
                )

    manifest_body = {
        "contract_sha256": contract["contract_sha256"],
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "row_count": len(records),
        "row_records_sha256": canonical_json_sha256(records),
        "parity": parity,
        "provenance": git_provenance([Path(__file__).resolve()]),
    }
    manifest = {
        "schema_version": 1,
        "kind": "public_relational_capture_manifest",
        "status": "success",
        **manifest_body,
        "manifest_sha256": canonical_json_sha256(manifest_body),
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
