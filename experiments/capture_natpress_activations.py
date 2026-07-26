"""Natpress activation capture: plan ($0) / run (pod, CUDA, bf16) / verify ($0).

Replays the frozen natpress banks' EXACT stored ``token_ids`` (no re-tokenization) through
Llama-3.1-8B in torch.bfloat16 with eager attention, capturing every-token residuals at
layers [12,16,19,20] for all conversations and packed head-wise attention weights at
layers [16,20] for the pre-registered smooth subset. batch_size=1 everywhere; the
matched-shape clone parity check (atol 0.0) is therefore a bitwise re-run reproducibility
gate (see src/geoprobe/models/natpress_capture.py for the frozen decisions).

Reuses the relational capture library by IMPORT ONLY (set_eager_attention,
capture_relational_batch, matched_shape_clone_batch, compare_captured_rows,
validate_relational_parity_entries); the structured relational row writer/validator
rejects free-prose rows by design and is not touched -- natpress rows get their own
atomic safetensors writer here.

Subcommands:
  plan    build + validate natpress_capture_plan.json from both banks' rows.jsonl ($0).
  run     pod-side capture honoring the plan verbatim (requires CUDA; refuses otherwise).
  verify  structural + parity validation of an output dir against the plan ($0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geoprobe.models.natpress_capture import (  # noqa: E402
    ATTENTION_PARITY_ATOL,
    NATPRESS_CAPTURE_ROW_SCHEMA,
    RESIDUAL_PARITY_ATOL,
    VERIFY_SCALAR_ROWS,
    build_natpress_capture_plan,
    canonical_json_sha256,
    derive_token_annotations,
    validate_natpress_capture_plan,
)
from geoprobe.provenance import git_provenance  # noqa: E402


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _row_paths(out_dir: Path, conversation_id: str) -> tuple[Path, Path]:
    stem = conversation_id.replace(":", "__")
    return out_dir / "rows" / f"{stem}.safetensors", out_dir / "rows" / f"{stem}.json"


# ======================================== plan ==========================================

def cmd_plan(args: argparse.Namespace) -> int:
    scripted = _load_rows(args.scripted_rows)
    adaptive = _load_rows(args.adaptive_rows)
    plan = build_natpress_capture_plan(
        scripted,
        adaptive,
        scripted_rows_sha256=file_sha256(args.scripted_rows),
        adaptive_rows_sha256=file_sha256(args.adaptive_rows),
    )
    validate_natpress_capture_plan(plan)
    artifact = dict(plan)
    artifact["git_provenance"] = git_provenance([args.scripted_rows, args.adaptive_rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    projection = plan["byte_projection"]
    print(
        f"plan OK: {plan['n_rows']} rows ({plan['source_banks']['n_scripted']} scripted + "
        f"{plan['source_banks']['n_adaptive']} adaptive), attention subset "
        f"{plan['n_attention_rows']} rows"
    )
    print(
        f"  bytes: residual {projection['total_residual_bytes'] / 1e9:.2f} GB + attention "
        f"{projection['total_attention_bytes'] / 1e9:.2f} GB + annotations "
        f"{projection['annotation_overhead_bytes'] / 1e9:.3f} GB = "
        f"{projection['projected_total_bytes'] / 1e9:.2f} GB "
        f"({100 * projection['cap_fraction']:.1f}% of the {projection['cap_bytes'] / 2**30:.0f} GiB cap)"
    )
    memory = plan["eager_attention_memory"]
    print(
        f"  eager-attention peak at max T={memory['max_token_length']}: "
        f"{memory['peak_scores_bytes_per_layer_at_max_T'] / 1e6:.0f} MB/layer x "
        f"{memory['hooked_layers']} hooked layers = "
        f"{memory['peak_retained_scores_bytes_at_max_T'] / 1e9:.2f} GB retained (batch 1)"
    )
    print(f"wrote {out}")
    return 0


# ======================================== run ===========================================

def cmd_run(args: argparse.Namespace) -> int:
    import torch  # deferred: plan/verify stay torch-free
    from transformers import AutoModelForCausalLM

    from geoprobe.models.relational_capture import (
        capture_relational_batch,
        compare_captured_rows,
        matched_shape_clone_batch,
        set_eager_attention,
    )

    if not torch.cuda.is_available():
        raise SystemExit("capture run requires CUDA (bf16 policy); refusing CPU run")

    plan = json.loads(Path(args.plan).read_text())
    validate_natpress_capture_plan(plan)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for source in (args.scripted_rows, args.adaptive_rows):
        for row in _load_rows(source):
            rows_by_id[str(row["conversation_id"])] = row
    missing = [r["conversation_id"] for r in plan["rows"] if r["conversation_id"] not in rows_by_id]
    if missing:
        raise SystemExit(f"plan rows missing from banks: {missing[:4]} (+{max(0, len(missing) - 4)})")

    out_dir = Path(args.out_dir)
    (out_dir / "rows").mkdir(parents=True, exist_ok=True)
    contract = {
        "kind": "natpress_capture_contract",
        "plan_sha256": plan["plan_sha256"],
        "model_id": args.model_id,
        "dtype": "bfloat16",
        "batch_size": plan["batch_size"],
        "residual_layers": plan["residual_layers"],
        "attention_layers": plan["attention_layers"],
        "parity": plan["parity"],
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    contract_path = out_dir / "capture_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text())
        if existing.get("contract_sha256") != contract["contract_sha256"]:
            raise SystemExit("existing capture contract differs; refusing to mix runs")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    set_eager_attention(model)
    capture_layers = plan["residual_layers"]  # attention hooks require every captured layer
    pad_token_id = int(model.config.eos_token_id if isinstance(model.config.eos_token_id, int)
                       else model.config.eos_token_id[0])

    # ---- parity: first VERIFY_SCALAR_ROWS plan rows (shortest first), clone batch of 1 ----
    parity_path = out_dir / "parity_checkpoint.json"
    parity_entries: list[dict[str, Any]] = []
    if parity_path.exists():
        checkpoint = json.loads(parity_path.read_text())
        if checkpoint.get("capture_contract_sha256") != contract["contract_sha256"]:
            raise SystemExit("parity checkpoint belongs to a different contract")
        parity_entries = list(checkpoint.get("entries", []))
    parity_ids = [row["conversation_id"] for row in plan["rows"][:VERIFY_SCALAR_ROWS]]
    for conversation_id in parity_ids:
        if any(entry["conversation_id"] == conversation_id for entry in parity_entries):
            continue
        token_ids = [int(t) for t in rows_by_id[conversation_id]["token_ids"]]
        reference = capture_relational_batch(
            model, [token_ids], layers=capture_layers, pad_token_id=pad_token_id
        )[0]
        clone = matched_shape_clone_batch([token_ids], batch_index=0)
        candidate = capture_relational_batch(
            model, clone, layers=capture_layers, pad_token_id=pad_token_id
        )[0]
        metrics = compare_captured_rows(
            reference, candidate,
            residual_atol=RESIDUAL_PARITY_ATOL, attention_atol=ATTENTION_PARITY_ATOL,
        )
        parity_entries.append({"conversation_id": conversation_id, **metrics})
        parity_path.write_text(json.dumps({
            "kind": "natpress_capture_parity_checkpoint",
            "capture_contract_sha256": contract["contract_sha256"],
            "entries": parity_entries,
        }, indent=2) + "\n")
        print(f"parity OK {conversation_id}: max metric "
              f"{max(metrics.values()):.3g} (atol {RESIDUAL_PARITY_ATOL})")

    # ---- capture rows, shortest first, batch 1, atomic writes, resumable ------------------
    from safetensors.torch import save_file

    n_written = n_skipped = 0
    for entry in plan["rows"]:
        conversation_id = entry["conversation_id"]
        tensor_path, record_path = _row_paths(out_dir, conversation_id)
        if tensor_path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("tensor_sha256") == file_sha256(tensor_path):
                n_skipped += 1
                continue
            raise SystemExit(f"corrupt existing artifact for {conversation_id}")
        row = rows_by_id[conversation_id]
        token_ids = [int(t) for t in row["token_ids"]]
        if canonical_json_sha256(token_ids) != entry["token_ids_sha256"]:
            raise SystemExit(f"token_ids drifted vs plan for {conversation_id}")
        captured = capture_relational_batch(
            model, [token_ids], layers=capture_layers, pad_token_id=pad_token_id
        )[0]
        annotations = derive_token_annotations(token_ids)
        import torch as _torch
        tensors = {
            "token_ids": captured.token_ids.contiguous(),
            "position_ids": _torch.arange(len(token_ids), dtype=_torch.int32),
            "token_role_ids": _torch.tensor(annotations["token_role_ids"], dtype=_torch.uint8),
            "token_turn_ids": _torch.tensor(annotations["token_turn_ids"], dtype=_torch.int16),
        }
        for layer in plan["residual_layers"]:
            tensors[f"residual.L{layer}"] = captured.residuals[layer].contiguous()
        if entry["capture_attention"]:
            for layer in plan["attention_layers"]:
                tensors[f"attention.L{layer}"] = captured.attentions[layer].contiguous()
        tmp_path = tensor_path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(tmp_path))
        os.replace(tmp_path, tensor_path)
        record = {
            "schema_version": NATPRESS_CAPTURE_ROW_SCHEMA,
            "conversation_id": conversation_id,
            "protocol": entry["protocol"],
            "bank": entry["bank"],
            "family": entry["family"],
            "scenario_id": entry["scenario_id"],
            "arm": entry["arm"],
            "sample_index": entry["sample_index"],
            "token_length": entry["token_length"],
            "token_ids_sha256": entry["token_ids_sha256"],
            "capture_attention": entry["capture_attention"],
            "residual_layers": plan["residual_layers"],
            "attention_layers": plan["attention_layers"] if entry["capture_attention"] else [],
            "capture_contract_sha256": contract["contract_sha256"],
            "tensor_path": f"rows/{tensor_path.name}",
            "tensor_bytes": tensor_path.stat().st_size,
            "tensor_sha256": file_sha256(tensor_path),
        }
        tmp_record = record_path.with_suffix(".json.tmp")
        tmp_record.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp_record, record_path)
        n_written += 1
        if n_written % 10 == 0:
            print(f"captured {n_written + n_skipped}/{plan['n_rows']}")

    manifest = {
        "kind": "natpress_capture_manifest",
        "plan_sha256": plan["plan_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "n_rows": plan["n_rows"],
        "n_written": n_written,
        "n_resumed": n_skipped,
        "parity_entries": len(parity_entries),
        "git_provenance": git_provenance([args.plan]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"capture complete: {n_written} written, {n_skipped} resumed, "
          f"{len(parity_entries)} parity rows -> {out_dir}")
    return 0


# ======================================== verify ========================================

def cmd_verify(args: argparse.Namespace) -> int:
    from geoprobe.models.relational_capture import validate_relational_parity_entries

    plan = json.loads(Path(args.plan).read_text())
    validate_natpress_capture_plan(plan)
    out_dir = Path(args.out_dir)
    contract = json.loads((out_dir / "capture_contract.json").read_text())
    if contract.get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit("VERIFY FAIL: contract does not bind this plan")
    problems: list[str] = []
    total_bytes = 0
    for entry in plan["rows"]:
        tensor_path, record_path = _row_paths(out_dir, entry["conversation_id"])
        if not tensor_path.exists() or not record_path.exists():
            problems.append(f"missing artifact: {entry['conversation_id']}")
            continue
        record = json.loads(record_path.read_text())
        if record.get("token_ids_sha256") != entry["token_ids_sha256"]:
            problems.append(f"token sha mismatch: {entry['conversation_id']}")
        if record.get("capture_contract_sha256") != contract["contract_sha256"]:
            problems.append(f"contract mismatch: {entry['conversation_id']}")
        actual_sha = file_sha256(tensor_path)
        if record.get("tensor_sha256") != actual_sha:
            problems.append(f"tensor sha mismatch: {entry['conversation_id']}")
        total_bytes += tensor_path.stat().st_size
    checkpoint = json.loads((out_dir / "parity_checkpoint.json").read_text())
    if checkpoint.get("capture_contract_sha256") != contract["contract_sha256"]:
        problems.append("parity checkpoint contract mismatch")
    else:
        try:
            validate_relational_parity_entries(
                checkpoint.get("entries", []),
                expected_conversation_ids=[
                    row["conversation_id"] for row in plan["rows"][:VERIFY_SCALAR_ROWS]
                ],
                layers=plan["residual_layers"],
                residual_atol=RESIDUAL_PARITY_ATOL,
                attention_atol=ATTENTION_PARITY_ATOL,
            )
        except ValueError as exc:
            problems.append(f"parity checkpoint invalid: {exc}")
    for problem in problems[:20]:
        print(f"  PROBLEM {problem}")
    verdict = "PASS" if not problems else "FAIL"
    print(f"verify: {verdict} rows={plan['n_rows']} bytes={total_bytes / 1e9:.2f} GB "
          f"problems={len(problems)}")
    return 0 if not problems else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="build + validate the capture plan ($0)")
    plan_parser.add_argument("--scripted-rows", required=True)
    plan_parser.add_argument("--adaptive-rows", required=True)
    plan_parser.add_argument("--out", required=True)

    run_parser = sub.add_parser("run", help="pod-side capture (CUDA, bf16)")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--scripted-rows", required=True)
    run_parser.add_argument("--adaptive-rows", required=True)
    run_parser.add_argument("--model-id", required=True)
    run_parser.add_argument("--out-dir", required=True)

    verify_parser = sub.add_parser("verify", help="structural + parity verification ($0)")
    verify_parser.add_argument("--plan", required=True)
    verify_parser.add_argument("--out-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "run":
        return cmd_run(args)
    return cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
