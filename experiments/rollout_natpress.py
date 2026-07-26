"""Roll out natural-pressure (``natpress_v1``) conversations into a ``natpress_rollout_v1`` bank.

Given a validated arcs config, plans every (scenario, arm, sample) conversation, generates the
assistant replies (real HF model, or a torch-free StubGenerator), and writes ``out/rows.jsonl`` +
``out/manifest.json``. ``--dry-run`` validates the config, builds the full plan + a model-free seed
table, estimates a token budget, writes ``out/dry_run_report.json`` and loads NO model.

Sibling sharing (bit-identical shared prefixes, zero regeneration) is handled by the driver: benign
conversations generate first and step/latedump copy their reply for the shared turns.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoprobe.io import file_sha256
from geoprobe.models.natpress_rollout import (
    SCHEMA_VERSION,
    StubGenerator,
    _sampling_block,
    generate_conversation,
    load_natpress_config,
    plan_conversations,
)
from geoprobe.provenance import git_provenance

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _scenario_index(cfg: dict) -> dict[str, dict]:
    return {scenario["scenario_id"]: scenario for scenario in cfg["scenarios"]}


def _limit_plan(plan: list[dict], limit: int | None) -> list[dict]:
    """Keep the first ``limit`` conversations, then drop any step/latedump orphaned from its benign."""
    if limit is None or limit >= len(plan):
        return plan
    kept = plan[:limit]
    benign_keys = {
        (entry["scenario_id"], entry["sample_index"])
        for entry in kept if entry["arm"] == "benign"
    }
    return [
        entry for entry in kept
        if entry["arm"] in ("benign", "smooth")
        or (entry["scenario_id"], entry["sample_index"]) in benign_keys
    ]


def _load_hf_generator(model_id: str, device: str | None):
    from geoprobe.models.loader import choose_device
    from geoprobe.models.natpress_rollout import HFGenerator
    from geoprobe.paths import ensure_hf_env

    ensure_hf_env()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or choose_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # BF16 REQUIRED on accelerators (standing rule: never fp16); fp32 on CPU.
    dtype = torch.float32 if resolved_device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)
    model.to(resolved_device)
    model.eval()
    return HFGenerator(model, tokenizer, resolved_device)


def run_rollout(cfg: dict, generator, out_dir: Path, *, model_id: str, config_sha256: str,
                provenance: dict, limit: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    scenarios = _scenario_index(cfg)
    plan = _limit_plan(plan_conversations(cfg), limit)
    benign_cache: dict[tuple[str, int], dict] = {}

    start = time.time()
    rows: list[dict] = []
    with rows_path.open("w") as handle:
        for entry in plan:
            row = generate_conversation(
                entry, scenarios[entry["scenario_id"]], cfg, generator, benign_cache,
                model_id=model_id, config_sha256=config_sha256, provenance=provenance,
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
    elapsed = time.time() - start

    manifest = _build_manifest(rows, cfg, model_id=model_id, config_sha256=config_sha256,
                               provenance=provenance, elapsed=elapsed, stub=bool(generator.is_stub))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _build_manifest(rows: list[dict], cfg: dict, *, model_id: str, config_sha256: str,
                    provenance: dict, elapsed: float, stub: bool) -> dict:
    lengths_by_arm: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        lengths_by_arm[row["arm"]].append(len(row["token_ids"]))
    token_length_stats = {
        arm: {
            "count": len(lengths),
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.fmean(lengths),
            "max": max(lengths),
        }
        for arm, lengths in sorted(lengths_by_arm.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": cfg["protocol"],
        "version": cfg["version"],
        "model_id": model_id,
        "stub": stub,
        "config_sha256": config_sha256,
        "sampling": _sampling_block(cfg),
        "num_conversations": len(rows),
        "per_arm_conversation_counts": {arm: len(v) for arm, v in sorted(lengths_by_arm.items())},
        "token_length_stats": token_length_stats,
        "elapsed_seconds": round(elapsed, 4),
        "git_provenance": provenance,
    }


def build_dry_run_report(cfg: dict, *, config_sha256: str, model_id: str, provenance: dict) -> dict:
    scenarios = _scenario_index(cfg)
    plan = plan_conversations(cfg)
    generator = StubGenerator()
    benign_cache: dict[tuple[str, int], dict] = {}

    seed_table: list[dict] = []
    for entry in plan:
        row = generate_conversation(
            entry, scenarios[entry["scenario_id"]], cfg, generator, benign_cache,
            model_id=model_id, config_sha256=config_sha256, provenance=provenance,
        )
        seed_table.append({
            "conversation_id": row["conversation_id"],
            "arm": row["arm"],
            "sample_index": row["sample_index"],
            "turns": [
                {
                    "turn_index": turn["turn_index"],
                    "turn_seed": turn["turn_seed"],
                    "prompt_prefix_sha256": turn["prompt_prefix_sha256"],
                    "shared_from": turn["shared_from"],
                }
                for turn in row["turns"]
            ],
        })

    max_new = int(cfg["sampling"]["max_new_tokens_per_turn"])
    prompt_chars = 0
    for entry in plan:
        scenario = scenarios[entry["scenario_id"]]
        prompt_chars += len(scenario["system"]) + sum(len(text) for text in entry["user_turns"])
    estimated_input_tokens = round(prompt_chars / 3.5)
    worst_case_output_tokens = len(plan) * 7 * max_new

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": cfg["protocol"],
        "version": cfg["version"],
        "config_sha256": config_sha256,
        "model_id": model_id,
        "sampling": _sampling_block(cfg),
        "num_conversations": len(plan),
        "per_arm_conversation_counts": dict(sorted(Counter(e["arm"] for e in plan).items())),
        "token_budget_estimate": {
            "method": "ESTIMATE only: input ~= prompt_chars / 3.5; output = conversations * 7 turns * max_new_tokens_per_turn (worst case)",
            "estimated_input_tokens": estimated_input_tokens,
            "worst_case_output_tokens": worst_case_output_tokens,
            "total_estimate_tokens": estimated_input_tokens + worst_case_output_tokens,
        },
        "seed_basis": (
            "stub_prefix_tokenization (model-free); a real --model-id run re-derives turn_seed and "
            "prompt_prefix_sha256 from the HF tokenizer and will differ. This report validates the "
            "plan/arm/sharing structure and the budget, not exact HF seeds."
        ),
        "seed_table": seed_table,
        "plan": [
            {
                "conversation_id": entry["conversation_id"],
                "family": entry["family"],
                "scenario_id": entry["scenario_id"],
                "arm": entry["arm"],
                "sample_index": entry["sample_index"],
                "num_user_turns": len(entry["user_turns"]),
            }
            for entry in plan
        ],
        "git_provenance": provenance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = load_natpress_config(config_path)
    config_sha256 = file_sha256(config_path)
    out_dir = Path(args.out)
    provenance = git_provenance([config_path])

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = build_dry_run_report(
            cfg, config_sha256=config_sha256, model_id=args.model_id, provenance=provenance
        )
        report_path = out_dir / "dry_run_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        budget = report["token_budget_estimate"]["total_estimate_tokens"]
        print(f"dry-run OK: {report['num_conversations']} conversations planned, "
              f"~{budget} token budget estimate (no model loaded) -> {report_path}")
        return 0

    generator = StubGenerator() if args.stub else _load_hf_generator(args.model_id, args.device)
    manifest = run_rollout(
        cfg, generator, out_dir, model_id=args.model_id,
        config_sha256=config_sha256, provenance=provenance, limit=args.limit,
    )
    label = "stub" if generator.is_stub else args.model_id
    print(f"wrote {manifest['num_conversations']} conversations ({label}) -> {out_dir/'rows.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
