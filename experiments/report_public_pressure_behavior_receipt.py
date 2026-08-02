"""Public behavioral receipt: deceptive-commitment rates per pressure program.

Reads the frozen presented-bank rollout rows (private path supplied via CLI,
SHA-256 pinned below), aggregates committed-status outcomes per two-slot
pressure program, and emits a compact hash-bound public receipt. Descriptive
only; developmental bank; retrospective unregistered descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "paper_artifacts" / "pressure_behavior_receipt.json"
FROZEN_ROWS_SHA256 = (
    "ebd99699ec1fbcc93da22c4ce768bd2d04e4ccc6d1416ba82aa866fd79157492"
)
PROGRAMS = ("NN", "AN", "D2N", "AA", "AB", "BA")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-rows", type=Path, required=True,
                        help="path to the frozen presented-bank rows.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    live = sha256_file(args.rollout_rows)
    if live != FROZEN_ROWS_SHA256:
        raise SystemExit(
            f"SHA-256 mismatch for rollout rows (expected {FROZEN_ROWS_SHA256}, "
            f"got {live}); the frozen source has changed."
        )

    per_program: dict[str, Counter] = defaultdict(Counter)
    knowledge = Counter()
    total = 0
    for line in args.rollout_rows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        program = row["conversation_id"].split(":")[1]
        per_program[program][row["status_outcome"]] += 1
        knowledge[bool(row["knowledge_correct"])] += 1
        total += 1

    programs_out = {}
    for program in PROGRAMS:
        counts = per_program[program]
        n = sum(counts.values())
        programs_out[program] = {
            "n": n,
            "deceptive": counts.get("deceptive", 0),
            "honest": counts.get("honest", 0),
            "skip": counts.get("skip", 0),
            "deceptive_rate": counts.get("deceptive", 0) / n,
        }

    receipt = {
        "schema_version": 1,
        "kind": "pressure_behavior_public_receipt",
        "status": "unregistered_descriptive",
        "producer": "experiments/report_public_pressure_behavior_receipt.py",
        "producer_sha256": sha256_file(Path(__file__)),
        "source_artifact": {
            "sha256": FROZEN_ROWS_SHA256,
            "byte_size": args.rollout_rows.stat().st_size,
        },
        "chronology": {
            "tier": "retrospective_unregistered_descriptive",
            "note": (
                "Aggregation of the frozen presented-bank rollout outcomes; "
                "developmental bank, no confirmatory reading."
            ),
        },
        "population": {
            "conversations": total,
            "knowledge_correct": knowledge.get(True, 0),
            "knowledge_incorrect": knowledge.get(False, 0),
        },
        "programs": programs_out,
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
