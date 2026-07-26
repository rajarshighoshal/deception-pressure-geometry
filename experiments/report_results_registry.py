"""Validate docs/results_registry.yaml and regenerate the claim table in README.md.

Thin CLI over geoprobe.registry: no logic here. Edit the registry, then run this to refresh the
block between the generated markers in README.md. `--check` validates and verifies the block is
in sync without writing (for CI / tests), exiting non-zero on drift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.registry import (  # noqa: E402
    inject_generated_block,
    load_registry,
    render_claim_table,
    validate_registry,
)

DEFAULT_REGISTRY = REPO_ROOT / "docs" / "results_registry.yaml"
DEFAULT_OUT = REPO_ROOT / "README.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check", action="store_true",
        help="Validate and verify the generated block is in sync; write nothing. Exit 1 on drift.",
    )
    args = parser.parse_args()

    reg = load_registry(args.registry)
    errors = validate_registry(reg)
    if errors:
        print(f"registry INVALID ({args.registry}):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    doc = args.out.read_text()
    updated = inject_generated_block(doc, render_claim_table(reg))

    if args.check:
        if updated != doc:
            print(f"{args.out} is OUT OF SYNC with {args.registry}; "
                  f"run experiments/report_results_registry.py.", file=sys.stderr)
            return 1
        print(f"registry valid; {args.out.name} in sync.")
        return 0

    args.out.write_text(updated)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
