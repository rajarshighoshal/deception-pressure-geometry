"""Validate and report a completed source relational gauge causal replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    return parser.parse_args(argv)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    outputs = (args.out_json.resolve(), args.out_markdown.resolve())
    inputs = (args.rows.resolve(), args.manifest.resolve())
    if outputs[0] == outputs[1] or any(output in inputs for output in outputs):
        raise ValueError("report input and output paths must be distinct")
    from geoprobe.eval.relational_gauge_source_report import (  # noqa: E402
        build_relational_gauge_source_report,
        load_relational_gauge_source_run,
        render_relational_gauge_source_markdown,
    )

    manifest, rows = load_relational_gauge_source_run(args.rows, args.manifest)
    report = build_relational_gauge_source_report(
        manifest,
        rows,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_text(
        args.out_json,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_text(
        args.out_markdown,
        render_relational_gauge_source_markdown(report),
    )
    print(
        json.dumps(
            {
                "status": "success",
                "report_sha256": report["report_sha256"],
                "out_json": str(args.out_json.resolve()),
                "out_markdown": str(args.out_markdown.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
