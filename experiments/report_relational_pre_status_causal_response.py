"""Render the immutable, hash-bound pre-status causal response diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geoprobe.eval.relational_pre_status_causal_response import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REGISTERED_CAUSAL_REPORT,
    DEFAULT_RUNNER_JSONL,
    DEFAULT_SOURCE_MANIFEST,
    build_relational_pre_status_causal_response_report,
    render_relational_pre_status_causal_response_report,
    REPORT_JSON_FILENAME,
    REPORT_MARKDOWN_FILENAME,
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-jsonl", type=Path, default=DEFAULT_RUNNER_JSONL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--registered-causal-report", type=Path, default=DEFAULT_REGISTERED_CAUSAL_REPORT
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR / REPORT_JSON_FILENAME)
    parser.add_argument(
        "--markdown-out", type=Path, default=DEFAULT_OUTPUT_DIR / REPORT_MARKDOWN_FILENAME
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    out = args.out.resolve()
    markdown_out = args.markdown_out.resolve()
    if out == markdown_out:
        raise ValueError("JSON and Markdown outputs must be distinct")

    canonical_out = (DEFAULT_OUTPUT_DIR / REPORT_JSON_FILENAME).resolve()
    canonical_markdown = (DEFAULT_OUTPUT_DIR / REPORT_MARKDOWN_FILENAME).resolve()
    if out != canonical_out or markdown_out != canonical_markdown:
        raise ValueError(
            "report outputs must use the canonical locations "
            f"{canonical_out} and {canonical_markdown}"
        )

    runner_jsonl = args.runner_jsonl.resolve()
    source_manifest = args.source_manifest.resolve()
    registered_causal_report = (
        args.registered_causal_report.resolve()
        if args.registered_causal_report is not None
        else None
    )
    if out.suffix != ".json":
        raise ValueError("--out must use .json")
    if markdown_out.suffix != ".md":
        raise ValueError("--markdown-out must use .md")

    inputs = {runner_jsonl, source_manifest}
    if registered_causal_report is not None:
        inputs.add(registered_causal_report)
    if out in inputs or markdown_out in inputs:
        raise ValueError("outputs must not overwrite inputs")
    if out.exists() or markdown_out.exists():
        raise FileExistsError("outputs are immutable and must not already exist")

    effective_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    report = build_relational_pre_status_causal_response_report(
        runner_jsonl,
        source_manifest,
        registered_causal_report_path=registered_causal_report,
        argv=effective_argv,
        extra_source_paths=[Path(__file__).resolve()],
    )

    _atomic_write(
        out,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
    _atomic_write(markdown_out, render_relational_pre_status_causal_response_report(report))

    print(
        json.dumps(
            {
                "status": "success",
                "json": str(out),
                "markdown": str(markdown_out),
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
