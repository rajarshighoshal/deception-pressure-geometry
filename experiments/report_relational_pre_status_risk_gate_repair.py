"""Report fold-safe repair tests for the sealed pre-status risk gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.eval.relational_pre_status_risk_gate_repair import (  # noqa: E402
    build_relational_pre_status_risk_gate_repair_report,
    render_relational_pre_status_risk_gate_repair_markdown,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-report", type=Path, required=True)
    parser.add_argument("--expected-sealed-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sealed_report = args.sealed_report.resolve()
    out = args.out.resolve()
    markdown_out = args.markdown_out.resolve()
    if out == markdown_out or out.suffix != ".json":
        raise ValueError("outputs must be distinct and --out must be JSON")
    if out == sealed_report or markdown_out == sealed_report:
        raise ValueError("outputs must not overwrite the sealed report")
    effective_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    report = build_relational_pre_status_risk_gate_repair_report(
        sealed_report,
        expected_sealed_report_sha256=args.expected_sealed_report_sha256,
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
    _atomic_write(
        markdown_out,
        render_relational_pre_status_risk_gate_repair_markdown(report),
    )
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
