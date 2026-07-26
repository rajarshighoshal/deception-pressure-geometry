"""Report hash-bound offline evaluations of pre-status risk and honestward fields."""

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
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--expected-rooted-star-manifest-sha256", required=True)
    parser.add_argument("--rooted-graph-artifact-root", type=Path, required=True)
    parser.add_argument("--outcome-report", type=Path, required=True)
    parser.add_argument("--expected-outcome-report-sha256", required=True)
    parser.add_argument("--frozen-orbit-roster", type=Path, required=True)
    parser.add_argument("--expected-frozen-orbit-roster-sha256", required=True)
    parser.add_argument("--expected-outcome-event-count", type=int, required=True)
    parser.add_argument("--expected-forward-edge-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    effective_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    bank_root = args.bank_root.resolve()
    graph_root = args.rooted_graph_artifact_root.resolve()
    outcome_report = args.outcome_report.resolve()
    roster = args.frozen_orbit_roster.resolve()
    out = args.out.resolve()
    markdown_out = args.markdown_out.resolve()
    inputs = {
        bank_root / "manifest.json",
        graph_root / "manifest.json",
        outcome_report,
        roster,
    }
    outputs = {out, markdown_out}
    if len(outputs) != 2:
        raise ValueError("--out and --markdown-out must be different paths")
    if outputs & inputs:
        raise ValueError("report outputs must not overwrite bound inputs")
    if out.suffix != ".json":
        raise ValueError("--out must use a .json suffix")
    if out.is_dir() or markdown_out.is_dir():
        raise ValueError("report outputs must be file paths")

    from geoprobe.data.relational_pre_status_rooted_star_store import (  # noqa: E402
        build_relational_pre_status_rooted_star_index,
    )
    from geoprobe.eval.relational_pre_status_field_report import (  # noqa: E402
        build_relational_pre_status_field_report,
        render_relational_pre_status_field_markdown,
    )
    from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (  # noqa: E402
        load_relational_pre_status_rooted_graph_artifacts,
    )
    from geoprobe.eval.relational_pre_status_supervision import (  # noqa: E402
        build_relational_pre_status_supervision,
    )

    index = build_relational_pre_status_rooted_star_index(bank_root)
    graphs = load_relational_pre_status_rooted_graph_artifacts(graph_root)
    supervision = build_relational_pre_status_supervision(
        index,
        outcome_report_path=outcome_report,
        expected_outcome_report_sha256=args.expected_outcome_report_sha256,
        roster_path=roster,
        expected_roster_sha256=args.expected_frozen_orbit_roster_sha256,
        expected_outcome_event_count=args.expected_outcome_event_count,
        expected_forward_edge_count=args.expected_forward_edge_count,
    )
    report = build_relational_pre_status_field_report(
        index,
        graphs,
        supervision,
        expected_rooted_star_manifest_sha256=args.expected_rooted_star_manifest_sha256,
        rooted_graph_artifact_root=graph_root,
        outcome_report_path=outcome_report,
        roster_path=roster,
        argv=effective_argv,
        extra_source_paths=[Path(__file__).resolve()],
    )
    _atomic_write(
        out,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    _atomic_write(markdown_out, render_relational_pre_status_field_markdown(report))
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
