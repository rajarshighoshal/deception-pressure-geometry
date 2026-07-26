"""Build the five-fold relational gauge-controller substrate and report."""

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
    parser.add_argument("--expected-rooted-graph-manifest-sha256", required=True)
    parser.add_argument("--outcome-shard-root", type=Path, required=True)
    parser.add_argument("--expected-outcome-shard-manifest-sha256", required=True)
    parser.add_argument("--expected-source-outcome-report-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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
    outcome_shard_root = args.outcome_shard_root.resolve()
    out_dir = args.out_dir.resolve()
    input_roots = (bank_root, graph_root, outcome_shard_root)
    if any(out_dir == root or out_dir.is_relative_to(root) for root in input_roots):
        raise ValueError("--out-dir must not overwrite or enter an input artifact")

    from geoprobe.data.relational_pre_status_rooted_star_store import (  # noqa: E402
        build_relational_pre_status_rooted_star_index,
    )
    from geoprobe.eval.relational_gauge_controller_artifact import (  # noqa: E402
        build_relational_gauge_controller_artifact,
        build_relational_gauge_controller_report,
        render_relational_gauge_controller_markdown,
    )
    from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (  # noqa: E402
        load_relational_pre_status_rooted_graph_artifacts,
    )

    index = build_relational_pre_status_rooted_star_index(bank_root)
    graphs = load_relational_pre_status_rooted_graph_artifacts(graph_root)
    manifest = build_relational_gauge_controller_artifact(
        index,
        graphs,
        artifact_root=out_dir,
        expected_rooted_star_manifest_sha256=(
            args.expected_rooted_star_manifest_sha256
        ),
        rooted_graph_artifact_root=graph_root,
        expected_rooted_graph_manifest_file_sha256=(
            args.expected_rooted_graph_manifest_sha256
        ),
        outcome_shard_root=outcome_shard_root,
        expected_outcome_shard_manifest_file_sha256=(
            args.expected_outcome_shard_manifest_sha256
        ),
        expected_source_report_file_sha256=(
            args.expected_source_outcome_report_sha256
        ),
        argv=effective_argv,
        extra_source_paths=[Path(__file__).resolve()],
    )
    report = build_relational_gauge_controller_report(manifest)
    report_path = out_dir / "report.json"
    markdown_path = out_dir / "report.md"
    _atomic_write(
        report_path,
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
        markdown_path,
        render_relational_gauge_controller_markdown(report),
    )
    print(
        json.dumps(
            {
                "status": "success",
                "manifest": str(out_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "report": str(report_path),
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
