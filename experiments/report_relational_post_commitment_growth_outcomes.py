"""Score post-commitment growth outcome projection prediction ledgers."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--state-graph-root", type=Path, required=True)
    parser.add_argument("--state-manifest-sha256", required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--prediction-ledger",
        action="append",
        required=True,
        type=Path,
        help="repeat exactly five times",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"prediction ledger must be a JSON object: {path}")
    return value


def _assert_distinct_prediction_ledgers(values: Sequence[Path]) -> None:
    resolved = [value.resolve() for value in values]
    if len(resolved) != 5:
        raise ValueError("--prediction-ledger must be provided exactly five times")
    if len(set(resolved)) != len(resolved):
        raise ValueError("prediction-ledger paths must be unique")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    from geoprobe.eval import (  # noqa: E402
        relational_post_commitment_growth_outcome_projection as projection,
    )
    from geoprobe.provenance import git_provenance  # noqa: E402

    out_json = args.out_json.resolve()
    out_markdown = args.out_markdown.resolve()
    if out_json.suffix != ".json":
        raise ValueError("--out-json must use .json")
    if out_json == out_markdown:
        raise ValueError("--out-json and --out-markdown must be distinct")

    _assert_distinct_prediction_ledgers(args.prediction_ledger)
    resolved_prediction_ledgers = [path.resolve() for path in args.prediction_ledger]
    prediction_ledgers = [_read_json(path) for path in resolved_prediction_ledgers]

    prepared = projection.prepare_relational_post_commitment_growth_outcome_projection(
        bank_root=args.bank_root.resolve(),
        candidate_path=args.candidate_path.resolve(),
        expected_candidate_sha256=args.candidate_sha256,
        graph_path=args.graph_path.resolve(),
        expected_graph_sha256=args.graph_sha256,
        state_graph_root=args.state_graph_root.resolve(),
        expected_state_manifest_sha256=args.state_manifest_sha256,
        protocol_path=args.protocol_path.resolve(),
        expected_protocol_sha256=args.protocol_sha256,
    )
    run_argv = list(sys.argv[1:] if argv is None else argv)
    provenance = git_provenance(
        [Path(__file__).resolve(), Path(projection.__file__).resolve()]
    )
    report = projection.score_relational_post_commitment_growth_outcomes(
        prepared,
        prediction_ledgers=prediction_ledgers,
        argv=run_argv,
        provenance=provenance,
    )

    _write_json(out_json, report)
    _atomic_write(
        out_markdown,
        projection.render_relational_post_commitment_growth_outcome_markdown(report),
    )
    print(
        json.dumps(
            {
                "status": "success",
                "json": str(out_json),
                "markdown": str(out_markdown),
                "held_out_fold_count": len(prediction_ledgers),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
