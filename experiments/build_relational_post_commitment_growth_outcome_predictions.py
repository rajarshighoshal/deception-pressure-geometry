"""Build one held-out fold prediction ledger for post-commitment growth outcomes."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FOLDS = ("outer_1", "outer_2", "outer_3", "outer_4", "outer_5")


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
    parser.add_argument("--held-out-family-fold", choices=FOLDS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roster-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    from geoprobe.eval import (  # noqa: E402
        relational_post_commitment_growth_outcome_projection as projection,
    )
    from geoprobe.provenance import git_provenance  # noqa: E402

    out = args.out.resolve()
    roster_out = args.roster_out.resolve() if args.roster_out is not None else None
    if out.suffix != ".json":
        raise ValueError("--out must use .json")
    if roster_out is not None and roster_out == out:
        raise ValueError("--out and --roster-out must be distinct")

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
    ledger = projection.build_relational_post_commitment_growth_fold_predictions(
        prepared,
        held_out_family_fold=args.held_out_family_fold,
        argv=run_argv,
        provenance=provenance,
    )

    if roster_out is not None:
        _write_json(roster_out, prepared.roster)
    _write_json(out, ledger)
    print(
        json.dumps(
            {
                "status": "success",
                "ledger": str(out),
                "roster": str(roster_out) if roster_out is not None else None,
                "held_out_family_fold": ledger.get("held_out_family_fold"),
                "query_unique_event_count": ledger.get("query_unique_event_count"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
