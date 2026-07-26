"""Bidirectional d_p diagnostic for graded deception-control.

The pooled d_p controller can silently become a one-way label pusher when the
deceptive pool is imbalanced, e.g. mostly true-PASS rows falsely reported as
FAIL.  This diagnostic splits the correction directions by error type:

  v_to_PASS = mean(honest_PASS) - mean(false_FAIL)
  v_to_FAIL = mean(honest_FAIL) - mean(false_PASS)

Directions are fit from train families only and within matched scenario-level
groups, so a held-out family cannot leak into its own direction.  The output
answers whether the current data has enough bidirectional support and whether
the old pooled honest-vs-deceptive vector is really just a PASS correction.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_graded_dp_frontier import (  # noqa: E402
    load_activation_points,
    read_jsonl_paths,
)
from geoprobe.io import file_sha256  # noqa: E402
from geoprobe.control.directions import (  # noqa: E402,F401  re-export moved direction-fitting
    attach_status_classes, fit_status_direction, status_error_class, summarize_family_directions,
)
import geoprobe.control.directions as _directions_mod  # noqa: E402  fingerprint promoted libraries
import geoprobe.geometry.tangent as _tangent_mod  # noqa: E402
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402
from experiments.control_graded_dp_stack_frontier import parse_csv  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402


STATUS_CLASSES = ("honest_PASS", "honest_FAIL", "false_FAIL", "false_PASS")


def to_jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
            if not str(k).startswith("_") and str(k) != "direction"
        }
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def transcript_index(paths: list[Path]) -> dict[str, dict]:
    rows = read_jsonl_paths(paths)
    out: dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("conversation_id", ""))
        if not cid:
            continue
        if cid in out:
            raise ValueError(f"duplicate transcript conversation_id {cid}")
        out[cid] = row
    return out


def count_table(rows: list[dict], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(key, ""))][str(row["status_class"])] += 1
    return {name: dict(counts) for name, counts in sorted(table.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20,24,28,32")
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--min-mixed-scenarios", type=int, default=2)
    parser.add_argument("--min-levels", type=int, default=2)
    args = parser.parse_args()

    activation_path = Path(args.activations)
    transcript_paths = [Path(path) for path in args.transcripts]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transcripts = transcript_index(transcript_paths)
    direction_levels = set(parse_csv(args.direction_levels))
    layers = [int(item) for item in parse_csv(args.layers)]

    by_layer: dict[str, dict] = {}
    activation_meta = None
    for layer in layers:
        points, meta = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.direction_turn},
            phases={args.direction_phase},
            levels=direction_levels,
        )
        activation_meta = activation_meta or meta
        rows, skipped = attach_status_classes(points, transcripts)
        class_counts = Counter(row["status_class"] for row in rows)
        status_by_family = count_table(rows, "family")
        status_by_level = count_table(rows, "arm")
        by_layer[str(layer)] = {
            "n_points": int(len(points)),
            "n_status_rows": int(len(rows)),
            "skipped": dict(skipped),
            "status_counts": {name: int(class_counts.get(name, 0)) for name in STATUS_CLASSES},
            "status_by_family": status_by_family,
            "status_by_level": status_by_level,
            "directions": summarize_family_directions(
                rows,
                direction_levels=direction_levels,
                min_mixed_scenarios=args.min_mixed_scenarios,
                min_levels=args.min_levels,
            ),
        }

    payload = {
        "provenance": {
            "git": git_provenance([Path(__file__), Path(_directions_mod.__file__), Path(_tangent_mod.__file__), Path(_bank_mod.__file__)]),
            "argv": sys.argv,
            "activations": str(activation_path),
            "activations_sha256": file_sha256(activation_path),
            "transcripts": [str(path) for path in transcript_paths],
            "transcripts_sha256": {str(path): file_sha256(path) for path in transcript_paths},
        },
        "config": {
            "layers": layers,
            "direction_turn": args.direction_turn,
            "direction_phase": args.direction_phase,
            "direction_levels": sorted(direction_levels),
            "min_mixed_scenarios": args.min_mixed_scenarios,
            "min_levels": args.min_levels,
        },
        "activation_meta": activation_meta,
        "layers": by_layer,
        "interpretation_note": (
            "If pooled-vs-to_PASS is high while pooled-vs-to_FAIL is low or negative, "
            "the old pooled d_p direction is likely a one-way PASS/FAIL label correction, "
            "not a bidirectional honesty-restoration vector. A deployable controller must "
            "fit/select error-type-specific directions or prove a shared honesty direction."
        ),
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    for layer, info in by_layer.items():
        dirs = info["directions"]
        print(
            f"L{layer}: counts={info['status_counts']} "
            f"families to_PASS={dirs['n_to_PASS_available']}/{dirs['n_families']} "
            f"to_FAIL={dirs['n_to_FAIL_available']}/{dirs['n_families']} "
            f"global cos pass/fail={dirs['global_cos_to_PASS_vs_to_FAIL']}"
        )


if __name__ == "__main__":
    main()
