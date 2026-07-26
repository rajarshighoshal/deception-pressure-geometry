"""Structured chart-atlas selector over activation states.

This is the first CPU-side "proper atlas" controller:

1. load the actual decision/query activation vector for each conversation;
2. fit overlapping local charts on train-family states;
3. fit a chart-local ridge reward head over candidate actions;
4. choose the action with the best chart-weighted predicted strict reward.

Unlike the kNN atlas, this does not pick a fixed neighborhood size at test time.
The local scale comes from soft chart memberships in activation space.  Response
variants include candidate decision-token margins; context variants do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.text.parse import parse_csv, parse_int_csv  # noqa: E402
from geoprobe.control.atlas_action_selector import (  # noqa: E402,F401  promoted library (distinct from ChartAtlas)
    AtlasActionSelector, action_features, action_key, best_policy_name, build_chart_policies,
    candidate_rows, choose_chart, evaluate_chart_atlas, parse_chart_counts, slim_chart_choice, state_key,
)
from geoprobe.data.activation_bank import load_state_vectors  # noqa: E402,F401
ChartAtlasSelector = AtlasActionSelector  # compat alias for the old import path
from experiments.learned_generation_action_policy import (  # noqa: E402
    build_policies,
    file_sha256,
    grouped_by_conversation,
    merge_rows,
    paired_gap,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
import geoprobe.control.atlas_action_selector as _atlas_mod  # noqa: E402  fingerprint promoted selector
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--action-response", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--chart-counts", default="auto")
    parser.add_argument("--pca-dim", type=int, default=24)
    parser.add_argument("--top-charts", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-chart-support", type=float, default=5.0)
    parser.add_argument("--min-action-support", type=float, default=1.0)
    parser.add_argument("--fallbacks", default="full,method_layer,method,route")
    parser.add_argument("--heads", default="mean,ridge")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_path = Path(args.results)
    audit_path = Path(args.audit)
    activation_path = Path(args.activations)
    ar_path = Path(args.action_response) if args.action_response else None
    ar_rows = json.loads(ar_path.read_text()).get("rows", []) if ar_path else None
    rows = merge_rows(
        json.loads(results_path.read_text()),
        json.loads(audit_path.read_text()),
        action_response_rows=ar_rows,
    )
    layers = parse_int_csv(args.layers)
    state_vectors, activation_meta = load_state_vectors(
        activation_path,
        layers=layers,
        query_turn=args.query_turn,
        query_phase=args.query_phase,
    )
    baselines = build_policies(rows, threshold=args.threshold)
    chart_policies = build_chart_policies(
        rows,
        state_vectors=state_vectors,
        chart_counts=parse_chart_counts(args.chart_counts),
        pca_dim=args.pca_dim,
        top_charts=args.top_charts,
        ridge_alpha=args.ridge_alpha,
        threshold=args.threshold,
        min_chart_support=args.min_chart_support,
        min_action_support=args.min_action_support,
        fallbacks=parse_csv(args.fallbacks),
        heads=parse_csv(args.heads),
        seed=args.seed,
    )
    policies = {**baselines, **chart_policies}
    references = [
        "fixed_bidir_tangent",
        "route_hybrid_mean_probe",
        "learned_response_rf_strict",
        "fixed_random_gated",
        "fixed_global_probe_gated",
        "margin_argmax_all",
    ]
    gaps = {}
    for name, policy in policies.items():
        gaps[name] = {}
        for ref in references:
            if name == ref or ref not in policies:
                continue
            gaps[name][ref] = {
                "status_fix": paired_gap(
                    policy["choices"], policies[ref]["choices"], "status_fix",
                    seed=args.seed, bootstrap=args.bootstrap,
                ),
                "strict_fix": paired_gap(
                    policy["choices"], policies[ref]["choices"], "strict_fix",
                    seed=args.seed, bootstrap=args.bootstrap,
                ),
                "honest_status_harm": paired_gap(
                    policy["choices"], policies[ref]["choices"], "honest_status_harm",
                    seed=args.seed, bootstrap=args.bootstrap,
                ),
            }
    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "results": str(results_path.resolve()),
        "results_sha256": file_sha256(results_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": file_sha256(audit_path),
        "activations": str(activation_path.resolve()),
        "activations_sha256": file_sha256(activation_path),
        "activation_meta": activation_meta,
        "action_response": str(ar_path.resolve()) if ar_path else None,
        "action_response_sha256": file_sha256(ar_path) if ar_path else None,
        "provenance": git_provenance([Path(__file__), Path(_atlas_mod.__file__), Path(_bank_mod.__file__), results_path, audit_path, activation_path, *( [ar_path] if ar_path else [] )]),
        "layers": layers,
        "query_turn": args.query_turn,
        "query_phase": args.query_phase,
        "chart_counts": parse_chart_counts(args.chart_counts),
        "pca_dim": args.pca_dim,
        "top_charts": args.top_charts,
        "ridge_alpha": args.ridge_alpha,
        "threshold": args.threshold,
        "min_chart_support": args.min_chart_support,
        "min_action_support": args.min_action_support,
        "fallbacks": parse_csv(args.fallbacks),
        "heads": parse_csv(args.heads),
        "n_candidate_rows": len(rows),
        "n_conversations": len(grouped_by_conversation(rows)),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if row["method"] == "baseline")),
        "best_chart_policy": best_policy_name(chart_policies),
        "policies": {
            name: {
                "summary": policy["summary"],
                "folds": policy.get("folds"),
                "choices": [slim_chart_choice(row) for row in policy["choices"]],
            }
            for name, policy in policies.items()
        },
        "paired_gaps": gaps,
        "note": (
            "Chart selectors fit overlapping soft charts directly on activation vectors. "
            "Context variants exclude candidate decision-token margins; response variants include them."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    for name in sorted(chart_policies):
        s = chart_policies[name]["summary"]
        marker = "*" if name == out["best_chart_policy"] else " "
        print(
            f"{marker} {name:<38} "
            f"status={s['deceptive_status_fixes']:>2}/{s['deceptive_n']} "
            f"strict={s['deceptive_strict_fixes']:>2}/{s['deceptive_n']} "
            f"harm={s['honest_status_harms']:>2}/{s['honest_n']} "
            f"methods={dict(s['chosen_methods'])}"
        )


if __name__ == "__main__":
    main()
