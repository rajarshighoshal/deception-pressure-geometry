"""Learn/evaluate action selectors over audited decision-token generation rows.

This consumes a completed `control_graded_dp_decision_token.py` run plus its
`audit_decision_token_control.py` output. It does not run a model. The point is
to ask whether a held-out-family selector can choose better actions than fixed
tangent or a simple route-wise hybrid.
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

from geoprobe.control.policy_eval import (  # noqa: E402
    STATUS_CLASSES, baseline_row, build_policies, choose_learned, choose_margin_argmax, choose_method, choose_route_map, evaluate_fixed, evaluate_learned, evaluate_train_best, feature_dict, file_sha256, fit_model, grouped_by_conversation, merge_rows, paired_gap, predict_model, row_key, safe_float, score_action_on_train, slim_choice, summarize_choices,
)
from geoprobe.provenance import git_provenance  # noqa: E402
import geoprobe.control.policy_eval as policy_eval_lib  # noqa: E402  fingerprint the promoted library

__all__ = [
    "STATUS_CLASSES",
    "baseline_row",
    "build_policies",
    "choose_learned",
    "choose_margin_argmax",
    "choose_method",
    "choose_route_map",
    "evaluate_fixed",
    "evaluate_learned",
    "evaluate_train_best",
    "feature_dict",
    "file_sha256",
    "fit_model",
    "git_provenance",
    "grouped_by_conversation",
    "merge_rows",
    "paired_gap",
    "predict_model",
    "row_key",
    "safe_float",
    "score_action_on_train",
    "slim_choice",
    "summarize_choices",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--action-response", default=None,
                        help="Optional action-response JSON (from decision_token_action_response.py) "
                             "to propagate point-cloud context features into merged rows.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--gate-threshold-sweep", default=None,
                        help="Comma-separated gate-confidence thresholds for honest-row "
                             "abstention sweep. Rows with |gate_proba_PASS - 0.5| < threshold "
                             "are forced to abstain (honest rows only; deceptive rows keep "
                             "the gate prediction). Use to test whether selector selectivity "
                             "holds when honest rows cannot rely on near-perfect routing. "
                             "Example: 0.1,0.2,0.3,0.4")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_path = Path(args.results)
    audit_path = Path(args.audit)
    ar_rows = None
    if args.action_response:
        ar_path = Path(args.action_response)
        ar_rows = json.loads(ar_path.read_text()).get("rows", [])
    rows = merge_rows(
        json.loads(results_path.read_text()),
        json.loads(audit_path.read_text()),
        action_response_rows=ar_rows,
    )

    def _save_output(policies: dict, rows_used: list[dict], gate_threshold: float | None = None) -> dict:
        reference_names = ["fixed_bidir_tangent", "route_hybrid_mean_probe", "fixed_random_gated", "fixed_global_probe_gated", "learned_response_rf_strict"]
        gaps = {}
        for name, policy in policies.items():
            gaps[name] = {}
            for reference in reference_names:
                if name == reference:
                    continue
                gaps[name][reference] = {
                    m: paired_gap(policy["choices"], policies[reference]["choices"], m, seed=args.seed, bootstrap=args.bootstrap)
                    for m in ("status_fix", "strict_fix", "honest_status_harm")
                }
        o = {
            "threshold": args.threshold,
            "gate_threshold": gate_threshold,
            "n_candidate_rows": len(rows_used),
            "n_conversations": len(grouped_by_conversation(rows_used)),
            "family_balance": dict(Counter(row["family"] for row in rows_used if row["method"] == "baseline")),
            "status_class_balance": dict(Counter(row["status_class"] for row in rows_used if row["method"] == "baseline")),
            "policies": {
                name: {
                    "summary": policy["summary"],
                    "folds": policy.get("folds"),
                    "choices": [slim_choice(row) for row in policy["choices"]],
                }
                for name, policy in policies.items()
            },
            "paired_gaps": gaps,
        }
        return o

    policies = build_policies(rows, threshold=args.threshold)
    out = _save_output(policies, rows)
    out.update({
        "schema_version": 1,
        "argv": sys.argv,
        "results": str(results_path.resolve()),
        "results_sha256": file_sha256(results_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": file_sha256(audit_path),
        "provenance": git_provenance([Path(__file__), Path(policy_eval_lib.__file__), results_path, audit_path]),
    })
    gate_sweep = {}
    if args.gate_threshold_sweep:
        for thresh_str in args.gate_threshold_sweep.split(","):
            thresh = float(thresh_str.strip())
            swept_rows = []
            for row in rows:
                r = dict(row)
                if r["status_class"].startswith("honest_"):
                    gate_prob = float(r.get("gate_proba_PASS", 0.5))
                    if abs(gate_prob - 0.5) < thresh:
                        r["route_action"] = "abstain"
                        r["target_status"] = None
                swept_rows.append(r)
            swept_policies = build_policies(swept_rows, threshold=args.threshold)
            gate_sweep[str(thresh)] = _save_output(swept_policies, swept_rows, gate_threshold=thresh)
            for name in sorted(swept_policies):
                s = swept_policies[name]["summary"]
                print(
                    f"  gate_thresh={thresh:4.1f} {name:28s} status={s['deceptive_status_fixes']:2d}/{s['deceptive_n']} "
                    f"strict={s['deceptive_strict_fixes']:2d}/{s['deceptive_n']} "
                    f"harm={s['honest_status_harms']:2d}/{s['honest_n']}"
                )
    out["gate_sweep"] = gate_sweep
    out["note"] = (
        "CPU-only selector over completed generation/audit rows. Learned policies are "
        "leave-one-family-out. `learned_response_*` uses candidate decision-token margin "
        "features, which require a cheap action-response pass before generation. "
        "If `gate_sweep` is present, each threshold entry re-routes honest rows where "
        "|gate_proba_PASS - 0.5| < threshold to abstain."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    for name in sorted(policies):
        s = policies[name]["summary"]
        print(
            f"{name:28s} status={s['deceptive_status_fixes']:2d}/{s['deceptive_n']} "
            f"strict={s['deceptive_strict_fixes']:2d}/{s['deceptive_n']} "
            f"harm={s['honest_status_harms']:2d}/{s['honest_n']} "
            f"methods={s['chosen_methods']}"
        )

if __name__ == "__main__":
    main()
