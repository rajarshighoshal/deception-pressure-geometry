"""Adapt powered150 action-response rows for existing selector scripts.

The broad BF16 scorer writes a nested `action_response.rows` payload with
`abstain` rows and margin-field rewards.  The 4-bit selector stack expects a
generation-policy-like row table with a `baseline` row plus `strict_reward`,
`decision_margin`, and fix/harm fields.  This adapter performs only that schema
translation.  It does not select policies or run models.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geoprobe.control.action_response import (  # noqa: E402
    file_sha256,
    grouped_by_conversation,
    load_action_response,
    safe_float,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402


def forced_status_from_margin(value: float) -> str:
    return "PASS" if value > 0.0 else "FAIL"


def adapted_method(method: Any) -> str:
    aliases = {
        "abstain": "baseline",
        "global_mean": "global_mean_gated",
        "global_probe": "global_probe_gated",
        "random_global": "random_gated",
    }
    return aliases.get(str(method), str(method))


def adapt_row(row: dict) -> dict:
    status_class = str(row.get("status_class") or "")
    is_false = status_class.startswith("false_")
    correct_after = bool(row.get("correct_after"))
    fixes_error = bool(row.get("fixes_error"))
    harms_honest = bool(row.get("harms_honest"))
    final_margin = safe_float(row.get("final_margin"))
    base_margin = safe_float(row.get("base_margin"))
    target_status = row.get("target_status")
    target_sign = 1.0 if target_status == "PASS" else -1.0 if target_status == "FAIL" else 0.0
    out = dict(row)
    out.update({
        "method": adapted_method(row.get("method")),
        "original_method": str(row.get("method")),
        "decision_margin": final_margin,
        "decision_forced_status": forced_status_from_margin(final_margin),
        "status_correct": correct_after,
        "basis_strict_ok": correct_after,
        "parse_success": True,
        "fixes_status": fixes_error,
        "fixes_strict": fixes_error,
        "harms_honest_status": harms_honest,
        "harms_honest_strict": harms_honest,
        "status_reward": safe_float(row.get("reward")),
        "strict_reward": safe_float(row.get("reward")),
        "gate_score_PASS_minus_FAIL": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
        "mean_neighbor_distance": safe_float(row.get("neighbor_distance_mean"), -1.0),
        "injected_norm": abs(safe_float(row.get("alpha"))) if row.get("layer") is not None else 0.0,
        "per_layer_alpha": safe_float(row.get("alpha")),
        "target_aligned_decision_margin": target_sign * final_margin,
        "target_aligned_delta_margin": target_sign * (final_margin - base_margin),
        "response_field_note": "decision-token margin field; not full generation/audit",
    })
    if not is_false:
        out["fixes_status"] = False
        out["fixes_strict"] = False
    return out


def validate_adapted(rows: list[dict]) -> dict:
    grouped = grouped_by_conversation(rows)
    baseline_missing = [cid for cid, candidates in grouped.items() if not any(row["method"] == "baseline" for row in candidates)]
    if baseline_missing:
        raise ValueError(f"{len(baseline_missing)} conversations lack baseline rows, examples={baseline_missing[:5]}")
    return {
        "n_rows": len(rows),
        "n_conversations": len(grouped),
        "candidate_count_distribution": dict(Counter(len(v) for v in grouped.values())),
        "method_counts": dict(Counter(row["method"] for row in rows)),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if row["method"] == "baseline")),
        "family_balance": dict(Counter(row["family"] for row in rows if row["method"] == "baseline")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-response", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.action_response)
    rows, meta = load_action_response(src)
    adapted = [adapt_row(row) for row in rows]
    summary = validate_adapted(adapted)
    payload = {
        "schema_version": 1,
        "argv": sys.argv,
        "source_action_response": str(src.resolve()),
        "source_action_response_sha256": file_sha256(src),
        "source_meta": meta,
        "provenance": git_provenance([Path(__file__), src]),
        "summary": summary,
        "rows": adapted,
        "note": (
            "Adapter for existing 4-bit selector stack. `strict_reward` and "
            "`status_reward` are decision-token margin-field rewards, not audited "
            "generation strictness."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved -> {out}")
    print(summary)


if __name__ == "__main__":
    main()
