"""Render an auditable Markdown report from powered150 selector-eval JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.control.structural_candidates import structural_candidate_records  # noqa: E402


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def rank_key(item: tuple[str, dict]) -> tuple[float, float, float, str]:
    name, policy = item
    s = policy["summary"]
    deceptive_n = max(safe_float(s.get("deceptive_n")), 1.0)
    honest_n = max(safe_float(s.get("honest_n")), 1.0)
    return (
        safe_float(s.get("mean_reward")),
        safe_float(s.get("fixes_error")) / deceptive_n,
        -safe_float(s.get("honest_harms")) / honest_n,
        name,
    )


def render_markdown(payload: dict, *, top_n: int) -> str:
    top = sorted(payload["policies"].items(), key=rank_key, reverse=True)[:top_n]
    lines = [
        "# Powered150 Selector Evaluation",
        "",
        "CPU-only comparison over the powered150 BF16 decision-token action-response field.",
        "This is not full generation/audit.",
        "",
        "## Inputs",
        "",
        f"- Action response: `{payload['action_response']}`",
        f"- Action-response SHA256: `{payload['action_response_sha256']}`",
        f"- Activations: `{payload.get('activations')}`",
        f"- Typed-graph import: `{payload.get('typed_graph')}`",
        f"- Raw rows: `{payload['n_rows']}`",
        f"- Conversations: `{payload['n_conversations']}`",
        f"- Adapted selector rows: `{payload['adapted_summary']['n_rows']}`",
        f"- Structural families: `{', '.join(payload.get('structural_families') or []) or 'none'}`",
        "",
        "## Class Balance",
        "",
        "```json",
        json.dumps(payload["status_class_balance"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top Policies By Mean Reward",
        "",
        "| policy | type | fixes | harm | mean_reward | aligned_margin | chosen methods |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, item in top:
        s = item["summary"]
        methods = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("chosen_methods", {}).items()))
        policy_type = payload["policy_types"].get(name, "unknown")
        lines.append(
            f"| `{name}` | {policy_type} | {s['fixes_error']}/{s['deceptive_n']} | "
            f"{s['honest_harms']}/{s['honest_n']} | {safe_float(s['mean_reward']):.4f} | "
            f"{safe_float(s['mean_aligned_margin']):.4f} | {methods} |"
        )
    lines.extend([
        "",
        "## Comparison References",
        "",
        "```json",
        json.dumps(payload["comparison_references"], indent=2),
        "```",
        "",
        "## Proposed Structural Candidates Not Run In This Artifact",
        "",
        "These are design candidates only. They are not evidence and should not enter result claims.",
        "",
        "| candidate | status | family | first eval | why add it |",
        "|---|---|---|---|---|",
    ])
    default_candidates = structural_candidate_records()
    for item in payload.get("next_structural_candidates_not_run", default_candidates):
        lines.append(
            f"| `{item['name']}` | {item['status']} | {item.get('family', '')} | "
            f"{item.get('first_eval_mode', '')} | {item['reason']} |"
        )
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- Response-aware and oracle policies are ceilings/probes, not context-only structural claims.",
        "- Source sweep used oracle/feasibility routing.",
        "- Existing 4-bit selector classes are reused through the powered150 adapter.",
        "- `strict_reward` in adapted rows is decision-margin reward, not audited generation strictness.",
        "- Full generation/audit is required before strict/coherence claims.",
        "- Paired gaps in JSON are clustered by `scenario_id`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    Path(args.out).write_text(render_markdown(payload, top_n=args.top_n))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
