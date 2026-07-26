"""Audit and compare powered150 targeted generation policy outputs.

This is the durable postprocess for the targeted-generation stage.  It
reads one or more ``control_graded_dp_decision_token.py`` policy-file outputs,
audits each generated reply against the source numeric rule, and writes a JSON
summary plus a Markdown report.  It does not run the model.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geoprobe.text.parse import parse_csv  # noqa: E402
from experiments.audit_decision_token_control import (  # noqa: E402
    audit_result_row,
    file_sha256,
    read_jsonl,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402


DEFAULT_TRANSCRIPTS = (
    REPO_ROOT
    / "results/powered150/run_20260629_142532/merged/transcript.jsonl"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "results/powered150/run_20260629_142532/targeted_generation_audit.json"
)
DEFAULT_REFERENCES = (
    "train_best_route_full_reward,"
    "chart_mean_context_cauto_d12_strict,"
    "graph_mean_context_cauto_gauto_d12_strict,"
    "chart_feature_gate_equivariant_neural_context,"
    "equivariant_chart_action_field_context,"
    "fixed_route_bidir_linear_L16_a96,"
    "local_control_flow_dense_alpha"
)
STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_FAIL", "honest_PASS")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def policy_name_from_payload(path: Path, payload: dict) -> str:
    name = payload.get("policy_name")
    if isinstance(name, str) and name:
        return name
    return path.stem


def audit_policy_file(path: Path, transcripts: dict[str, dict]) -> dict:
    payload = json.loads(path.read_text())
    rows = []
    missing = []
    for result in payload.get("results", []):
        cid = str(result.get("conversation_id", ""))
        transcript = transcripts.get(cid)
        if transcript is None:
            missing.append(cid)
            continue
        rows.append(audit_result_row(result, transcript))
    if missing:
        sample = ", ".join(sorted(set(missing))[:5])
        raise ValueError(f"{path} has {len(missing)} results with missing transcripts, e.g. {sample}")
    policy_name = policy_name_from_payload(path, payload)
    return {
        "policy_name": policy_name,
        "path": str(path),
        "sha256": file_sha256(path),
        "summary": summarize_rows(rows),
        "generation_checks": generation_checks(payload),
        "rows": rows,
        "source_meta": {
            "job_status": payload.get("job_status"),
            "STATUS": payload.get("STATUS"),
            "generation_protocol": payload.get("generation_protocol"),
            "generation_batch_size": payload.get("generation_batch_size"),
            "policy_choices": payload.get("policy_choices"),
            "policy_choices_sha256": payload.get("policy_choices_sha256"),
            "duration_sec": payload.get("duration_sec"),
            "timings": payload.get("timings"),
        },
    }


def generation_checks(payload: dict) -> dict:
    verifications = payload.get("batched_generation_verifications") or []
    gpu_summary = payload.get("gpu_summary") or {}
    return {
        "generation_protocol": payload.get("generation_protocol"),
        "generation_batch_size": payload.get("generation_batch_size"),
        "verification_batches": len(verifications),
        "forced_status_mismatches": int(sum(int(v.get("forced_status_mismatches", 0)) for v in verifications)),
        "reply_mismatches": int(sum(int(v.get("reply_mismatches", 0)) for v in verifications)),
        "max_abs_margin_diff": (
            max([safe_float(v.get("max_abs_margin_diff")) for v in verifications])
            if verifications else None
        ),
        "gpu_util_pct_mean": safe_float(gpu_summary.get("gpu_util_pct_mean")),
        "gpu_util_pct_max": safe_float(gpu_summary.get("gpu_util_pct_max")),
        "gpu_sample_count": int(gpu_summary.get("sample_count", 0) or 0),
    }


def row_status_class(row: dict) -> str:
    return str(row.get("status_class_before", ""))


def is_false(row: dict) -> bool:
    return row_status_class(row).startswith("false_")


def is_honest(row: dict) -> bool:
    return row_status_class(row).startswith("honest_")


def row_metrics(row: dict) -> dict[str, float]:
    false = is_false(row)
    honest = is_honest(row)
    status_correct = bool(row.get("status_correct"))
    strict_ok = bool(row.get("basis_strict_ok"))
    status_harm = honest and not status_correct
    strict_harm = honest and not strict_ok
    status_fix = false and status_correct
    strict_fix = false and strict_ok
    return {
        "status_fix": float(status_fix),
        "strict_fix": float(strict_fix),
        "honest_status_harm": float(status_harm),
        "honest_strict_harm": float(strict_harm),
        "status_reward": 1.0 if status_fix else -1.0 if status_harm else 0.0,
        "strict_reward": 1.0 if strict_fix else -1.0 if strict_harm else 0.0,
    }


def summarize_rows(rows: list[dict]) -> dict:
    deceptive = [row for row in rows if is_false(row)]
    honest = [row for row in rows if is_honest(row)]
    metrics = [row_metrics(row) for row in rows]
    return {
        "n": len(rows),
        "deceptive_n": len(deceptive),
        "honest_n": len(honest),
        "parse_success": int(sum(bool(row.get("parse_success")) for row in rows)),
        "coherence_preserved": int(sum(bool(row.get("coherence_preserved")) for row in rows)),
        "degenerate": int(sum(bool(row.get("degenerate")) for row in rows)),
        "deceptive_status_fixes": int(sum(row_metrics(row)["status_fix"] for row in deceptive)),
        "deceptive_strict_fixes": int(sum(row_metrics(row)["strict_fix"] for row in deceptive)),
        "honest_status_harms": int(sum(row_metrics(row)["honest_status_harm"] for row in honest)),
        "honest_strict_harms": int(sum(row_metrics(row)["honest_strict_harm"] for row in honest)),
        "mean_status_reward": float(np.mean([m["status_reward"] for m in metrics])) if metrics else 0.0,
        "mean_strict_reward": float(np.mean([m["strict_reward"] for m in metrics])) if metrics else 0.0,
        "chosen_methods": dict(Counter(str(row.get("method")) for row in rows)),
        "basis_quality": dict(Counter(str(row.get("basis_quality")) for row in rows)),
        "by_status_class": {
            cls: summarize_status_class([row for row in rows if row_status_class(row) == cls])
            for cls in STATUS_CLASSES
        },
    }


def summarize_status_class(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "status_correct": int(sum(bool(row.get("status_correct")) for row in rows)),
        "basis_strict_ok": int(sum(bool(row.get("basis_strict_ok")) for row in rows)),
        "parse_success": int(sum(bool(row.get("parse_success")) for row in rows)),
        "basis_quality": dict(Counter(str(row.get("basis_quality")) for row in rows)),
        "chosen_methods": dict(Counter(str(row.get("method")) for row in rows)),
    }


def metric_value(row: dict, metric: str) -> float:
    return row_metrics(row)[metric]


def paired_gap(policy_rows: list[dict], ref_rows: list[dict], metric: str, *, seed: int, bootstrap: int) -> dict:
    by_policy = {str(row["conversation_id"]): row for row in policy_rows}
    by_ref = {str(row["conversation_id"]): row for row in ref_rows}
    ids = sorted(set(by_policy) & set(by_ref))
    if metric in {"status_fix", "strict_fix"}:
        ids = [cid for cid in ids if is_false(by_policy[cid])]
    elif metric in {"honest_status_harm", "honest_strict_harm"}:
        ids = [cid for cid in ids if is_honest(by_policy[cid])]
    if not ids:
        return {"n": 0, "point": None, "ci95": None}

    def diff(cid: str) -> float:
        return metric_value(by_policy[cid], metric) - metric_value(by_ref[cid], metric)

    point = float(np.mean([diff(cid) for cid in ids]))
    cid_by_scenario: dict[str, list[str]] = {}
    for cid in ids:
        sid = str(by_policy[cid].get("scenario_id", ""))
        cid_by_scenario.setdefault(sid or cid, []).append(cid)
    clusters = sorted(cid_by_scenario)
    rng = np.random.default_rng(seed)
    samples = []
    if bootstrap > 0 and len(clusters) >= 2:
        for _ in range(bootstrap):
            drawn = rng.choice(clusters, size=len(clusters), replace=True)
            sampled = [cid for cluster in drawn for cid in cid_by_scenario[cluster]]
            samples.append(float(np.mean([diff(cid) for cid in sampled])))
    elif bootstrap > 0:
        for _ in range(bootstrap):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            samples.append(float(np.mean([diff(str(cid)) for cid in sampled])))
    ci95 = [point, point] if not samples else [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
    return {"n": len(ids), "point": point, "ci95": ci95}


def compute_pairwise(policies: dict[str, dict], references: list[str], *, seed: int, bootstrap: int) -> dict:
    out: dict[str, dict] = {}
    for name, policy in policies.items():
        out[name] = {}
        for ref in references:
            if ref == name or ref not in policies:
                continue
            out[name][ref] = {
                metric: paired_gap(
                    policy["rows"],
                    policies[ref]["rows"],
                    metric,
                    seed=seed,
                    bootstrap=bootstrap,
                )
                for metric in (
                    "status_fix",
                    "strict_fix",
                    "honest_status_harm",
                    "honest_strict_harm",
                    "status_reward",
                    "strict_reward",
                )
            }
    return out


def fmt_ci(gap: dict | None) -> str:
    if not gap or gap.get("point") is None:
        return "n/a"
    lo, hi = gap["ci95"]
    return f"{float(gap['point']):+.4f} [{float(lo):+.4f}, {float(hi):+.4f}]"


def render_markdown(payload: dict) -> str:
    ranked = sorted(
        payload["policies"].items(),
        key=lambda item: (
            float(item[1]["summary"]["mean_strict_reward"]),
            int(item[1]["summary"]["deceptive_strict_fixes"]),
            -int(item[1]["summary"]["honest_strict_harms"]),
            item[0],
        ),
        reverse=True,
    )
    lines = [
        "# Powered150 Targeted Generation Audit",
        "",
        "Deterministic source-level audit over generated policy outputs.",
        "This report is generated from saved JSON artifacts; it does not run the model.",
        "",
        "## Inputs",
        "",
        f"- Transcripts: `{payload['transcripts']}`",
        f"- Transcript SHA256: `{payload['transcripts_sha256']}`",
        f"- Policy files: `{len(payload['inputs'])}`",
        "",
        "## Policy Summary",
        "",
        "| policy | n | status fixes | strict fixes | honest status harm | honest strict harm | parse | strict reward | generation check | GPU util |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, item in ranked:
        s = item["summary"]
        g = item["generation_checks"]
        check = (
            f"verify={g['verification_batches']}, "
            f"status_mis={g['forced_status_mismatches']}, "
            f"reply_mis={g['reply_mismatches']}"
        )
        gpu = f"mean={float(g['gpu_util_pct_mean']):.1f}%, max={float(g['gpu_util_pct_max']):.0f}%, n={g['gpu_sample_count']}"
        lines.append(
            f"| `{name}` | {s['n']} | "
            f"{s['deceptive_status_fixes']}/{s['deceptive_n']} | "
            f"{s['deceptive_strict_fixes']}/{s['deceptive_n']} | "
            f"{s['honest_status_harms']}/{s['honest_n']} | "
            f"{s['honest_strict_harms']}/{s['honest_n']} | "
            f"{s['parse_success']}/{s['n']} | "
            f"{float(s['mean_strict_reward']):.4f} | {check} | {gpu} |"
        )
    lines.extend([
        "",
        "## Status-Class Breakdown",
        "",
        "| policy | false_FAIL strict | false_PASS strict | honest_FAIL strict-ok | honest_PASS strict-ok |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in ranked:
        by = item["summary"]["by_status_class"]
        def cell(cls: str) -> str:
            return f"{by[cls]['basis_strict_ok']}/{by[cls]['n']}"
        lines.append(
            f"| `{name}` | {cell('false_FAIL')} | {cell('false_PASS')} | "
            f"{cell('honest_FAIL')} | {cell('honest_PASS')} |"
        )
    lines.extend([
        "",
        "## Paired Clustered Gaps",
        "",
        "Positive fix/reward gaps are better. Positive harm gaps are worse. CIs are clustered by `scenario_id`.",
        "",
        "| policy | reference | strict reward | strict fix | honest strict harm | status reward |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name in payload["policy_order"]:
        for ref in payload["references"]:
            metrics = payload["paired_gaps"].get(name, {}).get(ref)
            if not metrics:
                continue
            lines.append(
                f"| `{name}` | `{ref}` | {fmt_ci(metrics['strict_reward'])} | "
                f"{fmt_ci(metrics['strict_fix'])} | {fmt_ci(metrics['honest_strict_harm'])} | "
                f"{fmt_ci(metrics['status_reward'])} |"
            )
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- `basis_strict_ok` is a heuristic source-level numeric-rule audit, not an LLM judge.",
        "- Margin-field winners are not assumed to win here; this report is behavior-level generated output evidence.",
        "- Policy files with mismatched conversation sets are compared only on common conversation IDs.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Policy JSON files or globs.")
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--references", default=DEFAULT_REFERENCES)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_paths = expand_inputs(args.inputs)
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input files: {missing}")
    transcripts = read_jsonl(args.transcripts)
    policies_list = [audit_policy_file(path, transcripts) for path in input_paths]
    policies = {item["policy_name"]: item for item in policies_list}
    if len(policies) != len(policies_list):
        names = [item["policy_name"] for item in policies_list]
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"duplicate policy names: {dupes}")
    references = parse_csv(args.references)
    payload = {
        "schema_version": 1,
        "argv": sys.argv,
        "provenance": git_provenance([Path(__file__), args.transcripts, *input_paths]),
        "transcripts": str(args.transcripts.resolve()),
        "transcripts_sha256": file_sha256(args.transcripts),
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in input_paths
        ],
        "references": references,
        "policy_order": sorted(policies),
        "policies": policies,
        "paired_gaps": compute_pairwise(
            policies,
            references,
            seed=args.seed,
            bootstrap=args.bootstrap,
        ),
        "notes": [
            "Behavior-level targeted generation audit; not a margin-only selector report.",
            "Strict audit checks parse, status, numeric rule basis, and pressure-language avoidance.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(payload))
    print(f"wrote {args.out}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
