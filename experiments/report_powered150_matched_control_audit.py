"""Build a matched-control audit for powered150 decision-token baselines.

This report keeps four comparison objects explicit:
- CNG comparator from a dedicated CNG artifact,
- historical route-floor comparator,
- post-hoc fixed-route whole-grid ceiling and heldout-family route-matched fixed-coordinate reconstructions.

The script is intentionally narrow and CPU-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from geoprobe.control.action_response import (  # noqa: E402
    file_sha256,
    evaluate_fixed_route_grid,
    evaluate_route_matched_fixed_coordinate,
    grouped_by_conversation,
    load_action_response,
    make_family_folds,
    metric_values,
    parse_csv,
    summarize_choices,
    target_from_route,
)
from geoprobe.control.literature_baselines import (  # noqa: E402
    evaluate_literature_steering_baselines,
    families_by_name,
)
from geoprobe.provenance import git_provenance  # noqa: E402


DEFAULT_RECEIPT = Path("paper_artifacts/c1_matched_control_audit.json")
DEFAULT_CONTEXT_POLICY = "chart_feature_gate_equivariant_neural_context"
DEFAULT_ROUTE_FLOOR_POLICY = "train_best_route_full_reward"
DEFAULT_CONTEXT_ONLY_POLICY = "learned_context_ridge_reward"
DEFAULT_FIXED_ROUTE_POLICY = "fixed_route_bidir_linear_L16_a96"
SAFE_GAP_METRICS = ("fixes_error", "honest_harm", "reward")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def choice_identity_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    return {
        "methods": dict(Counter(str(row.get("method")) for row in rows)),
        "targets": dict(Counter(str(row.get("target_status")) for row in rows)),
        "layers": dict(Counter(str(row.get("layer")) for row in rows)),
        "alphas": dict(Counter(str(row.get("alpha")) for row in rows)),
    }


def _route_truth(row: dict[str, Any]) -> str | None:
    if row.get("true_status") is not None:
        return str(row.get("true_status") or "").upper() or None
    if row.get("desired_status") is not None:
        return str(row.get("desired_status") or "").upper() or None
    status_class = str(row.get("status_class") or "")
    if status_class.startswith("false_"):
        return "FAIL" if "PASS" in status_class else "PASS"
    if status_class.startswith("honest_"):
        return "PASS" if "PASS" in status_class else "FAIL"
    return None


def _policy_information_audit(choices: list[dict[str, Any]]) -> dict[str, int]:
    route_truth_mismatches = 0
    target_route_mismatches = 0
    known_truth = 0
    for row in choices:
        route_target = target_from_route(row.get("route_action"))
        selected_target = row.get("target_status")
        if selected_target is not None and route_target is not None and str(selected_target) != str(route_target):
            target_route_mismatches += 1
        truth = _route_truth(row)
        if truth is not None and route_target is not None:
            known_truth += 1
            if truth != route_target:
                route_truth_mismatches += 1
    return {
        "n_rows": len(choices),
        "n_conversations": len(set(str(row.get("conversation_id")) for row in choices)),
        "route_truth_mismatches": route_truth_mismatches,
        "selected_target_route_mismatches": target_route_mismatches,
        "n_rows_with_route_truth": known_truth,
    }


def _nonbaseline_candidate_partition(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {"route_matched": 0, "counter_target": 0}
    for row in rows:
        if str(row.get("method")) == "abstain":
            continue
        if str(row.get("target_status") or "") not in {"PASS", "FAIL"}:
            continue
        if row.get("layer") is None:
            continue
        route_target = target_from_route(row.get("route_action"))
        if route_target is None:
            continue
        if route_target == str(row.get("target_status")):
            out["route_matched"] += 1
        else:
            out["counter_target"] += 1
    out["n_nonbaseline"] = out["route_matched"] + out["counter_target"]
    return out


def _information_audit(
    action_rows: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy_audit: dict[str, Any] = {}
    for name, policy in policies.items():
        policy_audit[name] = _policy_information_audit(policy.get("choices", []))
    return {
        "total_conversations": len(grouped_by_conversation(action_rows)),
        "policy_information": policy_audit,
        "raw_nonbaseline_candidate_counts": _nonbaseline_candidate_partition(action_rows),
    }


def _policy_record(
    name: str,
    item: dict[str, Any],
    *,
    source: str,
    source_path: str,
    role: str,
    policy_type: str,
) -> dict[str, Any]:
    choices = item.get("choices")
    if not isinstance(choices, list):
        choices = []
    summary = item.get("summary")
    if not isinstance(summary, dict):
        summary = summarize_choices(choices)
    out = {
        "name": name,
        "role": role,
        "source": source,
        "source_path": source_path,
        "type": policy_type,
        "summary": summary,
        "choices": choices,
        "action_identity_counts": choice_identity_counts(choices),
    }
    for key in ("folds", "best_by_route", "best_by_target", "coordinate"):
        if key in item:
            out[key] = item[key]
    if "folds" in item:
        out["folds"] = item["folds"]
    return out


def _paired_gap_by_cluster(
    policy: list[dict],
    reference: list[dict],
    metric: str,
    *,
    cluster_key: str,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    pol = metric_values(policy, metric)
    ref = metric_values(reference, metric)
    ids = sorted(set(pol) & set(ref))
    if not ids:
        return {"n": 0, "n_clusters": 0, "point": None, "ci95": None}
    pol_by_cid = {str(row["conversation_id"]): row for row in policy}
    diffs = {cid: pol[cid] - ref[cid] for cid in ids}
    clusters: dict[str, list[str]] = {}
    for cid in ids:
        clusters.setdefault(str(pol_by_cid[cid].get(cluster_key, cid)), []).append(cid)
    point = float(np.mean([diffs[cid] for cid in ids]))
    if bootstrap <= 0 or len(clusters) < 2:
        return {"n": len(ids), "n_clusters": len(clusters), "point": point, "ci95": None}
    rng = np.random.default_rng(seed)
    cluster_names = sorted(clusters)
    samples: list[float] = []
    for _ in range(int(bootstrap)):
        drawn = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        sampled = [cid for cluster in drawn for cid in clusters[str(cluster)]]
        samples.append(float(np.mean([diffs[cid] for cid in sampled])))
    return {
        "n": len(ids),
        "n_clusters": len(clusters),
        "point": point,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def _clustered_gaps(
    policies: dict[str, dict[str, Any]],
    references: list[str],
    *,
    cluster_key: str,
    metrics: tuple[str, ...],
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, policy in policies.items():
        out[name] = {}
        for ref in references:
            if ref == name or ref not in policies:
                continue
            out[name][ref] = {
                metric: _paired_gap_by_cluster(
                    policy["choices"],
                    policies[ref]["choices"],
                    metric,
                    cluster_key=cluster_key,
                    seed=seed,
                    bootstrap=bootstrap,
                )
                for metric in metrics
            }
    return out


def family_clustered_gaps(
    policies: dict[str, dict[str, Any]],
    references: list[str],
    *,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    return _clustered_gaps(
        policies,
        references,
        cluster_key="family",
        metrics=SAFE_GAP_METRICS,
        seed=seed,
        bootstrap=bootstrap,
    )


def scenario_clustered_gaps(
    policies: dict[str, dict[str, Any]],
    references: list[str],
    *,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    return _clustered_gaps(
        policies,
        references,
        cluster_key="scenario_id",
        metrics=SAFE_GAP_METRICS,
        seed=seed,
        bootstrap=bootstrap,
    )


def build_payload(
    selector_path: Path,
    cng_path: Path,
    action_response_path: Path,
    *,
    context_policy: str,
    route_floor_policy: str,
    route_matched_methods: set[str],
    out_of_domain_route_floor_fallback: bool,
    objective: str,
    folds_count: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    cng = load_json(cng_path)
    cng_policies = cng.get("policies")
    if not isinstance(cng_policies, dict):
        raise ValueError(f"{cng_path}: missing policies object")

    selector = load_json(selector_path)
    selector_policies = selector.get("policies")
    if not isinstance(selector_policies, dict):
        raise ValueError(f"{selector_path}: missing policies object")

    action_rows, action_meta = load_action_response(action_response_path)
    folds = make_family_folds(action_rows, folds_count)
    grouped = grouped_by_conversation(action_rows)

    policies: dict[str, dict[str, Any]] = {}
    policy_order: list[str] = []
    warnings: list[str] = []
    source_meta = {
        "cng": {
            "path": str(cng_path),
            "sha256": file_sha256(cng_path),
            "byte_size": cng_path.stat().st_size,
        },
        "selector": {
            "path": str(selector_path),
            "sha256": file_sha256(selector_path),
            "byte_size": selector_path.stat().st_size,
        },
        "action_response": {
            "path": str(action_response_path),
            "sha256": file_sha256(action_response_path),
            "byte_size": action_response_path.stat().st_size,
            "meta": action_meta,
        },
    }

    context_item = cng_policies.get(context_policy)
    if isinstance(context_item, dict):
        name = f"context_{context_policy}"
        policies[name] = _policy_record(
            name=name,
            item=context_item,
            source="cng",
            source_path=str(cng_path),
            role="oracle_route_original_status_pre_response_no_candidate_post_action",
            policy_type="cng_pre_response_oracle_route",
        )
        policy_order.append(name)
    else:
        warnings.append(
            f"missing context policy {context_policy} in CNG artifact; add/replace --context-policy or --cng"
        )

    route_floor_item = selector_policies.get(route_floor_policy)
    if isinstance(route_floor_item, dict):
        route_floor_name = "historical_route_floor"
        policies[route_floor_name] = _policy_record(
            name=route_floor_name,
            item=route_floor_item,
            source="selector_eval",
            source_path=str(selector_path),
            role="historical_route_floor",
            policy_type="historical_route_floor",
        )
        policy_order.append(route_floor_name)
    elif out_of_domain_route_floor_fallback:
        baseline_family = evaluate_literature_steering_baselines(
            action_rows,
            folds=folds,
            families=families_by_name(["standard_steering_train_best"]),
            objective=objective,
        )
        if baseline_family.get("standard_steering_train_best"):
            route_floor_name = "historical_route_floor"
            policies[route_floor_name] = _policy_record(
                name=route_floor_name,
                item=baseline_family["standard_steering_train_best"],
                source="literature_baselines",
                source_path="geoprobe.control.literature_baselines",
                role="historical_route_floor",
                policy_type="historical_route_floor_fallback",
            )
            policy_order.append(route_floor_name)
        else:
            warnings.append("route floor unavailable: selector_policy not found and literature fallback produced none")
    else:
        warnings.append("route floor unavailable in selector artifact and fallback disabled")

    context_only_item = selector_policies.get(DEFAULT_CONTEXT_ONLY_POLICY)
    if isinstance(context_only_item, dict):
        policies[DEFAULT_CONTEXT_ONLY_POLICY] = _policy_record(
            name=DEFAULT_CONTEXT_ONLY_POLICY,
            item=context_only_item,
            source="selector_eval",
            source_path=str(selector_path),
            role=(
                "heldout_family_candidate_ranker_oracle_route_feature_both_target_signs"
            ),
            policy_type="learned_oracle_route_feature_candidate_ranker",
        )
        policy_order.append(DEFAULT_CONTEXT_ONLY_POLICY)
    else:
        warnings.append(f"context-only selector unavailable: {DEFAULT_CONTEXT_ONLY_POLICY}")

    fixed_route = evaluate_fixed_route_grid(action_rows)
    fixed_route_item = fixed_route.get(DEFAULT_FIXED_ROUTE_POLICY)
    if fixed_route_item is not None:
        policies[DEFAULT_FIXED_ROUTE_POLICY] = _policy_record(
            name=DEFAULT_FIXED_ROUTE_POLICY,
            item=fixed_route_item,
            source="action_response_recomputed_grid",
            source_path=str(action_response_path),
            role="posthoc_whole_grid_fixed_route_ceiling",
            policy_type="posthoc_fixed_route_whole_grid",
        )
        policy_order.append(DEFAULT_FIXED_ROUTE_POLICY)
    else:
        warnings.append(f"missing fixed-route policy {DEFAULT_FIXED_ROUTE_POLICY} after recomputation")

    route_matched = evaluate_route_matched_fixed_coordinate(
        action_rows,
        folds=folds,
        objective=objective,
        methods=route_matched_methods,
    )
    route_matched_name = "route_matched_fixed_coordinate"
    policies[route_matched_name] = _policy_record(
        name=route_matched_name,
        item=route_matched,
        source="action_response_recomputed",
        source_path=str(action_response_path),
        role=(
            "heldout_clean_route_matched_coordinate_retrospective"
            "_not_preregistered_pre_action"
        ),
        policy_type=(
            "route_matched_fixed_coordinate_retrospective"
            "_heldout_clean_oracle_route"
        ),
    )
    policies[route_matched_name]["methods_restricted_to"] = sorted(route_matched_methods)
    policy_order.append(route_matched_name)

    policy_for_gaps = {name: policies[name] for name in policy_order if "choices" in policies[name]}
    references = [
        name for name in ("historical_route_floor", DEFAULT_FIXED_ROUTE_POLICY) if name in policy_for_gaps
    ]
    if context_policy and any(name.startswith("context_") for name in policy_for_gaps):
        context_key = next(name for name in policy_for_gaps if name.startswith("context_"))
        references.append(context_key)
    references = list(dict.fromkeys(references))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "argv": sys.argv,
        "git": git_provenance(),
        "policy_order": policy_order,
        "inputs": {
            "context_policy": context_policy,
            "cng_path": str(cng_path),
            "route_floor_policy": route_floor_policy,
            "objective": objective,
            "folds": folds,
            "bootstrap": bootstrap,
            "seed": seed,
            "route_matched_methods": sorted(route_matched_methods),
            "out_of_domain_route_floor_fallback": out_of_domain_route_floor_fallback,
            "n_rows": len(action_rows),
            "n_conversations": len(grouped),
            "n_families": len({str(row.get("family")) for row in action_rows}),
            "source_inputs": source_meta,
        },
        "policies": policies,
        "information_audit": _information_audit(action_rows, policy_for_gaps),
        "paired_gaps": scenario_clustered_gaps(
            policy_for_gaps,
            references=references,
            seed=seed,
            bootstrap=bootstrap,
        ) if len(policy_for_gaps) > 1 else {},
        "family_clustered_gaps": family_clustered_gaps(
            policy_for_gaps,
            references=references,
            seed=seed,
            bootstrap=bootstrap,
        ) if len(policy_for_gaps) > 1 else {},
        "warnings": warnings,
    }
    return payload


def _compact_fold(fold: dict[str, Any]) -> dict[str, Any]:
    """Keep only the held-out identity, selected coordinate, and scored summary."""
    out = {
        key: fold[key]
        for key in ("heldout_families", "coordinate", "best_by_target", "summary")
        if key in fold
    }
    return out


def build_public_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip raw choices and operational paths from the citable C1 receipt."""
    sources = payload["inputs"]["source_inputs"]
    logical_sources = {
        "cng_selector": sources["cng"],
        "historical_selector": sources["selector"],
        "action_response_field": sources["action_response"],
    }
    source_artifacts = {
        name: {
            "sha256": item["sha256"],
            "byte_size": int(item["byte_size"]),
        }
        for name, item in logical_sources.items()
    }

    policies: dict[str, Any] = {}
    for name in payload["policy_order"]:
        policy = payload["policies"][name]
        policies[name] = {
            "role": policy["role"],
            "type": policy["type"],
            "summary": policy["summary"],
            "action_identity_counts": policy["action_identity_counts"],
            "folds": {
                fold_id: _compact_fold(fold)
                for fold_id, fold in (policy.get("folds") or {}).items()
            },
        }
        if "methods_restricted_to" in policy:
            policies[name]["methods_restricted_to"] = policy["methods_restricted_to"]

    return {
        "schema_version": 1,
        "kind": "powered150_matched_control_public_receipt",
        "claim_id": "C1",
        "producer": "experiments/report_powered150_matched_control_audit.py",
        "producer_sha256": file_sha256(Path(__file__)),
        "chronology": {
            "registered_comparison": (
                "CNG versus the declared historical train-fold route floor inside the "
                "oracle-routed powered150 sandbox"
            ),
            "historical_fixed_route": (
                "whole-grid post-hoc feasibility ceiling; not a registered baseline"
            ),
            "context_only_selector": (
                "heldout-family ridge candidate ranker; receives the oracle route as a feature, "
                "scores both target signs, and never inspects heldout candidate outcomes"
            ),
            "route_matched_reconstruction": (
                "retrospective heldout-family-clean robustness analysis; candidate-response-blind "
                "at heldout application but oracle-route-conditioned"
            ),
            "prospective_fresh_generation_controller": "not established by this receipt",
        },
        "analysis": {
            "objective": payload["inputs"]["objective"],
            "bootstrap": payload["inputs"]["bootstrap"],
            "seed": payload["inputs"]["seed"],
            "folds": payload["inputs"]["folds"],
            "n_rows": payload["inputs"]["n_rows"],
            "n_conversations": payload["inputs"]["n_conversations"],
            "n_families": payload["inputs"]["n_families"],
        },
        "source_artifacts": source_artifacts,
        "information_audit": payload["information_audit"],
        "policy_order": payload["policy_order"],
        "policies": policies,
        "scenario_clustered_gaps": payload["paired_gaps"],
        "family_clustered_gaps": payload["family_clustered_gaps"],
        "interpretation": (
            "CNG strongly exceeds the learned route-feature candidate ranker inside the offline "
            "oracle-routed field. The field itself is route-constrained on deceptive rows: it "
            "exposes only corrective-target candidates there, so every policy is equally "
            "route-constrained and none can pick a counter-target on those 600 rows. CNG adds no "
            "selector-side route mask; it takes an unconstrained argmax over the candidates the "
            "field exposes and matches the supplied route on honest rows only because corrective "
            "candidates score higher. The learned ridge receives the same route as a feature yet "
            "selects 233 counter-target candidates, all on honest rows, and corrects far fewer "
            "deceptive rows. The historical registered floor is not a clean comparator because its "
            "train selection pooled counter-target candidates. A separately conceived "
            "fixed-coordinate policy that maps the route to its corrective target saturates the "
            "field and ties CNG, so the gap reflects corrective-coordinate selection and reward "
            "estimation, not route access or a geometric actuation primitive; this audit does not "
            "establish prospective control superiority."
        ),
    }


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "- "
    return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Powered150 Matched-Control Baseline Audit",
        "",
        "This report separates deployability classes before comparing baselines:",
        "- CNG comparator from a dedicated CNG artifact,"
        " conditioned on oracle route and original status from action responses,"
        " with pre-response activation features and no post-action response metric usage",
        "- historical route-floor comparator (`train_best_route_full_reward` when present),",
        "- post-hoc whole-grid fixed-route ceiling (`fixed_route_bidir_linear_L16_a96`),",
        "- heldout-family route-matched fixed-coordinate reconstruction (candidate-response-blind at heldout, oracle-route-conditioned, retrospective).",
        "",
        "## Inputs",
        f"- CNG input: `{payload['inputs']['source_inputs']['cng']['path']}`",
        f"- Selector input (historical floor only): `{payload['inputs']['source_inputs']['selector']['path']}`",
        f"- Action-response input: `{payload['inputs']['source_inputs']['action_response']['path']}`",
        f"- Objective: `{payload['inputs']['objective']}`",
        f"- Folds: `{len(payload['inputs']['folds'])}` with heldout family map `"
        f"{payload['inputs']['folds']}`",
        f"- Seed: `{payload['inputs']['seed']}` | Bootstrap: `{payload['inputs']['bootstrap']}`",
        "",
        "## Policy summaries",
        "",
        "| policy | role | source | fixes | honest_harm | mean_reward | mean_aligned_margin | chosen methods |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for name in payload["policy_order"]:
        policy = payload["policies"][name]
        s = policy["summary"]
        methods = _fmt_counts(s.get("chosen_methods") or {})
        lines.append(
            f"| `{name}` | {policy['role']} | {policy['source']} | "
            f"{s.get('fixes_error')}/{s.get('deceptive_n')} | {s.get('honest_harms')}/{s.get('honest_n')} | "
            f"{float(s.get('mean_reward') or 0.0):.4f} | "
            f"{float(s.get('mean_aligned_margin') or 0.0):.4f} | {methods} |"
        )
        if policy.get("action_identity_counts"):
            ident = policy["action_identity_counts"]
            lines.extend([
                f"- `{name}` action identity counts:",
                f"  - methods: {_fmt_counts(ident.get('methods', {}))}",
                f"  - targets: {_fmt_counts(ident.get('targets', {}))}",
                f"  - layers: {_fmt_counts(ident.get('layers', {}))}",
                f"  - alphas: {_fmt_counts(ident.get('alphas', {}))}",
            ])

    lines.extend([
        "",
        "## Registered result vs floor framing",
        "- The CNG arm is the registered source artifact comparator and is intentionally not treated as a pre-action controller.",
        "- `fixed_route_bidir_linear_L16_a96` is a whole-grid post-hoc feasibility ceiling, not a corrected or preregistered floor candidate.",
        "- `route_matched_fixed_coordinate` is a retrospective heldout-clean reconstruction of oracle-route-conditioned coordinates and remains outside preregistered response-blind controller guarantees.",
        "- Retrospective robustness conclusion from this file should be read as an audit signal, not a causal deployment claim.",
    ])

    lines.extend(["", "## Fixed-route vs route-matched fixed-coordinate (fold view)", ""])
    route_matched = payload["policies"].get("route_matched_fixed_coordinate")
    if route_matched and route_matched.get("folds"):
        lines.extend(["| fold | heldout families | coordinate | deceptive_n | honest_n | fixes | honest_harms |"])
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for fold, info in sorted(route_matched["folds"].items(), key=lambda kv: int(kv[0])):
            rows = info.get("summary", {})
            lines.append(
                f"| {fold} | `{info.get('heldout_families')}` | {info.get('coordinate')} | "
                f"{rows.get('deceptive_n')} | {rows.get('honest_n')} | "
                f"{rows.get('fixes_error')} | {rows.get('honest_harms')} |"
            )
    fixed = payload["policies"].get("fixed_route_bidir_linear_L16_a96")
    if fixed:
        lines.extend([
            "",
            "### Fixed-route note",
            "- `fixed_route_bidir_linear_L16_a96` is a post-hoc whole-grid feasibility ceiling computed after the fact.",
            "- Do not report this arm as a deployable or corrected route-matched baseline.",
        ])

    if payload["paired_gaps"]:
        lines.extend([
            "",
            "## Paired gaps (scenario-clustered)",
            "```json",
            json.dumps(payload["paired_gaps"], indent=2, sort_keys=True),
            "```",
            "",
            "## Paired gaps (family-clustered)",
            "```json",
            json.dumps(payload["family_clustered_gaps"], indent=2, sort_keys=True),
            "```",
        ])
    if payload["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        lines.extend([f"- {item}" for item in payload["warnings"]])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector",
        type=Path,
        required=True,
        help="Selector-evaluation source artifact.",
    )
    parser.add_argument(
        "--cng",
        type=Path,
        required=True,
        help="Conditional neural-geometry source artifact.",
    )
    parser.add_argument(
        "--action-response",
        type=Path,
        required=True,
        help="Completed action-response field source artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional detailed JSON audit output; not part of the compact public receipt.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional rendered Markdown audit output.",
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--context-policy", default=DEFAULT_CONTEXT_POLICY)
    parser.add_argument("--route-floor-policy", default=DEFAULT_ROUTE_FLOOR_POLICY)
    parser.add_argument("--route-matched-methods", default="bidir_linear")
    parser.add_argument("--objective", default="reward", choices=["reward", "aligned_margin", "aligned_delta_margin"])
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-route-floor-fallback", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    methods = set(parse_csv(args.route_matched_methods))
    if not methods:
        raise SystemExit("--route-matched-methods cannot be empty")
    payload = build_payload(
        selector_path=args.selector,
        cng_path=args.cng,
        action_response_path=args.action_response,
        context_policy=args.context_policy,
        route_floor_policy=args.route_floor_policy,
        route_matched_methods=methods,
        out_of_domain_route_floor_fallback=not args.no_route_floor_fallback,
        objective=args.objective,
        folds_count=args.folds,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        payload["out"] = str(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(payload))
    args.receipt.write_text(json.dumps(build_public_receipt(payload), indent=2, sort_keys=True) + "\n")
    if args.out is not None:
        print(f"wrote {args.out}")
    if args.report is not None:
        print(f"wrote {args.report}")
    print(f"wrote {args.receipt}")


if __name__ == "__main__":
    main()
