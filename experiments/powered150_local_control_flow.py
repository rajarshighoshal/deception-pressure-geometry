"""Evaluate option-3 local control-flow vectors on powered150.

This is CPU-only. It does not claim a new generated-output result; it tests
whether train-family local reward vectors over real steering directions predict
held-out useful directions before spending GPU on new generated directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.geometric_chart_atlas_selector import load_state_vectors  # noqa: E402
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402  fingerprint promoted load_state_vectors
from experiments.powered150_activation_base_space_scan import stratified_limit  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.control import geometry_map as gm  # noqa: E402
from geoprobe.control.action_response import (  # noqa: E402
    file_sha256,
    grouped_by_conversation,
    load_action_response,
    make_family_folds,
    paired_gap,
    parse_int_csv,
    prune_to_selected_library,
    safe_float,
    select_action_keys,
    slim_choice,
)
from geoprobe.control.local_control_flow import (  # noqa: E402
    LocalControlFlowConfig,
    LocalControlFlowEstimator,
    flow_alignment,
)
from geoprobe.control.local_deformation_field import SteeringSpecLookup  # noqa: E402


DEFAULT_ROOT = Path("results/powered150/run_20260629_142532")
DEFAULT_ACTIVATIONS = DEFAULT_ROOT / "merged/turns.pt"
DEFAULT_ACTION_RESPONSE = Path("results/spec_scored/powered150_broad_full/run_20260701_161044/action_response.json")
DEFAULT_SPEC_BANK = Path("results/spec_banks/powered150_broad")
DEFAULT_OUT = DEFAULT_ROOT / "local_control_flow_option3.json"


def filter_layers(rows: list[dict], layers: set[int]) -> list[dict]:
    out = []
    for row in rows:
        if gm.is_baseline(row):
            out.append(row)
        elif row.get("layer") is not None and int(row["layer"]) in layers:
            out.append(row)
    return out


def route_target(row: dict) -> str | None:
    action = str(row.get("route_action") or "")
    if action == "steer_to_PASS":
        return "PASS"
    if action == "steer_to_FAIL":
        return "FAIL"
    return None


def calibrate_threshold(
    estimator: LocalControlFlowEstimator,
    rows: list[dict],
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
) -> float:
    scored_groups = []
    best_scores = []
    for candidates in gm.grouped_by_state(rows).values():
        scores = estimator.score(candidates, state_vectors, exclude_same_state=True)
        scored_groups.append((candidates, scores))
        finite = scores[np.isfinite(scores) & (scores > -1e8)]
        best_scores.append(float(np.max(finite)) if finite.size else -1e9)
    if not best_scores:
        return 0.0
    quantiles = np.quantile(np.asarray(best_scores, dtype=np.float64), np.linspace(0.0, 0.95, 24))
    thresholds = sorted(set([-1e9, 0.0, *[float(x) for x in quantiles if np.isfinite(x)]]))
    best_threshold = 0.0
    best_key = None
    for threshold in thresholds:
        choices = [
            gm.choose_by_scores(candidates, scores, threshold=threshold)
            for candidates, scores in scored_groups
        ]
        summary = gm.summarize_choices(choices)
        deceptive_n = max(float(summary["deceptive_n"]), 1.0)
        honest_n = max(float(summary["honest_n"]), 1.0)
        key = (
            float(summary["mean_reward"]),
            -float(summary["honest_harms"]) / honest_n,
            float(summary["fixes_error"]) / deceptive_n,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def alignment_rows_for_fold(
    estimator: LocalControlFlowEstimator,
    full_test_rows: list[dict],
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
    *,
    layers: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for cid, group in grouped_by_conversation(full_test_rows).items():
        first = group[0]
        target = route_target(first)
        if target is None:
            continue
        for layer in sorted(layers):
            candidates = [
                row for row in group
                if row.get("layer") is not None
                and int(row["layer"]) == layer
                and str(row.get("target_status")) == target
            ]
            if not candidates:
                continue
            rep = candidates[0]
            pred = estimator.predict(rep, state_vectors, exclude_same_state=False)
            oracle = estimator.oracle_flow(candidates)
            align = flow_alignment(pred.vector, oracle)
            rows.append({
                "conversation_id": str(cid),
                "scenario_id": first.get("scenario_id"),
                "family": first.get("family"),
                "status_class": first.get("status_class"),
                "route_action": first.get("route_action"),
                "target_status": target,
                "layer": int(layer),
                "pred_norm": align["pred_norm"],
                "oracle_norm": align["oracle_norm"],
                "cosine": align["cosine"],
                "support_count": pred.support_count,
                "neighbor_state_ids": pred.neighbor_state_ids[:5],
                "neighbor_distances": pred.neighbor_distances[:5],
            })
    return rows


def summarize_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
        vals = np.asarray([safe_float(item.get("cosine")) for item in items], dtype=np.float64)
        pred_norm = np.asarray([safe_float(item.get("pred_norm")) for item in items], dtype=np.float64)
        oracle_norm = np.asarray([safe_float(item.get("oracle_norm")) for item in items], dtype=np.float64)
        if vals.size == 0:
            return {"n": 0, "mean_cosine": None, "median_cosine": None, "mean_pred_norm": None, "mean_oracle_norm": None}
        return {
            "n": int(vals.size),
            "mean_cosine": float(np.mean(vals)),
            "median_cosine": float(np.median(vals)),
            "p25_cosine": float(np.quantile(vals, 0.25)),
            "p75_cosine": float(np.quantile(vals, 0.75)),
            "mean_pred_norm": float(np.mean(pred_norm)),
            "mean_oracle_norm": float(np.mean(oracle_norm)),
        }

    by_status = {}
    for status in sorted({str(row.get("status_class")) for row in rows}):
        by_status[status] = summary([row for row in rows if str(row.get("status_class")) == status])
    return {"overall": summary(rows), "by_status_class": by_status}


def evaluate_flow_policy(
    rows: list[dict],
    *,
    policy_name: str,
    folds: list[list[str]],
    state_vectors: dict[gm.StateLayerKey, np.ndarray],
    spec_lookup: SteeringSpecLookup,
    layers: set[int],
    top_actions_per_target: int,
    config: LocalControlFlowConfig,
    threshold_mode: str,
) -> dict[str, Any]:
    choices = []
    alignments = []
    fold_out = {}
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_full = filter_layers([row for row in rows if str(row["family"]) not in heldout_set], layers)
        test_full = filter_layers([row for row in rows if str(row["family"]) in heldout_set], layers)
        selected = select_action_keys(train_full, top_per_target=top_actions_per_target, objective=config.objective)
        train_eval = prune_to_selected_library(train_full, selected)
        test = prune_to_selected_library(test_full, selected)
        estimator = LocalControlFlowEstimator(spec_lookup, config).fit(train_full, state_vectors)
        threshold = -1e9 if threshold_mode == "forced_pick" else 0.0
        if threshold_mode == "train_calibrated":
            threshold = calibrate_threshold(estimator, train_eval, state_vectors)
        fold_choices = []
        for candidates in gm.grouped_by_state(test).values():
            scores = estimator.score(candidates, state_vectors, exclude_same_state=False)
            chosen = gm.choose_by_scores(candidates, scores, threshold=threshold)
            if not gm.is_baseline(chosen):
                pred = estimator.predict(chosen, state_vectors, exclude_same_state=False)
                chosen = dict(chosen)
                chosen["policy_score"] = float(chosen.get("policy_score", 0.0))
                chosen["explanation"] = {
                    "model": policy_name,
                    "predicted_flow_norm": pred.confidence,
                    "support_count": pred.support_count,
                    "neighbor_state_ids": pred.neighbor_state_ids[:5],
                    "neighbor_distances": pred.neighbor_distances[:5],
                }
            fold_choices.append(chosen)
        fold_alignment = alignment_rows_for_fold(estimator, test_full, state_vectors, layers=layers)
        choices.extend(fold_choices)
        alignments.extend(fold_alignment)
        fold_out[str(fold_idx)] = {
            "heldout_families": heldout,
            "threshold": threshold,
            "n_train_rows": len(train_full),
            "n_train_eval_rows": len(train_eval),
            "n_test_rows": len(test),
            "summary": gm.summarize_choices(fold_choices),
            "alignment": summarize_alignment(fold_alignment),
            "selected_keys_per_target": {target: [list(key) for key in keys] for target, keys in selected.items()},
        }
        summary = fold_out[str(fold_idx)]["summary"]
        print(
            f"[local_control_flow] {policy_name} fold={fold_idx} "
            f"fixes={summary['fixes_error']}/{summary['deceptive_n']} "
            f"harm={summary['honest_harms']}/{summary['honest_n']} "
            f"reward={float(summary['mean_reward']):.4f}",
            flush=True,
        )
    slimmed = []
    for row in choices:
        item = slim_choice(row)
        item["desired_margin_sign"] = row.get("desired_margin_sign")
        item["desired_status"] = row.get("desired_status")
        item["true_status"] = row.get("true_status")
        slimmed.append(item)
    return {
        "summary": gm.summarize_choices(choices),
        "alignment": summarize_alignment(alignments),
        "folds": fold_out,
        "choices": slimmed,
        "alignment_rows": alignments,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.4f}"

    lines = [
        "# Powered150 Option-3 Local Control Flow",
        "",
        "CPU-only deployable proxy for learning a local control vector field.",
        "Train folds define reward-weighted steering directions; held-out states only query that train-local field.",
        "",
        "## Policies",
        "",
        "| policy | fixes | honest harm | reward | mean alignment cos | median alignment cos | pred norm | oracle norm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["policies"].items():
        summary = item["summary"]
        align = item["alignment"]["overall"]
        lines.append(
            f"| {name} | {summary['fixes_error']}/{summary['deceptive_n']} | "
            f"{summary['honest_harms']}/{summary['honest_n']} | {fmt(summary['mean_reward'])} | "
            f"{fmt(align['mean_cosine'])} | {fmt(align['median_cosine'])} | "
            f"{fmt(align['mean_pred_norm'])} | {fmt(align['mean_oracle_norm'])} |"
        )
    lines.extend([
        "",
        "## Alignment By Status Class",
        "",
        "| policy | status | n | mean cos | median cos | pred norm | oracle norm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in payload["policies"].items():
        for status, row in item["alignment"]["by_status_class"].items():
            lines.append(
                f"| {name} | {status} | {row['n']} | {fmt(row['mean_cosine'])} | "
                f"{fmt(row['median_cosine'])} | {fmt(row['mean_pred_norm'])} | {fmt(row['mean_oracle_norm'])} |"
            )
    lines.extend([
        "",
        "## Paired Gaps",
        "",
        "| policy | reference | reward gap | fix gap | harm gap |",
        "|---|---|---:|---:|---:|",
    ])
    for name, refs in payload["paired_gaps"].items():
        for ref, metrics in refs.items():
            lines.append(
                f"| {name} | {ref} | {fmt(metrics['reward']['point'])} | "
                f"{fmt(metrics['fixes_error']['point'])} | {fmt(metrics['honest_harm']['point'])} |"
            )
    lines.extend([
        "",
        "## Read",
        "",
        "- `local_control_flow_context` is the option-3 proxy: a route/target/layer local vector field over real steering directions.",
        "- `global_control_flow_context` removes local activation dependence and asks whether one train-fold vector per route/target/layer is enough.",
        "- Alignment uses held-out response labels only for analysis; it is not available to the policy.",
        "- A GPU step is only justified if local beats global and has positive held-out oracle-flow alignment on the hard false classes.",
        "",
    ])
    return "\n".join(lines)


def compute_gaps(policies: dict[str, dict], *, seed: int, bootstrap: int) -> dict[str, Any]:
    names = list(policies)
    out: dict[str, Any] = {}
    for name in names:
        out[name] = {}
        for ref in names:
            if ref == name:
                continue
            out[name][ref] = {
                metric: paired_gap(policies[name]["choices"], policies[ref]["choices"], metric, seed=seed, bootstrap=bootstrap)
                for metric in ("fixes_error", "honest_harm", "reward", "aligned_margin")
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-response", type=Path, default=DEFAULT_ACTION_RESPONSE)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--steering-spec-bank", type=Path, default=DEFAULT_SPEC_BANK)
    parser.add_argument("--action-response-sha256", default=None)
    parser.add_argument("--activations-sha256", default=None)
    parser.add_argument("--steering-spec-manifest-sha256", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", default="16")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--top-actions-per-target", type=int, default=16)
    parser.add_argument("--geometry-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--neighbor-pool-multiplier", type=int, default=4)
    parser.add_argument("--threshold-mode", choices=("forced_pick", "zero", "train_calibrated"), default="train_calibrated")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cids", type=int, default=0)
    args = parser.parse_args()

    layers = set(parse_int_csv(args.layers))
    rows, action_meta = load_action_response(args.action_response)
    records = {
        cid: {
            "conversation_id": cid,
            "family": group[0].get("family"),
            "status_class": group[0].get("status_class"),
        }
        for cid, group in grouped_by_conversation(rows).items()
    }
    keep_cids = stratified_limit(records, max_cids=args.max_cids, seed=args.seed)
    if keep_cids != set(records):
        rows = [row for row in rows if str(row.get("conversation_id")) in keep_cids]
    folds = make_family_folds(rows, args.folds)
    state_vectors, activation_meta = load_state_vectors(
        args.activations,
        layers=sorted(layers),
        query_turn=args.query_turn,
        query_phase=args.query_phase,
    )
    spec_lookup = SteeringSpecLookup(args.steering_spec_bank)
    base_config = {
        "top_k": args.top_k,
        "neighbor_pool_multiplier": args.neighbor_pool_multiplier,
        "geometry_dim": args.geometry_dim,
        "objective": "reward",
        "target_route_only": True,
        "seed": args.seed,
    }
    policies = {
        "local_control_flow_context": evaluate_flow_policy(
            rows,
            policy_name="local_control_flow_context",
            folds=folds,
            state_vectors=state_vectors,
            spec_lookup=spec_lookup,
            layers=layers,
            top_actions_per_target=args.top_actions_per_target,
            config=LocalControlFlowConfig(**base_config, use_local_neighbors=True),
            threshold_mode=args.threshold_mode,
        ),
        "global_control_flow_context": evaluate_flow_policy(
            rows,
            policy_name="global_control_flow_context",
            folds=folds,
            state_vectors=state_vectors,
            spec_lookup=spec_lookup,
            layers=layers,
            top_actions_per_target=args.top_actions_per_target,
            config=LocalControlFlowConfig(**base_config, use_local_neighbors=False),
            threshold_mode=args.threshold_mode,
        ),
    }
    payload = {
        "schema_version": 1,
        "argv": sys.argv,
        "action_response": str(args.action_response),
        "action_response_sha256": args.action_response_sha256 or file_sha256(args.action_response),
        "action_response_meta": action_meta,
        "activations": str(args.activations),
        "activations_sha256": args.activations_sha256 or file_sha256(args.activations),
        "activation_meta": activation_meta,
        "steering_spec_bank": str(args.steering_spec_bank),
        "steering_spec_manifest_sha256": args.steering_spec_manifest_sha256 or file_sha256(args.steering_spec_bank / "manifest.json"),
        "provenance": git_provenance([Path(__file__), Path(_bank_mod.__file__), args.action_response, args.activations, args.steering_spec_bank / "manifest.json"]),
        "config": {
            "layers": sorted(layers),
            "query_turn": args.query_turn,
            "query_phase": args.query_phase,
            "folds": folds,
            "top_actions_per_target": args.top_actions_per_target,
            "geometry_dim": args.geometry_dim,
            "top_k": args.top_k,
            "neighbor_pool_multiplier": args.neighbor_pool_multiplier,
            "threshold_mode": args.threshold_mode,
            "max_cids": args.max_cids,
        },
        "policies": policies,
        "paired_gaps": compute_gaps(policies, seed=args.seed, bootstrap=args.bootstrap),
        "notes": [
            "Option-3 CPU proxy: local predicted control vectors are used only to score existing candidate directions.",
            "Held-out oracle-flow alignment is post-hoc analysis, not a deployable input.",
            "A later GPU stage would score exported predicted-flow steering vectors directly.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = args.out.with_suffix(".md")
    md.write_text(render_markdown(payload) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "markdown": str(md),
        "policies": {name: item["summary"] for name, item in policies.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
