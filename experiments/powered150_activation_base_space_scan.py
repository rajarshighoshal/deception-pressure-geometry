"""Scan powered150 activation base-space choices before controller training.

The goal is not to train a controller.  It asks which activation state space
looks like a usable base manifold for control: stable held-out-family
neighborhoods, smooth route/status labels, and smooth action-response summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_graded_dp_frontier import (  # noqa: E402
    load_activation_bank,
    load_activation_points_from_bank,
    point_index,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.text.parse import parse_csv, parse_int_csv  # noqa: E402
from geoprobe.control.action_response import (  # noqa: E402
    aligned_margin,
    candidate_action_rows,
    file_sha256,
    grouped_by_conversation,
    load_action_response,
    make_family_folds,
    safe_float,
)


DEFAULT_ROOT = Path("results/powered150/run_20260629_142532")
DEFAULT_ACTIVATIONS = DEFAULT_ROOT / "merged/turns.pt"
DEFAULT_ACTION_RESPONSE = Path("results/spec_scored/powered150_broad_full/run_20260701_161044/action_response.json")
DEFAULT_OUT = DEFAULT_ROOT / "powered150_activation_base_space_scan.json"


def best_action(rows: list[dict]) -> dict:
    candidates = candidate_action_rows(rows)
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            safe_float(row.get("reward")),
            aligned_margin(row),
            -safe_float(row.get("alpha")),
            -safe_float(row.get("layer")),
            str(row.get("method")),
        ),
    )


def state_records_from_action_response(rows: list[dict]) -> dict[str, dict[str, Any]]:
    grouped = grouped_by_conversation(rows)
    out: dict[str, dict[str, Any]] = {}
    for cid, candidates in grouped.items():
        first = candidates[0]
        status_class = str(first.get("status_class"))
        action_candidates = candidate_action_rows(candidates)
        if status_class.startswith("false_"):
            response_density = float(np.mean([float(bool(row.get("fixes_error"))) for row in action_candidates]))
            response_metric = "fixes_error_fraction"
        elif status_class.startswith("honest_"):
            response_density = float(np.mean([float(bool(row.get("harms_honest"))) for row in action_candidates]))
            response_metric = "harms_honest_fraction"
        else:
            response_density = 0.0
            response_metric = "unknown"
        best = best_action(candidates)
        out[str(cid)] = {
            "conversation_id": str(cid),
            "scenario_id": str(first.get("scenario_id")),
            "family": str(first.get("family")),
            "status_class": status_class,
            "route_action": str(first.get("route_action")),
            "base_margin": safe_float(first.get("base_margin")),
            "response_density": response_density,
            "response_metric": response_metric,
            "best_method": str(best.get("method") or "NONE"),
            "best_target_status": str(best.get("target_status") or "NONE"),
            "best_layer": safe_float(best.get("layer"), -1.0),
            "best_alpha": safe_float(best.get("alpha"), -1.0),
            "best_reward": safe_float(best.get("reward")),
        }
    return out


def stratified_limit(records: dict[str, dict[str, Any]], max_cids: int, seed: int) -> list[str]:
    cids = sorted(records)
    if max_cids <= 0 or len(cids) <= max_cids:
        return cids
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for cid in cids:
        by_class[str(records[cid]["status_class"])].append(cid)
    selected: list[str] = []
    classes = sorted(by_class)
    while len(selected) < max_cids and any(by_class.values()):
        for cls in classes:
            if not by_class[cls] or len(selected) >= max_cids:
                continue
            idx = int(rng.integers(0, len(by_class[cls])))
            selected.append(by_class[cls].pop(idx))
    return sorted(selected)


def majority(values: list[Any]) -> Any:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def normalize_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def local_dimension_summary(z: np.ndarray, *, k: int, seed: int, max_samples: int) -> dict[str, float | int | None]:
    if z.shape[0] <= 2:
        return {"sample_count": 0, "median": None, "p90": None, "mean": None}
    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, z.shape[0])
    sample = rng.choice(np.arange(z.shape[0]), size=sample_count, replace=False)
    nn = NearestNeighbors(n_neighbors=min(k + 1, z.shape[0]), metric="euclidean")
    nn.fit(z)
    _, idx = nn.kneighbors(z[sample])
    dims = []
    for neigh in idx:
        neigh = neigh[1:] if len(neigh) > 1 else neigh
        if len(neigh) < 3:
            continue
        local = z[neigh] - z[neigh].mean(axis=0, keepdims=True)
        cov = local.T @ local / max(local.shape[0] - 1, 1)
        eig = np.linalg.eigvalsh(cov)
        eig = eig[eig > 1e-12]
        if eig.size == 0:
            continue
        dims.append(float((eig.sum() ** 2) / np.sum(eig ** 2)))
    if not dims:
        return {"sample_count": 0, "median": None, "p90": None, "mean": None}
    arr = np.asarray(dims, dtype=np.float64)
    return {
        "sample_count": int(len(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "mean": float(np.mean(arr)),
    }


def mean_entropy(count_rows: list[list[str]]) -> float:
    values = []
    for row in count_rows:
        counts = np.asarray(list(Counter(row).values()), dtype=np.float64)
        probs = counts / max(float(counts.sum()), 1e-12)
        values.append(float(-(probs * np.log(np.maximum(probs, 1e-12))).sum()))
    return float(np.mean(values)) if values else 0.0


def evaluate_base_space(
    x_by_cid: dict[str, np.ndarray],
    records: dict[str, dict[str, Any]],
    folds: list[list[str]],
    *,
    pca_dim: int,
    metric: str,
    k: int,
    seed: int,
    local_dim_samples: int,
) -> dict[str, Any]:
    cids = sorted(set(x_by_cid) & set(records))
    if not cids:
        raise ValueError("no overlapping activation/action-response IDs")
    x = np.vstack([x_by_cid[cid] for cid in cids]).astype(np.float64)
    family = np.asarray([records[cid]["family"] for cid in cids], dtype=object)
    status = np.asarray([records[cid]["status_class"] for cid in cids], dtype=object)
    route = np.asarray([records[cid]["route_action"] for cid in cids], dtype=object)
    best_method = np.asarray([records[cid]["best_method"] for cid in cids], dtype=object)
    best_target = np.asarray([records[cid]["best_target_status"] for cid in cids], dtype=object)
    best_layer = np.asarray([records[cid]["best_layer"] for cid in cids], dtype=np.float64)
    best_alpha = np.asarray([records[cid]["best_alpha"] for cid in cids], dtype=np.float64)
    response_density = np.asarray([records[cid]["response_density"] for cid in cids], dtype=np.float64)
    fold_rows = []
    all_pred = {
        "status": [],
        "route": [],
        "best_method": [],
        "best_target": [],
        "best_layer": [],
        "best_alpha": [],
        "response_density": [],
    }
    all_true = {
        "status": [],
        "route": [],
        "best_method": [],
        "best_target": [],
        "best_layer": [],
        "best_alpha": [],
        "response_density": [],
    }
    pca_vars = []
    local_dims = []
    neighbor_family_entropy = []
    neighbor_distance = []
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(map(str, heldout))
        test_mask = np.asarray([fam in heldout_set for fam in family], dtype=bool)
        train_mask = ~test_mask
        if int(train_mask.sum()) < 4 or int(test_mask.sum()) == 0:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_mask])
        x_test = scaler.transform(x[test_mask])
        n_components = max(1, min(int(pca_dim), x_train.shape[0] - 1, x_train.shape[1]))
        pca = PCA(n_components=n_components, random_state=seed + fold_idx)
        z_train = pca.fit_transform(x_train)
        z_test = pca.transform(x_test)
        if metric == "cosine":
            z_train_fit = normalize_rows(z_train)
            z_test_fit = normalize_rows(z_test)
            nn_metric = "cosine"
        elif metric == "euclidean":
            z_train_fit = z_train
            z_test_fit = z_test
            nn_metric = "euclidean"
        else:
            raise ValueError(f"unknown metric {metric!r}")
        n_neighbors = min(int(k), z_train_fit.shape[0])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=nn_metric)
        nn.fit(z_train_fit)
        dist, idx = nn.kneighbors(z_test_fit)
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        pred_status = []
        pred_route = []
        pred_method = []
        pred_target = []
        pred_layer = []
        pred_alpha = []
        pred_density = []
        fold_neighbor_families = []
        for row_idx in idx:
            neigh = train_indices[row_idx]
            pred_status.append(majority(status[neigh].tolist()))
            pred_route.append(majority(route[neigh].tolist()))
            pred_method.append(majority(best_method[neigh].tolist()))
            pred_target.append(majority(best_target[neigh].tolist()))
            pred_layer.append(float(np.median(best_layer[neigh])))
            pred_alpha.append(float(np.median(best_alpha[neigh])))
            pred_density.append(float(np.mean(response_density[neigh])))
            fold_neighbor_families.append(family[neigh].tolist())
        truth = {
            "status": status[test_indices].tolist(),
            "route": route[test_indices].tolist(),
            "best_method": best_method[test_indices].tolist(),
            "best_target": best_target[test_indices].tolist(),
            "best_layer": best_layer[test_indices].tolist(),
            "best_alpha": best_alpha[test_indices].tolist(),
            "response_density": response_density[test_indices].tolist(),
        }
        preds = {
            "status": pred_status,
            "route": pred_route,
            "best_method": pred_method,
            "best_target": pred_target,
            "best_layer": pred_layer,
            "best_alpha": pred_alpha,
            "response_density": pred_density,
        }
        for key in all_pred:
            all_pred[key].extend(preds[key])
            all_true[key].extend(truth[key])
        pca_vars.append(float(np.sum(pca.explained_variance_ratio_)))
        local_dims.append(local_dimension_summary(z_train, k=k, seed=seed + 17 * fold_idx, max_samples=local_dim_samples))
        neighbor_family_entropy.append(mean_entropy(fold_neighbor_families))
        neighbor_distance.append(float(np.mean(dist)))
        fold_rows.append({
            "fold": fold_idx,
            "heldout_families": sorted(heldout_set),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "pca_explained_variance": pca_vars[-1],
            "neighbor_family_entropy": neighbor_family_entropy[-1],
            "mean_neighbor_distance": neighbor_distance[-1],
            "status_acc": float(np.mean(np.asarray(pred_status, dtype=object) == status[test_indices])),
            "route_acc": float(np.mean(np.asarray(pred_route, dtype=object) == route[test_indices])),
            "best_method_acc": float(np.mean(np.asarray(pred_method, dtype=object) == best_method[test_indices])),
            "best_target_acc": float(np.mean(np.asarray(pred_target, dtype=object) == best_target[test_indices])),
            "response_density_mae": float(np.mean(np.abs(np.asarray(pred_density) - response_density[test_indices]))),
        })
    if not all_true["status"]:
        raise ValueError("no fold predictions")
    status_acc = float(np.mean(np.asarray(all_pred["status"], dtype=object) == np.asarray(all_true["status"], dtype=object)))
    route_acc = float(np.mean(np.asarray(all_pred["route"], dtype=object) == np.asarray(all_true["route"], dtype=object)))
    method_acc = float(np.mean(np.asarray(all_pred["best_method"], dtype=object) == np.asarray(all_true["best_method"], dtype=object)))
    target_acc = float(np.mean(np.asarray(all_pred["best_target"], dtype=object) == np.asarray(all_true["best_target"], dtype=object)))
    layer_mae = float(np.mean(np.abs(np.asarray(all_pred["best_layer"]) - np.asarray(all_true["best_layer"]))))
    alpha_mae = float(np.mean(np.abs(np.asarray(all_pred["best_alpha"]) - np.asarray(all_true["best_alpha"]))))
    density_mae = float(np.mean(np.abs(np.asarray(all_pred["response_density"]) - np.asarray(all_true["response_density"]))))
    smooth_score = float(np.mean([
        status_acc,
        route_acc,
        method_acc,
        target_acc,
        max(0.0, 1.0 - density_mae),
    ]))
    valid_local_dims = [item for item in local_dims if item.get("median") is not None]
    return {
        "n": int(len(cids)),
        "pca_dim_actual_mean": float(np.mean([min(int(pca_dim), len(cids) - 1, x.shape[1])])) if cids else None,
        "pca_explained_variance_mean": float(np.mean(pca_vars)) if pca_vars else None,
        "local_dim_participation_ratio": {
            "median_mean": float(np.mean([safe_float(item.get("median")) for item in valid_local_dims])) if valid_local_dims else None,
            "p90_mean": float(np.mean([safe_float(item.get("p90")) for item in valid_local_dims])) if valid_local_dims else None,
        },
        "neighbor_family_entropy_mean": float(np.mean(neighbor_family_entropy)) if neighbor_family_entropy else None,
        "mean_neighbor_distance": float(np.mean(neighbor_distance)) if neighbor_distance else None,
        "status_acc": status_acc,
        "route_acc": route_acc,
        "best_method_acc": method_acc,
        "best_target_acc": target_acc,
        "best_layer_mae": layer_mae,
        "best_alpha_mae": alpha_mae,
        "response_density_mae": density_mae,
        "base_space_smoothness_score": smooth_score,
        "folds": fold_rows,
    }


def extract_state_vectors(
    bank: dict,
    records: dict[str, dict[str, Any]],
    *,
    layer: int,
    turn: int,
    phase: str,
    max_cids: int,
    seed: int,
) -> dict[str, np.ndarray]:
    wanted = set(stratified_limit(records, max_cids=max_cids, seed=seed))
    points, _meta = load_activation_points_from_bank(bank, layer=layer, turns={turn}, phases={phase})
    indexed = point_index(points)
    return {
        cid: np.asarray(indexed[cid]["x"], dtype=np.float64)
        for cid in sorted(wanted & set(indexed))
    }


def rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["metrics"]
    return (
        safe_float(metrics.get("base_space_smoothness_score")),
        safe_float(metrics.get("response_density_mae")) * -1.0,
        safe_float(metrics.get("best_method_acc")),
        safe_float(metrics.get("status_acc")),
    )


def render_markdown(payload: dict[str, Any], *, top_n: int) -> str:
    top = sorted(payload["results"], key=rank_key, reverse=True)[:top_n]
    controller_available = [
        item for item in sorted(payload["results"], key=rank_key, reverse=True)
        if item["phase"] == "pre_response"
    ][:top_n]

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{safe_float(value):.4f}"

    def table_rows(items: list[dict[str, Any]]) -> list[str]:
        rows = []
        for item in items:
            m = item["metrics"]
            local = m.get("local_dim_participation_ratio", {})
            rows.append(
                f"| {item['layer']} | {item['turn']} | {item['phase']} | {item['pca_dim']} | {item['metric']} | "
                f"{fmt(m.get('base_space_smoothness_score'))} | {fmt(m.get('status_acc'))} | "
                f"{fmt(m.get('route_acc'))} | {fmt(m.get('best_method_acc'))} | "
                f"{fmt(m.get('best_target_acc'))} | {fmt(m.get('response_density_mae'))} | "
                f"{fmt(local.get('median_mean'))} | {fmt(m.get('pca_explained_variance_mean'))} |"
            )
        return rows

    lines = [
        "# Powered150 Activation Base-Space Scan",
        "",
        "CPU-only scan of candidate activation base spaces for later geometric control.",
        "This is not a controller and not generated-output evidence.",
        "",
        "## Grid",
        "",
        f"- Layers: `{payload['grid']['layers']}`",
        f"- Turns: `{payload['grid']['turns']}`",
        f"- Phases: `{payload['grid']['phases']}`",
        f"- PCA dims: `{payload['grid']['pca_dims']}`",
        f"- Metrics: `{payload['grid']['metrics']}`",
        f"- kNN k: `{payload['grid']['k']}`",
        f"- Max cids: `{payload['grid']['max_cids']}`",
        "",
        "## Top Base Spaces",
        "",
        "| layer | turn | phase | pca | metric | score | status | route | best method | best target | density MAE | local dim | var |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(table_rows(top))
    lines.extend([
        "",
        "## Top Controller-Available Base Spaces",
        "",
        "`pre_response` states are available before steering/generation. `post_response` states are diagnostic only.",
        "",
        "| layer | turn | phase | pca | metric | score | status | route | best method | best target | density MAE | local dim | var |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(table_rows(controller_available))
    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- High status/route accuracy means local neighborhoods preserve state/route information across held-out families.",
        "- High best-action agreement and low response-density MAE mean the action-response field is smooth over that base space.",
        "- Local dimension is a diagnostic of atlas complexity, not a win condition by itself.",
        "- `post_response` states can diagnose structure but are not deployable for pre-generation control.",
        "- This report should select base-space candidates for controllers; it is not a final structural-control claim.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--action-response", type=Path, default=DEFAULT_ACTION_RESPONSE)
    parser.add_argument("--activations-sha256", default=None)
    parser.add_argument("--action-response-sha256", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", default="8,12,16,19,20,24,28,32")
    parser.add_argument("--turns", default="0,1,2,3")
    parser.add_argument("--phases", default="pre_response,post_response")
    parser.add_argument("--pca-dims", default="8,16,32,64")
    parser.add_argument("--metrics", default="euclidean,cosine")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--max-cids", type=int, default=0, help="0 means all action-response cids.")
    parser.add_argument("--max-combos", type=int, default=0, help="Debug cap over layer/turn/phase/pca/metric combos.")
    parser.add_argument("--local-dim-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    action_rows, action_meta = load_action_response(args.action_response)
    records = state_records_from_action_response(action_rows)
    limited_cids = stratified_limit(records, max_cids=args.max_cids, seed=args.seed)
    records = {cid: records[cid] for cid in limited_cids}
    pseudo_rows = [
        {"conversation_id": cid, "family": record["family"], "method": "abstain"}
        for cid, record in records.items()
    ]
    folds = make_family_folds(pseudo_rows, args.n_folds)
    bank = load_activation_bank(args.activations)
    layers = parse_int_csv(args.layers)
    turns = parse_int_csv(args.turns)
    phases = parse_csv(args.phases)
    pca_dims = parse_int_csv(args.pca_dims)
    metrics = parse_csv(args.metrics)
    results = []
    combo_count = 0
    for layer in layers:
        for turn in turns:
            for phase in phases:
                x_by_cid = extract_state_vectors(
                    bank,
                    records,
                    layer=layer,
                    turn=turn,
                    phase=phase,
                    max_cids=0,
                    seed=args.seed,
                )
                if not x_by_cid:
                    continue
                for pca_dim in pca_dims:
                    for metric in metrics:
                        combo_count += 1
                        if args.max_combos and combo_count > args.max_combos:
                            break
                        metrics_out = evaluate_base_space(
                            x_by_cid,
                            records,
                            folds,
                            pca_dim=pca_dim,
                            metric=metric,
                            k=args.k,
                            seed=args.seed,
                            local_dim_samples=args.local_dim_samples,
                        )
                        results.append({
                            "layer": int(layer),
                            "turn": int(turn),
                            "phase": phase,
                            "pca_dim": int(pca_dim),
                            "metric": metric,
                            "metrics": metrics_out,
                        })
                    if args.max_combos and combo_count > args.max_combos:
                        break
                if args.max_combos and combo_count > args.max_combos:
                    break
            if args.max_combos and combo_count > args.max_combos:
                break
        if args.max_combos and combo_count > args.max_combos:
            break
    payload = {
        "schema_version": 1,
        "argv": sys.argv,
        "activations": str(args.activations),
        "activations_sha256": args.activations_sha256 or file_sha256(args.activations),
        "action_response": str(args.action_response),
        "action_response_sha256": args.action_response_sha256 or file_sha256(args.action_response),
        "action_response_meta": action_meta,
        "provenance": git_provenance([Path(__file__), args.activations, args.action_response]),
        "n_state_records": len(records),
        "folds": folds,
        "grid": {
            "layers": layers,
            "turns": turns,
            "phases": phases,
            "pca_dims": pca_dims,
            "metrics": metrics,
            "k": int(args.k),
            "max_cids": int(args.max_cids),
            "max_combos": int(args.max_combos),
        },
        "results": sorted(results, key=rank_key, reverse=True),
        "notes": [
            "This is a base-space diagnostic, not a controller.",
            "All preprocessing is fit inside held-out-family folds.",
            "Best-action labels are response-field diagnostics; do not use them as deployable inference inputs.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = args.out.with_suffix(".md")
    md.write_text(render_markdown(payload, top_n=args.top_n) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "markdown": str(md),
        "results": len(results),
        "top": payload["results"][0] if payload["results"] else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
