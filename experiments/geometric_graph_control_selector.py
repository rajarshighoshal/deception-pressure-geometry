"""Graph/diffusion chart selector for activation-state control.

This is the graph-geometric version of the chart atlas:

1. build a sparse activation graph on train-family states;
2. compute diffusion coordinates from that graph;
3. extend held-out-family states into the train diffusion coordinates;
4. fit chart-local action reward heads in diffusion space.

The graph is rebuilt inside each held-out-family fold, so test families do not
shape the graph geometry used to score their actions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.geometric_chart_atlas_selector import (  # noqa: E402
    ChartAtlasSelector,
    choose_chart,
    load_state_vectors,
    parse_chart_counts,
    parse_csv,
    parse_int_csv,
    slim_chart_choice,
    state_key,
)
from experiments.learned_generation_action_policy import (  # noqa: E402
    build_policies,
    file_sha256,
    grouped_by_conversation,
    merge_rows,
    paired_gap,
    summarize_choices,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
import geoprobe.control.atlas_action_selector as _atlas_mod  # noqa: E402  fingerprint promoted selector
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402


def parse_neighbors(value: str) -> int | str:
    value = value.strip()
    return "auto" if value == "auto" else int(value)


def unique_state_keys(rows: list[dict]) -> list[tuple[str, int]]:
    keys = {state_key(row) for row in rows if state_key(row) is not None}
    return sorted(key for key in keys if key is not None)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def resolve_neighbor_count(value: int | str, n: int) -> int:
    if n <= 1:
        return 1
    if value == "auto":
        return max(2, min(n - 1, int(math.ceil(math.sqrt(n)))))
    return max(1, min(int(value), n - 1))


def state_metadata(rows: list[dict]) -> dict[tuple[str, int], dict]:
    out = {}
    for row in rows:
        key = state_key(row)
        if key is None or key in out:
            continue
        out[key] = {
            "scenario_id": str(row.get("scenario_id", "")),
            "family": str(row.get("family", "")),
            "status_class": str(row.get("status_class", "")),
            "arm": str(row.get("arm", "")),
        }
    return out


class FoldDiffusionGraph:
    def __init__(
        self,
        *,
        graph_neighbors: int | str,
        diffusion_dim: int,
        diffusion_time: float,
        pre_pca_dim: int,
        scenario_edge_weight: float,
        seed: int,
    ) -> None:
        self.graph_neighbors_arg = graph_neighbors
        self.diffusion_dim = int(diffusion_dim)
        self.diffusion_time = float(diffusion_time)
        self.pre_pca_dim = int(pre_pca_dim)
        self.scenario_edge_weight = float(scenario_edge_weight)
        self.seed = int(seed)

    def fit(
        self,
        train_keys: list[tuple[str, int]],
        state_vectors: dict[tuple[str, int], np.ndarray],
        *,
        metadata: dict[tuple[str, int], dict],
    ) -> "FoldDiffusionGraph":
        if not train_keys:
            raise ValueError("cannot fit diffusion graph with no train states")
        self.train_keys_ = list(train_keys)
        x = np.vstack([state_vectors[key] for key in train_keys])
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)
        max_dim = min(max(1, self.pre_pca_dim), x_scaled.shape[0] - 1 if x_scaled.shape[0] > 1 else 1, x_scaled.shape[1])
        if max_dim < x_scaled.shape[1] and x_scaled.shape[0] > 1:
            self.pca_ = PCA(n_components=max_dim, whiten=True, random_state=self.seed)
            z = self.pca_.fit_transform(x_scaled)
        else:
            self.pca_ = None
            z = x_scaled
        self.train_pre_ = z
        n = z.shape[0]
        if n == 1:
            self.coords_ = np.zeros((1, max(1, self.diffusion_dim)), dtype=np.float64)
            self.sigmas_ = np.ones(1, dtype=np.float64)
            self.graph_neighbors_ = 1
            return self
        k = resolve_neighbor_count(self.graph_neighbors_arg, n)
        self.graph_neighbors_ = k
        d2 = squared_distances(z, z)
        np.fill_diagonal(d2, np.inf)
        order = np.argsort(d2, axis=1)
        kth = order[:, min(k - 1, order.shape[1] - 1)]
        sigmas = np.sqrt(np.maximum(d2[np.arange(n), kth], 1e-12))
        median_sigma = float(np.median(sigmas[np.isfinite(sigmas) & (sigmas > 1e-12)])) if np.any(np.isfinite(sigmas)) else 1.0
        sigmas[~np.isfinite(sigmas) | (sigmas <= 1e-12)] = max(median_sigma, 1e-6)
        self.sigmas_ = sigmas
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            nn = order[i, :k]
            vals = np.exp(-d2[i, nn] / np.maximum(sigmas[i] * sigmas[nn], 1e-12))
            w[i, nn] = vals
        w = np.maximum(w, w.T)
        if self.scenario_edge_weight > 0.0:
            scenarios = [metadata.get(key, {}).get("scenario_id", "") for key in train_keys]
            for i in range(n):
                for j in range(i + 1, n):
                    if scenarios[i] and scenarios[i] == scenarios[j]:
                        w[i, j] = max(w[i, j], self.scenario_edge_weight)
                        w[j, i] = max(w[j, i], self.scenario_edge_weight)
        w += np.eye(n, dtype=np.float64)
        degree = np.maximum(w.sum(axis=1), 1e-12)
        norm = w / np.sqrt(degree[:, None] * degree[None, :])
        vals, vecs = np.linalg.eigh(norm)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        start = 1 if len(vals) > 1 else 0
        end = min(start + self.diffusion_dim, len(vals))
        lambdas = np.maximum(vals[start:end], 0.0) ** self.diffusion_time
        coords = (vecs[:, start:end] / np.sqrt(degree[:, None])) * lambdas[None, :]
        if coords.shape[1] < self.diffusion_dim:
            coords = np.pad(coords, ((0, 0), (0, self.diffusion_dim - coords.shape[1])))
        self.coords_ = coords.astype(np.float64)
        return self

    def transform_raw(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler_.transform(np.asarray(x, dtype=np.float64))
        return self.pca_.transform(x_scaled) if self.pca_ is not None else x_scaled

    def transform_keys(
        self,
        keys: list[tuple[str, int]],
        state_vectors: dict[tuple[str, int], np.ndarray],
    ) -> dict[tuple[str, int], np.ndarray]:
        out = {key: self.coords_[idx] for idx, key in enumerate(self.train_keys_)}
        missing = [key for key in keys if key not in out]
        if not missing:
            return out
        z = self.transform_raw(np.vstack([state_vectors[key] for key in missing]))
        train_z = self.train_pre_
        d2 = squared_distances(z, train_z)
        k = min(self.graph_neighbors_, train_z.shape[0])
        order = np.argsort(d2, axis=1)[:, :k]
        for row_idx, key in enumerate(missing):
            nn = order[row_idx]
            test_sigma = float(np.sqrt(max(d2[row_idx, nn[-1]], 1e-12)))
            vals = np.exp(-d2[row_idx, nn] / np.maximum(test_sigma * self.sigmas_[nn], 1e-12))
            total = float(vals.sum())
            if total <= 1e-12:
                out[key] = np.zeros(self.coords_.shape[1], dtype=np.float64)
            else:
                out[key] = (vals[:, None] * self.coords_[nn]).sum(axis=0) / total
        return out


def fold_diffusion_vectors(
    train_rows: list[dict],
    test_rows: list[dict],
    raw_state_vectors: dict[tuple[str, int], np.ndarray],
    *,
    graph_neighbors: int | str,
    diffusion_dim: int,
    diffusion_time: float,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    seed: int,
) -> tuple[dict[tuple[str, int], np.ndarray], dict]:
    train_keys = [key for key in unique_state_keys(train_rows) if key in raw_state_vectors]
    test_keys = [key for key in unique_state_keys(test_rows) if key in raw_state_vectors]
    metadata = state_metadata(train_rows)
    graph = FoldDiffusionGraph(
        graph_neighbors=graph_neighbors,
        diffusion_dim=diffusion_dim,
        diffusion_time=diffusion_time,
        pre_pca_dim=pre_pca_dim,
        scenario_edge_weight=scenario_edge_weight,
        seed=seed,
    ).fit(train_keys, raw_state_vectors, metadata=metadata)
    coords = graph.transform_keys(sorted(set(train_keys + test_keys)), raw_state_vectors)
    meta = {
        "n_train_states": len(train_keys),
        "n_test_states": len(test_keys),
        "graph_neighbors": graph.graph_neighbors_,
        "diffusion_dim": diffusion_dim,
        "diffusion_time": diffusion_time,
        "pre_pca_dim": pre_pca_dim,
        "scenario_edge_weight": scenario_edge_weight,
    }
    return coords, meta


def evaluate_graph_atlas(
    rows: list[dict],
    *,
    raw_state_vectors: dict[tuple[str, int], np.ndarray],
    chart_count: int | str,
    pca_dim: int,
    top_charts: int,
    ridge_alpha: float,
    include_response_margin: bool,
    threshold: float,
    min_chart_support: float,
    min_action_support: float,
    fallbacks: list[str],
    head: str,
    graph_neighbors: int | str,
    diffusion_dim: int,
    diffusion_time: float,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    seed: int,
) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({str(row["family"]) for row in rows})
    choices = []
    folds = {}
    for family in families:
        train = [row for row in rows if str(row["family"]) != family]
        fold_rows = [row for row in rows if str(row["family"]) == family]
        state_vectors, graph_meta = fold_diffusion_vectors(
            train,
            fold_rows,
            raw_state_vectors,
            graph_neighbors=graph_neighbors,
            diffusion_dim=diffusion_dim,
            diffusion_time=diffusion_time,
            pre_pca_dim=pre_pca_dim,
            scenario_edge_weight=scenario_edge_weight,
            seed=seed,
        )
        selector = ChartAtlasSelector(
            chart_count=chart_count,
            pca_dim=pca_dim,
            top_charts=top_charts,
            ridge_alpha=ridge_alpha,
            include_response_margin=include_response_margin,
            objective="strict_reward",
            min_chart_support=min_chart_support,
            min_action_support=min_action_support,
            fallbacks=fallbacks,
            head=head,
            seed=seed,
        ).fit(train, state_vectors)
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) != family:
                continue
            fold_choices.append(choose_chart(candidates, selector, state_vectors, threshold=threshold))
        choices.extend(fold_choices)
        folds[family] = {
            "summary": summarize_choices(fold_choices),
            "graph": graph_meta,
            "chart_count": int(selector.chart_count_),
            "pca_dim": int(selector.pca_.n_components_),
            "chart_support": [float(x) for x in selector.chart_support_],
        }
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def build_graph_policies(
    rows: list[dict],
    *,
    raw_state_vectors: dict[tuple[str, int], np.ndarray],
    chart_counts: list[int | str],
    pca_dim: int,
    top_charts: int,
    ridge_alpha: float,
    threshold: float,
    min_chart_support: float,
    min_action_support: float,
    fallbacks: list[str],
    heads: list[str],
    graph_neighbors: int | str,
    diffusion_dim: int,
    diffusion_time: float,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    seed: int,
) -> dict[str, dict]:
    out = {}
    for head in heads:
        for include_response_margin, mode in [(False, "context"), (True, "response")]:
            for chart_count in chart_counts:
                count_name = str(chart_count)
                name = f"graph_{head}_{mode}_c{count_name}_g{graph_neighbors}_d{diffusion_dim}_strict"
                out[name] = evaluate_graph_atlas(
                    rows,
                    raw_state_vectors=raw_state_vectors,
                    chart_count=chart_count,
                    pca_dim=pca_dim,
                    top_charts=top_charts,
                    ridge_alpha=ridge_alpha,
                    include_response_margin=include_response_margin,
                    threshold=threshold,
                    min_chart_support=min_chart_support,
                    min_action_support=min_action_support,
                    fallbacks=fallbacks,
                    head=head,
                    graph_neighbors=graph_neighbors,
                    diffusion_dim=diffusion_dim,
                    diffusion_time=diffusion_time,
                    pre_pca_dim=pre_pca_dim,
                    scenario_edge_weight=scenario_edge_weight,
                    seed=seed,
                )
    return out


def best_policy_name(policies: dict[str, dict]) -> str:
    def key(item):
        name, value = item
        s = value["summary"]
        return (
            s["deceptive_strict_fixes"],
            s["deceptive_status_fixes"],
            -s["honest_status_harms"],
            s["parse_success"],
            name,
        )

    return max(policies.items(), key=key)[0] if policies else ""


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
    parser.add_argument("--chart-counts", default="auto,8,16")
    parser.add_argument("--chart-pca-dim", type=int, default=8)
    parser.add_argument("--top-charts", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-chart-support", type=float, default=5.0)
    parser.add_argument("--min-action-support", type=float, default=1.0)
    parser.add_argument("--fallbacks", default="full,method_layer,method,route")
    parser.add_argument("--heads", default="mean,ridge")
    parser.add_argument("--graph-neighbors", default="auto")
    parser.add_argument("--diffusion-dim", type=int, default=12)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument("--pre-pca-dim", type=int, default=32)
    parser.add_argument("--scenario-edge-weight", type=float, default=0.25)
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
    raw_state_vectors, activation_meta = load_state_vectors(
        activation_path,
        layers=layers,
        query_turn=args.query_turn,
        query_phase=args.query_phase,
    )
    graph_neighbors = parse_neighbors(args.graph_neighbors)
    baselines = build_policies(rows, threshold=args.threshold)
    graph_policies = build_graph_policies(
        rows,
        raw_state_vectors=raw_state_vectors,
        chart_counts=parse_chart_counts(args.chart_counts),
        pca_dim=args.chart_pca_dim,
        top_charts=args.top_charts,
        ridge_alpha=args.ridge_alpha,
        threshold=args.threshold,
        min_chart_support=args.min_chart_support,
        min_action_support=args.min_action_support,
        fallbacks=parse_csv(args.fallbacks),
        heads=parse_csv(args.heads),
        graph_neighbors=graph_neighbors,
        diffusion_dim=args.diffusion_dim,
        diffusion_time=args.diffusion_time,
        pre_pca_dim=args.pre_pca_dim,
        scenario_edge_weight=args.scenario_edge_weight,
        seed=args.seed,
    )
    policies = {**baselines, **graph_policies}
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
        "chart_pca_dim": args.chart_pca_dim,
        "top_charts": args.top_charts,
        "ridge_alpha": args.ridge_alpha,
        "threshold": args.threshold,
        "min_chart_support": args.min_chart_support,
        "min_action_support": args.min_action_support,
        "fallbacks": parse_csv(args.fallbacks),
        "heads": parse_csv(args.heads),
        "graph_neighbors": args.graph_neighbors,
        "diffusion_dim": args.diffusion_dim,
        "diffusion_time": args.diffusion_time,
        "pre_pca_dim": args.pre_pca_dim,
        "scenario_edge_weight": args.scenario_edge_weight,
        "n_candidate_rows": len(rows),
        "n_conversations": len(grouped_by_conversation(rows)),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if row["method"] == "baseline")),
        "best_graph_policy": best_policy_name(graph_policies),
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
            "Graph selectors rebuild a train-family activation graph per fold, "
            "embed states with diffusion coordinates, then fit chart-local action heads. "
            "Context variants exclude candidate decision-token margins; response variants include them."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    for name in sorted(graph_policies):
        s = graph_policies[name]["summary"]
        marker = "*" if name == out["best_graph_policy"] else " "
        print(
            f"{marker} {name:<46} "
            f"status={s['deceptive_status_fixes']:>2}/{s['deceptive_n']} "
            f"strict={s['deceptive_strict_fixes']:>2}/{s['deceptive_n']} "
            f"harm={s['honest_status_harms']:>2}/{s['honest_n']} "
            f"methods={dict(s['chosen_methods'])}"
        )


if __name__ == "__main__":
    main()
