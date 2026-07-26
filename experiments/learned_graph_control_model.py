"""CPU learned graph-action control model.

This is the first learned graph controller for the decision-token action bank:
fit a train-family activation graph, learn graph-smoothed state embeddings and
action embeddings, then score candidate actions on held-out families.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.geometric_chart_atlas_selector import (  # noqa: E402
    load_state_vectors,
    parse_csv,
    parse_int_csv,
    state_key,
)
from experiments.geometric_graph_control_selector import (  # noqa: E402
    parse_neighbors,
    resolve_neighbor_count,
    squared_distances,
    state_metadata,
    unique_state_keys,
)
from experiments.learned_generation_action_policy import (  # noqa: E402
    baseline_row,
    build_policies,
    file_sha256,
    grouped_by_conversation,
    merge_rows,
    paired_gap,
    safe_float,
    slim_choice,
    summarize_choices,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
import geoprobe.control.atlas_action_selector as _atlas_mod  # noqa: E402  fingerprint promoted selector
import geoprobe.data.activation_bank as _bank_mod  # noqa: E402


def candidate_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if str(row.get("method")) != "baseline"
        and str(row.get("route_action")) != "abstain"
        and row.get("target_status") is not None
        and row.get("layer") is not None
    ]


def action_features(row: dict, *, include_response_margin: bool) -> dict[str, str | float]:
    target = str(row.get("target_status") or "NONE")
    target_sign = 1.0 if target == "PASS" else -1.0 if target == "FAIL" else 0.0
    margin = safe_float(row.get("decision_margin"))
    out: dict[str, str | float] = {
        "method": str(row.get("method") or "unknown"),
        "route_action": str(row.get("route_action") or "unknown"),
        "target_status": target,
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "arm": str(row.get("arm") or ""),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "gate_score": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "abs_gate_score": abs(safe_float(row.get("gate_score_PASS_minus_FAIL"))),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
        "projection_fraction": safe_float(row.get("projection_fraction"), -1.0),
        "cos_to_raw": safe_float(row.get("cos_to_raw"), -1.0),
        "mean_neighbor_distance": safe_float(row.get("mean_neighbor_distance"), -1.0),
        "projected_norm": safe_float(row.get("projected_norm"), -1.0),
        "tangent_dim": safe_float(row.get("tangent_dim"), -1.0),
        "injected_norm": safe_float(row.get("injected_norm")),
        "per_layer_alpha": safe_float(row.get("per_layer_alpha")),
    }
    for key in sorted(row):
        if key.startswith("pc_"):
            out[key] = safe_float(row.get(key), 0.0)
    if include_response_margin:
        out.update({
            "decision_margin": margin,
            "abs_decision_margin": abs(margin),
            "target_aligned_decision_margin": target_sign * margin,
            "decision_forced_status": str(row.get("decision_forced_status") or "UNKNOWN"),
        })
    return out


class StatePreprocessor:
    def __init__(self, *, pre_pca_dim: int, seed: int) -> None:
        self.pre_pca_dim = int(pre_pca_dim)
        self.seed = int(seed)

    def fit(self, keys: list[tuple[str, int]], vectors: dict[tuple[str, int], np.ndarray]) -> "StatePreprocessor":
        x = np.vstack([vectors[key] for key in keys])
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(x)
        max_dim = min(max(1, self.pre_pca_dim), x_scaled.shape[0] - 1 if x_scaled.shape[0] > 1 else 1, x_scaled.shape[1])
        if max_dim < x_scaled.shape[1] and x_scaled.shape[0] > 1:
            self.pca_ = PCA(n_components=max_dim, whiten=True, random_state=self.seed)
            self.pca_.fit(x_scaled)
        else:
            self.pca_ = None
        return self

    def transform(self, keys: list[tuple[str, int]], vectors: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
        x = np.vstack([vectors[key] for key in keys])
        x_scaled = self.scaler_.transform(x)
        z = self.pca_.transform(x_scaled) if self.pca_ is not None else x_scaled
        return np.asarray(z, dtype=np.float32)


def normalized_graph(
    x: np.ndarray,
    keys: list[tuple[str, int]],
    *,
    metadata: dict[tuple[str, int], dict],
    graph_neighbors: int | str,
    scenario_edge_weight: float,
) -> tuple[np.ndarray, dict]:
    n = x.shape[0]
    if n <= 1:
        return np.eye(max(n, 1), dtype=np.float32), {"graph_neighbors": 1}
    k = resolve_neighbor_count(graph_neighbors, n)
    d2 = squared_distances(x.astype(np.float64), x.astype(np.float64))
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)
    kth = order[:, min(k - 1, order.shape[1] - 1)]
    sigmas = np.sqrt(np.maximum(d2[np.arange(n), kth], 1e-12))
    finite = sigmas[np.isfinite(sigmas) & (sigmas > 1e-12)]
    fill = float(np.median(finite)) if finite.size else 1.0
    sigmas[~np.isfinite(sigmas) | (sigmas <= 1e-12)] = max(fill, 1e-6)
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        nn = order[i, :k]
        w[i, nn] = np.exp(-d2[i, nn] / np.maximum(sigmas[i] * sigmas[nn], 1e-12))
    w = np.maximum(w, w.T)
    if scenario_edge_weight > 0.0:
        scenarios = [metadata.get(key, {}).get("scenario_id", "") for key in keys]
        for i in range(n):
            for j in range(i + 1, n):
                if scenarios[i] and scenarios[i] == scenarios[j]:
                    w[i, j] = max(w[i, j], scenario_edge_weight)
                    w[j, i] = max(w[j, i], scenario_edge_weight)
    w += np.eye(n, dtype=np.float64)
    degree = np.maximum(w.sum(axis=1), 1e-12)
    norm = w / np.sqrt(degree[:, None] * degree[None, :])
    return norm.astype(np.float32), {"graph_neighbors": int(k), "n_train_states": int(n)}


class GraphActionNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, *, hidden_dim: int, graph_layers: int, dropout: float) -> None:
        super().__init__()
        self.state_in = nn.Linear(state_dim, hidden_dim)
        self.graph_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(graph_layers)])
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.state_in(x))
        for layer in self.graph_layers:
            h = F.relu(layer(h))
        return h

    def encode_train(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.state_in(x))
        for layer in self.graph_layers:
            h_next = F.relu(layer(adj @ h))
            h = 0.5 * h + 0.5 * h_next
        return h

    def score(self, state_emb: torch.Tensor, action_x: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_net(action_x)
        joined = torch.cat([state_emb, action_emb, state_emb * action_emb], dim=-1)
        return self.score_net(joined).squeeze(-1)


class LearnedGraphSelector:
    def __init__(
        self,
        *,
        include_response_margin: bool,
        graph_neighbors: int | str,
        pre_pca_dim: int,
        scenario_edge_weight: float,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        epochs: int,
        extension_neighbors: int | str,
        extension_mix: float,
        seed: int,
    ) -> None:
        self.include_response_margin = include_response_margin
        self.graph_neighbors = graph_neighbors
        self.pre_pca_dim = int(pre_pca_dim)
        self.scenario_edge_weight = float(scenario_edge_weight)
        self.hidden_dim = int(hidden_dim)
        self.graph_layers = int(graph_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.extension_neighbors = extension_neighbors
        self.extension_mix = float(extension_mix)
        self.seed = int(seed)

    def fit(self, rows: list[dict], state_vectors: dict[tuple[str, int], np.ndarray]) -> "LearnedGraphSelector":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train = [row for row in candidate_rows(rows) if state_key(row) in state_vectors]
        if not train:
            self.model_ = None
            return self
        self.train_rows_ = train
        self.train_keys_ = [key for key in unique_state_keys(train) if key in state_vectors]
        self.state_index_ = {key: idx for idx, key in enumerate(self.train_keys_)}
        self.pre_ = StatePreprocessor(pre_pca_dim=self.pre_pca_dim, seed=self.seed).fit(self.train_keys_, state_vectors)
        x = self.pre_.transform(self.train_keys_, state_vectors)
        adj, graph_meta = normalized_graph(
            x,
            self.train_keys_,
            metadata=state_metadata(train),
            graph_neighbors=self.graph_neighbors,
            scenario_edge_weight=self.scenario_edge_weight,
        )
        self.graph_meta_ = graph_meta
        self.action_vectorizer_ = DictVectorizer(sparse=False)
        raw_action = self.action_vectorizer_.fit_transform([
            action_features(row, include_response_margin=self.include_response_margin)
            for row in train
        ])
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        self.action_scaler_ = StandardScaler()
        action_x = self.action_scaler_.fit_transform(raw_action).astype(np.float32)
        row_state_idx = np.asarray([self.state_index_[state_key(row)] for row in train], dtype=np.int64)
        y = np.asarray([1.0 if safe_float(row["strict_reward"]) > 0.0 else 0.0 for row in train], dtype=np.float32)
        pos = float(y.sum())
        neg = float(len(y) - y.sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        self.model_ = GraphActionNet(
            x.shape[1],
            action_x.shape[1],
            hidden_dim=self.hidden_dim,
            graph_layers=self.graph_layers,
            dropout=self.dropout,
        )
        xt = torch.tensor(x, dtype=torch.float32)
        at = torch.tensor(adj, dtype=torch.float32)
        action_t = torch.tensor(action_x, dtype=torch.float32)
        idx_t = torch.tensor(row_state_idx, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            state_emb = self.model_.encode_train(xt, at)
            logits = self.model_.score(state_emb[idx_t], action_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        self.model_.eval()
        with torch.no_grad():
            self.train_x_ = x
            self.train_emb_ = self.model_.encode_train(xt, at).detach().cpu().numpy()
        return self

    def _transform_action(self, rows: list[dict]) -> np.ndarray:
        raw = self.action_vectorizer_.transform([
            action_features(row, include_response_margin=self.include_response_margin)
            for row in rows
        ])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.action_scaler_.transform(raw).astype(np.float32)

    def _extend_state_embeddings(
        self,
        keys: list[tuple[str, int]],
        state_vectors: dict[tuple[str, int], np.ndarray],
    ) -> dict[tuple[str, int], np.ndarray]:
        out = {key: self.train_emb_[idx] for idx, key in enumerate(self.train_keys_)}
        missing = [key for key in keys if key not in out and key in state_vectors]
        if not missing:
            return out
        query_x = self.pre_.transform(missing, state_vectors)
        with torch.no_grad():
            raw_emb = self.model_.encode_raw(torch.tensor(query_x, dtype=torch.float32)).detach().cpu().numpy()
        d2 = squared_distances(query_x.astype(np.float64), self.train_x_.astype(np.float64))
        k = resolve_neighbor_count(self.extension_neighbors, self.train_x_.shape[0])
        order = np.argsort(d2, axis=1)[:, :k]
        for row_idx, key in enumerate(missing):
            nn = order[row_idx]
            local = np.sqrt(np.maximum(d2[row_idx, nn], 0.0))
            scale = float(np.median(local[local > 1e-12])) if np.any(local > 1e-12) else 1.0
            weights = np.exp(-local / max(scale, 1e-6))
            smooth = (weights[:, None] * self.train_emb_[nn]).sum(axis=0) / max(float(weights.sum()), 1e-12)
            out[key] = (1.0 - self.extension_mix) * raw_emb[row_idx] + self.extension_mix * smooth
        return out

    def score(self, rows: list[dict], state_vectors: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
        if self.model_ is None or not rows:
            return np.asarray([], dtype=np.float64)
        keys = [state_key(row) for row in rows if state_key(row) is not None]
        emb_by_key = self._extend_state_embeddings(keys, state_vectors)
        valid = []
        for row in rows:
            key = state_key(row)
            valid.append(key in emb_by_key if key is not None else False)
        if not any(valid):
            return np.full(len(rows), -1e9, dtype=np.float64)
        action_x = self._transform_action(rows)
        state_emb = np.vstack([
            emb_by_key[state_key(row)] if valid[idx] else np.zeros(self.hidden_dim, dtype=np.float32)
            for idx, row in enumerate(rows)
        ]).astype(np.float32)
        with torch.no_grad():
            logits = self.model_.score(torch.tensor(state_emb), torch.tensor(action_x))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs = probs.astype(np.float64)
        probs[~np.asarray(valid, dtype=bool)] = -1e9
        return probs


def choose_learned_graph(
    candidates: list[dict],
    selector: LearnedGraphSelector,
    state_vectors: dict[tuple[str, int], np.ndarray],
    *,
    threshold: float,
) -> dict:
    if str(candidates[0].get("route_action")) == "abstain":
        return baseline_row(candidates)
    pool = [row for row in candidates if str(row.get("method")) != "baseline"]
    if not pool or selector.model_ is None:
        return baseline_row(candidates)
    scores = selector.score(pool, state_vectors)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score <= threshold:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = best_score
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    chosen = dict(pool[best_idx])
    chosen["policy_score"] = best_score
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def evaluate_learned_graph(
    rows: list[dict],
    *,
    state_vectors: dict[tuple[str, int], np.ndarray],
    include_response_margin: bool,
    threshold: float,
    graph_neighbors: int | str,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    hidden_dim: int,
    graph_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    extension_neighbors: int | str,
    extension_mix: float,
    seed: int,
) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({str(row["family"]) for row in rows})
    choices = []
    folds = {}
    for fold_idx, family in enumerate(families):
        train = [row for row in rows if str(row["family"]) != family]
        selector = LearnedGraphSelector(
            include_response_margin=include_response_margin,
            graph_neighbors=graph_neighbors,
            pre_pca_dim=pre_pca_dim,
            scenario_edge_weight=scenario_edge_weight,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            extension_neighbors=extension_neighbors,
            extension_mix=extension_mix,
            seed=seed + fold_idx,
        ).fit(train, state_vectors)
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) != family:
                continue
            fold_choices.append(choose_learned_graph(candidates, selector, state_vectors, threshold=threshold))
        choices.extend(fold_choices)
        folds[family] = {
            "summary": summarize_choices(fold_choices),
            "graph": getattr(selector, "graph_meta_", {}),
        }
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def build_learned_graph_policies(
    rows: list[dict],
    *,
    state_vectors: dict[tuple[str, int], np.ndarray],
    modes: list[str],
    threshold: float,
    graph_neighbors: int | str,
    pre_pca_dim: int,
    scenario_edge_weight: float,
    hidden_dim: int,
    graph_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    extension_neighbors: int | str,
    extension_mix: float,
    seed: int,
) -> dict[str, dict]:
    out = {}
    for mode in modes:
        include_response = mode == "response"
        if mode not in {"context", "response"}:
            raise ValueError(f"unknown mode {mode!r}")
        name = f"learned_graph_{mode}_h{hidden_dim}_l{graph_layers}_g{graph_neighbors}_strict"
        out[name] = evaluate_learned_graph(
            rows,
            state_vectors=state_vectors,
            include_response_margin=include_response,
            threshold=threshold,
            graph_neighbors=graph_neighbors,
            pre_pca_dim=pre_pca_dim,
            scenario_edge_weight=scenario_edge_weight,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            extension_neighbors=extension_neighbors,
            extension_mix=extension_mix,
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
    parser.add_argument("--modes", default="context,response")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--graph-neighbors", default="auto")
    parser.add_argument("--pre-pca-dim", type=int, default=32)
    parser.add_argument("--scenario-edge-weight", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--extension-neighbors", default="auto")
    parser.add_argument("--extension-mix", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
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
    graph_neighbors = parse_neighbors(args.graph_neighbors)
    extension_neighbors = parse_neighbors(args.extension_neighbors)
    baselines = build_policies(rows, threshold=args.threshold)
    learned = build_learned_graph_policies(
        rows,
        state_vectors=state_vectors,
        modes=parse_csv(args.modes),
        threshold=args.threshold,
        graph_neighbors=graph_neighbors,
        pre_pca_dim=args.pre_pca_dim,
        scenario_edge_weight=args.scenario_edge_weight,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        extension_neighbors=extension_neighbors,
        extension_mix=args.extension_mix,
        seed=args.seed,
    )
    policies = {**baselines, **learned}
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
                "status_fix": paired_gap(policy["choices"], policies[ref]["choices"], "status_fix", seed=args.seed, bootstrap=args.bootstrap),
                "strict_fix": paired_gap(policy["choices"], policies[ref]["choices"], "strict_fix", seed=args.seed, bootstrap=args.bootstrap),
                "honest_status_harm": paired_gap(policy["choices"], policies[ref]["choices"], "honest_status_harm", seed=args.seed, bootstrap=args.bootstrap),
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
        "modes": parse_csv(args.modes),
        "threshold": args.threshold,
        "graph_neighbors": args.graph_neighbors,
        "pre_pca_dim": args.pre_pca_dim,
        "scenario_edge_weight": args.scenario_edge_weight,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "extension_neighbors": args.extension_neighbors,
        "extension_mix": args.extension_mix,
        "n_candidate_rows": len(rows),
        "n_conversations": len(grouped_by_conversation(rows)),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if row["method"] == "baseline")),
        "best_learned_graph_policy": best_policy_name(learned),
        "policies": {
            name: {
                "summary": policy["summary"],
                "folds": policy.get("folds"),
                "choices": [slim_choice(row) for row in policy["choices"]],
            }
            for name, policy in policies.items()
        },
        "paired_gaps": gaps,
        "note": (
            "Context mode excludes candidate decision-token response margins. "
            "Response mode includes them and is therefore a measured-response selector."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    for name in sorted(learned):
        s = learned[name]["summary"]
        marker = "*" if name == out["best_learned_graph_policy"] else " "
        print(
            f"{marker} {name:<46} "
            f"status={s['deceptive_status_fixes']:>2}/{s['deceptive_n']} "
            f"strict={s['deceptive_strict_fixes']:>2}/{s['deceptive_n']} "
            f"harm={s['honest_status_harms']:>2}/{s['honest_n']} "
            f"methods={dict(s['chosen_methods'])}"
        )


if __name__ == "__main__":
    main()
