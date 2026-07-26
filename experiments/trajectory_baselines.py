"""Trajectory-level probe baselines for SYCON."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.label_sycon_flips import trajectory_type
from geoprobe.geometry.trajectories import featurize_paths
from geoprobe.probes.registry import FAMILY, build_probe
from geoprobe.probes.torch_path import (  # noqa: E402,F401  re-export full torch-probe surface (compat shim)
    TorchBinaryProbe,
    TorchDiagonalRiemannianPathProbe,
    torch_cross_validated_auroc,
    torch_train_scores,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET_TYPES = {"sycophantic_flip": 1, "steadfast_correct": 0}
GEOMETRY_FEATURES = [
    "final",
    "mean",
    "delta",
    "path_flat",
    "relative_path",
    "centered_path",
    "velocity",
    "acceleration",
    "direction",
    "gram",
    "distances",
    "path_stats",
    "curvature",
    "curve_summary",
]
GEOMETRY_PROBES = [
    "linear",
    "torch_mlp",
    "mahalanobis",
    "class_mahalanobis",
    "centroid",
    "knn",
    "tangent_subspace",
    "graph_geodesic",
    "torch_riemannian",
]
DEFAULT_FEATURES = "geometry_full"
DEFAULT_PROBES = "geometry_full"
TORCH_PROBES = {"torch_linear", "torch_mlp", "torch_riemannian"}
PATH_FEATURES = {"path_flat", "relative_path", "centered_path"}


@dataclass(frozen=True)
class TrajectoryDataset:
    turns: dict
    labels: np.ndarray
    groups: np.ndarray
    conversation_ids: list[str]
    conversation_rows: dict[str, list[tuple[int, int]]]

    @property
    def n_sycophantic_flip(self) -> int:
        return int(self.labels.sum())

    @property
    def n_steadfast_correct(self) -> int:
        return int((1 - self.labels).sum())


def parse_csv(value: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    if aliases and value in aliases:
        return aliases[value]
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


from geoprobe.provenance import git_provenance  # noqa: E402,F401  re-export; canonical def in src


def load_stances(labels_path: Path) -> dict[str, dict[int, str]]:
    stances: dict[str, dict[int, str]] = defaultdict(dict)
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stances[row["conversation_id"]][int(row["turn_index"])] = row["stance"]
    return dict(stances)


def group_activation_rows(turns: dict, phase: str = "post_response") -> dict[str, list[tuple[int, int]]]:
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    turn_indices = turns["turn_index"].tolist()
    phases = turns.get("phase")
    if phases is None:
        phases = [phase] * len(turn_indices)
    for row_idx, (conversation_id, turn_index, row_phase) in enumerate(
        zip(turns["conversation_id"], turn_indices, phases)
    ):
        if row_phase != phase:
            continue
        rows[conversation_id].append((int(turn_index), row_idx))
    return {cid: sorted(items) for cid, items in rows.items()}


def paired_group_id(conversation_id: str) -> str:
    """Keep neutral/pressured arms of one synthetic scenario in one fold."""
    stem, sep, suffix = conversation_id.rpartition("_")
    if sep and suffix in {"n", "p"}:
        return stem
    return conversation_id


def load_dataset(config: dict, labels_path: Path, phase: str = "post_response") -> TrajectoryDataset:
    turns_path = Path(config["activations"]["output_dir"]) / "turns.pt"
    turns = torch.load(turns_path, map_location="cpu", weights_only=False)
    stances = load_stances(labels_path)
    conversation_rows = group_activation_rows(turns, phase=phase)

    labels: list[int] = []
    groups: list[str] = []
    conversation_ids: list[str] = []
    for conversation_id, turnmap in stances.items():
        if conversation_id not in conversation_rows:
            continue
        ordered_stances = [turnmap[t] for t in sorted(turnmap)]
        label = TARGET_TYPES.get(trajectory_type(ordered_stances))
        if label is None:
            continue
        conversation_ids.append(conversation_id)
        labels.append(label)
        groups.append(paired_group_id(conversation_id))

    return TrajectoryDataset(
        turns=turns,
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups),
        conversation_ids=conversation_ids,
        conversation_rows=conversation_rows,
    )


def layer_paths(dataset: TrajectoryDataset, layer: int) -> list[np.ndarray]:
    activations = dataset.turns["activations"][layer].numpy().astype(np.float32)
    paths = []
    for conversation_id in dataset.conversation_ids:
        rows = dataset.conversation_rows[conversation_id]
        paths.append(np.stack([activations[row_idx] for _, row_idx in rows]))
    return paths


def featurize(paths: list[np.ndarray], feature_name: str) -> np.ndarray:
    return featurize_paths(paths, feature_name)


def positive_scores(probe, features: np.ndarray) -> np.ndarray:
    if hasattr(probe, "decision_function"):
        return probe.decision_function(features)
    classes = probe.steps[-1][1].classes_ if hasattr(probe, "steps") else probe.classes_
    return probe.predict_proba(features)[:, list(classes).index(1)]


def cross_validated_auroc(
    probe_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    torch_epochs: int,
    torch_learning_rate: float,
    torch_hidden_features: int,
    path_shape: tuple[int, int] | None = None,
) -> float | None:
    if probe_name in TORCH_PROBES:
        return torch_cross_validated_auroc(
            probe_name,
            features,
            labels,
            groups,
            device,
            torch_epochs,
            torch_learning_rate,
            torch_hidden_features,
            path_shape,
        )

    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()), len(set(groups.tolist())))
    if n_splits < 2:
        return None

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    out_of_fold = np.full(len(labels), np.nan)
    for train_idx, test_idx in splitter.split(features, labels, groups):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        try:
            probe = build_probe(probe_name)
            probe.fit(features[train_idx], labels[train_idx])
            out_of_fold[test_idx] = positive_scores(probe, features[test_idx])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

    valid = ~np.isnan(out_of_fold)
    if valid.sum() == 0 or len(set(labels[valid].tolist())) < 2:
        return None
    return float(roc_auc_score(labels[valid], out_of_fold[valid]))


def probe_family(probe_name: str) -> str | None:
    if probe_name == "torch_riemannian":
        return "riemannian_approx"
    if probe_name in TORCH_PROBES:
        return "euclidean"
    return FAMILY.get(probe_name)


def probe_applies(probe_name: str, feature_name: str) -> bool:
    if probe_name == "torch_riemannian":
        return feature_name in PATH_FEATURES
    return True


def best_rows(results: dict, probes: list[str]) -> list[dict]:
    rows = []
    for feature_name, by_layer in results.items():
        for probe_name in probes:
            candidates = []
            for layer, layer_result in by_layer.items():
                score = layer_result["probes"].get(probe_name)
                if isinstance(score, float):
                    candidates.append((int(layer), score))
            best_layer, best_auroc = max(candidates, key=lambda item: item[1], default=(None, None))
            rows.append(
                {
                    "feature": feature_name,
                    "probe": probe_name,
                    "family": probe_family(probe_name),
                    "best_layer": best_layer,
                    "best_auroc": best_auroc,
                }
            )
    return rows


def infer_scope(labels_path: Path) -> str:
    return "kcfalse" if "kcfalse" in labels_path.name else "all"


def run(
    config: dict,
    labels_path: Path,
    feature_names: list[str],
    probe_names: list[str],
    device: torch.device,
    torch_epochs: int,
    torch_learning_rate: float,
    torch_hidden_features: int,
    phase: str = "post_response",
) -> dict:
    dataset = load_dataset(config, labels_path, phase=phase)
    by_feature: dict[str, dict] = {name: {} for name in feature_names}
    total_jobs = sum(
        1
        for _ in dataset.turns["layers"]
        for feature_name in feature_names
        for probe_name in probe_names
        if probe_applies(probe_name, feature_name)
    )
    completed_jobs = 0
    started_at = time.perf_counter()

    for layer in dataset.turns["layers"]:
        paths = layer_paths(dataset, int(layer))
        for feature_name in feature_names:
            features = featurize(paths, feature_name)
            path_shape = paths[0].shape if feature_name in PATH_FEATURES else None
            layer_result = {
                "n_features": int(features.shape[1]),
                "path_shape": list(path_shape) if path_shape is not None else None,
                "probes": {},
            }
            for probe_name in probe_names:
                if not probe_applies(probe_name, feature_name):
                    continue
                completed_jobs += 1
                job_started_at = time.perf_counter()
                print(
                    f"[{completed_jobs:03d}/{total_jobs:03d}] "
                    f"L{int(layer):02d} {feature_name}:{probe_name} "
                    f"(n={len(dataset.labels)}, d={features.shape[1]})",
                    flush=True,
                )
                try:
                    auroc = cross_validated_auroc(
                        probe_name,
                        features,
                        dataset.labels,
                        dataset.groups,
                        device,
                        torch_epochs,
                        torch_learning_rate,
                        torch_hidden_features,
                        path_shape,
                    )
                    layer_result["probes"][probe_name] = (
                        round(auroc, 4) if auroc is not None else None
                    )
                    elapsed = time.perf_counter() - job_started_at
                    score = f"{auroc:.4f}" if auroc is not None else "n/a"
                    print(f"    -> {score} in {elapsed:.1f}s", flush=True)
                except Exception as exc:
                    layer_result["probes"][probe_name] = {
                        "error": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                    elapsed = time.perf_counter() - job_started_at
                    print(f"    -> ERROR {type(exc).__name__} in {elapsed:.1f}s", flush=True)
            by_feature[feature_name][str(layer)] = layer_result

    print(f"finished {total_jobs} jobs in {time.perf_counter() - started_at:.1f}s", flush=True)

    return {
        "model_name": dataset.turns.get("model_name"),
        "torch_device": str(device),
        "phase": phase,
        "n": int(len(dataset.labels)),
        "n_sf": dataset.n_sycophantic_flip,
        "n_sc": dataset.n_steadfast_correct,
        "by_feature": by_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--scope")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--probes", default=DEFAULT_PROBES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", default="post_response", choices=["pre_response", "post_response"])
    parser.add_argument("--torch-epochs", type=int, default=250)
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    parser.add_argument("--torch-hidden", type=int, default=64)
    args = parser.parse_args()

    config_path = Path(args.config)
    labels_path = Path(args.labels)
    config = yaml.safe_load(config_path.read_text())
    feature_names = parse_csv(args.features, {"geometry_full": GEOMETRY_FEATURES})
    probe_names = parse_csv(args.probes, {"geometry_full": GEOMETRY_PROBES})
    device = resolve_device(args.device)

    result = run(
        config,
        labels_path,
        feature_names,
        probe_names,
        device,
        args.torch_epochs,
        args.torch_lr,
        args.torch_hidden,
        args.phase,
    )
    result.update(
        {
            "config": config["name"],
            "labels_path": str(labels_path),
            "scope": args.scope or infer_scope(labels_path),
            "task": "sycophantic_flip_vs_steadfast_correct",
            "phase": args.phase,
            "grouping": "paired_scenario_id",
            "features": feature_names,
            "probes": probe_names,
        }
    )
    result["summary"] = best_rows(result["by_feature"], probe_names)

    output_dir = Path(config["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trajectory_baselines_{result['scope']}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"\n{config['name']} {result['scope']}: n={result['n']} sf={result['n_sf']} sc={result['n_sc']}")
    top_rows = sorted(
        result["summary"],
        key=lambda row: (row["best_auroc"] is None, -(row["best_auroc"] or -1.0)),
    )[:12]
    for row in top_rows:
        score = row["best_auroc"] if row["best_auroc"] is not None else "n/a"
        print(f"{row['feature']:10s} {row['probe']:14s} {score} @ L{row['best_layer']}")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
