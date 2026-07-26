"""Tangent-geometry primitives for on-manifold steering (dp direction, local tangent bases,
off-tangent projection). Extracted from the control_graded_dp_frontier CLI."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.distance import cdist

from geoprobe.geometry.numerics import clean_matrix


def unit(vec: np.ndarray) -> np.ndarray:
    vec = clean_matrix(np.asarray(vec, dtype=np.float64))
    scale = float(np.max(np.abs(vec))) if vec.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(vec)
    scaled = vec / scale
    norm = float(np.linalg.norm(scaled))
    return scaled / norm if np.isfinite(norm) and norm > 1e-12 else np.zeros_like(vec)


def vector_stats(vec: np.ndarray | torch.Tensor) -> dict:
    arr = vec.detach().float().cpu().numpy() if torch.is_tensor(vec) else np.asarray(vec, dtype=np.float64)
    arr = clean_matrix(arr)
    return {
        "norm": float(np.linalg.norm(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max_abs": float(np.abs(arr).max()),
    }


def fit_dp_direction(
    direction_points: list[dict],
    *,
    heldout_family: str,
    direction_levels: set[str],
    min_mixed_scenarios: int,
    min_levels: int,
) -> dict | None:
    train = [
        row for row in direction_points
        if row["family"] != heldout_family and row["arm"] in direction_levels
    ]
    by_level_scenario: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        by_level_scenario[(row["arm"], row["scenario_id"])].append(row)

    level_diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    scenario_counts: dict[str, list[dict]] = defaultdict(list)
    for (level, scenario_id), group in sorted(by_level_scenario.items()):
        y = np.asarray([row["label"] for row in group], dtype=int)
        if int(y.sum()) < 1 or int(len(y) - y.sum()) < 1:
            continue
        x = clean_matrix(np.vstack([row["x"] for row in group]))
        diff = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        level_diffs[level].append(diff)
        scenario_counts[level].append({
            "scenario_id": scenario_id,
            "n_deceptive": int(y.sum()),
            "n_honest": int(len(y) - y.sum()),
        })

    usable_levels = [level for level in sorted(level_diffs) if len(level_diffs[level]) >= min_mixed_scenarios]
    if len(usable_levels) < min_levels:
        return None

    level_means = [clean_matrix(np.vstack(level_diffs[level])).mean(axis=0) for level in usable_levels]
    dp = unit(clean_matrix(np.vstack(level_means)).mean(axis=0))
    if not np.isfinite(dp).all() or np.linalg.norm(dp) <= 1e-8:
        return None

    direction = -dp
    return {
        "heldout_family": heldout_family,
        "direction_levels": usable_levels,
        "n_train_points": int(len(train)),
        "n_mixed_scenario_level_pairs": int(sum(len(level_diffs[level]) for level in usable_levels)),
        "mixed_scenarios_by_level": {
            level: scenario_counts[level][:100]
            for level in usable_levels
        },
        "dp_stats": vector_stats(dp),
        "direction_convention": "direction = honest - deceptive = -d_p",
        "direction_stats": vector_stats(direction),
        "direction": torch.from_numpy(direction.astype(np.float32)),
        "_direction_np": direction,
    }


def fit_tangent_cloud(
    tangent_points: list[dict],
    *,
    heldout_family: str | None = None,
    heldout_scenario_ids: set[str] | None = None,
) -> dict | None:
    train = [
        row for row in tangent_points
        if (heldout_family is None or row["family"] != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if len(train) < 4:
        return None
    x = clean_matrix(np.vstack([row["x"] for row in train]))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return {
        "heldout_family": heldout_family,
        "n_train": int(len(train)),
        "_train_x": x,
        "_train_scaled": (x - mean) / std,
        "_scale_mean": mean,
        "_scale_std": std,
    }


# Relative gap below which the k-th/(k+1)-th tangent-neighbor distances are
# treated as a tie and exact per-row distances are recomputed (cdist vs
# np.linalg.norm agree to ~1e-16; 1e-9 is far above that and far below any real
# neighbor gap on continuous activation clouds).
_TANGENT_TIE_TOL = 1e-9


def _tangent_basis_from_distances(
    train_scaled: np.ndarray,
    distances: np.ndarray,
    *,
    k: int,
    tangent_dim: int,
) -> tuple[np.ndarray | None, dict]:
    """Build the local tangent basis from one query's distances to the cloud.

    This is the part that depends ONLY on (query, cloud) -- not on the steering
    direction -- so it can be computed once per (row, layer) and reused for every
    method/target. The returned ``base_info`` carries neighbor ids/distances and
    singular values for spec-bank provenance (underscore keys are internal and are
    dropped from the public projection info). Returns (basis, base_info) or
    (None, reason). The basis (right singular vectors) is invariant to neighbor
    row order, so it does not matter whether neighbors come from cdist or a
    per-row norm as long as the top-k SET matches.
    """
    idx = np.argsort(distances)[:k]
    local_scaled = train_scaled[idx]
    local = clean_matrix(local_scaled - local_scaled.mean(axis=0, keepdims=True))
    max_dim = min(int(tangent_dim), local.shape[0] - 1, local.shape[1])
    if max_dim < 1:
        return None, {"reason": "no_tangent_dim", "neighbors": int(k)}
    try:
        _, singular_values, vh = np.linalg.svd(local, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, {"reason": "svd_failed", "neighbors": int(k), "tangent_dim": int(max_dim)}
    basis = clean_matrix(vh[:max_dim])
    return basis, {
        "neighbors": int(k),
        "tangent_dim": int(max_dim),
        "mean_neighbor_distance": float(distances[idx].mean()),
        "singular_values": [float(x) for x in singular_values[:max_dim]],
        "_neighbor_idx": idx,
        "_neighbor_distances": distances[idx],
    }


def _project_direction_onto_basis(
    raw_direction: np.ndarray,
    basis: np.ndarray,
    scale_std: np.ndarray,
    base_info: dict,
) -> tuple[torch.Tensor | None, dict]:
    """Project one raw steering direction onto a precomputed local tangent basis.

    This is the part that depends on the direction (method/target); it reuses the
    basis from ``_tangent_basis_from_distances``. Output is byte-for-byte the same
    dict the monolithic ``project_to_local_tangent`` returned.
    """
    raw_unit = unit(raw_direction)
    raw_scaled = clean_matrix(raw_unit / scale_std)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected_scaled = clean_matrix(basis.T @ (basis @ raw_scaled))
        projected = clean_matrix(projected_scaled * scale_std)
    projected_norm = float(np.linalg.norm(projected))
    raw_norm = float(np.linalg.norm(raw_unit))
    if not np.isfinite(projected_norm) or projected_norm < 1e-8:
        return None, {
            "reason": "zero_projection",
            "neighbors": base_info["neighbors"],
            "tangent_dim": base_info["tangent_dim"],
            "raw_norm": raw_norm,
            "projected_norm": projected_norm,
        }
    projected_unit = projected / projected_norm
    return torch.from_numpy(projected_unit.astype(np.float32)), {
        "neighbors": base_info["neighbors"],
        "tangent_dim": base_info["tangent_dim"],
        "raw_norm": raw_norm,
        "projected_norm": projected_norm,
        "projection_fraction": float(projected_norm / max(raw_norm, 1e-12)),
        "cos_to_raw": float(np.dot(projected_unit, raw_unit / max(raw_norm, 1e-12))),
        "mean_neighbor_distance": base_info["mean_neighbor_distance"],
        "singular_values": base_info["singular_values"],
    }


def project_to_local_tangent(
    raw_direction: np.ndarray,
    tangent_info: dict,
    query: np.ndarray,
    *,
    tangent_neighbors: int,
    tangent_dim: int,
) -> tuple[torch.Tensor | None, dict]:
    """Single-query local tangent projection (unchanged behavior).

    Kept bit-identical to the original (per-row ``np.linalg.norm`` distances); the
    batched ``local_tangent_bases`` is the fast path for many rows of one family.
    """
    train_scaled = tangent_info["_train_scaled"]
    q = clean_matrix(np.asarray(query, dtype=np.float64))
    q_scaled = (q - tangent_info["_scale_mean"]) / tangent_info["_scale_std"]
    k = min(int(tangent_neighbors), len(tangent_info["_train_x"]))
    if k < 2:
        return None, {"reason": "too_few_neighbors", "neighbors": int(k)}
    distances = np.linalg.norm(train_scaled - q_scaled[None, :], axis=1)
    basis, base_info = _tangent_basis_from_distances(train_scaled, distances, k=k, tangent_dim=tangent_dim)
    if basis is None:
        return None, base_info
    return _project_direction_onto_basis(raw_direction, basis, tangent_info["_scale_std"], base_info)


def local_tangent_bases(
    tangent_info: dict,
    queries: np.ndarray,
    *,
    tangent_neighbors: int,
    tangent_dim: int,
    query_chunk: int = 128,
) -> list[tuple[np.ndarray | None, dict]]:
    """Batched local tangent bases for many query points of ONE family/cloud.

    Replaces the per-row ``np.linalg.norm`` over the whole cloud (a multi-GB temp
    per call) with one chunked ``cdist`` per block of queries -- the kNN-loop
    bottleneck that left the A100 idle. Each result is reusable for any steering
    direction via ``project_direction_onto_tangent_basis``. Output (basis +
    projection) is identical to calling ``project_to_local_tangent`` per row:
    cdist matches the per-row norm to ~1e-13 and the basis is invariant to
    neighbor order.
    """
    train_scaled = tangent_info["_train_scaled"]
    mean = tangent_info["_scale_mean"]
    std = tangent_info["_scale_std"]
    Q = clean_matrix(np.asarray(queries, dtype=np.float64))
    if Q.ndim == 1:
        Q = Q[None, :]
    Q_scaled = (Q - mean) / std
    n_train = len(tangent_info["_train_x"])
    k = min(int(tangent_neighbors), n_train)
    results: list[tuple[np.ndarray | None, dict]] = []
    for start in range(0, Q_scaled.shape[0], int(query_chunk)):
        block = Q_scaled[start:start + int(query_chunk)]
        if k < 2:
            for _ in range(block.shape[0]):
                results.append((None, {"reason": "too_few_neighbors", "neighbors": int(k)}))
            continue
        dmat = cdist(block, train_scaled, metric="euclidean")
        for i in range(block.shape[0]):
            d = dmat[i]
            # cdist and the per-row np.linalg.norm reference agree only to ~1 ULP.
            # If the k-th and (k+1)-th nearest distances are within tolerance, the
            # top-k SET could differ from the canonical per-row selection (and so
            # could the steering vector). That requires near-duplicate cloud points
            # (~0 on continuous activation data) -- recompute exact distances there
            # so the basis is bit-identical to project_to_local_tangent.
            if n_train > k:
                kth = np.partition(d, [k - 1, k])
                if (kth[k] - kth[k - 1]) <= _TANGENT_TIE_TOL * max(1.0, float(kth[k])):
                    d = np.linalg.norm(train_scaled - block[i][None, :], axis=1)
            results.append(_tangent_basis_from_distances(train_scaled, d, k=k, tangent_dim=tangent_dim))
    return results


def project_direction_onto_tangent_basis(
    raw_direction: np.ndarray,
    tangent_info: dict,
    basis: np.ndarray,
    base_info: dict,
) -> tuple[torch.Tensor | None, dict]:
    """Project a raw direction onto a precomputed (batched) tangent basis.

    Public wrapper of ``_project_direction_onto_basis`` so callers (precompute,
    runner) reuse one basis for both tangent methods and both Z2 targets.
    """
    return _project_direction_onto_basis(raw_direction, basis, tangent_info["_scale_std"], base_info)


def off_tangent_direction(raw_direction: np.ndarray, tangent_direction: torch.Tensor | None) -> torch.Tensor | None:
    if tangent_direction is None:
        return None
    tangent = tangent_direction.detach().float().cpu().numpy().astype(np.float64)
    residual = clean_matrix(unit(raw_direction) - tangent * float(np.dot(unit(raw_direction), tangent)))
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return torch.from_numpy((residual / norm).astype(np.float32))


__all__ = ["unit", "vector_stats", "fit_dp_direction", "fit_tangent_cloud", "local_tangent_bases",
           "project_to_local_tangent", "project_direction_onto_tangent_basis", "off_tangent_direction"]
