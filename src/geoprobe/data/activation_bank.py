"""Activation-bank IO: load captured activation banks and index their points.
Extracted from the control_graded_dp_frontier CLI (Phase 3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def arr(values) -> np.ndarray:
    return values.numpy() if torch.is_tensor(values) else np.asarray(values)


def load_activation_bank(path: Path) -> dict:
    """Load the full activation bank .pt file once into memory."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _bank_meta(data: dict) -> dict:
    return {
        "model_name": data.get("model_name"),
        "backend": data.get("backend"),
        "device": data.get("device"),
        "layers": data.get("layers"),
        "capture": data.get("capture"),
    }


def _extract_points(
    data: dict,
    *,
    layer: int,
    turns: set[int] | None = None,
    phases: set[str] | None = None,
    levels: set[str] | None = None,
) -> list[dict]:
    if layer not in data["activations"]:
        raise ValueError(f"layer {layer} not in activation file; available={sorted(data['activations'])}")
    cids = np.asarray(data["conversation_id"]).astype(str)
    scenarios = np.asarray(data["scenario_id"]).astype(str)
    families = np.asarray(data["family"]).astype(str)
    arms = np.asarray(data["arm"]).astype(str)
    phases_all = np.asarray(data["phase"]).astype(str)
    true_status = np.asarray(data["true_status"]).astype(str)
    desired_status = np.asarray(data["desired_status"]).astype(str)
    sample_seed = arr(data["sample_seed"]).astype(int)
    turn_index = arr(data["turn_index"]).astype(int)
    labels = arr(data["deceptive"]).astype(int)
    x = data["activations"][layer]
    x_np = x.float().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)
    points: list[dict] = []
    for idx in range(len(cids)):
        if turns is not None and int(turn_index[idx]) not in turns:
            continue
        if phases is not None and phases_all[idx] not in phases:
            continue
        if levels is not None and arms[idx] not in levels:
            continue
        vec = np.asarray(x_np[idx], dtype=np.float64)
        if not np.isfinite(vec).all():
            continue
        points.append({
            "conversation_id": cids[idx],
            "scenario_id": scenarios[idx],
            "family": families[idx],
            "arm": arms[idx],
            "sample_seed": int(sample_seed[idx]),
            "turn_index": int(turn_index[idx]),
            "phase": phases_all[idx],
            "true_status": true_status[idx],
            "desired_status": desired_status[idx],
            "label": int(labels[idx]),
            "x": vec,
        })
    return points


def load_activation_points_from_bank(
    bank: dict,
    *,
    layer: int,
    turns: set[int] | None = None,
    phases: set[str] | None = None,
    levels: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Extract layer-specific activation points from a pre-loaded bank."""
    return _extract_points(bank, layer=layer, turns=turns, phases=phases, levels=levels), _bank_meta(bank)


def load_activation_points(
    path: Path,
    *,
    layer: int,
    turns: set[int] | None = None,
    phases: set[str] | None = None,
    levels: set[str] | None = None,
) -> tuple[list[dict], dict]:
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return _extract_points(data, layer=layer, turns=turns, phases=phases, levels=levels), _bank_meta(data)


def point_index(points: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in points:
        cid = row["conversation_id"]
        if cid in out:
            duplicates.append(cid)
        out[cid] = row
    if duplicates:
        raise ValueError(f"duplicate activation points for requested turn/phase: {duplicates[:5]}")
    return out


def load_state_vectors(
    activation_path: Path,
    *,
    layers: list[int],
    query_turn: int,
    query_phase: str,
) -> tuple[dict[tuple[str, int], np.ndarray], dict]:
    out: dict[tuple[str, int], np.ndarray] = {}
    metas = {}
    bank = load_activation_bank(activation_path)
    for layer in layers:
        points, meta = load_activation_points_from_bank(
            bank,
            layer=layer,
            turns={query_turn},
            phases={query_phase},
        )
        metas[str(layer)] = meta
        for cid, point in point_index(points).items():
            out[(str(cid), int(layer))] = np.asarray(point["x"], dtype=np.float64)
    return out, metas


__all__ = ["arr", "load_activation_bank", "load_activation_points_from_bank",
           "load_activation_points", "point_index", "load_state_vectors"]
