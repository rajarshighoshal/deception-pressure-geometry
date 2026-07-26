"""Product graph construction for activation-state/action control maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geoprobe.control import geometry_map as gm


@dataclass(frozen=True)
class ProductGraph:
    node_ids: list[str]
    node_types: list[str]
    edge_index: np.ndarray
    edge_types: list[str]
    action_node_for_row: dict[int, int]


def build_product_z2_graph(candidates: list[dict], *, include_z2: bool = True) -> ProductGraph:
    """Build a small inspectable graph for one state's candidate actions."""
    node_ids = ["state", "route", "chart"]
    node_types = ["state", "route", "chart"]
    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []

    def add_edge(src: int, dst: int, edge_type: str) -> None:
        edges.append((src, dst))
        edge_types.append(edge_type)

    action_node_for_row: dict[int, int] = {}
    key_to_node: dict[gm.ActionKey, int] = {}
    for idx, row in enumerate(candidates):
        if gm.is_baseline(row):
            continue
        node = len(node_ids)
        action_node_for_row[idx] = node
        key_to_node[gm.compact_action_key(row)] = node
        node_ids.append(f"action:{idx}:{gm.compact_action_key(row)}")
        node_types.append("action")
        for base_node, edge_type in [(0, "state_action"), (1, "route_action"), (2, "chart_action")]:
            add_edge(base_node, node, edge_type)
            add_edge(node, base_node, edge_type)

    if include_z2:
        for idx, row in enumerate(candidates):
            node = action_node_for_row.get(idx)
            partner = gm.z2_partner_key(row)
            partner_node = key_to_node.get(partner)
            if node is not None and partner_node is not None:
                add_edge(node, partner_node, "z2_partner")

    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T
    return ProductGraph(
        node_ids=node_ids,
        node_types=node_types,
        edge_index=edge_index,
        edge_types=edge_types,
        action_node_for_row=action_node_for_row,
    )


def z2_partner_feature_rows(candidates: list[dict], base_scores: np.ndarray) -> list[dict[str, float]]:
    """Return partner-score features aligned to `candidates`."""
    by_key = {
        gm.compact_action_key(row): float(base_scores[idx])
        for idx, row in enumerate(candidates)
        if not gm.is_baseline(row) and idx < len(base_scores)
    }
    out = []
    for idx, row in enumerate(candidates):
        partner = gm.z2_partner_key(row)
        partner_score = by_key.get(partner, 0.0)
        has_partner = 1.0 if partner in by_key else 0.0
        own_score = float(base_scores[idx]) if idx < len(base_scores) else 0.0
        out.append({
            "z2_has_partner": has_partner,
            "z2_partner_score": partner_score,
            "z2_score_gap": own_score - partner_score,
        })
    return out
