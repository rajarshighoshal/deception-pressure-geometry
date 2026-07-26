"""Registry for structural-control candidates that are proposed but not run."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuralCandidate:
    name: str
    status: str
    family: str
    reason: str
    first_eval_mode: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


STRUCTURAL_CANDIDATES: tuple[StructuralCandidate, ...] = (
    StructuralCandidate(
        name="equivariant_graph_atlas",
        status="not_run",
        family="graph_atlas",
        reason="Adds PASS/FAIL partner edges and Z2 action tying to the existing graph/diffusion atlas.",
        first_eval_mode="context_then_response_ceiling",
    ),
    StructuralCandidate(
        name="gauge_transport_diagnostic",
        status="not_run",
        family="gauge_transport",
        reason="Tests whether chart-local steering frames transported across overlaps predict held-out action response.",
        first_eval_mode="diagnostic_only",
    ),
    StructuralCandidate(
        name="latent_relation_graph",
        status="not_run",
        family="learned_edges",
        reason="Learns or weights edge types instead of assuming raw activation kNN is the right graph.",
        first_eval_mode="context_then_edge_ablation",
    ),
    StructuralCandidate(
        name="product_grid_graph_z2",
        status="not_run",
        family="product_geometry",
        reason="Combines layer/turn grid, activation graph, and PASS/FAIL symmetry rather than using one generic GNN.",
        first_eval_mode="context_then_response_ceiling",
    ),
)


def structural_candidate_records() -> list[dict[str, str]]:
    return [candidate.to_json() for candidate in STRUCTURAL_CANDIDATES]
