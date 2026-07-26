from __future__ import annotations

from geoprobe.control.structural_candidates import structural_candidate_records


def test_structural_candidate_registry_is_explicit_not_run():
    records = structural_candidate_records()
    names = {row["name"] for row in records}
    assert {
        "equivariant_graph_atlas",
        "gauge_transport_diagnostic",
        "latent_relation_graph",
        "product_grid_graph_z2",
    } <= names
    assert all(row["status"] == "not_run" for row in records)
    assert all(row["first_eval_mode"] for row in records)
