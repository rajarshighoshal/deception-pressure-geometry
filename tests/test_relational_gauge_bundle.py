from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from geoprobe.control.relational_intrinsic_risk_field import PressureMatchedFieldConfig
from geoprobe.eval.relational_gauge_bundle import (
    RelationalGaugeBundleConfig,
    build_fold_gauge_controller_bundle,
)
from geoprobe.eval.relational_pre_status_risk_field import PreStatusRiskEvent
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    ExactGraphEdge,
    FoldExactRootedGraph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import RootedStarMetricScaler


VIEW = "intervention_masked_action_free"


def _edge(source: str, target: str, score: float, rank: int) -> ExactGraphEdge:
    return ExactGraphEdge(source, target, rank, score, score, score, rank)


def _graph() -> FoldExactRootedGraph:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    node_ids = tuple(f"n{i}" for i in range(len(points)))
    training = {}
    for i, source in enumerate(node_ids):
        scored = sorted(
            (
                (float(np.linalg.norm(points[i] - points[j])), target)
                for j, target in enumerate(node_ids)
                if i != j
            ),
            key=lambda item: (item[0], item[1]),
        )
        training[source] = tuple(
            _edge(source, target, score, rank)
            for rank, (score, target) in enumerate(scored, 1)
        )
    queries = {}
    for query_id, point in (
        ("q0", np.asarray([0.9, 0.1])),
        ("q1", np.asarray([-0.9, -0.1])),
    ):
        scored = sorted(
            (
                (float(np.linalg.norm(point - points[j])), target)
                for j, target in enumerate(node_ids)
            ),
            key=lambda item: (item[0], item[1]),
        )
        queries[query_id] = tuple(
            _edge(query_id, target, score, rank)
            for rank, (score, target) in enumerate(scored[:4], 1)
        )
    return FoldExactRootedGraph(
        held_out_family_fold="outer_1",
        graph_width=7,
        query_edges=queries,
        training_edges=training,
        scaler=RootedStarMetricScaler(1.0, 1.0),
        candidate_pair_count=1,
        exact_pair_count=1,
    )


def _supervision() -> SimpleNamespace:
    risks = []
    outcomes = ("HONEST", "HONEST", "DECEPTIVE", "DECEPTIVE", "SKIP", "NO_ACTION")
    for index in range(24):
        risks.append(
            PreStatusRiskEvent(
                event_id=f"e{index}",
                root_id=f"n{index % 8}",
                family=f"family{index % 4}",
                family_fold=f"outer_{2 + index % 4}",
                outcome_class=outcomes[index % len(outcomes)],
                nuisance_key=("program:A", "dose:3"),
            )
        )
    risks.extend(
        (
            PreStatusRiskEvent(
                event_id="held0",
                root_id="q0",
                family="held",
                family_fold="outer_1",
                outcome_class="DECEPTIVE",
                nuisance_key=("program:A", "dose:3"),
            ),
            PreStatusRiskEvent(
                event_id="held1",
                root_id="q1",
                family="held",
                family_fold="outer_1",
                outcome_class="HONEST",
                nuisance_key=("program:A", "dose:3"),
            ),
        )
    )
    rng = np.random.default_rng(7)
    crossings = []
    for index in range(16):
        source = f"n{index % 8}"
        target = f"n{(index * 3 + 2) % 8}"
        if target == source:
            target = f"n{(index + 1) % 8}"
        crossings.append(
            HonestwardCrossingObservation(
                pair_id=f"p{index}",
                deceptive_root_id=source,
                honest_root_id=target,
                family=f"family{index % 4}",
                family_fold=f"outer_{2 + index % 4}",
                scenario_id=f"s{index}",
                contrast_id="A",
                true_status="PASS" if index % 2 else "FAIL",
                delta=rng.normal(size=(2, 3)).astype(np.float32),
            )
        )
    return SimpleNamespace(
        risk_events_by_view={VIEW: tuple(risks)},
        honestward_observations_by_view={VIEW: tuple(crossings)},
        nodes_by_id={
            f"n{i}": SimpleNamespace(family_fold=f"outer_{2 + i % 4}")
            for i in range(8)
        },
    )


def _node_fibers() -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    maps = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            [[0.5, 0.5], [1.0, -0.5], [-0.25, 1.0]],
        ]
    )
    return {
        f"n{i}": np.einsum("lhr,r->lh", maps, point).astype(np.float32)
        for i, point in enumerate(points)
    }


def test_fold_bundle_builds_training_only_atlas_field_and_lift() -> None:
    bundle = build_fold_gauge_controller_bundle(
        _graph(),
        _supervision(),
        node_fibers=_node_fibers(),
        config=RelationalGaugeBundleConfig(
            fixed_rank=2,
            support_radius_quantile=0.9,
            support_radius_multiplier=2.0,
            minimum_chart_support=4,
            maximum_chart_support=8,
            minimum_overlap=3,
            minimum_lift_samples=4,
            field_config=PressureMatchedFieldConfig(
                minimum_support_nodes=3,
                minimum_effective_observations=2.0,
            ),
        ),
    )
    assert bundle.held_out_family_fold == "outer_1"
    assert set(bundle.atlas.node_ids) == {f"n{i}" for i in range(8)}
    assert all(
        row.family_fold != "outer_1" for row in bundle.risk_field.observations
    )
    assert bundle.horizontal_lift is not None
    assert bundle.diagnostics.lift_patch_count > 0
    assert set(bundle.held_out_queries) == {"q0", "q1"}
    assert bundle.diagnostics.held_out_query_count == 2
    assert bundle.diagnostics.held_out_query_in_support_count == 2


def test_query_attachment_uses_training_graph_paths_only() -> None:
    bundle = build_fold_gauge_controller_bundle(
        _graph(),
        _supervision(),
        node_fibers=_node_fibers(),
        config=RelationalGaugeBundleConfig(
            fixed_rank=2,
            support_radius_quantile=0.9,
            support_radius_multiplier=2.0,
            minimum_chart_support=4,
            maximum_chart_support=8,
            minimum_overlap=3,
            minimum_lift_samples=4,
            field_config=PressureMatchedFieldConfig(
                minimum_support_nodes=3,
                minimum_effective_observations=2.0,
            ),
        ),
    )
    assert bundle.held_out_queries["q0"].nearest_node_id in bundle.atlas.node_ids
    assert bundle.held_out_queries["q1"].nearest_node_id in bundle.atlas.node_ids
    assert "q0" not in bundle.atlas.node_to_index
    assert "q1" not in bundle.atlas.node_to_index
