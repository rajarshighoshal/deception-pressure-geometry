"""Fold-safe construction of the source relational gauge-controller bundle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from geoprobe.control.relational_horizontal_lift import (
    GaugeLiftSample,
    HorizontalLift,
    HorizontalLiftPatch,
    RelationalHorizontalLiftError,
    fit_horizontal_lift_patch,
)
from geoprobe.control.relational_intrinsic_risk_field import (
    GaugeRiskObservation,
    PressureMatchedFieldConfig,
    PressureMatchedRiskField,
)
from geoprobe.eval.relational_pre_status_supervision import (
    RelationalPreStatusSupervision,
)
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
    load_rooted_star_root_residuals,
)
from geoprobe.geometry.relational_gauge_atlas import (
    GaugeQueryState,
    RelationalGaugeAtlas,
    RelationalGaugeAtlasConfig,
    RelationalGaugeAtlasError,
    WeightedGraphEdge,
)
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FoldExactRootedGraph


class RelationalGaugeBundleError(ValueError):
    """A fold bundle would leak outcomes or violate its geometric contract."""


@dataclass(frozen=True, slots=True)
class RelationalGaugeBundleConfig:
    view: str = "intervention_masked_action_free"
    fixed_rank: int = 3
    support_radius_quantile: float = 0.9
    support_radius_multiplier: float = 1.1
    minimum_chart_support: int = 6
    maximum_chart_support: int = 32
    minimum_overlap: int = 4
    lift_ridge: float = 1e-3
    lift_metric_ridge: float = 1e-3
    lift_trust_quantile: float = 0.9
    lift_fiber_cap_quantile: float = 0.9
    minimum_lift_samples: int = 6
    field_config: PressureMatchedFieldConfig = PressureMatchedFieldConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.view, str) or not self.view:
            raise RelationalGaugeBundleError("view must be non-empty")
        if not isinstance(self.fixed_rank, int) or self.fixed_rank < 1:
            raise RelationalGaugeBundleError("fixed_rank must be positive")
        for name in (
            "support_radius_quantile",
            "support_radius_multiplier",
            "lift_ridge",
            "lift_metric_ridge",
            "lift_trust_quantile",
            "lift_fiber_cap_quantile",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise RelationalGaugeBundleError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not 0.5 <= self.support_radius_quantile <= 1.0:
            raise RelationalGaugeBundleError(
                "support_radius_quantile must lie in [0.5, 1]"
            )
        if not 0.5 <= self.lift_trust_quantile <= 1.0:
            raise RelationalGaugeBundleError(
                "lift_trust_quantile must lie in [0.5, 1]"
            )
        if not 0.5 <= self.lift_fiber_cap_quantile <= 1.0:
            raise RelationalGaugeBundleError(
                "lift_fiber_cap_quantile must lie in [0.5, 1]"
            )
        if self.minimum_chart_support < self.fixed_rank + 1:
            raise RelationalGaugeBundleError(
                "minimum_chart_support must exceed the chart rank"
            )
        if self.maximum_chart_support < self.minimum_chart_support:
            raise RelationalGaugeBundleError(
                "maximum_chart_support is smaller than minimum"
            )
        if self.minimum_overlap < 2 or self.minimum_lift_samples < self.fixed_rank:
            raise RelationalGaugeBundleError(
                "connection or lift support minimum is invalid"
            )


@dataclass(frozen=True, slots=True)
class FoldGaugeBundleDiagnostics:
    held_out_family_fold: str
    training_node_count: int
    undirected_edge_count: int
    chart_count: int
    connection_count: int
    support_radius: float
    chart_stress_quantiles: tuple[float, float, float]
    training_risk_event_count: int
    training_crossing_count: int
    natural_lift_sample_count: int
    lift_patch_count: int
    lift_patch_coverage: float
    held_out_query_count: int
    held_out_query_in_support_count: int
    held_out_query_field_evaluated_count: int
    held_out_query_field_defined_count: int
    held_out_query_lift_defined_count: int


@dataclass(frozen=True, slots=True)
class FoldGaugeControllerBundle:
    held_out_family_fold: str
    view: str
    atlas: RelationalGaugeAtlas
    risk_field: PressureMatchedRiskField
    horizontal_lift: HorizontalLift | None
    held_out_queries: Mapping[str, GaugeQueryState]
    diagnostics: FoldGaugeBundleDiagnostics


def _training_edges(
    graph: FoldExactRootedGraph,
) -> tuple[WeightedGraphEdge, ...]:
    unique: dict[tuple[str, str], float] = {}
    for rows in graph.training_edges.values():
        for row in rows:
            if row.source_id == row.target_id:
                continue
            key = tuple(sorted((row.source_id, row.target_id)))
            score = float(row.joint_score)
            if not math.isfinite(score) or score <= 0.0:
                raise RelationalGaugeBundleError("training graph has invalid edge score")
            unique[key] = min(unique.get(key, math.inf), score)
    if not unique:
        raise RelationalGaugeBundleError("training graph has no undirected support")
    return tuple(
        WeightedGraphEdge(source, target, score)
        for (source, target), score in sorted(unique.items())
    )


def build_training_gauge_atlas(
    graph: FoldExactRootedGraph,
    *,
    config: RelationalGaugeBundleConfig,
) -> tuple[RelationalGaugeAtlas, int, float]:
    """Build the label-free training atlas before any outcome object is joined."""
    edges = _training_edges(graph)
    scores = np.asarray([edge.weight for edge in edges], dtype=np.float64)
    support_radius = float(
        np.quantile(scores, config.support_radius_quantile)
        * config.support_radius_multiplier
    )
    atlas = RelationalGaugeAtlas(
        edges,
        config=RelationalGaugeAtlasConfig(
            fixed_rank=config.fixed_rank,
            support_radius=support_radius,
            min_chart_support=config.minimum_chart_support,
            max_chart_support=config.maximum_chart_support,
            min_overlap=config.minimum_overlap,
        ),
    )
    return atlas, len(edges), support_radius


def _risk_observations(
    supervision: RelationalPreStatusSupervision,
    *,
    view: str,
    held_out_family_fold: str,
) -> tuple[GaugeRiskObservation, ...]:
    try:
        rows = supervision.risk_events_by_view[view]
    except KeyError as error:
        raise RelationalGaugeBundleError("supervision lacks the requested view") from error
    result = tuple(
        GaugeRiskObservation(
            observation_id=row.event_id,
            node_id=row.root_id,
            family_fold=row.family_fold,
            nuisance_key=row.nuisance_key,
            outcome_class=row.outcome_class,
        )
        for row in rows
        if row.family_fold != held_out_family_fold
    )
    if not result:
        raise RelationalGaugeBundleError("fold has no training risk observations")
    return result


def _query_distances(
    atlas: RelationalGaugeAtlas,
    edges: Sequence[object],
) -> np.ndarray:
    result = np.full(len(atlas.node_ids), np.inf, dtype=np.float64)
    for edge in edges:
        target = str(getattr(edge, "target_id"))
        score = float(getattr(edge, "joint_score"))
        target_index = atlas.node_to_index.get(target)
        if target_index is None or not math.isfinite(score) or score < 0.0:
            continue
        result = np.minimum(result, score + atlas.distances[target_index])
    if not np.isfinite(result).any():
        raise RelationalGaugeBundleError("query has no path into the training atlas")
    return result


def _anchored_lift_samples(
    atlas: RelationalGaugeAtlas,
    observations: Sequence[HonestwardCrossingObservation],
    *,
    held_out_family_fold: str,
) -> tuple[tuple[GaugeLiftSample, str], ...]:
    result: list[tuple[GaugeLiftSample, str]] = []
    for row in observations:
        if row.family_fold == held_out_family_fold:
            continue
        if row.deceptive_root_id not in atlas.node_to_index:
            continue
        if row.honest_root_id not in atlas.node_to_index:
            continue
        try:
            chart_id = atlas.chart_for_node(row.deceptive_root_id)
            chart = atlas.get_chart(chart_id)
            target_index = atlas.node_to_index[row.honest_root_id]
            target = atlas.locate_query(
                atlas.distances[target_index], preferred_chart_id=chart_id
            )
        except RelationalGaugeAtlasError:
            continue
        tangent = target.query_coordinates - chart.coordinate_for(
            row.deceptive_root_id
        )
        if float(np.linalg.norm(tangent)) <= 1e-10:
            continue
        source_index = atlas.node_to_index[row.deceptive_root_id]
        geodesic_length = float(atlas.distances[source_index, target_index])
        locality_ratio = geodesic_length / max(chart.support_radius, 1e-12)
        weight = math.exp(-max(locality_ratio - 1.0, 0.0)) / (1.0 + target.stress)
        if not math.isfinite(weight) or weight <= 1e-8:
            continue
        result.append(
            (
                GaugeLiftSample(
                    sample_id=row.pair_id,
                    chart_id=chart_id,
                    family_fold=row.family_fold,
                    tangent=tangent,
                    fiber_delta=row.delta,
                    weight=weight,
                ),
                row.deceptive_root_id,
            )
        )
    return tuple(result)


def load_training_node_fibers(
    index: RelationalPreStatusRootedStarIndex,
    supervision: RelationalPreStatusSupervision,
    atlas: RelationalGaugeAtlas,
    *,
    view: str,
    held_out_family_fold: str,
) -> Mapping[str, np.ndarray]:
    """Load one mean four-layer root fiber per outcome-blind training node."""
    result: dict[str, np.ndarray] = {}
    for node_id in atlas.node_ids:
        node = supervision.nodes_by_id.get(str(node_id))
        if (
            node is None
            or node.view != view
            or node.family_fold == held_out_family_fold
            or not node.representative_references
        ):
            raise RelationalGaugeBundleError(
                "training atlas node does not bind one legal quotient fiber"
            )
        values = [
            load_rooted_star_root_residuals(index, reference)
            .detach()
            .cpu()
            .float()
            .numpy()
            for reference in node.representative_references
        ]
        shape = values[0].shape
        if any(value.shape != shape or not np.isfinite(value).all() for value in values):
            raise RelationalGaugeBundleError("training node fibers are inconsistent")
        result[str(node_id)] = np.asarray(
            np.mean(np.stack(values, axis=0, dtype=np.float64), axis=0),
            dtype=np.float32,
        )
    return result


def _natural_lift_samples(
    atlas: RelationalGaugeAtlas,
    graph: FoldExactRootedGraph,
    node_fibers: Mapping[str, np.ndarray],
    node_family_folds: Mapping[str, str],
) -> tuple[tuple[GaugeLiftSample, str], ...]:
    """Create unlabeled local secants from the frozen intrinsic graph itself."""
    result: list[tuple[GaugeLiftSample, str]] = []
    seen: set[tuple[str, str]] = set()
    for source_id in sorted(graph.training_edges):
        if source_id not in atlas.node_to_index or source_id not in node_fibers:
            continue
        try:
            chart_id = atlas.chart_for_node(source_id)
            chart = atlas.get_chart(chart_id)
        except RelationalGaugeAtlasError:
            continue
        for edge in graph.training_edges[source_id]:
            target_id = edge.target_id
            key = (source_id, target_id)
            if key in seen or target_id not in node_fibers:
                continue
            seen.add(key)
            target_index = atlas.node_to_index.get(target_id)
            if target_index is None:
                continue
            try:
                target = atlas.locate_query(
                    atlas.distances[target_index], preferred_chart_id=chart_id
                )
            except RelationalGaugeAtlasError:
                continue
            tangent = target.query_coordinates - chart.coordinate_for(source_id)
            if float(np.linalg.norm(tangent)) <= 1e-10:
                continue
            fiber_delta = np.asarray(node_fibers[target_id]) - np.asarray(
                node_fibers[source_id]
            )
            if fiber_delta.ndim != 2 or not np.isfinite(fiber_delta).all():
                raise RelationalGaugeBundleError("natural fiber delta is invalid")
            locality = float(edge.joint_score) / max(chart.support_radius, 1e-12)
            weight = 1.0 / (1.0 + locality + target.stress)
            result.append(
                (
                    GaugeLiftSample(
                        sample_id=f"natural:{source_id}:{target_id}",
                        chart_id=chart_id,
                        family_fold=node_family_folds[source_id],
                        tangent=tangent,
                        fiber_delta=fiber_delta,
                        weight=weight,
                    ),
                    source_id,
                )
            )
    return tuple(result)


def _fit_lift_bank(
    atlas: RelationalGaugeAtlas,
    anchored: Sequence[tuple[GaugeLiftSample, str]],
    *,
    config: RelationalGaugeBundleConfig,
) -> tuple[HorizontalLift | None, Mapping[str, str]]:
    by_root: dict[str, list[GaugeLiftSample]] = defaultdict(list)
    for sample, root_id in anchored:
        by_root[root_id].append(sample)
    patches: dict[str, HorizontalLiftPatch] = {}
    failures: dict[str, str] = {}
    for chart_id, chart in atlas.charts.items():
        pooled: list[GaugeLiftSample] = []
        for root_id in chart.support_ids:
            for sample in by_root.get(str(root_id), ()):
                tangent = sample.tangent
                if sample.chart_id != chart_id:
                    try:
                        connection = atlas.get_connection(sample.chart_id, chart_id)
                    except RelationalGaugeAtlasError:
                        continue
                    tangent = connection.transport @ tangent
                pooled.append(
                    GaugeLiftSample(
                        sample_id=f"{sample.sample_id}@{chart_id}",
                        chart_id=chart_id,
                        family_fold=sample.family_fold,
                        tangent=tangent,
                        fiber_delta=sample.fiber_delta,
                        weight=sample.weight,
                    )
                )
        try:
            patches[chart_id] = fit_horizontal_lift_patch(
                pooled,
                chart_id=chart_id,
                ridge=config.lift_ridge,
                metric_ridge=config.lift_metric_ridge,
                trust_quantile=config.lift_trust_quantile,
                fiber_cap_quantile=config.lift_fiber_cap_quantile,
                minimum_samples=config.minimum_lift_samples,
            )
        except RelationalHorizontalLiftError as error:
            failures[chart_id] = str(error)
    if not patches:
        return None, failures
    return HorizontalLift(patches), failures


def build_fold_gauge_controller_bundle(
    graph: FoldExactRootedGraph,
    supervision: RelationalPreStatusSupervision,
    *,
    node_fibers: Mapping[str, np.ndarray] | None = None,
    config: RelationalGaugeBundleConfig | None = None,
) -> FoldGaugeControllerBundle:
    """Build one outer-fold controller without using held-out outcomes in fitting."""
    settings = config or RelationalGaugeBundleConfig()
    fold = graph.held_out_family_fold
    atlas, edge_count, support_radius = build_training_gauge_atlas(
        graph, config=settings
    )
    risk_rows = _risk_observations(
        supervision, view=settings.view, held_out_family_fold=fold
    )
    risk_field = PressureMatchedRiskField(
        risk_rows,
        config=settings.field_config,
        held_out_family_fold=fold,
    )
    try:
        crossing_rows = supervision.honestward_observations_by_view[settings.view]
    except KeyError as error:
        raise RelationalGaugeBundleError("supervision lacks lift observations") from error
    training_crossings = tuple(
        row for row in crossing_rows if row.family_fold != fold
    )
    if node_fibers is None:
        anchored = _anchored_lift_samples(
            atlas, training_crossings, held_out_family_fold=fold
        )
    else:
        anchored = _natural_lift_samples(
            atlas,
            graph,
            node_fibers,
            {
                str(node_id): supervision.nodes_by_id[str(node_id)].family_fold
                for node_id in atlas.node_ids
            },
        )
    lift, _ = _fit_lift_bank(atlas, anchored, config=settings)

    held_out_queries: dict[str, GaugeQueryState] = {}
    query_field_evaluated = 0
    query_field_defined = 0
    query_lift_defined = 0
    heldout_events = {
        row.root_id: row
        for row in supervision.risk_events_by_view[settings.view]
        if row.family_fold == fold
    }
    for root_id, edges in graph.query_edges.items():
        try:
            query = atlas.locate_query(_query_distances(atlas, edges))
        except (RelationalGaugeAtlasError, RelationalGaugeBundleError):
            continue
        held_out_queries[root_id] = query
        event = heldout_events.get(root_id)
        if event is not None:
            query_field_evaluated += 1
            chart = atlas.get_chart(query.chart_id)
            evaluation = risk_field.evaluate(
                chart,
                query.query_coordinates,
                nuisance_key=event.nuisance_key,
            )
            query_field_defined += int(evaluation.defined)
        if lift is not None and query.chart_id in lift.patches:
            query_lift_defined += 1

    stresses = np.asarray([chart.stress for chart in atlas.charts.values()])
    patch_count = 0 if lift is None else len(lift.patches)
    diagnostics = FoldGaugeBundleDiagnostics(
        held_out_family_fold=fold,
        training_node_count=len(atlas.node_ids),
        undirected_edge_count=edge_count,
        chart_count=len(atlas.charts),
        connection_count=len(atlas.connections),
        support_radius=support_radius,
        chart_stress_quantiles=tuple(
            float(value) for value in np.quantile(stresses, (0.1, 0.5, 0.9))
        ),
        training_risk_event_count=len(risk_rows),
        training_crossing_count=len(training_crossings),
        natural_lift_sample_count=len(anchored),
        lift_patch_count=patch_count,
        lift_patch_coverage=patch_count / len(atlas.charts),
        held_out_query_count=len(held_out_queries),
        held_out_query_in_support_count=sum(
            query.support_status for query in held_out_queries.values()
        ),
        held_out_query_field_evaluated_count=query_field_evaluated,
        held_out_query_field_defined_count=query_field_defined,
        held_out_query_lift_defined_count=query_lift_defined,
    )
    return FoldGaugeControllerBundle(
        held_out_family_fold=fold,
        view=settings.view,
        atlas=atlas,
        risk_field=risk_field,
        horizontal_lift=lift,
        held_out_queries=held_out_queries,
        diagnostics=diagnostics,
    )


__all__ = [
    "FoldGaugeBundleDiagnostics",
    "FoldGaugeControllerBundle",
    "RelationalGaugeBundleConfig",
    "RelationalGaugeBundleError",
    "build_fold_gauge_controller_bundle",
    "build_training_gauge_atlas",
    "load_training_node_fibers",
]
