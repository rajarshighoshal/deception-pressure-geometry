from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoprobe.data.relational_pre_status_rooted_star import LAYERS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RootedStarObservationBinding,
)
from geoprobe.geometry.relational_gauge_atlas import GaugeChart, GaugeQueryState
from geoprobe.models.relational_step_capture import RelationalStepCapture
from geoprobe.runtime.relational_gauge_replay_observation import (
    CaptureBoundPreStatusGaugeObservationBuilder,
    FrozenPreStatusGaugeReplayRow,
    RelationalGaugeReplayObservationError,
)


def _bf16(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).to(torch.float32).numpy().copy()


def _fixture() -> tuple[
    CaptureBoundPreStatusGaugeObservationBuilder,
    RelationalStepCapture,
]:
    chart = GaugeChart(
        chart_id="chart",
        center_node_id="train-a",
        support_ids=("train-a", "train-b"),
        support_distances=((0.0, 1.0), (1.0, 0.0)),
        coordinates=np.asarray([[0.0], [1.0]]),
        eigenvalues=np.asarray([1.0]),
        stress=0.0,
        support_radius=1.0,
    )
    query = GaugeQueryState(
        chart_id="chart",
        query_coordinates=np.asarray([0.25]),
        nearest_node_id="train-a",
        nearest_node_distance=0.25,
        stress=0.0,
        support_status=True,
        support_reason="sealed test query",
    )

    class Atlas:
        @staticmethod
        def get_chart(chart_id: str) -> GaugeChart:
            assert chart_id == "chart"
            return chart

    class Lift:
        @staticmethod
        def patch(chart_id: str) -> object:
            assert chart_id == "chart"
            return object()

    root = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    full_attention = torch.tensor(
        [[0.25, 0.50, 0.25], [0.60, 0.20, 0.20]], dtype=torch.float32
    )
    retained = full_attention[:, [0, 2]]
    retained /= retained.sum(dim=1, keepdim=True)
    binding = RootedStarObservationBinding(
        rooted_star_id="rooted-star",
        reference_id="reference",
        view="intervention_masked_action_free",
        geometry_sha256="a" * 64,
        prefix_token_count=3,
        retained_token_indices=np.asarray([0, 2]),
        root_residuals=_bf16(root),
        incoming_attention=np.stack([_bf16(retained) for _ in LAYERS]),
    )
    row = FrozenPreStatusGaugeReplayRow(
        row_id="row",
        held_out_family_fold="outer_1",
        root_id="held-out-root",
        expected_token_ids=(1, 2583, 25),
        nuisance_key=("0", "[]", "no_pressure", "PASS", "PASS"),
        rooted_stars=(binding,),
    )
    bundle = SimpleNamespace(
        horizontal_lift=Lift(),
        held_out_family_fold="outer_1",
        view="intervention_masked_action_free",
        held_out_queries={"held-out-root": query},
        atlas=Atlas(),
    )
    builder = CaptureBoundPreStatusGaugeObservationBuilder(
        bundle=bundle,
        rows=(row,),
        controller_artifact_sha256="b" * 64,
    )
    capture = RelationalStepCapture(
        residuals={
            layer: root[index].reshape(1, 1, 3).clone()
            for index, layer in enumerate(LAYERS)
        },
        attentions={
            layer: full_attention.reshape(1, 2, 1, 3).clone()
            for layer in LAYERS
        },
    )
    return builder, capture


def test_exact_live_capture_builds_artifact_bound_observation() -> None:
    builder, capture = _fixture()
    builder.validate_backend_row("row", (1, 2583, 25))
    observation = builder("row", 0, capture)
    assert observation.chart.chart_id == "chart"
    assert observation.query.query_coordinates.tolist() == [0.25]
    assert observation.nuisance_key[-2:] == ("PASS", "PASS")
    evidence = builder.evidence(observation.observation_id)
    assert evidence.row_id == "row"
    assert evidence.rooted_geometry_sha256 == "a" * 64
    assert evidence.maximum_residual_error == 0.0
    assert evidence.maximum_attention_error == 0.0
    assert evidence.minimum_residual_cosine == pytest.approx(1.0, abs=1e-9)
    assert evidence.maximum_residual_norm_ratio_deviation == 0.0
    assert evidence.minimum_attention_correlation == pytest.approx(1.0, abs=1e-9)


def test_token_and_atlas_domain_mismatches_fail_before_observation() -> None:
    builder, capture = _fixture()
    with pytest.raises(RelationalGaugeReplayObservationError, match="token IDs"):
        builder.validate_backend_row("row", (1, 3, 25))
    with pytest.raises(RelationalGaugeReplayObservationError, match="unsupported_atlas_domain"):
        builder("row", 1, capture)


def test_residual_or_attention_mismatch_fails_closed() -> None:
    builder, capture = _fixture()
    bad_residuals = dict(capture.residuals)
    bad_residuals[LAYERS[0]] = bad_residuals[LAYERS[0]] + 0.5
    with pytest.raises(RelationalGaugeReplayObservationError, match="root residuals"):
        builder(
            "row",
            0,
            RelationalStepCapture(bad_residuals, dict(capture.attentions)),
        )

    # Attention is telemetry, not a gate (launch-11 demotion): a perturbed
    # attention row must NOT abort the bind, but the evidence must record the
    # degraded correlation for offline v3-gate design.
    bad_attention = dict(capture.attentions)
    changed = bad_attention[LAYERS[0]].clone()
    changed[0, 0, 0, 0] = 0.95
    bad_attention[LAYERS[0]] = changed
    observation = builder(
        "row",
        0,
        RelationalStepCapture(dict(capture.residuals), bad_attention),
    )
    evidence = builder.evidence(observation.observation_id)
    assert evidence.minimum_attention_correlation < 1.0


def _realistic_fixture(
    seed: int = 7,
) -> tuple[CaptureBoundPreStatusGaugeObservationBuilder, torch.Tensor, torch.Tensor]:
    """Realistic-scale fixture: 256-dim residuals, 2 heads x 64 retained tokens."""
    rng = np.random.default_rng(seed)
    hidden, heads, tokens = 256, 2, 64
    chart = GaugeChart(
        chart_id="chart",
        center_node_id="train-a",
        support_ids=("train-a", "train-b"),
        support_distances=((0.0, 1.0), (1.0, 0.0)),
        coordinates=np.asarray([[0.0], [1.0]]),
        eigenvalues=np.asarray([1.0]),
        stress=0.0,
        support_radius=1.0,
    )
    query = GaugeQueryState(
        chart_id="chart",
        query_coordinates=np.asarray([0.25]),
        nearest_node_id="train-a",
        nearest_node_distance=0.25,
        stress=0.0,
        support_status=True,
        support_reason="sealed test query",
    )

    class Atlas:
        @staticmethod
        def get_chart(chart_id: str) -> GaugeChart:
            return chart

    class Lift:
        @staticmethod
        def patch(chart_id: str) -> object:
            return object()

    root = torch.tensor(
        rng.normal(0.0, 4.0, size=(len(LAYERS), hidden)), dtype=torch.float32
    )
    # Peaked attention rows: a softmax over random logits attends a few tokens.
    logits = rng.normal(0.0, 2.0, size=(heads, tokens))
    dense = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    full_attention = torch.tensor(dense, dtype=torch.float32)
    retained_indices = np.arange(tokens)
    binding = RootedStarObservationBinding(
        rooted_star_id="rooted-star",
        reference_id="reference",
        view="intervention_masked_action_free",
        geometry_sha256="a" * 64,
        prefix_token_count=tokens,
        retained_token_indices=retained_indices,
        root_residuals=_bf16(root),
        incoming_attention=np.stack([_bf16(full_attention) for _ in LAYERS]),
    )
    from geoprobe.models.relational_structured_action import STATUS_PREFIX_TOKEN_IDS

    prefix = tuple(range(100, 100 + tokens - len(STATUS_PREFIX_TOKEN_IDS))) + (
        STATUS_PREFIX_TOKEN_IDS
    )
    row = FrozenPreStatusGaugeReplayRow(
        row_id="row",
        held_out_family_fold="outer_1",
        root_id="held-out-root",
        expected_token_ids=prefix,
        nuisance_key=("0", "[]", "no_pressure", "PASS", "PASS"),
        rooted_stars=(binding,),
    )
    bundle = SimpleNamespace(
        horizontal_lift=Lift(),
        held_out_family_fold="outer_1",
        view="intervention_masked_action_free",
        held_out_queries={"held-out-root": query},
        atlas=Atlas(),
    )
    builder = CaptureBoundPreStatusGaugeObservationBuilder(
        bundle=bundle,
        rows=(row,),
        controller_artifact_sha256="b" * 64,
    )
    return builder, root, full_attention


def _capture_from(root: torch.Tensor, full_attention: torch.Tensor) -> RelationalStepCapture:
    heads, tokens = full_attention.shape
    return RelationalStepCapture(
        residuals={
            layer: root[index].reshape(1, 1, -1).clone()
            for index, layer in enumerate(LAYERS)
        },
        attentions={
            layer: full_attention.reshape(1, heads, 1, tokens).clone()
            for layer in LAYERS
        },
    )


def test_bind_accepts_cross_kernel_scale_drift() -> None:
    """Multi-ulp bf16 kernel drift (dense small noise + sparse outliers) must pass."""
    builder, root, full_attention = _realistic_fixture()
    rng = np.random.default_rng(11)
    noisy_root = root + torch.tensor(
        rng.normal(0.0, 0.05, size=tuple(root.shape)), dtype=torch.float32
    )
    # a few outlier channels at the observed live drift scale
    noisy_root[0, 3] += 0.1
    noisy_root[2, 100] -= 0.1
    attention = full_attention.numpy().copy()
    attention += rng.normal(0.0, 0.002, size=attention.shape)
    attention[0, 5] += 0.04  # sparse softmax outlier, launch-8/9 scale
    attention[1, 40] -= min(0.04, float(attention[1, 40]) * 0.5)
    attention = np.clip(attention, 1e-6, None)
    attention /= attention.sum(axis=1, keepdims=True)
    observation = builder(
        "row", 0, _capture_from(noisy_root, torch.tensor(attention, dtype=torch.float32))
    )
    evidence = builder.evidence(observation.observation_id)
    assert evidence.minimum_residual_cosine > 0.999
    assert evidence.minimum_attention_correlation > 0.99
    # amplitude telemetry records the drift without gating on it
    assert evidence.maximum_attention_error > 0.01


def test_bind_rejects_wrong_state_residuals_by_direction() -> None:
    """Channel-shuffled (wrong-state) residuals fail the cosine gate decisively."""
    builder, root, full_attention = _realistic_fixture()
    rng = np.random.default_rng(13)
    shuffled = root.numpy().copy()
    for index in range(shuffled.shape[0]):
        rng.shuffle(shuffled[index])
    with pytest.raises(RelationalGaugeReplayObservationError, match="root residuals") as info:
        builder("row", 0, _capture_from(torch.tensor(shuffled), full_attention))
    # margin evidence: a wrong state lands far below the 0.999 gate
    assert "min cosine=" in str(info.value)


def test_wrong_pattern_attention_recorded_as_low_correlation_telemetry() -> None:
    """Token-rolled attention no longer gates (launch-11 demotion) but its
    collapsed correlation must land in the evidence for offline v3-gate design;
    identity remains carried by the exact prefix + residual gates."""
    builder, root, full_attention = _realistic_fixture()
    rolled = np.roll(full_attention.numpy().copy(), shift=7, axis=1)
    observation = builder("row", 0, _capture_from(root, torch.tensor(rolled)))
    evidence = builder.evidence(observation.observation_id)
    assert evidence.minimum_attention_correlation < 0.5


def test_threshold_validation_rejects_out_of_range() -> None:
    builder, _, _ = _realistic_fixture()
    with pytest.raises(RelationalGaugeReplayObservationError, match="residual_min_cosine"):
        CaptureBoundPreStatusGaugeObservationBuilder(
            bundle=builder.bundle,
            rows=tuple(builder._rows.values()),
            controller_artifact_sha256="b" * 64,
            residual_min_cosine=1.5,
        )


def test_evidence_hash_binds_accepted_live_capture_values() -> None:
    builder, capture = _fixture()
    first = builder("row", 0, capture)
    changed_residuals = dict(capture.residuals)
    changed_residuals[LAYERS[0]] = changed_residuals[LAYERS[0]] + 0.005
    second = builder(
        "row",
        0,
        RelationalStepCapture(changed_residuals, dict(capture.attentions)),
    )
    assert second.observation_id != first.observation_id
    assert (
        builder.evidence(second.observation_id).live_root_residuals_sha256
        != builder.evidence(first.observation_id).live_root_residuals_sha256
    )
