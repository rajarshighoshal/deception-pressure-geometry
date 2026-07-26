from __future__ import annotations

import re

import numpy as np
import pytest

from geoprobe.control.relational_pre_status_causal import (
    CAUSAL_ARM_ORDER,
    CAUSAL_LAYERS,
    PRIMARY_ACTUATION_LAYER,
    RelationalPreStatusCausalError,
    build_relational_pre_status_causal_arm_bundle,
    select_primary_actuation_vectors,
)


def _event_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.array(
        [
            [1.2, 0.9, 1.4, -0.2, 0.3],
            [0.5, -1.0, 0.2, 2.1, -0.8],
            [1.0, 0.8, 0.5, -0.4, 1.7],
            [2.3, -0.6, 1.9, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    t = np.array(
        [
            [1.0, 0.4, 0.8, -0.5, 0.2],
            [0.1, -0.6, -0.5, 1.0, -0.3],
            [0.8, 0.2, 0.1, -1.1, 0.7],
            [1.8, -1.1, 1.5, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    g = np.array(
        [
            [0.2, 0.1, 0.2, -0.2, 0.2],
            [0.3, 0.7, 0.1, 0.9, 0.2],
            [0.4, 0.0, 0.6, 0.6, 0.9],
            [0.1, 0.4, 0.2, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    return h, t, g


def test_relational_pre_status_causal_arms_match_algebra() -> None:
    h, t, g = _event_vectors()
    bundle = build_relational_pre_status_causal_arm_bundle("event-1", "fold-a", h, t, g)
    s = h - t

    np.testing.assert_allclose(bundle.s, s)
    np.testing.assert_allclose(bundle.arm_vectors["noop"], np.zeros_like(h))
    np.testing.assert_allclose(bundle.arm_vectors["fixed_global_h"], g)
    np.testing.assert_allclose(bundle.arm_vectors["generic_t"], t)
    np.testing.assert_allclose(bundle.arm_vectors["specific_s"], s)
    np.testing.assert_allclose(bundle.arm_vectors["full_h"], h)
    np.testing.assert_allclose(bundle.arm_vectors["generic_minus_s"], t - s)
    np.testing.assert_allclose(bundle.arm_vectors["generic_plus_random_s"], t + bundle.random_s)
    assert tuple(bundle.arm_vectors) == CAUSAL_ARM_ORDER


def test_relational_pre_status_causal_random_arm_is_deterministic_and_domain_separated() -> None:
    h, t, g = _event_vectors()
    baseline = build_relational_pre_status_causal_arm_bundle("event-2", "fold-x", h, t, g)
    same = build_relational_pre_status_causal_arm_bundle("event-2", "fold-x", h, t, g)
    different_fold = build_relational_pre_status_causal_arm_bundle(
        "event-2", "fold-y", h, t, g
    )

    np.testing.assert_array_equal(
        baseline.arm_vectors["generic_plus_random_s"], same.arm_vectors["generic_plus_random_s"]
    )
    assert not np.array_equal(
        baseline.arm_vectors["generic_plus_random_s"],
        different_fold.arm_vectors["generic_plus_random_s"],
    )
    assert np.array_equal(baseline.arm_vectors["generic_t"], same.arm_vectors["generic_t"])


def test_relational_pre_status_causal_norm_parity_zero_layer() -> None:
    h, t, g = _event_vectors()
    h = h.copy()
    t = t.copy()
    h[0] = t[0]
    bundle = build_relational_pre_status_causal_arm_bundle("event-3", "fold-z", h, t, g)

    np.testing.assert_allclose(bundle.random_s[0], 0.0)
    np.testing.assert_allclose(
        bundle.arm_vectors["generic_plus_random_s"][0], bundle.arm_vectors["generic_t"][0]
    )
    assert bundle.per_layer_norms["specific_s"][0] == pytest.approx(0.0, abs=0.0)
    assert np.linalg.norm(bundle.random_s[0]) == pytest.approx(0.0, abs=0.0)


def test_relational_pre_status_causal_validation_rejects_invalid_inputs() -> None:
    h, t, g = _event_vectors()
    with pytest.raises(RelationalPreStatusCausalError, match="state_id"):
        build_relational_pre_status_causal_arm_bundle("", "fold", h, t, g)
    with pytest.raises(RelationalPreStatusCausalError, match="family_fold"):
        build_relational_pre_status_causal_arm_bundle("event", "", h, t, g)

    bad_shape = h[:-1]
    with pytest.raises(RelationalPreStatusCausalError, match="incompatible"):
        build_relational_pre_status_causal_arm_bundle("event", "fold", bad_shape, t, g)

    empty_hidden = np.empty((4, 0), dtype=np.float32)
    with pytest.raises(RelationalPreStatusCausalError, match="hidden dimension"):
        build_relational_pre_status_causal_arm_bundle(
            "event", "fold", empty_hidden, empty_hidden, empty_hidden
        )

    non_finite = h.copy()
    non_finite[0, 0] = np.nan
    with pytest.raises(RelationalPreStatusCausalError, match="finite"):
        build_relational_pre_status_causal_arm_bundle("event", "fold", non_finite, t, g)


def test_relational_pre_status_causal_hashes_and_read_only_outputs() -> None:
    h, t, g = _event_vectors()
    bundle = build_relational_pre_status_causal_arm_bundle("event-4", "fold-h", h, t, g)

    assert not bundle.h.flags.writeable
    assert not bundle.t.flags.writeable
    assert not bundle.g.flags.writeable
    assert not bundle.s.flags.writeable
    assert not bundle.random_s.flags.writeable
    for vector in bundle.arm_vectors.values():
        assert not vector.flags.writeable
    with pytest.raises(TypeError):
        bundle.arm_vectors["generic_t"] = bundle.h
    with pytest.raises(TypeError):
        bundle.per_layer_norms["generic_t"] = (0.0, 0.0, 0.0, 0.0)

    for digest in (*bundle.source_hashes.values(), *bundle.arm_hashes.values()):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    assert bundle.metadata["bundle_hash"] == bundle.bundle_hash
    assert set(bundle.metadata["random_seed_payload"]) == set(CAUSAL_LAYERS)
    assert bundle.metadata["kind"] == "relational_pre_status_causal_v1"
    assert bundle.metadata["state_id"] == "event-4"
    assert bundle.metadata["family_fold"] == "fold-h"
    assert bundle.metadata["layers"] == CAUSAL_LAYERS
    assert bundle.metadata["arm_order"] == CAUSAL_ARM_ORDER
    assert bundle.metadata["primary_actuation_layer"] == PRIMARY_ACTUATION_LAYER
    assert bundle.metadata["random_bit_generator"] == "PCG64"


def test_primary_actuation_selects_only_layer_12_cross_section() -> None:
    h, t, g = _event_vectors()
    bundle = build_relational_pre_status_causal_arm_bundle("event-5", "fold-a", h, t, g)

    selected = select_primary_actuation_vectors(bundle.arm_vectors)

    assert PRIMARY_ACTUATION_LAYER == 12
    assert tuple(selected) == CAUSAL_ARM_ORDER
    for name, vector in selected.items():
        np.testing.assert_array_equal(vector, bundle.arm_vectors[name][0])
        assert vector.ndim == 1
        assert not vector.flags.writeable


def test_primary_actuation_rejects_unfrozen_layer() -> None:
    h, t, g = _event_vectors()
    bundle = build_relational_pre_status_causal_arm_bundle("event-6", "fold-a", h, t, g)

    with pytest.raises(RelationalPreStatusCausalError, match="actuation layer"):
        select_primary_actuation_vectors(bundle.arm_vectors, layer=13)
