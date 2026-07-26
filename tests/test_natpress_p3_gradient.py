"""Tests for the P3 gradient-signature pure-math helpers (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.report_natpress_p3_gradient import (
    bootstrap_diff_ci,
    contrast_verdict,
    early_drift_fraction,
    kfold_indices,
    roc_auc,
    spearman_rho,
)


def test_edf_gradual_vs_late():
    # gradual climb: most displacement happens before the corner -> EDF near 1
    gradual = [0, 1, 2, 3, 4, 5, 6]
    assert early_drift_fraction(gradual) == pytest.approx(5 / 6)
    # late jump: flat then a corner spike -> EDF near 0
    late = [0, 0, 0, 0, 0, 0, 6]
    assert early_drift_fraction(late) == pytest.approx(0.0)


def test_edf_excludes_nonpositive_displacement():
    assert early_drift_fraction([5, 4, 3, 2, 1, 0, 0]) is None  # total displacement 0
    assert early_drift_fraction([6, 5, 4, 3, 2, 1, 0]) is None  # negative displacement


def test_edf_wrong_length_raises():
    with pytest.raises(ValueError):
        early_drift_fraction([0, 1, 2])


def test_spearman_monotone_and_flat():
    assert spearman_rho([1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert spearman_rho([6, 5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([3, 3, 3, 3]) == pytest.approx(0.0)


def test_spearman_with_ties():
    # non-decreasing with a tie -> strong positive but not exactly 1
    rho = spearman_rho([1, 2, 2, 3, 4, 5])
    assert 0.9 < rho < 1.0


def test_roc_auc_separable_and_random():
    assert roc_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([1.0, 0.9, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    # perfectly interleaved -> 0.5
    assert roc_auc([0.0, 1.0, 2.0, 3.0], [0, 1, 0, 1]) == pytest.approx(0.75)


def test_roc_auc_single_class_nan():
    assert np.isnan(roc_auc([0.1, 0.2, 0.3], [1, 1, 1]))


def test_kfold_partitions_cover_all():
    folds = kfold_indices(23, 5, seed=20260721)
    assert len(folds) == 5
    covered = np.concatenate(folds)
    assert sorted(covered.tolist()) == list(range(23))


def test_kfold_deterministic():
    a = kfold_indices(20, 5, seed=20260721)
    b = kfold_indices(20, 5, seed=20260721)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_bootstrap_deterministic_and_signed():
    a = [1.0, 1.1, 0.9, 1.2, 0.8]
    b = [0.1, 0.0, 0.2, -0.1, 0.05]
    ci1 = bootstrap_diff_ci(a, b, n_boot=2000, seed=20260721)
    ci2 = bootstrap_diff_ci(a, b, n_boot=2000, seed=20260721)
    assert ci1 == ci2
    assert ci1["point"] > 0 and ci1["lo"] > 0  # clearly separated groups


def test_contrast_verdict_mapping():
    assert contrast_verdict(0.3, 0.1, 0.5) == "found"
    assert contrast_verdict(-0.3, -0.5, -0.1) == "refuted-under-adequate-instrument"
    assert contrast_verdict(0.1, -0.05, 0.3) == "not-found-under-this-instrument"
    assert contrast_verdict(0.3, 0.0, 0.5) == "not-found-under-this-instrument"  # lo not > 0
