from __future__ import annotations

import numpy as np
import pytest

from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
    HonestwardSupportEdge,
    RelationalPreStatusHonestwardError,
    SharedPreStatusHonestwardField,
)


def _observation(
    pair: str,
    root: str,
    value: float,
    *,
    fold: str = "outer_1",
    contrast: str = "c1",
    truth: str = "PASS",
) -> HonestwardCrossingObservation:
    return HonestwardCrossingObservation(
        pair_id=pair,
        deceptive_root_id=root,
        honest_root_id=f"h-{pair}",
        family=f"family-{fold}",
        family_fold=fold,
        scenario_id=f"scenario-{pair}",
        contrast_id=contrast,
        true_status=truth,
        delta=np.full((2, 3), value, dtype=np.float32),
    )


def _edges(*roots: str) -> tuple[HonestwardSupportEdge, ...]:
    return tuple(
        HonestwardSupportEdge(target_id=root, rank=index, joint_score=float(index))
        for index, root in enumerate(roots, 1)
    )


def test_fit_rejects_heldout_crossings() -> None:
    with pytest.raises(RelationalPreStatusHonestwardError, match="held-out"):
        SharedPreStatusHonestwardField.fit(
            (_observation("p", "d", 1.0, fold="outer_5"),),
            held_out_family_fold="outer_5",
            training_edges={"d": ()},
        )


def test_multiple_targets_give_one_deceptive_root_vote() -> None:
    observations = (
        _observation("p1", "d1", 1.0),
        _observation("p2", "d1", 3.0),
        _observation("p3", "d2", 10.0),
    )
    graph = {"d1": _edges("d2"), "d2": _edges("d1")}
    field = SharedPreStatusHonestwardField.fit(
        observations, held_out_family_fold="outer_5", training_edges=graph
    )
    prediction = field.predict("query", _edges("d1", "d2"))
    assert prediction.defined
    assert prediction.support_count == 2
    assert prediction.local == pytest.approx(np.full((2, 3), 6.0))
    assert set(prediction.support_pair_ids) == {"p1", "p2", "p3"}


def test_leave_contrast_out_and_opposite_truth_filter_support() -> None:
    observations = (
        _observation("p1", "d1", 1.0, contrast="c1", truth="PASS"),
        _observation("p2", "d2", 5.0, contrast="c2", truth="FAIL"),
    )
    graph = {"d1": _edges("d2"), "d2": _edges("d1")}
    field = SharedPreStatusHonestwardField.fit(
        observations, held_out_family_fold="outer_5", training_edges=graph
    )
    cross = field.predict("query", _edges("d1", "d2"), exclude_contrast="c1")
    opposite = field.predict(
        "query", _edges("d1", "d2"), opposite_of_true_status="PASS"
    )
    assert cross.local == pytest.approx(np.full((2, 3), 5.0))
    assert opposite.local == pytest.approx(np.full((2, 3), 5.0))


def test_train_only_scalar_dose_is_nonnegative() -> None:
    observations = (
        _observation("p1", "d1", 1.0),
        _observation("p2", "d2", 2.0),
        _observation("p3", "d3", 3.0),
    )
    graph = {
        "d1": _edges("d2", "d3"),
        "d2": _edges("d1", "d3"),
        "d3": _edges("d1", "d2"),
    }
    field = SharedPreStatusHonestwardField.fit(
        observations, held_out_family_fold="outer_5", training_edges=graph
    )
    assert np.all(field.scalar_dose_by_layer >= 0.0)
    prediction = field.predict("query", _edges("d1", "d2"))
    assert prediction.dose_calibrated_local == pytest.approx(
        prediction.local * field.scalar_dose_by_layer[:, None]
    )
