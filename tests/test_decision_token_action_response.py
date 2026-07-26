"""Tests for the leak-safe point-cloud context features (family / scenario holdout).

Covers the pc_* fix: ``pointcloud_context_features`` must exclude neighbors by
family (default split) or by scenario (within-scenario split), and the two
guards must AND together. Default-family behavior must match the legacy call.
"""
import numpy as np

from experiments.decision_token_action_response import (
    PointcloudContextIndex,
    pointcloud_context_features,
)


def _row(family, scenario, status_class, x):
    return {
        "family": family,
        "scenario_id": scenario,
        "status_class": status_class,
        "x": np.array(x, dtype=float),
    }


# 6 rows: family A (scenario s1) x2, family B (s2) x2, family C (s3) x2
REGION = [
    _row("A", "s1", "false_PASS", [0.0, 0.0]),
    _row("A", "s1", "false_PASS", [0.1, 0.0]),
    _row("B", "s2", "honest_PASS", [5.0, 5.0]),
    _row("B", "s2", "honest_FAIL", [5.1, 5.0]),
    _row("C", "s3", "false_FAIL", [0.0, 0.1]),
    _row("C", "s3", "honest_PASS", [10.0, 10.0]),
]
QUERY = np.array([0.0, 0.0], dtype=float)


def test_no_holdout_keeps_all():
    out = pointcloud_context_features(query_x=QUERY, region_rows=REGION, k=3)
    assert out["pc_n_train"] == 6


def test_family_holdout_excludes_whole_family():
    out = pointcloud_context_features(query_x=QUERY, region_rows=REGION, heldout_family="A", k=3)
    assert out["pc_n_train"] == 4


def test_scenario_holdout_keeps_same_family_other_scenario():
    # add a same-family (A) / different-scenario (s9) row: family-holdout drops it,
    # scenario-holdout keeps it. This is the whole point of the within-scenario split.
    region = REGION + [_row("A", "s9", "honest_FAIL", [0.2, 0.0])]
    fam = pointcloud_context_features(query_x=QUERY, region_rows=region, heldout_family="A", k=5)
    scn = pointcloud_context_features(query_x=QUERY, region_rows=region, heldout_scenario_ids={"s1"}, k=5)
    assert fam["pc_n_train"] == 4   # excludes all of family A, including A/s9
    assert scn["pc_n_train"] == 5   # excludes only scenario s1, keeps A/s9


def test_both_holdouts_and_together():
    out = pointcloud_context_features(
        query_x=QUERY, region_rows=REGION, heldout_family="A", heldout_scenario_ids={"s1"}, k=3
    )
    # s1 lives inside family A, so the family exclusion already subsumes it
    assert out["pc_n_train"] == 4


def test_default_matches_legacy_family_only_call():
    # the augment path and the original call site pass heldout_family only
    out = pointcloud_context_features(query_x=QUERY, region_rows=REGION, heldout_family="B", k=3)
    assert out["pc_n_train"] == 4


def test_schema_keys_present():
    out = pointcloud_context_features(query_x=QUERY, region_rows=REGION, heldout_family="A", k=3)
    for key in (
        "pc_n_train", "pc_knn_k", "pc_knn_frac_false_PASS", "pc_centroid_dist_false_PASS",
        "pc_knn_entropy", "pc_false_frac", "pc_honest_frac",
    ):
        assert key in out


# --- batched context query equivalence (the kNN-loop optimization) ---

def _assert_feature_dicts_equal(a, b, tol=1e-9):
    assert set(a) == set(b)
    for key in a:
        va, vb = a[key], b[key]
        if va is None or vb is None:
            assert va is None and vb is None, f"{key}: {va!r} vs {vb!r}"
        else:
            assert abs(float(va) - float(vb)) <= tol * max(1.0, abs(float(va))), (
                f"{key}: {va} vs {vb}"
            )


def _synthetic_index(seed, dim, n_train, k, heldout):
    rng = np.random.default_rng(seed)
    classes = ["false_PASS", "false_FAIL", "honest_PASS", "honest_FAIL"]
    fams = ["fam0", "fam1", "fam2", "fam3"]
    region = []
    for i in range(n_train):
        x = rng.standard_normal(dim).astype(np.float32)
        region.append({
            "x": x,
            "status_class": classes[i % len(classes)],
            "family": fams[i % len(fams)],
            "scenario_id": f"s{i}",
        })
    # exercise NaN/inf sanitisation in both train and query
    region[3]["x"][0] = np.nan
    region[7]["x"][2] = np.inf
    return PointcloudContextIndex(region, heldout_family=heldout, k=k), rng


def test_query_batch_matches_per_row_query():
    idx, rng = _synthetic_index(seed=11, dim=64, n_train=200, k=15, heldout="fam1")
    queries = [rng.standard_normal(64).astype(np.float32) for _ in range(40)]
    queries[2][1] = np.nan  # query-side NaN must be handled identically
    batched = idx.query_batch(np.vstack(queries))
    assert len(batched) == len(queries)
    for q, b in zip(queries, batched):
        _assert_feature_dicts_equal(idx.query(q), b)


def test_query_batch_matches_standalone_function():
    rng = np.random.default_rng(5)
    classes = ["false_PASS", "false_FAIL", "honest_PASS", "honest_FAIL"]
    fams = ["fam0", "fam1", "fam2"]
    region = [
        {"x": rng.standard_normal(48).astype(np.float32),
         "status_class": classes[i % len(classes)],
         "family": fams[i % len(fams)],
         "scenario_id": f"s{i}"}
        for i in range(120)
    ]
    idx = PointcloudContextIndex(region, heldout_family="fam0", k=10)
    queries = [rng.standard_normal(48).astype(np.float32) for _ in range(25)]
    batched = idx.query_batch(np.vstack(queries))
    for q, b in zip(queries, batched):
        legacy = pointcloud_context_features(
            query_x=q, region_rows=region, heldout_family="fam0", k=10
        )
        _assert_feature_dicts_equal(legacy, b)


def test_query_batch_empty_index():
    # whole region held out -> empty train set; batch and per-row must agree
    region = [{"x": np.zeros(8, dtype=np.float32), "status_class": "false_PASS",
               "family": "only", "scenario_id": "s0"}]
    idx = PointcloudContextIndex(region, heldout_family="only", k=5)
    assert idx.xs is None
    queries = np.vstack([np.ones(8, dtype=np.float32), np.full(8, 2.0, dtype=np.float32)])
    batched = idx.query_batch(queries)
    assert len(batched) == 2
    for row, q in zip(batched, queries):
        _assert_feature_dicts_equal(idx.query(q), row)
        assert row["pc_knn_mean_dist"] is None
