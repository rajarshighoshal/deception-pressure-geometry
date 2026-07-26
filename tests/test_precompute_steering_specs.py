"""Tests for the steering-spec precompute (#2 of the CPU/GPU decouple).

The critical correctness property: a precomputed spec vector must equal what the
inline runner's action_vector would have produced. compute_vector here is tested
against a faithful reference of action_vector, fed identical mocked fits + a
batched tangent basis (which #1 already proved equals the per-row projection).
"""
import numpy as np

from experiments.control_graded_dp_decision_token import BIDIR_METHODS
from experiments.control_graded_dp_frontier import (
    fit_tangent_cloud,
    local_tangent_bases,
    off_tangent_direction,
    project_to_local_tangent,
    unit,
)
from experiments.decision_token_action_response import signed_global_direction
from experiments.precompute_steering_specs import compute_vector, shard_stem, spec_identity, targets_for_row
from experiments.resident_decision_token_bank_worker import usable_unit_vector

ALL_METHODS = [
    "bidir_linear", "bidir_tangent", "bidir_off_tangent",
    "global_mean", "global_probe", "random_global",
]


def _ref_action_vector(method, target, *, status_dir, global_dir, tangent_info, query_x, tn, td):
    """Faithful copy of resident_decision_token_pool_runner.action_vector's math."""
    if method in BIDIR_METHODS:
        di = status_dir.get(target)
        if di is None:
            return None
        raw = np.asarray(di["_direction_np"], dtype=np.float64)
        if method == "bidir_linear":
            return unit(raw)
        if tangent_info is None:
            return None
        tvec, _proj = project_to_local_tangent(raw, tangent_info, query_x, tangent_neighbors=tn, tangent_dim=td)
        if tvec is None:
            return None
        if method == "bidir_tangent":
            return unit(tvec.detach().float().cpu().numpy())
        off = off_tangent_direction(raw, tvec)
        if off is None:
            return None
        return unit(off.detach().float().cpu().numpy())
    di = global_dir.get(method)
    if di is None:
        return None
    return signed_global_direction(di, target)


def _cloud(seed, dim=48, n=300, heldout="famH"):
    rng = np.random.default_rng(seed)
    fams = ["fam0", "fam1", "fam2", heldout]
    pts = [{"x": rng.standard_normal(dim).astype(np.float32),
            "family": fams[i % len(fams)], "scenario_id": f"s{i}"} for i in range(n)]
    return fit_tangent_cloud(pts, heldout_family=heldout), rng


def test_compute_vector_matches_action_vector():
    info, rng = _cloud(seed=1)
    dim = 48
    status_dir = {
        "PASS": {"_direction_np": rng.standard_normal(dim)},
        "FAIL": {"_direction_np": rng.standard_normal(dim)},
    }
    global_dir = {m: {"_direction_np": rng.standard_normal(dim)} for m in ("global_mean", "global_probe", "random_global")}
    for _ in range(6):
        q = rng.standard_normal(dim).astype(np.float32)
        (basis, base_info), = local_tangent_bases(info, q[None, :], tangent_neighbors=16, tangent_dim=4)
        for method in ALL_METHODS:
            for target in ("PASS", "FAIL"):
                ref = _ref_action_vector(method, target, status_dir=status_dir, global_dir=global_dir,
                                         tangent_info=info, query_x=q, tn=16, td=4)
                vec, _di, _proj = compute_vector(method, target, status_dir=status_dir, global_dir=global_dir,
                                                 tangent_info=info, basis=basis, base_info=base_info)
                ref_clean = usable_unit_vector(ref)
                new_clean = usable_unit_vector(vec)
                if ref_clean is None:
                    assert new_clean is None, (method, target)
                else:
                    assert new_clean is not None, (method, target)
                    assert np.allclose(ref_clean, new_clean, atol=1e-6, rtol=0), (method, target)


def test_compute_vector_missing_fits():
    info, rng = _cloud(seed=2)
    q = rng.standard_normal(48).astype(np.float32)
    (basis, base_info), = local_tangent_bases(info, q[None, :], tangent_neighbors=16, tangent_dim=4)
    # no status direction -> bidir methods yield None with the right reason
    vec, di, proj = compute_vector("bidir_linear", "PASS", status_dir={}, global_dir={},
                                   tangent_info=info, basis=basis, base_info=base_info)
    assert vec is None and proj == {"reason": "missing_status_direction"}
    # no global direction
    vec, di, proj = compute_vector("global_mean", "PASS", status_dir={}, global_dir={},
                                   tangent_info=info, basis=basis, base_info=base_info)
    assert vec is None and proj == {"reason": "missing_global_direction"}
    # missing tangent cloud
    vec, di, proj = compute_vector("bidir_tangent", "PASS",
                                   status_dir={"PASS": {"_direction_np": rng.standard_normal(48)}},
                                   global_dir={}, tangent_info=None, basis=None, base_info={})
    assert vec is None and proj == {"reason": "missing_tangent_cloud"}


def test_targets_for_row_z2():
    assert targets_for_row("false_FAIL", ["PASS", "FAIL"], True) == ["PASS"]
    assert targets_for_row("false_PASS", ["PASS", "FAIL"], True) == ["FAIL"]
    assert targets_for_row("honest_PASS", ["PASS", "FAIL"], True) == ["PASS", "FAIL"]
    assert targets_for_row("false_FAIL", ["PASS", "FAIL"], False) == ["PASS", "FAIL"]


def test_shard_stem_is_collision_proof():
    # families that sanitize to the same name must NOT share a shard stem (else one
    # family's specs overwrite the other's). The raw-family hash keeps them distinct.
    a = shard_stem(20, "ai content safety")
    b = shard_stem(20, "ai_content_safety")
    c = shard_stem(20, "ai/content/safety")
    assert a != b and a != c and b != c
    # same family is deterministic and stable across calls
    assert shard_stem(20, "ai content safety") == a
    # different layer differs
    assert shard_stem(12, "ai content safety") != a


def test_spec_identity_has_vector_affecting_fields():
    import argparse
    args = argparse.Namespace(
        layers=[8, 12], methods=["bidir_tangent"], candidate_targets=["PASS", "FAIL"],
        direction_turn=2, direction_phase="pre_response", query_turn=3, query_phase="pre_response",
        eval_levels=["p3", "p4", "p5"], direction_levels=["p3", "p4", "p5", "p6"],
        tangent_levels=["p0", "p1"], tangent_turns=["0", "1"], tangent_phases=["pre_response"],
        min_mixed_scenarios=2, min_direction_levels=2, tangent_neighbors=16, tangent_dim=4,
        context_neighbors=32, equivariant_directions=True, unidirectional_targets=True,
        seed=7, limit=None, limit_per_status_class=40, limit_strategy="family_round_robin",
        scenario_include=None, scenario_exclude=None, num_shards=1, shard_index=0,
    )
    ident = spec_identity(args, transcript_hashes={"t": "h"}, activation_hash="a", config_hash="c")
    for key in ("activations_sha256", "layers", "methods", "tangent_levels", "tangent_neighbors",
                "tangent_dim", "unidirectional_targets", "limit_per_status_class", "lofo_rule",
                "direction_levels", "seed"):
        assert key in ident
    assert ident["lofo_rule"] == "leave_one_family_out"
    assert ident["activations_sha256"] == "a"
