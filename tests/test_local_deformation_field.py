from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geoprobe.control import geometry_map as gm
from geoprobe.control.local_deformation_field import (
    FullFieldLocalDeformationSelector,
    LocalDeformationConfig,
    SteeringSpecLookup,
)
from tests.test_action_fiber_models import make_row


def write_spec_bank(tmp_path: Path, rows: list[dict], vectors: dict[tuple[str, int, str, str], np.ndarray]) -> Path:
    bank = tmp_path / "bank"
    specs = bank / "specs"
    specs.mkdir(parents=True)
    meta = []
    vecs = []
    for row in rows:
        if gm.is_baseline(row) or row.get("layer") is None:
            continue
        raw_method = {
            "global_mean_gated": "global_mean",
            "global_probe_gated": "global_probe",
            "random_gated": "random_global",
        }.get(str(row.get("method")), str(row.get("method")))
        key = (gm.state_id(row), int(row["layer"]), raw_method, str(row["target_status"]))
        if key not in vectors:
            continue
        meta.append({
            "conversation_id": key[0],
            "family": row["family"],
            "layer": key[1],
            "method": key[2],
            "target_status": key[3],
            "direction_info": {"direction_convention": "toy"},
            "projection": None,
            "context": {},
        })
        vecs.append(vectors[key].astype(np.float32))
    np.savez_compressed(specs / "layer16_familytoy.npz", vectors=np.vstack(vecs))
    with (specs / "layer16_familytoy.jsonl").open("w") as handle:
        for item in meta:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    (bank / "manifest.json").write_text(json.dumps({
        "kind": "steering_spec_bank",
        "spec_schema_version": 1,
        "identity": {"layers": [16], "methods": ["global_mean", "bidir_linear"]},
        "shards": [{"family": "toy", "layer": 16, "stem": "layer16_familytoy", "n_specs": len(vecs)}],
        "n_specs": len(vecs),
    }))
    return bank


def toy_rows_states_vectors() -> tuple[list[dict], dict[gm.StateLayerKey, np.ndarray], dict[tuple[str, int, str, str], np.ndarray]]:
    rows: list[dict] = []
    states: dict[gm.StateLayerKey, np.ndarray] = {}
    vectors: dict[tuple[str, int, str, str], np.ndarray] = {}
    for family in ["train_a", "train_b", "held"]:
        for region, x0 in [("left", -2.0), ("right", 2.0)]:
            cid = f"{family}_{region}"
            states[(cid, 16)] = np.asarray([x0, 0.1 if family == "train_b" else 0.0, 0.0, 0.0], dtype=np.float32)
            rows.append(make_row(cid, "toy", region, method="baseline", target_status=None, layer=None))
            good_method = "global_mean_gated" if region == "left" else "bidir_linear"
            bad_method = "bidir_linear" if region == "left" else "global_mean_gated"
            rows.append(
                make_row(
                    cid,
                    "toy",
                    region,
                    method=good_method,
                    target_status="PASS",
                    layer=16,
                    reward=1.0,
                    fixes=True,
                )
            )
            rows.append(
                make_row(
                    cid,
                    "toy",
                    region,
                    method=bad_method,
                    target_status="PASS",
                    layer=16,
                    reward=0.0,
                    fixes=False,
                )
            )
            vectors[(cid, 16, "global_mean", "PASS")] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            vectors[(cid, 16, "bidir_linear", "PASS")] = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return rows, states, vectors


def cfg(**overrides: object) -> LocalDeformationConfig:
    values = {
        "hidden_dim": 12,
        "epochs": 10,
        "lr": 3e-3,
        "weight_decay": 0.0,
        "harm_penalty": 1.0,
        "top_k": 2,
        "neighbor_pool_multiplier": 2,
        "device": "cpu",
        "seed": 11,
    }
    values.update(overrides)
    return LocalDeformationConfig(**values)


def test_steering_spec_lookup_maps_adapted_global_alias(tmp_path: Path) -> None:
    rows, _states, vectors = toy_rows_states_vectors()
    bank = write_spec_bank(tmp_path, rows, vectors)
    lookup = SteeringSpecLookup(bank)
    row = next(item for item in rows if item.get("method") == "global_mean_gated")
    key = lookup.key_for_row(row)
    assert key is not None
    assert key[2] == "global_mean"
    np.testing.assert_allclose(lookup.vector_for(row), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_full_field_local_deformation_scores_heldout_candidates(tmp_path: Path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    test = [row for row in rows if gm.state_id(row) == "held_left"]
    selector = FullFieldLocalDeformationSelector(lookup, cfg()).fit(train, states)
    scores = selector.score(test, states)
    assert scores.shape == (len(test),)
    assert scores[0] < -1e8
    assert np.isfinite(scores[1:]).all()
    choice = gm.choose_by_scores(test, scores, threshold=-1e9)
    explanation = selector.explain_choice(choice, states)
    assert explanation["model"] == "full_field_local_deformation"
    assert explanation["geometry_dim"] == 0


def test_full_field_context_scores_ignore_mutated_query_response_fields(tmp_path: Path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    group = [row for row in rows if gm.state_id(row) == "held_right"]
    selector = FullFieldLocalDeformationSelector(lookup, cfg()).fit(train, states)
    mutated = []
    for row in group:
        copy = dict(row)
        copy.update({
            "final_margin": 999.0,
            "decision_margin": 999.0,
            "delta_margin": 999.0,
            "reward": -123.0,
            "strict_reward": -123.0,
            "correct_after": not bool(row.get("correct_after")),
            "fixes_error": not bool(row.get("fixes_error")),
            "fixes_strict": not bool(row.get("fixes_strict")),
            "harms_honest": not bool(row.get("harms_honest")),
            "harms_honest_strict": not bool(row.get("harms_honest_strict")),
        })
        mutated.append(copy)
    before = selector.score(group, states)
    after = selector.score(mutated, states)
    np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-7)


def test_pca64_ablation_uses_projected_geometry(tmp_path: Path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    selector = FullFieldLocalDeformationSelector(lookup, cfg(geometry_dim=2, epochs=1)).fit(train, states)
    assert selector.projector_.projector_ is not None
    assert selector.state_x_.shape[1] == 2


def test_exact_action_summaries_are_scalar_only_to_bound_full_field_memory(tmp_path: Path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    selector = FullFieldLocalDeformationSelector(lookup, cfg(epochs=1)).fit(train, states)
    assert selector.summary_any_
    assert selector.summary_method_
    assert selector.summary_action_
    assert all(summary.reward_vec_sum.size == states[("train_a_left", 16)].size for summary in selector.summary_any_.values())
    assert all(summary.reward_vec_sum.size == states[("train_a_left", 16)].size for summary in selector.summary_method_.values())
    assert all(summary.reward_vec_sum.size == 0 for summary in selector.summary_action_.values())
