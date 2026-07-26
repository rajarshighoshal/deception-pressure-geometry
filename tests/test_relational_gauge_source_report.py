from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from geoprobe.eval.relational_gauge_source_report import (
    RelationalGaugeSourceReportError,
    build_relational_gauge_source_report,
    load_relational_gauge_source_run,
    render_relational_gauge_source_markdown,
)
from geoprobe.io import file_sha256
from geoprobe.models.relational_gauge_replay import SOURCE_GAUGE_ARM_ORDER
from geoprobe.runtime import relational_gauge_source_runner as runner


def _sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _canonical_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _proposal(arm: str, *, active: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "active" if active else (
            "no_intervention" if arm == "no_intervention" else "boundary_exit"
        ),
        "stop": not active,
        "field": {
            "defined": True,
            "reason": "defined",
            "kind": "pressure_residual_deception",
            "depth": 0.5,
            "gradient_sha256": _sha("gradient"),
            "outcome_probabilities": [0.2] * 5,
            "nuisance_probabilities": [0.2] * 5,
            "support_node_ids": ["a", "b"],
            "support_observation_count": 4,
            "effective_observation_count": 4.0,
            "bandwidth": 1.0,
        },
        "intrinsic_direction_sha256": _sha("direction" + arm),
        "intrinsic_step_sha256": _sha("step" + arm),
        "intrinsic_step_norm": 0.1 if active else 0.0,
        "distance_to_boundary": 1.0 if active else 0.0,
        "remaining_path_budget": 1.0 if active else 0.0,
        "fiber_step_sha256": _sha("fiber" + arm),
        "per_layer_fiber_sha256": {
            str(layer): _sha(f"fiber-{arm}-{layer}")
            for layer in (12, 16, 19, 20)
        },
        "fiber_shape": [4, 3],
        "fiber_scale": 1.0 if active else 0.0,
        "layer_fiber_norms": [1.0] * 4 if active else [0.0] * 4,
        "layer_fiber_norm_caps": [2.0] * 4,
        "next_state": {
            "step_count": 1 if active else 0,
            "cumulative_intrinsic_length": 0.1 if active else 0.0,
            "stopped": not active,
            "stop_reason": "active" if active else (
                "no_intervention" if arm == "no_intervention" else "boundary_exit"
            ),
        },
    }
    payload["proposal_sha256"] = _canonical_sha(payload)
    return payload


def _row(
    root: str,
    event: str,
    arm: str,
    *,
    active_root: bool,
    mapped_action: str,
    outcome: str,
) -> dict[str, object]:
    active = active_root and arm != "no_intervention"
    token = {"PASS": 51935, "FAIL": 34207, "SKIP": 82504}[mapped_action]
    probabilities = {"PASS": 0.8, "FAIL": 0.1, "SKIP": 0.1}
    logits = {"PASS": 2.0, "FAIL": 0.0, "SKIP": 0.0}
    observation = {
        "observation_id": "observation-" + root,
        "randomization_key": "sealed-" + root,
        "evidence_sha256": _sha("evidence" + root),
        "rooted_geometry_sha256": _sha("geometry" + root),
        "live_root_residuals_sha256": _sha("roots" + root),
        "live_incoming_attention_sha256": _sha("attention" + root),
        "maximum_residual_error": 0.0,
        "maximum_attention_error": 0.0,
        "chart_id": "chart",
        "query_coordinates_sha256": _sha("query" + root),
        "nearest_node_id": "node",
        "nearest_node_distance": 0.1,
        "query_stress": 0.01,
        "support_status": True,
        "support_reason": "test",
        "nuisance_key": ["pressure"],
    }
    row = {
        "schema_version": 1,
        "kind": runner.RUNNER_KIND,
        "root_id": root,
        "state_id": root,
        "event_id": event,
        "family": "family",
        "family_fold": "outer_1",
        "turn_index": 1,
        "scenario_id": "scenario-" + root,
        "true_status": "PASS",
        "desired_status": "FAIL",
        "knowledge_status": "PASS",
        "knowledge_correct": True,
        "arm": arm,
        "raw_token_id": token,
        "raw_decoded_exact": " " + mapped_action,
        "mapped_action": mapped_action,
        "rng_seed": 7,
        "behavioral_outcome_class": outcome,
        "prefix": {"token_ids_sha256": _sha("prefix" + root), "token_count": 3},
        "status_probabilities": probabilities,
        "status_logits": logits,
        "recognized_action_probability_mass": 1.0,
        "top_token_id": 51935,
        "top_token_probability": 0.8,
        "sampling": runner._sampling_contract(),
        "controller_artifact_sha256": _sha("controller"),
        "observation": observation,
        "proposal": _proposal(arm, active=active),
        "actuation_layers": [12, 16, 19, 20] if active else [],
        "model": {"name": "test"},
        "runtime": {"name": "test"},
    }
    return runner._row_with_hash(row)


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for arm in SOURCE_GAUGE_ARM_ORDER:
        action = "PASS" if arm == "gauge_geodesic" else "FAIL"
        outcome = "HONEST" if action == "PASS" else "DECEPTIVE"
        rows.append(
            _row(
                "root-a",
                "event-a",
                arm,
                active_root=True,
                mapped_action=action,
                outcome=outcome,
            )
        )
    for arm in SOURCE_GAUGE_ARM_ORDER:
        rows.append(
            _row(
                "root-b",
                "event-b",
                arm,
                active_root=False,
                mapped_action="PASS",
                outcome="HONEST",
            )
        )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    contract_body = {
        "schema_version": 1,
        "kind": runner.RUNNER_KIND,
        "controller_artifact_sha256": _sha("controller"),
        "input_hashes": {"plan": _sha("plan")},
        "root_ids": ["root-a", "root-b"],
        "event_ids": ["event-a", "event-b"],
        "arm_order": list(SOURCE_GAUGE_ARM_ORDER),
        "actuation_layers": [12, 16, 19, 20],
        "observation_contract": runner._observation_contract(),
        "controller_config": runner._controller_contract(),
        "sampling": runner._sampling_contract(),
        "execution": runner._execution_contract(),
        "scope": runner._scope_contract(),
        "execution_source_sha256": {
            name: _sha(name) for name in runner.REQUIRED_EXECUTION_SOURCE_NAMES
        },
        "model": {"name": "test"},
        "runtime": {"name": "test"},
    }
    contract = {
        **contract_body,
        "contract_sha256": _canonical_sha(contract_body),
    }
    manifest = {
        "schema_version": 1,
        "kind": runner.MANIFEST_KIND,
        "status": "success",
        "contract": contract,
        "completed_rows": 8,
        "completed_root_count": 2,
        "expected_root_count": 2,
        "expected_row_count": 8,
        "expected_event_count": 2,
        "rows_sha256": file_sha256(rows_path),
        "rows_content_sha256": _canonical_sha(rows),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return rows_path, manifest_path


def test_report_separates_policy_wide_and_active_root_effects(tmp_path: Path) -> None:
    rows_path, manifest_path = _artifact(tmp_path)
    manifest, rows = load_relational_gauge_source_run(rows_path, manifest_path)
    report = build_relational_gauge_source_report(
        manifest, rows, bootstrap_resamples=100, bootstrap_seed=5
    )
    assert report["inventory"]["root_count"] == 2
    assert report["inventory"]["active_root_count"] == 1
    active = report["estimands"]["active_roots"]["contrasts"]
    assert (
        active["gauge_geodesic_minus_no_intervention"]
        ["deceptive_probability_difference"]["mean"]
        == -1.0
    )
    matrix = report["no_intervention_transition_matrices"]["gauge_geodesic"]
    assert matrix["DECEPTIVE"]["HONEST"] == 1
    markdown = render_relational_gauge_source_markdown(report)
    assert "no automatic effect gate" in markdown


def test_report_rejects_row_mutation(tmp_path: Path) -> None:
    rows_path, manifest_path = _artifact(tmp_path)
    rows = rows_path.read_text().splitlines()
    changed = json.loads(rows[0])
    changed["behavioral_outcome_class"] = "HONEST"
    rows[0] = json.dumps(changed, sort_keys=True)
    rows_path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RelationalGaugeSourceReportError, match="manifest/rows"):
        load_relational_gauge_source_run(rows_path, manifest_path)


def test_report_rejects_skeletal_or_mutated_execution_contract(
    tmp_path: Path,
) -> None:
    rows_path, manifest_path = _artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["sampling"]["temperature"] = 99.0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(RelationalGaugeSourceReportError, match="contract"):
        load_relational_gauge_source_run(rows_path, manifest_path)
