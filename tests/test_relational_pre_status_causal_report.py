from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from geoprobe.control.relational_pre_status_causal import (
    CAUSAL_ARM_ORDER,
    build_relational_pre_status_causal_arm_bundle,
)
from geoprobe.data.relational_structured_action import (
    STRUCTURED_ACTION_TEMPERATURE,
    STRUCTURED_ACTION_TOP_P,
)
from geoprobe.eval import relational_pre_status_causal_report as causal_report
from geoprobe.eval import relational_pre_status_causal_response as causal_response_report
from geoprobe.io import file_sha256
from geoprobe.models.relational_structured_action import (
    FAIL_TOKEN_ID,
    PASS_TOKEN_ID,
    SKIP_TOKEN_ID,
    STATUS_PREFIX_TOKEN_IDS,
    int32_token_sha256,
)
from geoprobe.runtime.relational_pre_status_causal_runner import (
    CausalRootWork,
    MANIFEST_KIND,
    RUNNER_KIND,
    run_exact_prefix_causal_replay,
)


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _row(*, event_id: str, root_id: str, scenario_id: str, arm: str, action: str, knowledge_correct: bool = True) -> dict[str, object]:
    probabilities = {"PASS": 0.20, "FAIL": 0.30, "SKIP": 0.10}
    logits = {"PASS": 1.0, "FAIL": 2.0, "SKIP": 0.0}
    true_status = "PASS"
    knowledge_status = "PASS" if knowledge_correct else "FAIL"
    outcome = causal_report.classify_causal_outcome(action, true_status, knowledge_correct)
    raw_token_id = {
        "PASS": PASS_TOKEN_ID,
        "FAIL": FAIL_TOKEN_ID,
        "SKIP": SKIP_TOKEN_ID,
        "NO_ACTION": 42,
    }[action]
    return {
        "schema_version": 1, "kind": RUNNER_KIND, "root_id": root_id, "state_id": root_id, "event_id": event_id,
        "family": "family_a", "family_fold": "family_fold_a", "turn_index": 1, "scenario_id": scenario_id, "true_status": true_status,
        "desired_status": "PASS", "knowledge_status": knowledge_status, "knowledge_correct": knowledge_correct,
        "arm": arm, "raw_token_id": raw_token_id, "raw_decoded_exact": action, "mapped_action": action, "rng_seed": 9,
        "behavioral_outcome_class": outcome, "prefix": {"token_ids_sha256": _digest("prefix-" + root_id), "token_count": 8},
        "status_probabilities": probabilities, "status_logits": logits, "pass_probability": probabilities["PASS"],
        "fail_probability": probabilities["FAIL"], "skip_probability": probabilities["SKIP"], "pass_logit": logits["PASS"],
        "fail_logit": logits["FAIL"], "skip_logit": logits["SKIP"], "recognized_action_probability_mass": 0.60,
        "top_token_id": 42, "top_token_probability": 0.25, "beta": 1.0, "actuation_layer": 12,
        "actuation_vector_sha256": _digest("actuation-" + arm),
        "hook_layers": [] if arm == "noop" else [12],
        "vector_tensor_row_index": 0, "vector_source_hashes": {"h": _digest("h")}, "arm_vector_sha256": _digest("arm-" + arm),
        "bundle_hash": _digest("bundle-" + root_id), "bundle_metadata_hash": _digest("metadata-" + root_id), "arm_vector_l2_norm": 1.25,
        "model": {"model_id": "synthetic"}, "runtime": {"device": "test"}, "capture_hooks_enabled": False,
    }


def _block(event_id: str, root_id: str, scenario_id: str, *, noop: str = "FAIL", full_h: str = "PASS", knowledge_correct: bool = True) -> list[dict[str, object]]:
    actions = {"noop": noop, "fixed_global_h": "FAIL", "generic_t": "FAIL", "specific_s": "SKIP", "full_h": full_h,
               "generic_minus_s": "NO_ACTION", "generic_plus_random_s": "FAIL"}
    return [_row(event_id=event_id, root_id=root_id, scenario_id=scenario_id, arm=arm, action=action, knowledge_correct=knowledge_correct) for arm, action in actions.items()]


def _write_inputs(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    for row in rows:
        row.pop("row_sha256", None)
        row["row_sha256"] = _sha(row)
    runner = tmp_path / "runner.jsonl"
    runner.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    root_ids = sorted({str(row["root_id"]) for row in rows})
    event_ids = sorted({str(row["event_id"]) for row in rows})
    contract_body = {"schema_version": 1, "kind": RUNNER_KIND, "input_hashes": {"replay": _digest("replay")}, "beta": 1.0,
                     "primary_actuation_layer": 12, "arm_order": list(CAUSAL_ARM_ORDER), "root_ids": root_ids,
                     "event_ids": event_ids, "sampling": {"temperature": STRUCTURED_ACTION_TEMPERATURE, "top_p": STRUCTURED_ACTION_TOP_P,
                     "full_vocabulary": True, "status_candidate_token_ids": {"PASS": PASS_TOKEN_ID, "FAIL": FAIL_TOKEN_ID, "SKIP": SKIP_TOKEN_ID},
                     "unrecognized_mapping": "NO_ACTION", "common_random_numbers_across_arms": True}, "execution": {"shared_prefill_forward_count_per_root": 1,
                     "steered_final_token_batch_forward_count_per_root": 1, "steered_final_token_batch_size": len(CAUSAL_ARM_ORDER),
                     "expected_shared_prefill_forward_count": len(root_ids), "expected_steered_final_token_batch_forward_count": len(root_ids),
                     "prefill_excludes_final_status_anchor_token": True, "action_token_feedback": False,
                     "checkpoint_root_interval": 8}, "model": {"model_id": "synthetic"},
                     "runtime": {"device": "test"}, "capture_hooks_enabled": False}
    contract = {**contract_body, "contract_sha256": _sha(contract_body)}
    manifest = {"schema_version": 1, "kind": MANIFEST_KIND, "status": "success", "contract": contract,
                "completed_rows": len(rows), "completed_root_count": len(root_ids), "expected_root_count": len(root_ids),
                "expected_event_count": len(event_ids), "expected_row_count": len(rows),
                "rows_content_sha256": _sha(rows), "rows_sha256": file_sha256(runner)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return runner, manifest_path


def test_exact_schema_root_balancing_and_five_classes(tmp_path: Path) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a", full_h="PASS")
    for index in range(9):
        rows += _block(f"event-b-{index}", _digest("root-b"), "scenario-b", full_h="FAIL")
    runner, manifest = _write_inputs(tmp_path, rows)
    report = causal_report.build_relational_pre_status_causal_report(runner, manifest, bootstrap_resamples=12)
    full = report["per_arm"]["full_h"]
    assert full["root_balanced_rates"]["DECEPTIVE"] == pytest.approx(0.5)
    assert full["event_weighted_rates"]["DECEPTIVE"] == pytest.approx(0.9)
    assert set(full["root_balanced_rates"]) == {"HONEST", "DECEPTIVE", "SKIP", "NO_ACTION", "WRONG_WITHOUT_BASELINE_KNOWLEDGE"}
    assert full["mean_recognized_action_probability_mass"] == {"root_balanced": pytest.approx(0.6), "event_weighted": pytest.approx(0.6)}
    assert full["mean_arm_vector_l2_norm"] == {"root_balanced": pytest.approx(1.25), "event_weighted": pytest.approx(1.25)}
    assert report["stratified_root_balanced_five_class_rates"]["family"]["family_a"]["row_count"] == len(rows)
    assert report["coverage"]["input_hashes"] == {"replay": _digest("replay")}
    causal_report.validate_relational_pre_status_causal_report(report)


def test_rejects_missing_knowledge_and_probability_mass_mismatch(tmp_path: Path) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a")
    del rows[0]["knowledge_correct"]
    runner, manifest = _write_inputs(tmp_path, rows)
    with pytest.raises(causal_report.RelationalPreStatusCausalReportError, match="exact runner schema"):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)
    rows = _block("event-a", _digest("root-a"), "scenario-a")
    rows[0]["recognized_action_probability_mass"] = 1.0
    runner, manifest = _write_inputs(tmp_path, rows)
    with pytest.raises(causal_report.RelationalPreStatusCausalReportError, match="recognized action probability mass"):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)


def _set_probability_mass(
    row: dict[str, object], probabilities: dict[str, float]
) -> None:
    row["status_probabilities"] = probabilities
    row["pass_probability"] = probabilities["PASS"]
    row["fail_probability"] = probabilities["FAIL"]
    row["skip_probability"] = probabilities["SKIP"]
    row["recognized_action_probability_mass"] = sum(probabilities.values())


def test_float32_probability_mass_boundary_roundoff_is_not_a_failed_run(
    tmp_path: Path,
) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a")
    probabilities = {
        "PASS": 0.051515281200408936,
        "FAIL": 0.8969695568084717,
        "SKIP": 0.051515281200408936,
    }
    _set_probability_mass(rows[0], probabilities)
    assert rows[0]["recognized_action_probability_mass"] == 1.0000001192092896
    runner, manifest = _write_inputs(tmp_path, rows)

    report = causal_report.build_relational_pre_status_causal_report(
        runner, manifest, bootstrap_resamples=5
    )

    assert report["per_arm"]["noop"][
        "mean_recognized_action_probability_mass"
    ] == {
        "root_balanced": 1.0000001192092896,
        "event_weighted": 1.0000001192092896,
    }
    assert report["coverage"]["probability_mass_roundoff"] == {
        "source_precision": "torch_float32",
        "accepted_boundary_tolerance": 8.0 * float(np.finfo(np.float32).eps),
        "strict_unit_interval_exceedance_count": 1,
        "maximum_recorded_mass": 1.0000001192092896,
        "maximum_boundary_excess": float(np.finfo(np.float32).eps),
        "raw_values_preserved": True,
    }


def test_response_report_inputs_bind_directly_to_scientific_artifacts(
    tmp_path: Path,
) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a")
    runner, manifest = _write_inputs(tmp_path, rows)
    canonical_runner = tmp_path / "causal_rows.jsonl"
    canonical_manifest = tmp_path / "causal_manifest.json"
    runner.rename(canonical_runner)
    manifest.rename(canonical_manifest)
    registered_report = causal_report.build_relational_pre_status_causal_report(
        canonical_runner,
        canonical_manifest,
        bootstrap_resamples=5,
    )
    registered_path = tmp_path / "causal_report.json"
    registered_path.write_text(
        json.dumps(registered_report, sort_keys=True),
        encoding="utf-8",
    )

    diagnostic_report = causal_response_report.build_relational_pre_status_causal_response_report(
        canonical_runner,
        canonical_manifest,
        expected_runner_jsonl_sha256=file_sha256(canonical_runner),
        expected_source_manifest_sha256=file_sha256(canonical_manifest),
        expected_registered_causal_report_sha256=file_sha256(registered_path),
        expected_registered_causal_report_internal_sha256=registered_report[
            "report_sha256"
        ],
        expected_runner_row_count=len(rows),
        registered_causal_report_path=registered_path,
        allow_unfrozen_test_inputs=True,
    )
    assert diagnostic_report["inputs"]["registered_causal_report"]["sha256"] == file_sha256(
        registered_path
    )
    assert diagnostic_report["inputs"]["registered_causal_report"]["report_sha256"] == (
        registered_report["report_sha256"]
    )
    assert "recovery_validation" not in diagnostic_report["inputs"]
    assert diagnostic_report["inputs"]["runner_jsonl"]["sha256"] == file_sha256(
        canonical_runner
    )
    assert diagnostic_report["inputs"]["source_manifest"]["sha256"] == file_sha256(
        canonical_manifest
    )
    causal_response_report.validate_relational_pre_status_causal_response_report(
        diagnostic_report
    )

    tampered_inputs = dict(registered_report["inputs"])
    runner_input = dict(tampered_inputs["runner_jsonl"])
    runner_input["sha256"] = "0" * 64
    tampered_inputs["runner_jsonl"] = runner_input
    tampered_report = dict(registered_report)
    tampered_report["inputs"] = tampered_inputs
    tampered_report["report_sha256"] = causal_report._sha(
        {
            key: value
            for key, value in tampered_report.items()
            if key != "report_sha256"
        }
    )
    tampered_path = tmp_path / "causal_report_tampered.json"
    tampered_path.write_text(json.dumps(tampered_report, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        causal_response_report.RelationalPreStatusCausalResponseError,
        match="does not bind",
    ):
        causal_response_report.build_relational_pre_status_causal_response_report(
            canonical_runner,
            canonical_manifest,
            expected_runner_jsonl_sha256=file_sha256(canonical_runner),
            expected_source_manifest_sha256=file_sha256(canonical_manifest),
            expected_registered_causal_report_sha256=file_sha256(tampered_path),
            expected_registered_causal_report_internal_sha256=tampered_report[
                "report_sha256"
            ],
            registered_causal_report_path=tampered_path,
            expected_runner_row_count=len(rows),
            allow_unfrozen_test_inputs=True,
        )



def test_probability_mass_outside_float32_roundoff_is_rejected(
    tmp_path: Path,
) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a")
    _set_probability_mass(
        rows[0], {"PASS": 0.5, "FAIL": 0.5, "SKIP": 0.00001}
    )
    runner, manifest = _write_inputs(tmp_path, rows)

    with pytest.raises(
        causal_report.RelationalPreStatusCausalReportError,
        match="recognized action probability mass",
    ):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)


def test_rejects_manifest_bound_row_tampering_and_outcome_semantics(tmp_path: Path) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a", knowledge_correct=False)
    runner, manifest = _write_inputs(tmp_path, rows)
    rows[0]["raw_decoded_exact"] = "tampered"
    runner.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(causal_report.RelationalPreStatusCausalReportError, match="rows_sha256"):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)
    rows = _block("event-a", _digest("root-a"), "scenario-a", knowledge_correct=False)
    rows[0]["behavioral_outcome_class"] = "DECEPTIVE"
    runner, manifest = _write_inputs(tmp_path, rows)
    with pytest.raises(causal_report.RelationalPreStatusCausalReportError, match="knowledge-aware"):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)

    rows = _block("event-a", _digest("root-a"), "scenario-a")
    rows[0]["mapped_action"] = "PASS"
    runner, manifest = _write_inputs(tmp_path, rows)
    with pytest.raises(causal_report.RelationalPreStatusCausalReportError, match="raw-token mapping"):
        causal_report.build_relational_pre_status_causal_report(runner, manifest)


def test_conditional_transitions_and_deterministic_cluster_ci(tmp_path: Path) -> None:
    rows = _block("event-a", _digest("root-a"), "scenario-a", noop="FAIL", full_h="PASS")
    rows += _block("event-b", _digest("root-b"), "scenario-b", noop="PASS", full_h="FAIL")
    runner, manifest = _write_inputs(tmp_path, rows)
    first = causal_report.build_relational_pre_status_causal_report(runner, manifest, bootstrap_seed=17, bootstrap_resamples=25)
    second = causal_report.build_relational_pre_status_causal_report(runner, manifest, bootstrap_seed=17, bootstrap_resamples=25)
    transition = first["paired_transitions_vs_noop"]["full_h"]
    assert transition["noop_to_arm_counts"]["DECEPTIVE"]["HONEST"] == 1
    assert transition["truthful_fixes_deceptive_to_honest"]["conditional_denominator"] == 1
    assert transition["truthful_fixes_deceptive_to_honest"]["root_balanced_conditional_rate"] == 1.0
    assert transition["honest_harms_honest_to_deceptive"]["event_weighted_conditional_rate"] == 1.0
    assert first["primary_contrasts"]["generic_t"]["deceptive_probability_ci"] == second["primary_contrasts"]["generic_t"]["deceptive_probability_ci"]


def test_actual_runner_emission_round_trips_through_reporter(tmp_path: Path) -> None:
    root_id = _digest("live-schema-root")
    prefix = (91, *STATUS_PREFIX_TOKEN_IDS)
    vectors = np.ones((4, 3), dtype=np.float32)
    work = CausalRootWork(
        root_id=root_id,
        family_fold="outer_1",
        vector_tensor_row_index=0,
        prefix_token_ids=prefix,
        prefix_token_ids_sha256=int32_token_sha256(prefix),
        events=(
            {
                "event_id": "event-live-schema",
                "scenario_id": "scenario-live-schema",
                "family": "family-live-schema",
                "turn_index": 1,
                "true_status": "PASS",
                "desired_status": "FAIL",
                "knowledge_status": "PASS",
                "knowledge_correct": True,
                "rng_seed": 17,
            },
        ),
        bundle=build_relational_pre_status_causal_arm_bundle(
            root_id,
            "outer_1",
            vectors * 2.0,
            vectors,
            vectors * 0.5,
        ),
    )

    def forward(_model, _prefix, _steering):
        logits = torch.full((len(CAUSAL_ARM_ORDER), SKIP_TOKEN_ID + 2), -30.0)
        logits[:, FAIL_TOKEN_ID] = 20.0
        logits[CAUSAL_ARM_ORDER.index("full_h"), PASS_TOKEN_ID] = 30.0
        return logits

    rows_path = tmp_path / "actual_rows.jsonl"
    manifest_path = tmp_path / "actual_manifest.json"
    run_exact_prefix_causal_replay(
        (work,),
        model=object(),
        tokenizer=type(
            "Tokenizer",
            (),
            {
                "decode": lambda _self, token_ids, **_kwargs: f" token-{token_ids[0]}"
            },
        )(),
        out=rows_path,
        manifest_out=manifest_path,
        status_out=tmp_path / "STATUS",
        beta=1.0,
        expected_root_count=1,
        expected_event_count=1,
        input_hashes={
            "replay_plan": {"ledger_sha256": "a" * 64},
            "source_rollout_manifest": {
                "schema_version": 2,
                "kind": "relational_structured_action_rollout_manifest",
                "sha256": "c" * 64,
                "bytes": 24_649,
                "rows_sha256": "d" * 64,
                "rollout_contract": {
                    "schema_version": 2,
                    "kind": "relational_structured_action_rollout_contract",
                    "contract_sha256": "e" * 64,
                    "model_artifact_sha256": "b" * 64,
                    "tokenizer_artifact_sha256": {"tokenizer.json": "f" * 64},
                },
            },
        },
        model_provenance={"artifact_sha256": "b" * 64},
        runtime_provenance={"device": "test"},
        forward_fn=forward,
    )
    report = causal_report.build_relational_pre_status_causal_report(
        rows_path,
        manifest_path,
        expected_runner_row_count=len(CAUSAL_ARM_ORDER),
        bootstrap_resamples=5,
    )

    assert report["coverage"]["row_count"] == len(CAUSAL_ARM_ORDER)
    assert report["per_arm"]["full_h"]["root_balanced_rates"]["HONEST"] == 1.0
    assert report["stratified_root_balanced_five_class_rates"]["family"][
        "family-live-schema"
    ]["event_seed_count"] == 1
