"""Drift and privacy tests for the c14 representation-structure receipt."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.report_public_representation_receipt import (
    FROZEN_SOURCE_HASHES,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
C14_RECEIPT = REPO_ROOT / "paper_artifacts" / "c14_representation_receipt.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Schema and status checks
# ---------------------------------------------------------------------------
def test_shipped_c14_receipt_status_is_unregistered_descriptive() -> None:
    receipt = _load(C14_RECEIPT)
    assert receipt["status"] == "unregistered_descriptive"
    assert receipt["claim_id"] == "C14_DESCRIPTIVE"
    assert receipt["kind"] == "c14_representation_structure_public_receipt"
    assert receipt["companion_of"] == ["C11", "C13"]


def test_c14_receipt_has_all_five_sections() -> None:
    receipt = _load(C14_RECEIPT)
    for section in (
        "honestward_field",
        "specificity_controls",
        "compression_frontier",
        "additive_compositional_transport",
        "endpoint_prototype_diagnostic",
    ):
        assert section in receipt, f"missing section {section}"
        assert isinstance(receipt[section], dict), f"{section} must be a dict"


def test_c14_receipt_source_hashes_match_frozen() -> None:
    receipt = _load(C14_RECEIPT)
    sa = receipt["source_artifacts"]
    expected = {
        "pre_status_honestward_field_sealed": FROZEN_SOURCE_HASHES["honestward"],
        "pre_status_specificity_controls": FROZEN_SOURCE_HASHES["specificity"],
        "pre_status_compression_frontier": FROZEN_SOURCE_HASHES["compression"],
        "additive_compositional_transport": FROZEN_SOURCE_HASHES["additive"],
        "endpoint_prototype_diagnostic": FROZEN_SOURCE_HASHES["endpoint"],
    }
    for key, expected_hash in expected.items():
        assert key in sa, f"missing source_artifacts.{key}"
        assert sa[key]["sha256"] == expected_hash, (
            f"source_artifacts.{key}: hash mismatch"
        )
        assert sa[key]["byte_size"] > 0, f"source_artifacts.{key}: byte_size missing"


# ---------------------------------------------------------------------------
# Chronology markers
# ---------------------------------------------------------------------------
def test_c14_chronology_all_descriptive() -> None:
    receipt = _load(C14_RECEIPT)
    chron = receipt["chronology"]
    assert chron["honestward_field_sealed"]["tier"] == "retrospective_unregistered_descriptive"
    assert chron["specificity_controls"]["tier"] == "retrospective_unregistered_descriptive"
    assert chron["compression_frontier"]["tier"] == "retrospective_unregistered_descriptive"
    assert chron["additive_compositional_transport"]["tier"] == "post_hoc_registered_follow_up"
    assert chron["endpoint_prototype_diagnostic"]["tier"] == "post_evidence_descriptive_diagnostic"


# ---------------------------------------------------------------------------
# Headline value checks (cross-reference against M6 spine)
# ---------------------------------------------------------------------------
def test_c14_honestward_headline_values() -> None:
    receipt = _load(C14_RECEIPT)
    h = receipt["honestward_field"]

    assert h["primary_view"] == "intervention_masked_action_free"
    assert h["population"]["deceptive_source_roots"] == 200
    assert h["population"]["family_held_out_folds"] == 5

    pv = h["primary_view_models"]
    assert pv["local"]["cosine_mean"] == pytest.approx(0.9288961670689143)
    assert pv["global_mean"]["cosine_mean"] == pytest.approx(0.4839248580990402)
    assert pv["nearest"]["cosine_mean"] == pytest.approx(0.9149124317657036)
    assert pv["local"]["defined_count"] == 193
    assert pv["local"]["total_count"] == 200

    comp = h["primary_view_comparisons"]
    assert comp["global_mean"]["mean_cosine_difference"] == pytest.approx(0.4251746738794928)
    assert comp["global_mean"]["scenario_cluster_ci"] == pytest.approx(
        [0.37993288700738514, 0.46375176933455575]
    )
    assert comp["nearest"]["mean_cosine_difference"] == pytest.approx(0.013957480923283444)
    assert comp["nearest"]["scenario_cluster_ci"][0] == pytest.approx(0.0007401030345003412)
    assert comp["nearest"]["scenario_cluster_ci"][1] == pytest.approx(0.02844740392055282)
    assert comp["shuffled"]["mean_cosine_difference"] == pytest.approx(0.5060698988111998)


def test_c14_specificity_headline_values() -> None:
    receipt = _load(C14_RECEIPT)
    s = receipt["specificity_controls"]

    assert s["interpretation"] == "generic_orbit_transport_explains_most_with_small_dh_directional_margin"
    assert s["primary_view"] == "intervention_masked_action_free"

    mod = s["models"]
    assert mod["honestward_local_calibrated"]["cosine_mean"] == pytest.approx(0.9288699126889872)
    assert mod["generic_all_orbit_local_calibrated"]["cosine_mean"] == pytest.approx(0.9059677666303881)
    assert mod["nuisance_matched_delta_shuffle"]["cosine_mean"] == pytest.approx(0.9174500907506532)

    cmp = s["comparisons"]
    assert cmp["honestward_minus_generic"]["mean_cosine_difference"] == pytest.approx(0.01353053592901028)
    assert cmp["honestward_minus_generic"]["cosine_scenario_cluster_ci"] == pytest.approx(
        [0.0034143359916806357, 0.023705997412509176]
    )
    assert cmp["honestward_minus_generic"]["mean_nse_improvement"] == pytest.approx(0.014373221139071368)


def test_c14_compression_headline_values() -> None:
    receipt = _load(C14_RECEIPT)
    c = receipt["compression_frontier"]

    assert c["interpretation"] == "rank32_output_subspace_retains_full_transport_landmarks_lag"

    mod = c["models"]
    assert mod["full_exemplar_local"]["cosine_mean"] == pytest.approx(0.8786807266613557)
    assert mod["low_rank_projected_full"]["cosine_mean"] == pytest.approx(0.8759664929098006)
    assert mod["global_mean"]["cosine_mean"] == pytest.approx(0.11297378003650456)
    assert mod["landmark_local"]["cosine_mean"] == pytest.approx(0.8472938472455879)

    cmp = c["comparisons"]
    assert cmp["full_minus_low_rank"]["mean_cosine_difference"] == pytest.approx(0.002714233751555027)

    # Every fold selects rank 32
    for fs in c["fold_selections"]:
        assert fs["selected_rank"] == 32
        assert 0.965 < fs["rank_variance_explained"] < 0.975


def test_c14_additive_headline_values() -> None:
    receipt = _load(C14_RECEIPT)
    a = receipt["additive_compositional_transport"]

    assert a["verdict"] == "PASS"
    assert a["action_family_macro_cosine"] == pytest.approx(0.680750493461921)
    assert a["additive_family_macro_cosine"] == pytest.approx(0.8914569246017339)
    assert a["delta_family_macro_cosine"] == pytest.approx(0.2107064311398129)

    # All 5 folds positive
    for fd in a["per_fold"]:
        assert fd["delta"] > 0

    assert a["registration_character"] == "post_hoc_registered_follow_up"


def test_c14_endpoint_headline_values() -> None:
    receipt = _load(C14_RECEIPT)
    e = receipt["endpoint_prototype_diagnostic"]

    assert e["verdict"] == "FAIL"
    assert e["constrained_family_macro_cosine"] == pytest.approx(0.8204666214374028)
    assert e["free_additive_family_macro_cosine"] == pytest.approx(0.8914569246017339)
    assert e["gap_free_minus_constrained"] == pytest.approx(0.07099030316433108)

    for fd in e["per_fold"]:
        assert fd["gap"] > 0

    assert e["registration_character"] == "post_evidence_descriptive_diagnostic"


# ---------------------------------------------------------------------------
# Privacy: no absolute paths or private repo paths
# ---------------------------------------------------------------------------
def test_c14_receipt_does_not_leak_source_paths() -> None:
    serialized = C14_RECEIPT.read_text()
    for pattern in (
        "/Users/",
        "/workspace/",
        "/home/",
        "/private/",
        "/tmp/",
        "/var/",
        "/opt/",
        "/mnt/",
        "/Volumes/",
        "/Applications/",
        "runpod_results",
        "lie-geometry-probes",
        "pod_id",
        "ssh",
        "cost",
        "approval",
        "provider",
    ):
        assert pattern not in serialized, f"privacy leak: {pattern!r} found in receipt"


def test_c14_receipt_has_no_raw_arrays() -> None:
    """Receipt must not contain raw activation arrays — only summary values."""
    receipt = _load(C14_RECEIPT)

    def _deep_check(obj, path: str = "$") -> None:
        if isinstance(obj, list) and len(obj) > 20 and all(
            isinstance(v, (int, float)) for v in obj
        ):
            raise AssertionError(f"raw numeric array at {path} (len={len(obj)})")
        if isinstance(obj, dict):
            for k, v in obj.items():
                _deep_check(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _deep_check(v, f"{path}[{i}]")

    _deep_check(receipt)


# ---------------------------------------------------------------------------
# CLI and producer checks
# ---------------------------------------------------------------------------
def test_c14_cli_requires_sources_and_help_is_provider_neutral(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in (
        "--pre-status-honestward",
        "--pre-status-specificity",
        "--pre-status-compression",
        "--additive-transport",
        "--endpoint-diagnostic",
    ):
        assert option in help_text
    assert "runpod_results" not in help_text
    assert "/Users/" not in help_text
    assert "/workspace/" not in help_text

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2


def test_c14_receipt_producer_sha_matches_current() -> None:
    receipt = _load(C14_RECEIPT)
    import hashlib
    producer_path = REPO_ROOT / "experiments" / "report_public_representation_receipt.py"
    current_sha = hashlib.sha256(producer_path.read_bytes()).hexdigest()
    assert receipt["producer_sha256"] == current_sha, (
        "embedded producer_sha256 is stale; regenerate receipt"
    )


def test_c14_receipt_checks_are_all_true() -> None:
    receipt = _load(C14_RECEIPT)
    for key, value in receipt["checks"].items():
        assert value is True, f"check {key} must be True"


def test_c14_hash_gate_rejects_mismatch(tmp_path: Path) -> None:
    """If a source file hash doesn't match the frozen value, the producer must exit 1."""

    from experiments.report_public_representation_receipt import build_c14_receipt

    # Create all five files with wrong hashes — the first hash check triggers exit
    honest = tmp_path / "h.json"
    honest.write_text("x")
    spec = tmp_path / "s.json"
    spec.write_text("x")
    comp = tmp_path / "c.json"
    comp.write_text("x")
    add_ = tmp_path / "a.json"
    add_.write_text("x")
    end = tmp_path / "e.json"
    end.write_text("x")

    with pytest.raises(SystemExit) as excinfo:
        build_c14_receipt(
            honestward_path=honest,
            specificity_path=spec,
            compression_path=comp,
            additive_path=add_,
            endpoint_path=end,
        )
    assert excinfo.value.code == 1
