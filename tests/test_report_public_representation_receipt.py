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
        "simple_address_baselines",
        "within_cell_retrieval",
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
        "simple_address_baselines": FROZEN_SOURCE_HASHES["simple_address"],
        "within_cell_retrieval": FROZEN_SOURCE_HASHES["within_cell"],
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
    simple = tmp_path / "sa.json"
    simple.write_text("x")
    within = tmp_path / "wc.json"
    within.write_text("x")

    with pytest.raises(SystemExit) as excinfo:
        build_c14_receipt(
            honestward_path=honest,
            specificity_path=spec,
            compression_path=comp,
            additive_path=add_,
            endpoint_path=end,
            simple_address_path=simple,
            within_cell_path=within,
        )
    assert excinfo.value.code == 1


def test_registry_compression_variance_prose_matches_receipt() -> None:
    """The registry's rank-32 variance range must equal the receipt's fold values.

    Drift guard: the two prose statements in docs/results_registry.yaml carry the
    fold-selection variance range as NN.NN--NN.NN%. They must match the receipt's
    min/max rank_variance_explained to two decimals, in both locations.
    """
    import re

    receipt = json.loads(
        (REPO_ROOT / "paper_artifacts" / "c14_representation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    fold_vars = [
        fold["rank_variance_explained"]
        for fold in receipt["compression_frontier"]["fold_selections"]
    ]
    expected = f"{min(fold_vars) * 100:.2f}--{max(fold_vars) * 100:.2f}%"

    registry_text = (REPO_ROOT / "docs" / "results_registry.yaml").read_text(
        encoding="utf-8"
    )
    stated = re.findall(r"(\d{2}\.\d{2}--\d{2}\.\d{2}%)\s*(?:variance retained|train variance)",
                        registry_text)
    assert stated, "registry no longer states a rank-32 variance range"
    assert len(stated) == 2, f"expected the range in exactly 2 places, found {len(stated)}"
    for value in stated:
        assert value == expected, (
            f"registry states {value!r} but receipt fold values give {expected!r}"
        )


def test_simple_address_block_matches_source_report() -> None:
    """The receipt's simple-address block must mirror the frozen source values."""
    receipt = json.loads(C14_RECEIPT.read_text(encoding="utf-8"))
    block = receipt["simple_address_baselines"]
    assert block["fidelity_gate_passed"] is True
    view = block["models"]["intervention_masked_action_free"]
    assert set(view) == {
        "raw_nn",
        "raw_k8",
        "design_cell_mean",
        "global_mean_recomputed",
    }
    for info in view.values():
        assert info["defined_count"] == info["total_count"] == 200
    comps = block["comparisons"]["intervention_masked_action_free"]
    graph_vs_k8 = comps["graph_local_minus_raw_k8"]
    lo, hi = graph_vs_k8["scenario_cluster_ci"]
    assert hi < 0.0, "raw_k8 should exceed graph local with a CI excluding zero"
    graph_vs_cell = comps["graph_local_minus_design_cell_mean"]
    lo2, hi2 = graph_vs_cell["scenario_cluster_ci"]
    assert lo2 < 0.0 < hi2, "design-cell comparison should be a statistical tie"
    assert receipt["checks"]["simple_addresses_match_or_exceed_graph_local"] is True
    assert receipt["checks"]["design_cell_mean_statistically_ties_graph_local"] is True


def test_within_cell_block_matches_source_report() -> None:
    """The within-cell block must mirror the frozen diagnostic's registered outputs."""
    receipt = json.loads(C14_RECEIPT.read_text(encoding="utf-8"))
    block = receipt["within_cell_retrieval"]
    assert block["fidelity_gate_passed"] is True
    assert block["baseline_replication_gate"] == "passed"
    view = block["models"]["intervention_masked_action_free"]
    within = view["within_cell_nn"]
    assert within["defined_count"] == 195
    assert within["total_count"] == 200
    assert abs(within["cosine_mean"] - 0.947135) < 5e-6
    comps = block["comparisons"]["intervention_masked_action_free"]
    tie = comps["within_cell_nn_minus_design_cell_mean"]
    lo, hi = tie["scenario_cluster_ci"]
    assert lo < 0.0 < hi, "within-cell selection should statistically tie the cell mean"
    assert tie["n_pairs"] == 195
    identical = comps["within_cell_nn_minus_raw_nn"]
    assert identical["mean_cosine_difference"] == 0.0
    assert identical["scenario_cluster_ci"] == [0.0, 0.0], (
        "the nearest neighbour must land in-cell on every covered query"
    )
    marginal = comps["raw_k8_minus_design_cell_mean"]
    lo2, hi2 = marginal["scenario_cluster_ci"]
    assert lo2 < 0.0 < hi2, "the marginal raw-vs-cell contrast is a statistical tie"
    support = block["within_cell_support"]["intervention_masked_action_free"]
    assert support["min_support_for_definition"] == 2
    assert receipt["checks"]["within_cell_selection_ties_cell_mean"] is True
    assert receipt["checks"][
        "nearest_neighbour_lands_in_cell_on_all_covered_queries"
    ] is True


def test_receipt_contains_no_private_repository_paths() -> None:
    """The public receipt must not reference files that exist only privately."""
    text = C14_RECEIPT.read_text(encoding="utf-8")
    assert "docs/registrations" not in text
    assert "/Users/" not in text
    assert "lie-geometry-probes" not in text
