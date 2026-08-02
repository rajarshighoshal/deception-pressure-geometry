from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from experiments.plot_public_figures import (
    FIGURE_NAMES,
    REPO_ROOT,
    RECEIPT_SPECS,
    REGISTRY_PATH,
    _extract_c10_truth_aware_from_receipt,
    main,
    parse_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Basic registration / determinism tests
# ---------------------------------------------------------------------------

def test_plot_public_figures_emit_exact_filenames(tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    assert main(["--out-dir", str(out_dir)]) == 0
    names = sorted(path.name for path in out_dir.glob("*.png"))
    assert names == sorted(FIGURE_NAMES)


def test_plot_public_figures_render_nonempty(tmp_path: Path) -> None:
    generated = tmp_path / "figs"
    assert main(["--out-dir", str(generated)]) == 0
    out_paths = sorted(generated.glob("*.png"))
    assert [path.name for path in out_paths] == sorted(FIGURE_NAMES)
    for path in out_paths:
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_public_figures_only_use_receipt_and_registry_inputs() -> None:
    assert REGISTRY_PATH == REPO_ROOT / "docs" / "results_registry.yaml"
    assert REGISTRY_PATH.parent.name == "docs"
    for spec in RECEIPT_SPECS.values():
        path = REPO_ROOT / spec["path"]
        assert path.exists()
        assert path.parent.name == "paper_artifacts"
        assert path.suffix == ".json"
        assert path.match("paper_artifacts/*.json")


def test_plot_public_figures_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "run1"
    second = tmp_path / "run2"
    assert main(["--out-dir", str(first)]) == 0
    assert main(["--out-dir", str(second)]) == 0

    first_hashes = {_digest(file): _sha256(file) for file in first.glob("*.png")}
    second_hashes = {_digest(file): _sha256(file) for file in second.glob("*.png")}
    assert first_hashes == second_hashes


def _readme_figure_names() -> list[str]:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"docs/figures/([^)]+\.png)", readme)))


def test_plot_public_figure_names_match_readme_and_figures() -> None:
    from experiments.plot_representation_figures import (
        COMBINED_STEM,
        FIGURE_STEMS as REPR_STEMS,
        MOBILE_FIGURE_STEMS,
        PRESSURE_STEM,
        SOCIAL_CARD_NAME,
    )

    readme_expected = sorted(set(FIGURE_NAMES) | {f"{stem}.png" for stem in REPR_STEMS})
    assert _readme_figure_names() == readme_expected

    article_only = {f"{stem}.png" for stem in MOBILE_FIGURE_STEMS} | {SOCIAL_CARD_NAME}
    blog_variants = {f"{stem}_blog.png" for stem in REPR_STEMS} | {
        f"{stem}_blog.png" for stem in MOBILE_FIGURE_STEMS
    }
    required = set(readme_expected) | article_only | blog_variants
    manuscript_optional = {f"{COMBINED_STEM}.png", f"{PRESSURE_STEM}.png"}
    figures_dir = REPO_ROOT / "docs" / "figures"
    actual = {p.name for p in figures_dir.glob("*.png")}
    assert required <= actual, sorted(required - actual)
    assert actual <= required | manuscript_optional, sorted(actual - required)


def _digest(path: Path) -> str:
    return path.name


# ---------------------------------------------------------------------------
# Semantic regression tests — parse_data() exposes correct records
# ---------------------------------------------------------------------------

def test_parse_c9_pressure_values() -> None:
    """Verify C9 pressure data is loaded with correct values from receipt."""
    c9 = parse_data()["c9"]

    # Late-compressed rates match receipt
    scr = c9["outcomes"]["scripted"]["arm_summary"]
    assert scr["latedump"]["p1b_deceptive_commitment"]["point"] == 0.5
    adp = c9["outcomes"]["adaptive"]["arm_summary"]
    assert adp["latedump"]["p1b_deceptive_commitment"]["point"] == 0.71875

    # Smooth vs late-compressed contrasts are registered
    assert c9["outcomes"]["adaptive"]["contrasts"]["P2a"]["status"] == "registered"
    assert c9["outcomes"]["adaptive"]["contrasts"]["P2a"]["verdict"] == "found"


def test_parse_c9_hazard_gamma_ci_excludes_zero() -> None:
    """Gamma CI excludes zero (lower bound > 0) and no n.s./p-value text."""
    c9 = parse_data()["c9"]
    gamma = c9["hazard"]["adaptive_bank"]["adaptive_coefficients"]["gamma"]
    assert gamma["lo"] > 0.0, f"gamma CI lower bound {gamma['lo']} should exceed zero"
    assert gamma["hi"] > gamma["lo"]
    # Ensure the receipt itself does not contain "n.s." or "p <"
    import json
    raw = json.dumps(c9["hazard"]["adaptive_bank"])
    assert "n.s." not in raw, "receipt must not contain 'n.s.' annotation"
    assert "p <" not in raw, "receipt must not contain 'p <' annotation"


def test_parse_c9_descriptive_flow() -> None:
    """Descriptive flow values match receipt."""
    c9 = parse_data()["c9"]
    flow = c9["descriptive_pressure_flow"]
    assert flow["n_pseudo_orbits"] == 1800
    assert flow["monotonicity"]["probe_depth"]["median_spearman"] == 0.7142857142857144
    assert flow["cross_family_field_cosine"]["median"] == 0.9072723686695099


def test_parse_c10_truth_aware_values() -> None:
    """Truth-aware C10 Brier values are loaded from receipt JSON."""
    data = parse_data()
    ta = data["truth_aware_c10"]

    # Verify values match receipt (full precision)
    assert ta["nuisance_prior_brier"] == pytest.approx(0.027482589785385363)
    assert ta["linear_brier"] == pytest.approx(0.0015015367445592714)
    assert ta["linear_gain_over_truth_aware"] == pytest.approx(0.025981053040826096)
    assert ta["graph_gain_over_truth_aware"] == pytest.approx(0.006964244566928011)
    assert ta["graph_gain_ci_crosses_zero"] is True
    assert ta["verdict"].startswith("refuted-under-adequate-instrument")
    assert "truth-aware" in ta["verdict"].casefold()
    assert ta["families_positive"] == (16, 20)

    # Verify graph_brier from receipt (exact value, not rounded computation)
    assert ta["graph_brier"] == pytest.approx(0.020518345218457357)

    # Receipt values for graph/linear (raw, not truth-aware)
    c10 = data["c10"]
    assert c10["primary"]["models"]["local_joint_top8"]["family_macro_brier"] > 0.020
    assert c10["linear_probe_comparator"]["family_macro_brier"]["registered_probe"] < 0.002


def test_c10_truth_aware_comes_from_receipt_not_registry() -> None:
    """Prove truth-aware C10 values come strictly from the C10 receipt JSON,
    not from parsing registry boundary text.
    """
    c10 = parse_data()["c10"]
    result = _extract_c10_truth_aware_from_receipt(c10)

    data = parse_data()
    ta = data["truth_aware_c10"]
    for key in ("nuisance_prior_brier", "graph_brier", "linear_brier",
                "graph_gain_over_truth_aware", "linear_gain_over_truth_aware",
                "graph_gain_ci_crosses_zero", "verdict"):
        assert result[key] == ta[key], (
            f"receipt parse mismatch for {key}: {result[key]} != {ta[key]}"
        )

    # Verify extracting from a receipt missing the section fails loudly
    c10_no_ta = dict(c10)
    c10_no_ta.pop("truth_aware_rescore")
    with pytest.raises(SystemExit):
        _extract_c10_truth_aware_from_receipt(c10_no_ta)


def test_parse_c11_pre_action_geometry_gain() -> None:
    """C11 primary geometry-only log-loss gain matches receipt."""
    c11 = parse_data()["c11"]
    risk = c11["risk_gate_repair"]
    geo_gain = risk["interpretation"]["primary_geometry_only_log_loss_gain_over_nuisance"]
    assert abs(geo_gain - (-0.021776526022848856)) < 1e-10
    geo_ci = risk["interpretation"]["primary_geometry_only_log_loss_gain_ci"]
    assert geo_ci["interval"][0] < 0
    assert geo_ci["interval"][1] > 0
    # Must have model_scores for nested models
    assert "nuisance_plus_geometry_logistic" in risk["model_scores"], "nested model scores missing"


def test_parse_c1_cng_harm_is_1_not_zero() -> None:
    """C1 CNG honest_harms is 1, not zero."""
    c1 = parse_data()["c1"]
    cng = c1["policies"]["context_chart_feature_gate_equivariant_neural_context"]["summary"]
    assert cng["honest_harms"] == 1, f"CNG harms should be 1, got {cng['honest_harms']}"


def test_parse_c1_route_ridge_data() -> None:
    """Route-feature ridge exists in policies and has correct mismatch counts."""
    c1 = parse_data()["c1"]
    # The ridge is stored under learned_context_ridge_reward
    assert "learned_context_ridge_reward" in c1["policies"]
    ridge = c1["policies"]["learned_context_ridge_reward"]
    assert ridge["summary"]["fixes_error"] == 170
    assert ridge["summary"]["honest_harms"] == 11
    # Check info audit mismatch
    info = c1["information_audit"]["policy_information"]["learned_context_ridge_reward"]
    assert info["selected_target_route_mismatches"] == 233
    assert info["route_truth_mismatches"] == 0


def test_parse_c1_route_count_1680() -> None:
    """Gate L20 routing diagnostic n=1680, accuracy=1.0."""
    c1 = parse_data()["c1"]
    gate = c1["descriptive_structural_evidence"]["gate_l20_routing_diagnostic"]
    assert gate["n"] == 1680
    assert gate["routing_correct_rate"] == 1.0
    assert gate["target_status_accuracy"] == 1.0


def test_parse_c1_fresh_atlas_values() -> None:
    """Fresh atlas policy fixes: context 71, response 79, route floor 64, margin 79."""
    c1 = parse_data()["c1"]
    fresh = c1["descriptive_structural_evidence"]["fresh_equivariant_atlas"]
    ps = fresh["policy_summaries"]
    assert ps["atlas_context_local_k21_strict"]["deceptive_strict_fixes"] == 71
    assert ps["atlas_response_local_k5_strict"]["deceptive_strict_fixes"] == 79
    assert ps["route_hybrid_mean_probe"]["deceptive_strict_fixes"] == 64
    assert ps["margin_argmax_all"]["deceptive_strict_fixes"] == 79


def test_parse_c1_local_vs_global_flow() -> None:
    """Local flow proxy 275/600 vs global 52/600."""
    c1 = parse_data()["c1"]
    loc = c1["descriptive_structural_evidence"]["locality_proxy"]["policy_summaries"]
    assert loc["local_control_flow_context"]["fixes_error"] == 275
    assert loc["global_control_flow_context"]["fixes_error"] == 52


def test_parse_c1_saved_field_values() -> None:
    """Saved-field candidate fixes from receipt."""
    c1 = parse_data()["c1"]
    sf = c1["descriptive_structural_evidence"]["saved_field"]
    assert sf["chart_distilled_context_rf"]["fixes_error"] == 584
    assert sf["product_z2_context_rf"]["fixes_error"] == 581
    assert sf["graph_mean_context_cauto_gauto_d12_strict"]["fixes_error"] == 598


def test_parse_c5_transition_counts() -> None:
    """C5 transition counts match receipt for all three arms."""
    c5 = parse_data()["c5"]
    arms = c5["evaluation"]["arms"]

    lin = arms["family_matched_linear"]["machine_status"]
    assert lin["deceptive_fixes"] == 21
    assert lin["deceptive_harms"] == 5
    assert lin["honest_fixes"] == 26
    assert lin["honest_harms"] == 6

    nat = arms["native_gated"]["machine_status"]
    assert nat["deceptive_fixes"] == 0
    assert nat["deceptive_harms"] == 0

    freq = arms["frequent_early_window"]["machine_status"]
    assert freq["deceptive_fixes"] == 3
    assert freq["deceptive_harms"] == 7
    assert freq["honest_fixes"] == 5
    assert freq["honest_harms"] == 5


def test_parse_c5_populations_and_family_counts() -> None:
    """C5 arm populations and family effects are consistent."""
    c5 = parse_data()["c5"]
    arms = c5["evaluation"]["arms"]

    for arm_key in ("native_gated", "frequent_early_window", "family_matched_linear"):
        arm = arms[arm_key]
        assert arm["population"]["deceptive"] == 48
        assert arm["population"]["honest"] == 48
        assert arm["population"]["total"] == 96

        family_data = arm["machine_status_by_family"]
        assert isinstance(family_data, dict)
        assert len(family_data) == 4, f"Expected 4 held-out families, got {len(family_data)}"

        # Sum of family fixes should match aggregate
        total_df = sum(f["deceptive_fixes"] for f in family_data.values())
        assert total_df == arm["machine_status"]["deceptive_fixes"], \
            f"{arm_key}: family sum {total_df} != agg {arm['machine_status']['deceptive_fixes']}"

        total_hf = sum(f["honest_fixes"] for f in family_data.values())
        assert total_hf == arm["machine_status"]["honest_fixes"]


def test_parse_c13_support_counts_sum_to_402() -> None:
    """C13 proposal support counts sum to 402 including zero_direction."""
    c13 = parse_data()["c13"]
    status = c13["causal_replay"]["proposal_status_counts"]
    total = (status["active"] + status["boundary_exit"] + status["field_undefined"]
             + status["off_support"] + status["zero_direction"])
    assert total == 402, f"Support counts sum to {total}, expected 402"
    assert status["active"] == 21
    assert status["boundary_exit"] == 333
    assert status["field_undefined"] == 37
    assert status["off_support"] == 10
    assert status["zero_direction"] == 1


def test_parse_c13_gauge_contrasts() -> None:
    """C13 gauge vs no-control is exactly zero."""
    c13 = parse_data()["c13"]
    all_roots = c13["causal_replay"]["contrasts"]["all_roots"]
    gnc = all_roots["gauge_geodesic_minus_no_intervention"]
    assert gnc["point"] == 0.0
    assert gnc["ci95"] == [0.0, 0.0]
    assert gnc["roots"] == 402


def test_parse_c13_no_transport_claim() -> None:
    """C13 receipt does not contain holonomy/curvature verdict text."""
    c13 = parse_data()["c13"]
    # holonomy instrument explicitly failed
    assert c13["holonomy_instrument"]["adequate_folds"] == 0
    assert c13["holonomy_instrument"]["verdict"] == "not-found-under-this-instrument"
