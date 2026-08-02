from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.plot_representation_figures import (
    EXPECTED_KIND,
    FIGURE_NAMES,
    MOBILE_FIGURE_STEMS,
    RECEIPT_PATH,
    SOCIAL_CARD_NAME,
    REPO_ROOT,
    load_receipt,
    main,
    parse_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_emit_exact_filenames(tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    assert main(["--out-dir", str(out_dir)]) == 0
    names = sorted(path.name for path in out_dir.glob("*.*"))
    assert names == sorted(FIGURE_NAMES)


def test_render_nonempty_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    assert main(["--out-dir", str(out_dir)]) == 0
    for path in sorted(out_dir.glob("*.png")):
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for path in sorted(out_dir.glob("*.pdf")):
        assert path.stat().st_size > 5_000
        assert path.read_bytes().startswith(b"%PDF")


def test_web_variants_and_social_card_are_registered() -> None:
    assert MOBILE_FIGURE_STEMS == [
        "representation_reconstruction_mobile",
        "representation_structure_mobile",
        "representation_factorization_mobile",
    ]
    assert SOCIAL_CARD_NAME == "addressability_social_card.png"
    assert SOCIAL_CARD_NAME in FIGURE_NAMES
    assert all(f"{stem}.png" in FIGURE_NAMES for stem in MOBILE_FIGURE_STEMS)


def test_only_input_is_the_committed_receipt() -> None:
    assert RECEIPT_PATH == REPO_ROOT / "paper_artifacts" / "c14_representation_receipt.json"
    assert RECEIPT_PATH.exists()
    receipt = load_receipt(RECEIPT_PATH)
    assert receipt["kind"] == EXPECTED_KIND


def test_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "run1"
    second = tmp_path / "run2"
    assert main(["--out-dir", str(first)]) == 0
    assert main(["--out-dir", str(second)]) == 0
    first_hashes = {p.name: _sha256(p) for p in first.glob("*.*")}
    second_hashes = {p.name: _sha256(p) for p in second.glob("*.*")}
    assert first_hashes == second_hashes


def test_no_hand_entered_headline_values_in_source() -> None:
    """Every figure-bearing value must be parsed from the receipt, not hardcoded."""
    source = (REPO_ROOT / "experiments" / "plot_representation_figures.py").read_text(
        encoding="utf-8"
    )
    forbidden = [
        "0.9289", "0.4839", "0.4228", "0.9149", "0.4252", "0.5061", "0.0140",
        "0.9060", "0.9175", "0.0135", "0.8787", "0.8760", "0.8473", "0.1130",
        "0.6808", "0.8915", "0.2107", "0.8205", "0.0710", "66.3", "96.75", "96.92",
        "0.9326", "0.9304", "0.9223", "0.0243", "0.0138",
    ]
    hits = [value for value in forbidden if value in source]
    assert not hits, f"hardcoded scientific values in generator source: {hits}"


def test_parsed_values_cover_all_panels() -> None:
    data = parse_data(load_receipt(RECEIPT_PATH))
    assert len(data["bars"]) == 7
    assert len(data["paired"]) == 7
    assert len(data["specificity"]) == 3
    assert len(data["compression"]) == 4
    assert data["bars"][0][0] == "Raw-activation k=8"
    assert len(data["action_folds"]) == 5
    assert len(data["additive_folds"]) == 5
    assert len(data["constrained_folds"]) == 5
    assert 0 < data["rank_var_min"] <= data["rank_var_max"] < 1
    assert data["additive_cos"] > data["constrained_cos"] > data["action_cos"]
