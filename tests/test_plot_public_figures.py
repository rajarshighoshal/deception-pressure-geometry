from __future__ import annotations

import hashlib
import re
from pathlib import Path

from experiments.plot_public_figures import (
    FIGURE_NAMES,
    REPO_ROOT,
    RECEIPT_SPECS,
    REGISTRY_PATH,
    main,
    parse_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plot_public_figures_emit_exact_filenames(tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    assert main(["--out-dir", str(out_dir)]) == 0

    names = sorted(path.name for path in out_dir.glob("*.png"))
    assert names == sorted(FIGURE_NAMES)


def test_plot_public_figures_only_use_receipt_and_registry_inputs() -> None:
    assert REGISTRY_PATH == REPO_ROOT / "docs" / "results_registry.yaml"
    assert REGISTRY_PATH.parent.name == "docs"
    for spec in RECEIPT_SPECS.values():
        path = REPO_ROOT / spec["path"]
        assert path.exists()
        assert path.parent.name == "paper_artifacts"
        assert path.suffix == ".json"
        assert path.match("paper_artifacts/*.json")


def test_pressure_figure_uses_registered_late_compressed_contrast() -> None:
    c9 = parse_data()["c9"]

    assert c9["arms"]["scripted_late"]["point"] == 0.5
    assert c9["arms"]["adaptive_late"]["point"] == 0.71875
    assert [row["name"] for row in c9["contrasts"]["p2a"]] == [
        "Scripted P2a",
        "Adaptive P2a",
    ]


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
    expected = sorted(FIGURE_NAMES)
    readme_names = _readme_figure_names()
    assert readme_names == expected

    figures_dir = REPO_ROOT / "docs" / "figures"
    actual = sorted(p.name for p in figures_dir.glob("*.png"))
    assert actual == expected


def test_plot_public_figures_match_shipped_assets(tmp_path: Path) -> None:
    generated = tmp_path / "figs"
    assert main(["--out-dir", str(generated)]) == 0

    shipped = REPO_ROOT / "docs" / "figures"
    generated_hashes = {_digest(path): _sha256(path) for path in generated.glob("*.png")}
    shipped_hashes = {_digest(path): _sha256(path) for path in shipped.glob("*.png")}

    assert set(generated_hashes) == set(shipped_hashes) == set(FIGURE_NAMES)
    assert generated_hashes == shipped_hashes


def _digest(path: Path) -> str:
    return path.name
