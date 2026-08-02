"""Keep the public longform article self-contained and publication-safe."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE = REPO_ROOT / "docs" / "index.md"
FIGURES = {
    "representation_reconstruction.png",
    "representation_structure.png",
    "representation_factorization.png",
}


def test_article_uses_the_shipped_figures_by_relative_path() -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    image_paths = re.findall(r"!\[[^\]]*\]\(figures/([^)]+\.png)\)", text)

    assert set(image_paths) == FIGURES
    assert len(image_paths) == len(FIGURES)
    assert "raw.githubusercontent.com" not in text


def test_article_preserves_the_human_authorship_boundary() -> None:
    text = ARTICLE.read_text(encoding="utf-8")

    assert "The author wrote the post" in text
    assert "owns the research design, methodology, claims, and prose" in text
    assert "AI-written" not in text
    assert "AI-coauthored" not in text


def test_article_contains_no_private_or_editor_capability_links() -> None:
    text = ARTICLE.read_text(encoding="utf-8").lower()

    forbidden = (
        ".private/",
        "/users/",
        "/workspace/",
        "editpost?",
        "runpod_results",
        "request_sha256=",
        "run_contracts_sha256=",
    )
    assert [marker for marker in forbidden if marker in text] == []


def test_pages_configuration_targets_the_project_site() -> None:
    config = (REPO_ROOT / "docs" / "_config.yml").read_text(encoding="utf-8")

    assert 'url: "https://rajarshighoshal.github.io"' in config
    assert 'baseurl: "/deception-pressure-geometry"' in config
