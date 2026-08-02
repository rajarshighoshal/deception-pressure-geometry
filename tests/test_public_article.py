"""Keep the public longform article self-contained, accurate, and accessible."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTICLE = REPO_ROOT / "docs" / "index.md"
LAYOUT = REPO_ROOT / "docs" / "_layouts" / "article.html"
DESKTOP_FIGURES = {
    "representation_reconstruction_blog.png",
    "representation_structure_blog.png",
    "representation_factorization_blog.png",
}
MOBILE_FIGURES = {name.replace("_blog.png", "_mobile_blog.png") for name in DESKTOP_FIGURES}


def test_article_uses_responsive_shipped_figures_by_relative_path() -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    desktop = set(re.findall(r'<img src="figures/([^"/]+\.png)"', text))
    mobile = set(re.findall(r'srcset="figures/([^"/]+\.png)"', text))

    assert desktop == DESKTOP_FIGURES
    assert mobile == MOBILE_FIGURES
    assert text.count('<figure class="evidence-figure">') == 3
    assert text.count("<figcaption") == 3
    assert "raw.githubusercontent.com" not in text
    for name in desktop | mobile:
        assert (REPO_ROOT / "docs" / "figures" / name).is_file()


def test_article_has_meaningful_figure_alt_text_and_table_captions() -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    alt_texts = re.findall(r'<img[^>]+alt="([^"]+)"', text)

    assert len(alt_texts) == 3
    assert all(len(alt) >= 100 for alt in alt_texts)
    assert all("chart" not in alt.lower() for alt in alt_texts)
    assert text.count("<caption") == 3
    assert text.count('class="responsive-table"') == 3
    assert text.count('class="visual-header') >= 7
    assert text.count('class="visual-caption"') >= 6
    assert text.count('class="technical-ledger"') == 2
    assert text.count('data-label="') >= 20



def test_visual_captions_are_integrated_into_result_cards() -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    css = (REPO_ROOT / "docs" / "assets" / "article.css").read_text(encoding="utf-8")

    assert text.count('caption class="sr-only"') == 3
    assert '<caption>Held-out-family' not in text
    assert '<caption>Specificity margins' not in text
    assert 'class="visual-card visual-card--boundary-table"' in text
    assert '.visual-header' in css
    assert '.visual-caption' in css
    assert '--paper: #e9e2d5' in css
    assert '--blue: #1f5e8c' in css
    assert '--amber: #b0722a' in css

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


def test_article_preserves_the_scientific_claim_boundaries() -> None:
    text = ARTICLE.read_text(encoding="utf-8")
    lower = text.lower()

    forbidden_phrases = (
        "two identical conversations",
        "where the state would move",
        "the deception-specific residue is real — it is positive on both metrics",
        "the source side is not as compressible",
    )
    assert [phrase for phrase in forbidden_phrases if phrase in lower] == []

    required = (
        "offline reconstruction only",
        "developmental bank",
        "56/60 against a required 57/60",
        "retrospective unregistered descriptive",
        "post-evidence registered descriptive",
        "privileged truth-aware explanatory baseline",
        "within the typed-graph estimator",
        "normalized-error interval crossing zero",
        "desired status",
        "destination program",
        "not a target-free",
        "has not been injected",
    )
    assert [phrase for phrase in required if phrase.lower() not in lower] == []


def test_article_has_skimmer_and_method_visuals() -> None:
    text = ARTICLE.read_text(encoding="utf-8")

    assert 'class="lead-dashboard"' in text
    assert 'class="lead-finding"' in text
    assert text.count('class="insight-tile ') == 2
    assert 'class="method-map"' in text
    assert 'class="visual-card visual-card--ranking"' in text
    assert 'class="visual-card visual-card--specificity"' in text
    assert 'class="visual-card visual-card--structure"' in text
    assert 'class="term-strip"' in text
    assert 'class="evidence-ladder"' in text
    assert "0.9326" in text
    assert "0.4839" in text


def test_article_metadata_uses_current_assets_and_dates() -> None:
    article = ARTICLE.read_text(encoding="utf-8")
    layout = LAYOUT.read_text(encoding="utf-8")

    assert 'published: "July 2026"' in article
    assert 'updated: "August 2, 2026"' in article
    assert 'og_image: "figures/addressability_social_card.png"' in article
    assert "page.published" in layout
    assert "page.updated" in layout
    assert "page.og_image" in layout
    assert "page.og_image_alt" in layout
    assert "structured_action_control_audit.png" not in layout
    assert "{{ page.byline }}" in layout


def test_pages_configuration_targets_the_project_site() -> None:
    config = (REPO_ROOT / "docs" / "_config.yml").read_text(encoding="utf-8")

    assert 'url: "https://rajarshighoshal.github.io"' in config
    assert 'baseurl: "/deception-pressure-geometry"' in config
