"""Consistency gate between README.md and the registry it advertises.

The README's "In brief" bullets cite claim IDs (C9, C10, ...) and point readers at
docs/results_registry.yaml as the source of those claims, with the claim table itself
generated into the README. This test makes that pointer load-bearing: every claim ID
cited in the README must exist in the registry, and every local path the README links
must exist in the tree.
"""

from __future__ import annotations

import re

from geoprobe.paths import REPO_ROOT
from geoprobe.registry import load_registry

README = REPO_ROOT / "README.md"
REGISTRY_PATH = REPO_ROOT / "docs" / "results_registry.yaml"

CLAIM_ID_RE = re.compile(r"\bC(\d{1,2})\b")
# Markdown links to local files: [text](path) or bare `path` mentions worth checking.
LOCAL_LINK_RE = re.compile(r"\]\(([^)]+)\)")
BACKTICK_PATH_RE = re.compile(r"`((?:docs|pilot|experiments|src|tests|configs|scripts)/[^`]+)`")


def _readme() -> str:
    return README.read_text()


def test_claim_ids_cited_in_readme_exist_in_registry() -> None:
    registry = load_registry(REGISTRY_PATH)
    known = {claim["id"] for claim in registry["claims"]}
    cited = {f"C{n}" for n in CLAIM_ID_RE.findall(_readme())}
    unknown = cited - known
    assert not unknown, f"README cites claim IDs missing from the registry: {sorted(unknown)}"


def test_readme_markdown_links_resolve() -> None:
    missing = []
    for target in LOCAL_LINK_RE.findall(_readme()):
        if "://" in target or target.startswith("#"):
            continue
        if not (REPO_ROOT / target).exists():
            missing.append(target)
    assert not missing, f"README links that do not resolve: {missing}"


def test_readme_backticked_paths_exist() -> None:
    missing = []
    for target in BACKTICK_PATH_RE.findall(_readme()):
        path = target.rstrip("/")
        if not (REPO_ROOT / path).exists():
            missing.append(target)
    assert not missing, f"README backticked paths missing from the tree: {missing}"
