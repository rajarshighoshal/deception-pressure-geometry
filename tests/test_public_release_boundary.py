"""Keep the citable tree free of private execution and workstation details."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = ("README.md", "docs", "configs", "experiments", "src")
TEXT_SUFFIXES = {"", ".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt"}
EXCLUDED = {Path("experiments/verify_paper_artifacts.py")}
FORBIDDEN = (
    ".private/",
    "runpod_results",
    "/users/",
    "/workspace/",
    "approval.json",
    "request_sha256=",
    "run_contracts_sha256=",
    "deadman",
    "podctl",
    "claude.md",
    "agents.md",
)


def _public_text_files() -> list[Path]:
    paths: list[Path] = []
    for root_name in PUBLIC_ROOTS:
        root = REPO_ROOT / root_name
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(REPO_ROOT)
            if relative in EXCLUDED or "__pycache__" in relative.parts:
                continue
            paths.append(path)
    return sorted(paths)


def test_public_scientific_tree_has_no_private_execution_paths() -> None:
    violations: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in FORBIDDEN:
            if marker in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {marker}")
    assert violations == []
