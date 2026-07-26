"""Run provenance: git commit hash, dirty flag, and optional file hashes."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from geoprobe.io import file_sha256
from geoprobe.paths import REPO_ROOT

def git_provenance(extra_paths: list[str | Path] | None = None) -> dict:
    """Best-effort commit hash + dirty flag + optional file hashes for run provenance."""
    try:
        reviewed = os.environ.get("REVIEWED_GIT_COMMIT", "")
        if reviewed and re.fullmatch(r"[0-9a-f]{40}", reviewed) is None:
            raise ValueError("REVIEWED_GIT_COMMIT is invalid")
        hashes = {str(Path(path)): file_sha256(Path(path)) for path in extra_paths or []}
        if reviewed:
            return {
                "git_hash": reviewed[:7],
                "git_dirty": False,
                "file_sha256": hashes,
            }
        repo = str(REPO_ROOT)
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        return {"git_hash": commit or None, "git_dirty": dirty, "file_sha256": hashes}
    except Exception:
        return {"git_hash": None, "git_dirty": None, "file_sha256": {}}
