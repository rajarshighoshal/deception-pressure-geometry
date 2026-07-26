"""Shared file-hashing / IO helpers (canonical home; stdlib-only at import time)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    """Write text atomically: temp file in the same directory, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, human-readable JSON document atomically."""
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays (recursively) into plain JSON-serializable values."""
    import numpy as np  # lazy: keeps this module stdlib-only at import time

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
