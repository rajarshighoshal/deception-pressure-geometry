"""Import smoke test: every module in experiments/ must import cleanly.

Catches dead imports in experiment CLIs (e.g. a geoprobe module deleted while an
experiments script still imports it) without running any experiment. Discovery is
static (glob over experiments/*.py), imports are in-process, and the whole file is
deterministic and offline: a full cold run of all modules takes ~17 s, dominated by a
one-time matplotlib font-cache build triggered by plot_public_results.

Skip-list policy: only modules whose import fails for reasons outside this repo's
control (or pending restoration work owned outside tests/experiments) may be listed
below, each with an explicit reason. Anything else that fails this test is a bug.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

# Modules whose import currently fails for known, externally-owned reasons. Keep
# this list empty: anything that fails this test is a bug to fix, not to skip.
SKIP_MODULES: dict[str, str] = {}

MODULE_NAMES = sorted(
    path.stem for path in EXPERIMENTS_DIR.glob("*.py") if path.stem != "__init__"
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_experiments_module_imports(module_name: str) -> None:
    skip_reason = SKIP_MODULES.get(module_name)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    importlib.import_module(f"experiments.{module_name}")
