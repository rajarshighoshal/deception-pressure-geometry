from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from experiments.prepare_relational_structured_action import (
    ENGINEERING_SCENARIO_COUNT,
    SENTINEL_PER_STATUS_CAVEAT_STRATUM,
    _select_engineering_scenarios,
    _select_neutral_sentinels,
)
from geoprobe.data.relational_structured_action import (
    select_relational_structured_action_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "data/raw/deception_intent/graded_ks_150fam_l7_scenarios.jsonl"

# SCENARIOS is a raw data artifact (gitignored); skip the tests that need it rather
# than failing on a fresh checkout.
requires_scenarios_source = pytest.mark.skipif(
    not SCENARIOS.exists(),
    reason="requires local data/raw/deception_intent scenarios file (gitignored)",
)


def _specs() -> list[dict]:
    scenarios = [
        json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()
    ]
    return select_relational_structured_action_specs(scenarios)


@requires_scenarios_source
def test_neutral_sentinels_are_balanced_preselected_and_family_distinct() -> None:
    sentinels = _select_neutral_sentinels(_specs())
    assert len(sentinels) == 4 * SENTINEL_PER_STATUS_CAVEAT_STRATUM
    assert len({str(row["scenario_id"]) for row in sentinels}) == len(sentinels)
    assert len({str(row["family"]) for row in sentinels}) == len(sentinels)
    assert Counter(
        (str(row["true_status"]), bool(row["caveat_present"]))
        for row in sentinels
    ) == {
        ("PASS", True): SENTINEL_PER_STATUS_CAVEAT_STRATUM,
        ("PASS", False): SENTINEL_PER_STATUS_CAVEAT_STRATUM,
        ("FAIL", True): SENTINEL_PER_STATUS_CAVEAT_STRATUM,
        ("FAIL", False): SENTINEL_PER_STATUS_CAVEAT_STRATUM,
    }
    assert all(
        row["intervention_program"] == "NN" and row["sample_index"] == 0
        for row in sentinels
    )


@requires_scenarios_source
def test_engineering_checkpoint_is_two_complete_scenario_blocks() -> None:
    specs = _specs()
    scenario_ids = _select_engineering_scenarios(specs, 20260711)
    assert len(scenario_ids) == ENGINEERING_SCENARIO_COUNT
    selected = [
        row for row in specs if str(row["scenario_id"]) in set(scenario_ids)
    ]
    assert len(selected) == 20
    assert Counter(str(row["scenario_id"]) for row in selected) == {
        scenario_id: 10 for scenario_id in scenario_ids
    }
