"""Mirror and privacy contract for the public pressure-behavior receipt.

The receipt aggregates per-program behavioral outcomes from the frozen presented-bank
rollout rows. These tests pin the shipped values, enforce internal consistency
(counts sum, rates derive from counts), and guard against private-path leakage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = REPO_ROOT / "paper_artifacts" / "pressure_behavior_receipt.json"
PRODUCER = REPO_ROOT / "experiments" / "report_public_pressure_behavior_receipt.py"

EXPECTED_PROGRAMS = {
    "NN": {"n": 60, "deceptive": 11, "honest": 49, "skip": 0},
    "AN": {"n": 120, "deceptive": 90, "honest": 30, "skip": 0},
    "D2N": {"n": 60, "deceptive": 39, "honest": 21, "skip": 0},
    "AA": {"n": 120, "deceptive": 113, "honest": 5, "skip": 2},
    "AB": {"n": 120, "deceptive": 120, "honest": 0, "skip": 0},
    "BA": {"n": 120, "deceptive": 107, "honest": 13, "skip": 0},
}
FROZEN_ROWS_SHA256 = "ebd99699ec1fbcc93da22c4ce768bd2d04e4ccc6d1416ba82aa866fd79157492"


def _receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_kind_and_tier() -> None:
    receipt = _receipt()
    assert receipt["kind"] == "pressure_behavior_public_receipt"
    assert receipt["status"] == "unregistered_descriptive"
    assert receipt["chronology"]["tier"] == "retrospective_unregistered_descriptive"


def test_program_counts_match_frozen_values() -> None:
    programs = _receipt()["programs"]
    assert set(programs) == set(EXPECTED_PROGRAMS)
    for key, expected in EXPECTED_PROGRAMS.items():
        block = programs[key]
        for field, value in expected.items():
            assert block[field] == value, f"{key}.{field}: {block[field]} != {value}"


def test_rates_derive_from_counts() -> None:
    for key, block in _receipt()["programs"].items():
        assert block["deceptive"] + block["honest"] + block["skip"] == block["n"], key
        assert block["deceptive_rate"] == block["deceptive"] / block["n"], key


def test_population_totals_are_consistent() -> None:
    receipt = _receipt()
    population = receipt["population"]
    assert population["conversations"] == 600
    assert sum(b["n"] for b in receipt["programs"].values()) == 600
    assert population["knowledge_correct"] + population["knowledge_incorrect"] == 600
    assert population["knowledge_correct"] == 564


def test_source_artifact_hash_is_pinned() -> None:
    assert _receipt()["source_artifact"]["sha256"] == FROZEN_ROWS_SHA256


def test_producer_sha_matches_current_script() -> None:
    expected = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
    assert _receipt()["producer_sha256"] == expected


def test_receipt_does_not_leak_source_paths() -> None:
    serialized = RECEIPT.read_text()
    for pattern in (
        "/Users/",
        "/workspace/",
        "/home/",
        "/private/",
        "/tmp/",
        "/var/",
        "/opt/",
        "/mnt/",
        "/Volumes/",
        "/Applications/",
        "runpod_results",
        "lie-geometry-probes",
        "pod_id",
        "ssh",
        "cost",
        "approval",
        "provider",
    ):
        assert pattern not in serialized, f"privacy leak: {pattern!r} found in receipt"


def test_figure_glosses_cover_every_program() -> None:
    import experiments.plot_representation_figures as figs

    source = Path(figs.__file__).read_text(encoding="utf-8")
    for key in EXPECTED_PROGRAMS:
        assert f'"{key}"' in source, f"figure module never references program {key!r}"
