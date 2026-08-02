"""Drift gate for docs/results_registry.yaml.

Fails if the registry is structurally invalid, or if it cites a produced_by script / main_module
/ test that does not exist, or an artifact whose artifact_state disagrees with git, or if the
generated claim block in README.md has drifted from the registry. This drift gate is what keeps
the registry load-bearing instead of decorative: every claim in it must point at artifacts and
code that actually exist in the repo.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from geoprobe.paths import REPO_ROOT
from geoprobe.registry import (
    MARKER_BEGIN,
    MARKER_END,
    inject_generated_block,
    load_registry,
    render_claim_table,
    validate_registry,
)

REGISTRY_PATH = REPO_ROOT / "docs" / "results_registry.yaml"


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_registry(REGISTRY_PATH)


def _git_tracked(path: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", path],
        capture_output=True, text=True,
    )
    return out.returncode == 0


def _git_ignored(path: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", path.rstrip("/")],
        capture_output=True, text=True,
    )
    return out.returncode == 0


# --- the shipped registry ---

def test_shipped_registry_is_structurally_valid(registry):
    assert validate_registry(registry) == []


requires_git = pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists(),
    reason="git-provenance checks require a git checkout",
)


@requires_git
def test_produced_by_scripts_are_tracked(registry):
    for res in registry["results"]:
        assert _git_tracked(res["produced_by"]), f"{res['id']}: produced_by not tracked: {res['produced_by']}"


def test_branch_main_modules_are_tracked(registry):
    for name, spec in registry["branches"].items():
        module = spec.get("main_module")
        if module:
            assert _git_tracked(module), f"branch {name}: main_module not tracked: {module}"


def test_claim_tests_exist(registry):
    for claim in registry["claims"]:
        for test_path in claim.get("tests") or []:
            assert (REPO_ROOT / test_path).exists(), f"claim {claim['id']}: cited test missing: {test_path}"


@requires_git
def test_artifact_state_matches_git(registry):
    for res in registry["results"]:
        artifact, state = res["artifact"], res["artifact_state"]
        if state == "tracked":
            assert _git_tracked(artifact), f"{res['id']}: marked tracked but not in git: {artifact}"
        elif state == "remote_gitignored":
            assert _git_ignored(artifact), f"{res['id']}: marked remote_gitignored but not ignored: {artifact}"
        # local_only: no git assertion by design.


def test_public_registry_is_one_receipt_per_claim(registry):
    """The paper registry is a citation closure, not a historical experiment ledger."""
    claims = {claim["id"]: claim for claim in registry["claims"]}
    results = {result["id"]: result for result in registry["results"]}
    scoped = {
        claim_id
        for role in ("primary", "supporting", "negative", "descriptive")
        for claim_id in registry["paper_scope"].get(role, [])
    }

    assert set(claims) == scoped
    assert len(claims) == len(results) == 9
    for claim_id, claim in claims.items():
        assert len(claim["evidence"]) == 1
        result = results[claim["evidence"][0]]
        assert result["claim"] == claim_id
        assert result["artifact_state"] == "tracked"
        assert result["artifact"].startswith("paper_artifacts/")

    assert all(result.get("claim") in claims for result in results.values())


def test_render_claim_table_covers_every_claim(registry):
    table = render_claim_table(registry)
    # The claim table only renders primary/supporting/negative scoped claims.
    # Descriptive companions (unregistered_descriptive) are intentionally excluded.
    scoped_claim_ids: set[str] = set()
    for role in ("primary", "supporting", "negative"):
        scoped_claim_ids.update(registry.get("paper_scope", {}).get(role, []))
    for claim in registry["claims"]:
        if claim["id"] in scoped_claim_ids:
            assert claim["id"] in table


def test_readme_claim_block_is_in_sync(registry):
    doc = (REPO_ROOT / "README.md").read_text()
    synced = inject_generated_block(doc, render_claim_table(registry))
    assert synced == doc, "README.md claim block is stale; run experiments/report_results_registry.py"


def test_report_cli_check_passes(monkeypatch):
    import experiments.report_results_registry as cli

    monkeypatch.setattr(sys, "argv", ["report_results_registry.py", "--check"])
    assert cli.main() == 0


def test_report_cli_check_fails_on_drift(monkeypatch, tmp_path):
    """--check must exit non-zero when the target's block no longer matches the registry."""
    import experiments.report_results_registry as cli

    stale = tmp_path / "README.md"
    stale.write_text(f"# stale\n\n{MARKER_BEGIN}\n| Claim | Status |\n{MARKER_END}\n")
    monkeypatch.setattr(
        sys, "argv", ["report_results_registry.py", "--check", "--out", str(stale)]
    )
    assert cli.main() == 1


# --- validator unit tests (synthetic bad registries must be caught) ---

def _minimal() -> dict:
    return {
        "schema_version": 1,
        "branches": {"b1": {"role": "main"}},
        "paper_scope": {"primary": ["C1"], "supporting": [], "negative": []},
        "results": [{
            "id": "r1", "branch": "b1", "status": "done",
            "produced_by": "experiments/x.py", "artifact": "a", "artifact_state": "tracked",
            "claim": "C1",
        }],
        "claims": [{
            "id": "C1",
            "statement": "s",
            "status": "supported",
            "registration_tier": "prospective",
            "boundary": "b",
        }],
    }


def test_valid_minimal_registry_has_no_errors():
    assert validate_registry(_minimal()) == []


def test_bad_schema_version_is_caught():
    reg = _minimal()
    reg["schema_version"] = 2
    assert any("schema_version" in e for e in validate_registry(reg))


def test_unknown_branch_role_is_caught():
    reg = _minimal()
    reg["branches"]["b1"]["role"] = "bogus"
    assert any("role" in e for e in validate_registry(reg))


def test_result_referencing_unknown_branch_is_caught():
    reg = _minimal()
    reg["results"][0]["branch"] = "ghost"
    assert any("branch" in e and "ghost" in e for e in validate_registry(reg))


def test_result_referencing_unknown_claim_is_caught():
    reg = _minimal()
    reg["results"][0]["claim"] = "C9"
    assert any("claim" in e and "C9" in e for e in validate_registry(reg))


def test_evidence_referencing_unknown_result_is_caught():
    reg = _minimal()
    reg["claims"][0]["evidence"] = ["nope"]
    assert any("evidence" in e for e in validate_registry(reg))


def test_duplicate_result_id_is_caught():
    reg = _minimal()
    reg["results"].append(dict(reg["results"][0]))
    assert any("duplicate result id" in e for e in validate_registry(reg))


def test_bad_artifact_state_is_caught():
    reg = _minimal()
    reg["results"][0]["artifact_state"] = "somewhere"
    assert any("artifact_state" in e for e in validate_registry(reg))


def test_bad_registration_tier_is_caught():
    reg = _minimal()
    reg["claims"][0]["registration_tier"] = "everything_was_preregistered"
    assert any("registration_tier" in e for e in validate_registry(reg))


def test_bad_paper_scope_is_caught():
    reg = _minimal()
    reg["paper_scope"] = {
        "primary": ["C1"],
        "supporting": ["C1"],
        "negative": ["missing"],
    }
    errors = validate_registry(reg)
    assert any("unique" in e for e in errors)
    assert any("unknown claim" in e for e in errors)


# ---------------------------------------------------------------------------
# descriptive_companions validation (M4)
# ---------------------------------------------------------------------------

COMPANION_RESULT_IDS = ["c1_matched_control_audit", "c5_natural_prose_control_receipt",
                         "c9_pressure_commitment_receipt", "c11_precommitment_warning_receipt",
                         "c13_gauge_control_receipt"]


def test_companion_status_is_unregistered_descriptive(registry):
    """Every descriptive companion must have status unregistered_descriptive."""
    results_by_id = {r["id"]: r for r in registry["results"]}
    for rid in COMPANION_RESULT_IDS:
        result = results_by_id.get(rid)
        assert result is not None, f"expected result {rid} for companions"
        companions = result.get("descriptive_companions")
        assert companions is not None, f"result {rid} missing descriptive_companions"
        assert len(companions) >= 1, f"result {rid} has empty descriptive_companions"
        for comp in companions:
            assert comp.get("status") == "unregistered_descriptive", (
                f"{comp.get('id')}: expected status unregistered_descriptive, "
                f"got {comp.get('status')!r}"
            )


@requires_git
def test_companion_producer_is_tracked(registry):
    """Every companion's produced_by script must be git-tracked."""
    results_by_id = {r["id"]: r for r in registry["results"]}
    for rid in COMPANION_RESULT_IDS:
        result = results_by_id[rid]
        for comp in result.get("descriptive_companions", []):
            path = comp.get("produced_by")
            assert path is not None, f"{comp.get('id')}: missing produced_by"
            assert _git_tracked(path), (
                f"{comp.get('id')}: produced_by not tracked: {path}"
            )


def test_companion_receipt_exists(registry):
    """Every companion's receipt path must exist on disk."""
    results_by_id = {r["id"]: r for r in registry["results"]}
    for rid in COMPANION_RESULT_IDS:
        result = results_by_id[rid]
        for comp in result.get("descriptive_companions", []):
            path = comp.get("receipt")
            assert path is not None, f"{comp.get('id')}: missing receipt"
            assert (REPO_ROOT / path).exists(), (
                f"{comp.get('id')}: receipt not found: {path}"
            )


def test_companion_boundary_is_nonempty(registry):
    """Every companion must have a non-empty boundary string."""
    results_by_id = {r["id"]: r for r in registry["results"]}
    for rid in COMPANION_RESULT_IDS:
        result = results_by_id[rid]
        for comp in result.get("descriptive_companions", []):
            boundary = comp.get("boundary")
            assert boundary is not None, f"{comp.get('id')}: missing boundary"
            assert isinstance(boundary, str), f"{comp.get('id')}: boundary must be a string"
            assert boundary.strip(), f"{comp.get('id')}: boundary is empty"


def test_companion_has_id_and_statement(registry):
    """Every companion must have an id and a non-empty statement."""
    results_by_id = {r["id"]: r for r in registry["results"]}
    for rid in COMPANION_RESULT_IDS:
        result = results_by_id[rid]
        for comp in result.get("descriptive_companions", []):
            assert comp.get("id"), f"companion in {rid} missing id"
            assert comp.get("statement"), f"{comp.get('id')}: missing statement"
            assert comp.get("statement").strip(), f"{comp.get('id')}: empty statement"


def test_companions_do_not_affect_registered_claim_count(registry):
    """The nine registered claims/results must remain unchanged (C14_DESCRIPTIVE added)."""
    claims = {c["id"] for c in registry["claims"]}
    results = {r["id"] for r in registry["results"]}
    assert len(claims) == 9
    assert len(results) == 9
    for role in ("primary", "supporting", "negative"):
        assert len(registry["paper_scope"][role]) == len(
            set(registry["paper_scope"][role])
        )
