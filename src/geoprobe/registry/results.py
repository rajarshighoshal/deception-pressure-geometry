"""Load, validate, and render the powered150 result registry (docs/results_registry.yaml).

`validate_registry` is pure and filesystem-free (vocab + cross-reference integrity) so it is
cheap to unit-test; the on-disk/git checks (artifact tracked vs ignored, cited tests exist)
live in tests/test_results_registry.py. That drift gate is what keeps the registry
load-bearing instead of decorative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
BRANCH_ROLES = {"main", "comparator", "diagnostic", "negative_baseline", "superseded"}
RESULT_STATUSES = {"done", "pending", "superseded", "failed"}
CLAIM_STATUSES = {
    "supported",
    "false_pending",
    "untested",
    "pending",
    "refuted",
    # the third verdict: the instrument could not resolve the effect, so neither
    # support nor refutation is earned
    "not_found_under_instrument",
    # a completed run whose producer is not shipped: recorded, never citable
    "unregistered_descriptive",
}
REGISTRATION_TIERS = {
    # The claim and decision rule were fixed before the evidence that decides it.
    "prospective",
    # The analysis was frozen before outcomes were opened, but was explicitly exploratory.
    "frozen_exploratory",
    # The method and comparator were registered after the evaluation bank already existed.
    "post_hoc_registered",
    # The public claim combines or summarizes analyses after their results were available.
    "retrospective_synthesis",
    # A completed descriptive record with no prospective claim status.
    "unregistered_descriptive",
}
ARTIFACT_STATES = {"tracked", "local_only", "remote_gitignored"}

_REQUIRED_RESULT = ("id", "branch", "status", "produced_by", "artifact", "artifact_state")
_REQUIRED_CLAIM = ("id", "statement", "status", "registration_tier", "boundary")


def load_registry(path: str | Path) -> dict[str, Any]:
    """Parse the registry YAML into a plain dict."""
    return yaml.safe_load(Path(path).read_text())


def validate_registry(reg: dict[str, Any]) -> list[str]:
    """Return a list of structural errors (empty == valid). Pure: no filesystem or git access."""
    errors: list[str] = []
    if not isinstance(reg, dict):
        return ["registry root is not a mapping"]

    if reg.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}, got {reg.get('schema_version')!r}")

    branches = reg.get("branches") or {}
    results = reg.get("results") or []
    claims = reg.get("claims") or []

    branch_ids = set(branches)
    for name, spec in branches.items():
        role = (spec or {}).get("role")
        if role not in BRANCH_ROLES:
            errors.append(f"branch {name!r}: role {role!r} not in {sorted(BRANCH_ROLES)}")

    claim_ids: set[str] = set()
    for claim in claims:
        cid = claim.get("id")
        for field in _REQUIRED_CLAIM:
            if not claim.get(field):
                errors.append(f"claim {cid!r}: missing required field {field!r}")
        if cid in claim_ids:
            errors.append(f"duplicate claim id {cid!r}")
        claim_ids.add(cid)
        if claim.get("status") not in CLAIM_STATUSES:
            errors.append(f"claim {cid!r}: status {claim.get('status')!r} not in {sorted(CLAIM_STATUSES)}")
        if claim.get("registration_tier") not in REGISTRATION_TIERS:
            errors.append(
                f"claim {cid!r}: registration_tier {claim.get('registration_tier')!r} "
                f"not in {sorted(REGISTRATION_TIERS)}"
            )

    paper_scope = reg.get("paper_scope") or {}
    scoped_claims: list[str] = []
    for role in ("primary", "supporting", "negative"):
        values = paper_scope.get(role)
        if not isinstance(values, list):
            errors.append(f"paper_scope.{role} must be a list")
            continue
        scoped_claims.extend(values)
    if not paper_scope.get("primary"):
        errors.append("paper_scope.primary must name at least one claim")
    if len(scoped_claims) != len(set(scoped_claims)):
        errors.append("paper_scope claim ids must be unique across roles")
    for cid in scoped_claims:
        if cid not in claim_ids:
            errors.append(f"paper_scope references unknown claim {cid!r}")

    result_ids: set[str] = set()
    for res in results:
        rid = res.get("id")
        for field in _REQUIRED_RESULT:
            if not res.get(field):
                errors.append(f"result {rid!r}: missing required field {field!r}")
        if rid in result_ids:
            errors.append(f"duplicate result id {rid!r}")
        result_ids.add(rid)
        if res.get("branch") not in branch_ids:
            errors.append(f"result {rid!r}: branch {res.get('branch')!r} not in branches")
        if res.get("status") not in RESULT_STATUSES:
            errors.append(f"result {rid!r}: status {res.get('status')!r} not in {sorted(RESULT_STATUSES)}")
        if res.get("artifact_state") not in ARTIFACT_STATES:
            errors.append(
                f"result {rid!r}: artifact_state {res.get('artifact_state')!r} not in {sorted(ARTIFACT_STATES)}"
            )
        if res.get("claim") is not None and res["claim"] not in claim_ids:
            errors.append(f"result {rid!r}: claim {res['claim']!r} not in claims")

    # Claim -> evidence must reference real results.
    for claim in claims:
        for ev in claim.get("evidence") or []:
            if ev not in result_ids:
                errors.append(f"claim {claim.get('id')!r}: evidence {ev!r} not a known result id")

    return errors


def iter_results(reg: dict[str, Any], *, branch: str | None = None, claim: str | None = None):
    """Yield results, optionally filtered by branch or claim id."""
    for res in reg.get("results") or []:
        if branch is not None and res.get("branch") != branch:
            continue
        if claim is not None and res.get("claim") != claim:
            continue
        yield res


def render_claim_table(reg: dict[str, Any]) -> str:
    """Render a compact paper-claim map for README; full boundaries stay in the registry."""
    lines = [
        "| Claim | Evidence status | Registration tier | Registered summary |",
        "|---|---|---|---|",
    ]
    claims = {claim.get("id"): claim for claim in reg.get("claims") or []}
    paper_scope = reg.get("paper_scope") or {}
    ordered_ids = [
        claim_id
        for role in ("primary", "supporting", "negative")
        for claim_id in paper_scope.get(role, [])
    ]
    for claim_id in ordered_ids:
        claim = claims[claim_id]
        statement = " ".join(str(claim.get("statement", "")).split())
        lines.append(
            f"| {claim.get('id')} | {claim.get('status')} | "
            f"{claim.get('registration_tier')} | {statement} |"
        )
    return "\n".join(lines)


MARKER_BEGIN = "<!-- BEGIN GENERATED: claims -->"
MARKER_END = "<!-- END GENERATED: claims -->"


def inject_generated_block(doc: str, block: str) -> str:
    """Replace the text between the generated markers with ``block`` (idempotent).

    Raises ValueError if the markers are missing, so a hand-edited doc cannot silently drop the
    generated section.
    """
    if MARKER_BEGIN not in doc or MARKER_END not in doc:
        raise ValueError(f"generated markers {MARKER_BEGIN!r}/{MARKER_END!r} not found in document")
    pre = doc[: doc.index(MARKER_BEGIN) + len(MARKER_BEGIN)]
    post = doc[doc.index(MARKER_END):]
    return f"{pre}\n{block}\n{post}"
