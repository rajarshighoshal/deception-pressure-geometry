"""Machine-readable powered150 result registry: load, validate, render."""
from geoprobe.registry.results import (
    ARTIFACT_STATES,
    BRANCH_ROLES,
    CLAIM_STATUSES,
    MARKER_BEGIN,
    MARKER_END,
    REGISTRATION_TIERS,
    RESULT_STATUSES,
    SCHEMA_VERSION,
    inject_generated_block,
    iter_results,
    load_registry,
    render_claim_table,
    validate_registry,
)

__all__ = [
    "ARTIFACT_STATES",
    "BRANCH_ROLES",
    "CLAIM_STATUSES",
    "MARKER_BEGIN",
    "MARKER_END",
    "REGISTRATION_TIERS",
    "RESULT_STATUSES",
    "SCHEMA_VERSION",
    "inject_generated_block",
    "iter_results",
    "load_registry",
    "render_claim_table",
    "validate_registry",
]
