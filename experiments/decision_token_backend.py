"""Compatibility shim — moved to geoprobe.models.decision_backend (Phase 2). Import from there."""
from geoprobe.models.decision_backend import (
    BACKENDS,
    GENERATION_PROTOCOL_HF_BATCHED,
    GENERATION_PROTOCOL_SEQUENTIAL,
    build_decision_prefix_cache,
    choose_status_then_complete,
    choose_status_then_complete_batch,
    decision_margin,
    decision_margin_with_cache,
    decision_margins,
    generation_protocol_name,
    load_decision_backend_model,
    public_model_meta,
    validate_generation_batch_config,
)

__all__ = [
    "BACKENDS",
    "GENERATION_PROTOCOL_HF_BATCHED",
    "GENERATION_PROTOCOL_SEQUENTIAL",
    "build_decision_prefix_cache",
    "choose_status_then_complete",
    "choose_status_then_complete_batch",
    "decision_margin",
    "decision_margin_with_cache",
    "decision_margins",
    "generation_protocol_name",
    "load_decision_backend_model",
    "public_model_meta",
    "validate_generation_batch_config",
]
