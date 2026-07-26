"""Decision-token primitives: the PASS/FAIL report prefix, prefix ids, and margin scoring.

Extracted from the activation_control_tomography experiment (Phase 2); that module re-exports
these, so its importers keep working. These are model-tokenization/scoring primitives the
decision backend needs, independent of the tomography analysis code.
"""
from __future__ import annotations

import numpy as np

from geoprobe.models.tokenization import chat_token_ids

REPORT_PREFIX = '{\n  "reported_status": "'


def decision_tokens(tokenizer) -> tuple[int, int]:
    """First token that distinguishes ...PASS from ...FAIL in the report-JSON context."""
    p = tokenizer.encode(REPORT_PREFIX + "PASS")
    f = tokenizer.encode(REPORT_PREFIX + "FAIL")
    i = 0
    while i < min(len(p), len(f)) and p[i] == f[i]:
        i += 1
    return p[i], f[i]


def decision_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Tokens up to the point where the model must emit PASS or FAIL."""
    prompt = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
    ids = chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    return list(ids) + list(tokenizer.encode(REPORT_PREFIX, add_special_tokens=False))


def margin(model, token_ids, pass_id: int, fail_id: int, steering=None) -> float:
    from geoprobe.models.mlx_capture import _forward_logits_with_steering
    import mlx.core as mx
    logits = _forward_logits_with_steering(model, token_ids, steering)
    last = logits[0, -1, :]
    mx.eval(last)
    arr = np.array(last.astype(mx.float32))
    return float(arr[pass_id] - arr[fail_id])


__all__ = ["REPORT_PREFIX", "decision_tokens", "decision_prefix_ids", "margin"]
