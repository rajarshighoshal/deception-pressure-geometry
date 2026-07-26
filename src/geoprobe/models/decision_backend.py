"""Backend helpers for PASS/FAIL decision-token control.

The MLX and HF/PyTorch runners need the same narrow operation:

1. score the PASS-vs-FAIL decision-token margin under optional residual steering;
2. force the winning status token into the report prefix;
3. complete the report with steering turned off.

Keeping this in one small module makes the HF path testable without
changing the existing MLX experiment semantics.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from geoprobe.models.decision import REPORT_PREFIX, decision_prefix_ids
from geoprobe.models.hf_capture import (
    HfPrefixCache,
    _pad_token_id as hf_pad_token_id,
    build_prefix_cache as hf_build_prefix_cache,
    decision_margins_batch_rows as hf_decision_margins_batch_rows,
    forward_logits_batch_with_steering as hf_forward_logits_batch_with_steering,
    forward_logits_batch_with_steering_cached as hf_forward_logits_batch_with_steering_cached,
    forward_logits_with_steering as hf_forward_logits_with_steering,
    forward_margin_with_cache as hf_forward_margin_with_cache,
    generate_greedy_batch as hf_generate_greedy_batch,
    generate_greedy_with_steering as hf_generate_greedy_with_steering,
)
from geoprobe.models.interface import PipelineMeta, SteeringSpec, resolve_torch_dtype
from geoprobe.models.loader import load_hf_model
from geoprobe.models.mlx_capture import (
    generate_greedy_with_steering as mlx_generate_greedy_with_steering,
    load_mlx_model,
)


BACKENDS = {"mlx", "hf"}
GENERATION_PROTOCOL_SEQUENTIAL = "sequential_decision_token_v1"
GENERATION_PROTOCOL_HF_BATCHED = "hf_leftpad_batched_decision_unsteered_kvcache_v1"


def generation_protocol_name(*, backend: str, generation_batch_size: int) -> str:
    if int(generation_batch_size) <= 1:
        return GENERATION_PROTOCOL_SEQUENTIAL
    if backend == "hf":
        return GENERATION_PROTOCOL_HF_BATCHED
    return f"unsupported_batched_{backend}"


def validate_generation_batch_config(
    *,
    backend: str,
    generation_batch_size: int,
    verify_generation_smoke: bool,
    skip_generation: bool = False,
) -> None:
    if int(generation_batch_size) < 1:
        raise ValueError("--generation-batch-size must be >= 1")
    if skip_generation or int(generation_batch_size) <= 1:
        return
    if backend != "hf":
        raise ValueError("--generation-batch-size > 1 is supported only with --backend hf")
    if not verify_generation_smoke:
        raise ValueError(
            "--generation-batch-size > 1 requires --verify-generation-smoke so scalar-vs-batch "
            "drift is recorded and can fail before a paid full run"
        )


def load_decision_backend_model(
    *,
    backend: str,
    model_key: str | None,
    mlx_model: str | None,
    dtype: str,
    device: str | None = None,
):
    """Load the concrete model/tokenizer pair for decision-token experiments."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)}")
    if backend == "mlx":
        if not mlx_model:
            raise ValueError("--backend mlx requires --mlx-model or config model.mlx_model")
        return load_mlx_model(str(mlx_model), dtype=dtype)
    if not model_key:
        raise ValueError("--backend hf requires --model or config model.name")
    model, tokenizer, info = load_hf_model(
        str(model_key),
        device=device,
        dtype=resolve_torch_dtype(dtype),
    )
    effective_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    meta = PipelineMeta(
        name=str(info.get("name", model_key)),
        backend="hf",
        device=str(info.get("device", device or "auto")),
        dtype=effective_dtype,
        n_layers=info.get("num_layers"),
        hidden_size=info.get("hidden_size"),
    )
    return model, tokenizer, meta


def public_model_meta(meta: Any) -> dict:
    """Return JSON-friendly model metadata for either backend."""
    if hasattr(meta, "__dataclass_fields__"):
        return asdict(meta)
    if isinstance(meta, dict):
        return dict(meta)
    return {"repr": repr(meta)}


def decision_margin(
    model: Any,
    token_ids: list[int],
    pass_id: int,
    fail_id: int,
    *,
    steering: SteeringSpec | None,
    backend: str,
) -> float:
    """Return logit(PASS) - logit(FAIL) at the next token."""
    if backend == "mlx":
        from geoprobe.models.decision import margin as mlx_margin

        return float(mlx_margin(model, token_ids, pass_id, fail_id, steering=steering))
    if backend != "hf":
        raise ValueError(f"unknown backend {backend!r}")
    logits = hf_forward_logits_with_steering(model, token_ids, steering=steering)
    last = logits[0, -1, :].detach().float().cpu()
    return float(last[int(pass_id)] - last[int(fail_id)])


def build_decision_prefix_cache(
    model: Any,
    token_ids: list[int],
    *,
    backend: str,
    batch_size: int,
) -> HfPrefixCache | None:
    """Build a reusable prompt cache for batched HF decision-margin scoring."""
    if backend == "mlx":
        return None
    if backend != "hf":
        raise ValueError(f"unknown backend {backend!r}")
    return hf_build_prefix_cache(model, token_ids, batch_size=batch_size)


def decision_margin_with_cache(
    model: Any,
    token_ids: list[int],
    pass_id: int,
    fail_id: int,
    *,
    backend: str,
    batch_size: int,
) -> tuple[float, HfPrefixCache | None]:
    """Compute base margin AND build prefix cache in a single forward pass.

    Halves the number of full-sequence forwards per conversation by reusing the
    KV cache from the base margin forward to build the batched prefix cache.
    """
    if backend == "mlx":
        from geoprobe.models.decision import margin as mlx_margin
        m = float(mlx_margin(model, token_ids, pass_id, fail_id, steering=None))
        return m, None
    if backend != "hf":
        raise ValueError(f"unknown backend {backend!r}")
    margin, cache = hf_forward_margin_with_cache(
        model, token_ids, pass_id, fail_id, batch_size=batch_size,
    )
    return margin, cache


def decision_margins(
    model: Any,
    token_ids: list[int],
    pass_id: int,
    fail_id: int,
    *,
    steering_batch: list[SteeringSpec | None],
    backend: str,
    prefix_cache: HfPrefixCache | None = None,
) -> list[float]:
    """Return PASS-vs-FAIL margins for one prompt under many steering specs."""
    if not steering_batch:
        return []
    if backend == "mlx":
        return [
            decision_margin(model, token_ids, pass_id, fail_id, steering=steering, backend=backend)
            for steering in steering_batch
        ]
    if backend != "hf":
        raise ValueError(f"unknown backend {backend!r}")
    if prefix_cache is not None:
        logits = hf_forward_logits_batch_with_steering_cached(
            model,
            prefix_cache,
            steering_batch=steering_batch,
        )
    else:
        logits = hf_forward_logits_batch_with_steering(model, token_ids, steering_batch=steering_batch)
    last = logits[:, -1, :].detach().float().cpu()
    return (last[:, int(pass_id)] - last[:, int(fail_id)]).tolist()


def choose_status_then_complete(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    *,
    pass_id: int,
    fail_id: int,
    steering: SteeringSpec | None,
    max_new_tokens: int,
    max_generation_seconds: float | None,
    backend: str,
) -> tuple[str, dict]:
    """Steer only the binary decision token, then finish the report unsteered."""
    ids = decision_prefix_ids(tokenizer, messages)
    steered_margin = decision_margin(
        model,
        ids,
        pass_id,
        fail_id,
        steering=steering,
        backend=backend,
    )
    status = "PASS" if steered_margin > 0 else "FAIL"
    status_id = pass_id if status == "PASS" else fail_id
    generator = (
        mlx_generate_greedy_with_steering
        if backend == "mlx"
        else hf_generate_greedy_with_steering
    )
    continuation = generator(
        model,
        tokenizer,
        ids + [int(status_id)],
        max_new_tokens=max_new_tokens,
        steering=None,
        max_generation_seconds=max_generation_seconds,
    )
    reply = REPORT_PREFIX + status + continuation
    return reply, {
        "pass_token": int(pass_id),
        "fail_token": int(fail_id),
        "margin": float(steered_margin),
        "forced_status": status,
        "scope": "decision_token_only_binary_status",
        "backend": backend,
    }


def choose_status_then_complete_batch(
    model: Any,
    tokenizer: Any,
    messages_batch: list[list[dict]],
    *,
    pass_id: int,
    fail_id: int,
    steering_batch: list[SteeringSpec | None],
    max_new_tokens: int,
    max_generation_seconds: float | None,
    backend: str,
) -> list[tuple[str, dict]]:
    """Batched HF variant of ``choose_status_then_complete``.

    Steering is applied only to the PASS/FAIL decision-token margin. The forced
    status continuation is then decoded unsteered in one left-padded batch with a
    KV cache. This intentionally does not cache a steered generation path.
    """
    if backend != "hf":
        raise ValueError("choose_status_then_complete_batch supports backend='hf' only")
    if len(messages_batch) != len(steering_batch):
        raise ValueError("messages_batch and steering_batch length mismatch")
    if not messages_batch:
        return []
    ids_batch = [decision_prefix_ids(tokenizer, messages) for messages in messages_batch]
    margins = hf_decision_margins_batch_rows(
        model,
        ids_batch,
        pass_id,
        fail_id,
        steering_batch=steering_batch,
        pad_id=hf_pad_token_id(tokenizer),
    )
    statuses = ["PASS" if float(margin) > 0.0 else "FAIL" for margin in margins]
    forced_ids = [
        ids + [int(pass_id if status == "PASS" else fail_id)]
        for ids, status in zip(ids_batch, statuses)
    ]
    continuations = hf_generate_greedy_batch(
        model,
        tokenizer,
        forced_ids,
        max_new_tokens=max_new_tokens,
        max_generation_seconds=max_generation_seconds,
    )
    out: list[tuple[str, dict]] = []
    for status, margin, continuation in zip(statuses, margins, continuations):
        out.append((
            REPORT_PREFIX + status + continuation,
            {
                "pass_token": int(pass_id),
                "fail_token": int(fail_id),
                "margin": float(margin),
                "forced_status": status,
                "scope": "decision_token_only_binary_status",
                "backend": backend,
            },
        ))
    return out
