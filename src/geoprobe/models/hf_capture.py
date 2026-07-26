"""PyTorch activation backend (CUDA/CPU on host, MPS on Mac fallback).

Mirrors ``mlx_capture.MlxActivationPipeline`` so the two are interchangeable behind the
``ActivationPipeline`` contract. Uses the shared canonical tokenizer and HF's
``output_hidden_states`` + ``last_token_residual`` extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Sequence
from contextlib import contextmanager
import time

import torch

from geoprobe.models.hooks import last_token_residual
from geoprobe.models.interface import PipelineMeta, ResidualSteeringSpec, SteeringSpec, resolve_torch_dtype
from geoprobe.models.loader import choose_device, cleanup, load_hf_model
from geoprobe.models.tokenization import chat_token_ids


@dataclass
class HfPrefixCache:
    """Reusable KV cache for scoring the last token of one shared prompt."""

    token_ids: tuple[int, ...]
    past_key_values: Any
    last_token_id: int
    context_length: int
    batch_size: int


def _transformer_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("cannot locate transformer block list for steering")


def _normalize_steering(steering: SteeringSpec | None) -> list[ResidualSteeringSpec]:
    if steering is None:
        return []
    if isinstance(steering, ResidualSteeringSpec):
        return [steering] if steering.alpha != 0 else []
    return [spec for spec in steering if spec.alpha != 0]


@contextmanager
def _residual_steering_hook(model: Any, steering: SteeringSpec | None):
    specs = _normalize_steering(steering)
    if not specs:
        yield
        return
    blocks = _transformer_layers(model)
    param = next(model.parameters())
    by_layer: dict[int, list[tuple[torch.Tensor, float]]] = {}
    for spec in specs:
        if spec.layer <= 0 or spec.layer > len(blocks):
            raise ValueError(f"HF residual steering layer must be in [1, {len(blocks)}], got {spec.layer}")
        vector = spec.direction.detach().to(param.device, dtype=param.dtype)
        by_layer.setdefault(int(spec.layer), []).append((vector, float(spec.alpha)))

    handles = []
    for layer_index, nudges in by_layer.items():
        def hook(_module, _inputs, output, *, nudges=nudges):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                for vector, alpha in nudges:
                    hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
                return (hidden, *output[1:])
            hidden = output.clone()
            for vector, alpha in nudges:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
            return hidden

        handles.append(blocks[layer_index - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward_logits_with_steering(
    model: Any,
    token_ids: list[int],
    *,
    steering: SteeringSpec | None = None,
    cleanup_after: bool = False,
) -> torch.Tensor:
    """Forward pass on explicit token ids with optional residual steering.

    This is the HF/PyTorch analogue of MLX ``_forward_logits_with_steering``.
    It is intentionally token-id based so decision-token control can force a
    status token and then continue generation from that exact prefix.
    """
    param = next(model.parameters())
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=param.device)
    attention_mask = torch.ones_like(input_ids)
    with _residual_steering_hook(model, steering):
        with torch.inference_mode():
            output = _forward_last_logits(model, input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits.detach()
    del output
    if cleanup_after:
        cleanup()
    return logits


def _supports_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _forward_last_logits(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any | None = None,
    use_cache: bool = False,
    position_ids: torch.Tensor | None = None,
):
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
        kwargs["use_cache"] = True
    if position_ids is not None and _supports_kwarg(model.forward, "position_ids"):
        kwargs["position_ids"] = position_ids
    if _supports_kwarg(model.forward, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    return model(**kwargs)


def build_prefix_cache(model: Any, token_ids: list[int], *, batch_size: int) -> HfPrefixCache:
    """Prefill all but the last prompt token once, then repeat cache for a batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(token_ids) < 2:
        raise ValueError("cached scoring needs at least two prompt tokens")
    param = next(model.parameters())
    context_ids = list(token_ids[:-1])
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=param.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = _forward_last_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = output.past_key_values
    del output
    if past_key_values is None:
        raise ValueError("model did not return past_key_values for cached scoring")
    if batch_size > 1:
        if not hasattr(past_key_values, "batch_repeat_interleave"):
            raise ValueError("cached scoring needs a Cache object with batch_repeat_interleave")
        past_key_values.batch_repeat_interleave(batch_size)
    return HfPrefixCache(
        token_ids=tuple(int(token_id) for token_id in token_ids),
        past_key_values=past_key_values,
        last_token_id=int(token_ids[-1]),
        context_length=len(context_ids),
        batch_size=int(batch_size),
    )


def forward_margin_with_cache(
    model: Any,
    token_ids: list[int],
    pass_id: int,
    fail_id: int,
    *,
    batch_size: int,
) -> tuple[float, HfPrefixCache | None]:
    """Compute base (unsteered) margin AND build prefix cache in one forward.

    Runs a single full-sequence forward with use_cache=True, extracts the PASS/FAIL
    margin from the logits, then crops the KV cache to context_length (all but the
    last token) and replicates it for batched scoring.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(token_ids) < 2:
        raise ValueError("cached scoring needs at least two prompt tokens")
    param = next(model.parameters())
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=param.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = _forward_last_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    last_logits = output.logits[0, -1, :].detach().float().cpu()
    margin = float(last_logits[int(pass_id)] - last_logits[int(fail_id)])
    past_key_values = output.past_key_values
    del output
    if past_key_values is None:
        return margin, None
    context_length = len(token_ids) - 1
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(context_length)
    else:
        return margin, None
    if batch_size > 1:
        if not hasattr(past_key_values, "batch_repeat_interleave"):
            return margin, None
        past_key_values.batch_repeat_interleave(batch_size)
    cache = HfPrefixCache(
        token_ids=tuple(int(t) for t in token_ids),
        past_key_values=past_key_values,
        last_token_id=int(token_ids[-1]),
        context_length=context_length,
        batch_size=int(batch_size),
    )
    return margin, cache


@contextmanager
def _batched_residual_steering_hook(model: Any, steering_batch: Sequence[SteeringSpec | None]):
    """Apply per-example residual nudges for a batch sharing the same token ids."""
    normalized = [_normalize_steering(steering) for steering in steering_batch]
    specs = [spec for example in normalized for spec in example]
    if not specs:
        yield
        return
    blocks = _transformer_layers(model)
    param = next(model.parameters())
    batch_size = len(steering_batch)
    hidden_size = int(specs[0].direction.numel())
    by_layer: dict[int, torch.Tensor] = {}
    for example_index, example_specs in enumerate(normalized):
        for spec in example_specs:
            if spec.layer <= 0 or spec.layer > len(blocks):
                raise ValueError(f"HF residual steering layer must be in [1, {len(blocks)}], got {spec.layer}")
            vector = spec.direction.detach().reshape(-1).to(param.device, dtype=param.dtype)
            if int(vector.numel()) != hidden_size:
                raise ValueError(f"batched steering vector size mismatch: got {int(vector.numel())}, expected {hidden_size}")
            if spec.layer not in by_layer:
                by_layer[spec.layer] = torch.zeros((batch_size, hidden_size), device=param.device, dtype=param.dtype)
            by_layer[spec.layer][example_index] += float(spec.alpha) * vector

    handles = []
    for layer_index, delta in by_layer.items():
        def hook(_module, _inputs, output, *, delta=delta):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                if hidden.shape[0] != delta.shape[0] or hidden.shape[-1] != delta.shape[-1]:
                    raise ValueError(
                        f"batched steering shape mismatch: hidden={tuple(hidden.shape)}, delta={tuple(delta.shape)}"
                    )
                hidden[:, -1, :] = hidden[:, -1, :] + delta.to(hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:])
            hidden = output.clone()
            if hidden.shape[0] != delta.shape[0] or hidden.shape[-1] != delta.shape[-1]:
                raise ValueError(f"batched steering shape mismatch: hidden={tuple(hidden.shape)}, delta={tuple(delta.shape)}")
            hidden[:, -1, :] = hidden[:, -1, :] + delta.to(hidden.device, dtype=hidden.dtype)
            return hidden

        handles.append(blocks[layer_index - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class _FinishedRowTracker:
    """Logits-processor-shaped observer: marks rows finished once an EOS appears in their
    generated region, so the controller can freeze them (HF generate keeps forwarding finished
    rows with pad tokens until the whole batch completes). Read via ``.finished``."""

    def __init__(self, eos_ids: set[int], prompt_width: int, batch_size: int):
        self._eos = sorted(eos_ids)
        self._prompt_width = int(prompt_width)
        self.finished = [False] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        gen = input_ids[:, self._prompt_width:]
        if gen.shape[1] > 0 and self._eos:
            eos = torch.tensor(self._eos, device=gen.device)
            self.finished = torch.isin(gen, eos).any(dim=1).cpu().tolist()
        return scores


@contextmanager
def per_step_controller_hook(model: Any, controller: Any | None, *, layer: int,
                             finished_mask: Any | None = None):
    """Closed-loop per-step steering: at every forward through block `layer`, feed the last-token
    residual state [B, d] to ``controller.vectors`` and add ``alpha * direction`` back in place.

    The controller is stateful per row (integration window, hysteresis, arming); the CALLER must
    reset it for the batch before entering. vectors() is invoked on every forward -- prefill
    included (the prefill last token is step 0, the pre_response state) -- so the depth history
    stays truthful even on steps that push nothing. With left padding the last position is a real
    token for every row. Hooks registered earlier (e.g. TrajectoryCapture) see the PRE-push state
    at this layer, matching the static-steering capture semantics.

    finished_mask: zero-arg callable returning a per-row bool list; rows marked True are frozen
    on the controller (no pushes, no telemetry updates from pad-token states).

    Freeze timing (pinned by test): the tracker updates in a logits processor, which runs AFTER
    each forward. So the forward whose INPUT is the row's EOS token still reaches the controller
    (it is a real state of the sequence -- its push affects only a discarded token), and every
    pad forward after it is frozen. One real terminal step, zero pad steps.
    """
    if controller is None:
        yield
        return
    import numpy as np

    blocks = _transformer_layers(model)
    if layer <= 0 or layer > len(blocks):
        raise ValueError(f"per-step controller layer must be in [1, {len(blocks)}], got {layer}")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if finished_mask is not None and hasattr(controller, "set_frozen"):
            controller.set_frozen(finished_mask())
        state = hidden[:, -1, :].detach().to(torch.float32).cpu().numpy()
        directions, alphas = controller.vectors(state)
        if not np.any(alphas > 0):
            return None
        delta = torch.from_numpy(alphas[:, None] * directions).to(hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = blocks[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_logits_batch_with_steering(
    model: Any,
    token_ids: list[int],
    *,
    steering_batch: Sequence[SteeringSpec | None],
    cleanup_after: bool = False,
) -> torch.Tensor:
    """Batched analogue of ``forward_logits_with_steering`` for one shared prompt."""
    if not steering_batch:
        raise ValueError("steering_batch must not be empty")
    param = next(model.parameters())
    input_ids = torch.tensor([token_ids] * len(steering_batch), dtype=torch.long, device=param.device)
    attention_mask = torch.ones_like(input_ids)
    with _batched_residual_steering_hook(model, steering_batch):
        with torch.inference_mode():
            output = _forward_last_logits(model, input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits.detach()
    del output
    if cleanup_after:
        cleanup()
    return logits


def forward_logits_batch_rows_with_steering(
    model: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    steering_batch: Sequence[SteeringSpec | None],
    pad_id: int,
    cleanup_after: bool = False,
) -> torch.Tensor:
    """Score heterogeneous prompts in one left-padded, per-row-steered forward.

    Unlike :func:`forward_logits_batch_with_steering`, rows may have different
    prefix lengths.  The last sequence position is consequently a real token
    for every row, which makes the existing last-token residual hook valid.
    """
    if len(token_ids_batch) != len(steering_batch):
        raise ValueError("token_ids_batch and steering_batch length mismatch")
    if not token_ids_batch:
        raise ValueError("token_ids_batch must not be empty")
    param = next(model.parameters())
    input_ids, attention_mask, position_ids = _left_pad_batch(
        token_ids_batch,
        pad_id=pad_id,
        device=param.device,
    )
    with _batched_residual_steering_hook(model, steering_batch):
        with torch.inference_mode():
            output = _forward_last_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
    logits = output.logits.detach()
    del output
    if cleanup_after:
        cleanup()
    return logits


def forward_logits_batch_with_steering_cached(
    model: Any,
    prefix_cache: HfPrefixCache,
    *,
    steering_batch: Sequence[SteeringSpec | None],
    cleanup_after: bool = False,
) -> torch.Tensor:
    """Score one prompt's last token under many nudges using a reusable KV cache."""
    if not steering_batch:
        raise ValueError("steering_batch must not be empty")
    batch_size = len(steering_batch)
    if batch_size != prefix_cache.batch_size:
        raise ValueError(
            f"cached steering batch size mismatch: got {batch_size}, expected {prefix_cache.batch_size}"
        )
    param = next(model.parameters())
    input_ids = torch.full(
        (batch_size, 1),
        int(prefix_cache.last_token_id),
        dtype=torch.long,
        device=param.device,
    )
    attention_mask = torch.ones(
        (batch_size, prefix_cache.context_length + 1),
        dtype=torch.long,
        device=param.device,
    )
    try:
        with _batched_residual_steering_hook(model, steering_batch):
            with torch.inference_mode():
                output = _forward_last_logits(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=prefix_cache.past_key_values,
                    use_cache=True,
                )
        logits = output.logits.detach()
        del output
    finally:
        if hasattr(prefix_cache.past_key_values, "crop"):
            prefix_cache.past_key_values.crop(prefix_cache.context_length)
    if cleanup_after:
        cleanup()
    return logits


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(item) for item in eos}
    return {int(eos)}


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _left_pad_batch(
    token_ids_batch: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not token_ids_batch:
        raise ValueError("token_ids_batch must not be empty")
    max_len = max(len(ids) for ids in token_ids_batch)
    if max_len < 1:
        raise ValueError("empty prompt in batch")
    input_ids = torch.full(
        (len(token_ids_batch), max_len),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, ids_raw in enumerate(token_ids_batch):
        ids = [int(token_id) for token_id in ids_raw]
        if not ids:
            raise ValueError("empty prompt in batch")
        start = max_len - len(ids)
        input_ids[row_idx, start:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_idx, start:] = 1
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids


def _right_pad_batch(
    token_ids_batch: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_ids_batch:
        raise ValueError("token_ids_batch must not be empty")
    max_len = max(len(ids) for ids in token_ids_batch)
    if max_len < 1:
        raise ValueError("empty prompt in batch")
    input_ids = torch.full(
        (len(token_ids_batch), max_len),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, ids_raw in enumerate(token_ids_batch):
        ids = [int(token_id) for token_id in ids_raw]
        if not ids:
            raise ValueError("empty prompt in batch")
        input_ids[row_idx, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_idx, :len(ids)] = 1
    return input_ids, attention_mask


def _pad_token_id(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    if isinstance(eos, (list, tuple, set)):
        return int(next(iter(eos)))
    return int(eos)


def decision_margins_batch_rows(
    model: Any,
    token_ids_batch: Sequence[Sequence[int]],
    pass_id: int,
    fail_id: int,
    *,
    steering_batch: Sequence[SteeringSpec | None],
    pad_id: int,
) -> list[float]:
    """Score different prompts in one left-padded HF forward."""
    logits = forward_logits_batch_rows_with_steering(
        model,
        token_ids_batch,
        steering_batch=steering_batch,
        pad_id=pad_id,
    )
    last = logits[:, -1, :].float().cpu()
    return (last[:, int(pass_id)] - last[:, int(fail_id)]).tolist()


def generate_greedy_with_steering(
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    steering: SteeringSpec | None = None,
    max_generation_seconds: float | None = None,
) -> str:
    """Greedy token-id generation with optional residual steering.

    Used for decision-token control parity with the MLX backend. The caller can
    pass a prefix that already includes a forced PASS/FAIL status token.

    The unsteered path uses a KV cache. The steered path keeps full recompute
    semantics because transient steering is meant to affect only the current
    decision logits, not future cached hidden states.
    """
    if not _normalize_steering(steering):
        return _generate_greedy_unsteered_cached(
            model,
            tokenizer,
            token_ids,
            max_new_tokens=max_new_tokens,
            max_generation_seconds=max_generation_seconds,
        )
    return _generate_greedy_full_recompute(
        model,
        tokenizer,
        token_ids,
        max_new_tokens=max_new_tokens,
        steering=steering,
        max_generation_seconds=max_generation_seconds,
    )


def _generate_greedy_full_recompute(
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    steering: SteeringSpec | None,
    max_generation_seconds: float | None,
) -> str:
    generated: list[int] = []
    ids = list(token_ids)
    eos_ids = _eos_token_ids(tokenizer)
    deadline = None
    if max_generation_seconds is not None and max_generation_seconds > 0:
        deadline = time.monotonic() + max_generation_seconds
    try:
        for _ in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                break
            logits = forward_logits_with_steering(model, ids, steering=steering)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if next_id in eos_ids:
                break
            generated.append(next_id)
            ids.append(next_id)
    finally:
        cleanup()
    return _decode_token_ids(tokenizer, generated).strip()


def _generate_greedy_unsteered_cached(
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    max_generation_seconds: float | None,
) -> str:
    generated: list[int] = []
    ids = [int(token_id) for token_id in token_ids]
    eos_ids = _eos_token_ids(tokenizer)
    deadline = None
    if max_generation_seconds is not None and max_generation_seconds > 0:
        deadline = time.monotonic() + max_generation_seconds
    param = next(model.parameters())
    try:
        input_ids = torch.tensor([ids], dtype=torch.long, device=param.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = _forward_last_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past_key_values = getattr(output, "past_key_values", None)
        next_id = int(torch.argmax(output.logits[0, -1, :]).item())
        del output
        if past_key_values is None:
            return _generate_greedy_full_recompute(
                model,
                tokenizer,
                token_ids,
                max_new_tokens=max_new_tokens,
                steering=None,
                max_generation_seconds=max_generation_seconds,
            )
        for _ in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if next_id in eos_ids:
                break
            generated.append(next_id)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=param.device),
                ],
                dim=1,
            )
            with torch.inference_mode():
                output = _forward_last_logits(
                    model,
                    input_ids=torch.tensor([[next_id]], dtype=torch.long, device=param.device),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = getattr(output, "past_key_values", None) or past_key_values
            next_id = int(torch.argmax(output.logits[0, -1, :]).item())
            del output
    finally:
        cleanup()
    return _decode_token_ids(tokenizer, generated).strip()


def generate_greedy_batch(
    model: Any,
    tokenizer: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    max_generation_seconds: float | None = None,
) -> list[str]:
    """Batched unsteered greedy decode over left-padded prompts."""
    if not token_ids_batch:
        return []
    eos_ids = _eos_token_ids(tokenizer)
    pad_id = _pad_token_id(tokenizer)
    deadline = None
    if max_generation_seconds is not None and max_generation_seconds > 0:
        deadline = time.monotonic() + max_generation_seconds
    param = next(model.parameters())
    batch_size = len(token_ids_batch)
    try:
        input_ids, attention_mask, position_ids = _left_pad_batch(
            token_ids_batch,
            pad_id=pad_id,
            device=param.device,
        )
        with torch.inference_mode():
            output = _forward_last_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
        past_key_values = getattr(output, "past_key_values", None)
        next_ids = torch.argmax(output.logits[:, -1, :], dim=-1)
        del output
        if past_key_values is None:
            return [
                generate_greedy_with_steering(
                    model,
                    tokenizer,
                    list(ids),
                    max_new_tokens=max_new_tokens,
                    steering=None,
                    max_generation_seconds=max_generation_seconds,
                )
                for ids in token_ids_batch
            ]
        real_lengths = torch.tensor(
            [len(ids) for ids in token_ids_batch],
            dtype=torch.long,
            device=param.device,
        )
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=param.device)
        for step in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                break
            for row_idx in range(batch_size):
                if bool(finished[row_idx]):
                    continue
                token = int(next_ids[row_idx].item())
                if token in eos_ids:
                    finished[row_idx] = True
                else:
                    generated[row_idx].append(token)
            if bool(finished.all()):
                break
            feed = torch.where(
                finished,
                torch.full_like(next_ids, pad_id),
                next_ids,
            )
            attention_mask = torch.cat(
                [attention_mask, (~finished).long().unsqueeze(1)],
                dim=1,
            )
            step_positions = (real_lengths + step).unsqueeze(1)
            with torch.inference_mode():
                output = _forward_last_logits(
                    model,
                    input_ids=feed.unsqueeze(1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=step_positions,
                )
            past_key_values = getattr(output, "past_key_values", None) or past_key_values
            next_ids = torch.argmax(output.logits[:, -1, :], dim=-1)
            del output
    finally:
        cleanup()
    return [_decode_token_ids(tokenizer, tokens).strip() for tokens in generated]


class HfActivationPipeline:
    """Backend-agnostic pipeline implemented on top of a HuggingFace model."""

    def __init__(self, model: Any, tokenizer: Any, meta: PipelineMeta, max_length: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.max_length = max_length

    @classmethod
    def load(
        cls,
        model_key: str,
        *,
        device: str | None = None,
        dtype: str = "float16",
        max_length: int = 2048,
    ) -> "HfActivationPipeline":
        device = device or choose_device()
        model, tokenizer, info = load_hf_model(
            model_key, device=device, dtype=resolve_torch_dtype(dtype)
        )
        effective = str(next(model.parameters()).dtype).replace("torch.", "")
        meta = PipelineMeta(
            name=info["name"],
            backend="hf",
            device=device,
            dtype=effective,
            n_layers=info.get("num_layers"),
            hidden_size=info.get("hidden_size"),
        )
        return cls(model, tokenizer, meta, max_length)

    def _input_ids(self, messages: list[dict], *, add_generation_prompt: bool) -> torch.Tensor:
        ids = chat_token_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            max_length=self.max_length,
        )
        return torch.tensor([ids], dtype=torch.long, device=self.meta.device)

    def capture(
        self,
        messages: list[dict],
        layers: list[int],
        *,
        add_generation_prompt: bool,
    ) -> dict[int, torch.Tensor]:
        input_ids = self._input_ids(messages, add_generation_prompt=add_generation_prompt)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        residual = last_token_residual(output.hidden_states, attention_mask)
        result = {int(layer): residual[int(layer)] for layer in layers}
        del output, residual
        cleanup()
        return result

    def capture_batch(
        self,
        messages_batch: Sequence[list[dict]],
        layers: list[int],
        *,
        add_generation_prompt: bool,
    ) -> dict[int, torch.Tensor]:
        """Last-token residuals for a left-padded batch of chat prompts.

        This is semantically the same as calling ``capture`` for every prompt,
        but it keeps CUDA fed for fixed replay/capture jobs where no generation
        or per-row steering is involved.
        """
        token_ids_batch = [
            chat_token_ids(
                self.tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                max_length=self.max_length,
            )
            for messages in messages_batch
        ]
        input_ids, attention_mask = _right_pad_batch(
            token_ids_batch,
            pad_id=_pad_token_id(self.tokenizer),
            device=torch.device(self.meta.device),
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        residual = last_token_residual(output.hidden_states, attention_mask)
        result = {int(layer): residual[int(layer)] for layer in layers}
        del output, residual, input_ids, attention_mask
        cleanup()
        return result

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_generation_seconds: float | None = None,
        steering: SteeringSpec | None = None,
    ) -> str:
        input_ids = self._input_ids(messages, add_generation_prompt=True)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if max_generation_seconds is not None:
            kwargs["max_time"] = max_generation_seconds
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        with _residual_steering_hook(self.model, steering):
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids), **kwargs
                )
        reply = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        del generated
        cleanup()
        return reply

    def generate_batch(
        self,
        messages_batch: Sequence[list[dict]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_generation_seconds: float | None = None,
        steering_batch: Sequence[SteeringSpec | None] | None = None,
        per_step_controller: Any | None = None,
        per_step_controller_layer: int = 16,
        per_step_arm_priors: Sequence[float | None] | None = None,
    ) -> list[str]:
        if not messages_batch:
            return []
        steering_items: Sequence[SteeringSpec | None] = steering_batch or [None] * len(messages_batch)
        if len(steering_items) != len(messages_batch):
            raise ValueError("messages_batch and steering_batch length mismatch")
        token_ids_batch = [
            chat_token_ids(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                max_length=self.max_length,
            )
            for messages in messages_batch
        ]
        input_ids, attention_mask, _position_ids = _left_pad_batch(
            token_ids_batch,
            pad_id=_pad_token_id(self.tokenizer),
            device=torch.device(self.meta.device),
        )
        if seed is not None:
            torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if max_generation_seconds is not None:
            kwargs["max_time"] = max_generation_seconds
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        finished_tracker = None
        if per_step_controller is not None:
            if per_step_arm_priors is not None:
                per_step_controller.reset(len(messages_batch), per_step_arm_priors)
            else:
                per_step_controller.reset(len(messages_batch))
            from transformers import LogitsProcessorList
            finished_tracker = _FinishedRowTracker(
                _eos_token_ids(self.tokenizer),
                prompt_width=int(input_ids.shape[1]),
                batch_size=len(messages_batch),
            )
            kwargs["logits_processor"] = LogitsProcessorList([finished_tracker])
        with _batched_residual_steering_hook(self.model, steering_items):
            with per_step_controller_hook(
                self.model, per_step_controller,
                layer=per_step_controller_layer,
                finished_mask=((lambda: finished_tracker.finished)
                               if finished_tracker is not None else None),
            ):
                with torch.inference_mode():
                    generated = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=_position_ids,
                        **kwargs,
                    )
        prompt_width = int(input_ids.shape[1])
        replies = [
            self.tokenizer.decode(generated[idx, prompt_width:], skip_special_tokens=True).strip()
            for idx in range(int(generated.shape[0]))
        ]
        del generated, input_ids, attention_mask, _position_ids
        cleanup()
        return replies

    def next_token_id(
        self,
        messages: list[dict],
        *,
        steering: SteeringSpec | None = None,
    ) -> int:
        token_ids = chat_token_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            max_length=self.max_length,
        )
        logits = forward_logits_with_steering(self.model, token_ids, steering=steering)
        token_id = int(torch.argmax(logits[0, -1, :]).item())
        del logits
        cleanup()
        return token_id

    def next_token_ids_batch(
        self,
        messages_batch: Sequence[list[dict]],
        *,
        steering_batch: Sequence[SteeringSpec | None] | None = None,
    ) -> list[int]:
        if not messages_batch:
            return []
        steering_items: Sequence[SteeringSpec | None] = steering_batch or [None] * len(messages_batch)
        if len(steering_items) != len(messages_batch):
            raise ValueError("messages_batch and steering_batch length mismatch")
        token_ids_batch = [
            chat_token_ids(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                max_length=self.max_length,
            )
            for messages in messages_batch
        ]
        input_ids, attention_mask, position_ids = _left_pad_batch(
            token_ids_batch,
            pad_id=_pad_token_id(self.tokenizer),
            device=torch.device(self.meta.device),
        )
        with _batched_residual_steering_hook(self.model, steering_items):
            with torch.inference_mode():
                output = _forward_last_logits(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        next_ids = torch.argmax(output.logits[:, -1, :], dim=-1).detach().cpu().tolist()
        del output, input_ids, attention_mask, position_ids
        cleanup()
        return [int(token_id) for token_id in next_ids]
