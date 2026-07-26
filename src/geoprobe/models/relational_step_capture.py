"""One-token HF relational captures with explicit cache rollback semantics."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from geoprobe.models.hf_capture import HfPrefixCache, _supports_kwarg, _transformer_layers


@dataclass(frozen=True, slots=True)
class RelationalStepCapture:
    """Native full-head relations and residuals for the forwarded token block.

    Tensors are detached CPU snapshots. ``residuals[layer]`` has shape
    ``[batch, query_tokens, hidden]`` and ``attentions[layer]`` has shape
    ``[batch, heads, query_tokens, key_tokens]``.
    """

    residuals: dict[int, torch.Tensor]
    attentions: dict[int, torch.Tensor]


@dataclass(frozen=True, slots=True)
class RelationalStepForward:
    """Outputs of one cached token forward, optionally rolled back by the caller."""

    logits: torch.Tensor
    capture: RelationalStepCapture
    past_key_values: Any


def _cache_length(cache: Any) -> int | None:
    get_length = getattr(cache, "get_seq_length", None)
    if get_length is None:
        return None
    value = get_length()
    if isinstance(value, torch.Tensor):
        value = value.item()
    return int(value)


def _crop_cache(cache: Any, length: int) -> None:
    crop = getattr(cache, "crop", None)
    if crop is None:
        raise ValueError("transactional relational capture needs a cache with crop()")
    crop(int(length))


def _validate_layers(model: Any, layers: Sequence[int]) -> tuple[int, ...]:
    blocks = _transformer_layers(model)
    selected = tuple(int(layer) for layer in layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("capture layers must be non-empty and unique")
    if min(selected) < 1 or max(selected) > len(blocks):
        raise ValueError(f"capture layers must be in [1, {len(blocks)}]")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("transactional full-head attention capture requires eager attention")
    return selected


def capture_relational_step_with_cache(
    model: Any,
    prefix_cache: HfPrefixCache,
    *,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    rollback: bool,
) -> RelationalStepForward:
    """Forward ``input_ids`` from a cached prestate and optionally restore it exactly.

    The cache object is deliberately mutated only for the duration of a
    rollback pass.  A committing pass leaves its KV entries in place, so the
    next token is conditioned on all earlier committed interventions.
    """
    selected = _validate_layers(model, layers)
    if input_ids.ndim != 2 or input_ids.shape[0] != prefix_cache.batch_size or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [prefix-cache batch, positive query length]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must have dtype torch.long")
    parameter = next(model.parameters())
    if input_ids.device != parameter.device:
        input_ids = input_ids.to(parameter.device)
    cache_length = _cache_length(prefix_cache.past_key_values)
    if cache_length is not None and cache_length != prefix_cache.context_length:
        raise ValueError(
            "prefix cache length does not match its declared transactional prestate: "
            f"{cache_length} != {prefix_cache.context_length}"
        )
    if rollback and not hasattr(prefix_cache.past_key_values, "crop"):
        raise ValueError("transactional relational capture needs a cache with crop()")

    batch_size, query_length = input_ids.shape
    attention_mask = torch.ones(
        (batch_size, prefix_cache.context_length + query_length),
        dtype=torch.long,
        device=parameter.device,
    )
    position_ids = torch.arange(
        prefix_cache.context_length,
        prefix_cache.context_length + query_length,
        dtype=torch.long,
        device=parameter.device,
    ).unsqueeze(0).expand(batch_size, -1)
    blocks = _transformer_layers(model)
    hidden: dict[int, torch.Tensor] = {}
    attention: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    for layer in selected:
        block = blocks[layer - 1]

        def block_hook(_module: Any, _inputs: Any, output: Any, *, layer: int = layer) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            hidden[layer] = tensor

        def attention_hook(_module: Any, _inputs: Any, output: Any, *, layer: int = layer) -> None:
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise ValueError(f"layer {layer} did not return dense attention weights")
            attention[layer] = output[1]

        handles.append(block.register_forward_hook(block_hook))
        handles.append(block.self_attn.register_forward_hook(attention_hook))

    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": prefix_cache.past_key_values,
        "use_cache": True,
        "output_attentions": True,
    }
    if _supports_kwarg(model.forward, "position_ids"):
        kwargs["position_ids"] = position_ids
    if _supports_kwarg(model.forward, "logits_to_keep"):
        kwargs["logits_to_keep"] = query_length
    try:
        with torch.inference_mode():
            output = model(**kwargs)
        logits = output.logits.detach()
        returned_cache = getattr(output, "past_key_values", None)
        if returned_cache is None:
            raise ValueError("model did not return past_key_values for relational step capture")
        prefix_cache.past_key_values = returned_cache
        expected_keys = prefix_cache.context_length + query_length
        residuals: dict[int, torch.Tensor] = {}
        attentions: dict[int, torch.Tensor] = {}
        expected_heads = int(model.config.num_attention_heads)
        for layer in selected:
            residual = hidden.get(layer)
            weights = attention.get(layer)
            if residual is None or weights is None:
                raise ValueError(f"capture hooks did not observe layer {layer}")
            if tuple(residual.shape[:2]) != (batch_size, query_length):
                raise ValueError(f"layer {layer} residual shape mismatch: {tuple(residual.shape)}")
            expected_attention_shape = (batch_size, expected_heads, query_length, expected_keys)
            if tuple(weights.shape) != expected_attention_shape:
                raise ValueError(
                    f"layer {layer} attention shape {tuple(weights.shape)} != {expected_attention_shape}"
                )
            if not torch.isfinite(residual).all().item() or not torch.isfinite(weights).all().item():
                raise ValueError(f"layer {layer} relational capture contains non-finite values")
            residuals[layer] = residual.detach().to("cpu").clone()
            attentions[layer] = weights.detach().to("cpu").clone()
        result = RelationalStepForward(
            logits=logits,
            capture=RelationalStepCapture(residuals=residuals, attentions=attentions),
            past_key_values=returned_cache,
        )
        del output
        return result
    finally:
        for handle in handles:
            handle.remove()
        if rollback:
            _crop_cache(prefix_cache.past_key_values, prefix_cache.context_length)


__all__ = [
    "RelationalStepCapture",
    "RelationalStepForward",
    "capture_relational_step_with_cache",
]
