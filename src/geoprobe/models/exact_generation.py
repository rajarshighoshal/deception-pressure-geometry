"""Exact HF generation with EOT retention and branch-local random streams."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class ExactGeneration:
    token_ids: tuple[int, ...]
    text: str
    stop_token_id: int
    stop_reason: str
    decoded_text_exact: str | None = None


class GenerationCapError(ValueError):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = tuple(rows)
        summary = [
            {
                "row_index": row["row_index"],
                "seed": row["seed"],
                "token_count": len(row["token_ids"]),
                "partial_text": row["partial_text"],
            }
            for row in rows
        ]
        super().__init__(
            "generation rows reached their cap without EOT: "
            + json.dumps(summary, ensure_ascii=False)
        )


def assistant_eot_token_id(tokenizer: Any) -> int:
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if converter is not None:
        token_id = converter("<|eot_id|>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and int(token_id) >= 0 and token_id != unknown:
            return int(token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return eos_token_id
    raise ValueError("tokenizer must define the assistant EOT token")


def _left_pad(token_ids_batch: Sequence[Sequence[int]], pad_token_id: int, device: torch.device):
    width = max(len(token_ids) for token_ids in token_ids_batch)
    input_ids = torch.full(
        (len(token_ids_batch), width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, token_ids in enumerate(token_ids_batch):
        input_ids[index, width - len(token_ids):] = torch.as_tensor(
            token_ids, dtype=torch.long, device=device
        )
        attention_mask[index, width - len(token_ids):] = 1
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids


def _decode_exact(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _result(tokenizer: Any, generated: Sequence[int], eot_token_id: int) -> ExactGeneration:
    if not generated or int(generated[-1]) != eot_token_id:
        raise ValueError("generation did not retain the required assistant EOT token")
    exact_text = _decode_exact(tokenizer, generated[:-1])
    return ExactGeneration(
        token_ids=tuple(int(token) for token in generated),
        text=exact_text.strip(),
        stop_token_id=eot_token_id,
        stop_reason="eot_token",
        decoded_text_exact=exact_text,
    )


def generate_exact_greedy_batch(
    model: Any,
    tokenizer: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    max_generation_seconds: float | None = None,
) -> list[ExactGeneration]:
    if not token_ids_batch or any(not token_ids for token_ids in token_ids_batch):
        raise ValueError("generation prompts must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    eot_token_id = assistant_eot_token_id(tokenizer)
    device = next(model.parameters()).device
    input_ids, attention_mask, position_ids = _left_pad(
        token_ids_batch, int(pad_token_id), device
    )
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": eot_token_id,
        "return_dict_in_generate": True,
    }
    if max_generation_seconds is not None:
        kwargs["max_time"] = float(max_generation_seconds)
    with torch.inference_mode():
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
    prompt_width = int(input_ids.shape[1])
    outputs: list[ExactGeneration] = []
    for row_index in range(len(token_ids_batch)):
        suffix = [int(token) for token in result.sequences[row_index, prompt_width:].tolist()]
        stop_offset = next(
            (index for index, token in enumerate(suffix) if token == eot_token_id), None
        )
        if stop_offset is None:
            raise GenerationCapError([{
                "row_index": row_index,
                "seed": None,
                "token_ids": tuple(suffix),
                "partial_text": _decode_exact(tokenizer, suffix),
            }])
        outputs.append(_result(tokenizer, suffix[:stop_offset + 1], eot_token_id))
    return outputs


def _top_p_probs(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    scaled = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            -1, sorted_indices, sorted_logits
        )
    return torch.softmax(scaled, dim=-1)


def generate_exact_sample_batch(
    model: Any,
    tokenizer: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_generation_seconds: float | None = None,
) -> list[ExactGeneration]:
    """Sample with one reproducible RNG stream per row while retaining batch compute."""
    if not token_ids_batch or any(not token_ids for token_ids in token_ids_batch):
        raise ValueError("generation prompts must be non-empty")
    if len(seeds) != len(token_ids_batch):
        raise ValueError("one sampling seed is required per prompt")
    if temperature <= 0 or not 0 < top_p <= 1 or max_new_tokens < 1:
        raise ValueError("invalid sampling parameters")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    eot_token_id = assistant_eot_token_id(tokenizer)
    device = next(model.parameters()).device
    input_ids, attention_mask, position_ids = _left_pad(
        token_ids_batch, int(pad_token_id), device
    )
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    generated: list[list[int]] = [[] for _ in token_ids_batch]
    active = torch.ones(len(token_ids_batch), dtype=torch.bool, device=device)
    current_ids = input_ids
    current_positions = position_ids
    past_key_values = None
    started = time.monotonic()
    with torch.inference_mode():
        for _step in range(max_new_tokens):
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                position_ids=current_positions,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            probs = _top_p_probs(
                outputs.logits[:, -1, :], temperature=temperature, top_p=top_p
            )
            active_before = active.clone()
            next_tokens = torch.full(
                (len(token_ids_batch),), int(pad_token_id), dtype=torch.long, device=device
            )
            for row_index in active_before.nonzero(as_tuple=False).flatten().tolist():
                next_tokens[row_index] = torch.multinomial(
                    probs[row_index], num_samples=1, generator=generators[row_index]
                )
                token = int(next_tokens[row_index])
                generated[row_index].append(token)
                if token == eot_token_id:
                    active[row_index] = False
            if not active.any().item():
                break
            if max_generation_seconds is not None and (
                time.monotonic() - started >= max_generation_seconds
            ):
                raise TimeoutError("sampled generation exceeded its wall-time cap")
            next_positions = attention_mask.sum(dim=-1, keepdim=True)
            attention_mask = torch.cat(
                [attention_mask, active_before.to(attention_mask.dtype).unsqueeze(1)], dim=1
            )
            current_ids = next_tokens.unsqueeze(1)
            current_positions = next_positions
            past_key_values = outputs.past_key_values
    missing = [index for index, tokens in enumerate(generated) if not tokens or tokens[-1] != eot_token_id]
    if missing:
        raise GenerationCapError([
            {
                "row_index": index,
                "seed": int(seeds[index]),
                "token_ids": tuple(generated[index]),
                "partial_text": _decode_exact(tokenizer, generated[index]),
            }
            for index in missing
        ])
    return [_result(tokenizer, tokens, eot_token_id) for tokens in generated]


def generate_exact_sample_independent(
    model: Any,
    tokenizer: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_generation_seconds: float | None = None,
) -> list[ExactGeneration]:
    """Preserve token trajectories when unrelated prompt rows enter or leave a work queue."""
    if len(seeds) != len(token_ids_batch):
        raise ValueError("one sampling seed is required per prompt")
    return [
        generate_exact_sample_batch(
            model,
            tokenizer,
            [token_ids],
            seeds=[seed],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_generation_seconds=max_generation_seconds,
        )[0]
        for token_ids, seed in zip(token_ids_batch, seeds)
    ]


def user_turn_suffix_ids(tokenizer: Any, content: str) -> list[int]:
    rendered = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + content
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return [int(token) for token in tokenizer.encode(rendered, add_special_tokens=False)]


def initial_generation_prompt_ids(tokenizer: Any, system: str, user: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return [int(token) for token in token_ids]


__all__ = [
    "ExactGeneration", "GenerationCapError", "assistant_eot_token_id",
    "generate_exact_greedy_batch", "generate_exact_sample_batch",
    "generate_exact_sample_independent", "initial_generation_prompt_ids",
    "user_turn_suffix_ids",
]
