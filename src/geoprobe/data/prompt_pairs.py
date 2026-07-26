from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPair:
    prompt_id: str
    neutral_prompt: str
    target_prompt: str
    label: int
    split: str = "train"
    metadata: dict | None = None
