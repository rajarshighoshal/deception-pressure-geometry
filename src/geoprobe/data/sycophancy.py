from __future__ import annotations

from typing import Any


DEFAULT_PROMPT_TEMPLATE = (
    "Human: {question}\n\n"
    "Assistant: I believe the best answer is"
)


def format_sycophancy_prompt(record: dict[str, Any], prompt_template: str | None = None) -> str:
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    question = record["question"].strip()
    return template.format(question=question)


def get_sycophancy_answers(record: dict[str, Any]) -> tuple[str, str]:
    return record["answer_matching_behavior"], record["answer_not_matching_behavior"]
