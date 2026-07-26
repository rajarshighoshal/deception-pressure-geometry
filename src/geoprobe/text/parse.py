"""CSV-style argument parsing helpers shared across geoprobe CLIs."""
from __future__ import annotations


def parse_csv(value: str) -> list[str]:
    """Split a comma-separated string into stripped, non-empty items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    """Split a comma-separated string into ints, ignoring empty items."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]
