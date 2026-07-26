"""Small numeric-hygiene helpers shared by geometry/direction code."""
from __future__ import annotations

import numpy as np


def clean_matrix(x: np.ndarray) -> np.ndarray:
    """Sanitize an array: NaN/+inf/-inf -> 0.0, cast to float64."""
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
