from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(y: pd.Series, horizon: int, m: int = 24) -> np.ndarray:
    """Repeat the last full seasonal cycle of length ``m``."""
    values = np.asarray(y, dtype=float)
    if len(values) < m:
        raise ValueError(f"training window ({len(values)}) is shorter than m={m}")
    repeats = int(np.ceil(horizon / m))
    return np.tile(values[-m:], repeats)[:horizon]
