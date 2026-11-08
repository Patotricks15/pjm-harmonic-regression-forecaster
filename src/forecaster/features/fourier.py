from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Fixed reference so that feature rows depend only on the timestamp.
EPOCH = pd.Timestamp("2000-01-01")


def hour_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Whole hours elapsed since EPOCH for each timestamp in ``index``."""
    elapsed = (pd.DatetimeIndex(index) - EPOCH) / pd.Timedelta(hours=1)
    return np.asarray(elapsed).round().astype(np.int64)


def fourier_terms(positions: np.ndarray, periods: Mapping[int, int]) -> pd.DataFrame:
    """Sin/cos columns for each seasonal period ``m`` and harmonic ``k <= K``.

    ``periods`` maps the period ``m`` to the maximum harmonic ``K``. Positions
    are absolute, so training and serving produce identical rows.
    """
    t = np.asarray(positions, dtype=float)
    columns: dict[str, np.ndarray] = {}
    for m, k_max in periods.items():
        if m < 2 or k_max < 1 or k_max > m // 2:
            raise ValueError(f"invalid Fourier spec: period={m}, harmonics={k_max}")
        for k in range(1, k_max + 1):
            angle = 2.0 * np.pi * k * t / m
            columns[f"sin_{m}_{k}"] = np.sin(angle)
            columns[f"cos_{m}_{k}"] = np.cos(angle)
    return pd.DataFrame(columns, index=pd.RangeIndex(len(t)))
