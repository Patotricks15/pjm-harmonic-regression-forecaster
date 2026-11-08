from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_hourly_csv(
    path: str | Path,
    value_column: str = "PJME_MW",
    time_column: str = "Datetime",
) -> pd.Series:
    """Load an hourly series and fail fast on any structural problem.

    The loader does not impute. Duplicated or missing hours raise, so no
    interpolation can leak future values into the training windows.
    """
    frame = pd.read_csv(path, parse_dates=[time_column])
    series = frame.set_index(time_column)[value_column].astype(float).sort_index()

    if series.index.has_duplicates:
        duplicated = int(series.index.duplicated().sum())
        raise ValueError(f"{duplicated} duplicated timestamps; deduplicate before use")
    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} missing values in {value_column}")

    expected = pd.date_range(series.index[0], series.index[-1], freq="h")
    missing = expected.difference(series.index)
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} missing hourly timestamps; first: {missing[0]}")

    return series.asfreq("h")
