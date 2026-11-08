from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_hourly_csv(
    path: str | Path,
    value_column: str = "PJME_MW",
    time_column: str = "Datetime",
) -> pd.Series:
    """Load an hourly series with a fixed, causal repair policy.

    Duplicated timestamps (the repeated hour at the end of daylight saving time)
    keep the first record in file order. Missing hours are forward-filled from
    the last observed value, so a repaired value never depends on later data.
    NaN values in the file raise an error.
    """
    frame = pd.read_csv(path, parse_dates=[time_column])
    series = frame.set_index(time_column)[value_column].astype(float)
    series = series.sort_index(kind="stable")
    series = series[~series.index.duplicated(keep="first")]

    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} missing values in {value_column}")

    return series.asfreq("h").ffill()
