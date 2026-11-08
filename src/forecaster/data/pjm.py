from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_hourly_csv(
    path: str | Path,
    value_column: str = "PJME_MW",
    time_column: str = "Datetime",
) -> pd.Series:
    """Load an hourly series with a fixed, documented repair policy.

    Duplicated timestamps (the repeated hour at the end of daylight saving time)
    keep the first record in file order. Missing hours are filled by linear
    interpolation over the full history, before any backtest split. Missing
    values inside the file (NaN) still raise.
    """
    frame = pd.read_csv(path, parse_dates=[time_column])
    series = frame.set_index(time_column)[value_column].astype(float)
    series = series.sort_index(kind="stable")
    series = series[~series.index.duplicated(keep="first")]

    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} missing values in {value_column}")

    hourly = series.asfreq("h")
    return hourly.interpolate(method="linear", limit_area="inside")
