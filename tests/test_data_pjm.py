import pandas as pd
import pytest

from forecaster.data.pjm import load_hourly_csv


def _write(tmp_path, timestamps, values=None):
    values = values if values is not None else list(range(len(timestamps)))
    path = tmp_path / "PJME_hourly.csv"
    pd.DataFrame({"Datetime": timestamps, "PJME_MW": values}).to_csv(path, index=False)
    return path


def test_loads_regular_hourly_series(tmp_path):
    ts = pd.date_range("2015-01-01", periods=10, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    series = load_hourly_csv(_write(tmp_path, list(ts)))
    assert len(series) == 10
    assert series.index.freqstr == "h"


def test_duplicate_timestamps_raise(tmp_path):
    ts = ["2015-01-01 00:00:00", "2015-01-01 01:00:00", "2015-01-01 01:00:00"]
    with pytest.raises(ValueError, match="duplicated"):
        load_hourly_csv(_write(tmp_path, ts))


def test_missing_hours_raise_without_imputation(tmp_path):
    ts = ["2015-01-01 00:00:00", "2015-01-01 01:00:00", "2015-01-01 03:00:00"]
    with pytest.raises(ValueError, match="missing hourly"):
        load_hourly_csv(_write(tmp_path, ts))
