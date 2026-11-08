import numpy as np
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


def test_duplicate_timestamp_keeps_first_record(tmp_path):
    ts = ["2014-11-02 00:00:00", "2014-11-02 01:00:00", "2014-11-02 01:00:00", "2014-11-02 02:00:00"]
    series = load_hourly_csv(_write(tmp_path, ts, [1.0, 2.0, 9.0, 3.0]))
    assert len(series) == 3
    assert series.iloc[1] == 2.0


def test_missing_hour_is_forward_filled_from_past_only(tmp_path):
    ts = ["2015-01-01 00:00:00", "2015-01-01 01:00:00", "2015-01-01 03:00:00"]
    series = load_hourly_csv(_write(tmp_path, ts, [10.0, 20.0, 40.0]))
    assert series.index.freqstr == "h"
    assert series.iloc[2] == pytest.approx(20.0)


def test_repair_never_uses_future_values(tmp_path):
    ts = [
        "2015-01-01 00:00:00",
        "2015-01-01 01:00:00",
        "2015-01-01 03:00:00",
        "2015-01-01 04:00:00",
    ]
    base = load_hourly_csv(_write(tmp_path, ts, [10.0, 20.0, 40.0, 50.0]))
    changed = load_hourly_csv(_write(tmp_path, ts, [10.0, 20.0, 40.0, 999.0]))
    pd.testing.assert_series_equal(base.iloc[:3], changed.iloc[:3])



def test_nan_values_raise(tmp_path):
    ts = ["2015-01-01 00:00:00", "2015-01-01 01:00:00"]
    with pytest.raises(ValueError, match="missing values"):
        load_hourly_csv(_write(tmp_path, ts, [1.0, np.nan]))
