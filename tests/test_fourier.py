import numpy as np
import pandas as pd
import pytest

from forecaster.features.fourier import EPOCH, fourier_terms, hour_positions


def test_columns_and_shape():
    df = fourier_terms(np.arange(48), {24: 2, 168: 1})
    assert list(df.columns) == [
        "sin_24_1",
        "cos_24_1",
        "sin_24_2",
        "cos_24_2",
        "sin_168_1",
        "cos_168_1",
    ]
    assert len(df) == 48
    assert not df.isna().any().any()


def test_daily_term_repeats_every_24_hours():
    df = fourier_terms(np.arange(72), {24: 3})
    np.testing.assert_allclose(df.iloc[:24].to_numpy(), df.iloc[24:48].to_numpy(), atol=1e-12)


def test_hour_positions_are_hours_since_epoch():
    idx = pd.DatetimeIndex([EPOCH, EPOCH + pd.Timedelta(hours=25)])
    np.testing.assert_array_equal(hour_positions(idx), [0, 25])


@pytest.mark.parametrize("periods", [{24: 13}, {24: 0}, {1: 1}])
def test_invalid_spec_raises(periods):
    with pytest.raises(ValueError):
        fourier_terms(np.arange(10), periods)
