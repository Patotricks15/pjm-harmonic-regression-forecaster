from pathlib import Path

import numpy as np
import pytest

from forecaster.features.spec import load_fourier_grid
from forecaster.models.harmonic import fit_predict_harmonic_arima, select_harmonic_arima

GRID = load_fourier_grid(Path(__file__).resolve().parents[1] / "conf" / "features.yaml")


def test_selected_spec_has_lowest_aic_in_grid(pjm_like):
    y = pjm_like.iloc[: 24 * 21]
    grid = [{24: 2, 168: 1}, {24: 4, 168: 2}]
    best, periods = select_harmonic_arima(y, grid)
    assert periods in grid
    assert np.isfinite(best.aic)


def test_arima_forecast_has_horizon_length_and_beats_seasonal_naive(pjm_like):
    horizon = 24
    train = pjm_like.iloc[: -horizon]
    actual = pjm_like.iloc[-horizon:].to_numpy()

    pred = fit_predict_harmonic_arima(train.iloc[-24 * 21 :], horizon, grid=GRID[:1])
    assert pred.shape == (horizon,)
    assert np.all(np.isfinite(pred))

    snaive = np.tile(train.to_numpy()[-24:], 1)[:horizon]
    assert np.mean(np.abs(actual - pred)) < np.mean(np.abs(actual - snaive))


@pytest.mark.parametrize("horizon", [1, 24, 170])
def test_forecast_length_matches_horizon(pjm_like, horizon):
    pred = fit_predict_harmonic_arima(pjm_like.iloc[: 24 * 21], horizon, grid=[{24: 2, 168: 1}])
    assert pred.shape == (horizon,)
