from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from forecaster.features.fourier import fourier_terms, hour_positions


def _design(index: pd.DatetimeIndex, periods: Mapping[int, int]) -> np.ndarray:
    return fourier_terms(hour_positions(index), periods).to_numpy()


def _future_index(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")


def select_harmonic_arima(
    y: pd.Series,
    grid: Sequence[Mapping[int, int]],
    order: tuple[int, int, int] = (2, 0, 1),
) -> tuple[SARIMAXResults, Mapping[int, int]]:
    """Fit regression-with-ARIMA-errors for each Fourier spec and keep the lowest AIC."""
    values = np.asarray(y, dtype=float)
    index = pd.DatetimeIndex(y.index)
    best: tuple[SARIMAXResults, Mapping[int, int]] | None = None
    for periods in grid:
        model = SARIMAX(values, exog=_design(index, periods), order=order, trend="c")
        result = model.fit(disp=False, maxiter=200)
        if best is None or result.aic < best[0].aic:
            best = (result, periods)
    if best is None:
        raise ValueError("grid must contain at least one Fourier specification")
    return best


def fit_predict_harmonic_arima(
    y: pd.Series,
    horizon: int,
    grid: Sequence[Mapping[int, int]],
    order: tuple[int, int, int] = (2, 0, 1),
) -> np.ndarray:
    """Forecast ``horizon`` steps ahead from the end of the training window ``y``."""
    result, periods = select_harmonic_arima(y, grid, order)
    future_exog = _design(_future_index(pd.DatetimeIndex(y.index), horizon), periods)
    return np.asarray(result.get_forecast(steps=horizon, exog=future_exog).predicted_mean)


def fit_predict_harmonic_ols(
    y: pd.Series,
    horizon: int,
    grid: Sequence[Mapping[int, int]],
) -> np.ndarray:
    """Harmonic regression without ARIMA errors, using the same AIC grid search."""
    values = np.asarray(y, dtype=float)
    index = pd.DatetimeIndex(y.index)
    best = None
    for periods in grid:
        design = sm.add_constant(_design(index, periods), has_constant="add")
        result = sm.OLS(values, design).fit()
        if best is None or result.aic < best[0].aic:
            best = (result, periods)
    if best is None:
        raise ValueError("grid must contain at least one Fourier specification")
    result, periods = best
    future = sm.add_constant(
        _design(_future_index(index, horizon), periods), has_constant="add"
    )
    return np.asarray(result.predict(future))
