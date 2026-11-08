import numpy as np
import pandas as pd

from forecaster.evaluation.run import evaluate_forecasters, iter_folds
from forecaster.features.spec import load_fourier_grid
from forecaster.models.baselines import seasonal_naive
from forecaster.models.harmonic import fit_predict_harmonic_ols
from pathlib import Path

GRID = load_fourier_grid(Path(__file__).resolve().parents[1] / "conf" / "features.yaml")
HORIZON = 24
N_FOLDS = 4
WINDOW = 24 * 28


def test_future_injection_does_not_change_training_or_forecasts(pjm_like):
    # Overwrite only the final test window; every fold trains strictly before it.
    corrupted = pjm_like.copy()
    rng = np.random.default_rng(1)
    corrupted.iloc[-HORIZON:] = rng.normal(1e6, 1e5, HORIZON)

    def forecast(y, h):
        return fit_predict_harmonic_ols(y, h, grid=GRID)

    for clean, dirty in zip(
        iter_folds(pjm_like, HORIZON, N_FOLDS, WINDOW),
        iter_folds(corrupted, HORIZON, N_FOLDS, WINDOW),
        strict=True,
    ):
        pd.testing.assert_series_equal(clean.train, dirty.train)
        np.testing.assert_allclose(forecast(clean.train, HORIZON), forecast(dirty.train, HORIZON))


def test_training_window_ends_before_the_test_window(pjm_like):
    for fd in iter_folds(pjm_like, HORIZON, N_FOLDS, WINDOW):
        next_hour = fd.origin + pd.Timedelta(hours=1)
        assert next_hour in pjm_like.index
        assert fd.train.index.max() == fd.origin
        assert len(fd.actual) == HORIZON


def test_shuffled_time_removes_the_model_advantage(pjm_like):
    shuffled = pjm_like.copy()
    shuffled[:] = np.random.default_rng(2).permutation(pjm_like.to_numpy())

    forecasters = {
        "harmonic_ols": lambda y, h: fit_predict_harmonic_ols(y, h, grid=GRID),
        "seasonal_naive": lambda y, h: seasonal_naive(y, h, m=24),
    }
    original = evaluate_forecasters(pjm_like, forecasters, HORIZON, N_FOLDS, WINDOW)
    scrambled = evaluate_forecasters(shuffled, forecasters, HORIZON, N_FOLDS, WINDOW)

    mase_orig = original.groupby("model")["mase"].mean()
    mase_shuf = scrambled.groupby("model")["mase"].mean()

    assert mase_orig["harmonic_ols"] < mase_orig["seasonal_naive"]
    assert mase_shuf["harmonic_ols"] > 2 * mase_orig["harmonic_ols"]
