import numpy as np
import pytest

from forecaster.evaluation.metrics import mae, mase, rmse, smape


def test_mase_of_naive_forecast_on_train_is_one_at_m1():
    y_train = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y_true = np.array([16.0, 22.0])
    y_pred = np.array([11.0, 11.0])
    expected = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))
    assert mase(y_true, y_pred, y_train, m=1) == pytest.approx(expected)


def test_mase_zero_scale_raises():
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), np.array([3.0] * 30), m=1)


def test_smape_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_mae_and_rmse_known_values():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 1.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
