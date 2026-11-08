from __future__ import annotations

import numpy as np


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 24) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal naive error."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        raise ValueError("y_train must be longer than the seasonal period m")
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        raise ValueError("in-sample naive error is zero; MASE is undefined")
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))) / scale)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent, as used in the M4 competition."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))
