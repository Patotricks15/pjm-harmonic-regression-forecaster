from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.evaluation.backtest import rolling_origin_folds
from forecaster.evaluation.metrics import mae, mase, rmse, smape

Forecaster = Callable[[pd.Series, int], np.ndarray]


@dataclass(frozen=True)
class FoldData:
    fold: int
    origin: pd.Timestamp  # timestamp of the last training observation
    train: pd.Series  # only observations strictly before the test window
    actual: np.ndarray


def iter_folds(
    series: pd.Series,
    horizon: int,
    n_folds: int,
    window: int | None = None,
) -> Iterator[FoldData]:
    """Yield each fold's training window (optionally truncated) and test actuals."""
    for i, fold in enumerate(rolling_origin_folds(len(series), horizon, n_folds)):
        start = 0 if window is None else max(0, fold.train_end - window)
        yield FoldData(
            fold=i,
            origin=series.index[fold.train_end - 1],
            train=series.iloc[start : fold.train_end],
            actual=series.iloc[fold.test_start : fold.test_end].to_numpy(dtype=float),
        )


def evaluate_forecasters(
    series: pd.Series,
    forecasters: Mapping[str, Forecaster],
    horizon: int,
    n_folds: int,
    window: int | None = None,
    m: int = 24,
) -> pd.DataFrame:
    """Return one row per (model, fold) with scale-free and scale-dependent errors."""
    rows: list[dict] = []
    for fd in iter_folds(series, horizon, n_folds, window):
        train_values = fd.train.to_numpy(dtype=float)
        for name, forecast in forecasters.items():
            pred = np.asarray(forecast(fd.train, horizon), dtype=float)
            if pred.shape != (horizon,) or not np.all(np.isfinite(pred)):
                raise ValueError(f"{name} returned an invalid forecast in fold {fd.fold}")
            rows.append(
                {
                    "model": name,
                    "fold": fd.fold,
                    "origin": fd.origin,
                    "mase": mase(fd.actual, pred, train_values, m=m),
                    "smape": smape(fd.actual, pred),
                    "mae": mae(fd.actual, pred),
                    "rmse": rmse(fd.actual, pred),
                }
            )
    return pd.DataFrame(rows)
