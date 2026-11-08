# PJM Harmonic Regression Forecaster (TS02)

Hourly electricity-load forecasting on PJM (PJME zone) with a classical method:
harmonic regression with ARIMA errors, where the Fourier order is chosen by AIC.

- Horizon: 168 hours (7 days), hourly frequency.
- Evaluation: rolling-origin backtests (5 folds, step = horizon, training window
  truncated to the last 8 weeks).
- Metrics: MASE (m = 24), sMAPE, MAE, RMSE.
- Baselines: seasonal naive (m = 24 and m = 168), harmonic regression by OLS.

## Method

Let $y_t$ be the load and $x_t$ the Fourier regressors for the periods in
`conf/features.yaml` (daily $m=24$ with $K \in \{2,4,6\}$, weekly $m=168$ with
$K \in \{1,2,3\}$):

$$
y_t = \mu + \sum_{m} \sum_{k=1}^{K_m} \left(a_{m,k}\sin\tfrac{2\pi k t}{m} + b_{m,k}\cos\tfrac{2\pi k t}{m}\right) + \eta_t,
\qquad \eta_t \sim \text{ARIMA}(2,0,1).
$$

The regression is fitted as a SARIMAX model with exogenous Fourier terms. For
each candidate $(K_{24}, K_{168})$ the model is fitted and the spec with the
lowest AIC wins. Fourier positions are whole hours since a fixed epoch
(2000-01-01), so features depend only on timestamps and are identical in
training and forecasting.

## Data

The real series is PJM hourly load from Kaggle,
[`robikscube/hourly-energy-consumption`](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
(CC0). Download `PJME_hourly.csv` and pass its path to the CLI. This repository
does not ship the data.

The loader is strict and does not impute: duplicate timestamps, NaNs, and
missing hours raise an error. Daylight-saving fall-back can create duplicated
local timestamps in this file; resolving them is a pending decision (see below).

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Tests and dependency gate (same as CI)
.venv/bin/python scripts/check_neural_deps.py
.venv/bin/python -m pytest

# Backtest on the real file
.venv/bin/python scripts/run_backtest.py --csv data/PJME_hourly.csv \
    --horizon 168 --folds 5 --window-hours 1344 --out outputs/backtest.csv
```

The CLI prints a mean/std summary per model and writes one row per
(model, fold) to `--out`. Expect the harmonic ARIMA runs to be slow: each fold
fits 9 SARIMAX models for the AIC grid.

## Tests

- `test_fourier.py`, `test_features_schema.py`, `test_features_parity.py`:
  Fourier shapes, periodicity, serving-batch parity, spec agreement.
- `test_features_leakage.py`: future values do not change training or forecasts;
  training ends at the origin; shuffling time degrades the model.
- `test_data_pjm.py`: loader rejects duplicates and gaps.
- `test_harmonic.py`, `test_backtest_splits.py`, `test_metrics.py`: model
  outputs, fold geometry, metric definitions.

## Status

Implemented: method core, rolling-origin backtest, metrics, CLI, tests, CI.

Not yet done (from the project plan's definition of done):

- Terraform and Floci setup.
- Architecture and method diagrams.
- LaTeX write-up.
- Results on the real PJM data (the file is not in the repository).
- Decision on DST duplicate timestamps.
- Order selection for ARIMA errors is fixed at (2,0,1); it is not searched.
