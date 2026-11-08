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
(CC0). Download it with the Kaggle CLI, using a `KAGGLE_API_TOKEN` in the
environment (the token is never stored in the repository):

```bash
KAGGLE_API_TOKEN=... .venv/bin/kaggle datasets download -d robikscube/hourly-energy-consumption -p data --unzip
```

The PJME zone is `data/PJME_hourly.csv` (columns `Datetime`, `PJME_MW`). The
data directory is git-ignored.

Repair policy in `load_hourly_csv` (causal, so no repaired value depends on future data):

- Duplicated timestamps (the repeated hour at the end of daylight saving time)
  keep the first record in file order. This affects 4 hours in the PJME file.
- Missing hours are forward-filled from the last observed value. This affects
  30 hours in the PJME file. The last of them (2018-03-11) is before the first
  evaluated test window (2018-06-29), so repairs do not enter any evaluation.
- NaN values in the file raise an error.

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
(model, fold) to `--out`.

## Results (PJME, horizon 168 h, 5 rolling-origin folds, 8-week training window)

| model | MASE mean (std) | sMAPE % mean (std) | MAE mean (std) | RMSE mean (std) |
|---|---|---|---|---|
| seasonal_naive_24 | **1.447** (0.747) | **11.62** (5.13) | **4342** (1966) | **5372** (2210) |
| harmonic_arima | 1.688 (1.266) | 13.63 (8.46) | 4977 (3257) | 5965 (3489) |
| harmonic_ols | 1.747 (1.404) | 14.13 (9.72) | 5131 (3639) | 5996 (3762) |
| seasonal_naive_168 | 1.890 (1.067) | 15.23 (6.96) | 5602 (2670) | 6601 (2763) |

On this data the harmonic models do not beat the daily seasonal naive. The
harmonic ARIMA beats the OLS harmonic model and the weekly seasonal naive. The
folds have high variance, so these averages should be read with the per-fold
CSV. Interpretation and next steps are not yet written up.

## Tests

- `test_fourier.py`, `test_features_schema.py`, `test_features_parity.py`:
  Fourier shapes, periodicity, serving-batch parity, spec agreement.
- `test_features_leakage.py`: future values do not change training or forecasts;
  training ends at the origin; shuffling time degrades the model.
- `test_data_pjm.py`: loader keeps the first duplicate, forward-fills gaps from the past only, and rejects NaNs.
- `test_harmonic.py`, `test_backtest_splits.py`, `test_metrics.py`: model
  outputs, fold geometry, metric definitions.

## Status

Implemented: method core, rolling-origin backtest, metrics, CLI, tests, CI.

## Scope

Out of scope for this project: Terraform, Floci, and architecture or method
diagrams. Not yet done:

- Results on PJME are reported above; the harmonic models do not yet beat the daily seasonal naive.
- LaTeX write-up.
- Order selection for ARIMA errors is fixed at (2,0,1); it is not searched.
