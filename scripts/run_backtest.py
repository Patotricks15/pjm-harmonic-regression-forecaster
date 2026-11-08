"""Run the TS02 rolling-origin backtest on a PJM hourly CSV."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from forecaster.data.pjm import load_hourly_csv
from forecaster.evaluation.run import evaluate_forecasters
from forecaster.features.spec import load_fourier_grid
from forecaster.models.baselines import seasonal_naive
from forecaster.models.harmonic import fit_predict_harmonic_arima, fit_predict_harmonic_ols

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to PJME_hourly.csv")
    parser.add_argument("--column", default="PJME_MW")
    parser.add_argument("--horizon", type=int, default=168, help="Forecast horizon in hours")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--window-hours",
        type=int,
        default=8 * 168,
        help="Training window length in hours (ARIMA fits are costly on long histories)",
    )
    parser.add_argument("--spec", type=Path, default=ROOT / "conf" / "features.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "backtest.csv")
    args = parser.parse_args()

    series = load_hourly_csv(args.csv, args.column)
    grid = load_fourier_grid(args.spec)
    forecasters = {
        "seasonal_naive_24": partial(seasonal_naive, m=24),
        "seasonal_naive_168": partial(seasonal_naive, m=168),
        "harmonic_ols": partial(fit_predict_harmonic_ols, grid=grid),
        "harmonic_arima": partial(fit_predict_harmonic_arima, grid=grid),
    }

    results = evaluate_forecasters(
        series,
        forecasters,
        horizon=args.horizon,
        n_folds=args.folds,
        window=args.window_hours,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    summary = results.groupby("model")[["mase", "smape", "mae", "rmse"]].agg(["mean", "std"])
    print(summary.round(3).to_string())
    print(f"\nPer-fold results written to {args.out}")


if __name__ == "__main__":
    main()
