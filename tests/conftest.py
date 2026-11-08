import numpy as np
import pandas as pd
import pytest


def make_pjm_like(n_days: int = 56, seed: int = 0) -> pd.Series:
    """Hourly synthetic load with daily and weekly cycles plus AR(1) noise."""
    n = 24 * n_days
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 150.0, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    values = (
        30000.0
        + 4000.0 * np.sin(2 * np.pi * t / 24)
        + 1500.0 * np.cos(2 * np.pi * t / 24)
        + 2500.0 * np.sin(2 * np.pi * t / 168)
        + noise
    )
    index = pd.date_range("2015-01-01", periods=n, freq="h")
    return pd.Series(values, index=index, name="PJME_MW")


@pytest.fixture
def pjm_like() -> pd.Series:
    return make_pjm_like()
