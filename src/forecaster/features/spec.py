from __future__ import annotations

import itertools
from pathlib import Path

import yaml


def load_fourier_grid(path: str | Path) -> list[dict[int, int]]:
    """Expand ``seasonal_fourier`` in the feature spec into all harmonic combinations."""
    with Path(path).open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    entries = doc["seasonal_fourier"]
    periods = [int(entry["period"]) for entry in entries]
    if len(set(periods)) != len(periods):
        raise ValueError(f"duplicate periods in {path}")

    choices = [[int(k) for k in entry["harmonics"]] for entry in entries]
    return [dict(zip(periods, combo)) for combo in itertools.product(*choices)]
