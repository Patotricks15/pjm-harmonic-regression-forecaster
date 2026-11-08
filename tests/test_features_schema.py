from pathlib import Path

import pytest

from forecaster.features.fourier import fourier_terms
from forecaster.features.spec import load_fourier_grid

SPEC = Path(__file__).resolve().parents[1] / "conf" / "features.yaml"


def test_grid_is_full_cartesian_product():
    grid = load_fourier_grid(SPEC)
    assert len(grid) == 3 * 3
    assert all(set(combo) == {24, 168} for combo in grid)
    assert {combo[24] for combo in grid} == {2, 4, 6}
    assert {combo[168] for combo in grid} == {1, 2, 3}


def test_declared_columns_match_generated_columns():
    for combo in load_fourier_grid(SPEC):
        expected = [
            f"{fn}_{m}_{k}"
            for m, k_max in combo.items()
            for k in range(1, k_max + 1)
            for fn in ("sin", "cos")
        ]
        generated = fourier_terms(range(5), combo).columns.tolist()
        assert generated == expected


def test_duplicate_period_raises(tmp_path):
    spec = tmp_path / "features.yaml"
    spec.write_text(
        "seasonal_fourier:\n"
        "  - period: 24\n    harmonics: [1]\n"
        "  - period: 24\n    harmonics: [2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_fourier_grid(spec)
