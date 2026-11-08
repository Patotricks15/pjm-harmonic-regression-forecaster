from pathlib import Path

import numpy as np
import pandas as pd

from forecaster.features.fourier import fourier_terms, hour_positions
from forecaster.features.spec import load_fourier_grid

SPEC = Path(__file__).resolve().parents[1] / "conf" / "features.yaml"


def test_batch_rows_match_single_timestamp_serving_rows(pjm_like):
    combo = load_fourier_grid(SPEC)[0]
    full = fourier_terms(hour_positions(pjm_like.index), combo)

    serving_idx = pjm_like.index[[100, 500, 1000]]
    served = fourier_terms(hour_positions(serving_idx), combo)

    np.testing.assert_allclose(served.to_numpy(), full.iloc[[100, 500, 1000]].to_numpy())


def test_slice_features_equal_full_features_slice(pjm_like):
    combo = load_fourier_grid(SPEC)[-1]
    full = fourier_terms(hour_positions(pjm_like.index), combo)
    part = fourier_terms(hour_positions(pjm_like.index[200:260]), combo)
    pd.testing.assert_frame_equal(part, full.iloc[200:260].reset_index(drop=True))
