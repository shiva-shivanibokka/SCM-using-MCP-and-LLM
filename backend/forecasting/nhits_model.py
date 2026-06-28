"""N-HiTS forecaster via neuralforecast (CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import validate_forecast


def forecast_nhits(history: list[float], horizon: int = 30) -> dict:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss

    n = len(history)
    input_size = min(max(2 * horizon, 30), n - 1)
    df = pd.DataFrame({
        "unique_id": "series",
        "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
        "y": [float(x) for x in history],
    })
    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MQLoss(level=[80]),
        max_steps=200,
        scaler_type="standard",
        enable_progress_bar=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    fc = nf.predict()
    # MQLoss with level=[80] yields NHITS-lo-80, NHITS-median, NHITS-hi-80
    lo = fc.filter(like="-lo-80").iloc[:, 0].to_numpy()
    mid = fc.filter(like="-median").iloc[:, 0].to_numpy()
    hi = fc.filter(like="-hi-80").iloc[:, 0].to_numpy()
    lo = np.maximum(lo, 0.0)
    mid = np.maximum(mid, lo)
    hi = np.maximum(hi, mid)
    result = {"p10": lo.tolist(), "p50": mid.tolist(), "p90": hi.tolist()}
    return validate_forecast(result, horizon)
