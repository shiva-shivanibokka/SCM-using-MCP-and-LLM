"""CatBoost quantile baseline forecaster (CPU, always available)."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

from .contract import validate_forecast

_LAGS = (1, 2, 3, 7, 14, 28)


def _make_supervised(series: np.ndarray):
    rows, targets = [], []
    max_lag = max(_LAGS)
    for t in range(max_lag, len(series)):
        rows.append([series[t - lag] for lag in _LAGS])
        targets.append(series[t])
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def _fit_quantile(X, y, alpha):
    m = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=200, depth=4, learning_rate=0.1, verbose=False,
    )
    m.fit(X, y)
    return m


def forecast_catboost(history: list[float], horizon: int = 30) -> dict:
    series = np.asarray([float(x) for x in history], dtype=float)
    X, y = _make_supervised(series)
    models = {a: _fit_quantile(X, y, a) for a in (0.1, 0.5, 0.9)}

    preds = {0.1: [], 0.5: [], 0.9: []}
    working = list(series)
    for _ in range(horizon):
        feat = np.array([[working[-lag] for lag in _LAGS]], dtype=float)
        for a in (0.1, 0.5, 0.9):
            preds[a].append(float(models[a].predict(feat)[0]))
        working.append(preds[0.5][-1])  # roll forward on median

    p10 = np.maximum(preds[0.1], 0.0)
    p50 = np.maximum(preds[0.5], p10)
    p90 = np.maximum(preds[0.9], p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    return validate_forecast(result, horizon)
