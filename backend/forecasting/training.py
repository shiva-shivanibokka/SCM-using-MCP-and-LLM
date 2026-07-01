"""On-demand CatBoost retraining + backtest.

This is what the MLOps "Trigger fine-tune" button actually runs: it pulls the
latest demand for the busiest SKUs, retrains CatBoost, and backtests it by
holding out the most recent `horizon` days and measuring sMAPE against actuals.
The resulting score is written to the model-registry logbook (registry.py).

sMAPE (symmetric mean absolute percentage error) is used instead of MAPE because
demand has zero-days, which make plain MAPE divide by zero.
"""
from __future__ import annotations

import logging

import numpy as np

from ..data_access import load_demand
from .catboost_model import _LAGS, _fit_quantile, _make_supervised

logger = logging.getLogger(__name__)


def _smape(actual, forecast) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = np.abs(a) + np.abs(f)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(f - a)[mask] / denom[mask]) * 100.0)


def _top_skus(n: int) -> list[str]:
    df = load_demand()
    totals = df.groupby("sku_id")["demand"].sum().sort_values(ascending=False)
    return [str(s) for s in totals.head(n).index]


def _series(sku_id: str) -> np.ndarray:
    df = load_demand()
    s = (df[df["sku_id"] == sku_id]
         .sort_values("date")
         .groupby("date")["demand"].sum())
    return s.to_numpy(dtype=float)


def _backtest_one(series: np.ndarray, horizon: int) -> float | None:
    """Train on all-but-last-`horizon` days, forecast, score sMAPE vs actuals."""
    if len(series) < max(_LAGS) + horizon + 10:
        return None
    train, test = series[:-horizon], series[-horizon:]
    X, y = _make_supervised(train)
    model = _fit_quantile(X, y, 0.5)  # median is enough to score accuracy

    working = list(train)
    preds = []
    for _ in range(horizon):
        feat = np.array([[working[-lag] for lag in _LAGS]], dtype=float)
        p = max(float(model.predict(feat)[0]), 0.0)
        preds.append(p)
        working.append(p)
    return _smape(test, preds)


def retrain_catboost(n_skus: int = 8, horizon: int = 30) -> dict:
    """Retrain + backtest CatBoost, then train and DURABLY persist the served model.

    Two steps:
      1. A fast, leak-free backtest (train on all-but-last-`horizon`, score sMAPE)
         for the registry metric.
      2. Train the persistent CatBoost (leak-free temporal split) and save it to
         forecasting/.model_cache *and* to Neon (artifact_store), so the fine-tune
         actually produces served weights that survive an HF Space restart.
    """
    skus = _top_skus(n_skus)
    scores: list[float] = []
    rows = 0
    for sku in skus:
        series = _series(sku)
        rows += int(len(series))
        score = _backtest_one(series, horizon)
        if score is not None:
            scores.append(score)
    logger.info("retrain_catboost: %d/%d SKUs scored, %d rows", len(scores), len(skus), rows)

    # Step 2 — train + durably persist the served model (best-effort). Call the
    # CatBoost path directly: the generic train() prefers TFT (20-40 min), far too
    # slow for a button; CatBoost trains in ~1 min, leak-free, and _save_catboost
    # persists it to .model_cache + Neon.
    weights_persisted = False
    val_mape = None
    try:
        from forecasting.ml_forecast import _train_catboost, is_trained

        metrics = _train_catboost(load_demand())
        val_mape = metrics.get("mape")
        weights_persisted = is_trained()
    except Exception as e:  # never fail the backtest/registry log over this
        logger.warning("durable train/persist failed: %s", e)

    return {
        "model_name": "catboost",
        "backtest_smape": round(float(np.mean(scores)), 2) if scores else None,
        "val_mape": round(float(val_mape), 2) if val_mape is not None else None,
        "n_skus": len(scores),
        "training_rows": rows,
        "horizon": horizon,
        "weights_persisted": weights_persisted,
    }
