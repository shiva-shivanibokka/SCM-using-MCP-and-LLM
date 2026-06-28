"""Frozen-weight ensemble of Chronos + N-HiTS + CatBoost."""
from __future__ import annotations

import logging
import numpy as np

from .contract import validate_forecast
from .chronos_model import forecast_chronos
from .nhits_model import forecast_nhits
from .catboost_model import forecast_catboost

logger = logging.getLogger(__name__)

# (chronos, nhits, catboost) — frozen between quarterly fine-tunes
ENSEMBLE_WEIGHTS = (0.5, 0.35, 0.15)
_NAMES = ("chronos", "nhits", "catboost")


def ensemble_forecast(history: list[float], horizon: int = 30,
                      components: bool = False) -> dict:
    funcs = (forecast_chronos, forecast_nhits, forecast_catboost)
    got, weights, comp_out = [], [], {}
    for name, w, fn in zip(_NAMES, ENSEMBLE_WEIGHTS, funcs):
        try:
            r = fn(history, horizon)
            got.append(r)
            weights.append(w)
            comp_out[name] = r
        except Exception as e:  # degrade gracefully
            logger.warning("ensemble component %s failed: %s", name, e)
    if not got:
        raise RuntimeError("all ensemble components failed")

    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    blended = {}
    for q in ("p10", "p50", "p90"):
        stacked = np.array([r[q] for r in got])          # (k, horizon)
        blended[q] = (np.array(weights) @ stacked).tolist()
    # re-assert ordering after blending
    p10 = np.array(blended["p10"])
    p50 = np.maximum(blended["p50"], p10)
    p90 = np.maximum(blended["p90"], p50)
    result = {"p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist()}
    validate_forecast(result, horizon)
    if components:
        result["components"] = comp_out
    return result
