"""Intermittent-demand forecasting — Croston & TSB.

Many retail SKUs sell lumpily: long runs of zero-demand days punctuated by
occasional spikes. Standard ML/time-series models (Chronos, N-HiTS, CatBoost)
smear a smooth average across those zeros and forecast poorly. Croston's method
(and the TSB variant, which handles obsolescence) are the classic remedies —
they model demand SIZE and demand INTERVAL/PROBABILITY separately.

This module provides:
  - classify_demand()  — Syntetos-Boylan quadrant (smooth/erratic/intermittent/lumpy)
  - is_intermittent()  — routing predicate for the ensemble
  - croston_tsb_forecast() — a Croston/TSB forecast in the shared p10/p50/p90 contract

Ported in spirit from the pre-revamp HUFT forecast engine's Croston/TSB routing.
"""
from __future__ import annotations

import numpy as np

from .contract import validate_forecast

# Syntetos-Boylan cutoffs.
ADI_CUTOFF = 1.32   # average demand interval
CV2_CUTOFF = 0.49   # squared coefficient of variation of non-zero demand sizes


def classify_demand(history: list[float]) -> str:
    """Return one of: smooth, erratic, intermittent, lumpy, none."""
    y = np.asarray(history, dtype=float)
    if y.size == 0:
        return "none"
    nz = y[y > 0]
    if nz.size == 0:
        return "none"
    adi = y.size / nz.size
    mean = nz.mean()
    cv2 = (nz.std() / mean) ** 2 if mean > 0 else 0.0
    if adi < ADI_CUTOFF:
        return "erratic" if cv2 >= CV2_CUTOFF else "smooth"
    return "lumpy" if cv2 >= CV2_CUTOFF else "intermittent"


def is_intermittent(history: list[float]) -> bool:
    """True when the series is intermittent or lumpy (Croston/TSB territory)."""
    return classify_demand(history) in ("intermittent", "lumpy")


def _croston_rate(y: np.ndarray, alpha: float = 0.1) -> float:
    """Croston: exponentially-smoothed demand size / interval → per-period rate."""
    nz = np.flatnonzero(y)
    if nz.size == 0:
        return 0.0
    z = float(y[nz[0]])   # demand size estimate
    x = 1.0               # interval estimate
    q = 1                 # periods since last non-zero demand
    for t in range(nz[0] + 1, y.size):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            x += alpha * (q - x)
            q = 1
        else:
            q += 1
    return z / x if x > 0 else 0.0


def _tsb_rate(y: np.ndarray, alpha: float = 0.1, beta: float = 0.05) -> float:
    """TSB (Teunter-Syntetos-Babai): smooth demand size and demand PROBABILITY.
    Updates every period, so it decays the forecast for SKUs going obsolete."""
    n = y.size
    nz = np.flatnonzero(y)
    if nz.size == 0:
        return 0.0
    z = float(y[nz[0]])
    p = nz.size / n
    for t in range(n):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            p += beta * (1 - p)
        else:
            p += beta * (0 - p)
    return p * z


def croston_tsb_forecast(history: list[float], horizon: int = 30,
                         method: str | None = None) -> dict:
    """Flat-rate intermittent forecast with a demand-variability band, returned in
    the shared {p10, p50, p90} contract. `method` forces 'croston' or 'tsb';
    default picks TSB for lumpy series and Croston otherwise."""
    y = np.asarray(history, dtype=float)
    if method is None:
        method = "tsb" if classify_demand(history) == "lumpy" else "croston"
    rate = _tsb_rate(y) if method == "tsb" else _croston_rate(y)

    nz = y[y > 0]
    if nz.size > 0:
        p = nz.size / y.size
        mean_size = float(nz.mean())
        var_size = float(nz.var())
        # Variance of intermittent per-period demand ≈ p·E[size²] − (p·E[size])².
        per_period_var = p * (var_size + mean_size ** 2) - (p * mean_size) ** 2
        sd = float(np.sqrt(max(0.0, per_period_var)))
    else:
        sd = 0.0

    p50 = max(0.0, rate)
    p10 = max(0.0, p50 - 1.2816 * sd)
    p90 = p50 + 1.2816 * sd
    result = {
        "p10": [round(p10, 3)] * horizon,
        "p50": [round(p50, 3)] * horizon,
        "p90": [round(p90, 3)] * horizon,
        "method": method,
    }
    validate_forecast(result, horizon)
    return result
