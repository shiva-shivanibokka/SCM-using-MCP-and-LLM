"""Forecast output contract shared by all forecasters."""
from __future__ import annotations

REQUIRED_KEYS = ("p10", "p50", "p90")


def validate_forecast(result: dict, horizon: int) -> dict:
    for k in REQUIRED_KEYS:
        if k not in result:
            raise ValueError(f"forecast missing key: {k}")
        if len(result[k]) != horizon:
            raise ValueError(
                f"forecast['{k}'] has length {len(result[k])}, expected {horizon}"
            )
    for i in range(horizon):
        lo, mid, hi = result["p10"][i], result["p50"][i], result["p90"][i]
        if not (lo <= mid <= hi):
            raise ValueError(
                f"quantiles unordered at step {i}: p10={lo} p50={mid} p90={hi}"
            )
    return result
