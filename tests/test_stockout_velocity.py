"""Regression test: a one-day demand spike must not make a SKU 'critical'.

Bug (pre-fix intelligence/stockout.py): velocity = the latest single day's
demand, so a SKU with a steady ~75/day that spikes to 300 on the last day was
classified critical. Velocity is now a trailing mean (default 28 days).
"""
import pandas as pd

from intelligence.stockout import predict_stockouts


def _history(days=30, base=75, spike=300, inventory=400, lead=3):
    dates = pd.date_range("2025-12-02", periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "store_id": "S1", "sku_id": "A", "name": "Prod A",
                     "category": "Grooming", "brand": "X",
                     "demand": spike if i == days - 1 else base,
                     "inventory": inventory, "lead_time_days": lead})
    return pd.DataFrame(rows)


def test_single_day_spike_is_not_critical():
    r = predict_stockouts(_history())
    row = r["rows"][0]
    # ~400 units / ~83 per day (28-day mean incl. spike) ≈ 4.8 days > 3-day lead time
    assert row["risk"] != "critical", row
    assert 75 <= row["daily_velocity"] <= 90, row


def test_window_one_reproduces_old_behaviour():
    r = predict_stockouts(_history(), velocity_window_days=1)
    assert r["rows"][0]["daily_velocity"] == 300.0
    assert r["rows"][0]["risk"] == "critical"


def test_single_snapshot_falls_back_to_snapshot_demand():
    snap = _history().tail(1)
    r = predict_stockouts(snap)
    assert r["rows"][0]["daily_velocity"] == 300.0
