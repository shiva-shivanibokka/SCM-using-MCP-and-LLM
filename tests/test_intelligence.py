"""Unit tests for the ported intelligence engines and intermittent forecasting."""
import pandas as pd

from intelligence.stockout import predict_stockouts
from intelligence.anomaly import run_anomaly_detection
from intelligence.whatif import simulate_discount_impact, simulate_restock_impact
from backend.forecasting.intermittent import (
    classify_demand, is_intermittent, croston_tsb_forecast,
)
from backend.forecasting.contract import validate_forecast


def _inv():
    # Two SKUs across two stores, latest snapshot.
    return pd.DataFrame([
        {"date": "2025-12-31", "store_id": "S1", "sku_id": "A", "name": "Prod A",
         "category": "Food", "brand": "X", "inventory": 10, "demand": 20,
         "lead_time_days": 5, "days_of_supply": 0.5, "risk_status": "CRITICAL",
         "price_inr": 100, "cost_inr": 60},
        {"date": "2025-12-31", "store_id": "S2", "sku_id": "B", "name": "Prod B",
         "category": "Toys", "brand": "Y", "inventory": 5000, "demand": 2,
         "lead_time_days": 7, "days_of_supply": 2500, "risk_status": "OK",
         "price_inr": 200, "cost_inr": 120},
    ])


def test_stockout_buckets_and_reorder():
    r = predict_stockouts(_inv())
    rows = {x["sku_id"]: x for x in r["rows"]}
    assert rows["A"]["risk"] == "critical"        # 10 units / 20 per day = 0.5 days
    assert rows["A"]["reorder_qty"] > 0
    assert rows["B"]["risk"] == "excess"          # 2500 days of cover
    assert r["summary"]["total_skus"] == 2


def test_anomaly_detects_velocity_risk():
    txn = pd.DataFrame(columns=["date", "sku_id", "net_revenue_inr", "quantity",
                                "unit_price_inr", "discount_pct", "channel", "txn_id"])
    out = run_anomaly_detection(txn, _inv())
    types = {a["type"] for a in out["anomalies"]}
    assert "Velocity-vs-stock risk" in types      # SKU A has <14 days cover
    assert out["summary"]["total"] >= 1


def test_whatif_discount_shape():
    txn = pd.DataFrame({
        "category": ["Food"] * 6,
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-06-01",
                                "2025-06-02", "2025-12-01", "2025-12-02"]),
        "sku_id": ["A"] * 6,
        "quantity": [10, 12, 20, 22, 30, 28],
        "unit_price_inr": [100] * 6,
        "discount_pct": [0, 5, 20, 25, 40, 35],
        "net_revenue_inr": [1000, 1140, 1600, 1650, 1800, 1820],
    })
    r = simulate_discount_impact(txn, category="Food", new_discount_pct=20)
    assert -1.8 <= r["elasticity"] <= -0.5
    assert "baseline" in r and "projected" in r and "delta" in r


def test_whatif_restock_roi():
    products = pd.DataFrame([{"sku_id": "A", "name": "Prod A",
                             "price_inr": 100, "cost_inr": 60}])
    r = simulate_restock_impact(_inv(), products, "A", 500)
    assert r["economics"]["restock_cost"] == 500 * 60
    assert r["economics"]["gross_profit"] == 500 * 40
    assert r["projected"]["inventory"] == 10 + 500


def test_intermittent_classification_and_contract():
    interm = [0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 4, 0, 0, 6] * 5
    smooth = [50, 52, 48, 51, 49, 50, 53, 47] * 10
    assert is_intermittent(interm) is True
    assert is_intermittent(smooth) is False
    assert classify_demand(smooth) == "smooth"
    f = croston_tsb_forecast(interm, horizon=30)
    validate_forecast(f, 30)                       # raises if contract violated
    assert f["method"] in ("croston", "tsb")
    assert all(f["p10"][i] <= f["p50"][i] <= f["p90"][i] for i in range(30))
