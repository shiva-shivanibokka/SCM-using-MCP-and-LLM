from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_executive_kpis():
    r = client.get("/api/executive/kpis")
    assert r.status_code == 200
    body = r.json()
    for k in ("revenue", "gross_margin_pct", "revenue_by_region", "top_skus"):
        assert k in body
    assert body["revenue"] > 0  # real revenue from transactions
    assert len(body["revenue_by_region"]) > 0


def test_inventory_health():
    r = client.get("/api/inventory/health")
    assert r.status_code == 200
    for k in ("reorder_list", "stockout_risk", "dead_stock"):
        assert k in r.json()
