from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_sku_list():
    r = client.get("/api/forecast/skus")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and "sku_id" in body[0]


def test_forecast_shape():
    sku = client.get("/api/forecast/skus").json()[0]["sku_id"]
    r = client.get(f"/api/forecast/{sku}?horizon=7")
    assert r.status_code == 200
    body = r.json()
    assert len(body["p50"]) == 7


def test_mlops_registry():
    r = client.get("/api/mlops/registry")
    assert r.status_code == 200
    assert "weights" in r.json()
