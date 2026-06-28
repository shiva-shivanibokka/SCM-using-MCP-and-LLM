from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_suppliers():
    r = client.get("/api/suppliers/scorecard")
    assert r.status_code == 200 and "suppliers" in r.json()
    assert len(r.json()["suppliers"]) > 0


def test_stores():
    r = client.get("/api/stores/grid")
    assert r.status_code == 200 and "stores" in r.json()
    assert len(r.json()["stores"]) >= 80


def test_analytics():
    r = client.get("/api/analytics/overview")
    assert r.status_code == 200 and "segments" in r.json()
    assert len(r.json()["segments"]) > 0
