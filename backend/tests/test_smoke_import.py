from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_app_imports():
    # Behavioral smoke check: the app boots and its key routes are reachable.
    # (We assert reachability via the TestClient rather than introspecting
    # app.routes/.path, which enumerates included-router paths differently
    # across Starlette versions and gave false negatives in CI.)
    assert client.get("/health").status_code == 200
    assert client.get("/api/forecast/skus").status_code == 200
