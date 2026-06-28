def test_app_imports():
    from backend.main import app
    routes = {getattr(r, "path", "") for r in app.routes}
    assert "/health" in routes
    assert "/api/forecast/skus" in routes
