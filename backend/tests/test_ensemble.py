from backend.forecasting.contract import validate_forecast
from backend.forecasting import ensemble


def test_ensemble_blends_with_stubs(monkeypatch):
    def stub(val):
        def f(history, horizon=30):
            return {"p10": [val]*horizon, "p50": [val]*horizon, "p90": [val]*horizon}
        return f
    monkeypatch.setattr(ensemble, "forecast_chronos", stub(10.0))
    monkeypatch.setattr(ensemble, "forecast_nhits", stub(20.0))
    monkeypatch.setattr(ensemble, "forecast_catboost", stub(30.0))

    out = ensemble.ensemble_forecast([1.0]*50, horizon=3)
    validate_forecast(out, horizon=3)
    # 0.5*10 + 0.35*20 + 0.15*30 = 16.5
    assert abs(out["p50"][0] - 16.5) < 1e-6


def test_ensemble_survives_component_failure(monkeypatch):
    def boom(history, horizon=30):
        raise RuntimeError("model down")

    def ok(history, horizon=30):
        return {"p10": [5.0]*horizon, "p50": [5.0]*horizon, "p90": [5.0]*horizon}
    monkeypatch.setattr(ensemble, "forecast_chronos", boom)
    monkeypatch.setattr(ensemble, "forecast_nhits", ok)
    monkeypatch.setattr(ensemble, "forecast_catboost", ok)
    out = ensemble.ensemble_forecast([1.0]*50, horizon=2)
    validate_forecast(out, horizon=2)
    assert abs(out["p50"][0] - 5.0) < 1e-6
