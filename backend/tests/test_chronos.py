import pytest
from backend.forecasting.contract import validate_forecast

chronos = pytest.importorskip("chronos", reason="chronos deps not installed")
from backend.forecasting import chronos_model  # noqa: E402


def test_chronos_shape():
    history = [float(x % 7) + 10 for x in range(120)]
    out = chronos_model.forecast_chronos(history, horizon=14)
    validate_forecast(out, horizon=14)


def test_chronos_short_history_still_returns():
    out = chronos_model.forecast_chronos([10.0] * 20, horizon=5)
    validate_forecast(out, horizon=5)
