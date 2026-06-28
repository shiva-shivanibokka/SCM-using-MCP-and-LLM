import pytest
from backend.forecasting.contract import validate_forecast

pytest.importorskip("neuralforecast", reason="neuralforecast deps not installed")
from backend.forecasting import nhits_model  # noqa: E402


def test_nhits_shape():
    history = [float(x % 7) + 10 for x in range(220)]
    out = nhits_model.forecast_nhits(history, horizon=14)
    validate_forecast(out, horizon=14)
