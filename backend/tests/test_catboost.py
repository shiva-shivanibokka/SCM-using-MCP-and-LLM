from backend.forecasting.contract import validate_forecast
from backend.forecasting import catboost_model


def test_catboost_shape():
    history = [float(x % 7) + 10 for x in range(120)]
    out = catboost_model.forecast_catboost(history, horizon=10)
    validate_forecast(out, horizon=10)
