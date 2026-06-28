import pytest
from backend.forecasting.contract import validate_forecast


def test_valid_passes():
    r = {"p10": [1.0, 2.0], "p50": [2.0, 3.0], "p90": [3.0, 4.0]}
    assert validate_forecast(r, horizon=2) == r


def test_wrong_length_raises():
    r = {"p10": [1.0], "p50": [2.0], "p90": [3.0]}
    with pytest.raises(ValueError):
        validate_forecast(r, horizon=2)


def test_missing_key_raises():
    with pytest.raises(ValueError):
        validate_forecast({"p10": [1.0], "p50": [1.0]}, horizon=1)


def test_quantiles_must_be_ordered():
    r = {"p10": [5.0], "p50": [2.0], "p90": [3.0]}
    with pytest.raises(ValueError):
        validate_forecast(r, horizon=1)
