import pandas as pd
from backend import data_access


def test_load_products_cached():
    a = data_access.load_products()
    b = data_access.load_products()
    assert isinstance(a, pd.DataFrame)
    assert a is b  # same cached object


def test_sku_history_returns_floats():
    demand = data_access.load_demand()
    sku = demand["sku_id"].iloc[0]
    hist = data_access.sku_history(sku)
    assert isinstance(hist, list) and len(hist) > 0
    assert all(isinstance(x, float) for x in hist)
