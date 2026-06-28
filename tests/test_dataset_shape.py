import subprocess, sys
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def _gen_once():
    if not (DATA / "huft_daily_demand.csv").exists():
        subprocess.run([sys.executable, str(DATA / "generate_data.py")], check=True)


def test_dataset_dimensions():
    _gen_once()
    products = pd.read_csv(DATA / "huft_products.csv")
    stores = pd.read_csv(DATA / "huft_stores.csv")
    customers = pd.read_csv(DATA / "huft_customers.csv")
    txns = pd.read_csv(DATA / "huft_sales_transactions.csv")
    demand = pd.read_csv(DATA / "huft_daily_demand.csv")

    assert products["sku_id"].nunique() >= 150, products["sku_id"].nunique()
    assert stores["store_id"].nunique() >= 80, stores["store_id"].nunique()
    assert len(customers) >= 25_000, len(customers)
    assert len(txns) >= 300_000, len(txns)
    assert demand["date"].nunique() >= 1095, demand["date"].nunique()


def test_demand_has_no_negatives():
    _gen_once()
    demand = pd.read_csv(DATA / "huft_daily_demand.csv")
    assert (demand["demand"] >= 0).all()
