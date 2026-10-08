"""Security regression tests for the guarded SQL tool (intelligence.sql).

Files used here are created under pytest's tmp_path only.
"""
import time

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from intelligence import sql as S  # noqa: E402


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    pd.DataFrame({"sku_id": ["A", "B"], "name": ["Dog Food", "Cat Toy"],
                  "category": ["Food", "Toys"]}).to_csv(d / "huft_products.csv", index=False)
    pd.DataFrame({"x": range(3000)}).to_csv(d / "huft_stores.csv", index=False)
    S._CON_CACHE.clear()
    yield d
    S._CON_CACHE.clear()


def test_normal_select_still_works(data_dir):
    r = S.run_query("SELECT sku_id FROM products ORDER BY sku_id", data_dir)
    assert r["error"] is None and [x["sku_id"] for x in r["rows"]] == ["A", "B"]


def test_quoted_path_access_is_blocked(data_dir, tmp_path):
    secret = tmp_path / "canary.csv"
    secret.write_text("k,v\nTOKEN,canary-xyz\n", encoding="utf-8")
    r = S.run_query(f"SELECT * FROM '{secret.as_posix()}'", data_dir)
    assert r["error"] is not None
    assert "canary-xyz" not in str(r["rows"])


def test_expensive_query_times_out(data_dir):
    t0 = time.time()
    r = S.run_query("SELECT sum(a.x * b.x * c.x) FROM stores a, stores b, stores c", data_dir,
                    timeout_s=2)
    assert r["error"] is not None and "timed out" in r["error"].lower()
    assert time.time() - t0 < 20
