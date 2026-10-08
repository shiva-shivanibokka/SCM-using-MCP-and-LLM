"""Reproduce the single-day-velocity bug on the ORIGINAL code (git HEAD of the
base commit) and on the fixed code, using (1) a synthetic spike and (2) the
real data for EXT_059. Run from repo root:
  python eval_sop/repro/repro_stockout.py
"""
import importlib.util, os, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = "65c1c06"
sys.path.insert(0, str(ROOT))


def load(src_text, name):
    d = Path(os.getenv("REPRO_TMP", str(ROOT.parents[1] / "repro_tmp"))); d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.py"
    p.write_text(src_text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

old_src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:intelligence/stockout.py"]).decode()
old = load(old_src, "stockout_base")
new = load((ROOT / "intelligence/stockout.py").read_text(encoding="utf-8"), "stockout_fixed")

sys.path.insert(0, str(ROOT / "tests"))
from test_stockout_velocity import _history  # noqa: E402
h = _history()
for tag, m in [("BASE", old), ("FIXED", new)]:
    r = m.predict_stockouts(h)["rows"][0]
    print(f"[synthetic 30d @75/day, last day 300, inv 400, LT 3] {tag}: velocity={r['daily_velocity']} "
          f"days_to_zero={r['days_to_zero']} risk={r['risk']}")

inv = pd.read_csv(ROOT / "data" / "store_daily_inventory.csv", parse_dates=["date"])
for tag, m in [("BASE", old), ("FIXED", new)]:
    res = m.predict_stockouts(inv)
    rows = {x["sku_id"]: x for x in res["rows"]}
    x = rows["EXT_059"]
    print(f"[real data EXT_059] {tag}: velocity={x['daily_velocity']} days_to_zero={x['days_to_zero']} "
          f"reorder_qty={x['reorder_qty']} risk={x['risk']} | summary critical={res['summary']['critical']} "
          f"warning={res['summary']['warning']}")
sku = inv[inv.sku_id == "EXT_059"].groupby("date").demand.sum().tail(8)
print("EXT_059 summed store demand, last 8 days:", sku.to_dict())
