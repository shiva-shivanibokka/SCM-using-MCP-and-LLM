"""Reproduce the misleading 'not found' message in get_sku_360 /
get_inventory_status: it advertises prefixes (DOG, CAT, MED, ACC) that match 0
SKUs in the data. Run from repo root (DATABASE_URL empty)."""
import os, re, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.environ["DATABASE_URL"] = ""
sys.path.insert(0, str(ROOT))
BASE = "90606bc"

real = sorted({re.match(r"[A-Z]+", s).group() for s in pd.read_csv(ROOT / "data/huft_products.csv")["sku_id"]})
print("real SKU prefixes in data:", real)

base_src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE}:mcp_server/server.py"]).decode()
advertised = re.findall(r"Valid prefixes: ([^\"]+)\.", base_src)
print(f"prefixes advertised by {BASE} 'not found' message:", advertised[:1])
adv = [p.strip() for p in advertised[0].split(",")] if advertised else []
bogus = [p for p in adv if p not in real]
print("advertised prefixes that match ZERO SKUs:", bogus)
print("RESULT:", "BUG reproduced — message lists only non-existent prefixes" if set(adv).isdisjoint(real)
      else "message overlaps real prefixes")
