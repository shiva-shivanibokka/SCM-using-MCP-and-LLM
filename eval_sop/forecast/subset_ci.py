"""Subset / per-origin confidence intervals for CatBoost global vs MA28.

Review follow-up: RESULTS.md stated the excl-Diwali delta (+0.17 pp) as a bare
point estimate.  This script re-uses the backtest's OWN bootstrap helpers
(`eval_sop.forecast.backtest.paired` -> `boot_ci`, 2,000 reps, seed 20251231,
cluster = SKU) on the committed `results/per_sku_origin.csv`, so every interval
quoted in RESULTS.md §2a is reproducible without re-running the 70-minute
backtest.

Reported:
  * all 4 origins              (SKU-clustered, matches summary.json)
  * excluding 2025-10-31       (the Diwali origin)
  * each origin separately
  * an ORIGIN-clustered CI over the 4 origin-level mean diffs

Run from the repo root:
    python eval_sop/forecast/subset_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_sop.forecast.backtest import (  # noqa: E402
    BOOT_REPS,
    BOOT_SEED,
    boot_ci,
    paired,
)

RES = Path(__file__).resolve().parent / "results"
PER = RES / "per_sku_origin.csv"
DIWALI = "2025-10-31"
CB = "catboost_global"
BASE = "ma28"
MET = "smape"


def seed_averaged(per: pd.DataFrame) -> pd.DataFrame:
    """Reproduce backtest.main's `catboost_global[seedmean]` rows."""
    cb = (per[per["model"] == CB]
          .groupby(["origin", "sku_id"], as_index=False)[[MET]].mean())
    cb["model"] = f"{CB}[seedmean]"
    return pd.concat([per[per["model"] == BASE][["model", "origin", "sku_id", MET]], cb],
                     ignore_index=True)


def main() -> None:
    per = pd.read_csv(PER)
    combo = seed_averaged(per)
    cbm = f"{CB}[seedmean]"
    origins = sorted(combo["origin"].unique())

    def mask_for(keep):
        idx = combo[(combo["model"] == BASE) & combo["origin"].isin(keep)]
        return pd.MultiIndex.from_frame(idx[["sku_id", "origin"]])

    out = {"bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED,
                         "cluster": "SKU (all kept origins of a SKU move together)"},
           "metric": MET, "model_a": cbm, "model_b": BASE, "subsets": {}}

    out["subsets"]["all_4_origins"] = paired(combo, cbm, BASE, MET, mask_for(origins))
    keep = [o for o in origins if o != DIWALI]
    out["subsets"]["excl_diwali_2025-10-31"] = paired(combo, cbm, BASE, MET, mask_for(keep))
    for o in origins:
        out["subsets"][f"origin_{o}"] = paired(combo, cbm, BASE, MET, mask_for([o]))

    # Origin-clustered CI: resample the 4 origin-level mean diffs.
    a = combo[combo["model"] == cbm].groupby("origin")[MET].mean()
    b = combo[combo["model"] == BASE].groupby("origin")[MET].mean()
    od = (a - b).reindex(origins).to_numpy(float)
    lo, hi = boot_ci(od, np.random.default_rng(BOOT_SEED))
    out["origin_clustered_all_4"] = {"origin_mean_diffs": dict(zip(origins, od.round(4))),
                                     "mean_diff_a_minus_b": float(od.mean()),
                                     "ci95": [lo, hi],
                                     "cluster": "origin (4 clusters)"}

    (RES / "subset_ci.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    lines = [f"CatBoost global (seed-mean) minus {BASE}, {MET} pp; "
             f"SKU-clustered bootstrap {BOOT_REPS} reps, seed {BOOT_SEED}", ""]
    lines.append(f"{'subset':28s} {'n pairs':>7s} {'n SKUs':>6s}  {'delta':>7s}  95% CI")
    for k, v in out["subsets"].items():
        lo_, hi_ = v["ci95"]
        lines.append(f"{k:28s} {v['n_pairs']:7d} {'':>6s}  {v['mean_diff_a_minus_b']:+7.3f}  "
                     f"[{lo_:+.3f}, {hi_:+.3f}]  win_rate_catboost={v['win_rate_a']:.3f}")
    oc = out["origin_clustered_all_4"]
    lines += ["", "Origin-clustered (4 origin-level mean diffs resampled):",
              f"  per-origin deltas: " + ", ".join(f"{k}: {v:+.3f}" for k, v in oc["origin_mean_diffs"].items()),
              f"  delta {oc['mean_diff_a_minus_b']:+.3f} pp, 95% CI "
              f"[{oc['ci95'][0]:+.3f}, {oc['ci95'][1]:+.3f}]"]
    text = "\n".join(lines) + "\n"
    (RES / "subset_ci.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
