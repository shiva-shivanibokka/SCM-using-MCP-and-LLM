"""Quantify the early-stopping leak in forecasting/ml_forecast.py::_train_catboost.

Runs the repo's own _train_catboost twice on data/huft_daily_demand.csv with the
same seed (random_seed=42, hard-coded in the repo):
  * BEFORE: the pre-fix file from git commit 65c1c06 (eval_set = the reported
    validation window X_va -> early stopping selects the iteration on the
    same rows that are then scored).
  * AFTER : the current working-tree file (early stopping on an inner split =
    last 30 days of the training window; X_va never seen during fitting).

Persistence (_save_catboost -> local .model_cache + Postgres) is monkeypatched
to a no-op, so nothing is written to disk or to any database.

Usage (from repo root):
    DATABASE_URL= python eval_sop/forecast/leak_check.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BEFORE_COMMIT = "65c1c0606823882ebd742f639ae39b8aee325d54"
OUT = Path(__file__).resolve().parent / "results" / "leak_check.json"

os.environ["DATABASE_URL"] = ""  # belt and braces: never touch remote Postgres


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = ROOT / "data"          # promotions CSV location
    mod._save_catboost = lambda: None     # no disk / DB writes
    return mod


def _smape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    d = np.abs(a) + np.abs(f)
    t = np.where(d > 0, 2 * np.abs(f - a) / np.where(d > 0, d, 1), 0.0)
    return float(t.mean() * 100)


def main():
    df = pd.read_csv(ROOT / "data" / "huft_daily_demand.csv", parse_dates=["date"])
    src = subprocess.check_output(
        ["git", "show", f"{BEFORE_COMMIT}:forecasting/ml_forecast.py"], cwd=ROOT
    )
    # temp copy lives inside the repo (no system temp dir); removed right after
    # import (the module stays loaded in sys.modules)
    tmp = Path(__file__).resolve().parent / "_tmp_ml_forecast_before.py"
    tmp.write_bytes(src)
    try:
        _load(tmp, "mlf_before_leaky")
    finally:
        tmp.unlink(missing_ok=True)

    res = {}
    for label, path in [("before_leaky", tmp), ("after_fixed", ROOT / "forecasting" / "ml_forecast.py")]:
        mod = sys.modules["mlf_before_leaky"] if path == tmp else _load(path, f"mlf_{label}")
        t0 = time.time()
        metrics = mod._train_catboost(df)
        # also sMAPE of the p50 on the reported validation window
        cutoff = df["date"].max() - pd.Timedelta(days=90)
        feat = mod._make_cb_features(df).dropna(subset=mod._CB_FEATURES)
        va = feat[feat["date"] > cutoff]
        p50 = np.maximum(mod._cb_models["p50"].predict(va[mod._CB_FEATURES]), 0)
        metrics = {k: v for k, v in metrics.items()}
        metrics["val_smape_p50"] = round(_smape(va["demand"].values, p50), 3)
        metrics["runtime_s"] = round(time.time() - t0, 1)
        res[label] = metrics
        print(label, json.dumps(metrics, default=str))

    res["note"] = (
        "Both numbers are ONE-STEP-AHEAD on the last 90 days using TRUE lags "
        "(lag_1 etc. come from observed demand inside the validation window); "
        "they are not multi-step forecast accuracy. 'after' also fits on 30 fewer "
        "days (inner ES split), so the delta mixes leak removal with slightly less "
        "training data. Seed 42 (repo default), thread_count=-1 (repo default)."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
