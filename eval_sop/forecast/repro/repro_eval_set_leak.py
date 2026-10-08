"""Reproduce the early-stopping (eval_set) leak in _train_catboost.

Property tested: the rows passed to CatBoost as `eval_set` (which drive early
stopping / use_best_model) must be DISJOINT from the rows the reported
validation metric (`mape`) is computed on.

CatBoostRegressor is replaced by a recording stub (no real training), so this
runs in seconds. It records the DataFrame index of every eval_set and of every
predict() input, then checks their overlap.

Runs against:
  * ORIGINAL code: `git show 65c1c06:forecasting/ml_forecast.py` (pre-fix HEAD
    at the time of the fix), loaded as a temp module  -> expected FAIL
  * FIXED code   : working-tree forecasting/ml_forecast.py      -> expected PASS

Exit code 0 iff ORIGINAL fails and FIXED passes.
Usage (repo root):  DATABASE_URL= python eval_sop/forecast/repro/repro_eval_set_leak.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = ""

import catboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
ORIG_COMMIT = "65c1c0606823882ebd742f639ae39b8aee325d54"


class RecordingStub:
    eval_idx: list = []
    pred_idx: list = []

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y, eval_set=None, use_best_model=False):
        if eval_set is not None:
            RecordingStub.eval_idx.append(set(eval_set[0].index))
        return self

    def predict(self, X):
        RecordingStub.pred_idx.append(set(X.index))
        return np.zeros(len(X))

    def get_best_iteration(self):
        return 0


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.DATA_DIR = ROOT / "data"
    mod._save_catboost = lambda: None
    return mod


def check(path: Path, label: str, df: pd.DataFrame) -> bool:
    RecordingStub.eval_idx, RecordingStub.pred_idx = [], []
    mod = load(path, f"mlf_{label}")
    mod._train_catboost(df)
    scored = set().union(*RecordingStub.pred_idx)
    evals = set().union(*RecordingStub.eval_idx)
    overlap = len(scored & evals)
    ok = overlap == 0
    print(f"[{label}] eval_set rows={len(evals)}  scored(validation) rows={len(scored)}  "
          f"overlap={overlap} ({100*overlap/max(len(scored),1):.1f}% of scored rows)  -> "
          f"{'PASS' if ok else 'FAIL (leak: early stopping sees the scored rows)'}")
    return ok


def main() -> int:
    catboost.CatBoostRegressor = RecordingStub  # module does `from catboost import ...` at call time
    df = pd.read_csv(ROOT / "data" / "huft_daily_demand.csv", parse_dates=["date"])
    # temp copy lives inside the repo (no system temp dir); removed after loading
    tmp = Path(__file__).resolve().parent / "_tmp_ml_forecast_orig.py"
    tmp.write_bytes(subprocess.check_output(
        ["git", "show", f"{ORIG_COMMIT}:forecasting/ml_forecast.py"], cwd=ROOT))
    try:
        orig_ok = check(tmp, "ORIGINAL_65c1c06", df)
    finally:
        tmp.unlink(missing_ok=True)
    fixed_ok = check(ROOT / "forecasting" / "ml_forecast.py", "FIXED_worktree", df)
    good = (not orig_ok) and fixed_ok
    print("RESULT:", "leak reproduced on original and absent after fix" if good else "UNEXPECTED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
