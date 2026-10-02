"""Rolling-origin, true multi-step backtest of the repo's forecasters vs baselines.

Design (see eval_sop/forecast/README.md for the full write-up):
  * Data      : data/huft_daily_demand.csv (synthetic, data/generate_data.py).
  * Origins   : last observed training day o in ORIGINS; test = o+1 .. o+30.
                Every test window ends <= 2025-12-31.
  * Horizon   : H = 30 days, TRUE multi-step: every forecast for o+h uses only
                demand observed <= o (recursive models feed back their own
                predictions; no actual demand from the test window is ever used).
  * SKUs      : all 160 SKUs (default) or a seeded stratified sample (--n-skus).
  * Models    :
      catboost_global      repo feature design (_CB_FEATURES / _make_cb_features
                           from forecasting/ml_forecast.py), one global
                           Quantile:alpha=0.5 model per origin, repo
                           hyper-parameters except iterations=500 & threads;
                           early stopping on an INNER split (last 30 days of the
                           training window). Recursive forecast with features
                           recomputed consistently with the training definition.
      catboost_repo_infer  same trained model, but forecasts produced by the
                           repo's own forecasting/ml_forecast.py::_forecast_catboost
                           (as served). That function's rolling features differ
                           from training (window shifted by one day, np.std ddof=0
                           vs pandas ddof=1) -- reported separately, unmodified.
      catboost_backend     backend/forecasting/catboost_model.py design (per-SKU,
                           lags (1,2,3,7,14,28), 200 it, depth 4, lr 0.1, median),
                           recursive with clipping at 0 exactly like
                           backend/forecasting/training.py::_backtest_one; the
                           only change is an explicit random_seed.
      naive, snaive7, ma28 last value / last week repeated / 28-day mean.
      croston, tsb         implemented here (alpha=0.1; TSB beta=0.1).
  * Metrics   : sMAPE, MASE (m=7 and m=1 scaling on the training window), MAE.
  * Seeds     : CatBoost random_seed in {0,1,2}. Bootstrap seed 20251231.

Run from repo root:
    DATABASE_URL= python eval_sop/forecast/backtest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ["DATABASE_URL"] = ""  # never touch remote Postgres (artifact_store)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import catboost  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

import forecasting.ml_forecast as mlf  # noqa: E402
from backend.forecasting.catboost_model import _LAGS as BK_LAGS  # noqa: E402
from backend.forecasting.catboost_model import _make_supervised  # noqa: E402

mlf._save_catboost = lambda: None  # never persist anything

DATA = ROOT / "data" / "huft_daily_demand.csv"
OUTDIR = Path(__file__).resolve().parent / "results"
ORIGINS = ["2025-06-30", "2025-08-31", "2025-10-31", "2025-12-01"]
H = 30
SEEDS = [0, 1, 2]
BOOT_REPS = 2000
BOOT_SEED = 20251231
SAMPLE_SEED = 7
CROSTON_ALPHA = 0.1
TSB_ALPHA, TSB_BETA = 0.1, 0.1
THREADS = 6
ADI_CUT, CV2_CUT = 1.32, 0.49


# ───────────────────────────── metrics ─────────────────────────────────────
def smape(a, f):
    """100/H * sum 2|f-a| / (|a|+|f|); a term with a=f=0 contributes 0."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    d = np.abs(a) + np.abs(f)
    t = np.where(d > 0, 2.0 * np.abs(f - a) / np.where(d > 0, d, 1.0), 0.0)
    return float(t.mean() * 100.0)


def mase(a, f, y_train, m):
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))) / scale) if scale > 0 else np.nan


def sb_class(y):
    nz = y[y > 0]
    if nz.size == 0:
        return "none", np.inf, np.nan
    adi = y.size / nz.size
    cv2 = float((nz.std() / nz.mean()) ** 2)
    if adi < ADI_CUT:
        c = "erratic" if cv2 >= CV2_CUT else "smooth"
    else:
        c = "lumpy" if cv2 >= CV2_CUT else "intermittent"
    return c, float(adi), cv2


# ───────────────────────────── baselines ───────────────────────────────────
def f_naive(y):
    return np.full(H, y[-1], float)


def f_snaive7(y):
    return np.tile(y[-7:], H // 7 + 1)[:H].astype(float)


def f_ma28(y):
    return np.full(H, y[-28:].mean(), float)


def f_croston(y, alpha=CROSTON_ALPHA):
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(H)
    z, x, q = float(y[nz[0]]), float(nz[0] + 1), 1  # init: first size, first interval
    for t in range(nz[0] + 1, y.size):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            x += alpha * (q - x)
            q = 1
        else:
            q += 1
    return np.full(H, z / x)


def f_tsb(y, alpha=TSB_ALPHA, beta=TSB_BETA):
    nz = y[y > 0]
    if nz.size == 0:
        return np.zeros(H)
    z, p = float(nz.mean()), nz.size / y.size  # init: mean size, demand frequency
    for v in y:
        if v > 0:
            z += alpha * (v - z)
            p += beta * (1 - p)
        else:
            p += beta * (0 - p)
    return np.full(H, p * z)


BASELINES = {"naive": f_naive, "snaive7": f_snaive7, "ma28": f_ma28,
             "croston": f_croston, "tsb": f_tsb}


# ───────────────────────────── CatBoost (repo ml_forecast design) ──────────
def cb_params(seed):
    # repo base_params (forecasting/ml_forecast.py::_train_catboost) except
    # random_seed (swept) and thread_count (modest, shared machine).
    return dict(loss_function="Quantile:alpha=0.5", iterations=500, learning_rate=0.04,
                depth=7, l2_leaf_reg=3.0, min_data_in_leaf=30, subsample=0.8,
                colsample_bylevel=0.8, random_seed=seed, thread_count=THREADS,
                verbose=0, early_stopping_rounds=50)


def build_future_static(df, skus, origin, promos):
    """Known-future features for every (sku, o+h): calendar + promo calendar."""
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=H)
    last = df[df["date"] <= origin].groupby("sku_id").tail(1).set_index("sku_id").loc[skus]
    rep_dates = np.tile(dates.values, len(skus))
    rep_cat = pd.Series(np.repeat(last["category"].values, H))
    pf = mlf._build_promo_features(rep_dates, rep_cat, promos).reset_index(drop=True)
    di = pd.DatetimeIndex(rep_dates)
    fut = pd.DataFrame({
        "dayofweek": di.dayofweek, "month": di.month, "quarter": di.quarter,
        "isoweek": di.isocalendar().week.values.astype(int), "dayofyear": di.dayofyear,
        "is_weekend": (di.dayofweek >= 5).astype(int),
        "log_price": np.repeat(np.log1p(last["price_inr"].astype(float).values), H),
        "lead_time": np.repeat(last["lead_time_days"].astype(float).values, H),
        "sku_code": np.repeat(last.index.map(mlf._cb_sku_encoder).values, H),
        "cat_code": np.repeat(last["category"].map(mlf._cb_cat_encoder).fillna(0).astype(int).values, H),
    })
    for c in ["is_promo_active", "promo_discount_pct", "is_festival_week",
              "is_diwali_season", "is_monsoon"]:
        fut[c] = pf[c].values
    return fut  # row index = sku_idx * H + (h-1)


def recursive_cb(model, hist, fut):
    """hist: (n_sku, T) observed demand <= origin. Vectorised over SKUs."""
    n = hist.shape[0]
    work = [hist[i].astype(float).tolist() for i in range(n)]
    out = np.zeros((n, H))
    for h in range(H):
        rows = fut.iloc[np.arange(n) * H + h].reset_index(drop=True).copy()
        W = np.array([w[-29:] for w in work])  # last 29 values (lag up to 28)
        for lag in mlf._CB_LAGS:
            rows[f"lag_{lag}"] = W[:, -lag]
        for w in mlf._CB_WINDOWS:
            rows[f"roll_mean_{w}"] = W[:, -w:].mean(1)       # = shift(1).rolling(w)
            rows[f"roll_std_{w}"] = W[:, -w:].std(1, ddof=1)  # pandas default ddof
        p = np.maximum(model.predict(rows[mlf._CB_FEATURES]), 0.0)
        out[:, h] = p
        for i in range(n):
            work[i].append(float(p[i]))
    return out


def repo_infer_cb(model, df, skus, origin):
    mlf._cb_models = {"p10": model, "p50": model, "p90": model}
    mlf._cb_trained = True
    out = np.zeros((len(skus), H))
    for i, s in enumerate(skus):
        sdf = df[(df["sku_id"] == s) & (df["date"] <= origin)].sort_values("date").reset_index(drop=True)
        out[i] = mlf._forecast_catboost(s, sdf, H)["p50"]
    return out


# ───────────────────────────── backend lags-only CatBoost ─────────────────
def backend_cb(y, seed):
    X, t = _make_supervised(y.astype(float))
    m = CatBoostRegressor(loss_function="Quantile:alpha=0.5", iterations=200, depth=4,
                          learning_rate=0.1, verbose=False, random_seed=seed,
                          thread_count=THREADS)
    m.fit(X, t)
    work, preds = list(y.astype(float)), []
    for _ in range(H):
        feat = np.array([[work[-lag] for lag in BK_LAGS]], dtype=float)
        p = max(float(m.predict(feat)[0]), 0.0)
        preds.append(p)
        work.append(p)
    return np.array(preds)


# ───────────────────────────── aggregation ────────────────────────────────
def boot_ci(sku_vals, rng, reps=BOOT_REPS):
    v = np.asarray(sku_vals, float)
    if v.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, v.size, (reps, v.size))
    means = v[idx].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarise(per, models, subset_mask_col=None):
    rows = []
    for model in models:
        d = per[per["model"] == model]
        if subset_mask_col is not None:
            d = d[d[subset_mask_col]]
        r = {"model": model, "n_pairs": int(len(d)), "n_skus": int(d["sku_id"].nunique())}
        for met in ["smape", "mase7", "mase1", "mae"]:
            if len(d) == 0:
                r.update({f"{met}_mean": np.nan, f"{met}_std": np.nan, f"{met}_median": np.nan,
                          f"{met}_ci_lo": np.nan, f"{met}_ci_hi": np.nan})
                continue
            sku_means = d.groupby("sku_id")[met].mean().values
            lo, hi = boot_ci(sku_means, np.random.default_rng(BOOT_SEED))
            r.update({f"{met}_mean": d[met].mean(), f"{met}_std": d[met].std(ddof=1),
                      f"{met}_median": d[met].median(), f"{met}_ci_lo": lo, f"{met}_ci_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows)


def paired(per, model_a, model_b, met, mask=None):
    a = per[per["model"] == model_a].set_index(["sku_id", "origin"])[met]
    b = per[per["model"] == model_b].set_index(["sku_id", "origin"])[met]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if mask is not None:
        j = j[j.index.isin(mask)]
    if len(j) == 0:
        return {"n_pairs": 0}
    diff = j["a"] - j["b"]
    sku_d = diff.groupby(level=0).mean().values
    lo, hi = boot_ci(sku_d, np.random.default_rng(BOOT_SEED))
    wins = float(((j["a"] < j["b"]).sum() + 0.5 * (j["a"] == j["b"]).sum()) / len(j))
    return {"model_a": model_a, "model_b": model_b, "metric": met, "n_pairs": int(len(j)),
            "mean_diff_a_minus_b": float(diff.mean()), "ci95": [lo, hi],
            "median_diff": float(diff.median()), "win_rate_a": wins}


# ───────────────────────────── main ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skus", type=int, default=0, help="0 = all SKUs")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--origins", nargs="+", default=ORIGINS)
    ap.add_argument("--skip-repo-infer", action="store_true")
    ap.add_argument("--skip-backend", action="store_true")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()
    t_start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["sku_id", "date"]).reset_index(drop=True)
    all_skus = sorted(df["sku_id"].unique())
    dates_all = pd.DatetimeIndex(sorted(df["date"].unique()))
    assert len(df) == len(all_skus) * len(dates_all), "panel is not rectangular"

    origins = [pd.Timestamp(o) for o in args.origins]
    for o in origins:
        assert o + pd.Timedelta(days=H) <= df["date"].max(), f"test window past data end: {o}"

    # SKU sample: stratified by Syntetos-Boylan class on the earliest training window
    first_tr = df[df["date"] <= origins[0]]
    cls0 = {s: sb_class(g["demand"].values.astype(float))[0] for s, g in first_tr.groupby("sku_id")}
    if args.n_skus and args.n_skus < len(all_skus):
        rng = np.random.default_rng(SAMPLE_SEED)
        cls_ser = pd.Series(cls0)
        picks = []
        for c, grp in cls_ser.groupby(cls_ser):
            k = max(1, round(args.n_skus * len(grp) / len(cls_ser)))
            picks += list(rng.choice(grp.index.values, size=min(k, len(grp)), replace=False))
        skus = sorted(picks)
        sku_note = f"stratified random sample (seed {SAMPLE_SEED}) of {len(skus)} SKUs"
    else:
        skus = all_skus
        sku_note = f"all {len(skus)} SKUs"
    print(f"[backtest] {sku_note}; origins={args.origins}; seeds={args.seeds}", flush=True)

    # Repo encoders (sorted, as in _train_catboost) + repo feature matrix computed
    # once on the full panel; only rows with date <= origin are ever used for
    # training, and every feature at row t depends on demand < t or the calendar.
    mlf._cb_sku_encoder = {v: i for i, v in enumerate(all_skus)}
    mlf._cb_cat_encoder = {v: i for i, v in enumerate(sorted(df["category"].unique()))}
    t0 = time.time()
    feat = mlf._make_cb_features(df).dropna(subset=mlf._CB_FEATURES)
    print(f"[backtest] repo features built in {time.time()-t0:.0f}s", flush=True)
    promos = mlf._load_promotions()

    wide = df.pivot(index="sku_id", columns="date", values="demand").loc[skus]
    records = []
    timings = {}

    def add(model, seed, o, s, y_tr, y_te, f, info, fit_info=None):
        records.append({
            "model": model, "seed": seed, "origin": o.date().isoformat(), "sku_id": s,
            "sb_class": info[0], "adi": info[1], "cv2": info[2],
            "intermittent": info[0] in ("intermittent", "lumpy"),
            "train_mean": float(y_tr.mean()), "test_mean": float(y_te.mean()),
            "smape": smape(y_te, f), "mase7": mase(y_te, f, y_tr, 7),
            "mase1": mase(y_te, f, y_tr, 1), "mae": float(np.mean(np.abs(y_te - f))),
            "fc_mean": float(np.mean(f)), **(fit_info or {}),
        })

    for o in origins:
        tr_cols = wide.columns[wide.columns <= o]
        te_cols = pd.date_range(o + pd.Timedelta(days=1), periods=H)
        hist = wide[tr_cols].values.astype(float)
        test = wide[te_cols].values.astype(float)
        infos = [sb_class(hist[i]) for i in range(len(skus))]

        # baselines
        t0 = time.time()
        for name, fn in BASELINES.items():
            for i, s in enumerate(skus):
                add(name, -1, o, s, hist[i], test[i], fn(hist[i]), infos[i])
        timings[f"baselines_{o.date()}"] = time.time() - t0

        # global CatBoost (repo ml_forecast design)
        tr = feat[feat["date"] <= o]
        es_cut = o - pd.Timedelta(days=30)
        fit, es = tr[tr["date"] <= es_cut], tr[tr["date"] > es_cut]
        fut = build_future_static(df, skus, o, promos)
        for seed in args.seeds:
            t0 = time.time()
            m = CatBoostRegressor(**cb_params(seed))
            m.fit(fit[mlf._CB_FEATURES], fit["demand"],
                  eval_set=(es[mlf._CB_FEATURES], es["demand"]), use_best_model=True)
            bi = m.get_best_iteration()
            t_fit = time.time() - t0
            fc = recursive_cb(m, hist, fut)
            for i, s in enumerate(skus):
                add("catboost_global", seed, o, s, hist[i], test[i], fc[i], infos[i],
                    {"best_iter": bi})
            if not args.skip_repo_infer:
                fc2 = repo_infer_cb(m, df, skus, o)
                for i, s in enumerate(skus):
                    add("catboost_repo_infer", seed, o, s, hist[i], test[i], fc2[i], infos[i],
                        {"best_iter": bi})
            timings[f"catboost_global_{o.date()}_s{seed}"] = time.time() - t0
            print(f"[backtest] origin {o.date()} seed {seed}: fit {t_fit:.0f}s, best_iter={bi}, "
                  f"total {time.time()-t0:.0f}s", flush=True)

            if not args.skip_backend:
                t0 = time.time()
                for i, s in enumerate(skus):
                    add("catboost_backend", seed, o, s, hist[i], test[i],
                        backend_cb(hist[i], seed), infos[i])
                timings[f"catboost_backend_{o.date()}_s{seed}"] = time.time() - t0
                print(f"[backtest]   backend per-SKU CatBoost: {time.time()-t0:.0f}s", flush=True)

    per = pd.DataFrame(records)
    per.to_csv(outdir / "per_sku_origin.csv", index=False)

    # seed-averaged metric rows for the CatBoost variants
    cb_models = [m for m in ["catboost_global", "catboost_repo_infer", "catboost_backend"]
                 if m in per["model"].unique()]
    avg = (per[per["model"].isin(cb_models)]
           .groupby(["model", "origin", "sku_id"], as_index=False)
           .agg({**{k: "mean" for k in ["smape", "mase7", "mase1", "mae", "fc_mean"]},
                 **{k: "first" for k in ["sb_class", "adi", "cv2", "intermittent",
                                         "train_mean", "test_mean"]}}))
    avg["model"] = avg["model"] + "[seedmean]"
    avg["seed"] = -1
    per_seed = per[per["model"].isin(cb_models)].copy()
    per_seed["model"] = per_seed["model"] + "[s" + per_seed["seed"].astype(str) + "]"
    combo = pd.concat([per[~per["model"].isin(cb_models)], avg, per_seed], ignore_index=True)

    model_order = list(BASELINES) + [f"{m}[seedmean]" for m in cb_models] + \
        sorted(per_seed["model"].unique())
    s_all = summarise(combo, model_order).assign(subset="all")
    s_int = summarise(combo, model_order, "intermittent").assign(subset="intermittent")
    summary = pd.concat([s_all, s_int], ignore_index=True)
    summary.to_csv(outdir / "summary.csv", index=False)

    base_means = s_all[s_all["model"].isin(BASELINES)].set_index("model")
    best_smape = base_means["smape_mean"].idxmin()
    best_mase = base_means["mase7_mean"].idxmin()
    pairs = {}
    int_idx = per[(per["model"] == "naive") & per["intermittent"]].set_index(["sku_id", "origin"]).index
    for m in cb_models:
        mm = f"{m}[seedmean]"
        pairs[f"{mm}_vs_{best_smape}_smape"] = paired(combo, mm, best_smape, "smape")
        pairs[f"{mm}_vs_{best_mase}_mase7"] = paired(combo, mm, best_mase, "mase7")
        pairs[f"{mm}_vs_{best_smape}_smape_intermittent"] = paired(combo, mm, best_smape, "smape", int_idx)
    # seed variation of the aggregate mean
    seed_var = {}
    for m in cb_models:
        d = per[per["model"] == m].groupby("seed")[["smape", "mase7"]].mean()
        seed_var[m] = {"per_seed_mean_smape": d["smape"].round(4).to_dict(),
                       "per_seed_mean_mase7": d["mase7"].round(4).to_dict(),
                       "std_across_seeds_smape": float(d["smape"].std(ddof=1)) if len(d) > 1 else 0.0,
                       "range_across_seeds_smape": float(d["smape"].max() - d["smape"].min())}
    class_counts = (per[per["model"] == "naive"].groupby(["origin", "sb_class"]).size()
                    .unstack(fill_value=0).to_dict(orient="index"))

    meta = {
        "data_file": "data/huft_daily_demand.csv", "data_sha256": sha,
        "data_rows": int(len(df)), "n_skus_total": len(all_skus), "sku_selection": sku_note,
        "origins_last_train_day": args.origins, "horizon_days": H, "seeds": args.seeds,
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED, "unit": "SKU (cluster: all origins of a SKU resampled together)"},
        "croston_alpha": CROSTON_ALPHA, "tsb_alpha": TSB_ALPHA, "tsb_beta": TSB_BETA,
        "sb_class_counts_by_origin": class_counts,
        "n_intermittent_pairs": int(len(int_idx)),
        "best_baseline_by_mean_smape": best_smape, "best_baseline_by_mean_mase7": best_mase,
        "paired": pairs, "seed_variation": seed_var,
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
        "runtime_s": round(time.time() - t_start, 1), "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    out = {"meta": meta, "summary": summary.replace({np.nan: None}).to_dict(orient="records")}
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    pd.set_option("display.width", 220)
    cols = ["model", "subset", "n_pairs", "smape_mean", "smape_std", "smape_ci_lo", "smape_ci_hi",
            "mase7_mean", "mase7_std", "mase7_ci_lo", "mase7_ci_hi", "mae_mean"]
    print(summary[cols].round(3).to_string(index=False))
    print(json.dumps({"paired": pairs, "seed_variation": seed_var,
                      "class_counts": class_counts, "runtime_s": meta["runtime_s"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
