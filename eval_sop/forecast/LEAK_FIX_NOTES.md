# Early-stopping leak in `_train_catboost`: what it was, the fix, and how big it was

## What the code did (pre-fix commit 65c1c06, `forecasting/ml_forecast.py`)
- Line 1125: `cutoff = df["date"].max() - pd.Timedelta(days=90)`. Train is `date <= cutoff`, validation `X_va` is `date > cutoff` (lines 1144-1147).
- Line 1166: `m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)` with `early_stopping_rounds=50`. That means the best iteration is **picked on `X_va`**.
- Lines 1171-1172: the reported `mape` is then computed on that same `X_va`. **The leak is real**: model selection and scoring use the same rows.
- Validation is **one-step-ahead with true lags**. `_make_cb_features` builds `lag_1`, `roll_mean_7` and the rest from observed demand inside the validation window, so the number measures 1-day-ahead accuracy, not accuracy over a 30-day horizon.
- The metric is plain MAPE (`/(y+1e-6)`), not sMAPE. The README's "sMAPE, reported out-of-sample" (README.md:160) describes `backend/forecasting/training.py::_backtest_one`. That function is a recursive multi-step sMAPE backtest, but it runs on only the top-8 SKUs by volume, one fold, the lag-only per-SKU CatBoost, and no baseline. The `val_mape` that README.md:299 shows next to it comes from the leaky path above.

## Fix (minimal; `forecasting/ml_forecast.py`, `_train_catboost`)
```diff
-    X_tr, y_tr = tr[_CB_FEATURES], tr["demand"]
+    es_cutoff = cutoff - pd.Timedelta(days=30)
+    tr_fit = tr[tr["date"] <= es_cutoff]
+    tr_es = tr[tr["date"] > es_cutoff]
+    X_tr, y_tr = tr_fit[_CB_FEATURES], tr_fit["demand"]
+    X_es, y_es = tr_es[_CB_FEATURES], tr_es["demand"]
     X_va, y_va = va[_CB_FEATURES], va["demand"]
 ...
-        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
+        m.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True)
 ...
+        "es_rows": int(len(X_es)),
```
Early stopping now runs on an inner split: the last 30 days of the training window. `X_va` is never seen during fitting. Nothing else changed: hyper-parameters, seed 42, the metric definition, and the one-step-ahead nature of the validation are all the same.

## Measured effect (same seed 42, same data; `eval_sop/forecast/leak_check.py`, output in `results/leak_check.json`)
| | val MAPE (p50) | val sMAPE (p50) | MAE | best iterations p10/p50/p90 |
|---|---|---|---|---|
| before (leaky) | 11.96 % | 12.31 % | 15.25 | 498 / 498 / 490 |
| after (fixed)  | 11.97 % | 12.33 % | 15.26 | 492 / 496 / 449 |

**On this dataset the leak's optimism is negligible (about 0.01-0.02 pp).** Early stopping almost never fires: the best iteration sits near the 500-iteration cap in both runs, so picking the iteration on `X_va` hardly changes the model. The fix is still correct, because it removes test-set model selection on principle. But it does not materially change the reported number here. The "after" model also fits on 30 fewer days, so the tiny delta mixes leak removal with less training data.

Both numbers are still **one-step-ahead, true-lag** accuracy. They should not be quoted as 30-day forecast accuracy. For real multi-step, rolling-origin numbers, see `README.md` in this folder.
