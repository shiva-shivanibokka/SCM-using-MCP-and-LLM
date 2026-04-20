"""
MLOps Monitor — CSV-based (compatible with HF Spaces persistent storage).
Tracks forecast predictions, computes drift metrics, logs agent queries.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Storage paths (HF Spaces uses /data/ as persistent volume)
_LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
try:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    _LOGS_DIR = Path("/tmp/scm_logs")
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

PRED_LOG_PATH = _LOGS_DIR / "predictions.csv"
QUERY_LOG_PATH = _LOGS_DIR / "query_log.csv"
DRIFT_LOG_PATH = _LOGS_DIR / "drift_metrics.csv"

_lock = threading.Lock()

# Prediction Log

PRED_COLS = [
    "logged_at",
    "sku_id",
    "horizon_days",
    "p10_total",
    "p50_total",
    "p90_total",
    "p50_daily",
    "forecast_source",
    "model_version",
]


def log_prediction(
    sku_id: str,
    p10_total: float,
    p50_total: float,
    p90_total: float,
    p50_daily: float,
    horizon_days: int = 30,
    forecast_source: str = "TFT",
    model_version: str = "v2.0",
) -> None:
    """Append a forecast prediction to the CSV log."""
    row = pd.DataFrame(
        [
            {
                "logged_at": datetime.utcnow().isoformat(),
                "sku_id": sku_id.upper(),
                "horizon_days": horizon_days,
                "p10_total": round(p10_total, 2),
                "p50_total": round(p50_total, 2),
                "p90_total": round(p90_total, 2),
                "p50_daily": round(p50_daily, 2),
                "forecast_source": forecast_source,
                "model_version": model_version,
            }
        ]
    )
    with _lock:
        if PRED_LOG_PATH.exists():
            row.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(PRED_LOG_PATH, index=False)


def get_prediction_log(limit: int = 200) -> pd.DataFrame:
    """Return the most recent N prediction log entries.

    Reads only the last (limit + header) lines from the CSV to avoid loading
    the entire file into memory on long-running deployments (L-04 fix).
    """
    if not PRED_LOG_PATH.exists():
        return pd.DataFrame(columns=PRED_COLS)
    with _lock:
        try:
            # Use context manager to avoid file handle leak on Windows
            with open(PRED_LOG_PATH, encoding="utf-8") as _fh:
                total_lines = sum(1 for _ in _fh)
            skip = max(1, total_lines - limit - 1)  # keep header (line 0) + last N rows
            if skip > 1:
                # Header=0 already preserves column names — no need
                # for a second pd.read_csv call to re-read the header (was redundant I/O).
                df = pd.read_csv(PRED_LOG_PATH, skiprows=list(range(1, skip)), header=0)
            else:
                df = pd.read_csv(PRED_LOG_PATH)
        except Exception:
            df = pd.read_csv(PRED_LOG_PATH)
    if df.empty:
        return df
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return (
        df.sort_values("logged_at", ascending=False).head(limit).reset_index(drop=True)
    )


def get_most_queried_skus(top_n: int = 10) -> pd.DataFrame:
    """Return the most frequently forecasted SKUs."""
    df = get_prediction_log(limit=10000)
    if df.empty:
        return pd.DataFrame(columns=["sku_id", "count"])
    return (
        df.groupby("sku_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def get_sku_accuracy_chart(
    data_csv_path: str | Path | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compare each SKU's most recent P50_daily forecast against actual average
    daily demand over the last 30 days.  Returns a DataFrame with columns:
      sku_id, p50_daily, actual_daily, abs_error_pct, grade
    sorted by abs_error_pct descending (worst performers first).

    This replaces the meaningless "Most Queried SKUs" count chart with
    actionable model accuracy information.
    """
    pred_df = get_prediction_log(limit=10000)
    if pred_df.empty:
        return pd.DataFrame(
            columns=["sku_id", "p50_daily", "actual_daily", "abs_error_pct", "grade"]
        )

    # Most recent forecast per SKU
    latest_preds = pred_df.sort_values("logged_at", ascending=False).drop_duplicates(
        "sku_id"
    )[["sku_id", "p50_daily", "horizon_days", "logged_at"]]

    # Load actual demand
    if data_csv_path is None:
        data_csv_path = Path(__file__).parent.parent / "data" / "huft_daily_demand.csv"
    if not Path(data_csv_path).exists():
        return pd.DataFrame(
            columns=["sku_id", "p50_daily", "actual_daily", "abs_error_pct", "grade"]
        )

    demand_df = pd.read_csv(data_csv_path, parse_dates=["date"])

    # Compare each forecast against the 30d window FOLLOWING its logged_at,
    # not the most recent 30 days (which could be months after the forecast was made).
    latest_preds["logged_at"] = pd.to_datetime(latest_preds["logged_at"])
    actuals_rows = []
    for _, row in latest_preds.iterrows():
        forecast_start = pd.to_datetime(row["logged_at"]).normalize()
        forecast_end = forecast_start + pd.Timedelta(
            days=int(row.get("horizon_days", 30))
        )
        window = demand_df[
            (demand_df["sku_id"] == row["sku_id"])
            & (demand_df["date"] >= forecast_start)
            & (demand_df["date"] < forecast_end)
        ]
        if not window.empty:
            actuals_rows.append(
                {
                    "sku_id": row["sku_id"],
                    "p50_daily": row["p50_daily"],
                    "actual_daily": float(window["demand"].mean()),
                }
            )

    if not actuals_rows:
        # Fallback: use most recent 30 days if no forecast window has elapsed yet
        latest_date = demand_df["date"].max()
        cutoff = latest_date - pd.Timedelta(days=30)
        actuals = (
            demand_df[demand_df["date"] >= cutoff]
            .groupby("sku_id")["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "actual_daily"})
        )
        merged = latest_preds.merge(actuals, on="sku_id", how="inner")
    else:
        merged = pd.DataFrame(actuals_rows)
    if merged.empty:
        return pd.DataFrame(
            columns=["sku_id", "p50_daily", "actual_daily", "abs_error_pct", "grade"]
        )

    merged["abs_error_pct"] = (
        (merged["p50_daily"] - merged["actual_daily"]).abs()
        / (merged["actual_daily"] + 1e-6)
        * 100
    ).round(1)

    def _grade(e: float) -> str:
        if e < 10:
            return "Excellent"
        if e < 20:
            return "Good"
        if e < 35:
            return "Fair"
        return "Poor"

    merged["grade"] = merged["abs_error_pct"].apply(_grade)
    # Sort descending so worst performers appear first — this is what an
    # MLOps monitor should surface, not the best-performing SKUs.
    return (
        merged[["sku_id", "p50_daily", "actual_daily", "abs_error_pct", "grade"]]
        .sort_values("abs_error_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# Query Log

QUERY_COLS = [
    "queried_at",
    "session_id",
    "provider",
    "model",
    "user_query",
    "tools_called",
    "duration_ms",
]


def log_query(
    user_query: str,
    provider: str,
    model: str,
    tools_called: list[str],
    duration_ms: int = 0,
    session_id: str = "",
) -> None:
    row = pd.DataFrame(
        [
            {
                "queried_at": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "provider": provider,
                "model": model,
                "user_query": user_query[:300],
                "tools_called": json.dumps(tools_called),
                "duration_ms": duration_ms,
            }
        ]
    )
    with _lock:
        if QUERY_LOG_PATH.exists():
            row.to_csv(QUERY_LOG_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(QUERY_LOG_PATH, index=False)


def get_query_log(limit: int = 100) -> pd.DataFrame:
    if not QUERY_LOG_PATH.exists():
        return pd.DataFrame(columns=QUERY_COLS)
    with _lock:
        df = pd.read_csv(QUERY_LOG_PATH)
    if df.empty:
        return df
    df["queried_at"] = pd.to_datetime(df["queried_at"])
    return (
        df.sort_values("queried_at", ascending=False).head(limit).reset_index(drop=True)
    )


# Drift Detection


def compute_drift_metrics(data_csv_path: str | Path | None = None) -> dict[str, Any]:
    """
    Compare logged forecast P50 values against actual demand from the CSV.
    Returns a drift metrics dict.
    """
    pred_df = get_prediction_log(limit=5000)
    if pred_df.empty:
        return {"status": "no_predictions", "message": "No predictions logged yet."}

    # Load actual demand
    if data_csv_path is None:
        data_csv_path = Path(__file__).parent.parent / "data" / "huft_daily_demand.csv"

    if not Path(data_csv_path).exists():
        return {"status": "no_data", "message": "Demand CSV not found."}

    demand_df = pd.read_csv(data_csv_path, parse_dates=["date"])
    latest_date = demand_df["date"].max()
    recent_cutoff = latest_date - pd.Timedelta(days=30)
    recent_demand = (
        demand_df[demand_df["date"] >= recent_cutoff]
        .groupby("sku_id")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "actual_daily_avg"})
    )

    # Deduplicate to one row per SKU (most recent forecast) before calibration.
    # Without this, a SKU forecasted 50 times gets 50× weight and n_evaluated
    # is meaningless (H-01 fix).
    pred_dedup = pred_df.sort_values("logged_at", ascending=False).drop_duplicates(
        "sku_id"
    )

    # Merge with predictions
    merged = pred_dedup.merge(recent_demand, on="sku_id", how="inner")
    if merged.empty:
        return {
            "status": "no_overlap",
            "message": "No overlap between predicted SKUs and demand data.",
        }

    # MAE
    merged["abs_error"] = abs(merged["p50_daily"] - merged["actual_daily_avg"])
    mae = float(merged["abs_error"].mean())

    # Baseline: predict global mean for every SKU
    global_mean = float(recent_demand["actual_daily_avg"].mean())
    baseline_mae = float(abs(recent_demand["actual_daily_avg"] - global_mean).mean())

    # Calibration: % of actuals within p10-p90 band
    # Fillna before clip so NaN horizon entries don't deflate calibration_pct
    horizon = merged["horizon_days"].fillna(30).clip(lower=1)
    in_band = (merged["actual_daily_avg"] >= merged["p10_total"] / horizon) & (
        merged["actual_daily_avg"] <= merged["p90_total"] / horizon
    )
    calibration_pct = float(in_band.mean() * 100)

    # Drift flag
    drift_threshold = baseline_mae * 1.5
    drift_detected = mae > drift_threshold

    metrics = {
        "status": "ok",
        "n_predictions": len(pred_df),
        "n_evaluated": len(merged),
        "forecast_mae": round(mae, 2),
        "baseline_mae": round(baseline_mae, 2),
        "calibration_pct": round(calibration_pct, 1),
        "drift_detected": drift_detected,
        "drift_threshold": round(drift_threshold, 2),
        "worst_skus": (
            merged.nlargest(5, "abs_error")[
                ["sku_id", "p50_daily", "actual_daily_avg", "abs_error"]
            ]
            .round(2)
            .to_dict("records")
        ),
        "computed_at": datetime.utcnow().isoformat(),
    }

    # Append to drift log — rate-limited to once per 5 minutes to prevent
    # flooding from rapid UI auto-refresh calls (L-07 fix).
    _DRIFT_MIN_INTERVAL_SECONDS = 300
    should_log = True
    with _lock:
        if DRIFT_LOG_PATH.exists():
            try:
                last_df = pd.read_csv(DRIFT_LOG_PATH, usecols=["logged_at"])
                if not last_df.empty:
                    last_ts = pd.to_datetime(last_df["logged_at"].iloc[-1])
                    elapsed = (
                        datetime.utcnow() - last_ts.to_pydatetime().replace(tzinfo=None)
                    ).total_seconds()
                    if elapsed < _DRIFT_MIN_INTERVAL_SECONDS:
                        should_log = False
            except Exception:
                pass  # if we can't read, just write

    if should_log:
        log_row = pd.DataFrame(
            [
                {
                    "logged_at": metrics["computed_at"],
                    "forecast_mae": metrics["forecast_mae"],
                    "baseline_mae": metrics["baseline_mae"],
                    "calibration_pct": metrics["calibration_pct"],
                    "drift_detected": metrics["drift_detected"],
                    "n_evaluated": metrics["n_evaluated"],
                }
            ]
        )
        with _lock:
            if DRIFT_LOG_PATH.exists():
                log_row.to_csv(DRIFT_LOG_PATH, mode="a", header=False, index=False)
            else:
                log_row.to_csv(DRIFT_LOG_PATH, index=False)

    return metrics


def get_drift_history() -> pd.DataFrame:
    if not DRIFT_LOG_PATH.exists():
        return pd.DataFrame()
    with _lock:
        df = pd.read_csv(DRIFT_LOG_PATH)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return df.sort_values("logged_at").reset_index(drop=True)


# Summary stats for UI dashboard


def get_forecast_summary() -> dict[str, Any]:
    """Return high-level forecast stats for the MLOps dashboard."""
    pred_df = get_prediction_log()
    if pred_df.empty:
        return {
            "total_forecasts": 0,
            "unique_skus": 0,
            "providers_used": [],
            "last_forecast_at": None,
        }

    query_df = get_query_log()
    providers = list(query_df["provider"].unique()) if not query_df.empty else []

    return {
        "total_forecasts": len(pred_df),
        "unique_skus": pred_df["sku_id"].nunique(),
        "providers_used": providers,
        "last_forecast_at": str(pred_df["logged_at"].max()),
    }
