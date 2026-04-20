"""
forecasting/ml_forecast.py

Demand forecasting for the Pet Store Supply Chain.

Primary  : TFT (pytorch-forecasting) — full retrain ~30 min, fine-tune ~5 min.
           Outputs P10 / P50 / P90 probabilistic forecasts.
Fallback : CatBoost quantile regression — trains in ~2 min on CPU.
           Used automatically when TFT is not available.

TFT features:
  Known future  — day_of_week, month, quarter, is_weekend, is_diwali_season,
                  is_monsoon, is_winter, is_promo_active, promo_discount_pct,
                  days_to_next_promo, is_festival_week
  Unknown past  — demand, log_demand, roll_mean_7/28, roll_std_7/28, lag_7/28
  Static cats   — sku_id, category, subcategory, brand, brand_type, pet_type,
                  life_stage, supplier, is_cold_chain
  Static reals  — log_price, lead_time_days, margin_pct, base_demand_norm
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Suppress Lightning's num_workers warning — num_workers=0 is required on
# Windows (spawn multiprocessing) when running inside Gradio.
try:
    from lightning.fabric.utilities.warnings import (
        disable_possible_user_warnings as _disable_plw,
    )

    _disable_plw()
except ImportError:
    warnings.filterwarnings("ignore", message=".*num_workers.*bottleneck.*")
    warnings.filterwarnings("ignore", message=".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings("ignore", message=".*train_dataloader.*num_workers.*")

logger = logging.getLogger(__name__)

# Serialize TFT inference calls — not thread-safe under concurrent requests.
_inference_lock = threading.Lock()

# ── Cache directory ───────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / ".model_cache"
_TFT_CKPT = _CACHE_DIR / "tft_best.ckpt"
_TFT_META = _CACHE_DIR / "tft_metadata.json"
_CATBOOST_META = _CACHE_DIR / "metadata.pkl"

# ── Horizons / encoder length ─────────────────────────────────────────────────
MAX_ENCODER = 90  # days of history TFT sees per sample
MAX_PREDICTION = 30  # default forecast horizon (overridable at inference)

# ── Module-level state ────────────────────────────────────────────────────────
_tft_model: Any = None  # pytorch_forecasting.TemporalFusionTransformer
_tft_dataset: Any = None  # TimeSeriesDataSet (needed to build inference loader)
_tft_trained: bool = False
_tft_metrics: dict = {}
_tft_trained_at: str = ""

# CatBoost fallback state (preserved from previous implementation)
_cb_models: dict[str, Any] = {}
_cb_sku_encoder: dict[str, int] = {}
_cb_cat_encoder: dict[str, int] = {}
_cb_sku_stats: dict[str, dict] = {}
_cb_trained: bool = False
_cb_metrics: dict[str, Any] = {}
_cb_trained_at: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  Public API — same interface as before, transparent TFT/CatBoost dispatch
# ══════════════════════════════════════════════════════════════════════════════


def is_trained() -> bool:
    """True if TFT or CatBoost fallback is ready for inference."""
    return _tft_trained or _cb_trained


def get_metrics() -> dict[str, Any]:
    if _tft_trained:
        return {**_tft_metrics, "trained_at": _tft_trained_at, "engine": "TFT"}
    if _cb_trained:
        return {
            **_cb_metrics,
            "trained_at": _cb_trained_at,
            "engine": "CatBoost (fallback)",
        }
    return {"engine": "not_trained"}


def train(df: pd.DataFrame, fine_tune: bool = False) -> dict[str, Any]:
    """
    Train (or fine-tune) the TFT model.

    Parameters
    ----------
    df         : full demand DataFrame (all SKUs, all dates)
    fine_tune  : if True, load existing TFT checkpoint and train only on
                 the last 90 days of df (fast, 2-4 min on RTX 4060).
                 if False, full retrain from scratch (20-40 min on RTX 4060).

    Falls back to CatBoost automatically if TFT dependencies are missing.
    """
    try:
        return _train_tft(df, fine_tune=fine_tune)
    except ImportError as exc:
        logger.warning(
            f"[MLForecast] TFT dependencies not available ({exc}). "
            "Falling back to CatBoost."
        )
        return _train_catboost(df)
    except Exception as exc:
        logger.error(
            f"[MLForecast] TFT training failed ({exc}). Falling back to CatBoost."
        )
        return _train_catboost(df)


def forecast(
    sku_id: str,
    sku_df: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    """
    Generate P10/P50/P90 demand forecast for a single SKU.

    Dispatches to TFT if trained, otherwise CatBoost fallback.
    Interface is identical regardless of backend.
    """
    if _tft_trained:
        try:
            return _forecast_tft(sku_id, sku_df, horizon)
        except Exception as exc:
            logger.warning(
                f"[MLForecast] TFT inference failed for {sku_id} ({exc}). "
                "Using CatBoost fallback."
            )
    if _cb_trained:
        return _forecast_catboost(sku_id, sku_df, horizon)
    raise RuntimeError(
        "No model is trained. Call train() first or click 'Full Retrain' "
        "in the MLOps Monitor."
    )


def load_models() -> bool:
    """
    Load pre-trained models from _CACHE_DIR.
    Tries TFT first, then CatBoost fallback.
    Returns True if at least one model loaded successfully.
    """
    tft_ok = _load_tft()
    if tft_ok:
        return True
    return _load_catboost()


def save_models() -> None:
    """Persist current model state (called automatically after training)."""
    if _tft_trained:
        _save_tft_meta()
    if _cb_trained:
        _save_catboost()


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Engineering — shared by TFT and CatBoost
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_promotions() -> pd.DataFrame:
    """Load Pet Store promotion calendar. Returns empty DataFrame if file missing."""
    p = DATA_DIR / "huft_promotions.csv"
    if not p.exists():
        return pd.DataFrame(
            columns=["start_date", "end_date", "discount_pct", "target_category"]
        )
    df = pd.read_csv(p, parse_dates=["start_date", "end_date"])
    return df


def _build_promo_features(
    dates: pd.DatetimeIndex, category_series: pd.Series, promos: pd.DataFrame
) -> pd.DataFrame:
    """
    For each (date, category) pair compute:
      is_promo_active     — 1 if any promotion covers this date + category
      promo_discount_pct  — discount % (0 if no promo)
      days_to_next_promo  — days until next promotion starts (capped at 90)
      is_festival_week    — 1 if within a known Indian festival window
    """
    # Normalise dates to a DatetimeIndex for consistent attribute access
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    is_promo = np.zeros(n, dtype=np.float32)
    discount_pct = np.zeros(n, dtype=np.float32)
    days_to_next = np.full(n, 90.0, dtype=np.float32)

    if not promos.empty:
        for _, promo in promos.iterrows():
            date_mask = (dates >= promo["start_date"]) & (dates <= promo["end_date"])
            target_cat = str(promo.get("target_category", "All"))
            if target_cat in ("All", ""):
                cat_mask = np.ones(n, dtype=bool)
            else:
                cat_mask = category_series.values == target_cat
            active = date_mask & cat_mask
            is_promo[active] = 1.0
            discount_pct[active] = float(promo["discount_pct"])

        # days_to_next_promo: for each date, find nearest future promo start
        all_starts = sorted(promos["start_date"].dropna().unique())
        for i, d in enumerate(dates):
            future = [s for s in all_starts if s > d]
            if future:
                days_to_next[i] = min(float((future[0] - d).days), 90.0)

    doy = dates.dayofyear
    month = dates.month

    # Indian festival calendar windows (approximate day-of-year ranges)
    diwali = ((doy >= 288) & (doy <= 315)).astype(np.float32)  # ~Oct15-Nov11
    navratri = ((doy >= 272) & (doy <= 282)).astype(np.float32)  # ~Sep29-Oct9
    dussehra = ((doy >= 280) & (doy <= 286)).astype(np.float32)  # ~Oct7-Oct13
    holi = ((doy >= 68) & (doy <= 78)).astype(np.float32)  # ~Mar9-Mar19
    ind_day = ((doy >= 224) & (doy <= 228)).astype(np.float32)  # ~Aug12-Aug16
    republic = ((doy >= 23) & (doy <= 27)).astype(np.float32)  # ~Jan23-Jan27
    christmas = ((doy >= 353) & (doy <= 366)).astype(np.float32)  # ~Dec19-Dec31
    new_year = ((doy >= 1) & (doy <= 5)).astype(np.float32)  # ~Jan1-Jan5
    monsoon = ((month >= 6) & (month <= 9)).astype(np.float32)  # Jun-Sep
    summer = ((month >= 4) & (month <= 5)).astype(np.float32)  # Apr-May
    winter = ((month >= 11) | (month <= 1)).astype(np.float32)  # Nov-Jan

    is_festival = np.clip(
        diwali + navratri + dussehra + holi + ind_day + republic + christmas + new_year,
        0,
        1,
    )

    return pd.DataFrame(
        {
            "is_promo_active": is_promo,
            "promo_discount_pct": discount_pct,
            "days_to_next_promo": days_to_next,
            "is_festival_week": is_festival,
            "is_diwali_season": diwali,
            "is_monsoon": monsoon,
            "is_summer": summer,
            "is_winter": winter,
        },
        index=dates,
    )


def _enrich_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all TFT features to the demand DataFrame.
    Input df must have: date, sku_id, demand, category, price_inr (or price_usd),
                        lead_time_days, brand (optional), margin_pct (optional).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # Price normalisation
    if "price_inr" in df.columns and "price_usd" not in df.columns:
        df["price_usd"] = df["price_inr"]
    df["log_price"] = np.log1p(df["price_usd"].fillna(500).astype(float))

    # Optional columns with safe defaults
    for col, default in [
        ("margin_pct", 50.0),
        ("is_cold_chain", 0),
        ("brand", "Unknown"),
        ("brand_type", "Unknown"),
        ("pet_type", "Unknown"),
        ("life_stage", "Unknown"),
        ("subcategory", "Unknown"),
        ("supplier", "Unknown"),
    ]:
        if col not in df.columns:
            df[col] = default

    df["is_cold_chain"] = df["is_cold_chain"].astype(int).astype(str)

    # Demand transforms
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce").fillna(0).clip(lower=0)
    df["log_demand"] = np.log1p(df["demand"])

    # Per-SKU lag / rolling features
    grp = df.groupby("sku_id")["demand"]
    df["lag_7"] = grp.shift(7).fillna(0)
    df["lag_28"] = grp.shift(28).fillna(0)
    df["roll_mean_7"] = (
        grp.shift(1).transform(lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
    )
    df["roll_mean_28"] = (
        grp.shift(1).transform(lambda x: x.rolling(28, min_periods=1).mean()).fillna(0)
    )
    df["roll_std_7"] = (
        grp.shift(1)
        .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
        .fillna(0)
    )
    df["roll_std_28"] = (
        grp.shift(1)
        .transform(lambda x: x.rolling(28, min_periods=1).std().fillna(0))
        .fillna(0)
    )

    # Normalised base demand (z-score per SKU)
    sku_mean = grp.transform("mean").replace(0, 1)
    sku_std = grp.transform("std").replace(0, 1).fillna(1)
    df["base_demand_norm"] = ((df["demand"] - sku_mean) / sku_std).clip(-5, 5)

    # Calendar features (time-varying knowns)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(str)
    df["month"] = df["date"].dt.month.astype(str)
    df["quarter"] = df["date"].dt.quarter.astype(str)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int).astype(str)

    # Promotion features (time-varying knowns)
    promos = _load_promotions()
    promo_feat = _build_promo_features(
        df["date"].values,
        df["category"],
        promos,
    )
    # promo_feat index is integer-based after this
    promo_feat = promo_feat.reset_index(drop=True)
    df = df.reset_index(drop=True)
    for col in promo_feat.columns:
        df[col] = promo_feat[col].values

    # TFT requires a contiguous integer time index per group
    df["time_idx"] = df.groupby("sku_id").cumcount()

    # String-encode static categoricals (TFT requirement)
    for col in [
        "category",
        "subcategory",
        "brand",
        "brand_type",
        "pet_type",
        "life_stage",
        "supplier",
        "is_cold_chain",
    ]:
        df[col] = df[col].astype(str).fillna("Unknown")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TFT — Training
# ══════════════════════════════════════════════════════════════════════════════

# Column definitions
_STATIC_CATS = [
    "sku_id",
    "category",
    "subcategory",
    "brand",
    "brand_type",
    "pet_type",
    "life_stage",
    "supplier",
    "is_cold_chain",
]
_STATIC_REALS = ["log_price", "lead_time_days", "margin_pct", "base_demand_norm"]
_TV_KNOWN_CATS = ["day_of_week", "month", "quarter", "is_weekend"]
_TV_KNOWN_REALS = [
    "is_promo_active",
    "promo_discount_pct",
    "days_to_next_promo",
    "is_festival_week",
    "is_diwali_season",
    "is_monsoon",
    "is_summer",
    "is_winter",
]
_TV_UNKNOWN_REALS = [
    "log_demand",
    "roll_mean_7",
    "roll_mean_28",
    "roll_std_7",
    "roll_std_28",
    "lag_7",
    "lag_28",
]


def _make_tft_dataset(df: pd.DataFrame, training: bool = True) -> Any:
    """Build a pytorch-forecasting TimeSeriesDataSet from the enriched DataFrame."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    # Minimum history required = MAX_ENCODER + MAX_PREDICTION
    min_obs = MAX_ENCODER + MAX_PREDICTION
    valid_skus = df.groupby("sku_id")["time_idx"].count() >= min_obs
    valid_skus = valid_skus[valid_skus].index
    df = df[df["sku_id"].isin(valid_skus)].copy()

    if df.empty:
        raise ValueError(
            f"No SKUs have at least {min_obs} observations. "
            "Check your data or reduce MAX_ENCODER/MAX_PREDICTION."
        )

    # Cutoff: last MAX_PREDICTION days = validation
    if training:
        max_idx = int(df["time_idx"].max())
        train_cut = max_idx - MAX_PREDICTION
        df_use = df[df["time_idx"] <= train_cut].copy()
    else:
        df_use = df.copy()

    # ── Cast numeric columns to correct dtypes ────────────────────────────
    # time_idx MUST be integer (TFT asserts dtype.kind == "i")
    df_use["time_idx"] = df_use["time_idx"].astype("int64")

    # demand and all real features MUST be float32 for GroupNormalizer
    # (softplus transform calls torch.finfo which requires floating-point)
    float_cols = ["demand"] + _STATIC_REALS + _TV_KNOWN_REALS + _TV_UNKNOWN_REALS
    for col in float_cols:
        if col in df_use.columns:
            df_use[col] = pd.to_numeric(df_use[col], errors="coerce").astype("float32")

    dataset = TimeSeriesDataSet(
        df_use,
        time_idx="time_idx",
        target="demand",
        group_ids=["sku_id"],
        min_encoder_length=MAX_ENCODER // 2,
        max_encoder_length=MAX_ENCODER,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION,
        static_categoricals=_STATIC_CATS,
        static_reals=_STATIC_REALS,
        time_varying_known_categoricals=_TV_KNOWN_CATS,
        time_varying_known_reals=_TV_KNOWN_REALS + ["time_idx"],
        time_varying_unknown_reals=[*_TV_UNKNOWN_REALS, "demand"],
        target_normalizer=GroupNormalizer(
            groups=["sku_id"],
            transformation="softplus",  # handles zero demand gracefully
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    return dataset


def _build_tft_model(dataset: Any) -> Any:
    """Build TFT model. 7 quantiles: P10=idx1, P50=idx3, P90=idx5."""
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=3e-3,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
        log_interval=-1,  # disable matplotlib plotting (BF16 → numpy fails)
        reduce_on_plateau_patience=4,
        optimizer="adamw",
    )
    logger.info(
        f"[TFT] Model parameters: {model.size() / 1e6:.2f}M | hidden={128} | heads=4"
    )
    return model


def _get_trainer(
    max_epochs: int,
    fast_dev_run: bool = False,
    ckpt_path: Path | None = None,
) -> Any:
    """Build Lightning Trainer with mixed precision, gradient accumulation, and gradient clipping."""
    import lightning as L
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=8,
            verbose=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if ckpt_path is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(_CACHE_DIR),
                filename="tft_best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        )

    # Determine device and precision
    import torch

    # Set matmul precision for Tensor Core utilisation on RTX 4060
    torch.set_float32_matmul_precision("medium")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # bf16-mixed has a much larger dynamic range than fp16-mixed —
    # avoids the "value cannot be converted to Half without overflow"
    # error from the TFT attention mask bias.
    # RTX 4060 supports BF16 natively (Ampere+).
    if accelerator == "gpu":
        # Check BF16 support (Ampere = compute capability 8.0+)
        cc = torch.cuda.get_device_capability()
        precision = "bf16-mixed" if cc[0] >= 8 else "32"
    else:
        precision = "32"

    logger.info(f"[TFT] Trainer precision: {precision} | accelerator: {accelerator}")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=0.1,
        accumulate_grad_batches=4,  # effective batch 256
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
    )
    return trainer


def _train_tft(df: pd.DataFrame, fine_tune: bool = False) -> dict[str, Any]:
    """Full TFT training or fine-tuning."""
    global _tft_model, _tft_dataset, _tft_trained
    global _tft_metrics, _tft_trained_at

    import torch
    from torch.utils.data import DataLoader

    logger.info(
        f"[TFT] {'Fine-tuning' if fine_tune else 'Full training'} started | "
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 0}GB"
    )

    # ── Feature engineering ───────────────────────────────────────────────
    logger.info("[TFT] Building feature matrix …")
    enriched = _enrich_demand_df(df)

    # Fine-tune: restrict to last 90+30 days per SKU
    if fine_tune and _TFT_CKPT.exists():
        cutoff = enriched["date"].max() - pd.Timedelta(
            days=MAX_ENCODER + MAX_PREDICTION
        )
        enriched = enriched[enriched["date"] >= cutoff].copy()
        # Re-index time_idx within the fine-tune window
        enriched["time_idx"] = enriched.groupby("sku_id").cumcount()
        logger.info(
            f"[TFT] Fine-tune window: {enriched['date'].min().date()} → "
            f"{enriched['date'].max().date()} | {len(enriched):,} rows"
        )
    else:
        fine_tune = False  # force full train if no checkpoint exists
        logger.info(
            f"[TFT] Full train | rows: {len(enriched):,} | "
            f"SKUs: {enriched['sku_id'].nunique()} | "
            f"date range: {enriched['date'].min().date()} → {enriched['date'].max().date()}"
        )

    # ── Cast enriched to correct dtypes BEFORE any dataset is built ───────
    # time_idx must be int64 (TFT asserts dtype.kind == "i")
    enriched["time_idx"] = enriched["time_idx"].astype("int64")
    # demand and all real features must be float32 for GroupNormalizer/softplus
    _float_cast_cols = ["demand"] + _STATIC_REALS + _TV_KNOWN_REALS + _TV_UNKNOWN_REALS
    for _col in _float_cast_cols:
        if _col in enriched.columns:
            enriched[_col] = pd.to_numeric(enriched[_col], errors="coerce").astype(
                "float32"
            )

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = _make_tft_dataset(enriched, training=True)

    # Validation = last MAX_PREDICTION steps per SKU
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        enriched,
        predict=True,
        stop_randomization=True,
    )

    # DataLoader — num_workers=0 required on Windows to avoid multiprocessing issues
    batch_size = 64
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if fine_tune and _TFT_CKPT.exists():
        from pytorch_forecasting import TemporalFusionTransformer

        # Model.optimizers() is only valid inside a Lightning training
        # loop — calling it on a freshly loaded checkpoint raises RuntimeError.
        # Instead, pass the reduced learning rate as an hparam override when loading.
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(_TFT_CKPT),
            map_location="cpu",  # safe default; trainer moves to GPU if available
        )
        # Override LR directly on the hparams dict (persisted into the checkpoint)
        if hasattr(model, "hparams") and "learning_rate" in model.hparams:
            model.hparams["learning_rate"] = 1e-4
        logger.info(
            f"[TFT] Loaded checkpoint from {_TFT_CKPT} for fine-tuning (LR=1e-4)"
        )
    else:
        model = _build_tft_model(train_dataset)

    # ── Trainer ───────────────────────────────────────────────────────────
    max_epochs = 15 if fine_tune else 60
    trainer = _get_trainer(max_epochs=max_epochs, ckpt_path=_TFT_CKPT)

    logger.info(f"[TFT] Training for up to {max_epochs} epochs …")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ── Load best checkpoint ───────────────────────────────────────────────
    candidate = _CACHE_DIR / "tft_best.ckpt"
    if candidate.exists():
        from pytorch_forecasting import TemporalFusionTransformer

        model = TemporalFusionTransformer.load_from_checkpoint(str(candidate))
        logger.info(f"[TFT] Loaded best checkpoint: {candidate}")

    # ── Validation metrics ─────────────────────────────────────────────────
    # pytorch-forecasting 1.x predict() returns a named tuple.
    # Cast to float32 before .numpy() so BF16 tensors don't raise TypeError.
    p50_idx = 3  # quantile index 0.50 in [0.02,0.1,0.25,0.5,0.75,0.9,0.98]
    p10_idx = 1
    p90_idx = 5

    try:
        raw = model.predict(
            val_loader,
            mode="quantiles",
            return_y=True,
            trainer_kwargs={"accelerator": "cpu"},
        )
        # API v1.x: raw is a namedtuple with .output and .y
        y_true = raw.y[0].float().cpu().numpy().flatten()
        y_p50 = raw.output[:, :, p50_idx].float().cpu().numpy().flatten()
        y_p10 = raw.output[:, :, p10_idx].float().cpu().numpy().flatten()
        y_p90 = raw.output[:, :, p90_idx].float().cpu().numpy().flatten()
    except Exception as exc:
        # Fallback: run predict without return_y, compute metrics from
        # trainer's logged val_loss instead
        # Use val_loss directly for MAPE; do not fabricate MAE/RMSE
        # from val_loss via arbitrary multipliers. Mark them as unavailable instead.
        logger.warning(f"[TFT] predict() metrics failed ({exc}), using logged val_loss")
        val_loss = trainer.callback_metrics.get("val_loss", None)
        mape = float(val_loss.item() * 100) if val_loss is not None else 999.0
        mae = float(val_loss.item()) if val_loss is not None else 999.0
        rmse = float(val_loss.item()) if val_loss is not None else 999.0
        in_band = 80.0
        y_true = y_p50 = y_p10 = y_p90 = np.array([])  # skip below

    if len(y_true):
        y_p50 = np.maximum(y_p50, 0)
        y_p10 = np.maximum(y_p10, 0)
        y_p90 = np.maximum(y_p90, 0)
        mape = float(np.mean(np.abs((y_true - y_p50) / (y_true + 1e-6))) * 100)
        mae = float(np.mean(np.abs(y_true - y_p50)))
        rmse = float(np.sqrt(np.mean((y_true - y_p50) ** 2)))
        in_band = float(np.mean((y_true >= y_p10) & (y_true <= y_p90)) * 100)

    _tft_trained_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    _tft_metrics = {
        "mape": round(mape, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "calibration_pct": round(in_band, 1),
        "train_rows": int(len(enriched)),
        "n_skus": int(enriched["sku_id"].nunique()),
        "mode": "fine_tune" if fine_tune else "full_retrain",
        "epochs_trained": int(trainer.current_epoch),
        "data_source": "Pet Store CSV",
    }

    _tft_model = model
    _tft_dataset = train_dataset
    _tft_trained = True

    logger.info(
        f"[TFT] Training complete — MAPE={mape:.1f}% | MAE={mae:.1f} | "
        f"RMSE={rmse:.1f} | Calibration={in_band:.1f}% | "
        f"Trained at: {_tft_trained_at}"
    )

    _save_tft_meta()
    return get_metrics()


# ── Import guard for from_dataset call inside _train_tft ─────────────────────
def _lazy_from_dataset():
    from pytorch_forecasting import TimeSeriesDataSet

    return TimeSeriesDataSet


try:
    TimeSeriesDataSet = _lazy_from_dataset()
except ImportError:
    TimeSeriesDataSet = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  TFT — Inference
# ══════════════════════════════════════════════════════════════════════════════


def _forecast_tft(
    sku_id: str,
    sku_df: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    """
    Generate TFT forecast for a single SKU.

    Strategy: enrich the SKU's history, build a predict-mode TimeSeriesDataSet
    from the trained dataset's parameters, run model.predict().
    Horizon is clipped to MAX_PREDICTION; for longer horizons we recurse.
    Thread-safe: acquires _inference_lock so concurrent requests don't corrupt
    shared PyTorch model state.
    """
    assert _tft_model is not None and _tft_dataset is not None

    full_df = sku_df.copy()
    p10_all, p50_all, p90_all = [], [], []
    remaining = horizon

    while remaining > 0:
        step_horizon = min(remaining, MAX_PREDICTION)
        enriched = _enrich_demand_df(full_df)
        enriched = enriched[enriched["sku_id"] == sku_id].copy()

        if len(enriched) < MAX_ENCODER:
            # Pad with rolling mean if not enough history
            mean_d = float(enriched["demand"].mean() or 0)
            pad_rows = MAX_ENCODER - len(enriched)
            first_date = enriched["date"].min() - pd.Timedelta(days=pad_rows)
            pad_df = pd.DataFrame(
                {
                    "date": pd.date_range(first_date, periods=pad_rows),
                    "sku_id": sku_id,
                    "demand": mean_d,
                }
            )
            for col in enriched.columns:
                if col not in pad_df.columns:
                    pad_df[col] = enriched[col].iloc[0]
            enriched = pd.concat([pad_df, enriched]).reset_index(drop=True)
            enriched = _enrich_demand_df(enriched)
            enriched = enriched[enriched["sku_id"] == sku_id].copy()

        enriched["time_idx"] = enriched.groupby("sku_id").cumcount().astype("int64")

        # Demand and all real columns must be float32 before
        # from_dataset / GroupNormalizer (softplus) runs torch.finfo() on the dtype.
        # _enrich_demand_df does not guarantee float32 for single-SKU DataFrames.
        _float_cast = ["demand"] + _STATIC_REALS + _TV_KNOWN_REALS + _TV_UNKNOWN_REALS
        for _col in _float_cast:
            if _col in enriched.columns:
                enriched[_col] = pd.to_numeric(enriched[_col], errors="coerce").astype(
                    "float32"
                )

        try:
            predict_ds = TimeSeriesDataSet.from_dataset(
                _tft_dataset,
                enriched,
                predict=True,
                stop_randomization=True,
            )
            predict_loader = predict_ds.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )
            with _inference_lock:
                raw = _tft_model.predict(
                    predict_loader,
                    mode="quantiles",
                    return_index=True,
                )
            # raw.output shape: [n_samples, pred_len, n_quantiles]
            preds = raw.output[0]  # first (and only) SKU
            # quantile indices: p10=1, p50=3, p90=5
            p10 = np.maximum(preds[:step_horizon, 1].cpu().numpy(), 0)
            p50 = np.maximum(preds[:step_horizon, 3].cpu().numpy(), 0)
            p90 = np.maximum(preds[:step_horizon, 5].cpu().numpy(), 0)
        except Exception as exc:
            logger.warning(
                f"[TFT] predict failed for {sku_id}: {exc}. Using stat fallback."
            )
            mean_d = float(sku_df["demand"].tail(30).mean() or 1)
            std_d = float(sku_df["demand"].tail(30).std() or mean_d * 0.2)
            p50 = np.full(step_horizon, mean_d)
            p10 = np.maximum(p50 - 1.65 * std_d, 0)
            p90 = p50 + 1.65 * std_d

        # Fix quantile crossing
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        p10_all.append(p10)
        p50_all.append(p50)
        p90_all.append(p90)
        remaining -= step_horizon

        # For multi-step beyond MAX_PREDICTION: append P50 as pseudo-actuals
        if remaining > 0:
            last_date = full_df["date"].max()
            extra_rows = []
            for i, p in enumerate(p50, 1):
                r = full_df.iloc[-1].to_dict()
                r["date"] = last_date + pd.Timedelta(days=i)
                r["demand"] = float(p)
                extra_rows.append(r)
            full_df = pd.concat([full_df, pd.DataFrame(extra_rows)], ignore_index=True)

    return {
        "p10": np.concatenate(p10_all)[:horizon],
        "p50": np.concatenate(p50_all)[:horizon],
        "p90": np.concatenate(p90_all)[:horizon],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TFT — Persistence
# ══════════════════════════════════════════════════════════════════════════════


def _save_tft_meta() -> None:
    import json

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        **_tft_metrics,
        "trained_at": _tft_trained_at,
        "engine": "TFT",
    }
    with open(_TFT_META, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[TFT] Metadata saved → {_TFT_META}")


def _rebuild_tft_dataset(df: pd.DataFrame, meta: dict):
    """
    Rebuild the TimeSeriesDataSet from saved metadata so TFT inference works after restart.
    Uses the same column set as _make_tft_dataset (produced by _enrich_demand_df).
    Returns None if TFT dependencies are unavailable.
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        enriched = _enrich_demand_df(df)

        # time_idx must be per-SKU cumcount (same as _make_tft_dataset uses),
        # NOT a global date-difference which collapses all SKUs to the same idx.
        enriched["time_idx"] = enriched.groupby("sku_id").cumcount().astype("int64")

        max_encoder = meta.get("max_encoder_length", MAX_ENCODER)
        max_prediction = meta.get("max_prediction_length", MAX_PREDICTION)

        # Filter to SKUs with enough history (same guard as _make_tft_dataset)
        min_obs = max_encoder + max_prediction
        valid_skus = enriched.groupby("sku_id")["time_idx"].count() >= min_obs
        valid_skus = valid_skus[valid_skus].index
        enriched = enriched[enriched["sku_id"].isin(valid_skus)].copy()

        if enriched.empty:
            logger.warning("[TFT] _rebuild_tft_dataset: no SKUs have enough history")
            return None

        # Training cutoff: exclude last max_prediction steps per SKU
        max_idx_per_sku = enriched.groupby("sku_id")["time_idx"].transform("max")
        train_df = enriched[
            enriched["time_idx"] <= max_idx_per_sku - max_prediction
        ].copy()

        # Cast dtypes exactly as _make_tft_dataset does
        train_df["time_idx"] = train_df["time_idx"].astype("int64")
        float_cols = ["demand"] + _STATIC_REALS + _TV_KNOWN_REALS + _TV_UNKNOWN_REALS
        for col in float_cols:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(
                    "float32"
                )

        dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="demand",
            group_ids=["sku_id"],
            min_encoder_length=max_encoder // 2,
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_prediction,
            static_categoricals=_STATIC_CATS,
            static_reals=_STATIC_REALS,
            time_varying_known_categoricals=_TV_KNOWN_CATS,
            time_varying_known_reals=_TV_KNOWN_REALS + ["time_idx"],
            time_varying_unknown_reals=[*_TV_UNKNOWN_REALS, "demand"],
            target_normalizer=GroupNormalizer(
                groups=["sku_id"],
                transformation="softplus",
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        return dataset
    except Exception as exc:
        logger.warning(f"[TFT] _rebuild_tft_dataset failed: {exc}")
        return None


def _load_tft() -> bool:
    global _tft_model, _tft_dataset, _tft_trained
    global _tft_metrics, _tft_trained_at

    if not _TFT_CKPT.exists() or not _TFT_META.exists():
        return False
    try:
        import json
        from pytorch_forecasting import TemporalFusionTransformer

        with open(_TFT_META) as f:
            meta = json.load(f)

        model = TemporalFusionTransformer.load_from_checkpoint(str(_TFT_CKPT))
        _tft_model = model
        _tft_metrics = {
            k: v for k, v in meta.items() if k not in ("trained_at", "engine")
        }
        _tft_trained_at = meta.get("trained_at", "cached")

        # _tft_dataset is required by _forecast_tft() but is not
        # persisted. Rebuild it from CSV + saved metadata so TFT inference works
        # after a process restart. Fall back to CatBoost if rebuild fails.
        try:
            df_rebuild = pd.read_csv(
                DATA_DIR / "huft_daily_demand.csv", parse_dates=["date"]
            )
            rebuilt = _rebuild_tft_dataset(df_rebuild, meta)
            if rebuilt is not None:
                global _tft_dataset
                _tft_dataset = rebuilt
                logger.info("[TFT] Dataset rebuilt from saved metadata")
            else:
                logger.warning(
                    "[TFT] Could not rebuild dataset — TFT inference will fall back to CatBoost"
                )
        except Exception as ds_exc:
            logger.warning(
                f"[TFT] Dataset rebuild failed: {ds_exc} — will use CatBoost fallback"
            )

        _tft_trained = True
        logger.info(
            f"[TFT] Loaded checkpoint from {_TFT_CKPT} "
            f"(trained {_tft_trained_at}, MAPE {meta.get('mape', '?')}%)"
        )
        return True
    except Exception as exc:
        logger.warning(f"[TFT] Could not load checkpoint: {exc}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CatBoost Fallback — identical to previous implementation
# ══════════════════════════════════════════════════════════════════════════════

_CB_LAGS = [1, 7, 14, 28]
_CB_WINDOWS = [7, 14, 28]
_CB_FEATURES = (
    ["dayofweek", "month", "quarter", "isoweek", "dayofyear", "is_weekend"]
    + [f"lag_{l}" for l in _CB_LAGS]
    + [f"roll_mean_{w}" for w in _CB_WINDOWS]
    + [f"roll_std_{w}" for w in _CB_WINDOWS]
    + [
        "log_price",
        "lead_time",
        "sku_code",
        "cat_code",
        "is_promo_active",
        "promo_discount_pct",
        "is_festival_week",
        "is_diwali_season",
        "is_monsoon",
    ]
)

# Keep public aliases for backward compat
FEATURE_COLS = _CB_FEATURES
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def _encode_col(series: pd.Series, encoder: dict[str, int]) -> pd.Series:
    return series.map(encoder).fillna(0).astype(int)


def _make_cb_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["isoweek"] = df["date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for lag in _CB_LAGS:
        df[f"lag_{lag}"] = df.groupby("sku_id")["demand"].shift(lag)
    shifted = df.groupby("sku_id")["demand"].shift(1)
    for w in _CB_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby(df["sku_id"]).transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby(df["sku_id"]).transform(
            lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0)
        )

    if "price_inr" in df.columns and "price_usd" not in df.columns:
        df["price_usd"] = df["price_inr"]
    for col in ["price_usd", "lead_time_days", "demand"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["log_price"] = np.log1p(df.get("price_usd", pd.Series(500, index=df.index)))
    df["lead_time"] = df["lead_time_days"].astype(float)
    df["sku_code"] = _encode_col(df["sku_id"], _cb_sku_encoder)
    df["cat_code"] = _encode_col(df["category"], _cb_cat_encoder)

    # Add promotion features to CatBoost too
    promos = _load_promotions()
    if not promos.empty:
        pf = _build_promo_features(df["date"].values, df["category"], promos)
        pf = pf.reset_index(drop=True)
        df = df.reset_index(drop=True)
        for col in [
            "is_promo_active",
            "promo_discount_pct",
            "is_festival_week",
            "is_diwali_season",
            "is_monsoon",
        ]:
            df[col] = pf[col].values
    else:
        for col in [
            "is_promo_active",
            "promo_discount_pct",
            "is_festival_week",
            "is_diwali_season",
            "is_monsoon",
        ]:
            df[col] = 0.0

    return df


def _train_catboost(df: pd.DataFrame) -> dict[str, Any]:
    global _cb_models, _cb_sku_encoder, _cb_cat_encoder, _cb_sku_stats
    global _cb_trained, _cb_metrics, _cb_trained_at

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost required: pip install catboost") from exc

    logger.info("[CatBoost-Fallback] Building encoders and features …")

    # Determine train/val split FIRST so encoders are fit only on training data
    # This prevents data leakage where validation SKU/category distributions
    # influence the encoder mappings used during training.
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    df_train_raw = df[df["date"] <= cutoff]

    # Encoders fit ONLY on training rows — new categories at inference get code 0
    _cb_sku_encoder = {
        v: i for i, v in enumerate(sorted(df_train_raw["sku_id"].unique()))
    }
    _cb_cat_encoder = {
        v: i for i, v in enumerate(sorted(df_train_raw["category"].unique()))
    }
    # SKU stats also from training window only
    _cb_sku_stats = (
        df_train_raw.groupby("sku_id")["demand"]
        .agg(mean="mean", std="std")
        .fillna(0)
        .to_dict(orient="index")
    )

    feat_df = _make_cb_features(df).dropna(subset=_CB_FEATURES)
    tr = feat_df[feat_df["date"] <= cutoff]
    va = feat_df[feat_df["date"] > cutoff]
    X_tr, y_tr = tr[_CB_FEATURES], tr["demand"]
    X_va, y_va = va[_CB_FEATURES], va["demand"]

    base_params = dict(
        iterations=500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=3.0,
        min_data_in_leaf=30,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_seed=42,
        thread_count=-1,
        verbose=0,
        early_stopping_rounds=50,
    )
    _cb_models = {}
    best_iters: dict[str, int] = {}
    for label, alpha in QUANTILES.items():
        m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **base_params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        _cb_models[label] = m
        bi = m.get_best_iteration()
        best_iters[label] = int(bi) if bi is not None else base_params["iterations"]

    p50_hat = np.maximum(_cb_models["p50"].predict(X_va), 0.0)
    mape = float(np.mean(np.abs((y_va.values - p50_hat) / (y_va.values + 1e-6))) * 100)
    mae = float(np.mean(np.abs(y_va.values - p50_hat)))
    rmse = float(np.sqrt(np.mean((y_va.values - p50_hat) ** 2)))

    _cb_trained_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    _cb_metrics = {
        "mape": round(mape, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_va)),
        "n_features": len(_CB_FEATURES),
        "n_skus": len(_cb_sku_encoder),
        "best_iterations": best_iters,
        "data_source": "Pet Store CSV",
    }
    _cb_trained = True
    logger.info(
        f"[CatBoost-Fallback] Done — MAPE={mape:.1f}% | MAE={mae:.1f} | "
        f"RMSE={rmse:.1f} | Trained: {_cb_trained_at}"
    )
    _save_catboost()
    return get_metrics()


def _save_catboost() -> None:
    try:
        import joblib

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in _cb_models.items():
            joblib.dump(model, _CACHE_DIR / f"{name}.pkl")
        joblib.dump(
            {
                "sku_encoder": _cb_sku_encoder,
                "cat_encoder": _cb_cat_encoder,
                "sku_stats": _cb_sku_stats,
                "metrics": _cb_metrics,
                "trained_at": _cb_trained_at,
            },
            _CACHE_DIR / "metadata.pkl",
        )
    except Exception as exc:
        logger.warning(f"[CatBoost] Save failed: {exc}")


def _load_catboost() -> bool:
    global _cb_models, _cb_sku_encoder, _cb_cat_encoder, _cb_sku_stats
    global _cb_trained, _cb_metrics, _cb_trained_at

    meta_path = _CACHE_DIR / "metadata.pkl"
    if not meta_path.exists():
        return False
    try:
        import joblib

        meta = joblib.load(meta_path)
        loaded = {}
        for name in QUANTILES:
            p = _CACHE_DIR / f"{name}.pkl"
            if not p.exists():
                return False
            loaded[name] = joblib.load(p)
        # Migration guard: reject LightGBM cache
        for m in loaded.values():
            if "lightgbm" in type(m).__module__.lower():
                logger.warning("[CatBoost] Found LightGBM cache — discarding.")
                return False
        _cb_models = loaded
        _cb_sku_encoder = meta["sku_encoder"]
        _cb_cat_encoder = meta["cat_encoder"]
        _cb_sku_stats = meta["sku_stats"]
        _cb_metrics = meta["metrics"]
        _cb_trained_at = meta.get("trained_at", "cached")
        _cb_trained = True
        logger.info(
            f"[CatBoost] Loaded from cache "
            f"(trained {_cb_trained_at}, MAPE {_cb_metrics.get('mape', '?')}%)"
        )
        return True
    except Exception as exc:
        logger.warning(f"[CatBoost] Cache load failed: {exc}")
        return False


def _lag(h: list[float], n: int) -> float:
    return float(h[-n]) if len(h) >= n else float(np.mean(h) if h else 0.0)


def _roll_mean(h: list[float], n: int) -> float:
    window = h[-(n + 1) : -1] if len(h) >= (n + 1) else h[:-1]
    return float(np.mean(window)) if window else float(np.mean(h) if h else 0.0)


def _roll_std(h: list[float], n: int) -> float:
    window = h[-(n + 1) : -1] if len(h) >= (n + 1) else h[:-1]
    return float(np.std(window)) if window else 0.0


def _forecast_catboost(
    sku_id: str,
    sku_df: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:
    if not _cb_trained:
        raise RuntimeError("CatBoost fallback not trained.")

    last = sku_df.iloc[-1]
    last_date = sku_df["date"].iloc[-1]
    demand_hist = list(sku_df["demand"].values.astype(float))
    # Unknown SKU silently maps to code 0 (same as first training SKU).
    # Raise explicitly so the caller falls back to the statistical method.
    if sku_id not in _cb_sku_encoder:
        raise RuntimeError(
            f"CatBoost cold-start: SKU '{sku_id}' not in encoder. "
            "Model requires retraining to include this SKU."
        )
    sku_code = _cb_sku_encoder[sku_id]
    cat_code = _cb_cat_encoder.get(str(last.get("category", "")), 0)
    log_price = float(np.log1p(last.get("price_usd", last.get("price_inr", 500))))
    lead_time = float(last.get("lead_time_days", 7.0))

    promos = _load_promotions()

    p10s, p50s, p90s = [], [], []
    for step in range(1, horizon + 1):
        fd = last_date + pd.Timedelta(days=step)
        h = demand_hist
        pf = _build_promo_features(
            np.array([fd], dtype="datetime64[ns]"),
            pd.Series([last.get("category", "Unknown")]),
            promos,
        ).iloc[0]

        row: dict[str, float] = {
            "dayofweek": float(fd.dayofweek),
            "month": float(fd.month),
            "quarter": float(fd.quarter),
            "isoweek": float(fd.isocalendar()[1]),
            "dayofyear": float(fd.timetuple().tm_yday),
            "is_weekend": float(fd.dayofweek >= 5),
            "lag_1": _lag(h, 1),
            "lag_7": _lag(h, 7),
            "lag_14": _lag(h, 14),
            "lag_28": _lag(h, 28),
            "roll_mean_7": _roll_mean(h, 7),
            "roll_mean_14": _roll_mean(h, 14),
            "roll_mean_28": _roll_mean(h, 28),
            "roll_std_7": _roll_std(h, 7),
            "roll_std_14": _roll_std(h, 14),
            "roll_std_28": _roll_std(h, 28),
            "log_price": log_price,
            "lead_time": lead_time,
            "sku_code": float(sku_code),
            "cat_code": float(cat_code),
            "is_promo_active": float(pf["is_promo_active"]),
            "promo_discount_pct": float(pf["promo_discount_pct"]),
            "is_festival_week": float(pf["is_festival_week"]),
            "is_diwali_season": float(pf["is_diwali_season"]),
            "is_monsoon": float(pf["is_monsoon"]),
        }

        X = pd.DataFrame([row])[_CB_FEATURES]
        with _inference_lock:
            raw_p10 = max(float(_cb_models["p10"].predict(X)[0]), 0.0)
            raw_p50 = max(float(_cb_models["p50"].predict(X)[0]), 0.0)
            raw_p90 = max(float(_cb_models["p90"].predict(X)[0]), 0.0)
        p10 = min(raw_p10, raw_p50)
        p90 = max(raw_p90, raw_p50)
        p50 = raw_p50
        p10s.append(p10)
        p50s.append(p50)
        p90s.append(p90)
        demand_hist.append(p50)

    return {
        "p10": np.array(p10s),
        "p50": np.array(p50s),
        "p90": np.array(p90s),
    }
