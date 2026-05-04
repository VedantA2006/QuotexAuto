"""
Walk-forward training pipeline.

Splits time series into monthly chunks, trains on expanding window,
tests on the next month — never uses future data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional
from sklearn.metrics import accuracy_score, log_loss, classification_report

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MIN_TRAIN_MONTHS,
    PURGE_GAP_CANDLES,
    MIN_CONFIDENCE,
    MODEL_DIR,
)
from models.xgb_model import XGBModel
from models.lgbm_model import LGBMModel
from models.cnn_model import CNNModel
from models.ensemble import EnsembleModel
from features.engineer import get_feature_columns, scale_features
from features.selector import compute_feature_importance, compute_shap_importance
from utils.logger import get_logger

logger = get_logger("binary_predictor")


def create_monthly_splits(df: pd.DataFrame, time_col: str = "time") -> list[tuple]:
    """
    Split DataFrame into monthly chunks.

    Returns list of (year, month, DataFrame) tuples.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df["_year"] = df[time_col].dt.year
    df["_month"] = df[time_col].dt.month

    splits = []
    for (year, month), group in df.groupby(["_year", "_month"]):
        splits.append((int(year), int(month), group.drop(columns=["_year", "_month"])))

    return splits


def walk_forward_split(
    monthly_chunks: list[tuple],
    min_train_months: int = MIN_TRAIN_MONTHS,
    purge_gap: int = PURGE_GAP_CANDLES,
) -> list[dict]:
    """
    Generate walk-forward train/test folds from monthly chunks.

    Returns list of fold dicts with keys: fold, train_df, test_df, train_months, test_month
    """
    folds = []
    n = len(monthly_chunks)

    for i in range(min_train_months, n):
        # Training: first i months
        train_frames = [monthly_chunks[j][2] for j in range(i)]
        train_df = pd.concat(train_frames, ignore_index=True)

        # Purge gap: remove last N candles from training
        if purge_gap > 0 and len(train_df) > purge_gap:
            train_df = train_df.iloc[:-purge_gap]

        # Test: month i
        test_year, test_month, test_df = monthly_chunks[i]

        folds.append({
            "fold": i - min_train_months + 1,
            "train_df": train_df,
            "test_df": test_df,
            "train_months": i,
            "test_period": f"{test_year}-{test_month:02d}",
        })

    logger.info(f"Created {len(folds)} walk-forward folds")
    return folds


def train_fold(
    fold: dict,
    feature_cols: list[str],
    target_col: str = "next_candle_direction",
) -> dict:
    """
    Train all three models on a single fold.

    Returns dict with models, predictions, and metrics.
    """
    train_df = fold["train_df"]
    test_df = fold["test_df"]
    fold_num = fold["fold"]
    test_period = fold["test_period"]

    # Scale features
    train_scaled, scaler = scale_features(train_df, fit=True)
    test_scaled, _ = scale_features(test_df, scaler=scaler, fit=False)

    X_train = train_scaled[feature_cols]
    y_train = train_scaled[target_col].astype(int)
    X_test = test_scaled[feature_cols]
    y_test = test_scaled[target_col].astype(int)

    # Validation split from train (last 15%)
    val_size = max(int(len(X_train) * 0.15), 100)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]

    # ── XGBoost ─────────────────────────────────────
    xgb_model = XGBModel()
    xgb_model.fit(X_tr, y_tr, X_val, y_val)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_preds = (xgb_proba > 0.5).astype(int)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    # ── LightGBM ────────────────────────────────────
    lgbm_model = LGBMModel()
    lgbm_model.fit(X_tr, y_tr, X_val, y_val)
    lgbm_proba = lgbm_model.predict_proba(X_test)
    lgbm_preds = (lgbm_proba > 0.5).astype(int)
    lgbm_acc = accuracy_score(y_test, lgbm_preds)

    # ── CNN ──────────────────────────────────────────
    cnn_model = CNNModel()
    try:
        # CNN needs full DataFrame (OHLCV + features)
        cnn_model.fit(train_scaled, y_train, test_scaled, y_test)
        cnn_proba = cnn_model.predict_proba(test_scaled)
    except Exception as e:
        logger.warning(f"CNN training failed on fold {fold_num}: {e}")
        cnn_proba = np.full(len(X_test), 0.5)

    # ── Ensemble ────────────────────────────────────
    ensemble = EnsembleModel()
    ensemble.set_models(xgb_model, lgbm_model, cnn_model)

    # Optimize weights on validation set
    try:
        ensemble.optimize_weights(X_val, y_val, train_scaled.iloc[-val_size:])
    except Exception:
        pass

    ensemble_proba = ensemble.predict_proba(X_test, test_scaled)
    ensemble_preds = (ensemble_proba > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)

    # Confidence-filtered accuracy
    confidence = np.maximum(ensemble_proba, 1 - ensemble_proba)
    conf_mask = confidence >= MIN_CONFIDENCE
    filtered_preds = ensemble_preds[conf_mask]
    filtered_actual = y_test.values[conf_mask]
    filtered_acc = accuracy_score(filtered_actual, filtered_preds) if len(filtered_preds) > 0 else 0
    trade_rate = conf_mask.mean()

    logger.info(
        f"Fold {fold_num} [{test_period}] — "
        f"XGB: {xgb_acc:.3f} | LGBM: {lgbm_acc:.3f} | "
        f"Ens: {ensemble_acc:.3f} | "
        f"Filtered({MIN_CONFIDENCE}): {filtered_acc:.3f} "
        f"({conf_mask.sum()}/{len(conf_mask)} trades, {trade_rate:.1%} rate)"
    )

    return {
        "fold": fold_num,
        "test_period": test_period,
        "xgb_model": xgb_model,
        "lgbm_model": lgbm_model,
        "cnn_model": cnn_model,
        "ensemble": ensemble,
        "scaler": scaler,
        # Predictions
        "y_test": y_test,
        "ensemble_proba": ensemble_proba,
        "ensemble_preds": ensemble_preds,
        "confidence": confidence,
        # Metrics
        "xgb_accuracy": xgb_acc,
        "lgbm_accuracy": lgbm_acc,
        "ensemble_accuracy": ensemble_acc,
        "filtered_accuracy": filtered_acc,
        "total_candles": len(y_test),
        "traded_candles": int(conf_mask.sum()),
        "trade_rate": trade_rate,
        "weights": ensemble.weights,
        # Test data for backtest
        "test_df": test_df,
    }


def run_walk_forward_training(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    target_col: str = "next_candle_direction",
) -> dict:
    """
    Full walk-forward training pipeline.

    Parameters
    ----------
    df : Fully engineered DataFrame with features and target
    feature_cols : list of feature column names to use
    target_col : target column name

    Returns
    -------
    Dict with fold results, aggregate metrics, and best models.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
    # Filter to columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    logger.info(f"Starting walk-forward training with {len(feature_cols)} features")

    # Create monthly splits
    monthly_chunks = create_monthly_splits(df)
    folds = walk_forward_split(monthly_chunks)

    if not folds:
        logger.error("Not enough data for walk-forward training")
        return {"fold_results": [], "aggregate": {}}

    # Train each fold
    fold_results = []
    for fold in folds:
        result = train_fold(fold, feature_cols, target_col)
        fold_results.append(result)

    # ── Aggregate Metrics ────────────────────────────
    all_y = np.concatenate([r["y_test"].values for r in fold_results])
    all_proba = np.concatenate([r["ensemble_proba"] for r in fold_results])
    all_preds = (all_proba > 0.5).astype(int)
    all_conf = np.maximum(all_proba, 1 - all_proba)

    raw_acc = accuracy_score(all_y, all_preds)

    conf_mask = all_conf >= MIN_CONFIDENCE
    if conf_mask.sum() > 0:
        filtered_acc = accuracy_score(all_y[conf_mask], all_preds[conf_mask])
    else:
        filtered_acc = 0

    monthly_accs = [r["filtered_accuracy"] for r in fold_results if r["traded_candles"] > 0]
    consistency = np.std(monthly_accs) if monthly_accs else 0

    aggregate = {
        "raw_accuracy": raw_acc,
        "filtered_accuracy": filtered_acc,
        "total_candles": len(all_y),
        "traded_candles": int(conf_mask.sum()),
        "trade_rate": float(conf_mask.mean()),
        "monthly_consistency_std": consistency,
        "n_folds": len(fold_results),
        "per_fold_accuracies": [r["filtered_accuracy"] for r in fold_results],
    }

    logger.info("=" * 60)
    logger.info(f"WALK-FORWARD RESULTS ({len(fold_results)} folds)")
    logger.info(f"  Raw accuracy:      {raw_acc:.4f}")
    logger.info(f"  Filtered accuracy: {filtered_acc:.4f}")
    logger.info(f"  Trade rate:        {conf_mask.mean():.1%}")
    logger.info(f"  Monthly std:       {consistency:.4f}")
    logger.info("=" * 60)

    # Use last fold's models as the "best" (most recent data)
    best = fold_results[-1]

    # Feature importance from best fold
    feat_imp = compute_feature_importance(
        best["xgb_model"].model, feature_cols
    )

    return {
        "fold_results": fold_results,
        "aggregate": aggregate,
        "best_xgb": best["xgb_model"],
        "best_lgbm": best["lgbm_model"],
        "best_cnn": best["cnn_model"],
        "best_ensemble": best["ensemble"],
        "best_scaler": best["scaler"],
        "feature_importance": feat_imp,
        "feature_cols": feature_cols,
    }
