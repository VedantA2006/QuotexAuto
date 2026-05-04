"""
train.py — Main training entry point.

Orchestrates data download, feature engineering, walk-forward training,
backtesting, and report generation.

Usage:
    python train.py --asset EURUSD --timeframe 1M
    python train.py --asset GBPUSD --timeframe 5M --min-confidence 0.65
    python train.py --skip-download  # if data already cached
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ASSETS,
    TIMEFRAME,
    MIN_CONFIDENCE,
    DATA_DIR,
    MODEL_DIR,
    BASE_DIR,
)
from utils.logger import setup_logger
from utils.helpers import resample_to_timeframe

logger = setup_logger(
    "binary_predictor",
    log_file=BASE_DIR / "logs" / "app.log",
    level="INFO",
)


def main():
    parser = argparse.ArgumentParser(
        description="Binary Options Candle Direction Predictor — Training Pipeline"
    )
    parser.add_argument("--asset", type=str, default="EURUSD",
                        choices=["EURUSD", "GBPUSD", "USDJPY"],
                        help="Currency pair to train on")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME,
                        choices=["1M", "5M"],
                        help="Candle timeframe")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help="Minimum confidence threshold")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use cached CSV)")
    parser.add_argument("--years", type=int, default=2,
                        help="Years of historical data to fetch")
    args = parser.parse_args()

    pair = args.asset.upper()
    timeframe = args.timeframe
    min_conf = args.min_confidence

    logger.info("=" * 70)
    logger.info("  BINARY OPTIONS CANDLE DIRECTION PREDICTOR — TRAINING")
    logger.info(f"  Asset: {pair} | Timeframe: {timeframe} | Min Confidence: {min_conf}")
    logger.info("=" * 70)

    start_time = time.time()

    # ══════════════════════════════════════════════
    # STEP 1: Download / load historical data
    # ══════════════════════════════════════════════
    logger.info("\n📥 STEP 1: Loading historical data...")

    from data.fetcher_historical import download_historical, load_cached_data

    if args.skip_download:
        df_raw = load_cached_data(pair)
        if df_raw is None:
            logger.error(f"No cached data found for {pair}. Run without --skip-download")
            sys.exit(1)
    else:
        df_raw = download_historical(pair, years=args.years, save=True)

    if df_raw is None or df_raw.empty:
        logger.error("Failed to obtain historical data. Exiting.")
        sys.exit(1)

    logger.info(f"Raw data: {len(df_raw):,} candles from {df_raw['time'].min()} to {df_raw['time'].max()}")

    # Resample if needed
    if timeframe == "5M":
        df_raw = resample_to_timeframe(df_raw, "5T")
        logger.info(f"Resampled to 5-min: {len(df_raw):,} candles")

    # ══════════════════════════════════════════════
    # STEP 2: Feature engineering
    # ══════════════════════════════════════════════
    logger.info("\n🔧 STEP 2: Engineering features...")

    from features.engineer import engineer_features, get_feature_columns

    df = engineer_features(df_raw, add_target_col=True, drop_na=True)
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    logger.info(f"Engineered data: {len(df):,} valid rows, {len(feature_cols)} features")

    # ══════════════════════════════════════════════
    # STEP 3: Walk-forward training
    # ══════════════════════════════════════════════
    logger.info("\n🧠 STEP 3: Walk-forward model training...")

    from models.trainer import run_walk_forward_training

    results = run_walk_forward_training(df, feature_cols)

    if not results.get("fold_results"):
        logger.error("Training failed — no fold results produced.")
        sys.exit(1)

    # ══════════════════════════════════════════════
    # STEP 4: Run backtest
    # ══════════════════════════════════════════════
    logger.info("\n📊 STEP 4: Running backtest...")

    from backtest.engine import BacktestEngine

    engine = BacktestEngine(min_confidence=min_conf, pair=pair)
    bt_metrics = engine.run_from_fold_results(results["fold_results"])

    # ══════════════════════════════════════════════
    # STEP 5: Generate HTML report
    # ══════════════════════════════════════════════
    logger.info("\n📄 STEP 5: Generating backtest report...")

    from backtest.report import generate_html_report

    report_path = generate_html_report(
        metrics=bt_metrics,
        fold_results=results["fold_results"],
        feature_importance=results.get("feature_importance"),
        pair=pair,
        timeframe=timeframe,
        output_path=BASE_DIR / "backtest" / f"report_{pair}_{timeframe}.html",
    )

    # ══════════════════════════════════════════════
    # STEP 6: Save trained models
    # ══════════════════════════════════════════════
    logger.info("\n💾 STEP 6: Saving trained models...")

    best_ensemble = results.get("best_ensemble")
    if best_ensemble:
        best_ensemble.save(MODEL_DIR)

    # Save scaler
    best_scaler = results.get("best_scaler")
    if best_scaler:
        import joblib
        joblib.dump(best_scaler, MODEL_DIR / "scaler.pkl")

    # Save feature importance
    feat_imp = results.get("feature_importance")
    if feat_imp is not None and not feat_imp.empty:
        feat_imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # ══════════════════════════════════════════════
    # STEP 7: Final summary
    # ══════════════════════════════════════════════
    elapsed = time.time() - start_time
    agg = results.get("aggregate", {})

    logger.info("\n" + "═" * 70)
    logger.info("  TRAINING COMPLETE")
    logger.info("═" * 70)
    logger.info(f"  Asset:              {pair}")
    logger.info(f"  Timeframe:          {timeframe}")
    logger.info(f"  Raw Accuracy:       {agg.get('raw_accuracy', 0):.4f}")
    logger.info(f"  Filtered Accuracy:  {agg.get('filtered_accuracy', 0):.4f}")
    logger.info(f"  Trade Rate:         {agg.get('trade_rate', 0):.1%}")
    logger.info(f"  Total Folds:        {agg.get('n_folds', 0)}")
    logger.info(f"  Backtest Win Rate:  {bt_metrics.get('win_rate', 0):.2%}")
    logger.info(f"  Backtest Trades:    {bt_metrics.get('total_trades', 0)}")
    logger.info(f"  Net P&L:            ${bt_metrics.get('net_pnl', 0):,.2f}")
    logger.info(f"  Profit Factor:      {bt_metrics.get('profit_factor', 0):.3f}")
    logger.info(f"  Max Drawdown:       {bt_metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  Final Balance:      ${bt_metrics.get('final_balance', 0):,.2f}")
    logger.info(f"  Time Elapsed:       {elapsed:.1f}s")
    logger.info(f"  Report:             {report_path}")
    logger.info(f"  Models:             {MODEL_DIR}")
    logger.info("═" * 70)


if __name__ == "__main__":
    main()
