"""
train.py -- Main training entry point (multi-pair, multi-timeframe).

Orchestrates data download, feature engineering, walk-forward training,
backtesting, and report generation for all pair x timeframe combinations.

Usage:
    python train.py --asset EURUSD --timeframe 1M
    python train.py --asset EURUSD GBPUSD USDJPY --timeframe 1M 5M
    python train.py --asset EURUSD GBPUSD USDJPY AUDUSD USDCAD EURJPY GBPJPY EURCAD --timeframe 1M 5M
    python train.py --skip-download
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
    TIMEFRAMES,
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

ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY", "EURCAD"]


def train_single(pair: str, timeframe: str, min_conf: float,
                 skip_download: bool, years: int) -> dict:
    """
    Train a single pair on a single timeframe.
    Returns a summary dict with metrics.
    """
    logger.info("=" * 70)
    logger.info(f"  TRAINING: {pair} | {timeframe} | Min Confidence: {min_conf}")
    logger.info("=" * 70)

    start_time = time.time()

    # --- Model save directory (timeframe-specific) ---
    save_dir = BASE_DIR / "models" / f"saved_{timeframe}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # STEP 1: Download / load historical data
    # ==================================================
    logger.info("\n>> STEP 1: Loading historical data...")

    from data.fetcher_historical import download_historical, load_cached_data

    if skip_download:
        df_raw = load_cached_data(pair)
        if df_raw is None:
            logger.error(f"No cached data found for {pair}. Run without --skip-download")
            return {"pair": pair, "timeframe": timeframe, "status": "FAILED", "reason": "No cached data"}
    else:
        df_raw = download_historical(pair, years=years, save=True)

    if df_raw is None or df_raw.empty:
        logger.error(f"Failed to obtain historical data for {pair}.")
        return {"pair": pair, "timeframe": timeframe, "status": "FAILED", "reason": "No data"}

    logger.info(f"Raw data: {len(df_raw):,} candles from {df_raw['time'].min()} to {df_raw['time'].max()}")

    # Resample if needed
    if timeframe == "5M":
        df_raw = resample_to_timeframe(df_raw, "5T")
        logger.info(f"Resampled to 5-min: {len(df_raw):,} candles")

    # ==================================================
    # STEP 2: Feature engineering
    # ==================================================
    logger.info("\n>> STEP 2: Engineering features...")

    from features.engineer import engineer_features, get_feature_columns

    df = engineer_features(df_raw, add_target_col=True, drop_na=True)
    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    logger.info(f"Engineered data: {len(df):,} valid rows, {len(feature_cols)} features")

    # ==================================================
    # STEP 3: Walk-forward training
    # ==================================================
    logger.info("\n>> STEP 3: Walk-forward model training...")

    from models.trainer import run_walk_forward_training

    results = run_walk_forward_training(df, feature_cols)

    if not results.get("fold_results"):
        logger.error(f"Training failed for {pair}/{timeframe} -- no fold results.")
        return {"pair": pair, "timeframe": timeframe, "status": "FAILED", "reason": "No folds"}

    # ==================================================
    # STEP 4: Run backtest
    # ==================================================
    logger.info("\n>> STEP 4: Running backtest...")

    from backtest.engine import BacktestEngine

    engine = BacktestEngine(min_confidence=min_conf, pair=pair)
    bt_metrics = engine.run_from_fold_results(results["fold_results"])

    # ==================================================
    # STEP 5: Generate HTML report
    # ==================================================
    logger.info("\n>> STEP 5: Generating backtest report...")

    from backtest.report import generate_html_report

    report_dir = BASE_DIR / "backtest"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = generate_html_report(
        metrics=bt_metrics,
        fold_results=results["fold_results"],
        feature_importance=results.get("feature_importance"),
        pair=pair,
        timeframe=timeframe,
        output_path=report_dir / f"report_{pair}_{timeframe}.html",
    )

    # ==================================================
    # STEP 6: Save trained models (to TF-specific dir)
    # ==================================================
    logger.info(f"\n>> STEP 6: Saving models to {save_dir}...")

    best_ensemble = results.get("best_ensemble")
    if best_ensemble:
        best_ensemble.save(save_dir)

    # Also save to default MODEL_DIR for backward compat (last trained wins)
    if best_ensemble:
        best_ensemble.save(MODEL_DIR)

    # Save scaler
    best_scaler = results.get("best_scaler")
    if best_scaler:
        import joblib
        joblib.dump(best_scaler, save_dir / "scaler.pkl")
        joblib.dump(best_scaler, MODEL_DIR / "scaler.pkl")

    # Save feature importance
    feat_imp = results.get("feature_importance")
    if feat_imp is not None and not feat_imp.empty:
        feat_imp.to_csv(save_dir / "feature_importance.csv", index=False)
        feat_imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    elapsed = time.time() - start_time
    agg = results.get("aggregate", {})

    summary = {
        "pair": pair,
        "timeframe": timeframe,
        "status": "OK",
        "raw_accuracy": agg.get("raw_accuracy", 0),
        "filtered_accuracy": agg.get("filtered_accuracy", 0),
        "trade_rate": agg.get("trade_rate", 0),
        "n_folds": agg.get("n_folds", 0),
        "win_rate": bt_metrics.get("win_rate", 0),
        "total_trades": bt_metrics.get("total_trades", 0),
        "net_pnl": bt_metrics.get("net_pnl", 0),
        "profit_factor": bt_metrics.get("profit_factor", 0),
        "max_drawdown": bt_metrics.get("max_drawdown", 0),
        "final_balance": bt_metrics.get("final_balance", 0),
        "elapsed": elapsed,
        "report": str(report_path),
        "model_dir": str(save_dir),
    }

    logger.info(f"\n  {pair}/{timeframe} done in {elapsed:.0f}s -- "
                f"WR={summary['win_rate']:.2%}, Trades={summary['total_trades']}, "
                f"PF={summary['profit_factor']:.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Binary Options Candle Direction Predictor -- Training Pipeline"
    )
    parser.add_argument("--asset", type=str, nargs="+", default=["EURUSD"],
                        choices=ALL_PAIRS,
                        help="Currency pair(s) to train on (space-separated)")
    parser.add_argument("--timeframe", type=str, nargs="+", default=[TIMEFRAME],
                        choices=["1M", "5M"],
                        help="Candle timeframe(s) (space-separated)")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help="Minimum confidence threshold")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use cached CSV)")
    parser.add_argument("--years", type=int, default=2,
                        help="Years of historical data to fetch")
    args = parser.parse_args()

    pairs = [a.upper() for a in args.asset]
    timeframes = [t.upper() for t in args.timeframe]
    min_conf = args.min_confidence

    total_combos = len(pairs) * len(timeframes)

    logger.info("=" * 70)
    logger.info("  BINARY OPTIONS CANDLE DIRECTION PREDICTOR -- TRAINING")
    logger.info(f"  Pairs:      {pairs}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Total jobs: {total_combos} ({len(pairs)} pairs x {len(timeframes)} TFs)")
    logger.info(f"  Confidence: {min_conf}")
    logger.info("=" * 70)

    global_start = time.time()
    all_summaries = []

    for pair in pairs:
        for timeframe in timeframes:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  JOB: {pair} / {timeframe}  "
                        f"({len(all_summaries) + 1} of {total_combos})")
            logger.info(f"{'#' * 70}")

            summary = train_single(
                pair=pair,
                timeframe=timeframe,
                min_conf=min_conf,
                skip_download=args.skip_download,
                years=args.years,
            )
            all_summaries.append(summary)

    # ==================================================
    # Final summary table
    # ==================================================
    total_elapsed = time.time() - global_start

    logger.info("\n" + "=" * 70)
    logger.info("  ALL TRAINING COMPLETE")
    logger.info("=" * 70)

    # Build table
    header = f"{'Pair':<10} {'TF':<5} {'Status':<8} {'Win Rate':>9} {'Trades':>7} {'PF':>6} {'Balance':>10} {'Time':>6}"
    logger.info(header)
    logger.info("-" * 70)

    for s in all_summaries:
        if s["status"] == "OK":
            row = (
                f"{s['pair']:<10} {s['timeframe']:<5} {s['status']:<8} "
                f"{s['win_rate']:>8.1%} {s['total_trades']:>7} "
                f"{s['profit_factor']:>6.2f} {s['final_balance']:>9.0f}$ "
                f"{s['elapsed']:>5.0f}s"
            )
        else:
            row = (
                f"{s['pair']:<10} {s['timeframe']:<5} {s['status']:<8} "
                f"{'---':>9} {'---':>7} {'---':>6} {'---':>10} {'---':>6}"
            )
        logger.info(row)

    logger.info("-" * 70)

    ok_results = [s for s in all_summaries if s["status"] == "OK"]
    if ok_results:
        avg_wr = np.mean([s["win_rate"] for s in ok_results])
        total_trades = sum(s["total_trades"] for s in ok_results)
        logger.info(f"  Avg Win Rate: {avg_wr:.1%} | Total Trades: {total_trades} | "
                    f"Jobs: {len(ok_results)}/{total_combos} OK")

    logger.info(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
