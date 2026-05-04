"""
Live signal generator -- multi-pair, multi-timeframe async loop.

Runs all pairs x timeframes simultaneously using asyncio.gather().
Implements per-pair session filtering, loss-streak tracking, and
multi-timeframe agreement logic.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ASSETS,
    ASIAN_PAIRS,
    LONDON_NY_PAIRS,
    TIMEFRAME,
    TIMEFRAMES,
    TIMEFRAME_SECONDS,
    MIN_CONFIDENCE,
    MTF_AGREEMENT_BOOST,
    SIGNAL_LOG,
    MODEL_DIR,
    SKIP_ASIAN,
    MAX_DAILY_TRADES,
    MAX_DAILY_TRADES_PER_PAIR,
    MAX_LOSS_STREAK,
    MAX_GLOBAL_LOSS_STREAK,
    LONDON_OPEN,
    LONDON_CLOSE,
    NY_OPEN,
    NY_CLOSE,
    OVERLAP_START,
    OVERLAP_END,
    ASIAN_START,
    ASIAN_END,
)
from data.fetcher_live import LiveFetcher
from features.engineer import engineer_features, get_feature_columns, scale_features, load_scaler
from models.ensemble import EnsembleModel
from utils.helpers import append_signal_log, now_utc
from utils.logger import get_logger

logger = get_logger("binary_predictor")


class SignalGenerator:
    """
    Multi-pair, multi-timeframe signal generator.

    Produces UP/DOWN trading signals from live candle data with:
    - Per-pair session filtering (Asian pairs trade Asian, etc.)
    - Per-pair loss streak tracking
    - Global loss streak circuit breaker
    - Multi-timeframe agreement boosting
    """

    def __init__(
        self,
        pairs: list[str] = None,
        timeframes: list[str] = None,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.pairs = pairs or ASSETS
        self.timeframes = timeframes or [TIMEFRAME]
        self.min_confidence = min_confidence

        self.fetcher = LiveFetcher()
        # One ensemble per timeframe
        self._ensembles: dict[str, EnsembleModel] = {}
        self._scalers: dict[str, object] = {}

        self._running = False
        self._callbacks: list[Callable] = []
        self._latest_signals: dict[str, dict] = {}

        # --- Per-pair tracking ---
        self._daily_trades_global: int = 0
        self._daily_trades_per_pair: dict[str, int] = {p: 0 for p in self.pairs}
        self._loss_streak_per_pair: dict[str, int] = {p: 0 for p in self.pairs}
        self._global_loss_streak: int = 0
        self._today_str: str = ""
        self._pair_blocked: dict[str, bool] = {p: False for p in self.pairs}
        self._global_blocked: bool = False

    async def initialize(self) -> bool:
        """
        Load models for each timeframe and connect to Quotex.
        """
        for tf in self.timeframes:
            ensemble = EnsembleModel(min_confidence=self.min_confidence)
            tf_model_dir = MODEL_DIR
            # Check for timeframe-specific model dirs
            tf_specific = MODEL_DIR.parent / f"saved_{tf}"
            if tf_specific.exists():
                tf_model_dir = tf_specific

            try:
                ensemble.load(tf_model_dir)
                self._ensembles[tf] = ensemble
                logger.info(f"[OK] Ensemble loaded for {tf}")
            except Exception as e:
                logger.warning(f"Could not load model for {tf}: {e}")
                # Fall back to default models
                if tf_model_dir != MODEL_DIR:
                    try:
                        ensemble.load(MODEL_DIR)
                        self._ensembles[tf] = ensemble
                        logger.info(f"[OK] Fallback ensemble loaded for {tf}")
                    except Exception as e2:
                        logger.error(f"Failed to load any model for {tf}: {e2}")

            # Load scaler
            scaler_path = tf_model_dir / "scaler.pkl"
            if not scaler_path.exists():
                scaler_path = MODEL_DIR / "scaler.pkl"
            if scaler_path.exists():
                self._scalers[tf] = load_scaler(scaler_path)
                logger.info(f"[OK] Scaler loaded for {tf}")

        if not self._ensembles:
            logger.error("No models loaded for any timeframe")
            return False

        # Connect to Quotex
        connected = await self.fetcher.connect()
        if not connected:
            logger.error("Failed to connect to Quotex")
            return False

        n_channels = len(self.pairs) * len(self.timeframes)
        logger.info(
            f"[OK] Signal generator ready: "
            f"{len(self.pairs)} pairs x {len(self.timeframes)} timeframes = "
            f"{n_channels} signal channels active"
        )
        return True

    def add_callback(self, callback: Callable):
        """Add a callback that fires on each signal."""
        self._callbacks.append(callback)

    async def run(self):
        """
        Main signal generation loop.
        All pairs processed concurrently via asyncio.gather().
        """
        self._running = True
        logger.info(
            f"Signal loop started: pairs={self.pairs}, "
            f"timeframes={self.timeframes}, conf>={self.min_confidence}"
        )

        try:
            while self._running:
                self._check_day_reset()

                if self._global_blocked:
                    logger.info("Global loss streak limit hit. Sleeping until next day.")
                    await asyncio.sleep(60)
                    continue

                # Process all pairs concurrently
                tasks = [self._process_pair(pair) for pair in self.pairs]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep until next candle close
                await self._sleep_until_next_candle()

        except asyncio.CancelledError:
            logger.info("Signal generator cancelled")
        finally:
            self._running = False
            await self.fetcher.disconnect()

    async def _process_pair(self, pair: str):
        """
        Process a single pair across all timeframes.
        Generates signals per-TF, merges via MTF agreement, and emits.
        """
        # Skip if pair is blocked
        if self._pair_blocked.get(pair, False):
            return

        # Per-pair daily limit
        if self._daily_trades_per_pair.get(pair, 0) >= MAX_DAILY_TRADES_PER_PAIR:
            return

        # Global daily limit
        if self._daily_trades_global >= MAX_DAILY_TRADES:
            return

        # Session filter
        if not self._is_pair_in_session(pair):
            return

        try:
            # Generate signal on each timeframe concurrently
            tf_signals: dict[str, Optional[dict]] = {}
            tasks = {}
            for tf in self.timeframes:
                if tf in self._ensembles:
                    tasks[tf] = self._generate_signal(pair, tf)

            results = await asyncio.gather(
                *[tasks[tf] for tf in tasks],
                return_exceptions=True,
            )

            for tf, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Signal error {pair}/{tf}: {result}")
                    tf_signals[tf] = None
                else:
                    tf_signals[tf] = result

            # --- Multi-timeframe merging ---
            if len(self.timeframes) >= 2 and "1M" in tf_signals and "5M" in tf_signals:
                ensemble = self._ensembles.get("1M") or list(self._ensembles.values())[0]
                merged = ensemble.merge_mtf_signals(
                    tf_signals.get("1M"),
                    tf_signals.get("5M"),
                )
                merged["asset"] = pair
                merged["time"] = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
                merged["timeframe"] = "MTF"
                merged["expiry"] = 60

                self._latest_signals[pair] = merged
                await self._emit_signal(merged)
            else:
                # Single timeframe mode
                for tf, sig in tf_signals.items():
                    if sig:
                        sig["mtf_agreement"] = False
                        self._latest_signals[pair] = sig
                        await self._emit_signal(sig)

        except Exception as e:
            logger.error(f"Signal generation error for {pair}: {e}")

    async def _generate_signal(self, pair: str, timeframe: str) -> Optional[dict]:
        """
        Generate a single signal for one pair on one timeframe.
        """
        interval = 60 if timeframe == "1M" else 300

        # Fetch latest candles
        df = await self.fetcher.fetch_candles(pair, count=100, timeframe=interval)
        if df.empty or len(df) < 50:
            return None

        # Engineer features
        featured = engineer_features(df, add_target_col=False, drop_na=True)
        if featured.empty or len(featured) < 5:
            return None

        # Scale features
        feature_cols = [c for c in get_feature_columns() if c in featured.columns]
        scaler = self._scalers.get(timeframe)
        if scaler:
            featured_scaled, _ = scale_features(featured, scaler=scaler, fit=False)
        else:
            featured_scaled = featured

        X = featured_scaled[feature_cols]

        # Predict
        ensemble = self._ensembles[timeframe]
        signal_info = ensemble.predict_single(X, featured_scaled)
        now = now_utc()

        signal = {
            "time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "asset": pair,
            "direction": signal_info["direction"],
            "confidence": signal_info["confidence"],
            "timeframe": timeframe,
            "expiry": interval,
            "action": signal_info["action"],
            "reason": signal_info.get("reason", ""),
            "probability": signal_info.get("probability", 0.5),
        }

        return signal

    def _is_pair_in_session(self, pair: str) -> bool:
        """
        Check if a pair should be traded in the current UTC hour.

        - ASIAN_PAIRS trade UTC 0-8 (Asian) + overlap (13-17)
        - LONDON_NY_PAIRS trade UTC 8-22 (London + NY)
        - All pairs always trade during overlap (13-17)
        """
        hour = now_utc().hour

        # Overlap is always active for all pairs
        if OVERLAP_START <= hour < OVERLAP_END:
            return True

        if pair in ASIAN_PAIRS:
            # Asian session (0-8) + London/NY (8-22)
            return True  # Asian pairs trade all sessions

        if pair in LONDON_NY_PAIRS:
            # Only London + NY hours
            return LONDON_OPEN <= hour < NY_CLOSE

        # Default: trade London + NY
        return LONDON_OPEN <= hour < NY_CLOSE

    def _check_day_reset(self):
        """Reset daily counters at midnight UTC."""
        today = now_utc().strftime("%Y-%m-%d")
        if today != self._today_str:
            self._today_str = today
            self._daily_trades_global = 0
            self._daily_trades_per_pair = {p: 0 for p in self.pairs}
            self._loss_streak_per_pair = {p: 0 for p in self.pairs}
            self._global_loss_streak = 0
            self._pair_blocked = {p: False for p in self.pairs}
            self._global_blocked = False
            logger.info(f"Day reset: {today} -- all counters cleared")

    async def _emit_signal(self, signal: dict):
        """Log and broadcast signal."""
        action = signal.get("action", "SKIP")
        direction = signal.get("direction", "?")
        conf = signal.get("confidence", 0)
        pair = signal.get("asset", "?")
        mtf = signal.get("mtf_agreement", False)
        mtf_tag = " [MTF]" if mtf else ""

        if action == "TRADE":
            logger.info(
                f"SIGNAL: {pair} -> {direction} "
                f"(conf={conf:.2%}){mtf_tag} | {signal.get('reason', '')}"
            )

            # Update counters
            self._daily_trades_global += 1
            self._daily_trades_per_pair[pair] = self._daily_trades_per_pair.get(pair, 0) + 1
        else:
            logger.debug(
                f"SKIP: {pair} -> {direction} "
                f"(conf={conf:.2%}){mtf_tag} | {signal.get('reason', '')}"
            )

        # Log to CSV
        append_signal_log(SIGNAL_LOG, signal)

        # Write latest signals to shared file (for dashboard)
        signal_file = Path(MODEL_DIR).parent / "latest_signals.json"
        try:
            # Read existing, update, write
            existing = {}
            if signal_file.exists():
                try:
                    existing = json.loads(signal_file.read_text())
                except Exception:
                    existing = {}
            existing[pair] = signal
            signal_file.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

        # Also write single latest signal for backward compat
        single_file = Path(MODEL_DIR).parent / "latest_signal.json"
        try:
            single_file.write_text(json.dumps(signal, indent=2))
        except Exception:
            pass

        # Fire callbacks
        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(signal)
                else:
                    cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    async def _sleep_until_next_candle(self):
        """Sleep until a few seconds before the next candle close."""
        now = now_utc()
        # Use the smallest timeframe interval
        interval = 60  # 1M is the fastest
        seconds_in_candle = now.second + now.microsecond / 1e6
        remaining = interval - seconds_in_candle

        sleep_secs = max(remaining - 8, 1)
        logger.debug(f"Sleeping {sleep_secs:.0f}s until next candle close")
        await asyncio.sleep(sleep_secs)

    def record_outcome(self, pair: str, won: bool):
        """
        Record trade outcome for adaptive threshold and streak tracking.
        """
        # Update ensemble adaptive threshold
        for ensemble in self._ensembles.values():
            ensemble.record_outcome(won)

        if won:
            self._loss_streak_per_pair[pair] = 0
            self._global_loss_streak = 0
        else:
            self._loss_streak_per_pair[pair] = self._loss_streak_per_pair.get(pair, 0) + 1
            self._global_loss_streak += 1

            # Per-pair circuit breaker
            if self._loss_streak_per_pair[pair] >= MAX_LOSS_STREAK:
                self._pair_blocked[pair] = True
                logger.warning(
                    f"CIRCUIT BREAKER: {pair} blocked for today "
                    f"({self._loss_streak_per_pair[pair]} consecutive losses)"
                )

            # Global circuit breaker
            if self._global_loss_streak >= MAX_GLOBAL_LOSS_STREAK:
                self._global_blocked = True
                logger.warning(
                    f"GLOBAL CIRCUIT BREAKER: ALL pairs blocked "
                    f"({self._global_loss_streak} consecutive losses across all pairs)"
                )

    def get_latest_signal(self, pair: str) -> Optional[dict]:
        """Get the most recent signal for a pair."""
        return self._latest_signals.get(pair)

    def get_all_latest_signals(self) -> dict[str, dict]:
        """Get all latest signals keyed by pair."""
        return self._latest_signals.copy()

    def get_active_pairs_count(self) -> int:
        """Return how many pairs are still active (not blocked)."""
        return sum(1 for p in self.pairs if not self._pair_blocked.get(p, False))

    def stop(self):
        """Stop the signal generator."""
        self._running = False


async def run_signal_generator(
    pairs: list[str] = None,
    timeframes: list[str] = None,
):
    """Convenience function to start the signal generator."""
    gen = SignalGenerator(pairs=pairs, timeframes=timeframes)
    ok = await gen.initialize()
    if ok:
        await gen.run()
    else:
        logger.error("Signal generator initialization failed")


if __name__ == "__main__":
    asyncio.run(run_signal_generator())
