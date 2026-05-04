"""
Live signal generator — async loop that produces trading signals.

Runs on each candle close, applies feature engineering and ensemble
prediction, then emits signals with confidence filtering.
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
    TIMEFRAME,
    TIMEFRAME_SECONDS,
    MIN_CONFIDENCE,
    SIGNAL_LOG,
    MODEL_DIR,
    SKIP_ASIAN,
    MAX_DAILY_TRADES,
    MAX_LOSS_STREAK,
)
from data.fetcher_live import LiveFetcher
from features.engineer import engineer_features, get_feature_columns, scale_features, load_scaler
from models.ensemble import EnsembleModel
from utils.helpers import append_signal_log, now_utc
from utils.logger import get_logger

logger = get_logger("binary_predictor")


class SignalGenerator:
    """
    Produces UP/DOWN trading signals from live candle data.
    
    Runs asynchronously, emitting signals at each candle close.
    """

    def __init__(
        self,
        pairs: list[str] = None,
        timeframe: str = TIMEFRAME,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.pairs = pairs or ASSETS
        self.timeframe = timeframe
        self.interval = 60 if timeframe == "1M" else 300
        self.min_confidence = min_confidence

        self.fetcher = LiveFetcher()
        self.ensemble = EnsembleModel(min_confidence=min_confidence)
        self.scaler = None

        self._running = False
        self._daily_trades: dict[str, int] = {}
        self._loss_streak = 0
        self._callbacks: list[Callable] = []
        self._latest_signals: dict[str, dict] = {}

    async def initialize(self) -> bool:
        """
        Load models and connect to Quotex.
        """
        # Load trained models
        try:
            self.ensemble.load(MODEL_DIR)
            logger.info("✓ Ensemble model loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

        # Load scaler
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = load_scaler(scaler_path)
            logger.info("✓ Scaler loaded")

        # Connect to Quotex
        connected = await self.fetcher.connect()
        if not connected:
            logger.error("Failed to connect to Quotex")
            return False

        return True

    def add_callback(self, callback: Callable):
        """Add a callback that fires on each signal."""
        self._callbacks.append(callback)

    async def run(self):
        """
        Main signal generation loop.
        Runs until cancelled.
        """
        self._running = True
        logger.info(
            f"Signal generator started: pairs={self.pairs}, "
            f"tf={self.timeframe}, conf≥{self.min_confidence}"
        )

        try:
            while self._running:
                for pair in self.pairs:
                    try:
                        signal = await self._generate_signal(pair)
                        if signal:
                            self._latest_signals[pair] = signal
                            await self._emit_signal(signal)
                    except Exception as e:
                        logger.error(f"Signal generation error for {pair}: {e}")

                # Sleep until a few seconds before next candle close
                await self._sleep_until_next_candle()

        except asyncio.CancelledError:
            logger.info("Signal generator cancelled")
        finally:
            self._running = False
            await self.fetcher.disconnect()

    async def _generate_signal(self, pair: str) -> Optional[dict]:
        """
        Generate a single signal for one pair.
        """
        # Fetch latest candles
        df = await self.fetcher.fetch_candles(pair, count=100, timeframe=self.interval)
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient candle data for {pair}: {len(df)} candles")
            return None

        # Engineer features (no target needed for live)
        featured = engineer_features(df, add_target_col=False, drop_na=True)
        if featured.empty or len(featured) < 5:
            logger.warning(f"Feature engineering produced too few rows for {pair}")
            return None

        # Scale features
        feature_cols = [c for c in get_feature_columns() if c in featured.columns]
        if self.scaler:
            featured_scaled, _ = scale_features(featured, scaler=self.scaler, fit=False)
        else:
            featured_scaled = featured

        X = featured_scaled[feature_cols]

        # Predict
        signal_info = self.ensemble.predict_single(X, featured_scaled)
        now = now_utc()

        # Apply additional filters
        hour = now.hour
        date_str = now.strftime("%Y-%m-%d")

        # Asian session filter
        if SKIP_ASIAN and 0 <= hour < 8:
            signal_info["action"] = "SKIP"
            signal_info["reason"] = "Asian session"

        # Daily trade limit
        daily = self._daily_trades.get(date_str, 0)
        if daily >= MAX_DAILY_TRADES:
            signal_info["action"] = "SKIP"
            signal_info["reason"] = "Daily limit reached"

        # Loss streak
        if self._loss_streak >= MAX_LOSS_STREAK:
            signal_info["action"] = "SKIP"
            signal_info["reason"] = "Loss streak circuit breaker"

        # Build signal dict
        reason = self._build_reason(featured.iloc[-1], signal_info)

        signal = {
            "time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "asset": pair,
            "direction": signal_info["direction"],
            "confidence": signal_info["confidence"],
            "timeframe": self.timeframe,
            "expiry": self.interval,
            "action": signal_info["action"],
            "reason": reason,
            "probability": signal_info.get("probability", 0.5),
        }

        # Update daily count if trading
        if signal["action"] == "TRADE":
            self._daily_trades[date_str] = daily + 1

        return signal

    def _build_reason(self, row: pd.Series, signal_info: dict) -> str:
        """Build a human-readable reason string."""
        reasons = []

        if signal_info["action"] == "SKIP":
            return signal_info.get("reason", "Low confidence")

        # Check patterns
        if row.get("bullish_engulfing", 0) == 1:
            reasons.append("Bullish engulfing")
        elif row.get("bearish_engulfing", 0) == 1:
            reasons.append("Bearish engulfing")
        elif row.get("hammer", 0) == 1:
            reasons.append("Hammer")
        elif row.get("shooting_star", 0) == 1:
            reasons.append("Shooting star")

        # RSI
        rsi = row.get("rsi_7", 50)
        if rsi < 30:
            reasons.append("RSI oversold")
        elif rsi > 70:
            reasons.append("RSI overbought")
        elif row.get("rsi_slope", 0) > 0:
            reasons.append("RSI rising")
        elif row.get("rsi_slope", 0) < 0:
            reasons.append("RSI falling")

        # Session
        if row.get("is_overlap", 0) == 1:
            reasons.append("London/NY overlap")
        elif row.get("is_ny_open", 0) == 1:
            reasons.append("NY session")
        elif row.get("is_london_open", 0) == 1:
            reasons.append("London session")

        # EMA cross
        if row.get("ema_cross_signal", 0) == 1:
            reasons.append("EMA5>EMA13")
        else:
            reasons.append("EMA5<EMA13")

        return " + ".join(reasons[:3]) if reasons else "Model consensus"

    async def _emit_signal(self, signal: dict):
        """Log and broadcast signal."""
        action = signal["action"]
        direction = signal["direction"]
        conf = signal["confidence"]
        pair = signal["asset"]

        if action == "TRADE":
            logger.info(
                f"🎯 SIGNAL: {pair} → {direction} "
                f"(conf={conf:.2%}) | {signal['reason']}"
            )
        else:
            logger.debug(
                f"⏭ SKIP: {pair} → {direction} "
                f"(conf={conf:.2%}) | {signal.get('reason', '')}"
            )

        # Log to CSV
        append_signal_log(SIGNAL_LOG, signal)

        # Write latest signal to shared file (for dashboard)
        signal_file = Path(MODEL_DIR).parent / "latest_signal.json"
        try:
            signal_file.write_text(json.dumps(signal, indent=2))
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
        seconds_in_candle = now.second + now.microsecond / 1e6
        if self.interval == 60:
            remaining = 60 - seconds_in_candle
        else:
            elapsed = (now.minute * 60 + now.second) % self.interval
            remaining = self.interval - elapsed

        # Wake up 5-10 seconds before candle close
        sleep_secs = max(remaining - 8, 1)
        logger.debug(f"Sleeping {sleep_secs:.0f}s until next candle close")
        await asyncio.sleep(sleep_secs)

    def record_outcome(self, won: bool):
        """Record trade outcome for adaptive threshold."""
        self.ensemble.record_outcome(won)
        if won:
            self._loss_streak = 0
        else:
            self._loss_streak += 1

    def get_latest_signal(self, pair: str) -> Optional[dict]:
        """Get the most recent signal for a pair."""
        return self._latest_signals.get(pair)

    def stop(self):
        """Stop the signal generator."""
        self._running = False


async def run_signal_generator(
    pairs: list[str] = None,
    timeframe: str = TIMEFRAME,
):
    """Convenience function to start the signal generator."""
    gen = SignalGenerator(pairs=pairs, timeframe=timeframe)
    ok = await gen.initialize()
    if ok:
        await gen.run()
    else:
        logger.error("Signal generator initialization failed")


if __name__ == "__main__":
    asyncio.run(run_signal_generator())
