"""
Walk-forward backtesting engine.

Simulates binary options trading with realistic conditions:
  - Fixed payout/loss, spread costs, confidence filtering
  - Session and news filters, loss streak circuit breaker
  - Tracks equity curve, drawdown, and per-trade logs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    INITIAL_BALANCE,
    TRADE_SIZE_PCT,
    PAYOUT_RATE,
    LOSS_RATE,
    SPREAD_COST_PIPS,
    MIN_CONFIDENCE,
    MAX_DAILY_TRADES,
    MAX_LOSS_STREAK,
    SKIP_ASIAN,
    SKIP_NEWS_MINUTES,
    HIGH_VOLATILITY_HOURS,
)
from utils.logger import get_logger
from utils.helpers import pip_value

logger = get_logger("binary_predictor")


class BacktestEngine:
    """
    Walk-forward backtesting engine for binary options.
    """

    def __init__(
        self,
        initial_balance: float = INITIAL_BALANCE,
        trade_size_pct: float = TRADE_SIZE_PCT,
        payout_rate: float = PAYOUT_RATE,
        loss_rate: float = LOSS_RATE,
        spread_pips: float = SPREAD_COST_PIPS,
        min_confidence: float = MIN_CONFIDENCE,
        max_daily_trades: int = MAX_DAILY_TRADES,
        max_loss_streak: int = MAX_LOSS_STREAK,
        skip_asian: bool = SKIP_ASIAN,
        skip_news_minutes: int = SKIP_NEWS_MINUTES,
        pair: str = "EURUSD",
    ):
        self.initial_balance = initial_balance
        self.trade_size_pct = trade_size_pct
        self.payout_rate = payout_rate
        self.loss_rate = loss_rate
        self.spread_pips = spread_pips
        self.min_confidence = min_confidence
        self.max_daily_trades = max_daily_trades
        self.max_loss_streak = max_loss_streak
        self.skip_asian = skip_asian
        self.skip_news_minutes = skip_news_minutes
        self.pair = pair
        self._pip = pip_value(pair)

        # State
        self.balance = initial_balance
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self._daily_count: dict[str, int] = {}
        self._loss_streak = 0
        self._max_drawdown = 0
        self._peak_balance = initial_balance

        # News events (loaded externally)
        self.news_events: list[datetime] = []

    def set_news_events(self, events: list[datetime]):
        """Set list of high-impact news event times."""
        self.news_events = sorted(events)

    def _is_near_news(self, candle_time: datetime) -> bool:
        """Check if candle_time is within ±skip_news_minutes of any news event."""
        if not self.news_events:
            return False
        window = timedelta(minutes=self.skip_news_minutes)
        for event in self.news_events:
            if abs((candle_time - event).total_seconds()) < window.total_seconds():
                return True
        return False

    def _is_asian_session(self, hour: int) -> bool:
        """Check if hour falls in Asian session (0-8 UTC)."""
        return 0 <= hour < 8

    def _check_filters(
        self,
        candle_time: datetime,
        confidence: float,
        date_str: str,
    ) -> tuple[bool, str]:
        """
        Apply all trading filters.
        Returns (should_trade, skip_reason).
        """
        # Confidence filter
        if confidence < self.min_confidence:
            return False, "low_confidence"

        # Asian session filter
        hour = candle_time.hour if hasattr(candle_time, 'hour') else pd.Timestamp(candle_time).hour
        if self.skip_asian and self._is_asian_session(hour):
            return False, "asian_session"

        # Daily trade limit
        daily_count = self._daily_count.get(date_str, 0)
        if daily_count >= self.max_daily_trades:
            return False, "daily_limit"

        # Loss streak circuit breaker
        if self._loss_streak >= self.max_loss_streak:
            return False, "loss_streak"

        # News filter
        if self._is_near_news(candle_time):
            return False, "near_news"

        return True, ""

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        confidences: np.ndarray,
        actuals: np.ndarray,
        times: pd.Series = None,
    ) -> dict:
        """
        Run backtest on a dataset.

        Parameters
        ----------
        df : DataFrame with candle data
        predictions : array of predicted directions (0=DOWN, 1=UP)
        confidences : array of confidence scores [0, 1]
        actuals : array of actual next candle directions
        times : series of candle timestamps

        Returns
        -------
        Dict with all backtest results and metrics.
        """
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = []
        self._daily_count = {}
        self._loss_streak = 0
        self._peak_balance = self.initial_balance
        self._max_drawdown = 0

        n = len(predictions)
        if times is None:
            times = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")

        wins = 0
        losses = 0

        for i in range(n):
            candle_time = pd.Timestamp(times.iloc[i] if hasattr(times, 'iloc') else times[i])
            if candle_time.tzinfo is None:
                candle_time = candle_time.tz_localize("UTC")
            date_str = candle_time.strftime("%Y-%m-%d")

            # Apply filters
            should_trade, skip_reason = self._check_filters(
                candle_time, confidences[i], date_str
            )

            if not should_trade:
                self.equity_curve.append({
                    "time": candle_time,
                    "balance": self.balance,
                    "action": "skip",
                    "reason": skip_reason,
                })
                # Reset loss streak on new day
                if skip_reason == "loss_streak" and date_str not in self._daily_count:
                    self._loss_streak = 0
                continue

            # Place trade
            stake = self.balance * self.trade_size_pct
            predicted_dir = int(predictions[i])
            actual_dir = int(actuals[i])

            # Apply spread cost (reduces effective payout)
            spread_cost = self.spread_pips * self._pip * stake * 100

            won = predicted_dir == actual_dir

            if won:
                pnl = stake * self.payout_rate - spread_cost
                wins += 1
                self._loss_streak = 0
            else:
                pnl = -(stake * self.loss_rate) - spread_cost
                losses += 1
                self._loss_streak += 1

            self.balance += pnl

            # Update daily count
            self._daily_count[date_str] = self._daily_count.get(date_str, 0) + 1

            # Track drawdown
            if self.balance > self._peak_balance:
                self._peak_balance = self.balance
            dd = (self._peak_balance - self.balance) / self._peak_balance
            if dd > self._max_drawdown:
                self._max_drawdown = dd

            trade = {
                "time": candle_time,
                "prediction": "UP" if predicted_dir else "DOWN",
                "actual": "UP" if actual_dir else "DOWN",
                "confidence": round(confidences[i], 4),
                "won": won,
                "stake": round(stake, 2),
                "pnl": round(pnl, 2),
                "balance": round(self.balance, 2),
                "drawdown": round(dd, 4),
            }
            self.trades.append(trade)

            self.equity_curve.append({
                "time": candle_time,
                "balance": self.balance,
                "action": "trade",
                "won": won,
            })

        # ── Compute Metrics ──────────────────────────
        return self._compute_metrics(wins, losses)

    def _compute_metrics(self, wins: int, losses: int) -> dict:
        """Compute all backtest performance metrics."""
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0

        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "net_pnl": 0,
                "final_balance": self.balance,
            }

        gross_profit = trades_df.loc[trades_df["won"], "pnl"].sum()
        gross_loss = abs(trades_df.loc[~trades_df["won"], "pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        net_pnl = self.balance - self.initial_balance

        # Sharpe ratio (daily returns)
        trades_df["date"] = pd.to_datetime(trades_df["time"]).dt.date
        daily_pnl = trades_df.groupby("date")["pnl"].sum()
        sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

        # Monthly win rates
        trades_df["month"] = pd.to_datetime(trades_df["time"]).dt.to_period("M")
        monthly_wr = trades_df.groupby("month")["won"].mean()
        monthly_consistency = monthly_wr.std()

        # Calmar ratio
        annual_return = net_pnl / self.initial_balance
        calmar = annual_return / self._max_drawdown if self._max_drawdown > 0 else 0

        # Max consecutive wins/losses
        streaks = trades_df["won"].astype(int).values
        max_win_streak = max_loss_streak = current = 0
        for s in streaks:
            if s == 1:
                current += 1
                max_win_streak = max(max_win_streak, current)
            else:
                current = 0
        current = 0
        for s in streaks:
            if s == 0:
                current += 1
                max_loss_streak = max(max_loss_streak, current)
            else:
                current = 0

        metrics = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "net_pnl": round(net_pnl, 2),
            "final_balance": round(self.balance, 2),
            "profit_factor": round(profit_factor, 3),
            "max_drawdown": round(self._max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 3),
            "calmar_ratio": round(calmar, 3),
            "monthly_consistency_std": round(monthly_consistency, 4) if not pd.isna(monthly_consistency) else 0,
            "monthly_win_rates": monthly_wr.to_dict(),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "initial_balance": self.initial_balance,
            "trades_df": trades_df,
            "equity_curve_df": pd.DataFrame(self.equity_curve),
        }

        logger.info("─" * 50)
        logger.info("BACKTEST RESULTS")
        logger.info(f"  Trades:       {total_trades}")
        logger.info(f"  Win Rate:     {win_rate:.2%}")
        logger.info(f"  Net P&L:      ${net_pnl:,.2f}")
        logger.info(f"  Final Bal:    ${self.balance:,.2f}")
        logger.info(f"  Profit Factor:{profit_factor:.3f}")
        logger.info(f"  Max Drawdown: {self._max_drawdown:.2%}")
        logger.info(f"  Sharpe:       {sharpe:.3f}")
        logger.info("─" * 50)

        return metrics

    def run_from_fold_results(self, fold_results: list[dict]) -> dict:
        """
        Run backtest across all walk-forward folds.
        Concatenates fold predictions and runs the backtest.
        """
        all_preds = []
        all_conf = []
        all_actual = []
        all_times = []

        for result in fold_results:
            n = len(result["y_test"])
            all_preds.append(result["ensemble_preds"])
            conf = result.get("confidence", np.maximum(
                result["ensemble_proba"], 1 - result["ensemble_proba"]
            ))
            all_conf.append(conf)
            all_actual.append(result["y_test"].values)

            test_df = result["test_df"]
            if "time" in test_df.columns:
                all_times.append(test_df["time"].values[:n])
            else:
                all_times.append(pd.date_range(
                    "2023-01-01", periods=n, freq="1min", tz="UTC"
                ).values)

        predictions = np.concatenate(all_preds)
        confidences = np.concatenate(all_conf)
        actuals = np.concatenate(all_actual)
        times = pd.Series(np.concatenate(all_times))

        return self.run(
            pd.DataFrame(),
            predictions,
            confidences,
            actuals,
            times,
        )
