"""
Live data fetcher — connects to Quotex via pyquotex.

Streams real-time candle data and maintains a rolling buffer.
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    QUOTEX_EMAIL,
    QUOTEX_PASSWORD,
    TIMEFRAME_SECONDS,
    ASSETS,
)
from utils.logger import setup_logger, get_logger

logger = setup_logger("binary_predictor")


class LiveFetcher:
    """
    Asynchronous live candle fetcher using pyquotex.
    
    Maintains a rolling buffer of recent candles per asset
    and auto-reconnects on disconnect.
    """

    def __init__(
        self,
        email: str = QUOTEX_EMAIL,
        password: str = QUOTEX_PASSWORD,
        buffer_size: int = 200,
    ):
        self.email = email
        self.password = password
        self.buffer_size = buffer_size
        self.client = None
        self._connected = False
        self._buffers: dict[str, deque] = {}
        self._retry_delay = 1.0
        self._max_retry_delay = 60.0
        self._last_candle_time: dict[str, datetime] = {}

    async def connect(self) -> bool:
        """
        Authenticate and connect to Quotex.
        Returns True on success.
        """
        try:
            from quotexapi.stable_api import Quotex

            self.client = Quotex(
                email=self.email,
                password=self.password,
                lang="en",
            )

            check, reason = await self.client.connect()
            if check:
                self._connected = True
                self._retry_delay = 1.0
                logger.info("✓ Connected to Quotex successfully")
                return True
            else:
                logger.error(f"✗ Quotex connection failed: {reason}")
                return False

        except ImportError:
            logger.error(
                "pyquotex not installed. Install with: pip install pyquotex"
            )
            return False
        except Exception as e:
            logger.error(f"✗ Connection error: {e}")
            return False

    async def reconnect(self) -> bool:
        """
        Reconnect with exponential backoff.
        """
        while True:
            logger.info(
                f"Reconnecting in {self._retry_delay:.0f}s..."
            )
            await asyncio.sleep(self._retry_delay)
            success = await self.connect()
            if success:
                return True
            self._retry_delay = min(
                self._retry_delay * 2, self._max_retry_delay
            )

    async def disconnect(self):
        """Gracefully disconnect from Quotex."""
        if self.client and self._connected:
            try:
                self.client.close()
            except Exception:
                pass
            self._connected = False
            logger.info("Disconnected from Quotex")

    async def fetch_candles(
        self,
        pair: str,
        count: int = 100,
        timeframe: int = TIMEFRAME_SECONDS,
    ) -> pd.DataFrame:
        """
        Fetch recent candles from Quotex for the given pair.

        Parameters
        ----------
        pair : str
            e.g. "EURUSD"
        count : int
            Number of candles to fetch
        timeframe : int
            Candle period in seconds (60 or 300)

        Returns
        -------
        DataFrame with columns [time, open, high, low, close, volume]
        """
        if not self._connected:
            await self.reconnect()

        try:
            # pyquotex asset format
            asset = pair[:3] + "/" + pair[3:]  # "EUR/USD"

            candles = await self.client.get_candles(
                asset, timeframe, count
            )

            if candles is None or len(candles) == 0:
                logger.warning(f"No candles returned for {pair}")
                return pd.DataFrame()

            rows = []
            for c in candles:
                rows.append(
                    {
                        "time": datetime.fromtimestamp(
                            c.get("time", c.get("from", 0)),
                            tz=timezone.utc,
                        ),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("max", c.get("high", 0))),
                        "low": float(c.get("min", c.get("low", 0))),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0)),
                    }
                )

            df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

            # Update rolling buffer
            self._update_buffer(pair, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching candles for {pair}: {e}")
            self._connected = False
            return pd.DataFrame()

    def _update_buffer(self, pair: str, df: pd.DataFrame):
        """Update the in-memory rolling buffer for a pair."""
        if pair not in self._buffers:
            self._buffers[pair] = deque(maxlen=self.buffer_size)

        for _, row in df.iterrows():
            ts = row["time"]
            last = self._last_candle_time.get(pair)
            if last is None or ts > last:
                self._buffers[pair].append(row.to_dict())
                self._last_candle_time[pair] = ts

    def get_latest_candles(
        self, pair: str, count: int = 50
    ) -> pd.DataFrame:
        """
        Get the latest N candles from the in-memory buffer.
        Non-async — reads from the buffer without network calls.
        """
        if pair not in self._buffers or len(self._buffers[pair]) == 0:
            return pd.DataFrame()

        data = list(self._buffers[pair])
        df = pd.DataFrame(data).tail(count).reset_index(drop=True)
        return df

    def get_buffer_size(self, pair: str) -> int:
        """Return current buffer size for a pair."""
        return len(self._buffers.get(pair, []))

    @property
    def is_connected(self) -> bool:
        return self._connected

    def time_since_last_candle(self, pair: str) -> float:
        """Seconds since last candle was received."""
        last = self._last_candle_time.get(pair)
        if last is None:
            return float("inf")
        return (datetime.now(timezone.utc) - last).total_seconds()


async def run_live_fetcher(
    pairs: list[str] = ASSETS,
    interval: int = TIMEFRAME_SECONDS,
    callback=None,
):
    """
    Continuously fetch candles for all pairs.
    Calls `callback(pair, df)` after each fetch.
    """
    fetcher = LiveFetcher()
    await fetcher.connect()

    logger.info(f"Starting live fetcher for {pairs}, interval={interval}s")

    try:
        while True:
            for pair in pairs:
                df = await fetcher.fetch_candles(pair, count=100, timeframe=interval)
                if not df.empty and callback:
                    await callback(pair, df)

            # Sleep until a few seconds before next candle close
            sleep_time = max(interval - 10, 5)
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("Live fetcher cancelled")
    finally:
        await fetcher.disconnect()


if __name__ == "__main__":
    async def _print_callback(pair, df):
        print(f"\n{pair}: {len(df)} candles, latest close={df.iloc[-1]['close']:.5f}")

    asyncio.run(run_live_fetcher(callback=_print_callback))
