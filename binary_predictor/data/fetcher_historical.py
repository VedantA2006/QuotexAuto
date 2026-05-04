"""
Historical data fetcher — Binance klines API.

Downloads 1-minute (or 5-minute) OHLCV candle data from Binance
and stores as CSV. No API key required.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_DIR,
    HISTORICAL_YEARS,
    BINANCE_BASE_URL,
    BINANCE_SYMBOL_MAP,
    BINANCE_INTERVAL_MAP,
)
from utils.logger import setup_logger, get_logger

logger = setup_logger("binary_predictor")


# ═══════════════════════════════════════════════════════════════
# Binance klines fetcher
# ═══════════════════════════════════════════════════════════════

def fetch_binance_klines(
    pair: str,
    interval: str = "1M",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit_per_request: int = 1000,
    sleep_between: float = 0.1,
) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from Binance klines API.

    Parameters
    ----------
    pair : str
        Internal pair name, e.g. "EURUSD"
    interval : str
        Timeframe key from BINANCE_INTERVAL_MAP, e.g. "1M", "5M"
    start_date : datetime
        Start of the date range (UTC). Defaults to `HISTORICAL_YEARS` ago.
    end_date : datetime
        End of the date range (UTC). Defaults to now.
    limit_per_request : int
        Max candles per API call (Binance caps at 1000).
    sleep_between : float
        Seconds to sleep between requests to respect rate limits.

    Returns
    -------
    DataFrame with columns [time, open, high, low, close, volume]
    """
    symbol = BINANCE_SYMBOL_MAP.get(pair.upper())
    if symbol is None:
        logger.error(f"No Binance symbol mapping for pair: {pair}")
        return pd.DataFrame()

    bi_interval = BINANCE_INTERVAL_MAP.get(interval.upper(), "1m")

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=365 * HISTORICAL_YEARS)

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    logger.info(
        f"Fetching Binance klines: {symbol} ({pair}) | {bi_interval} | "
        f"{start_date.date()} → {end_date.date()}"
    )

    all_rows = []
    current_ms = start_ms

    # Estimate total requests for progress bar
    if bi_interval == "1m":
        candle_ms = 60_000
    elif bi_interval == "5m":
        candle_ms = 300_000
    elif bi_interval == "15m":
        candle_ms = 900_000
    else:
        candle_ms = 60_000

    total_candles = (end_ms - start_ms) // candle_ms
    total_requests = max(total_candles // limit_per_request, 1)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    with tqdm(total=total_requests, desc=f"Downloading {pair}") as pbar:
        while current_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": bi_interval,
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": limit_per_request,
            }

            try:
                resp = session.get(BINANCE_BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.warning(f"Binance request failed: {e}")
                time.sleep(1)
                continue
            except ValueError as e:
                logger.warning(f"Binance JSON decode error: {e}")
                time.sleep(1)
                continue

            if not data or len(data) == 0:
                break

            for kline in data:
                # Binance kline format:
                # [open_time, open, high, low, close, volume,
                #  close_time, quote_vol, trades, taker_buy_base,
                #  taker_buy_quote, ignore]
                all_rows.append({
                    "time": datetime.fromtimestamp(
                        kline[0] / 1000, tz=timezone.utc
                    ),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                })

            # Move start cursor past the last candle received
            last_open_time = data[-1][0]
            current_ms = last_open_time + candle_ms

            pbar.update(1)
            time.sleep(sleep_between)

    if not all_rows:
        logger.warning(f"No data fetched for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    logger.info(f"Fetched {len(df):,} candles for {pair}")
    return df


def fetch_and_save(
    pair: str,
    interval: str = "1M",
    years: int = HISTORICAL_YEARS,
) -> pd.DataFrame:
    """
    Fetch full date range of candles from Binance and save to CSV.

    Parameters
    ----------
    pair : str
        Internal pair name, e.g. "EURUSD"
    interval : str
        Timeframe key, e.g. "1M", "5M"
    years : int
        Years of history to download

    Returns
    -------
    DataFrame with OHLCV data.
    """
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    df = fetch_binance_klines(
        pair, interval=interval,
        start_date=start_date, end_date=end_date,
    )

    if not df.empty:
        bi_interval = BINANCE_INTERVAL_MAP.get(interval.upper(), "1m")
        out_path = DATA_DIR / f"{pair.upper()}_{bi_interval}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df):,} rows -> {out_path}")

    return df


# ═══════════════════════════════════════════════════════════════
# Aggregator (resample)
# ═══════════════════════════════════════════════════════════════

def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str = "5T",
) -> pd.DataFrame:
    """
    Resample OHLCV bars to a coarser timeframe.
    Always handles UTC alignment.

    Parameters
    ----------
    df : DataFrame with columns [time, open, high, low, close, volume]
    timeframe : pandas offset alias, e.g. '5T', '15T', '1H'

    Returns
    -------
    Resampled DataFrame.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = df.resample(timeframe).agg(agg).dropna(subset=["open"])
    return resampled.reset_index()


def load_cached_data(pair: str, interval: str = "1M") -> Optional[pd.DataFrame]:
    """
    Load previously downloaded CSV data for a pair.

    Parameters
    ----------
    pair : str
        Internal pair name, e.g. "EURUSD"
    interval : str
        Timeframe key, e.g. "1M", "5M"

    Returns
    -------
    DataFrame or None if not found.
    """
    bi_interval = BINANCE_INTERVAL_MAP.get(interval.upper(), "1m")
    path = DATA_DIR / f"{pair.upper()}_{bi_interval}.csv"

    # Also check legacy filename format
    legacy_path = DATA_DIR / f"{pair.upper()}_1min.csv"

    for p in [path, legacy_path]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["time"])
            logger.info(f"Loaded cached data: {pair} - {len(df):,} rows from {p.name}")
            return df

    return None


# ═══════════════════════════════════════════════════════════════
# Convenience alias (used by train.py)
# ═══════════════════════════════════════════════════════════════

def download_historical(
    pair: str,
    years: int = HISTORICAL_YEARS,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical data. Alias for fetch_and_save.
    """
    return fetch_and_save(pair, interval="1M", years=years)


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download historical OHLCV data from Binance")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair")
    parser.add_argument("--interval", default="1M", help="Timeframe: 1M, 5M, 15M")
    parser.add_argument("--years", type=int, default=2, help="Years of history")
    args = parser.parse_args()

    fetch_and_save(args.pair, args.interval, args.years)
