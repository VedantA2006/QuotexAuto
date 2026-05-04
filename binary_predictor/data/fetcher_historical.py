"""
Historical data fetcher — Dukascopy + TrueFX.

Downloads 1-minute OHLCV candle data and stores as CSV.
"""

import io
import lzma
import struct
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
    DUKASCOPY_BASE_URL,
    DUKASCOPY_INSTRUMENTS,
    HISTORICAL_YEARS,
)
from utils.logger import setup_logger, get_logger

logger = setup_logger("binary_predictor")


# ═══════════════════════════════════════════════════════════════
# A) Dukascopy fetcher
# ═══════════════════════════════════════════════════════════════

def _decode_bi5(data: bytes, point: float = 1e-5) -> list[dict]:
    """
    Decode a Dukascopy .bi5 (LZMA-compressed) 1-minute candle file.

    Each record is 24 bytes:
        4 bytes: time offset in seconds from start of day (uint32)
        4 bytes: open  (uint32, multiply by point)
        4 bytes: high  (uint32)
        4 bytes: low   (uint32)
        4 bytes: close (uint32)
        4 bytes: volume (float32)
    """
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []

    record_size = 24
    records = []
    for i in range(0, len(raw), record_size):
        if i + record_size > len(raw):
            break
        chunk = raw[i : i + record_size]
        ts_offset, o, h, l, c, v = struct.unpack(">IIIIIf", chunk)
        records.append(
            {
                "time_offset": ts_offset,
                "open": o * point,
                "high": h * point,
                "low": l * point,
                "close": c * point,
                "volume": round(v, 2),
            }
        )
    return records


def _get_point(pair: str) -> float:
    """Point multiplier for decoding prices."""
    if "JPY" in pair.upper():
        return 1e-3
    return 1e-5


def fetch_dukascopy_day(
    pair: str,
    date: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch 1-minute candle data for a single day from Dukascopy.

    Returns an OHLCV DataFrame or empty DataFrame on failure.
    """
    instrument = DUKASCOPY_INSTRUMENTS.get(pair.upper(), pair.upper())
    # Dukascopy months are 0-indexed
    url = (
        f"{DUKASCOPY_BASE_URL}/{instrument}/"
        f"{date.year}/{date.month - 1:02d}/{date.day:02d}/"
        f"BID_candles_min_1.bi5"
    )

    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=15)
        if resp.status_code != 200 or len(resp.content) == 0:
            return pd.DataFrame()
    except requests.RequestException as e:
        logger.warning(f"Dukascopy request failed for {pair} {date.date()}: {e}")
        return pd.DataFrame()

    point = _get_point(pair)
    records = _decode_bi5(resp.content, point)
    if not records:
        return pd.DataFrame()

    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    rows = []
    for r in records:
        ts = day_start + timedelta(seconds=r["time_offset"])
        rows.append(
            {
                "time": ts,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def fetch_dukascopy_range(
    pair: str,
    start_date: datetime,
    end_date: datetime,
    sleep_between: float = 0.15,
) -> pd.DataFrame:
    """
    Fetch multiple days of 1-minute data from Dukascopy.
    """
    logger.info(
        f"Fetching Dukascopy data: {pair} from {start_date.date()} to {end_date.date()}"
    )
    all_frames = []
    current = start_date
    total_days = (end_date - start_date).days
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )

    with tqdm(total=total_days, desc=f"Downloading {pair}") as pbar:
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                df = fetch_dukascopy_day(pair, current, session)
                if not df.empty:
                    all_frames.append(df)
                time.sleep(sleep_between)
            current += timedelta(days=1)
            pbar.update(1)

    if not all_frames:
        logger.warning(f"No data fetched for {pair}")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values("time").reset_index(drop=True)
    result = result.drop_duplicates(subset=["time"])
    logger.info(f"Fetched {len(result):,} candles for {pair}")
    return result


def download_historical(
    pair: str,
    years: int = HISTORICAL_YEARS,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download `years` years of 1-min data for `pair` from Dukascopy.
    Saves to CSV in DATA_DIR.
    """
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    df = fetch_dukascopy_range(pair, start_date, end_date)

    if save and not df.empty:
        out_path = DATA_DIR / f"{pair.upper()}_1min.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df):,} rows → {out_path}")

    return df


# ═══════════════════════════════════════════════════════════════
# B) TrueFX fetcher (backup)
# ═══════════════════════════════════════════════════════════════

def parse_truefx_ticks(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a TrueFX tick CSV into 1-minute OHLC bars.

    TrueFX format: Currency,DateTime,Bid,Ask
    e.g.: EUR/USD,20240101 00:00:00.123,1.10234,1.10245
    """
    logger.info(f"Parsing TrueFX tick file: {filepath}")
    df = pd.read_csv(
        filepath,
        header=None,
        names=["pair", "datetime", "bid", "ask"],
    )
    df["time"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H:%M:%S.%f", utc=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Resample to 1-minute bars
    df = df.set_index("time")
    ohlc = df["mid"].resample("1T").ohlc().dropna()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = df["mid"].resample("1T").count()
    ohlc = ohlc.reset_index()
    logger.info(f"Parsed {len(ohlc):,} 1-min bars from TrueFX")
    return ohlc


def load_truefx_directory(directory: str | Path, pair: str) -> pd.DataFrame:
    """
    Load all TrueFX CSVs for a pair from a directory, combine and save.
    """
    directory = Path(directory)
    files = sorted(directory.glob(f"*{pair}*"))
    if not files:
        logger.warning(f"No TrueFX files found for {pair} in {directory}")
        return pd.DataFrame()

    frames = [parse_truefx_ticks(f) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"])

    out_path = DATA_DIR / f"{pair.upper()}_truefx_1min.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved TrueFX data: {len(df):,} rows → {out_path}")
    return df


# ═══════════════════════════════════════════════════════════════
# C) Aggregator (resample)
# ═══════════════════════════════════════════════════════════════

def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str = "5T",
) -> pd.DataFrame:
    """
    Resample 1-min bars to a coarser timeframe.
    Always handles UTC alignment.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = df.resample(timeframe).agg(agg).dropna(subset=["open"])
    return resampled.reset_index()


def load_cached_data(pair: str) -> Optional[pd.DataFrame]:
    """Load previously downloaded CSV data for a pair."""
    path = DATA_DIR / f"{pair.upper()}_1min.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["time"])
        logger.info(f"Loaded cached data: {pair} — {len(df):,} rows")
        return df
    return None


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair")
    parser.add_argument("--years", type=int, default=2, help="Years of history")
    args = parser.parse_args()

    download_historical(args.pair, args.years)
