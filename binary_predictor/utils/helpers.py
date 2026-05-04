"""
Utility helpers used across the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import json
import csv
from datetime import datetime, timezone


def ensure_utc(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Make sure a datetime column (or index) is tz-aware UTC."""
    if col in df.columns:
        s = pd.to_datetime(df[col], utc=True)
        df[col] = s
    elif df.index.name == col or isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str = "5T",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Resample a 1-minute OHLCV DataFrame to a coarser timeframe.
    
    Parameters
    ----------
    df : DataFrame with columns [time, open, high, low, close, volume]
    timeframe : pandas offset alias, e.g. '5T', '15T', '1H'
    time_col : name of the datetime column

    Returns
    -------
    Resampled DataFrame.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col)
    df = df.sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    resampled = df.resample(timeframe).agg(agg).dropna(subset=["open"])
    resampled = resampled.reset_index()
    resampled.rename(columns={resampled.columns[0]: time_col}, inplace=True)
    return resampled


def append_signal_log(
    filepath: str | Path,
    signal: dict,
) -> None:
    """Append a signal dict to the CSV log."""
    filepath = Path(filepath)
    file_exists = filepath.exists()
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=signal.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal)


def load_signal_log(filepath: str | Path) -> pd.DataFrame:
    """Load the signal log CSV into a DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath, parse_dates=["time"])


def now_utc() -> datetime:
    """Current datetime in UTC."""
    return datetime.now(timezone.utc)


def pip_value(pair: str) -> float:
    """Return pip size for a currency pair."""
    if "JPY" in pair.upper():
        return 0.01
    return 0.0001


def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """Check that a DataFrame has the required columns and no NaN/Inf."""
    for col in required_cols:
        if col not in df.columns:
            return False
    if df[required_cols].isnull().any().any():
        return False
    if np.isinf(df[required_cols].select_dtypes(include=[np.number]).values).any():
        return False
    return True


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
