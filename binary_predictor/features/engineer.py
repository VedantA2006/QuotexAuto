"""
Feature engineering — compute all 56 features for candle prediction.

Every feature is computed from PAST data only (no lookahead bias).
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import RobustScaler
from typing import Optional
import joblib
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_DIR
from utils.logger import get_logger

logger = get_logger("binary_predictor")

# Feature column list (for ordering / reference)
FEATURE_COLUMNS = [
    # Candle pattern features (1-11)
    "body_ratio", "upper_wick", "lower_wick", "is_doji",
    "engulf_score", "prev_candle_dir", "prev2_candle_dir",
    "sequence_3", "body_size_normalized", "range_vs_atr",
    # Price/Indicator features (12-29)
    "rsi_7", "rsi_14", "rsi_slope",
    "ema5", "ema13", "ema_cross_dist", "ema_cross_signal",
    "bb_upper", "bb_lower", "bb_mid", "bb_percent_b", "bb_squeeze",
    "atr_7", "atr_normalized",
    "macd_histogram", "macd_hist_direction",
    "price_vs_ema5", "price_vs_ema13",
    "momentum_5", "momentum_10",
    # Multi-candle patterns (30-37)
    "hammer", "shooting_star",
    "bullish_engulfing", "bearish_engulfing",
    "three_white_soldiers", "three_black_crows",
    "inside_bar", "outside_bar",
    # Session/Time features (38-48)
    "hour_of_day", "minute_of_hour", "day_of_week",
    "is_london_open", "is_ny_open", "is_overlap",
    "is_asian_session", "minutes_to_session_open",
    "is_monday", "is_friday", "is_high_volatility_hour",
    # Market structure features (49-55)
    "recent_high_20", "recent_low_20",
    "distance_to_high", "distance_to_low",
    "price_position",
    "support_bounce", "resistance_reject",
    # Noise reduction (56)
    "rolling_win_rate_20",
]


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle pattern features (1-11)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    full_range = h - l + 1e-9

    df["candle_direction"] = (c > o).astype(int)
    df["body_ratio"] = body / full_range
    df["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / full_range
    df["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / full_range
    df["is_doji"] = (df["body_ratio"] < 0.1).astype(int)

    # Engulfing score
    prev_body = body.shift(1)
    df["engulf_score"] = (body / (prev_body + 1e-9)).clip(upper=3.0)

    # Previous candle directions
    df["prev_candle_dir"] = df["candle_direction"].shift(1)
    df["prev2_candle_dir"] = df["candle_direction"].shift(2)

    # 3-candle sequence encoded as integer 0-7
    df["sequence_3"] = (
        df["candle_direction"].shift(2).fillna(0).astype(int) * 4
        + df["candle_direction"].shift(1).fillna(0).astype(int) * 2
        + df["candle_direction"].fillna(0).astype(int)
    )

    # ATR(7) for normalization
    atr_ind = AverageTrueRange(h, l, c, window=7)
    atr_7 = atr_ind.average_true_range()
    df["atr_7"] = atr_7

    df["body_size_normalized"] = (c - o) / (atr_7 + 1e-9)
    df["range_vs_atr"] = (h - l) / (atr_7 + 1e-9)

    return df


def compute_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price/indicator features (12-29)."""
    c = df["close"]
    h, l = df["high"], df["low"]
    atr_7 = df["atr_7"]

    # RSI
    rsi7 = RSIIndicator(c, window=7).rsi()
    rsi14 = RSIIndicator(c, window=14).rsi()
    df["rsi_7"] = rsi7
    df["rsi_14"] = rsi14
    df["rsi_slope"] = rsi7 - rsi7.shift(3)

    # EMA
    ema5 = EMAIndicator(c, window=5).ema_indicator()
    ema13 = EMAIndicator(c, window=13).ema_indicator()
    df["ema5"] = ema5
    df["ema13"] = ema13
    df["ema_cross_dist"] = (ema5 - ema13) / (atr_7 + 1e-9)
    df["ema_cross_signal"] = (ema5 > ema13).astype(int)

    # Bollinger Bands
    bb = BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    bb_range = df["bb_upper"] - df["bb_lower"] + 1e-9
    df["bb_percent_b"] = (c - df["bb_lower"]) / bb_range
    df["bb_squeeze"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

    # ATR normalized
    df["atr_normalized"] = atr_7 / (c + 1e-9)

    # MACD
    macd = MACD(c, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd.macd_diff()
    df["macd_histogram"] = macd_hist
    df["macd_hist_direction"] = (macd_hist > macd_hist.shift(1)).astype(int)

    # Price vs EMAs
    df["price_vs_ema5"] = (c - ema5) / (atr_7 + 1e-9)
    df["price_vs_ema13"] = (c - ema13) / (atr_7 + 1e-9)

    # Momentum
    df["momentum_5"] = c / c.shift(5) - 1
    df["momentum_10"] = c / c.shift(10) - 1

    return df


def compute_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-candle pattern features (30-37)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    full_range = h - l + 1e-9
    direction = df["candle_direction"]

    # Body position within range
    body_top = pd.concat([o, c], axis=1).max(axis=1)
    body_bottom = pd.concat([o, c], axis=1).min(axis=1)

    upper_wick_size = h - body_top
    lower_wick_size = body_bottom - l

    # Hammer: body in upper third, long lower wick (>2x body), small upper wick
    df["hammer"] = (
        (lower_wick_size > 2 * body)
        & (upper_wick_size < body * 0.5)
        & (body_top > l + full_range * 0.66)
    ).astype(int)

    # Shooting star: body in lower third, long upper wick, small lower wick
    df["shooting_star"] = (
        (upper_wick_size > 2 * body)
        & (lower_wick_size < body * 0.5)
        & (body_bottom < l + full_range * 0.33)
    ).astype(int)

    # Bullish engulfing
    prev_dir = direction.shift(1)
    prev_body = body.shift(1)
    df["bullish_engulfing"] = (
        (direction == 1) & (prev_dir == 0) & (body > prev_body)
    ).astype(int)

    # Bearish engulfing
    df["bearish_engulfing"] = (
        (direction == 0) & (prev_dir == 1) & (body > prev_body)
    ).astype(int)

    # Three white soldiers
    df["three_white_soldiers"] = (
        (direction == 1)
        & (direction.shift(1) == 1)
        & (direction.shift(2) == 1)
        & (c > c.shift(1))
        & (c.shift(1) > c.shift(2))
    ).astype(int)

    # Three black crows
    df["three_black_crows"] = (
        (direction == 0)
        & (direction.shift(1) == 0)
        & (direction.shift(2) == 0)
        & (c < c.shift(1))
        & (c.shift(1) < c.shift(2))
    ).astype(int)

    # Inside bar
    df["inside_bar"] = (
        (h < h.shift(1)) & (l > l.shift(1))
    ).astype(int)

    # Outside bar
    df["outside_bar"] = (
        (h > h.shift(1)) & (l < l.shift(1))
    ).astype(int)

    return df


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute session/time features (38-48)."""
    t = pd.to_datetime(df["time"], utc=True)

    df["hour_of_day"] = t.dt.hour
    df["minute_of_hour"] = t.dt.minute
    df["day_of_week"] = t.dt.dayofweek

    df["is_london_open"] = ((t.dt.hour >= 8) & (t.dt.hour < 17)).astype(int)
    df["is_ny_open"] = ((t.dt.hour >= 13) & (t.dt.hour < 22)).astype(int)
    df["is_overlap"] = ((t.dt.hour >= 13) & (t.dt.hour < 17)).astype(int)
    df["is_asian_session"] = ((t.dt.hour >= 0) & (t.dt.hour < 8)).astype(int)

    # Minutes since last major session open
    def _minutes_to_session(hour):
        session_opens = [0, 8, 13]  # Asian, London, NY
        mins_since = []
        for so in session_opens:
            diff = (hour - so) % 24
            mins_since.append(diff * 60)
        return min(mins_since)

    df["minutes_to_session_open"] = df["hour_of_day"].apply(_minutes_to_session)

    df["is_monday"] = (t.dt.dayofweek == 0).astype(int)
    df["is_friday"] = (t.dt.dayofweek == 4).astype(int)

    high_vol_hours = {8, 9, 13, 14, 15, 16}
    df["is_high_volatility_hour"] = t.dt.hour.isin(high_vol_hours).astype(int)

    return df


def compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market structure features (49-55)."""
    c = df["close"]
    h, l = df["high"], df["low"]
    atr_7 = df["atr_7"]
    direction = df["candle_direction"]

    df["recent_high_20"] = h.rolling(20).max()
    df["recent_low_20"] = l.rolling(20).min()

    rng = df["recent_high_20"] - df["recent_low_20"] + 1e-9

    df["distance_to_high"] = (df["recent_high_20"] - c) / (atr_7 + 1e-9)
    df["distance_to_low"] = (c - df["recent_low_20"]) / (atr_7 + 1e-9)
    df["price_position"] = (c - df["recent_low_20"]) / rng

    # Support bounce
    df["support_bounce"] = (
        ((c - df["recent_low_20"]) < 0.5 * atr_7)
        & (direction == 1)
    ).astype(int)

    # Resistance reject
    df["resistance_reject"] = (
        ((df["recent_high_20"] - c) < 0.5 * atr_7)
        & (direction == 0)
    ).astype(int)

    return df


def compute_noise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute noise reduction features (56)."""
    df["rolling_win_rate_20"] = df["candle_direction"].rolling(20).mean()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add next candle direction as the prediction target."""
    df["next_candle_direction"] = df["candle_direction"].shift(-1)
    return df


def engineer_features(
    df: pd.DataFrame,
    add_target_col: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Master function: compute ALL features from raw OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns [time, open, high, low, close, volume]
    add_target_col : whether to add the target column (next_candle_direction)
    drop_na : whether to drop rows with NaN values

    Returns
    -------
    DataFrame with all features added.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Ensure numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0

    # Compute all feature groups
    df = compute_candle_features(df)
    df = compute_indicator_features(df)
    df = compute_pattern_features(df)
    df = compute_session_features(df)
    df = compute_structure_features(df)
    df = compute_noise_features(df)

    # Add target
    if add_target_col:
        df = add_target(df)

    # Drop NaN rows (first ~30 rows lose data due to lookback)
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    logger.info(
        f"Engineered {len(FEATURE_COLUMNS)} features, {len(df):,} valid rows"
    )
    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names."""
    return FEATURE_COLUMNS.copy()


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit: bool = True,
    save_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Scale continuous features using RobustScaler.
    
    Parameters
    ----------
    df : DataFrame with feature columns
    scaler : existing scaler (for transform-only on test data)
    fit : whether to fit the scaler (True for train, False for test)
    save_path : path to save the fitted scaler

    Returns
    -------
    (scaled DataFrame, fitted scaler)
    """
    # Identify continuous features (exclude binary/categorical)
    binary_cols = {
        "is_doji", "prev_candle_dir", "prev2_candle_dir",
        "ema_cross_signal", "macd_hist_direction",
        "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
        "three_white_soldiers", "three_black_crows", "inside_bar", "outside_bar",
        "is_london_open", "is_ny_open", "is_overlap", "is_asian_session",
        "is_monday", "is_friday", "is_high_volatility_hour",
        "support_bounce", "resistance_reject",
    }

    continuous_cols = [c for c in FEATURE_COLUMNS if c in df.columns and c not in binary_cols]

    if scaler is None:
        scaler = RobustScaler()

    df = df.copy()
    if fit:
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    else:
        df[continuous_cols] = scaler.transform(df[continuous_cols])

    if save_path:
        joblib.dump(scaler, save_path)
        logger.info(f"Saved scaler → {save_path}")

    return df, scaler


def load_scaler(path: Path) -> RobustScaler:
    """Load a saved scaler."""
    return joblib.load(path)
