"""
Binary Options Candle Direction Predictor - Configuration
All configurable parameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models" / "saved"
LOG_DIR = BASE_DIR / "logs"
SIGNAL_LOG = BASE_DIR / "signals_log.csv"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Quotex Credentials ──────────────────────────────────────
QUOTEX_EMAIL = os.getenv("QUOTEX_EMAIL", "")
QUOTEX_PASSWORD = os.getenv("QUOTEX_PASSWORD", "")

# ─── Assets & Timeframe ──────────────────────────────────────
ASSETS = ["EURUSD", "GBPUSD"]
TIMEFRAME = "1M"  # "1M" or "5M"
TIMEFRAME_SECONDS = 60 if TIMEFRAME == "1M" else 300

# ─── Model Parameters ────────────────────────────────────────
MIN_CONFIDENCE = 0.62
ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]  # XGBoost, LightGBM, CNN

# ─── XGBoost Params ──────────────────────────────────────────
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1,

    'eval_metric': 'logloss',
    'random_state': 42
}
XGB_EARLY_STOPPING = 30

# ─── LightGBM Params ─────────────────────────────────────────
LGBM_PARAMS = {
    'n_estimators': 600,
    'max_depth': 6,
    'learning_rate': 0.015,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 20,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'random_state': 42,
    'verbose': -1
}

# ─── CNN Params ───────────────────────────────────────────────
CNN_SEQUENCE_LENGTH = 20
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 128
CNN_PATIENCE = 5
CNN_LR = 0.001

# ─── Risk Management ─────────────────────────────────────────
INITIAL_BALANCE = 1000.0
TRADE_SIZE_PCT = 0.02        # 2% of balance per trade
PAYOUT_RATE = 0.80           # 80% profit on win
LOSS_RATE = 1.0              # 100% loss on loss
SPREAD_COST_PIPS = 0.5
MAX_DAILY_TRADES = 20
MAX_LOSS_STREAK = 5          # Stop trading after N consecutive losses

# ─── Session Filters (UTC) ───────────────────────────────────
TRADE_SESSIONS = ["london", "newyork", "overlap"]
SKIP_ASIAN = True

# Session hours (UTC)
LONDON_OPEN = 8
LONDON_CLOSE = 17
NY_OPEN = 13
NY_CLOSE = 22
OVERLAP_START = 13
OVERLAP_END = 17
ASIAN_START = 0
ASIAN_END = 8

HIGH_VOLATILITY_HOURS = [8, 9, 13, 14, 15, 16]

# ─── News Filter ─────────────────────────────────────────────
SKIP_NEWS_MINUTES = 30       # Skip ±30 min around high-impact news

# ─── Walk-Forward Training ────────────────────────────────────
MIN_TRAIN_MONTHS = 6
PURGE_GAP_CANDLES = 10       # Skip candles between train/test split

# ─── Adaptive Confidence ─────────────────────────────────────
ADAPTIVE_CONFIDENCE = True
ADAPTIVE_WINDOW = 20         # Rolling window for adaptive threshold
ADAPTIVE_LOW_WR = 0.50       # If win rate drops below this...
ADAPTIVE_HIGH_WR = 0.65      # If win rate rises above this...
ADAPTIVE_HIGH_CONF = 0.68    # Raise confidence to this
ADAPTIVE_LOW_CONF = 0.58     # Lower confidence to this

# ─── Model Retraining ────────────────────────────────────────
RETRAIN_INTERVAL_DAYS = 14   # Retrain every 2 weeks

# ─── Regime Detection ────────────────────────────────────────
REGIME_DETECTION = True
REGIME_STATES = 3            # trending, ranging, volatile
REGIME_TRADE_STATES = [0]    # Only trade in state 0 (trending)

# ─── Dashboard ────────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 30
DASHBOARD_PORT = 8501

# ─── Logging ──────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = "INFO"

# ─── Historical Data (Binance) ───────────────────────────────
HISTORICAL_YEARS = 2

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

BINANCE_SYMBOL_MAP = {
    "EURUSD": "EURUSDT",
    "GBPUSD": "GBPUSDT",
    "USDJPY": "USDTJPY",
}

BINANCE_INTERVAL_MAP = {
    "1M": "1m",
    "5M": "5m",
    "15M": "15m",
}
