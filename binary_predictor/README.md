# Binary Options Candle Direction Predictor

> Predict the direction (UP/DOWN) of the next candle on Quotex with maximum accuracy.  
> Target win rate: **70%+** using ML ensemble + multi-layer filtering.

---

## 🎯 Overview

A complete, production-ready system for predicting binary options candle direction on the Quotex trading platform. Combines **XGBoost**, **LightGBM**, and a **1D CNN** in a weighted ensemble, filtered by confidence thresholds, session timing, and market regime detection.

### How it achieves 70%+ win rate

The key insight: **don't trade every candle**. Only trade the ~30-40% of candles where all conditions align:

| Filter Layer | Estimated Win Rate |
|---|---|
| Raw model accuracy | 55-60% |
| + Confidence filtering (>0.62) | 63-70% |
| + Session filtering (London/NY overlap) | 65-72% |
| + News event skipping | 67-74% |
| + Regime filtering (trending only) | 70-78% |

---

## 🏗️ Project Structure

```
binary_predictor/
├── config.py                  # All settings in one place
├── train.py                   # Main training entry point
├── live.py                    # Main live trading entry point
├── data/
│   ├── fetcher_historical.py  # Download from Dukascopy/TrueFX
│   ├── fetcher_live.py        # Real-time candles via pyquotex
│   └── raw/                   # CSV storage for OHLC data
├── features/
│   ├── engineer.py            # 56 features: candle, indicator, session, structure
│   └── selector.py            # Feature importance + SHAP selection
├── models/
│   ├── trainer.py             # Walk-forward training pipeline
│   ├── xgb_model.py           # XGBoost wrapper
│   ├── lgbm_model.py          # LightGBM wrapper
│   ├── cnn_model.py           # 1D CNN (TensorFlow/Keras)
│   ├── ensemble.py            # Weighted ensemble + adaptive confidence
│   └── saved/                 # Serialized trained models
├── backtest/
│   ├── engine.py              # Walk-forward backtesting engine
│   └── report.py              # HTML report with Plotly charts
├── signals/
│   └── generator.py           # Live signal loop (async)
├── dashboard/
│   └── app.py                 # Streamlit dark-themed dashboard
├── utils/
│   ├── logger.py              # Centralized logging
│   └── helpers.py             # Utility functions
├── logs/                      # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd binary_predictor
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
# Edit .env with your Quotex email and password
```

### 3. Train Models

```bash
# Train on EUR/USD 1-minute data (downloads 2 years of history)
python train.py --asset EURUSD --timeframe 1M

# Train on GBP/USD with custom confidence
python train.py --asset GBPUSD --min-confidence 0.65

# Skip download if data is already cached
python train.py --asset EURUSD --skip-download

# Use 5-minute candles
python train.py --asset EURUSD --timeframe 5M
```

Training will:
1. Download historical data from Dukascopy
2. Engineer 56 features per candle
3. Run walk-forward training (monthly folds)
4. Backtest with all filters applied
5. Generate an HTML report
6. Save trained models

### 4. Go Live

```bash
# Start live signal generator + dashboard
python live.py

# Custom options
python live.py --asset EURUSD GBPUSD --timeframe 1M --min-confidence 0.62

# Without dashboard
python live.py --no-dashboard
```

### 5. View Dashboard

Open your browser to `http://localhost:8501` after starting `live.py`.

The dashboard shows:
- **Live signal panel** — current UP/DOWN signal with confidence gauge
- **Performance tracking** — win rate, P&L, streaks
- **Model health** — rolling win rate, feature importance

---

## 🧠 Features (56 total)

### Candle Patterns (11)
Body ratio, upper/lower wick, doji detection, engulfing score, 3-candle sequence encoding, ATR-normalized body size

### Price Indicators (18)
RSI (7,14) with slope, EMA (5,13) crossover, Bollinger Bands %B + squeeze, MACD histogram, momentum (5,10)

### Multi-Candle Patterns (8)
Hammer, shooting star, bullish/bearish engulfing, three white soldiers, three black crows, inside bar, outside bar

### Session/Time (11)
Hour, day-of-week, London/NY/overlap session flags, high-volatility hour detection

### Market Structure (8)
20-bar high/low channels, price position, support bounce, resistance rejection, rolling win rate

---

## ⚙️ Key Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE` | 0.62 | Minimum prediction confidence to trade |
| `ENSEMBLE_WEIGHTS` | [0.4, 0.4, 0.2] | XGB, LGBM, CNN weights |
| `TRADE_SIZE_PCT` | 0.02 | 2% of balance per trade |
| `PAYOUT_RATE` | 0.80 | 80% payout on win |
| `MAX_DAILY_TRADES` | 20 | Max trades per day |
| `MAX_LOSS_STREAK` | 5 | Stop after N consecutive losses |
| `SKIP_ASIAN` | True | Skip low-momentum Asian session |
| `ADAPTIVE_CONFIDENCE` | True | Auto-adjust threshold based on performance |

---

## 📊 Walk-Forward Training

Unlike simple train/test splits, this system uses **walk-forward validation**:

1. Split data into monthly chunks
2. Train on months 1..N, test on month N+1
3. Slide forward, retrain on 1..N+1, test on N+2
4. **Purge gap**: Skip 10 candles between train/test to prevent leakage
5. Minimum training window: 6 months

This ensures the model is always evaluated on truly unseen future data.

---

## 🛡️ Risk Management

- **2% position sizing** (Kelly-conservative)
- **5-loss streak circuit breaker** — stops trading for the day
- **20 max daily trades** — prevents overtrading
- **Adaptive confidence** — raises threshold when performance drops
- **Spread cost deduction** — realistic P&L accounting
- **Session filtering** — avoids choppy Asian session

---

## 📈 Backtest Report

After training, find the HTML report at:
```
backtest/report_EURUSD_1M.html
```

The report includes:
- Equity curve (interactive Plotly chart)
- Monthly win rate bar chart
- Per-fold walk-forward results table
- Feature importance (top 20)
- Trade P&L distribution histogram
- Confusion matrix

---

## 🔄 Model Retraining

Retrain every 2 weeks with fresh data:

```bash
python train.py --asset EURUSD --skip-download
```

The system compares new model performance against the current model before replacing it.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Binary options trading involves substantial risk of loss. Past performance does not guarantee future results. Always trade responsibly and never risk money you cannot afford to lose.

---

## 📋 License

MIT License — see LICENSE file for details.
