"""
Streamlit Dashboard — Binary Options Candle Direction Predictor.

Dark-themed dashboard with live signals, performance tracking,
model health monitoring, and settings controls.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ASSETS,
    SIGNAL_LOG,
    MODEL_DIR,
    MIN_CONFIDENCE,
    BASE_DIR,
    DASHBOARD_REFRESH_SECONDS,
)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Binary Predictor Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f0f2e 100%);
        color: #e0e0e0;
    }

    .signal-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
    }

    .signal-up {
        border-color: rgba(0, 212, 170, 0.4);
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.1);
    }

    .signal-down {
        border-color: rgba(255, 71, 87, 0.4);
        box-shadow: 0 0 30px rgba(255, 71, 87, 0.1);
    }

    .signal-skip {
        border-color: rgba(255, 165, 2, 0.3);
    }

    .big-arrow {
        font-size: 4em;
        margin: 10px 0;
    }

    .conf-text {
        font-size: 1.4em;
        font-weight: 700;
    }

    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.6em;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.8em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 15, 46, 0.95);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: #aaa;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(124, 92, 252, 0.2);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────

def load_latest_signal() -> dict:
    """Load the latest signal from shared JSON file."""
    signal_file = BASE_DIR / "latest_signal.json"
    if signal_file.exists():
        try:
            return json.loads(signal_file.read_text())
        except Exception:
            pass
    return {}


def load_signals_log() -> pd.DataFrame:
    """Load full signal history."""
    log_path = Path(SIGNAL_LOG)
    if log_path.exists():
        try:
            df = pd.read_csv(log_path)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception:
            pass
    return pd.DataFrame()


# ─── Sidebar Settings ────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    min_conf = st.slider(
        "Min Confidence Threshold",
        min_value=0.50,
        max_value=0.80,
        value=MIN_CONFIDENCE,
        step=0.01,
        help="Only show signals above this confidence level",
    )

    selected_asset = st.selectbox("Asset", ASSETS, index=0)

    selected_tf = st.radio("Timeframe", ["1M", "5M"], index=0, horizontal=True)

    skip_asian = st.checkbox("Skip Asian Session", value=True)

    auto_refresh = st.checkbox("Auto-Refresh (30s)", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    model_meta_path = MODEL_DIR / "ensemble_meta.pkl"
    if model_meta_path.exists():
        st.success("Models loaded ✓")
        st.caption(f"Path: `{MODEL_DIR}`")
    else:
        st.warning("No trained models found")
        st.caption("Run `python train.py` first")

    st.markdown("---")
    st.markdown(
        "<p style='color:#555;font-size:0.75em;text-align:center'>"
        "Binary Predictor v1.0</p>",
        unsafe_allow_html=True,
    )


# ─── Main Layout ─────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(135deg,#00d4aa,#7c5cfc);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.2em;margin-bottom:5px'>🎯 Binary Predictor</h1>"
    "<p style='text-align:center;color:#888;margin-bottom:30px'>"
    "Candle Direction Predictor for Quotex</p>",
    unsafe_allow_html=True,
)

# ─── Tab Layout ───────────────────────────────────────────────
tab_signal, tab_perf, tab_health = st.tabs([
    "📡 Live Signal", "📈 Performance", "🔧 Model Health"
])

# ═══════════════════════════════════════════════
# TAB 1: Live Signal
# ═══════════════════════════════════════════════
with tab_signal:
    signal = load_latest_signal()

    col1, col2 = st.columns([2, 1])

    with col1:
        if signal:
            direction = signal.get("direction", "—")
            confidence = signal.get("confidence", 0)
            action = signal.get("action", "SKIP")
            reason = signal.get("reason", "")
            sig_time = signal.get("time", "—")

            if action == "TRADE":
                css_class = "signal-up" if direction == "UP" else "signal-down"
                arrow = "⬆" if direction == "UP" else "⬇"
                color = "#00d4aa" if direction == "UP" else "#ff4757"
            else:
                css_class = "signal-skip"
                arrow = "⏸"
                color = "#ffa502"

            st.markdown(f"""
            <div class="signal-card {css_class}">
                <div style="color:#888;font-size:0.85em;margin-bottom:8px">
                    {signal.get('asset', selected_asset)} • {sig_time}
                </div>
                <div class="big-arrow" style="color:{color}">{arrow}</div>
                <div style="font-size:2em;font-weight:800;color:{color};margin:8px 0">
                    {direction if action == "TRADE" else "WAITING"}
                </div>
                <div class="conf-text" style="color:{color}">
                    {confidence:.1%} confidence
                </div>
                <div style="color:#888;margin-top:12px;font-size:0.85em">
                    {reason}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="signal-card signal-skip">
                <div class="big-arrow" style="color:#555">⏳</div>
                <div style="font-size:1.5em;color:#555;font-weight:600">
                    No Signal Yet
                </div>
                <div style="color:#666;margin-top:8px;font-size:0.85em">
                    Start the live engine to receive signals
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Confidence gauge
        conf_val = signal.get("confidence", 0) if signal else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_val * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#e0e0e0"}},
            gauge={
                "axis": {"range": [50, 100], "tickcolor": "#555"},
                "bar": {"color": "#7c5cfc"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [50, 55.6], "color": "rgba(255,71,87,0.2)"},
                    {"range": [55.6, 62], "color": "rgba(255,165,2,0.2)"},
                    {"range": [62, 100], "color": "rgba(0,212,170,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#00d4aa", "width": 3},
                    "thickness": 0.8,
                    "value": min_conf * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e0e0e0"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Countdown placeholder
        if signal:
            tf_sec = 60 if selected_tf == "1M" else 300
            now = datetime.now(timezone.utc)
            elapsed = now.second if tf_sec == 60 else (now.minute * 60 + now.second) % tf_sec
            remaining = tf_sec - elapsed
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-value' style='color:#7c5cfc'>{remaining}s</div>"
                f"<div class='metric-label'>Next candle</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Recent signals table
    st.markdown("### Recent Signals")
    signals_df = load_signals_log()
    if not signals_df.empty:
        recent = signals_df.tail(10).iloc[::-1]
        display_cols = [c for c in ["time", "asset", "direction", "confidence", "action", "reason"] if c in recent.columns]
        st.dataframe(
            recent[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No signals recorded yet.")

# ═══════════════════════════════════════════════
# TAB 2: Performance
# ═══════════════════════════════════════════════
with tab_perf:
    signals_df = load_signals_log()
    traded = signals_df[signals_df.get("action", pd.Series()) == "TRADE"] if not signals_df.empty and "action" in signals_df.columns else pd.DataFrame()

    if not traded.empty and "won" in traded.columns:
        col1, col2, col3, col4 = st.columns(4)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = traded[traded["time"].astype(str).str.startswith(today_str)] if "time" in traded.columns else pd.DataFrame()

        with col1:
            today_wr = today_trades["won"].mean() * 100 if len(today_trades) > 0 else 0
            wr_color = "#00d4aa" if today_wr >= 55.6 else "#ff4757"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:{wr_color}">{today_wr:.1f}%</div>
                <div class="metric-label">Today's Win Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:#7c5cfc">{len(today_trades)}</div>
                <div class="metric-label">Today's Trades</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            total_wr = traded["won"].mean() * 100
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color:#00d4aa">{total_wr:.1f}%</div>
                <div class="metric-label">All-Time Win Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(traded)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            """, unsafe_allow_html=True)

        # Equity curve
        if "balance" in traded.columns:
            st.markdown("### 💰 Balance Over Time")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=traded["time"],
                y=traded["balance"],
                mode="lines",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.08)",
            ))
            fig_eq.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=50, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Win/Loss streak
        if len(traded) > 0:
            streak_vals = traded["won"].astype(int).values
            current_streak = 0
            streak_type = "—"
            for v in reversed(streak_vals):
                if current_streak == 0:
                    current_streak = 1
                    streak_type = "Win" if v == 1 else "Loss"
                elif (v == 1 and streak_type == "Win") or (v == 0 and streak_type == "Loss"):
                    current_streak += 1
                else:
                    break

            streak_color = "#00d4aa" if streak_type == "Win" else "#ff4757"
            st.markdown(
                f"<div class='metric-box' style='display:inline-block;padding:12px 24px'>"
                f"<span style='color:{streak_color};font-weight:700;font-size:1.2em'>"
                f"{current_streak}x {streak_type} Streak</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No trade data available yet. Start trading to see performance.")

# ═══════════════════════════════════════════════
# TAB 3: Model Health
# ═══════════════════════════════════════════════
with tab_health:
    signals_df = load_signals_log()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Rolling Win Rate (Last 50 Trades)")
        if not signals_df.empty and "won" in signals_df.columns:
            traded = signals_df[signals_df["action"] == "TRADE"] if "action" in signals_df.columns else signals_df
            if len(traded) >= 10:
                traded = traded.copy()
                traded["rolling_wr"] = traded["won"].rolling(50, min_periods=10).mean()

                fig_rolling = go.Figure()
                fig_rolling.add_trace(go.Scatter(
                    x=list(range(len(traded))),
                    y=traded["rolling_wr"] * 100,
                    mode="lines",
                    line=dict(color="#7c5cfc", width=2),
                ))
                fig_rolling.add_hline(y=55.6, line_dash="dash", line_color="#ffa502")
                fig_rolling.add_hline(y=70, line_dash="dash", line_color="#00d4aa")
                fig_rolling.update_layout(
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=50, r=20, t=20, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[30, 90], title="Win Rate %"),
                    xaxis=dict(title="Trade #"),
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
            else:
                st.info("Need at least 10 trades for rolling win rate.")
        else:
            st.info("No trade outcome data yet.")

    with col2:
        st.markdown("### 🏆 Top Features")
        feat_imp_path = MODEL_DIR / "feature_importance.csv"
        if feat_imp_path.exists():
            feat_df = pd.read_csv(feat_imp_path)
            top10 = feat_df.head(10).iloc[::-1]

            fig_feat = go.Figure(go.Bar(
                x=top10["importance"],
                y=top10["feature"],
                orientation="h",
                marker_color="#7c5cfc",
            ))
            fig_feat.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=140, r=20, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Train models to see feature importance.")

    # Last retrain info
    st.markdown("### 🔄 Model Status")
    model_files = list(MODEL_DIR.glob("*.pkl")) if MODEL_DIR.exists() else []
    if model_files:
        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        st.success(f"Last model update: **{mtime.strftime('%Y-%m-%d %H:%M')}**")
    else:
        st.warning("No trained models found. Run `python train.py`.")


# ─── Auto Refresh ─────────────────────────────────────────────
if auto_refresh:
    time.sleep(DASHBOARD_REFRESH_SECONDS)
    st.rerun()
