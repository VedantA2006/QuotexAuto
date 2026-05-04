"""
Streamlit Dashboard -- Binary Options Multi-Pair Signal Monitor.

Dark-themed dashboard with multi-pair signal table, performance tracking,
model health monitoring, and per-pair status.
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
    MAX_DAILY_TRADES_PER_PAIR,
    MAX_LOSS_STREAK,
)

# --- Page Config -----------------------------------------------------------
st.set_page_config(
    page_title="Binary Predictor Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ------------------------------------------------------------
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

    .pair-row-trade {
        background: rgba(0, 212, 170, 0.06);
        border-left: 3px solid #00d4aa;
    }
    .pair-row-skip-conf {
        background: rgba(255, 165, 2, 0.04);
        border-left: 3px solid #ffa502;
    }
    .pair-row-skip-streak {
        background: rgba(255, 71, 87, 0.06);
        border-left: 3px solid #ff4757;
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


# --- Data Loading ----------------------------------------------------------

def load_all_latest_signals() -> dict:
    """Load latest signals for all pairs from shared JSON."""
    signal_file = BASE_DIR / "latest_signals.json"
    if signal_file.exists():
        try:
            return json.loads(signal_file.read_text())
        except Exception:
            pass
    return {}


def load_latest_signal() -> dict:
    """Load the single latest signal (backward compat)."""
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


# --- Sidebar Settings ------------------------------------------------------

with st.sidebar:
    st.markdown("## Settings")

    min_conf = st.slider(
        "Min Confidence Threshold",
        min_value=0.50,
        max_value=0.80,
        value=MIN_CONFIDENCE,
        step=0.01,
        help="Only show signals above this confidence level",
    )

    selected_pairs = st.multiselect(
        "Filter Pairs",
        ASSETS,
        default=ASSETS,
    )

    auto_refresh = st.checkbox("Auto-Refresh (30s)", value=True)

    st.markdown("---")
    st.markdown("### Model Info")
    model_meta_path = MODEL_DIR / "ensemble_meta.pkl"
    if model_meta_path.exists():
        st.success("Models loaded")
        st.caption(f"Path: `{MODEL_DIR}`")
    else:
        st.warning("No trained models found")
        st.caption("Run `python train.py` first")

    st.markdown("---")
    st.markdown(
        "<p style='color:#555;font-size:0.75em;text-align:center'>"
        "Binary Predictor v2.0 | Multi-Pair MTF</p>",
        unsafe_allow_html=True,
    )


# --- Main Layout -----------------------------------------------------------

st.markdown(
    "<h1 style='text-align:center;background:linear-gradient(135deg,#00d4aa,#7c5cfc);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.2em;margin-bottom:5px'>Binary Predictor</h1>"
    "<p style='text-align:center;color:#888;margin-bottom:30px'>"
    "Multi-Pair Multi-Timeframe Signal Monitor</p>",
    unsafe_allow_html=True,
)

# --- Tab Layout ------------------------------------------------------------
tab_signals, tab_perf, tab_health = st.tabs([
    "Live Signals", "Performance", "Model Health"
])

# =====================================================
# TAB 1: Live Signals -- Multi-Pair Table
# =====================================================
with tab_signals:
    all_signals = load_all_latest_signals()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    active_trades = sum(
        1 for p, s in all_signals.items()
        if s.get("action") == "TRADE" and p in selected_pairs
    )
    active_pairs = len([p for p in selected_pairs if p in all_signals])
    blocked_pairs = sum(
        1 for p, s in all_signals.items()
        if s.get("action") == "SKIP" and "streak" in s.get("reason", "").lower()
        and p in selected_pairs
    )

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#00d4aa">{active_trades}</div>
            <div class="metric-label">Active Signals</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#7c5cfc">{active_pairs}</div>
            <div class="metric-label">Pairs Reporting</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#ffa502">{len(selected_pairs) - blocked_pairs}</div>
            <div class="metric-label">Pairs Active Today</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        now_utc = datetime.now(timezone.utc)
        remaining_1m = 60 - now_utc.second
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:#7c5cfc">{remaining_1m}s</div>
            <div class="metric-label">Next Candle</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Signal Table")

    # Build table data
    table_rows = []
    for pair in selected_pairs:
        sig = all_signals.get(pair, {})
        if not sig:
            table_rows.append({
                "Pair": pair,
                "1M Signal": "---",
                "5M Signal": "---",
                "MTF Agreement": "---",
                "Confidence": 0.0,
                "Action": "WAITING",
            })
            continue

        direction = sig.get("direction", "---")
        confidence = sig.get("confidence", 0)
        action = sig.get("action", "SKIP")
        mtf = sig.get("mtf_agreement", False)
        conf_1m = sig.get("conf_1m", confidence)
        conf_5m = sig.get("conf_5m", 0)
        tf = sig.get("timeframe", "1M")

        if tf == "MTF" or mtf:
            sig_1m = f"{direction} ({conf_1m:.1%})" if conf_1m else "---"
            sig_5m = f"{direction} ({conf_5m:.1%})" if conf_5m else "---"
        else:
            sig_1m = f"{direction} ({confidence:.1%})" if tf == "1M" else "---"
            sig_5m = f"{direction} ({confidence:.1%})" if tf == "5M" else "---"

        table_rows.append({
            "Pair": pair,
            "1M Signal": sig_1m,
            "5M Signal": sig_5m,
            "MTF Agreement": "YES" if mtf else "NO",
            "Confidence": confidence,
            "Action": action,
        })

    if table_rows:
        df_table = pd.DataFrame(table_rows)

        # Color-code via styling
        def color_action(val):
            if val == "TRADE":
                return "background-color: rgba(0,212,170,0.15); color: #00d4aa; font-weight: 700"
            elif val == "SKIP":
                return "background-color: rgba(255,165,2,0.08); color: #ffa502"
            else:
                return "color: #555"

        def color_mtf(val):
            if val == "YES":
                return "color: #00d4aa; font-weight: 700"
            elif val == "NO":
                return "color: #ff4757"
            return "color: #555"

        def color_conf(val):
            if isinstance(val, (int, float)):
                if val >= 0.7:
                    return "color: #00d4aa; font-weight: 700"
                elif val >= 0.62:
                    return "color: #7c5cfc"
                elif val > 0:
                    return "color: #ffa502"
            return "color: #555"

        styled = df_table.style.applymap(
            color_action, subset=["Action"]
        ).applymap(
            color_mtf, subset=["MTF Agreement"]
        ).applymap(
            color_conf, subset=["Confidence"]
        ).format({
            "Confidence": "{:.1%}",
        })

        st.dataframe(styled, use_container_width=True, hide_index=True, height=380)
    else:
        st.info("No signals yet. Start the live engine to receive signals.")

    # Recent signals log
    st.markdown("### Recent Signals")
    signals_df = load_signals_log()
    if not signals_df.empty:
        recent = signals_df.tail(15).iloc[::-1]
        display_cols = [c for c in ["time", "asset", "direction", "confidence", "timeframe", "action", "reason"]
                       if c in recent.columns]
        st.dataframe(
            recent[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No signals recorded yet.")


# =====================================================
# TAB 2: Performance
# =====================================================
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

        # Per-pair performance breakdown
        st.markdown("### Per-Pair Performance")
        if "asset" in traded.columns:
            pair_stats = traded.groupby("asset").agg(
                trades=("won", "count"),
                wins=("won", "sum"),
                win_rate=("won", "mean"),
            ).reset_index()
            pair_stats["win_rate"] = (pair_stats["win_rate"] * 100).round(1)
            pair_stats = pair_stats.sort_values("win_rate", ascending=False)

            st.dataframe(
                pair_stats.rename(columns={
                    "asset": "Pair", "trades": "Trades",
                    "wins": "Wins", "win_rate": "Win Rate %",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # Equity curve
        if "balance" in traded.columns:
            st.markdown("### Balance Over Time")
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

    else:
        st.info("No trade data available yet. Start trading to see performance.")


# =====================================================
# TAB 3: Model Health
# =====================================================
with tab_health:
    signals_df = load_signals_log()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Rolling Win Rate (Last 50 Trades)")
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
        st.markdown("### Top Features")
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
    st.markdown("### Model Status")
    model_files = list(MODEL_DIR.glob("*.pkl")) if MODEL_DIR.exists() else []
    if model_files:
        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        st.success(f"Last model update: **{mtime.strftime('%Y-%m-%d %H:%M')}**")
    else:
        st.warning("No trained models found. Run `python train.py`.")


# --- Auto Refresh ----------------------------------------------------------
if auto_refresh:
    time.sleep(DASHBOARD_REFRESH_SECONDS)
    st.rerun()
