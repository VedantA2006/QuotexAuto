"""
Backtest reporting — generates HTML report with interactive charts.

Uses Plotly for equity curves, bar charts, and feature importance plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR
from utils.logger import get_logger

logger = get_logger("binary_predictor")


def _equity_curve_chart(metrics: dict) -> str:
    """Generate equity curve plotly chart as HTML div."""
    eq_df = metrics.get("equity_curve_df", pd.DataFrame())
    if eq_df.empty or "balance" not in eq_df.columns:
        return "<p>No equity curve data available.</p>"

    trade_rows = eq_df[eq_df["action"] == "trade"].copy()
    if trade_rows.empty:
        trade_rows = eq_df.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_rows["time"],
        y=trade_rows["balance"],
        mode="lines",
        name="Equity",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
    ))

    fig.add_hline(
        y=metrics.get("initial_balance", 1000),
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Initial Balance",
    )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Balance ($)",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _monthly_winrate_chart(metrics: dict) -> str:
    """Generate monthly win rate bar chart."""
    monthly_wr = metrics.get("monthly_win_rates", {})
    if not monthly_wr:
        return "<p>No monthly data available.</p>"

    months = [str(k) for k in monthly_wr.keys()]
    rates = [v * 100 for v in monthly_wr.values()]

    colors = ["#00d4aa" if r >= 55.6 else "#ff4757" for r in rates]

    fig = go.Figure(go.Bar(
        x=months,
        y=rates,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
    ))

    fig.add_hline(y=55.6, line_dash="dash", line_color="#ffa502",
                  annotation_text="Breakeven (55.6%)")
    fig.add_hline(y=70, line_dash="dash", line_color="#00d4aa",
                  annotation_text="Target (70%)")

    fig.update_layout(
        title="Monthly Win Rate",
        xaxis_title="Month",
        yaxis_title="Win Rate (%)",
        template="plotly_dark",
        height=350,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _feature_importance_chart(feature_importance: pd.DataFrame, top_n: int = 20) -> str:
    """Generate feature importance bar chart."""
    if feature_importance is None or feature_importance.empty:
        return "<p>No feature importance data available.</p>"

    top = feature_importance.head(top_n).iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color="#7c5cfc",
    ))

    fig.update_layout(
        title=f"Top {top_n} Feature Importance",
        xaxis_title="Importance",
        template="plotly_dark",
        height=500,
        margin=dict(l=180, r=30, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _trade_distribution_chart(metrics: dict) -> str:
    """Generate trade PnL distribution histogram."""
    trades_df = metrics.get("trades_df", pd.DataFrame())
    if trades_df.empty:
        return "<p>No trade data available.</p>"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trades_df["pnl"],
        nbinsx=50,
        marker_color="#7c5cfc",
        opacity=0.8,
    ))

    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _confusion_matrix_chart(metrics: dict) -> str:
    """Generate confusion matrix as HTML table."""
    trades_df = metrics.get("trades_df", pd.DataFrame())
    if trades_df.empty:
        return "<p>No trade data available.</p>"

    # Build confusion matrix
    tp = ((trades_df["prediction"] == "UP") & (trades_df["actual"] == "UP")).sum()
    tn = ((trades_df["prediction"] == "DOWN") & (trades_df["actual"] == "DOWN")).sum()
    fp = ((trades_df["prediction"] == "UP") & (trades_df["actual"] == "DOWN")).sum()
    fn = ((trades_df["prediction"] == "DOWN") & (trades_df["actual"] == "UP")).sum()

    html = f"""
    <table class="confusion-matrix">
        <tr><th></th><th colspan="2">Predicted</th></tr>
        <tr><th></th><th>UP</th><th>DOWN</th></tr>
        <tr><th>Actual UP</th><td class="tp">{tp}</td><td class="fn">{fn}</td></tr>
        <tr><th>Actual DOWN</th><td class="fp">{fp}</td><td class="tn">{tn}</td></tr>
    </table>
    """
    return html


def _fold_summary_table(fold_results: list[dict]) -> str:
    """Generate per-fold summary table."""
    if not fold_results:
        return "<p>No fold results available.</p>"

    rows = ""
    for r in fold_results:
        color = "#00d4aa" if r["filtered_accuracy"] >= 0.556 else "#ff4757"
        rows += f"""
        <tr>
            <td>{r['fold']}</td>
            <td>{r['test_period']}</td>
            <td>{r['xgb_accuracy']:.3f}</td>
            <td>{r['lgbm_accuracy']:.3f}</td>
            <td>{r['ensemble_accuracy']:.3f}</td>
            <td style="color:{color};font-weight:bold">{r['filtered_accuracy']:.3f}</td>
            <td>{r['traded_candles']}/{r['total_candles']}</td>
            <td>{r['trade_rate']:.1%}</td>
        </tr>
        """

    html = f"""
    <table class="fold-table">
        <thead>
            <tr>
                <th>Fold</th><th>Period</th><th>XGB</th><th>LGBM</th>
                <th>Ensemble</th><th>Filtered</th><th>Trades</th><th>Rate</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """
    return html


def generate_html_report(
    metrics: dict,
    fold_results: list[dict] = None,
    feature_importance: pd.DataFrame = None,
    pair: str = "EURUSD",
    timeframe: str = "1M",
    output_path: str | Path = None,
) -> Path:
    """
    Generate a complete HTML backtest report.

    Returns the path to the generated HTML file.
    """
    if output_path is None:
        output_path = BASE_DIR / "backtest" / "report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate chart HTML
    equity_html = _equity_curve_chart(metrics)
    monthly_html = _monthly_winrate_chart(metrics)
    feat_html = _feature_importance_chart(feature_importance)
    dist_html = _trade_distribution_chart(metrics)
    confusion_html = _confusion_matrix_chart(metrics)
    fold_html = _fold_summary_table(fold_results or [])

    wr_pct = metrics.get("win_rate", 0) * 100
    wr_color = "#00d4aa" if wr_pct >= 55.6 else "#ff4757"
    pnl = metrics.get("net_pnl", 0)
    pnl_color = "#00d4aa" if pnl >= 0 else "#ff4757"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binary Predictor Backtest Report — {pair} {timeframe}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2em;
            background: linear-gradient(135deg, #00d4aa, #7c5cfc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .subtitle {{ color: #888; font-size: 0.9em; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .metric-card .label {{ color: #888; font-size: 0.85em; text-transform: uppercase; }}
        .chart-section {{
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{
            padding: 10px 14px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        th {{ background: rgba(124,92,252,0.15); color: #b8a9ff; font-weight: 600; }}
        .confusion-matrix td {{ font-size: 1.3em; font-weight: 700; width: 100px; }}
        .tp, .tn {{ color: #00d4aa; }}
        .fp, .fn {{ color: #ff4757; }}
        .fold-table {{ font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Binary Predictor — Backtest Report</h1>
        <p class="subtitle">{pair} | {timeframe} | Generated {now}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="value" style="color:{wr_color}">{wr_pct:.1f}%</div>
            <div class="label">Win Rate</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('total_trades', 0):,}</div>
            <div class="label">Total Trades</div>
        </div>
        <div class="metric-card">
            <div class="value" style="color:{pnl_color}">${pnl:,.2f}</div>
            <div class="label">Net P&L</div>
        </div>
        <div class="metric-card">
            <div class="value">${metrics.get('final_balance', 0):,.2f}</div>
            <div class="label">Final Balance</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('profit_factor', 0):.2f}</div>
            <div class="label">Profit Factor</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('max_drawdown', 0):.1%}</div>
            <div class="label">Max Drawdown</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            <div class="label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('calmar_ratio', 0):.2f}</div>
            <div class="label">Calmar Ratio</div>
        </div>
    </div>

    <div class="chart-section">{equity_html}</div>
    <div class="chart-section">{monthly_html}</div>

    <div class="chart-section">
        <h3 style="margin-bottom:12px;color:#b8a9ff">Walk-Forward Fold Results</h3>
        {fold_html}
    </div>

    <div class="chart-section">{feat_html}</div>
    <div class="chart-section">{dist_html}</div>

    <div class="chart-section">
        <h3 style="margin-bottom:12px;color:#b8a9ff">Confusion Matrix</h3>
        {confusion_html}
    </div>

    <div style="text-align:center;padding:30px 0;color:#555;font-size:0.8em;">
        Binary Options Candle Direction Predictor v1.0 &mdash; Walk-Forward Backtest
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report saved → {output_path}")
    return output_path
