"""
Crypto Sentiment Alpha "” Production Dashboard
Analyzes Hyperliquid trader behavior across Fear & Greed sentiment regimes.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import load_and_process_data
from src.trader_segmentation import segment_traders
from src.metrics_engine import (
    calculate_cohort_metrics,
    calculate_leverage_efficiency,
    detect_greed_trap,
    get_trader_summary,
)
from src.statistical_tests import (
    run_kruskal_wallis,
    run_pairwise_mannwhitney,
    run_cohort_significance,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Sentiment Alpha",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Shadcn-inspired light-mode professional palette ─── */
:root {
    --bg-primary:    #F5F6FA;
    --bg-surface:    #FFFFFF;
    --bg-surface-2:  #F8F9FB;
    --bg-sidebar:    #18181B;
    --border:        #E4E7EC;
    --border-accent: #F97316;
    --accent:        #F97316;
    --accent-hover:  #EA6C00;
    --accent-dark:   #18181B;
    --green:         #16A34A;
    --red:           #DC2626;
    --amber:         #D97706;
    --blue:          #2563EB;
    --purple:        #7C3AED;
    --text-primary:  #0F172A;
    --text-secondary:#475569;
    --text-muted:    #94A3B8;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:     0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-hover:  0 8px 24px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.06);
    --radius:        12px;
    --radius-sm:     8px;
}

/* ── Base ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp {
    background: var(--bg-primary) !important;
}

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * {
    color: #A1A1AA !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: var(--radius-sm) !important;
    color: #E4E4E7 !important;
}

/* ── Metric Cards ──────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px 22px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-hover) !important;
    transform: translateY(-2px);
    border-color: var(--border-accent) !important;
}
[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: var(--bg-surface) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 9px 18px !important;
    font-size: 0.83rem !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-surface-2) !important;
    color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-dark) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Dividers ──────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Clean Card ────────────────────────────────────── */
.glass-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 14px;
    transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}
.glass-card:hover {
    box-shadow: var(--shadow-hover);
    border-color: var(--border-accent);
    transform: translateY(-1px);
}

/* ── Accent Badge ──────────────────────────────────── */
.header-badge {
    display: inline-block;
    background: var(--accent-dark);
    color: #fff;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Insight Box ───────────────────────────────────── */
.insight-box {
    background: #FFF7ED;
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.65;
}

/* ── Section Headings ──────────────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 3px 0;
    letter-spacing: -0.02em;
}
.section-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 0 0 18px 0;
}

/* ── Hide Streamlit chrome ─────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Inputs ────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── DataFrames ────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Spinner ───────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ─────────────────────────────────────────────────────────
# ── Plotly light theme matching Shadcn reference ─────────────
# ── Plotly light theme helper matching Shadcn reference ──────
def get_layout(**kwargs):
    res = {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(family="Inter, sans-serif", color="#0F172A", size=12),
        "hoverlabel": dict(
            bgcolor="#18181B",
            bordercolor="#F97316",
            font=dict(color="#FAFAFA", family="Inter", size=12),
        ),
    }
    if "margin" not in kwargs:
        res["margin"] = dict(l=36, r=36, t=48, b=36)
    if "legend" not in kwargs:
        res["legend"] = dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color="#475569", size=11),
        )
    res.update(kwargs)
    return res

# Sentiment palette — vivid but professional on light bg
SENTIMENT_COLORS = {
    "Extreme Fear": "#DC2626",   # bold red
    "Fear":         "#F97316",   # orange
    "Neutral":      "#6B7280",   # gray
    "Greed":        "#16A34A",   # green
    "Extreme Greed":"#2563EB",   # blue
}

# Cohort palette — distinct, readable on white
COHORT_COLORS = {
    "Degen":             "#DC2626",  # red
    "Sentiment Follower":"#F97316",  # orange
    "Contrarian":        "#16A34A",  # green
    "Standard Retail":   "#6B7280",  # slate
}

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ── Data Loading ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    """Load, clean, merge, segment, and compute all analytics."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    trader_path = os.path.join(data_dir, "historical_data.csv")
    sentiment_path = os.path.join(data_dir, "fear_greed_index.csv")

    merged = load_and_process_data(trader_path, sentiment_path)
    segmented = segment_traders(merged)
    metrics = calculate_cohort_metrics(segmented)
    leverage_eff = calculate_leverage_efficiency(segmented)
    greed_trap = detect_greed_trap(segmented)
    trader_summary = get_trader_summary(segmented)
    kw_result = run_kruskal_wallis(segmented)
    pairwise = run_pairwise_mannwhitney(segmented)
    cohort_sig = run_cohort_significance(segmented)

    return {
        "raw": segmented,
        "metrics": metrics,
        "leverage_eff": leverage_eff,
        "greed_trap": greed_trap,
        "trader_summary": trader_summary,
        "kw_result": kw_result,
        "pairwise": pairwise,
        "cohort_sig": cohort_sig,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.8rem; margin-bottom: 4px;">⚡</div>
        <div style="font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #F97316, #7B2FFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">
            Crypto Sentiment Alpha
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;">
            Hyperliquid · Fear &amp; Greed Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("###  Navigation")

# ── Load Data ────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading & processing 211K+ trades..."):
    data = load_all_data()

df = data["raw"]
metrics_df = data["metrics"]
trader_summary = data["trader_summary"]

# ── Column name references (from backend) ───────────────────────────────
# Backend columns: fg_classification, sentiment_regime, direction_clean,
# implied_leverage, Closed PnL, Size USD, Account, Coin, date

# Sidebar filters
with st.sidebar:
    available_cohorts = sorted(df["cohort"].unique().tolist())
    selected_cohorts = st.multiselect(
        "Filter Cohorts",
        available_cohorts,
        default=available_cohorts,
        key="cohort_filter",
    )

    top_coins = df["Coin"].value_counts().head(20).index.tolist()
    selected_coins = st.multiselect(
        "Filter Coins (Top 20)",
        top_coins,
        default=top_coins,
        key="coin_filter",
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#6B7280; text-align:center;">'
        "Built for institutional-grade analysis<br>"
        "Data: Hyperliquid DEX · Alternative.me F&G Index"
        "</div>",
        unsafe_allow_html=True,
    )

# Apply filters
mask = df["cohort"].isin(selected_cohorts) & df["Coin"].isin(selected_coins)
df_filtered = df[mask].copy()

# ── Tabs ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Executive Summary",
    "Sentiment Regimes",
    "Trader Archetypes",
    "Risk Metrics",
    "Statistical Tests",
    "The Greed Trap",
])

# 
# TAB 1: EXECUTIVE SUMMARY
# 
with tabs[0]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">Executive Summary</span>
        <span class="header-badge">LIVE DATA</span>
    </div>
    <p class="section-subtitle">
        High-level overview of Hyperliquid trades analyzed against the Crypto Fear &amp; Greed Index.
    </p>
    """, unsafe_allow_html=True)

    # KPI Row
    total_trades = len(df_filtered)
    total_pnl = df_filtered["Closed PnL"].sum()
    unique_traders = df_filtered["Account"].nunique()
    unique_coins = df_filtered["Coin"].nunique()
    nonzero_pnl = df_filtered[df_filtered["Closed PnL"] != 0]
    overall_win_rate = (
        (nonzero_pnl["Closed PnL"] > 0).sum() / len(nonzero_pnl) * 100
        if len(nonzero_pnl) > 0 else 0
    )
    avg_trade_size = df_filtered["Size USD"].mean()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Trades", f"{total_trades:,}")
    k2.metric("Total PnL", f"${total_pnl:,.0f}")
    k3.metric("Win Rate", f"{overall_win_rate:.1f}%")
    k4.metric("Unique Traders", f"{unique_traders}")
    k5.metric("Coins Traded", f"{unique_coins}")
    k6.metric("Avg Trade Size", f"${avg_trade_size:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-title">PnL by Sentiment Regime</div>', unsafe_allow_html=True)
        pnl_by_sentiment = (
            df_filtered.groupby("sentiment_regime")["Closed PnL"]
            .agg(["sum", "mean", "count"])
            .reindex(SENTIMENT_ORDER)
            .dropna()
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pnl_by_sentiment.index,
            y=pnl_by_sentiment["sum"],
            marker=dict(
                color=[SENTIMENT_COLORS.get(s, "#888") for s in pnl_by_sentiment.index],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"${v:,.0f}" for v in pnl_by_sentiment["sum"]],
            textposition="outside",
            textfont=dict(size=11, color="#0F172A"),
            hovertemplate="<b>%{x}</b><br>Total PnL: $%{y:,.0f}<br>Trades: %{customdata:,}<extra></extra>",
            customdata=pnl_by_sentiment["count"],
        ))
        fig.update_layout(get_layout(height=380,
            yaxis=dict(title="Total PnL ($)", gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
            xaxis=dict(title=""),
            showlegend=False,))
        st.plotly_chart(fig, width="stretch", key="pnl_by_sentiment_bar")

    with col_right:
        st.markdown('<div class="section-title">Trader Archetype Distribution</div>', unsafe_allow_html=True)
        cohort_counts = df_filtered.groupby("cohort")["Account"].nunique()

        fig2 = go.Figure(data=[go.Pie(
            labels=cohort_counts.index,
            values=cohort_counts.values,
            hole=0.55,
            marker=dict(
                colors=[COHORT_COLORS.get(c, "#888") for c in cohort_counts.index],
                line=dict(color="#E4E7EC", width=2),
            ),
            textfont=dict(size=12, color="#0F172A"),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Traders: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig2.update_layout(get_layout(height=380,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{unique_traders}</b><br><span style='font-size:11px;color:#9CA3AF'>Traders</span>",
                x=0.5, y=0.5, font_size=22, showarrow=False,
                font=dict(color="#0F172A"),
            )],))
        st.plotly_chart(fig2, width="stretch", key="cohort_pie")

    # PnL Over Time
    st.markdown('<div class="section-title">Cumulative PnL Over Time</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Daily cumulative P&L colored by prevailing market sentiment</p>', unsafe_allow_html=True)

    daily_pnl = (
        df_filtered.groupby("date")
        .agg(daily_pnl=("Closed PnL", "sum"), sentiment=("sentiment_regime", "first"))
        .reset_index()
        .sort_values("date")
    )
    daily_pnl["cum_pnl"] = daily_pnl["daily_pnl"].cumsum()

    fig3 = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        mask_s = daily_pnl["sentiment"] == sentiment
        if mask_s.any():
            fig3.add_trace(go.Scatter(
                x=daily_pnl.loc[mask_s, "date"],
                y=daily_pnl.loc[mask_s, "cum_pnl"],
                mode="markers",
                marker=dict(color=SENTIMENT_COLORS.get(sentiment, "#888"), size=5, opacity=0.7),
                name=sentiment,
                hovertemplate=f"<b>{sentiment}</b><br>Date: %{{x}}<br>Cum PnL: $%{{y:,.0f}}<extra></extra>",
            ))

    fig3.add_trace(go.Scatter(
        x=daily_pnl["date"], y=daily_pnl["cum_pnl"],
        mode="lines", line=dict(color="#F97316", width=2),
        name="Cumulative PnL",
        hovertemplate="Date: %{x}<br>Cum PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig3.update_layout(get_layout(height=350,
        yaxis=dict(title="Cumulative PnL ($)", gridcolor="#F1F5F9"),
        xaxis=dict(title="", gridcolor="#F1F5F9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),))
    st.plotly_chart(fig3, width="stretch", key="cum_pnl_timeline")

    # Key insight
    best_regime = pnl_by_sentiment["sum"].idxmax() if len(pnl_by_sentiment) > 0 else "N/A"
    worst_regime = pnl_by_sentiment["sum"].idxmin() if len(pnl_by_sentiment) > 0 else "N/A"
    best_val = pnl_by_sentiment["sum"].max() if len(pnl_by_sentiment) > 0 else 0
    worst_val = pnl_by_sentiment["sum"].min() if len(pnl_by_sentiment) > 0 else 0

    st.markdown(f"""
    <div class="insight-box">
        <strong> Key Insight:</strong> Traders generated the highest total PnL during
        <strong>{best_regime}</strong> (${best_val:,.0f}) and the lowest during
        <strong>{worst_regime}</strong> (${worst_val:,.0f}).
        The overall win rate across all sentiment regimes is <strong>{overall_win_rate:.1f}%</strong>.
    </div>
    """, unsafe_allow_html=True)


# 
# TAB 2: SENTIMENT REGIME ANALYSIS
# 
with tabs[1]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">Sentiment Regime Analysis</span>
        <span class="header-badge">DEEP DIVE</span>
    </div>
    <p class="section-subtitle">
        How do traders behave and perform across different market sentiment states?
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Trade Volume by Sentiment</div>', unsafe_allow_html=True)
        vol_by_sent = (
            df_filtered.groupby("sentiment_regime")
            .agg(trade_count=("Closed PnL", "count"), total_volume=("Size USD", "sum"))
            .reindex(SENTIMENT_ORDER)
            .dropna()
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=vol_by_sent.index, y=vol_by_sent["trade_count"],
                name="Trade Count",
                marker=dict(color=[SENTIMENT_COLORS.get(s, "#888") for s in vol_by_sent.index], cornerradius=4, opacity=0.8),
                hovertemplate="<b>%{x}</b><br>Trades: %{y:,}<extra></extra>",
            ), secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=vol_by_sent.index, y=vol_by_sent["total_volume"],
                name="Volume ($)", mode="lines+markers",
                line=dict(color="#F97316", width=2), marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Volume: $%{y:,.0f}<extra></extra>",
            ), secondary_y=True,
        )
        fig.update_layout(get_layout(height=380,
            yaxis=dict(title="Trade Count", gridcolor="#F1F5F9"),
            yaxis2=dict(title="Total Volume ($)", gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),))
        st.plotly_chart(fig, width="stretch", key="vol_by_sentiment")

    with col2:
        st.markdown('<div class="section-title">PnL Distribution by Sentiment</div>', unsafe_allow_html=True)
        nonzero = df_filtered[df_filtered["Closed PnL"] != 0].copy()
        clip_hi = nonzero["Closed PnL"].quantile(0.99)
        clip_lo = nonzero["Closed PnL"].quantile(0.01)
        nonzero_clipped = nonzero[(nonzero["Closed PnL"] >= clip_lo) & (nonzero["Closed PnL"] <= clip_hi)]

        fig = go.Figure()
        for sentiment in SENTIMENT_ORDER:
            subset = nonzero_clipped[nonzero_clipped["sentiment_regime"] == sentiment]["Closed PnL"]
            if len(subset) > 0:
                fig.add_trace(go.Violin(
                    y=subset, name=sentiment,
                    line_color=SENTIMENT_COLORS.get(sentiment, "#888"),
                    fillcolor=SENTIMENT_COLORS.get(sentiment, "#888"),
                    opacity=0.5, meanline_visible=True, box_visible=True, points=False,
                ))
        fig.update_layout(get_layout(height=380,
            yaxis=dict(title="Closed PnL ($)", gridcolor="#F1F5F9"),
            showlegend=False, xaxis=dict(title=""),))
        st.plotly_chart(fig, width="stretch", key="pnl_violin")

    # Heatmap: Sentiment × Leverage × PnL
    st.markdown('<div class="section-title">Sentiment × Leverage × PnL Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Average Closed PnL across sentiment regimes and leverage bands</p>', unsafe_allow_html=True)

    lev_eff = data["leverage_eff"]
    if len(lev_eff) > 0:
        # Backend returns: leverage_band, sentiment_regime, Avg PnL ($), Trade Count
        heatmap_pivot = lev_eff.pivot_table(
            index="leverage_band", columns="sentiment_regime", values="Avg PnL ($)", aggfunc="mean"
        )
        col_order = [c for c in SENTIMENT_ORDER if c in heatmap_pivot.columns]
        if col_order:
            heatmap_pivot = heatmap_pivot[col_order]

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.tolist(),
                y=heatmap_pivot.index.astype(str).tolist(),
                colorscale=[
                    [0.0, "#DC2626"], [0.35, "#1F2937"], [0.5, "#1F2937"],
                    [0.65, "#1F2937"], [1.0, "#16A34A"],
                ],
                zmid=0,
                text=[[f"${v:,.0f}" if not pd.isna(v) else "" for v in row] for row in heatmap_pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate="Sentiment: %{x}<br>Leverage: %{y}<br>Avg PnL: $%{z:,.2f}<extra></extra>",
                colorbar=dict(title=dict(text="Avg PnL ($)", font=dict(color="#64748B")), tickfont=dict(color="#64748B")),
            ))
            fig.update_layout(get_layout(height=400,
                xaxis=dict(title="Sentiment Regime"),
                yaxis=dict(title="Leverage Band", autorange="reversed"),))
            st.plotly_chart(fig, width="stretch", key="leverage_heatmap")
        else:
            st.info("Heatmap: no matching sentiment columns in data.")
    else:
        st.info("Insufficient leverage data for heatmap.")

    # Regime Summary Stats
    st.markdown('<div class="section-title">Regime Summary Statistics</div>', unsafe_allow_html=True)

    regime_stats = (
        df_filtered[df_filtered["Closed PnL"] != 0]
        .groupby("sentiment_regime")
        .agg(
            total_trades=("Closed PnL", "count"),
            total_pnl=("Closed PnL", "sum"),
            avg_pnl=("Closed PnL", "mean"),
            median_pnl=("Closed PnL", "median"),
            std_pnl=("Closed PnL", "std"),
            avg_size=("Size USD", "mean"),
        )
        .reindex(SENTIMENT_ORDER)
        .dropna()
    )
    regime_stats.columns = [
        "Total Trades", "Total PnL ($)", "Avg PnL ($)",
        "Median PnL ($)", "Std Dev ($)", "Avg Trade Size ($)",
    ]

    st.dataframe(
        regime_stats.style.format({
            "Total Trades": "{:,.0f}",
            "Total PnL ($)": "${:,.0f}",
            "Avg PnL ($)": "${:,.2f}",
            "Median PnL ($)": "${:,.2f}",
            "Std Dev ($)": "${:,.0f}",
            "Avg Trade Size ($)": "${:,.0f}",
        }).background_gradient(subset=["Total PnL ($)"], cmap="RdYlGn"),
        width="stretch", height=250,
    )

    # Direction breakdown by sentiment
    st.markdown('<div class="section-title">Trade Direction by Sentiment Regime</div>', unsafe_allow_html=True)
    dir_by_sent = (
        df_filtered.groupby(["sentiment_regime", "direction_clean"])
        .size()
        .reset_index(name="count")
    )
    dir_colors = {
        "Open Long": "#16A34A", "Close Long": "#2563EB",
        "Open Short": "#DC2626", "Close Short": "#EF4444",
        "Spot": "#F97316", "Liquidation": "#FF0000", "Other": "#94A3B8",
    }

    fig = go.Figure()
    for direction in ["Open Long", "Open Short", "Close Long", "Close Short", "Spot", "Liquidation", "Other"]:
        subset = dir_by_sent[dir_by_sent["direction_clean"] == direction]
        if len(subset) > 0:
            subset = subset.set_index("sentiment_regime").reindex(SENTIMENT_ORDER).fillna(0).reset_index()
            fig.add_trace(go.Bar(
                x=subset["sentiment_regime"], y=subset["count"],
                name=direction,
                marker_color=dir_colors.get(direction, "#888"),
                marker_cornerradius=2,
            ))
    fig.update_layout(get_layout(height=380,
        barmode="stack",
        yaxis=dict(title="Trade Count", gridcolor="#F1F5F9"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),))
    st.plotly_chart(fig, width="stretch", key="direction_stacked")


# 
# TAB 3: TRADER ARCHETYPES
# 
with tabs[2]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">Trader Archetype Analysis</span>
        <span class="header-badge">BEHAVIORAL</span>
    </div>
    <p class="section-subtitle">
        Segmentation of traders into behavioral cohorts based on execution patterns, leverage, and sentiment alignment.
    </p>
    """, unsafe_allow_html=True)

    # Cohort KPI Cards
    cohort_kpi = (
        df_filtered[df_filtered["Closed PnL"] != 0]
        .groupby("cohort")
        .agg(
            traders=("Account", "nunique"),
            total_trades=("Closed PnL", "count"),
            total_pnl=("Closed PnL", "sum"),
            avg_pnl=("Closed PnL", "mean"),
            win_rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
        )
    )

    if len(cohort_kpi) > 0:
        cohort_cols = st.columns(len(cohort_kpi))
        for i, (cohort, row) in enumerate(cohort_kpi.iterrows()):
            color = COHORT_COLORS.get(cohort, "#888")
            with cohort_cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 3px solid {color};">
                    <div style="font-size: 0.78rem; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">
                        {cohort}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <div style="font-size: 0.7rem; color: #94A3B8;">Traders</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{int(row['traders'])}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #94A3B8;">Win Rate</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: {'#16A34A' if row['win_rate'] > 50 else '#DC2626'};">{row['win_rate']:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #94A3B8;">Total PnL</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: {'#16A34A' if row['total_pnl'] > 0 else '#DC2626'};">${row['total_pnl']:,.0f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #94A3B8;">Trades</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{int(row['total_trades']):,}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Avg PnL by Cohort × Sentiment</div>', unsafe_allow_html=True)
        cohort_sent_pnl = (
            df_filtered[df_filtered["Closed PnL"] != 0]
            .groupby(["cohort", "sentiment_regime"])["Closed PnL"]
            .mean()
            .reset_index()
        )

        fig = go.Figure()
        for cohort in (cohort_kpi.index if len(cohort_kpi) > 0 else []):
            subset = cohort_sent_pnl[cohort_sent_pnl["cohort"] == cohort]
            subset = subset.set_index("sentiment_regime").reindex(SENTIMENT_ORDER).dropna().reset_index()
            fig.add_trace(go.Bar(
                x=subset["sentiment_regime"], y=subset["Closed PnL"],
                name=cohort,
                marker_color=COHORT_COLORS.get(cohort, "#888"),
                marker_cornerradius=4,
                hovertemplate=f"<b>{cohort}</b><br>%{{x}}<br>Avg PnL: $%{{y:,.2f}}<extra></extra>",
            ))
        fig.update_layout(get_layout(height=400, barmode="group",
            yaxis=dict(title="Avg Closed PnL ($)", gridcolor="#F1F5F9"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),))
        st.plotly_chart(fig, width="stretch", key="cohort_sentiment_bar")

    with col_b:
        st.markdown('<div class="section-title">Win Rate by Cohort × Sentiment</div>', unsafe_allow_html=True)
        cohort_sent_wr = (
            df_filtered[df_filtered["Closed PnL"] != 0]
            .groupby(["cohort", "sentiment_regime"])["Closed PnL"]
            .apply(lambda x: (x > 0).mean() * 100)
            .reset_index()
            .rename(columns={"Closed PnL": "win_rate"})
        )

        fig = go.Figure()
        for cohort in (cohort_kpi.index if len(cohort_kpi) > 0 else []):
            subset = cohort_sent_wr[cohort_sent_wr["cohort"] == cohort]
            subset = subset.set_index("sentiment_regime").reindex(SENTIMENT_ORDER).dropna().reset_index()
            fig.add_trace(go.Scatter(
                x=subset["sentiment_regime"], y=subset["win_rate"],
                mode="lines+markers", name=cohort,
                line=dict(color=COHORT_COLORS.get(cohort, "#888"), width=2.5),
                marker=dict(size=8),
                hovertemplate=f"<b>{cohort}</b><br>%{{x}}<br>Win Rate: %{{y:.1f}}%<extra></extra>",
            ))
        fig.add_hline(y=50, line_dash="dash", line_color="#CBD5E1", annotation_text="50% Baseline")
        fig.update_layout(get_layout(height=400,
            yaxis=dict(title="Win Rate (%)", gridcolor="#F1F5F9", range=[0, 100]),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),))
        st.plotly_chart(fig, width="stretch", key="cohort_winrate_lines")

    # Individual Trader Leaderboard
    st.markdown('<div class="section-title">Trader Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Top-performing and worst-performing individual trader accounts</p>', unsafe_allow_html=True)

    if len(trader_summary) > 0:
        display_summary = trader_summary.copy()
        # Rename columns to display format
        display_summary = display_summary.rename(columns={
            "total_pnl": "Total PnL ($)",
            "total_trades": "Total Trades",
            "avg_trade_size": "Avg Trade Size ($)",
            "win_rate_pct": "Win Rate (%)",
            "cohort": "Cohort",
            "favorite_coin": "Favorite Coin",
        })
        display_summary["Account"] = display_summary["Account"].apply(
            lambda x: x[:8] + "..." + x[-6:] if len(str(x)) > 14 else x
        )

        display_cols = ["Account", "Cohort", "Total PnL ($)", "Win Rate (%)", "Total Trades", "Avg Trade Size ($)", "Favorite Coin"]
        available_cols = [c for c in display_cols if c in display_summary.columns]

        st.dataframe(
            display_summary[available_cols].sort_values("Total PnL ($)", ascending=False).style.format({
                "Total PnL ($)": "${:,.0f}",
                "Win Rate (%)": "{:.1f}%",
                "Avg Trade Size ($)": "${:,.0f}",
                "Total Trades": "{:,.0f}",
            }).background_gradient(subset=["Total PnL ($)"], cmap="RdYlGn"),
            width="stretch", height=400,
        )

    # Top coins by PnL
    st.markdown('<div class="section-title">Top Coins by Total PnL</div>', unsafe_allow_html=True)
    coin_pnl = df_filtered.groupby("Coin")["Closed PnL"].sum().sort_values(ascending=False).head(15)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coin_pnl.index, y=coin_pnl.values,
        marker=dict(
            color=["#16A34A" if v > 0 else "#DC2626" for v in coin_pnl.values],
            cornerradius=4,
        ),
        text=[f"${v:,.0f}" for v in coin_pnl.values],
        textposition="outside",
        textfont=dict(size=10, color="#0F172A"),
        hovertemplate="<b>%{x}</b><br>Total PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(get_layout(height=350,
        yaxis=dict(title="Total PnL ($)", gridcolor="#F1F5F9"),
        xaxis=dict(title="", tickangle=-45),
        showlegend=False,))
    st.plotly_chart(fig, width="stretch", key="coin_pnl_bar")


# 
# TAB 4: RISK METRICS
# 
with tabs[3]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">Institutional Risk Metrics</span>
        <span class="header-badge">QUANTITATIVE</span>
    </div>
    <p class="section-subtitle">
        Win Rate, Profit Factor, Expectancy, and Leverage Efficiency across all cohorts and sentiment regimes.
    </p>
    """, unsafe_allow_html=True)

    # Metrics Table
    st.markdown('<div class="section-title">Performance Matrix: Cohort × Sentiment</div>', unsafe_allow_html=True)

    display_metrics = metrics_df[metrics_df["Cohort"].isin(selected_cohorts)].copy()

    if len(display_metrics) > 0:
        # Replace inf with a large number for display
        display_metrics_styled = display_metrics.copy()
        display_metrics_styled["Profit Factor"] = display_metrics_styled["Profit Factor"].replace([np.inf], 999.99)

        st.dataframe(
            display_metrics_styled.style.format({
                "Total Trades": "{:,.0f}",
                "Win Rate (%)": "{:.1f}",
                "Profit Factor": "{:.2f}",
                "Expectancy ($)": "${:,.2f}",
                "Avg Trade Size ($)": "${:,.0f}",
                "Total PnL ($)": "${:,.0f}",
            }).background_gradient(subset=["Win Rate (%)"], cmap="RdYlGn", vmin=30, vmax=70)
            .background_gradient(subset=["Profit Factor"], cmap="RdYlGn", vmin=0, vmax=10)
            .background_gradient(subset=["Expectancy ($)"], cmap="RdYlGn"),
            width="stretch", height=400,
        )

    # Profit Factor Radar
    st.markdown('<div class="section-title">Profit Factor Radar by Cohort</div>', unsafe_allow_html=True)

    if len(display_metrics) > 0:
        fig = go.Figure()
        for cohort in display_metrics["Cohort"].unique():
            subset = display_metrics[display_metrics["Cohort"] == cohort]
            subset = subset.set_index("Sentiment Regime").reindex(SENTIMENT_ORDER).dropna()
            if len(subset) > 0:
                pf_values = subset["Profit Factor"].replace([np.inf], subset["Profit Factor"][subset["Profit Factor"] != np.inf].max() * 1.5 if (subset["Profit Factor"] != np.inf).any() else 10).tolist()
                values = pf_values + [pf_values[0]]
                categories = subset.index.tolist() + [subset.index.tolist()[0]]
                fig.add_trace(go.Scatterpolar(
                    r=values, theta=categories,
                    fill="toself", name=cohort,
                    line=dict(color=COHORT_COLORS.get(cohort, "#888"), width=2),
                    fillcolor=COHORT_COLORS.get(cohort, "#888"),
                    opacity=0.3,
                ))
        fig.update_layout(get_layout(height=450,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="#F1F5F9", tickfont=dict(color="#94A3B8")),
                angularaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),))
        st.plotly_chart(fig, width="stretch", key="profit_factor_radar")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown('<div class="section-title">Expectancy by Cohort</div>', unsafe_allow_html=True)
        if len(display_metrics) > 0:
            exp_by_cohort = display_metrics.groupby("Cohort")["Expectancy ($)"].mean().sort_values(ascending=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=exp_by_cohort.values, y=exp_by_cohort.index, orientation="h",
                marker=dict(color=["#16A34A" if v > 0 else "#DC2626" for v in exp_by_cohort.values], cornerradius=4),
                text=[f"${v:,.2f}" for v in exp_by_cohort.values],
                textposition="outside",
                textfont=dict(size=11, color="#0F172A"),
                hovertemplate="<b>%{y}</b><br>Expectancy: $%{x:,.2f}<extra></extra>",
            ))
            fig.update_layout(get_layout(height=300,
                xaxis=dict(title="Avg Expectancy ($)", gridcolor="#F1F5F9"),
                yaxis=dict(title=""),))
            st.plotly_chart(fig, width="stretch", key="expectancy_bar")

    with col_e2:
        st.markdown('<div class="section-title">Win Rate vs Profit Factor</div>', unsafe_allow_html=True)
        if len(display_metrics) > 0:
            scatter_df = display_metrics.copy()
            scatter_df["Profit Factor"] = scatter_df["Profit Factor"].replace([np.inf], scatter_df["Profit Factor"][scatter_df["Profit Factor"] != np.inf].max() * 1.2 if (scatter_df["Profit Factor"] != np.inf).any() else 10)

            fig = go.Figure()
            for cohort in scatter_df["Cohort"].unique():
                subset = scatter_df[scatter_df["Cohort"] == cohort]
                max_trades = scatter_df["Total Trades"].max()
                sizes = (subset["Total Trades"] / max_trades * 30 + 8) if max_trades > 0 else 15

                fig.add_trace(go.Scatter(
                    x=subset["Win Rate (%)"], y=subset["Profit Factor"],
                    mode="markers+text", name=cohort,
                    marker=dict(
                        color=COHORT_COLORS.get(cohort, "#888"),
                        size=sizes, opacity=0.8,
                        line=dict(width=1, color="#CBD5E1"),
                    ),
                    text=subset["Sentiment Regime"].apply(lambda x: x[:3] if isinstance(x, str) else ""),
                    textposition="top center",
                    textfont=dict(size=9, color="#64748B"),
                    hovertemplate=f"<b>{cohort}</b><br>%{{customdata}}<br>Win Rate: %{{x:.1f}}%<br>PF: %{{y:.2f}}<extra></extra>",
                    customdata=subset["Sentiment Regime"],
                ))
            fig.add_hline(y=1, line_dash="dash", line_color="#CBD5E1", annotation_text="PF = 1")
            fig.add_vline(x=50, line_dash="dash", line_color="#CBD5E1", annotation_text="50% WR")
            fig.update_layout(get_layout(height=300,
                xaxis=dict(title="Win Rate (%)", gridcolor="#F1F5F9"),
                yaxis=dict(title="Profit Factor", gridcolor="#F1F5F9"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),))
            st.plotly_chart(fig, width="stretch", key="wr_vs_pf_scatter")


# 
# TAB 5: STATISTICAL TESTS
# 
with tabs[4]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">Statistical Validation</span>
        <span class="header-badge">SIGNIFICANCE</span>
    </div>
    <p class="section-subtitle">
        Proving that sentiment-driven performance differences are not random noise.
    </p>
    """, unsafe_allow_html=True)

    kw = data["kw_result"]
    h_stat = kw.get("H_statistic", 0) or 0
    p_val = kw.get("p_value", 1) or 1
    interp = kw.get("interpretation", "N/A")
    is_significant = p_val < 0.05 if not np.isnan(p_val) else False

    st.markdown(f"""
    <div class="glass-card">
        <div style="font-size: 0.78rem; color: #F97316; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;">
            Kruskal-Wallis H-Test (Non-Parametric ANOVA)
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            <div>
                <div style="font-size: 0.7rem; color: #94A3B8;">H-Statistic</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #0F172A;">{h_stat:.2f}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #94A3B8;">p-value</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {'#16A34A' if is_significant else '#DC2626'};">{p_val:.2e}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Verdict</div>
                <div style="font-size: 1rem; font-weight: 600; color: {'#16A34A' if is_significant else '#DC2626'}; margin-top: 8px;">
                    {"✅ Statistically Significant" if is_significant else "âš ï¸ Not Significant"}
                </div>
            </div>
        </div>
        <div style="margin-top: 16px; font-size: 0.85rem; color: #64748B; line-height: 1.6;">
            {interp}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1.3, 1])

    with col_p1:
        st.markdown('<div class="section-title">Pairwise Mann-Whitney U Tests</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Comparing PnL distributions between each pair of sentiment regimes (Bonferroni-corrected)</p>', unsafe_allow_html=True)

        pairwise = data["pairwise"]
        if len(pairwise) > 0:
            # Rename columns for display
            pw_display = pairwise.rename(columns={
                "U Statistic": "U-Statistic",
                "p-value (raw)": "p-value",
                "p-value (Bonferroni)": "p-value (corrected)",
                "Effect Size (rank-biserial)": "Effect Size",
            })
            display_pw_cols = ["Regime A", "Regime B", "N_A", "N_B", "U-Statistic", "p-value", "p-value (corrected)", "Effect Size", "Significant"]
            available_pw_cols = [c for c in display_pw_cols if c in pw_display.columns]

            st.dataframe(
                pw_display[available_pw_cols].style.format({
                    "U-Statistic": "{:,.0f}",
                    "p-value": "{:.4e}",
                    "p-value (corrected)": "{:.4e}",
                    "Effect Size": "{:.4f}",
                }),
                width="stretch", height=350,
            )
        else:
            st.info("Insufficient data for pairwise comparisons.")

    with col_p2:
        st.markdown('<div class="section-title">Effect Size Visualization</div>', unsafe_allow_html=True)

        if len(pairwise) > 0:
            pw_viz = pairwise.copy()
            pw_viz["pair"] = pw_viz["Regime A"] + " vs\n" + pw_viz["Regime B"]
            pw_viz = pw_viz.sort_values("Effect Size (rank-biserial)", ascending=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pw_viz["Effect Size (rank-biserial)"],
                y=pw_viz["pair"],
                orientation="h",
                marker=dict(
                    color=["#16A34A" if s else "#6B7280" for s in pw_viz["Significant"]],
                    cornerradius=4,
                ),
                hovertemplate="<b>%{y}</b><br>Effect Size: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(get_layout(height=350,
                xaxis=dict(title="Rank-Biserial Effect Size", gridcolor="#F1F5F9"),
                yaxis=dict(title="", tickfont=dict(size=10)),))
            st.plotly_chart(fig, width="stretch", key="effect_size_bar")

    # Per-Cohort Significance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Per-Cohort Sentiment Significance</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Does sentiment regime significantly affect PnL within each trader archetype?</p>', unsafe_allow_html=True)

    cohort_sig = data["cohort_sig"]
    if len(cohort_sig) > 0:
        # Filter to just Kruskal-Wallis rows for the summary cards
        kw_rows = cohort_sig[cohort_sig["Test"] == "Kruskal-Wallis"]
        if len(kw_rows) > 0:
            sig_cols = st.columns(len(kw_rows))
            for i, (_, row) in enumerate(kw_rows.iterrows()):
                color = COHORT_COLORS.get(row["Cohort"], "#888")
                is_sig_cohort = row["Significant"]
                stat_val = row["Statistic"] if not pd.isna(row["Statistic"]) else 0
                p_val_cohort = row["p-value"] if not pd.isna(row["p-value"]) else 1
                with sig_cols[i]:
                    st.markdown(f"""
                    <div class="glass-card" style="border-top: 3px solid {color}; text-align: center;">
                        <div style="font-size: 0.78rem; color: {color}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">
                            {row['Cohort']}
                        </div>
                        <div style="font-size: 2rem; margin-bottom: 4px;">
                            {'✅' if is_sig_cohort else 'âŒ'}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">
                            H = {stat_val:.2f}<br>
                            p = {p_val_cohort:.2e}
                        </div>
                        <div style="font-size: 0.75rem; color: {'#16A34A' if is_sig_cohort else '#DC2626'}; margin-top: 8px; font-weight: 600;">
                            {'Significant' if is_sig_cohort else 'Not Significant'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Full cohort significance table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detailed Per-Cohort Test Results</div>', unsafe_allow_html=True)
        st.dataframe(
            cohort_sig.style.format({
                "Statistic": "{:,.2f}",
                "p-value": "{:.4e}",
                "p-value (Bonferroni)": "{:.4e}",
                "Effect Size": "{:.4f}",
                "N": "{:,.0f}",
            }),
            width="stretch", height=350,
        )
    else:
        st.info("Insufficient data for cohort significance testing.")


# 
# TAB 6: THE GREED TRAP
# 
with tabs[5]:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span class="section-title">🔥 The Greed Liquidation Trap</span>
        <span class="header-badge">ALPHA INSIGHT</span>
    </div>
    <p class="section-subtitle">
        Investigating whether trading during Extreme Greed leads to catastrophic losses "”
        and proposing a contrarian counter-strategy.
    </p>
    """, unsafe_allow_html=True)

    greed_trap = data["greed_trap"]

    # Even if no high-leverage greed traps, show Extreme Greed analysis
    eg_trades = df_filtered[df_filtered["sentiment_regime"] == "Extreme Greed"]
    eg_nonzero = eg_trades[eg_trades["Closed PnL"] != 0]

    if len(eg_nonzero) > 0:
        eg_losses = eg_nonzero[eg_nonzero["Closed PnL"] < 0]
        eg_wins = eg_nonzero[eg_nonzero["Closed PnL"] > 0]
        eg_loss_rate = len(eg_losses) / len(eg_nonzero) * 100
        eg_avg_loss = eg_losses["Closed PnL"].mean() if len(eg_losses) > 0 else 0
        eg_total_pnl = eg_nonzero["Closed PnL"].sum()
        eg_win_rate = len(eg_wins) / len(eg_nonzero) * 100

        st.markdown(f"""
        <div class="insight-box">
            <strong> Key Finding:</strong> During <strong>Extreme Greed</strong> periods,
            <strong>{eg_loss_rate:.1f}%</strong> of trades resulted in losses,
            with an average loss of <strong>${abs(eg_avg_loss):,.2f}</strong> per losing trade.
            Total PnL during Extreme Greed: <strong style="color: {'#16A34A' if eg_total_pnl > 0 else '#DC2626'};">${eg_total_pnl:,.0f}</strong>.
        </div>
        """, unsafe_allow_html=True)

        # KPIs for Extreme Greed
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Trades in Extreme Greed", f"{len(eg_nonzero):,}")
        g2.metric("Win Rate", f"{eg_win_rate:.1f}%")
        g3.metric("Total PnL", f"${eg_total_pnl:,.0f}")
        g4.metric("Avg Loss", f"${abs(eg_avg_loss):,.2f}" if len(eg_losses) > 0 else "N/A")

        st.markdown("<br>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown('<div class="section-title">PnL Distribution: Extreme Greed</div>', unsafe_allow_html=True)

            clip_q_lo = eg_nonzero["Closed PnL"].quantile(0.02)
            clip_q_hi = eg_nonzero["Closed PnL"].quantile(0.98)
            eg_clipped = eg_nonzero[
                (eg_nonzero["Closed PnL"] >= clip_q_lo) & (eg_nonzero["Closed PnL"] <= clip_q_hi)
            ]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=eg_clipped["Closed PnL"], nbinsx=50,
                marker_color="#DC2626", opacity=0.7,
                hovertemplate="PnL Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="#CBD5E1", annotation_text="Break-even")
            fig.update_layout(get_layout(height=350,
                xaxis=dict(title="Closed PnL ($)", gridcolor="#F1F5F9"),
                yaxis=dict(title="Frequency", gridcolor="#F1F5F9"),
                showlegend=False,))
            st.plotly_chart(fig, width="stretch", key="greed_pnl_hist")

        with col_g2:
            st.markdown('<div class="section-title">Cohort Performance in Extreme Greed</div>', unsafe_allow_html=True)

            eg_by_cohort = (
                eg_nonzero.groupby("cohort")
                .agg(
                    count=("Closed PnL", "count"),
                    total_pnl=("Closed PnL", "sum"),
                    avg_pnl=("Closed PnL", "mean"),
                    win_rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
                )
                .sort_values("total_pnl")
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=eg_by_cohort["total_pnl"],
                y=eg_by_cohort.index,
                orientation="h",
                marker=dict(
                    color=[COHORT_COLORS.get(c, "#888") for c in eg_by_cohort.index],
                    cornerradius=4,
                ),
                text=[f"${v:,.0f} ({c} trades)" for v, c in zip(eg_by_cohort["total_pnl"], eg_by_cohort["count"])],
                textposition="outside",
                textfont=dict(size=10, color="#0F172A"),
                hovertemplate="<b>%{y}</b><br>Total PnL: $%{x:,.0f}<br>Win Rate: %{customdata:.1f}%<extra></extra>",
                customdata=eg_by_cohort["win_rate"],
            ))
            fig.update_layout(get_layout(height=350,
                xaxis=dict(title="Total PnL ($)", gridcolor="#F1F5F9"),
                yaxis=dict(title=""),))
            st.plotly_chart(fig, width="stretch", key="greed_cohort_bar")

        # Sentiment regime comparison: Extreme Greed vs Extreme Fear
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Extreme Greed vs Other Regimes</div>', unsafe_allow_html=True)

        regime_comparison = (
            df_filtered[df_filtered["Closed PnL"] != 0]
            .groupby("sentiment_regime")
            .agg(
                total_pnl=("Closed PnL", "sum"),
                avg_pnl=("Closed PnL", "mean"),
                win_rate=("Closed PnL", lambda x: (x > 0).mean() * 100),
                count=("Closed PnL", "count"),
                avg_size=("Size USD", "mean"),
            )
            .reindex(SENTIMENT_ORDER)
            .dropna()
        )

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Total PnL ($)", "Avg PnL ($)", "Win Rate (%)"],
            horizontal_spacing=0.08,
        )
        colors = [SENTIMENT_COLORS.get(s, "#888") for s in regime_comparison.index]

        fig.add_trace(go.Bar(
            x=regime_comparison.index, y=regime_comparison["total_pnl"],
            marker=dict(color=colors, cornerradius=4),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Total PnL: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=regime_comparison.index, y=regime_comparison["avg_pnl"],
            marker=dict(color=colors, cornerradius=4),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Avg PnL: $%{y:,.2f}<extra></extra>",
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=regime_comparison.index, y=regime_comparison["win_rate"],
            marker=dict(color=colors, cornerradius=4),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
        ), row=1, col=3)

        fig.update_layout(get_layout(height=350))
        fig.update_yaxes(gridcolor="#F1F5F9")
        fig.update_annotations(font=dict(color="#64748B", size=12))
        st.plotly_chart(fig, width="stretch", key="regime_comparison_trio")

        # Strategy Hypothesis
        st.markdown("<br>", unsafe_allow_html=True)

        # Find which cohort performs worst during Extreme Greed
        if len(eg_by_cohort) > 0:
            worst_cohort = eg_by_cohort["total_pnl"].idxmin()
            worst_pnl = eg_by_cohort.loc[worst_cohort, "total_pnl"]
            worst_wr = eg_by_cohort.loc[worst_cohort, "win_rate"]
            worst_count = eg_by_cohort.loc[worst_cohort, "count"]

            # Calculate expectancy for the worst cohort during extreme greed
            worst_eg = eg_nonzero[eg_nonzero["cohort"] == worst_cohort]
            w = worst_eg[worst_eg["Closed PnL"] > 0]
            l = worst_eg[worst_eg["Closed PnL"] < 0]
            wr_frac = len(w) / len(worst_eg) if len(worst_eg) > 0 else 0
            avg_w = w["Closed PnL"].mean() if len(w) > 0 else 0
            avg_l = abs(l["Closed PnL"].mean()) if len(l) > 0 else 0
            exp = (wr_frac * avg_w) - ((1 - wr_frac) * avg_l)

            st.markdown(f"""
            <div class="glass-card" style="border: 1px solid rgba(255,184,0,0.3);">
                <div style="font-size: 0.78rem; color: #F97316; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">
                    💡 Trading Strategy Hypothesis
                </div>
                <div style="font-size: 0.92rem; color: #0F172A; line-height: 1.7;">
                    The <strong>{worst_cohort}</strong> cohort shows the worst performance during <strong>Extreme Greed</strong>
                    with {int(worst_count)} trades, a win rate of <strong>{worst_wr:.1f}%</strong>,
                    and total PnL of <strong>${worst_pnl:,.0f}</strong>.
                    <br><br>
                    <strong>Per-trade Expectancy:</strong> <strong style="color: {'#16A34A' if exp > 0 else '#DC2626'};">${exp:,.2f}</strong>
                    (Avg Win: ${avg_w:,.2f} | Avg Loss: ${avg_l:,.2f})
                    <br><br>
                    <strong>Counter-Strategy:</strong> A strategy that <em>takes the opposite position of {worst_cohort}
                    traders during Extreme Greed</em> would capture this expectancy delta. During Extreme Greed,
                    this cohort's overconfidence creates predictable entry/exit patterns that can be exploited.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Specific greed trap trades (if any)
        if len(greed_trap) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">High-Leverage Greed Trap Events</div>', unsafe_allow_html=True)
            st.markdown('<p class="section-subtitle">Trades with high leverage during Extreme Greed that resulted in significant losses</p>', unsafe_allow_html=True)

            trap_display = greed_trap.head(30).copy()
            trap_display["Account"] = trap_display["Account"].apply(
                lambda x: x[:8] + "..." + x[-6:] if len(str(x)) > 14 else x
            )
            st.dataframe(
                trap_display.style.format({
                    "Size USD": "${:,.0f}",
                    "Closed PnL": "${:,.2f}",
                    "implied_leverage": "{:.1f}x",
                    "fg_value": "{:.0f}",
                }).background_gradient(subset=["Closed PnL"], cmap="Reds_r"),
                width="stretch", height=400,
            )

    elif len(df_filtered[df_filtered["sentiment_regime"].isin(["Greed", "Extreme Greed"])]) > 0:
        # Show Greed analysis even if no Extreme Greed
        greed_all = df_filtered[df_filtered["sentiment_regime"].isin(["Greed", "Extreme Greed"])]
        greed_nonzero = greed_all[greed_all["Closed PnL"] != 0]
        if len(greed_nonzero) > 0:
            total_greed_pnl = greed_nonzero["Closed PnL"].sum()
            greed_wr = (greed_nonzero["Closed PnL"] > 0).mean() * 100
            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Greed Period Analysis:</strong> During Greed/Extreme Greed periods,
                <strong>{len(greed_nonzero):,}</strong> trades were executed with a
                win rate of <strong>{greed_wr:.1f}%</strong> and total PnL of
                <strong>${total_greed_pnl:,.0f}</strong>.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No trades with non-zero PnL during Greed/Extreme Greed periods in the filtered data.")
    else:
        st.info("No trades during Greed/Extreme Greed sentiment regimes in the filtered data.")

