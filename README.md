# ⚡ Crypto Sentiment Alpha

**Quantitative Analysis of Hyperliquid Trader Behavior Across Crypto Market Sentiment Regimes**

[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

This research platform analyzes **211,000+ trades** from **32 unique accounts** on the Hyperliquid decentralized exchange, cross-referenced with the Crypto Fear & Greed Index, to quantify how market sentiment impacts trader performance. Through behavioral segmentation (Sentiment Followers, Contrarians, Standard Retail) and institutional-grade risk metrics (Win Rate, Profit Factor, Expectancy), we demonstrate **statistically significant** performance variation across sentiment regimes (Kruskal-Wallis H = 959.03, p = 2.70 × 10⁻²⁰⁶).

---

## Key Findings

| Metric | Value |
|--------|-------|
| **Total Trades Analyzed** | 211,219 |
| **Unique Traders** | 32 |
| **Coins Traded** | 246 |
| **Date Range** | Apr 2023 – May 2025 |
| **Statistical Significance** | p = 2.70 × 10⁻²⁰⁶ |

### Trader Archetype Performance (Highlights)

| Cohort | Best Regime | Win Rate | Profit Factor | Expectancy |
|--------|-------------|----------|---------------|------------|
| **Sentiment Follower** | Extreme Greed | 90.5% | 11.94 | $113.00 |
| **Standard Retail** | Extreme Greed | 90.7% | 65.02 | $165.55 |
| **Contrarian** | Fear | 68.2% | 5.59 | $347.79 |

### The Greed Liquidation Trap

Analysis identified **3,213 greed trap events** — trades during Greed/Extreme Greed periods that resulted in bottom-decile losses. The largest single loss was **-$117,990** on ETH during Extreme Greed (F&G value: 84).

> **Strategy Hypothesis**: Contrarian traders who achieve $347.79 expectancy during Fear demonstrate that counter-cyclical positioning during sentiment extremes generates the highest per-trade alpha, despite lower win rates.

---

## Architecture

```
crypto-sentiment-alpha/
├── app.py                     # Streamlit dashboard (6 tabs)
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Dark theme & layout config
├── data/
│   ├── historical_data.csv    # Hyperliquid trade data (211K rows)
│   └── fear_greed_index.csv   # Fear & Greed Index (2,644 days)
├── src/
│   ├── __init__.py            # Package exports
│   ├── data_processor.py      # Phase 1: Data sync & cleaning
│   ├── trader_segmentation.py # Phase 2: Behavioral cohort segmentation
│   ├── metrics_engine.py      # Phase 3: Quantitative metrics
│   └── statistical_tests.py   # Phase 4: Statistical validation
└── notebooks/                 # Exploratory analysis notebooks
```

---

## Methodology

### Phase 1: Advanced Data Synchronization

- **Timestamp Normalization**: IST timestamps (DD-MM-YYYY HH:MM) are converted to UTC and matched to daily sentiment readings.
- **Look-Ahead Bias Mitigation**: Each trade on day *X* is matched to the Fear & Greed value from day *X−1*, ensuring only historically available information influences the analysis.
- **Direction Mapping**: Raw trade directions (Open Long, Close Long, Open Short, Close Short, Buy, Sell, Liquidation, etc.) are cleaned into standardized categories.

### Phase 2: Trader Archetype Segmentation

Traders are classified into behavioral cohorts based on:

| Cohort | Classification Criteria |
|--------|------------------------|
| **Sentiment Follower** | ≥65% long bias during Greed/Extreme Greed with ≥3% of trades in greed periods |
| **Contrarian** | ≤35% long ratio during Greed OR ≥60% short ratio during Fear |
| **Standard Retail** | Default classification |

*Degen archetype (≥5x implied leverage) is defined but not triggered in this dataset.*

### Phase 3: Institutional Risk Metrics

For each cohort × sentiment regime combination:

- **Win Rate**: `Winning Trades / Total Non-Zero PnL Trades`
- **Profit Factor**: `Gross Profits / |Gross Losses|`
- **Expectancy**: `(WinRate × AvgWin) − (LossRate × AvgLoss)`
- **Leverage Efficiency**: PnL heatmap across leverage bands × sentiment regimes

### Phase 4: Statistical Validation

- **Kruskal-Wallis H-test**: Non-parametric ANOVA across all 5 sentiment regimes
- **Mann-Whitney U**: Pairwise regime comparisons with Bonferroni correction
- **Per-Cohort Testing**: Intra-cohort regime significance with effect sizes (rank-biserial correlation)

---

## Interactive Dashboard

The Streamlit dashboard features 6 analysis tabs:

1. **📊 Executive Summary** — KPI cards, PnL by sentiment, cumulative PnL timeline
2. **🌡️ Sentiment Regimes** — Volume analysis, PnL distributions, sentiment × leverage heatmap
3. **🧬 Trader Archetypes** — Cohort KPIs, cross-regime performance, trader leaderboard
4. **📈 Risk Metrics** — Performance matrix, radar charts, win rate vs. profit factor scatter
5. **🔬 Statistical Tests** — Kruskal-Wallis, pairwise Mann-Whitney, per-cohort significance
6. **🔥 The Greed Trap** — Extreme Greed analysis, counter-strategy hypothesis

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/crypto-sentiment-alpha.git
cd crypto-sentiment-alpha

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Data Sources

| Dataset | Source | Records | Update Frequency |
|---------|--------|---------|------------------|
| **Hyperliquid Trades** | Hyperliquid DEX | 211,224 | Historical |
| **Fear & Greed Index** | [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) | 2,644 | Daily |

---

## Statistical Results Summary

### Overall Kruskal-Wallis Test

| Metric | Value |
|--------|-------|
| H-Statistic | 959.03 |
| p-value | 2.70 × 10⁻²⁰⁶ |
| Result | **Highly Significant** |

### Pairwise Comparisons (9/10 significant after Bonferroni correction)

The only non-significant pair is **Extreme Fear vs. Extreme Greed** (p = 1.0 corrected), suggesting that extreme sentiment in either direction produces similar PnL variance patterns.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for institutional-grade Web3 analytics. Data processed with look-ahead bias mitigation and Bonferroni-corrected statistical tests.*
