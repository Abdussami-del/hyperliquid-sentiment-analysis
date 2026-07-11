# ⚡ Crypto Sentiment Analysis

**Data Analysis of Hyperliquid Trader Behavior based on Crypto Market Sentiment**

🔴 **Live Interactive Dashboard:** [View on Streamlit Cloud](https://hyperliquid-sentiment-analysis-azrtevmdttezvp4kpcnez3.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Project Overview

This project analyzes over **211,000 trades** from **32 unique accounts** on the Hyperliquid decentralized exchange. By comparing this trading data with the Crypto Fear & Greed Index, the goal is to see how market mood affects trader profits. We grouped traders by their behavior (Sentiment Followers, Contrarians, Standard Retail) and tracked standard risk metrics like Win Rate and Profit Factor. The results show a clear link between market sentiment and trading success.

---

## Key Findings

| Metric | Value |
|--------|-------|
| **Total Trades Analyzed** | 211,219 |
| **Unique Traders** | 32 |
| **Coins Traded** | 246 |
| **Date Range** | Apr 2023 – May 2025 |

### How Different Traders Performed

| Trader Group | Best Market Mood | Win Rate | Profit Factor | Avg Profit Per Trade |
|--------|-------------|----------|---------------|------------|
| **Sentiment Follower** | Extreme Greed | 90.5% | 11.94 | $113.00 |
| **Standard Retail** | Extreme Greed | 90.7% | 65.02 | $165.55 |
| **Contrarian** | Fear | 68.2% | 5.59 | $347.79 |

### The Greed Liquidation Trap

We found **3,213 "greed trap" events** — these were trades made during Greed or Extreme Greed periods that resulted in massive losses. The largest single loss was **-$117,990** on ETH during Extreme Greed.

> **Trading Strategy Idea**: Traders who bet against the crowd (Contrarians) during "Fear" periods had the highest average profit per trade ($347.79). Even though they win less often, their winning trades are much larger.

---

## Code Structure

```text
crypto-sentiment-alpha/
├── app.py                     # Streamlit dashboard code
├── requirements.txt           # Python packages needed
├── data/                      # Folder for datasets
├── src/
│   ├── data_processor.py      # Cleans and merges the data
│   ├── trader_segmentation.py # Groups traders by their behavior
│   ├── metrics_engine.py      # Calculates performance numbers
│   └── statistical_tests.py   # Runs stats to prove findings
└── notebooks/                 # Jupyter notebooks for early testing
```

---

## How the Data Was Processed

### 1. Data Cleaning
- **Time Fixing**: Timestamps were converted to UTC to match the daily Fear & Greed Index.
- **Preventing Data Leaks**: A trade made on Tuesday is matched with Monday's sentiment score, so we don't accidentally use future information.
- **Simplifying Directions**: Complex trade actions were grouped into simple terms (like Open Long, Close Short, etc.).

### 2. Grouping Traders
Traders were sorted into groups based on how they trade:
- **Sentiment Follower**: Trades mostly in the same direction as the market mood (e.g., buying during Greed).
- **Contrarian**: Bets against the market mood (e.g., buying during Fear).
- **Standard Retail**: Everyone else.

### 3. Measuring Risk
For each group and market mood, we calculated:
- **Win Rate**: Percentage of profitable trades.
- **Profit Factor**: Total money made divided by total money lost.
- **Expectancy**: Average expected profit per trade.

### 4. Statistical Proof
We ran standard statistical tests to make sure our findings were mathematically solid and not just random luck. The tests confirmed that the differences in profit across different market moods are highly significant.

---

## Interactive Dashboard Tabs

The live dashboard includes 6 sections:
1. **📊 Executive Summary** — High-level numbers and charts.
2. **🌡️ Sentiment Regimes** — How trades change based on market mood.
3. **🧬 Trader Archetypes** — Comparing the different trader groups.
4. **📈 Risk Metrics** — Deep dive into win rates and profit factors.
5. **🔬 Statistical Tests** — The math proving the results.
6. **🔥 The Greed Trap** — A closer look at massive losses during Greed periods.

---

## How to Run It Locally

If you want to run the code on your own computer instead of using the live link:

```bash
# Clone the repository
git clone https://github.com/Abdussami-del/hyperliquid-sentiment-analysis.git
cd hyperliquid-sentiment-analysis

# Install the required packages
pip install -r requirements.txt

# Start the dashboard
streamlit run app.py
```

---

## Data Sources

| Dataset | Source | Records | 
|---------|--------|---------|
| **Hyperliquid Trades** | Hyperliquid DEX | 211,224 |
| **Fear & Greed Index** | Alternative.me | 2,644 | 
