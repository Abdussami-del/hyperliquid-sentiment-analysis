"""
data_processor.py — Data loading, cleaning, and merging module.

Loads historical trade data and fear/greed sentiment index,
cleans both datasets, and produces a merged DataFrame ready
for downstream analytics.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Direction-mapping table ─────────────────────────────────────
_DIRECTION_MAP = {
    "Open Long": "Open Long",
    "Close Long": "Close Long",
    "Open Short": "Open Short",
    "Close Short": "Close Short",
    "Buy": "Spot",
    "Sell": "Spot",
    "Spot Dust Conversion": "Spot",
    "Short > Long": "Other",
    "Long > Short": "Other",
    "Auto-Deleveraging": "Liquidation",
    "Liquidated Isolated Short": "Liquidation",
    "Settlement": "Other",
}


def _load_historical(path: str) -> pd.DataFrame:
    """Load and clean historical trade CSV."""
    df = pd.read_csv(path, low_memory=False)

    # ── Timestamp conversion ────────────────────────────────────
    # The 'Timestamp' column is heavily rounded (many identical values),
    # so we use 'Timestamp IST' (DD-MM-YYYY HH:MM in IST) as primary.
    df["datetime_ist"] = pd.to_datetime(
        df["Timestamp IST"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    # Convert IST → UTC (IST = UTC+05:30)
    df["datetime_utc"] = (
        df["datetime_ist"]
        .dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )

    # Fallback: use the Unix timestamp for any rows where IST parsing failed
    mask_missing = df["datetime_utc"].isna()
    if mask_missing.any():
        ts_numeric = pd.to_numeric(df.loc[mask_missing, "Timestamp"], errors="coerce")
        df.loc[mask_missing, "datetime_utc"] = pd.to_datetime(
            ts_numeric, unit="ms", errors="coerce", utc=True
        )

    # Extract date for merge key
    df["date"] = df["datetime_utc"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # ── Numeric columns ────────────────────────────────────────
    for col in ["Execution Price", "Size Tokens", "Size USD", "Closed PnL", "Fee"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── Implied leverage ────────────────────────────────────────
    denominator = df["Execution Price"] * df["Size Tokens"]
    # Guard against division by zero / near-zero
    safe_denom = denominator.where(denominator.abs() > 1e-8, other=np.nan)
    df["implied_leverage"] = (df["Size USD"] / safe_denom).fillna(1.0)
    # Clip extreme values (data artifacts) to a sensible ceiling
    df["implied_leverage"] = df["implied_leverage"].clip(lower=0.5, upper=200.0)

    # ── Clean Direction ─────────────────────────────────────────
    df["direction_clean"] = df["Direction"].map(_DIRECTION_MAP).fillna("Other")

    return df


def _load_sentiment(path: str) -> pd.DataFrame:
    """Load and clean fear/greed sentiment CSV."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Rename for clarity after merge
    df = df.rename(columns={
        "value": "fg_value",
        "classification": "fg_classification",
    })

    # ── Look-ahead bias mitigation ──────────────────────────────
    # A trade on day X should use sentiment from day X-1 (the most
    # recent *known* reading at the time of the trade).
    df["date"] = df["date"] + pd.Timedelta(days=1)

    # Keep only columns needed for merge
    df = df[["date", "fg_value", "fg_classification"]].drop_duplicates(subset="date")

    return df


def load_and_process_data(
    trader_path: str = r"C:\Users\Nalbu\crypto-sentiment-alpha\data\historical_data.csv",
    sentiment_path: str = r"C:\Users\Nalbu\crypto-sentiment-alpha\data\fear_greed_index.csv",
) -> pd.DataFrame:
    """
    End-to-end loader: read, clean, merge trades with sentiment.

    Parameters
    ----------
    trader_path : str
        Path to the historical_data.csv file.
    sentiment_path : str
        Path to the fear_greed_index.csv file.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with trade data enriched by fear/greed
        sentiment (shifted by 1 day to avoid look-ahead bias).
    """
    trades = _load_historical(trader_path)
    sentiment = _load_sentiment(sentiment_path)

    # Inner join on date — drops trades outside sentiment date range
    merged = trades.merge(sentiment, on="date", how="inner")

    # ── Sentiment regime buckets ────────────────────────────────
    regime_map = {
        "Extreme Fear": "Extreme Fear",
        "Fear": "Fear",
        "Neutral": "Neutral",
        "Greed": "Greed",
        "Extreme Greed": "Extreme Greed",
    }
    merged["sentiment_regime"] = merged["fg_classification"].map(regime_map).fillna("Neutral")

    return merged
