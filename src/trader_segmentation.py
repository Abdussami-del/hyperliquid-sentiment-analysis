"""
trader_segmentation.py — Trader archetype classification.

Assigns each account to a behavioural cohort based on leverage,
directional bias, and sentiment-aligned trading patterns.

Cohort Definitions:
- Degen: High average implied leverage (>= 5x)
- Sentiment Follower: Traders who trade disproportionately during Greed periods
  AND with directional alignment (long-biased during greed)
- Contrarian: Traders who trade counter to sentiment (shorting during greed,
  or going long during fear)
- Standard Retail: Default classification
"""

import pandas as pd
import numpy as np


def _compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Account, compute the features used for segmentation.

    Returns a DataFrame indexed by Account with columns:
        avg_leverage, total_trades, long_ratio,
        pct_trades_in_greed, pct_trades_in_fear,
        long_ratio_during_greed, short_ratio_during_fear,
        pnl_during_greed, pnl_during_fear, total_pnl
    """
    features = []

    for account, grp in df.groupby("Account"):
        total_trades = len(grp)
        avg_leverage = grp["implied_leverage"].mean()

        # ── Directional trade masks ─────────────────────────────
        is_long_open = grp["direction_clean"].isin(["Open Long"])
        is_short_open = grp["direction_clean"] == "Open Short"
        is_spot_buy = (grp["direction_clean"] == "Spot") & (grp["Side"] == "BUY")

        is_long = is_long_open | is_spot_buy
        is_short = is_short_open

        directional = is_long | is_short
        n_directional = directional.sum()
        long_ratio = is_long.sum() / n_directional if n_directional > 0 else 0.5

        # ── Sentiment regime masks ──────────────────────────────
        greed_mask = grp["fg_classification"].isin(["Greed", "Extreme Greed"])
        fear_mask = grp["fg_classification"].isin(["Fear", "Extreme Fear"])

        n_greed_trades = greed_mask.sum()
        n_fear_trades = fear_mask.sum()

        pct_trades_in_greed = n_greed_trades / total_trades if total_trades > 0 else 0
        pct_trades_in_fear = n_fear_trades / total_trades if total_trades > 0 else 0

        # ── Directional ratios within sentiment regimes ─────────
        # Of directional trades during greed, what fraction are longs?
        directional_in_greed = (directional & greed_mask).sum()
        long_ratio_during_greed = (
            (is_long & greed_mask).sum() / directional_in_greed
            if directional_in_greed > 0 else 0.5
        )

        # Of directional trades during fear, what fraction are shorts?
        directional_in_fear = (directional & fear_mask).sum()
        short_ratio_during_fear = (
            (is_short & fear_mask).sum() / directional_in_fear
            if directional_in_fear > 0 else 0.5
        )

        # What fraction of all longs happen during greed?
        n_longs_total = is_long.sum()
        greed_long_ratio = (
            (is_long & greed_mask).sum() / n_longs_total
            if n_longs_total > 0 else 0.0
        )

        # PnL during different regimes (for classification quality)
        pnl_greed = grp.loc[greed_mask, "Closed PnL"].sum()
        pnl_fear = grp.loc[fear_mask, "Closed PnL"].sum()
        total_pnl = grp["Closed PnL"].sum()

        features.append({
            "Account": account,
            "avg_leverage": avg_leverage,
            "total_trades": total_trades,
            "long_ratio": long_ratio,
            "pct_trades_in_greed": pct_trades_in_greed,
            "pct_trades_in_fear": pct_trades_in_fear,
            "long_ratio_during_greed": long_ratio_during_greed,
            "short_ratio_during_fear": short_ratio_during_fear,
            "greed_long_ratio": greed_long_ratio,
            "pnl_during_greed": pnl_greed,
            "pnl_during_fear": pnl_fear,
            "total_pnl": total_pnl,
        })

    return pd.DataFrame(features)


def _classify(row: pd.Series) -> str:
    """Apply priority-ordered classification rules to one account.
    
    Classification Logic (applied in priority order):
    1. Degen: avg leverage >= 5x regardless of sentiment behavior
    2. Sentiment Follower: Goes long during greed at above-median rate
       AND has above-average greed trade participation
    3. Contrarian: Goes short during fear at below-median rate OR
       has negative PnL during greed (fighting the trend)
    4. Standard Retail: Everyone else
    """
    # 1. Degen — high leverage regardless of sentiment
    if row["avg_leverage"] >= 5:
        return "Degen"

    # 2. Sentiment Follower — trades aligned with herd sentiment
    # Long-heavy during greed AND actively trading during greed periods
    if (row["long_ratio_during_greed"] >= 0.65 and
            row["pct_trades_in_greed"] >= 0.03):
        return "Sentiment Follower"

    # 3. Contrarian — trades against sentiment
    # Either: heavily short during greed, or long during fear,
    # or loses money during greed (fighting the prevailing trend)
    if (row["long_ratio_during_greed"] <= 0.35 and
            row["pct_trades_in_greed"] >= 0.01):
        return "Contrarian"

    if (row["short_ratio_during_fear"] >= 0.6 and
            row["pct_trades_in_fear"] >= 0.01):
        return "Contrarian"

    # 4. Standard Retail
    return "Standard Retail"


def segment_traders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each Account into a behavioural cohort and merge
    the label back into the main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Merged trade + sentiment DataFrame (output of
        ``data_processor.load_and_process_data``).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added ``cohort`` column.
    """
    if df.empty:
        df["cohort"] = pd.Series(dtype="str")
        return df

    acct_features = _compute_account_features(df)
    acct_features["cohort"] = acct_features.apply(_classify, axis=1)

    # Merge cohort label back into every trade row
    df = df.merge(
        acct_features[["Account", "cohort"]],
        on="Account",
        how="left",
    )
    df["cohort"] = df["cohort"].fillna("Standard Retail")

    return df
