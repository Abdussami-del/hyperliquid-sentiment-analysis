"""
metrics_engine.py — Quantitative performance metrics.

Computes cohort-level and trader-level metrics including win rate,
profit factor, expectancy, leverage efficiency, and greed-trap detection.
"""

import pandas as pd
import numpy as np


def _safe_profit_factor(wins_sum: float, losses_sum: float) -> float:
    """Profit factor = gross profits / |gross losses|. Safe against div-by-zero."""
    abs_losses = abs(losses_sum)
    if abs_losses < 1e-10:
        return np.inf if wins_sum > 0 else 0.0
    return wins_sum / abs_losses


def calculate_cohort_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute key performance metrics grouped by cohort × sentiment regime.

    Only trades with non-zero Closed PnL are counted for win/loss stats.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: cohort, sentiment_regime, Closed PnL, Size USD.

    Returns
    -------
    pd.DataFrame
        Columns: Cohort, Sentiment Regime, Total Trades, Win Rate (%),
        Profit Factor, Expectancy ($), Avg Trade Size ($), Total PnL ($).
    """
    # Filter to trades with non-zero PnL (position closes)
    pnl_df = df[df["Closed PnL"] != 0].copy()

    if pnl_df.empty:
        return pd.DataFrame(columns=[
            "Cohort", "Sentiment Regime", "Total Trades", "Win Rate (%)",
            "Profit Factor", "Expectancy ($)", "Avg Trade Size ($)", "Total PnL ($)",
        ])

    rows = []
    for (cohort, regime), grp in pnl_df.groupby(["cohort", "sentiment_regime"]):
        wins = grp[grp["Closed PnL"] > 0]
        losses = grp[grp["Closed PnL"] < 0]

        n_total = len(grp)
        n_wins = len(wins)
        n_losses = len(losses)

        win_rate = (n_wins / n_total * 100) if n_total > 0 else 0.0

        gross_profit = wins["Closed PnL"].sum()
        gross_loss = losses["Closed PnL"].sum()  # negative number
        profit_factor = _safe_profit_factor(gross_profit, gross_loss)

        avg_win = wins["Closed PnL"].mean() if n_wins > 0 else 0.0
        avg_loss = abs(losses["Closed PnL"].mean()) if n_losses > 0 else 0.0
        loss_rate = 1.0 - (win_rate / 100)
        expectancy = (win_rate / 100 * avg_win) - (loss_rate * avg_loss)

        rows.append({
            "Cohort": cohort,
            "Sentiment Regime": regime,
            "Total Trades": n_total,
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 4) if np.isfinite(profit_factor) else np.inf,
            "Expectancy ($)": round(expectancy, 2),
            "Avg Trade Size ($)": round(grp["Size USD"].mean(), 2),
            "Total PnL ($)": round(grp["Closed PnL"].sum(), 2),
        })

    return pd.DataFrame(rows)


def calculate_leverage_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show avg PnL across leverage bands × sentiment regime.

    Leverage bands: [0-2), [2-5), [5-10), [10-25), [25+)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: implied_leverage, sentiment_regime, Closed PnL.

    Returns
    -------
    pd.DataFrame
        Pivot-style table: leverage_band × sentiment_regime → avg PnL.
    """
    pnl_df = df[df["Closed PnL"] != 0].copy()
    if pnl_df.empty:
        return pd.DataFrame()

    bins = [0, 2, 5, 10, 25, np.inf]
    labels = ["0-2x", "2-5x", "5-10x", "10-25x", "25x+"]
    pnl_df["leverage_band"] = pd.cut(
        pnl_df["implied_leverage"], bins=bins, labels=labels, right=False
    )

    result = (
        pnl_df
        .groupby(["leverage_band", "sentiment_regime"], observed=False)["Closed PnL"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg PnL ($)", "count": "Trade Count"})
    )
    return result


def detect_greed_trap(
    df: pd.DataFrame,
    pnl_percentile: float = 10,
) -> pd.DataFrame:
    """
    Identify trades during Greed/Extreme Greed that resulted in
    large negative PnL ("greed traps").

    The greed trap captures instances where bullish sentiment
    preceded significant losses — a hallmark of retail traders
    over-leveraging at market tops.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged DataFrame with cohort column.
    pnl_percentile : float
        Bottom-percentile of PnL to flag (default: bottom 10 %).

    Returns
    -------
    pd.DataFrame
        Subset of trades matching greed-trap criteria, sorted by PnL.
    """
    pnl_df = df[df["Closed PnL"] != 0].copy()
    if pnl_df.empty:
        return pd.DataFrame()

    pnl_cutoff = pnl_df["Closed PnL"].quantile(pnl_percentile / 100)

    # Detect losses during greed/extreme greed — the classic retail trap
    mask = (
        pnl_df["sentiment_regime"].isin(["Extreme Greed", "Greed"])
        & (pnl_df["Closed PnL"] <= pnl_cutoff)
    )

    traps = pnl_df.loc[mask].sort_values("Closed PnL")

    output_cols = [
        "Account", "Coin", "direction_clean", "implied_leverage",
        "Size USD", "Closed PnL", "fg_value", "sentiment_regime", "date",
    ]
    # Include cohort if available
    if "cohort" in traps.columns:
        output_cols.append("cohort")

    available_cols = [c for c in output_cols if c in traps.columns]
    return traps[available_cols].copy() if not traps.empty else pd.DataFrame()


def get_trader_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account summary: total PnL, trade count, win rate, cohort,
    favourite coin, average trade size.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged DataFrame with cohort column.

    Returns
    -------
    pd.DataFrame
        One row per Account.
    """
    if df.empty:
        return pd.DataFrame()

    pnl_trades = df[df["Closed PnL"] != 0]

    # Basic aggregations on ALL trades
    summary = df.groupby("Account").agg(
        total_trades=("Closed PnL", "size"),
        total_pnl=("Closed PnL", "sum"),
        avg_trade_size=("Size USD", "mean"),
        cohort=("cohort", "first"),
        favorite_coin=("Coin", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    # Win rate from non-zero PnL trades only
    if not pnl_trades.empty:
        wr = (
            pnl_trades
            .groupby("Account")["Closed PnL"]
            .apply(lambda s: (s > 0).sum() / len(s) * 100 if len(s) > 0 else 0.0)
            .reset_index()
            .rename(columns={"Closed PnL": "win_rate_pct"})
        )
        summary = summary.merge(wr, on="Account", how="left")
    else:
        summary["win_rate_pct"] = 0.0

    summary["win_rate_pct"] = summary["win_rate_pct"].fillna(0.0).round(2)
    summary["total_pnl"] = summary["total_pnl"].round(2)
    summary["avg_trade_size"] = summary["avg_trade_size"].round(2)

    return summary.sort_values("total_pnl", ascending=False).reset_index(drop=True)
