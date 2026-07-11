"""
statistical_tests.py — Statistical validation of PnL differences.

Kruskal-Wallis test across sentiment regimes, pairwise Mann-Whitney U
with Bonferroni correction, and per-cohort cross-regime significance.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu


def _nonzero_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with non-zero Closed PnL."""
    return df[df["Closed PnL"] != 0].copy()


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    if n1 * n2 == 0:
        return 0.0
    return 1 - (2 * u) / (n1 * n2)


def run_kruskal_wallis(df: pd.DataFrame) -> dict:
    """
    Kruskal-Wallis H-test: are PnL distributions different across
    sentiment regimes?

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sentiment_regime, Closed PnL.

    Returns
    -------
    dict
        Keys: H_statistic, p_value, interpretation, n_groups, group_sizes.
    """
    pnl_df = _nonzero_pnl(df)
    if pnl_df.empty:
        return {"H_statistic": np.nan, "p_value": np.nan,
                "interpretation": "No data", "n_groups": 0, "group_sizes": {}}

    groups = []
    group_sizes = {}
    for regime, grp in pnl_df.groupby("sentiment_regime"):
        vals = grp["Closed PnL"].values
        if len(vals) >= 2:  # need at least 2 observations
            groups.append(vals)
            group_sizes[regime] = len(vals)

    if len(groups) < 2:
        return {"H_statistic": np.nan, "p_value": np.nan,
                "interpretation": "Fewer than 2 groups with sufficient data",
                "n_groups": len(groups), "group_sizes": group_sizes}

    h_stat, p_val = kruskal(*groups)

    if p_val < 0.001:
        interp = "Highly significant difference across regimes (p < 0.001)"
    elif p_val < 0.01:
        interp = "Significant difference across regimes (p < 0.01)"
    elif p_val < 0.05:
        interp = "Marginally significant difference across regimes (p < 0.05)"
    else:
        interp = "No significant difference across regimes (p ≥ 0.05)"

    return {
        "H_statistic": round(h_stat, 4),
        "p_value": p_val,
        "interpretation": interp,
        "n_groups": len(groups),
        "group_sizes": group_sizes,
    }


def run_pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests between every pair of sentiment
    regimes with Bonferroni-corrected p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sentiment_regime, Closed PnL.

    Returns
    -------
    pd.DataFrame
        Columns: Regime A, Regime B, U Statistic, p-value (raw),
        p-value (Bonferroni), Effect Size (rank-biserial), Significant.
    """
    pnl_df = _nonzero_pnl(df)
    if pnl_df.empty:
        return pd.DataFrame()

    regime_groups = {
        regime: grp["Closed PnL"].values
        for regime, grp in pnl_df.groupby("sentiment_regime")
        if len(grp) >= 2
    }

    pairs = list(combinations(sorted(regime_groups.keys()), 2))
    n_comparisons = len(pairs) if len(pairs) > 0 else 1

    rows = []
    for a, b in pairs:
        vals_a = regime_groups[a]
        vals_b = regime_groups[b]

        u_stat, p_raw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_bonf = min(p_raw * n_comparisons, 1.0)
        effect = _rank_biserial(u_stat, len(vals_a), len(vals_b))

        rows.append({
            "Regime A": a,
            "Regime B": b,
            "N_A": len(vals_a),
            "N_B": len(vals_b),
            "U Statistic": round(u_stat, 2),
            "p-value (raw)": p_raw,
            "p-value (Bonferroni)": round(p_bonf, 6),
            "Effect Size (rank-biserial)": round(effect, 4),
            "Significant": p_bonf < 0.05,
        })

    return pd.DataFrame(rows)


def run_cohort_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cohort, test whether PnL distributions differ significantly
    across sentiment regimes (Kruskal-Wallis), then do pairwise
    Mann-Whitney for cohorts with significant overall results.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: cohort, sentiment_regime, Closed PnL.

    Returns
    -------
    pd.DataFrame
        Columns: Cohort, Test, Comparison, Statistic, p-value,
        p-value (Bonferroni), Effect Size, Significant, N.
    """
    pnl_df = _nonzero_pnl(df)
    if pnl_df.empty:
        return pd.DataFrame()

    rows = []

    for cohort, cgrp in pnl_df.groupby("cohort"):
        # ── Overall Kruskal-Wallis within this cohort ───────────
        regime_groups = {
            r: g["Closed PnL"].values
            for r, g in cgrp.groupby("sentiment_regime")
            if len(g) >= 2
        }

        if len(regime_groups) < 2:
            rows.append({
                "Cohort": cohort, "Test": "Kruskal-Wallis",
                "Comparison": "All regimes", "Statistic": np.nan,
                "p-value": np.nan, "p-value (Bonferroni)": np.nan,
                "Effect Size": np.nan, "Significant": False,
                "N": len(cgrp),
            })
            continue

        h_stat, p_val = kruskal(*regime_groups.values())
        rows.append({
            "Cohort": cohort, "Test": "Kruskal-Wallis",
            "Comparison": "All regimes", "Statistic": round(h_stat, 4),
            "p-value": p_val, "p-value (Bonferroni)": np.nan,
            "Effect Size": np.nan, "Significant": p_val < 0.05,
            "N": len(cgrp),
        })

        # ── Pairwise only if overall is significant ─────────────
        if p_val < 0.05:
            pairs = list(combinations(sorted(regime_groups.keys()), 2))
            n_comp = len(pairs) if len(pairs) > 0 else 1

            for a, b in pairs:
                va, vb = regime_groups[a], regime_groups[b]
                u, p_raw = mannwhitneyu(va, vb, alternative="two-sided")
                p_bonf = min(p_raw * n_comp, 1.0)
                eff = _rank_biserial(u, len(va), len(vb))

                rows.append({
                    "Cohort": cohort, "Test": "Mann-Whitney U",
                    "Comparison": f"{a} vs {b}",
                    "Statistic": round(u, 2),
                    "p-value": p_raw,
                    "p-value (Bonferroni)": round(p_bonf, 6),
                    "Effect Size": round(eff, 4),
                    "Significant": p_bonf < 0.05,
                    "N": len(va) + len(vb),
                })

    return pd.DataFrame(rows)
