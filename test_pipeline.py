"""Quick verification of the full data pipeline."""
import pandas as pd
import numpy as np
import sys
import os

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

print("=" * 60)
print("STEP 1: DATA PROCESSING")
print("=" * 60)
merged = load_and_process_data()
print(f"Merged rows: {len(merged):,}")
print(f"Columns: {len(merged.columns)}")
print(f"Accounts: {merged['Account'].nunique()}")
print(f"Coins: {merged['Coin'].nunique()}")
print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
print(f"\nSentiment regime distribution:")
print(merged['sentiment_regime'].value_counts().to_string())
print(f"\nImplied leverage stats:")
print(merged['implied_leverage'].describe().to_string())
print(f"\nDirection distribution:")
print(merged['direction_clean'].value_counts().to_string())

print("\n" + "=" * 60)
print("STEP 2: TRADER SEGMENTATION")
print("=" * 60)
segmented = segment_traders(merged)
print(f"\nCohort distribution:")
print(segmented.groupby('cohort')['Account'].nunique().to_string())
print(f"\nCohort trade counts:")
print(segmented['cohort'].value_counts().to_string())

print("\n" + "=" * 60)
print("STEP 3: METRICS")
print("=" * 60)
metrics = calculate_cohort_metrics(segmented)
print(f"\nMetrics table ({len(metrics)} rows):")
print(metrics.to_string())

lev_eff = calculate_leverage_efficiency(segmented)
print(f"\nLeverage efficiency ({len(lev_eff)} rows):")
if len(lev_eff) > 0:
    print(lev_eff.head(10).to_string())

greed_trap = detect_greed_trap(segmented)
print(f"\nGreed trap events: {len(greed_trap)}")
if len(greed_trap) > 0:
    print(greed_trap.head(5).to_string())

trader_summary = get_trader_summary(segmented)
print(f"\nTrader summary ({len(trader_summary)} traders):")
print(trader_summary.head(5).to_string())

print("\n" + "=" * 60)
print("STEP 4: STATISTICAL TESTS")
print("=" * 60)
kw = run_kruskal_wallis(segmented)
print(f"\nKruskal-Wallis: H={kw['H_statistic']}, p={kw['p_value']}")
print(f"Interpretation: {kw['interpretation']}")
print(f"Group sizes: {kw['group_sizes']}")

pairwise = run_pairwise_mannwhitney(segmented)
print(f"\nPairwise tests ({len(pairwise)} comparisons):")
if len(pairwise) > 0:
    print(pairwise.to_string())

cohort_sig = run_cohort_significance(segmented)
print(f"\nCohort significance ({len(cohort_sig)} rows):")
if len(cohort_sig) > 0:
    print(cohort_sig.to_string())

print("\n" + "=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY ✅")
print("=" * 60)
