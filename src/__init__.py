"""
src — Crypto Sentiment Alpha analytics package.

Convenience imports so downstream code can do:
    from src import load_and_process_data, segment_traders, ...
"""

from .data_processor import load_and_process_data
from .trader_segmentation import segment_traders
from .metrics_engine import (
    calculate_cohort_metrics,
    calculate_leverage_efficiency,
    detect_greed_trap,
    get_trader_summary,
)
from .statistical_tests import (
    run_kruskal_wallis,
    run_pairwise_mannwhitney,
    run_cohort_significance,
)

__all__ = [
    "load_and_process_data",
    "segment_traders",
    "calculate_cohort_metrics",
    "calculate_leverage_efficiency",
    "detect_greed_trap",
    "get_trader_summary",
    "run_kruskal_wallis",
    "run_pairwise_mannwhitney",
    "run_cohort_significance",
]
