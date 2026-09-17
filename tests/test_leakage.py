# tests/test_leakage.py — the one to read first
"""
The most important test file in the repo. A leakage-prevention library
that isn't tested against an actual leak is just an assertion of good
intentions -- these two tests are the proof.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from quantlab import signals as S
from quantlab import validation as V


def _clean_pipeline(prices):
    returns = prices.pct_change()
    return S.momentum_meanrev_blend(prices, returns, mom_window=30, mr_window=5,
                                     mom_weight=0.5, zscore_window=100)


def _leaky_pipeline(prices):
    """Deliberately broken: blends in tomorrow's return."""
    returns = prices.pct_change()
    sig = S.momentum_meanrev_blend(prices, returns, 30, 5, 0.5, 100)
    future_leak = returns.shift(-1).fillna(0)
    return sig + 0.5 * future_leak


def test_clean_pipeline_passes_leak_audit(market):
    prices = market["prices"]
    check_dates = list(prices.index[200:550:70])
    result = V.verify_vectorized_vs_iterative(_clean_pipeline, prices, check_dates)
    assert result["passed"].all()


def test_leaky_pipeline_is_caught(market):
    prices = market["prices"]
    check_dates = list(prices.index[200:550:70])
    with pytest.raises(AssertionError, match="LOOK-AHEAD BIAS DETECTED"):
        V.verify_vectorized_vs_iterative(_leaky_pipeline, prices, check_dates)