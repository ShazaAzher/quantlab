# features.py
"""
Every feature uses pandas .rolling() (already causal) then `.shift(lag)`,
default lag=1. Baking the shift into the feature layer, rather than
trusting every downstream module to remember it, is the single biggest
leakage-prevention design choice in this library.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _lag(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    if lag < 0:
        raise ValueError("negative lag would shift the future backward -- leakage by definition.")
    return df.shift(lag)


def momentum(prices: pd.DataFrame, window: int, lag: int = 1) -> pd.DataFrame:
    return _lag(prices.pct_change(window), lag)


def realized_vol(returns: pd.DataFrame, window: int, lag: int = 1, annualize: bool = True) -> pd.DataFrame:
    raw = returns.rolling(window).std()
    if annualize:
        raw = raw * np.sqrt(252)
    return _lag(raw, lag)


def zscore(feature: pd.DataFrame, window: int, lag: int = 1) -> pd.DataFrame:
    """Pass lag=0 if `feature` was already lagged upstream -- double-lagging
    wastes signal but isn't wrong. Defaults to lag=1 to fail safe."""
    mu = feature.rolling(window).mean()
    sd = feature.rolling(window).std()
    raw = (feature - mu) / sd.replace(0, np.nan)
    return _lag(raw, lag)


def cross_sectional_rank(feature: pd.DataFrame) -> pd.DataFrame:
    """Safe with lag=0 by construction -- ranks across columns at a fixed
    t, never down through time. `feature` itself must already be causal."""
    return feature.rank(axis=1, pct=True) - 0.5


def mean_reversion_score(prices: pd.DataFrame, window: int, lag: int = 1) -> pd.DataFrame:
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    return _lag(-(prices - ma) / sd.replace(0, np.nan), lag)

# No cheap "assert_causal" static checker here on purpose -- a function
# inspecting a feature frame in isolation can't actually prove causality,
# it would just be false confidence. The real test is
# validation.verify_vectorized_vs_iterative, below.
