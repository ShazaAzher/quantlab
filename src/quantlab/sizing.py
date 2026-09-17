# sizing.py
"""
Position sizing + volatility targeting. The vol estimate used to size
today's trade must never see today's own return -- that's a subtle,
common leak (sizing a trade using the realized move it's about to make).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import features as F


def signal_to_raw_weights(signal: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    abs_sum = signal.abs().sum(axis=1)
    scale = gross_leverage / abs_sum.replace(0, np.nan)
    return signal.mul(scale, axis=0).fillna(0.0)


def volatility_target_scalar(portfolio_gross_returns: pd.Series, target_annual_vol: float,
                              lookback: int = 63, max_leverage: float = 3.0) -> pd.Series:
    """`portfolio_gross_returns` must be the unlevered raw-weight book's
    historical returns. Returns a scalar for day t built from realized
    vol through t-1 only (explicit .shift(1))."""
    realized = portfolio_gross_returns.rolling(lookback).std() * np.sqrt(252)
    scalar = (target_annual_vol / realized.replace(0, np.nan)).clip(upper=max_leverage)
    return scalar.shift(1)


def asset_level_vol_scaling(signal: pd.DataFrame, returns: pd.DataFrame,
                             vol_window: int = 20, target_asset_vol: float = 0.15) -> pd.DataFrame:
    vol = F.realized_vol(returns, vol_window, lag=1, annualize=True)
    scale = (target_asset_vol / vol.replace(0, np.nan)).clip(upper=5.0)
    return signal * scale