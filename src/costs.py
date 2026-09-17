# costs.py
"""
Linear cost (commission + half-spread) + square-root market-impact
model (Almgren et al. functional form), both driven only by the trade
itself and TRAILING average volume -- never future prices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def linear_costs(weights: pd.DataFrame, cost_bps: float = 2.0) -> pd.Series:
    return (weights.diff().abs().sum(axis=1) * cost_bps / 1e4).fillna(0.0)


def market_impact_costs(weights: pd.DataFrame, nav: float, volume: pd.DataFrame,
                         impact_coef: float = 0.1, vol_avg_window: int = 20) -> pd.Series:
    """impact_bps = impact_coef * sqrt(trade_notional / trailing_ADV)."""
    trades_notional = weights.diff().abs() * nav
    adv_dollar = volume.rolling(vol_avg_window).mean().shift(1)  # causal
    participation = (trades_notional / adv_dollar.replace(0, np.nan)).clip(lower=0)
    impact_frac = impact_coef * np.sqrt(participation)
    cost_dollars = (impact_frac * trades_notional).sum(axis=1).fillna(0.0)
    return cost_dollars / nav


def total_costs(weights, nav, volume, linear_bps=2.0, impact_coef=0.1) -> pd.Series:
    lin = linear_costs(weights, linear_bps)
    imp = market_impact_costs(weights, nav, volume, impact_coef)
    return (lin + imp).reindex(weights.index).fillna(0.0)