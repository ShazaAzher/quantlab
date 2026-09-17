# risk.py
"""Static, order-independent clips applied after sizing. No lookback,
so nothing here can leak."""
from __future__ import annotations
import pandas as pd


def apply_position_limits(weights: pd.DataFrame, max_position: float = 0.15) -> pd.DataFrame:
    return weights.clip(lower=-max_position, upper=max_position)


def apply_gross_leverage_cap(weights: pd.DataFrame, max_gross: float = 3.0) -> pd.DataFrame:
    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0)
    return weights.mul(scale, axis=0)


def apply_net_exposure_band(weights: pd.DataFrame, max_abs_net: float = 0.3) -> pd.DataFrame:
    net = weights.sum(axis=1)
    excess = (net.abs() - max_abs_net).clip(lower=0) * pd.Series(
        [1 if n >= 0 else -1 for n in net], index=net.index)
    return weights.sub(excess / weights.shape[1], axis=0)


def apply_all(weights: pd.DataFrame, max_position=0.15, max_gross=3.0, max_abs_net=0.3) -> pd.DataFrame:
    w = apply_position_limits(weights, max_position)
    w = apply_gross_leverage_cap(w, max_gross)
    return apply_net_exposure_band(w, max_abs_net)