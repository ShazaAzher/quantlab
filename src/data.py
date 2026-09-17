# data.py
"""
Point-in-time data container. The single most important object in the
whole library from a leakage-prevention standpoint: every other module
is only allowed to see data through `.asof(t)`, never the raw frame.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class PointInTimeData:
    def __init__(self, df: pd.DataFrame, name: str = "data"):
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{name}: index must be sorted ascending")
        if df.index.has_duplicates:
            raise ValueError(f"{name}: duplicate timestamps in index")
        self._df = df.copy()
        self.name = name

    def asof(self, t) -> pd.DataFrame:
        """Everything with index <= t. Inclusive of t's own bar."""
        return self._df.loc[:t]

    def asof_strict(self, t) -> pd.DataFrame:
        """Everything with index < t. For intraday decisions where
        today's own close isn't visible yet."""
        return self._df.loc[self._df.index < t]

    def full_unsafe(self) -> pd.DataFrame:
        """Escape hatch, named `_unsafe` so it screams at code review."""
        return self._df


def make_synthetic_market(n_assets=8, n_days=2000, seed=7, start="2015-01-02") -> dict:
    """Synthetic multi-asset OHLCV panel: common market factor + fat-tailed
    idiosyncratic noise + mild vol clustering + a small, realistic
    autocorrelation edge. For pipeline testing only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"AST{i:02d}" for i in range(n_assets)]

    market = np.zeros(n_days)
    vol = np.full(n_assets, 0.015)
    asset_rets = np.zeros((n_days, n_assets))
    betas = rng.uniform(0.5, 1.5, n_assets)
    ac = rng.uniform(-0.03, 0.03, n_assets)

    prev = np.zeros(n_assets)
    for t in range(n_days):
        market[t] = rng.standard_t(6) * 0.008
        vol = 0.94 * vol + 0.06 * np.abs(prev) + 0.10 * 0.015
        idio = rng.standard_t(6, n_assets) * vol
        asset_rets[t] = betas * market[t] + idio + ac * prev
        prev = asset_rets[t]

    prices = 100 * np.exp(np.cumsum(asset_rets, axis=0))
    prices = pd.DataFrame(prices, index=dates, columns=tickers)

    base_vol = rng.uniform(5e6, 5e7, n_assets)
    volume = pd.DataFrame(
        base_vol[None, :] * (1 + rng.standard_normal((n_days, n_assets)) * 0.2).clip(0.3),
        index=dates, columns=tickers,
    ).clip(lower=1e5)

    factor_returns = pd.DataFrame({"MKT": market}, index=dates)
    return {"prices": prices, "volume": volume, "factor_returns": factor_returns,
            "true_betas": pd.Series(betas, index=tickers)}