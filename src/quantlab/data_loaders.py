"""
data_loaders.py
----------------
Real-data equivalents of data.make_synthetic_market. NOT importable
without the optional "data" extra (`pip install -e ".[data]"`) -- these
functions import yfinance/pandas_datareader lazily, inside the function
body, so the rest of the package works fine without those dependencies
installed.


IMPORTANT, repeating the README: neither of these functions does
anything about survivorship bias. If you pass in today's S&P 500
constituent list and backtest back to 2015, you will silently exclude
every company that was delisted, acquired, or went bankrupt along the
way, and your results will be overstated. That is a data-sourcing
problem this module does not solve.
"""

from __future__ import annotations
import pandas as pd


def load_yfinance_prices_volume(tickers: list[str], start: str, end: str | None = None):
    """
    Returns (prices, volume) as wide DataFrames (date x ticker), in the
    exact shape quantlab expects everywhere else in the pipeline.

    prices: split/dividend-adjusted close (auto_adjust=True), so a
            5-for-1 split or a dividend doesn't show up as a fake
            -80% return or a fake dividend-day pop.
    volume: converted to approximate DOLLAR volume (shares * price),
            because costs.market_impact_costs expects dollar volume,
            not share count -- passing raw share volume in there will
            silently give you a wrong (and probably tiny, if you're
            trading a small number of shares against a huge share-count
            ADV) impact estimate.

    If this breaks: yfinance's multi-ticker download has changed its
    column-grouping convention across versions more than once (top
    level = field vs. top level = ticker). If `raw["Close"]` raises a
    KeyError, run `print(raw.columns)` and adjust -- the fix is a
    one-line change to how `prices`/`volume` are sliced out below, not
    a change to anything downstream.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if len(tickers) == 1:
        # yfinance sometimes returns a flat (non-MultiIndex) frame for a
        # single ticker depending on version -- normalize to the same
        # wide shape as the multi-ticker case.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})
    else:
        prices = raw["Close"]
        volume = raw["Volume"]

    volume = volume * prices  # shares -> approximate dollar volume

    # Drop days where EVERY ticker is NaN (exchange holiday), then
    # forward-fill remaining single-ticker gaps (halts, provider gaps).
    # ffill is causal -- it only ever copies a PAST value forward, never
    # a future one -- so this does not reintroduce look-ahead bias.
    prices = prices.dropna(how="all").ffill()
    volume = volume.reindex(prices.index).ffill()

    return prices, volume


def load_fama_french_factors(start: str, end: str | None = None,
                              include_momentum: bool = True) -> pd.DataFrame:
    """
    Pulls the daily Fama-French factors (Mkt-RF, SMB, HML, and
    optionally Momentum) from Ken French's data library via
    pandas_datareader, and converts from percent to decimal returns
    (the raw data is in percent, e.g. 0.53 meaning 0.53%, not 0.53%
    already-decimal -- dividing by 100 is not optional, it's a unit fix).

    Returns a DataFrame with columns like ["Mkt-RF", "SMB", "HML", "Mom"]
    -- pass this straight into metrics.factor_exposure() or
    metrics.full_report(factor_returns=...).

    If this breaks: pandas_datareader's FamaFrench reader occasionally
    lags Ken French's own site through a format change, and it can be
    slow/flaky on first call (it downloads and unzips a file from
    Dartmouth's servers, not a fast JSON API). Retry once before
    assuming it's broken. Dataset names below match the reader's own
    naming ("F-F_Research_Data_Factors_daily", "F-F_Momentum_Factor_daily")
    as of when this was written; if pandas_datareader errors with a
    "dataset not found"-style message, search
    pandas_datareader.famafrench.get_available_datasets() for the
    current exact name.
    """
    import pandas_datareader.data as web

    ff3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start, end=end)[0]
    factors = ff3[["Mkt-RF", "SMB", "HML"]] / 100.0

    if include_momentum:
        mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start, end=end)[0]
        mom_col = mom.columns[0]  # reader sometimes names it "Mom   " with trailing spaces
        factors["Mom"] = mom[mom_col].reindex(factors.index) / 100.0

    # factors.index = pd.to_datetime(factors.index)
    factors.index = factors.index.to_timestamp()  # pandas_datareader returns a PeriodIndex, not a DatetimeIndex
    return factors


def load_real_market(tickers: list[str], start: str, end: str | None = None,
                      include_momentum_factor: bool = True) -> dict:
    """
    Convenience wrapper matching data.make_synthetic_market's return
    shape, so you can swap this in for that call with no other changes
    anywhere in run_example.py / run_ablation.py / your own scripts.

    Returns {"prices": ..., "volume": ..., "factor_returns": ...}.
    Unlike make_synthetic_market, there is no "true_betas" key -- you
    don't get to know the ground truth for real data, which is rather
    the point of testing on it.
    """
    prices, volume = load_yfinance_prices_volume(tickers, start, end)
    factor_returns = load_fama_french_factors(start, end, include_momentum=include_momentum_factor)
    factor_returns = factor_returns.reindex(prices.index).ffill()
    return {"prices": prices, "volume": volume, "factor_returns": factor_returns}