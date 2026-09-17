# tests/test_execution.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from quantlab import execution as E


def test_held_weights_are_shifted_by_exactly_one_bar():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    held = E.held_weights_from_target(target)
    assert held.iloc[0]["A"] == 0.0          # no position before any decision existed
    assert held.iloc[1]["A"] == target.iloc[0]["A"]
    assert held.iloc[-1]["A"] == target.iloc[-2]["A"]


def test_signal_on_day_t_cannot_earn_pnl_on_day_t(market):
    """The core anti-leakage invariant of the execution layer: a target
    weight decided using day-t information must not appear in the
    return realized on day t."""
    prices = market["prices"]
    returns = market["returns"]
    volume = market["volume"]

    # Weight that goes to 1.0 on a single day and 0 elsewhere.
    target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pivot_date = prices.index[300]
    target.loc[pivot_date, prices.columns[0]] = 1.0

    sim = E.simulate(target, returns, volume, nav=1_000_000.0)
    held = sim["held_weights"]

    assert held.loc[pivot_date, prices.columns[0]] == 0.0
    next_date = prices.index[301]
    assert held.loc[next_date, prices.columns[0]] == 1.0