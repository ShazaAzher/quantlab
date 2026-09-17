# tests/test_costs.py
import pandas as pd
from src import costs as C


def test_zero_turnover_for_constant_weights():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    weights = pd.DataFrame({"A": [0.5] * 10, "B": [-0.5] * 10}, index=idx)
    tno = C.turnover(weights)
    assert (tno.iloc[1:] == 0).all()  # no trades after the first bar


def test_linear_costs_scale_with_bps():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
    cost_2bps = C.linear_costs(weights, cost_bps=2.0)
    cost_4bps = C.linear_costs(weights, cost_bps=4.0)
    assert cost_4bps.iloc[1] == 2 * cost_2bps.iloc[1]


def test_market_impact_grows_with_trade_size(market):
    weights_small = pd.DataFrame(0.0, index=market["prices"].index, columns=market["prices"].columns)
    weights_large = weights_small.copy()
    t = market["prices"].index[50]
    weights_small.loc[t:, market["prices"].columns[0]] = 0.05
    weights_large.loc[t:, market["prices"].columns[0]] = 0.50

    small_cost = C.market_impact_costs(weights_small, 1_000_000.0, market["volume"])
    large_cost = C.market_impact_costs(weights_large, 1_000_000.0, market["volume"])
    assert large_cost.loc[t] > small_cost.loc[t]