# tests/test_metrics.py — includes the bug catch above
import numpy as np
import pandas as pd
from src import metrics as M


def test_sharpe_of_constant_returns_is_nan():
    r = pd.Series([0.001] * 100)
    assert np.isnan(M.sharpe_ratio(r))  # zero variance -> undefined, not zero or inf


def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    assert M.max_drawdown(r)["max_drawdown"] <= 0


def test_psr_is_bounded_0_1():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 300))
    assert 0.0 <= M.probabilistic_sharpe_ratio(r) <= 1.0


def test_dsr_decreases_as_n_trials_increases():
    """Searching harder for a winning strategy should never make the
    SAME return series look MORE significant."""
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0008, 0.01, 400))
    dsr_few = M.deflated_sharpe_ratio(r, n_trials=2)["deflated_sharpe_ratio"]
    dsr_many = M.deflated_sharpe_ratio(r, n_trials=200)["deflated_sharpe_ratio"]
    assert dsr_many <= dsr_few


def test_factor_exposure_recovers_known_beta():
    rng = np.random.default_rng(3)
    n = 500
    factor = pd.DataFrame({"MKT": rng.normal(0, 0.01, n)}, index=pd.date_range("2020-01-01", periods=n))
    true_beta = 1.3
    strat_returns = pd.Series(true_beta * factor["MKT"].values + rng.normal(0, 0.002, n), index=factor.index)
    result = M.factor_exposure(strat_returns, factor)
    assert abs(result["betas"]["MKT"] - true_beta) < 0.1