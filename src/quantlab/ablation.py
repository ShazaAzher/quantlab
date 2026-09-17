# ablation.py
"""
The empirical core of the paper: how much does each individual piece of
backtest rigor actually cost you in reported Sharpe? Six stages, each
one changing exactly one thing relative to the one before it, so the
Sharpe delta between consecutive stages is attributable to that one
change and nothing else.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from .. import signals as S
from .. import features as F
from .. import sizing as Z
from .. import risk as R
from .. import execution as E
from .. import metrics as M
from .. import validation as V
from .. import backtest as B


def naive_momentum_meanrev_blend(prices, returns, mom_window=60, mr_window=5,
                                  mom_weight=0.5, zscore_window=252) -> pd.DataFrame:
    """
    The bug this whole module exists to quantify: identical to
    signals.momentum_meanrev_blend except every feature is built with
    lag=0 -- today's momentum/mean-reversion score is allowed to use
    today's own close. This is what a careless first draft of a
    strategy script actually looks like; nothing here is a strawman.

    Concretely for mean-reversion: mr_score(t) is a function of
    price(t), and returns(t) = price(t)/price(t-1) - 1 is ALSO a
    function of price(t). Correlating a same-bar feature with the
    same-bar return isn't predicting the future from the past -- it's
    partially reconstructing today's move from today's move.
    """
    mom = F.momentum(prices, mom_window, lag=0)
    mom_z = F.zscore(mom, zscore_window, lag=0)
    mr = F.mean_reversion_score(prices, mr_window, lag=0)
    mr_z = F.zscore(mr, zscore_window, lag=0)
    blended = mom_weight * mom_z + (1 - mom_weight) * mr_z
    return F.cross_sectional_rank(blended)


def _build_target_weights(prices, returns, params, signal_fn, gross_leverage=1.0,
                           target_annual_vol=0.10, vol_lookback=63):
    """Mirrors backtest.build_target_weights, but takes the signal
    function as an argument so the ablation can swap in the naive
    (unlagged) signal for the early stages and the real, structurally
    safe one for the later stages -- everything downstream of the
    signal (sizing, vol targeting, risk caps) is identical either way."""
    raw_signal = signal_fn(
        prices, returns, mom_window=params["mom_window"], mr_window=params["mr_window"],
        mom_weight=params["mom_weight"], zscore_window=params.get("zscore_window", 252))
    raw_weights = Z.signal_to_raw_weights(raw_signal, gross_leverage=gross_leverage)
    unlevered_held = E.held_weights_from_target(raw_weights)
    unlevered_gross_rets = (unlevered_held * returns).sum(axis=1)
    vol_scalar = Z.volatility_target_scalar(unlevered_gross_rets, target_annual_vol, lookback=vol_lookback)
    scaled_weights = raw_weights.mul(vol_scalar, axis=0).fillna(0.0)
    return R.apply_all(scaled_weights)


def _build_and_run(prices, returns, volume, params, signal_fn, lag_bars, linear_cost_bps,
                    impact_coef, nav=1_000_000.0, target_annual_vol=0.10):
    weights = _build_target_weights(prices, returns, params, signal_fn, target_annual_vol=target_annual_vol)
    sim = E.simulate(weights, returns, volume, nav=nav, linear_cost_bps=linear_cost_bps,
                      impact_coef=impact_coef, lag_bars=lag_bars)
    return weights, sim


def _grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def _in_sample_stage(prices, returns, volume, param_grid, signal_fn, lag_bars,
                      linear_cost_bps, impact_coef):
    """Stages 1-3: fit AND evaluate on the full history -- the classic
    'grid-searched, looked at the whole equity curve, shipped the best
    one' workflow."""
    combos = _grid(param_grid)
    best = None
    for params in combos:
        _, sim = _build_and_run(prices, returns, volume, params, signal_fn, lag_bars,
                                 linear_cost_bps, impact_coef)
        rets = sim["net_returns"].dropna()
        if len(rets) < 30:
            continue
        score = M.sharpe_ratio(rets)
        if best is None or (not np.isnan(score) and score > best["score"]):
            best = {"score": score, "params": params, "sim": sim, "returns": rets}
    return best, len(combos)


def _summarize(name, description, returns, n_trials=1, trial_sr_std=None) -> dict:
    r = returns.dropna()
    if len(r) < 30:
        return {"stage": name, "description": description, "n_obs": len(r),
                "sharpe": np.nan, "annualized_return": np.nan, "max_drawdown": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "psr": np.nan, "dsr": np.nan}
    tt = M.significance_ttest(r)
    return {
        "stage": name, "description": description, "n_obs": len(r),
        "sharpe": M.sharpe_ratio(r),
        "annualized_return": M.annualized_return(r),
        "max_drawdown": M.max_drawdown(r)["max_drawdown"],
        "t_stat": tt["t_stat"], "p_value": tt["p_value"],
        "psr": M.probabilistic_sharpe_ratio(r, 0.0),
        "dsr": M.deflated_sharpe_ratio(r, n_trials, trial_sr_std)["deflated_sharpe_ratio"] if n_trials > 1 else np.nan,
    }


def run_full_ablation(prices, returns, volume, param_grid, train_size=500, test_size=126,
                       purge_proper=90, embargo_proper=20,
                       realistic_linear_cost_bps=2.0, realistic_impact_coef=0.1) -> pd.DataFrame:
    """
    1  naive (lag=0) features, NO execution lag, no costs, in-sample fit.
    2  SAME naive features, execution-level T+1 lag turned ON. Isolates
       the lag fix alone: one shift, anywhere, should be enough.
    3  Switch to real structurally-lagged features + realistic costs,
       still in-sample. Isolates what's left once the temporal leak is gone.
    4  Walk-forward OOS, purge=0/embargo=0. Isolates in-sample -> OOS.
    5  Add purge/embargo. Isolates boundary-leak removal.
    6  Same Stage-5 returns, naive stats vs PSR/DSR. Isolates the
       statistical correction alone -- no new backtest.
    """
    rows = []

    best1, n_trials = _in_sample_stage(prices, returns, volume, param_grid,
                                        naive_momentum_meanrev_blend,
                                        lag_bars=0, linear_cost_bps=0.0, impact_coef=0.0)
    rows.append(_summarize("1_naive_features_no_lag",
        "Unlagged (same-bar) features, same-bar execution, zero cost, in-sample fit",
        best1["returns"], n_trials=1))

    best2, _ = _in_sample_stage(prices, returns, volume, param_grid,
                                 naive_momentum_meanrev_blend,
                                 lag_bars=1, linear_cost_bps=0.0, impact_coef=0.0)
    rows.append(_summarize("2_same_features_plus_lag",
        "Same unlagged features, T+1 execution lag added, zero cost, in-sample fit",
        best2["returns"], n_trials=1))

    best3, _ = _in_sample_stage(prices, returns, volume, param_grid,
                                 S.momentum_meanrev_blend,
                                 lag_bars=1, linear_cost_bps=realistic_linear_cost_bps,
                                 impact_coef=realistic_impact_coef)
    rows.append(_summarize("3_real_features_plus_costs",
        f"Structurally-lagged features, T+1 lag, realistic costs "
        f"({realistic_linear_cost_bps}bps + impact), in-sample fit",
        best3["returns"], n_trials=1))

    wf_no_purge = B.walk_forward_backtest(
        prices, returns, volume, param_grid, train_size=train_size, test_size=test_size,
        purge=0, embargo=0, linear_cost_bps=realistic_linear_cost_bps, impact_coef=realistic_impact_coef)
    rows.append(_summarize("4_walk_forward_no_purge",
        "Walk-forward OOS, T+1 lag, realistic costs, NO purge/embargo",
        wf_no_purge["oos_returns"], n_trials=wf_no_purge["n_trials"],
        trial_sr_std=wf_no_purge["trial_sr_std_daily"]))

    wf_proper = B.walk_forward_backtest(
        prices, returns, volume, param_grid, train_size=train_size, test_size=test_size,
        purge=purge_proper, embargo=embargo_proper,
        linear_cost_bps=realistic_linear_cost_bps, impact_coef=realistic_impact_coef)
    rows.append(_summarize("5_walk_forward_purged",
        f"Walk-forward OOS, T+1 lag, realistic costs, purge={purge_proper}/embargo={embargo_proper}",
        wf_proper["oos_returns"], n_trials=wf_proper["n_trials"],
        trial_sr_std=wf_proper["trial_sr_std_daily"]))

    rows.append(_summarize("6_same_returns_dsr_corrected",
        "Identical return series to Stage 5 -- naive Sharpe/p-value vs PSR/DSR side by side",
        wf_proper["oos_returns"], n_trials=wf_proper["n_trials"],
        trial_sr_std=wf_proper["trial_sr_std_daily"]))

    table = pd.DataFrame(rows).set_index("stage")
    table["sharpe_inflation_vs_stage5"] = table["sharpe"] / table.loc["5_walk_forward_purged", "sharpe"]
    return table