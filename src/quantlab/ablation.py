"""
ablation.py
-----------
The empirical core of the paper: how much does each individual piece of
backtest rigor actually cost you in reported Sharpe? Seven stages, each
one changing exactly one thing relative to the one before it, so the
Sharpe delta between consecutive stages is attributable to that one
change and nothing else. Full stage-by-stage description lives on
run_full_ablation's docstring below, since that's the function that
actually orders and runs them.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from . import signals as S
from . import features as F
from . import sizing as Z
from . import risk as R
from . import execution as E
from . import metrics as M
from . import validation as V
from . import backtest as B


def naive_momentum_meanrev_blend(prices, returns, mom_window=60, mr_window=5,
                                  mom_weight=0.5, zscore_window=252) -> pd.DataFrame:
    """
    The bug this whole module exists to quantify: identical to
    signals.momentum_meanrev_blend except every feature is built with
    lag=0 -- i.e. today's momentum/mean-reversion score is allowed to
    use today's own close. This is what a careless first draft of a
    strategy script actually looks like; nothing here is a strawman.

    Note what this means concretely for mean-reversion: mr_score(t) is
    a function of price(t), and returns(t) = price(t)/price(t-1) - 1 is
    ALSO a function of price(t). Correlating a same-bar feature with the
    same-bar return isn't predicting the future from the past -- it's
    partially reconstructing today's move from today's move. That's the
    leak, in one sentence.
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
    """Sizing/risk stack is IDENTICAL across every stage. Only signal_fn
    (naive vs. structurally-safe features), lag_bars, and cost params
    vary. This is deliberate: the ablation is about backtest
    methodology, not about changing the strategy."""
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


def _walk_forward_stage(prices, returns, volume, param_grid, train_size, test_size,
                         purge, embargo, linear_cost_bps, impact_coef):
    result = B.walk_forward_backtest(
        prices, returns, volume, param_grid,
        train_size=train_size, test_size=test_size, purge=purge, embargo=embargo,
        linear_cost_bps=linear_cost_bps, impact_coef=impact_coef,
    )
    return result


def _summarize(name: str, description: str, returns: pd.Series, n_trials: int = 1,
               trial_sr_std: float | None = None) -> dict:
    r = returns.dropna()
    if len(r) < 30:
        return {"stage": name, "description": description, "n_obs": len(r),
                "sharpe": np.nan, "annualized_return": np.nan, "max_drawdown": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "psr": np.nan, "dsr": np.nan}
    tt = M.significance_ttest(r)
    return {
        "stage": name,
        "description": description,
        "n_obs": len(r),
        "sharpe": M.sharpe_ratio(r),
        "annualized_return": M.annualized_return(r),
        "max_drawdown": M.max_drawdown(r)["max_drawdown"],
        "t_stat": tt["t_stat"],
        "p_value": tt["p_value"],
        "psr": M.probabilistic_sharpe_ratio(r, 0.0),
        "dsr": M.deflated_sharpe_ratio(r, n_trials, trial_sr_std)["deflated_sharpe_ratio"] if n_trials > 1 else np.nan,
    }


def run_full_ablation(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    param_grid: dict,
    train_size: int = 500,
    test_size: int = 126,
    purge_proper: int = 90,
    embargo_proper: int = 20,
    realistic_linear_cost_bps: float = 2.0,
    realistic_impact_coef: float = 0.1,
) -> pd.DataFrame:
    """
    Runs all seven stages and returns a comparison table -- the table
    the paper's results section is built around. Each stage changes
    EXACTLY one thing relative to the one before it -- this is the
    fixed version: the original 6-stage design changed both the feature
    layer and the cost model in a single step (old stage 2->3), which
    meant that transition's Sharpe delta couldn't be attributed to
    either cause alone. A cost-only stage is now inserted between them.

      1  naive (lag=0) features, NO execution lag, no costs, in-sample
         fit. The actual bug: today's feature is allowed to use today's
         own close, and today's fill happens on today's own bar.
      2  SAME naive (lag=0) features, execution-level T+1 lag turned ON,
         still no costs. Isolates the lag fix alone: one shift, applied
         anywhere in the pipeline, should be enough to stop the leak
         even if the feature layer itself was never made safe.
      3  SAME naive features, SAME lag, realistic costs turned ON.
         Isolates the cost effect alone -- nothing else changes here.
      4  Switch to the real, structurally-lagged feature layer
         (signals.momentum_meanrev_blend), same lag setting, same costs.
         Isolates what the feature-layer switch itself does -- note this
         inherently adds a second day of effective lag (features are
         internally lagged 1, then shifted again at execution), so a
         Sharpe change here reflects "using the safe feature layer",
         not costs or the raw lag toggle, both of which are already
         controlled for by stages 2 and 3.
      5  Walk-forward OOS instead of in-sample fit, purge=0, embargo=0.
         Isolates the cost of moving from in-sample to out-of-sample
         evaluation, before boundary leakage is handled.
      6  Add purge + embargo. Isolates whatever boundary contamination
         adjacent folds were allowing through.
      7  Same Stage-6 return series, naive Sharpe/p-value reported next
         to PSR/DSR. No new backtest -- isolates what the significance
         correction alone removes.
    """
    rows = []

    # Stage 1: naive features + no execution lag + no costs, in-sample fit.
    best1, n_trials = _in_sample_stage(prices, returns, volume, param_grid,
                                        naive_momentum_meanrev_blend,
                                        lag_bars=0, linear_cost_bps=0.0, impact_coef=0.0)
    rows.append(_summarize(
        "1_naive_features_no_lag",
        "Unlagged (same-bar) features, same-bar execution, zero cost, in-sample fit",
        best1["returns"], n_trials=1,
    ))

    # Stage 2: SAME naive features, execution lag turned on, still no costs.
    best2, _ = _in_sample_stage(prices, returns, volume, param_grid,
                                 naive_momentum_meanrev_blend,
                                 lag_bars=1, linear_cost_bps=0.0, impact_coef=0.0)
    rows.append(_summarize(
        "2_same_features_plus_lag",
        "Same unlagged features, T+1 execution lag added, zero cost, in-sample fit",
        best2["returns"], n_trials=1,
    ))

    # Stage 3 (NEW): SAME naive features, SAME lag, costs turned on.
    # This isolates the cost effect on its own -- nothing else changes
    # relative to Stage 2. This is the stage that didn't exist before
    # and whose absence was the confound.
    best3, _ = _in_sample_stage(prices, returns, volume, param_grid,
                                 naive_momentum_meanrev_blend,
                                 lag_bars=1, linear_cost_bps=realistic_linear_cost_bps,
                                 impact_coef=realistic_impact_coef)
    rows.append(_summarize(
        "3_same_features_plus_costs",
        f"Same unlagged features, T+1 lag, realistic costs added "
        f"({realistic_linear_cost_bps}bps + impact), in-sample fit",
        best3["returns"], n_trials=1,
    ))

    # Stage 4: switch to the real feature layer, same lag setting, same costs.
    # Isolates the feature-layer switch alone, holding lag-toggle and
    # costs fixed relative to Stage 3.
    best4, _ = _in_sample_stage(prices, returns, volume, param_grid,
                                 S.momentum_meanrev_blend,
                                 lag_bars=1, linear_cost_bps=realistic_linear_cost_bps,
                                 impact_coef=realistic_impact_coef)
    rows.append(_summarize(
        "4_real_features",
        "Structurally-lagged features (real feature layer), T+1 lag, "
        "realistic costs, in-sample fit",
        best4["returns"], n_trials=1,
    ))

    # Stage 5: walk-forward, purge=0, embargo=0.
    wf_no_purge = B.walk_forward_backtest(
        prices, returns, volume, param_grid, train_size=train_size, test_size=test_size,
        purge=0, embargo=0, linear_cost_bps=realistic_linear_cost_bps, impact_coef=realistic_impact_coef,
    )
    rows.append(_summarize(
        "5_walk_forward_no_purge",
        "Walk-forward OOS, T+1 lag, realistic costs, NO purge/embargo",
        wf_no_purge["oos_returns"], n_trials=wf_no_purge["n_trials"],
        trial_sr_std=wf_no_purge["trial_sr_std_daily"],
    ))

    # Stage 6: proper walk-forward with purge + embargo -- the "honest" backtest.
    wf_proper = B.walk_forward_backtest(
        prices, returns, volume, param_grid, train_size=train_size, test_size=test_size,
        purge=purge_proper, embargo=embargo_proper,
        linear_cost_bps=realistic_linear_cost_bps, impact_coef=realistic_impact_coef,
    )
    rows.append(_summarize(
        "6_walk_forward_purged",
        f"Walk-forward OOS, T+1 lag, realistic costs, purge={purge_proper}/embargo={embargo_proper}",
        wf_proper["oos_returns"], n_trials=wf_proper["n_trials"],
        trial_sr_std=wf_proper["trial_sr_std_daily"],
    ))

    # Stage 7: identical Stage-6 returns; isolates the statistical-correction step alone.
    rows.append(_summarize(
        "7_same_returns_dsr_corrected",
        "Identical return series to Stage 6 -- naive Sharpe/p-value vs "
        "PSR/DSR side by side, no new backtest run",
        wf_proper["oos_returns"], n_trials=wf_proper["n_trials"],
        trial_sr_std=wf_proper["trial_sr_std_daily"],
    ))

    table = pd.DataFrame(rows).set_index("stage")
    table["sharpe_inflation_vs_stage6"] = table["sharpe"] / table.loc["6_walk_forward_purged", "sharpe"]
    return table