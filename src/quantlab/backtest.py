# backtest.py
"""
For each fold: try every param combo (built on FULL history -- safe,
since every feature is causal so signal(t) can't depend on t'>t no
matter how much history it's handed), score each on the fold's TRAIN
window only, keep the winner, record its TEST-window performance.
Stitch all folds' OOS returns -> honest walk-forward result. The number
of grid combos tried is fed straight into DSR so the final significance
number is penalized for how much you searched to find it.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from ... import signals as S, sizing as Z, risk as R, execution as E, metrics as M, validation as V


def build_target_weights(prices, returns, params, gross_leverage=1.0,
                          target_annual_vol=0.10, vol_lookback=63):
    raw_signal = S.momentum_meanrev_blend(
        prices, returns, mom_window=params["mom_window"], mr_window=params["mr_window"],
        mom_weight=params["mom_weight"], zscore_window=params.get("zscore_window", 252))
    raw_weights = Z.signal_to_raw_weights(raw_signal, gross_leverage=gross_leverage)

    unlevered_held = E.held_weights_from_target(raw_weights)
    unlevered_gross_rets = (unlevered_held * returns).sum(axis=1)
    vol_scalar = Z.volatility_target_scalar(unlevered_gross_rets, target_annual_vol, lookback=vol_lookback)
    scaled_weights = raw_weights.mul(vol_scalar, axis=0).fillna(0.0)
    return R.apply_all(scaled_weights)


def run_single_config(prices, returns, volume, params, nav=1_000_000.0,
                       linear_cost_bps=2.0, impact_coef=0.1, **kw):
    weights = build_target_weights(prices, returns, params, **kw)
    sim = E.simulate(weights, returns, volume, nav=nav, linear_cost_bps=linear_cost_bps, impact_coef=impact_coef)
    return weights, sim


def walk_forward_backtest(prices, returns, volume, param_grid, train_size=500, test_size=126,
                           purge=60, embargo=20, nav=1_000_000.0, linear_cost_bps=2.0,
                           impact_coef=0.1, target_annual_vol=0.10) -> dict:
    folds = V.walk_forward_folds(prices.index, train_size, test_size, purge, embargo)
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]

    fold_records, oos_returns_parts, oos_turnover_parts, all_trial_scores_daily = [], [], [], []

    for fi, fold in enumerate(folds):
        best = None
        for params in combos:
            weights, sim = run_single_config(prices, returns, volume, params, nav=nav,
                linear_cost_bps=linear_cost_bps, impact_coef=impact_coef, target_annual_vol=target_annual_vol)
            train_rets = sim["net_returns"].reindex(fold.train_idx).dropna()
            if len(train_rets) < 30:
                continue
            score = M.sharpe_ratio(train_rets)
            if not np.isnan(score):
                all_trial_scores_daily.append(score / np.sqrt(252))
            if best is None or (not np.isnan(score) and score > best["score"]):
                best = {"score": score, "params": params, "weights": weights, "sim": sim}

        if best is None:
            continue
        test_rets = best["sim"]["net_returns"].reindex(fold.test_idx).dropna()
        test_tno = best["sim"]["turnover"].reindex(fold.test_idx).dropna()
        oos_returns_parts.append(test_rets)
        oos_turnover_parts.append(test_tno)
        fold_records.append({"fold": fi, "train_start": fold.train_idx[0], "train_end": fold.train_idx[-1],
            "test_start": fold.test_idx[0], "test_end": fold.test_idx[-1], "chosen_params": best["params"],
            "train_sharpe": best["score"], "test_sharpe": M.sharpe_ratio(test_rets) if len(test_rets) > 5 else np.nan})

    oos_returns = pd.concat(oos_returns_parts).sort_index()
    oos_turnover = pd.concat(oos_turnover_parts).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_turnover = oos_turnover[~oos_turnover.index.duplicated(keep="first")]

    return {"fold_records": pd.DataFrame(fold_records), "oos_returns": oos_returns,
            "oos_turnover": oos_turnover, "n_trials": len(combos),
            "trial_sr_std_daily": float(np.std(all_trial_scores_daily, ddof=1)) if len(all_trial_scores_daily) > 1 else 1.0}