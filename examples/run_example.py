# examples/run_example.py
import pandas as pd
from quantlab import data as D
from quantlab import signals as S
from quantlab import backtest as B
from quantlab import validation as V
from quantlab import metrics as M

pd.set_option("display.width", 140)

# ---- 1. synthetic market ----
mkt = D.make_synthetic_market(n_assets=10, n_days=1500, seed=11)
prices, volume, factor_returns = mkt["prices"], mkt["volume"], mkt["factor_returns"]
returns = prices.pct_change()

print("=" * 70)
print("STEP 1: LEAK AUDIT ON THE SIGNAL FUNCTION ITSELF")
print("=" * 70)

def pipeline_fn(price_slice):
    ret_slice = price_slice.pct_change()
    return S.momentum_meanrev_blend(price_slice, ret_slice,
                                     mom_window=60, mr_window=5,
                                     mom_weight=0.5, zscore_window=252)

check_dates = list(prices.index[400:1400:150])
audit = V.verify_vectorized_vs_iterative(pipeline_fn, prices, check_dates)
print(audit)
print("\nNo look-ahead bias detected in signal construction: PASS\n")

print("=" * 70)
print("STEP 2: WALK-FORWARD BACKTEST (with purge + embargo)")
print("=" * 70)

param_grid = {
    "mom_window": [40, 60, 90],
    "mr_window": [3, 5, 10],
    "mom_weight": [0.3, 0.5, 0.7],
}

result = B.walk_forward_backtest(
    prices, returns, volume, param_grid,
    train_size=500, test_size=126, purge=90, embargo=20,
    nav=1_000_000.0, linear_cost_bps=2.0, impact_coef=0.1,
    target_annual_vol=0.10,
)

print(result["fold_records"].to_string(index=False))
print(f"\nTotal OOS days stitched: {len(result['oos_returns'])}")
print(f"Grid combinations tried (fed into DSR): {result['n_trials']}")

print("\n" + "=" * 70)
print("STEP 3: OUT-OF-SAMPLE PERFORMANCE REPORT")
print("=" * 70)

report = M.full_report(
    result["oos_returns"], result["oos_turnover"],
    factor_returns=factor_returns.reindex(result["oos_returns"].index),
    n_trials_for_dsr=result["n_trials"],
    trial_sr_std_daily=result["trial_sr_std_daily"],
)

print(f"Annualized return       : {report['annualized_return']:.2%}")
print(f"Annualized vol          : {report['annualized_vol']:.2%}")
print(f"Sharpe                  : {report['sharpe']:.2f}")
print(f"Sortino                 : {report['sortino']:.2f}")
print(f"Max drawdown            : {report['max_drawdown']:.2%}")
print(f"Calmar                  : {report['calmar']:.2f}")
print(f"Avg daily turnover      : {report['avg_daily_turnover']:.2%}")
print(f"t-stat / p-value        : {report['ttest']['t_stat']:.2f} / {report['ttest']['p_value']:.4f}  (n={report['ttest']['n']})")
print(f"PSR (vs SR*=0)          : {report['psr_vs_zero_sharpe']:.2%}")
print(f"Deflated Sharpe Ratio   : {report['dsr']['deflated_sharpe_ratio']:.2%}  (n_trials={report['dsr']['n_trials_assumed']})")
fe = report["factor_exposure"]
print(f"Factor alpha (ann.)     : {fe['alpha_annualized']:.2%}")
print(f"Factor betas            :\n{fe['betas']}")
print(f"Factor R^2              : {fe['r_squared']:.3f}")

print("\n" + "=" * 70)
print("STEP 4: NEGATIVE CONTROL -- deliberately inject a leaky feature")
print("        and show the audit catches it")
print("=" * 70)

def leaky_pipeline_fn(price_slice):
    ret_slice = price_slice.pct_change()
    sig = S.momentum_meanrev_blend(price_slice, ret_slice, 60, 5, 0.5, 252)
    # BUG: uses tomorrow's return, shifted backward -- classic leak
    future_leak = ret_slice.shift(-1)
    return sig + 0.5 * future_leak.fillna(0)

try:
    V.verify_vectorized_vs_iterative(leaky_pipeline_fn, prices, check_dates)
    print("ERROR: audit failed to catch the injected leak!")
except AssertionError as e:
    print("Leak correctly detected:\n")
    print(str(e)[:400])