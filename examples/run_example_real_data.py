"""
run_example_real_data.py
-------------------------
Same as run_example.py, but on real prices instead of the synthetic
generator. Requires: pip install -e ".[data]"

NOT verified against live data in the environment this was written in
(no network access there) -- if the yfinance or pandas_datareader calls
error, see the troubleshooting notes in quantlab/data_loaders.py's
docstrings before assuming the rest of the pipeline is at fault.
"""
import pandas as pd
from quantlab import data_loaders as DL
from quantlab import signals as S
from quantlab import backtest as B
from quantlab import validation as V
from quantlab import metrics as M

pd.set_option("display.width", 140)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG"]
START, END = "2015-01-01", "2024-12-31"

print("Downloading real data (this hits the network and can take a minute)...")
market = DL.load_real_market(TICKERS, START, END, include_momentum_factor=True)
prices, volume, factor_returns = market["prices"], market["volume"], market["factor_returns"]
returns = prices.pct_change()

print(f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} trading days, "
      f"{prices.index.min().date()} to {prices.index.max().date()}")

# ---- leak audit, same as the synthetic example ----
def pipeline_fn(price_slice):
    ret_slice = price_slice.pct_change()
    return S.momentum_meanrev_blend(price_slice, ret_slice, 60, 5, 0.5, 252)

check_dates = list(prices.index[400: len(prices.index) - 100: 150])
audit = V.verify_vectorized_vs_iterative(pipeline_fn, prices, check_dates)
print("\nLeak audit:\n", audit)

# ---- walk-forward backtest ----
param_grid = {"mom_window": [40, 60, 90], "mr_window": [3, 5, 10], "mom_weight": [0.3, 0.5, 0.7]}
result = B.walk_forward_backtest(prices, returns, volume, param_grid,
                                  train_size=500, test_size=126, purge=90, embargo=20)

print(f"\nTotal OOS days: {len(result['oos_returns'])}")

report = M.full_report(
    result["oos_returns"], result["oos_turnover"],
    factor_returns=factor_returns.reindex(result["oos_returns"].index),
    n_trials_for_dsr=result["n_trials"],
    trial_sr_std_daily=result["trial_sr_std_daily"],
)

print(f"Sharpe: {report['sharpe']:.2f}  |  DSR: {report['dsr']['deflated_sharpe_ratio']:.2%}  "
      f"|  Max DD: {report['max_drawdown']:.2%}")
print(f"Factor betas:\n{report['factor_exposure']['betas']}")