"""
run_ablation_real_data.py
--------------------------
Same as run_ablation.py, but on real prices instead of the synthetic
generator -- this is the table the paper's results section should
actually be built around if the claim is meant to generalize beyond
one synthetic seed. Requires: pip install -e ".[data]"
"""
import pandas as pd
from quantlab import data_loaders as DL
from quantlab import ablation as A

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG"]
START, END = "2015-01-01", "2024-12-31"

print("Downloading real data (this hits the network and can take a minute)...")
market = DL.load_real_market(TICKERS, START, END, include_momentum_factor=True)
prices, volume = market["prices"], market["volume"]
returns = prices.pct_change()

print(f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} trading days, "
      f"{prices.index.min().date()} to {prices.index.max().date()}")

param_grid = {
    "mom_window": [40, 60, 90],
    "mr_window": [3, 5, 10],
    "mom_weight": [0.3, 0.5, 0.7],
}

table = A.run_full_ablation(prices, returns, volume, param_grid,
                             train_size=500, test_size=126,
                             purge_proper=90, embargo_proper=20)

cols = ["description", "n_obs", "sharpe", "annualized_return", "max_drawdown",
        "p_value", "psr", "dsr", "sharpe_inflation_vs_stage6"]
print(table[cols].to_string())

table.to_csv("ablation_real_data_results.csv")
print("\nSaved full table to ablation_real_data_results.csv")
