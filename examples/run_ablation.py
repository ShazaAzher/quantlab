# examples/run_ablation.py
import pandas as pd
from quantlab import data as D
from quantlab import ablation as A

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

mkt = D.make_synthetic_market(n_assets=10, n_days=1500, seed=11)
prices, volume = mkt["prices"], mkt["volume"]
returns = prices.pct_change()

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