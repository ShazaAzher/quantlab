"""
seed_sensitivity.py
--------------------
Answers the open question from the ablation study: does Stage 1's
sign-flipped (negative) Sharpe generalize across random draws of the
synthetic market, or is it an artifact of one specific seed's
autocorrelation structure? Run this BEFORE writing that result into
the paper as a general claim.
"""
import pandas as pd
from quantlab import data as D
from quantlab import ablation as A

param_grid = {"mom_window": [40, 60, 90], "mr_window": [3, 5, 10], "mom_weight": [0.3, 0.5, 0.7]}
seeds = [1, 2, 3, 7, 11, 42, 99, 123]

rows = []
for seed in seeds:
    mkt = D.make_synthetic_market(n_assets=10, n_days=1500, seed=seed)
    prices, volume = mkt["prices"], mkt["volume"]
    returns = prices.pct_change()
    table = A.run_full_ablation(prices, returns, volume, param_grid,
                                 train_size=500, test_size=126,
                                 purge_proper=90, embargo_proper=20)
    rows.append({
        "seed": seed,
        "stage1_sharpe": table.loc["1_naive_features_no_lag", "sharpe"],
        "stage2_sharpe": table.loc["2_same_features_plus_lag", "sharpe"],
        "stage6_sharpe": table.loc["6_walk_forward_purged", "sharpe"],
    })

result = pd.DataFrame(rows).set_index("seed")
print(result)
print(f"\nStage 1 negative in {(result['stage1_sharpe'] < 0).sum()}/{len(seeds)} seeds")