"""
universe_sensitivity_real_data.py
-----------------------------------
Real-data analogue of seed_sensitivity.py: answers whether the ablation
study's headline finding (naive in-sample Sharpe is wildly inflated vs.
a proper walk-forward/purged/DSR-corrected Sharpe) generalizes across
different stock universes and date windows, or is an artifact of the
one 10-large-cap-tech-heavy sample used in run_ablation_real_data.py.

Requires: pip install -e ".[data]"
"""
import pandas as pd
from quantlab import data_loaders as DL
from quantlab import ablation as A

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

param_grid = {"mom_window": [40, 60, 90], "mr_window": [3, 5, 10], "mom_weight": [0.3, 0.5, 0.7]}

UNIVERSES = {
    "big_tech": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM", "JNJ", "PG"],
    "industrials_value": ["CAT", "GE", "HON", "MMM", "UPS", "BA", "LMT", "DE", "EMR", "ETN"],
    "consumer_defensive": ["KO", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "MDLZ", "CLX", "HSY"],
    "financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
}
WINDOWS = {
    "2005_2014": ("2005-01-01", "2014-12-31"),
    "2015_2024": ("2015-01-01", "2024-12-31"),
}

rows = []
for uni_name, tickers in UNIVERSES.items():
    for win_name, (start, end) in WINDOWS.items():
        print(f"\n=== universe={uni_name} window={win_name} ===")
        try:
            market = DL.load_real_market(tickers, start, end, include_momentum_factor=True)
            prices, volume = market["prices"], market["volume"]
            returns = prices.pct_change()
            print(f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} trading days, "
                  f"{prices.index.min().date()} to {prices.index.max().date()}")

            table = A.run_full_ablation(prices, returns, volume, param_grid,
                                         train_size=500, test_size=126,
                                         purge_proper=90, embargo_proper=20)

            rows.append({
                "universe": uni_name,
                "window": win_name,
                "stage1_sharpe": table.loc["1_naive_features_no_lag", "sharpe"],
                "stage6_sharpe": table.loc["6_walk_forward_purged", "sharpe"],
                "stage6_dsr": table.loc["6_walk_forward_purged", "dsr"],
                "sharpe_inflation_vs_stage6": table.loc["1_naive_features_no_lag", "sharpe_inflation_vs_stage6"],
            })
        except Exception as e:
            print(f"FAILED ({uni_name}, {win_name}): {e!r}")
            rows.append({"universe": uni_name, "window": win_name, "error": str(e)})

result = pd.DataFrame(rows).set_index(["universe", "window"])
print("\n\n=== Summary across universes/windows ===")
print(result.to_string())

result.to_csv("universe_sensitivity_real_data_results.csv")
print("\nSaved to universe_sensitivity_real_data_results.csv")

if "stage1_sharpe" in result.columns:
    n_valid = result["stage1_sharpe"].notna().sum()
    n_stage1_positive = (result["stage1_sharpe"] > 0).sum()
    print(f"\nStage 1 (naive) positive in {n_stage1_positive}/{n_valid} runs")
    print(f"Median sharpe_inflation_vs_stage6: {result['sharpe_inflation_vs_stage6'].median():.2f}")
