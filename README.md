# README.md

# QuantLab

A walk-forward backtesting framework for systematic multi-asset strategies,
built around one non-negotiable constraint: **every stage of the pipeline is
causal by construction, and that claim is checked by code, not asserted in
prose.** It also includes a seven-stage ablation study that measures, on the
same data and the same strategy, how much each individual piece of backtest
rigor actually changes the reported result.


This is a research tool for evaluating rule-based strategies, not a live
trading system. It does not place orders, model the order book, or handle
intraday microstructure.

## Why this exists

Most homegrown backtesters get a good Sharpe ratio by accident, via one of a
small number of well-known mistakes: a rolling feature that isn't shifted, a
trade that gets to earn P&L on the same bar it was decided, position sizing
that reads the volatility of the move it's about to make, or a hyperparameter
grid search with no penalty for how many combinations were tried.

QuantLab's answer is structural rather than procedural: leakage prevention is
enforced in the type of object each stage is allowed to touch, not left to
the discipline of whoever writes the strategy. The claim "this pipeline is
leak-free" is falsifiable — `validation.verify_vectorized_vs_iterative`
recomputes the pipeline's output at spot-check dates using only truncated
history and asserts it's identical to the full-history run — and the
`ablation` module goes one step further: it deliberately reintroduces the
bug the rest of the codebase prevents, and measures what it costs.

## Install

```bash
git clone https://github.com/<you>/quantlab.git
cd quantlab
pip install -e ".[dev]"      # + ".[data]" for the yfinance loader
```

`requirements.txt` is also included for setups that expect a plain pip file
instead of `pyproject.toml` — both install the same dependencies.

## Running it

```bash
# 1. run the test suite (start here — fastest way to confirm your
#    environment is set up correctly and the leak detector works)
pytest tests/ -v

# 2. run the standard walk-forward backtest on synthetic data
python examples/run_example.py

# 3. run the seven-stage methodology ablation
python examples/run_ablation.py
```

Expect `run_example.py` to print a leak-audit pass, a fold-by-fold
walk-forward table, and a final performance report (Sharpe, Sortino, max
drawdown, PSR, DSR, factor exposure). Expect `run_ablation.py` to print a
seven-row comparison table — see below for how to read it.

To point either script at real data instead of the synthetic generator,
replace the `make_synthetic_market(...)` call with your own `prices` /
`volume` / `factor_returns` DataFrames in the same shape — or use the
real-data scripts described next.

## Running it on real data

`pip install -e ".[data]"` pulls in `yfinance` and `pandas-datareader`, which
`quantlab/data_loaders.py` wraps to produce prices/volume/factor_returns in
the exact shape the rest of the pipeline expects.

```bash
# single-universe walk-forward backtest + ablation on 10 real tickers
python examples/run_example_real_data.py
python examples/run_ablation_real_data.py

# does the ablation's headline finding generalize across universes/periods?
python examples/universe_sensitivity_real_data.py
```

See **Results on real data** below for what these actually produced.

## Design

| Module | Responsibility | Leakage-relevant guarantee |
|---|---|---|
| `data.py` | Point-in-time data access | Downstream code only sees `.asof(t)`, never the raw frame |
| `features.py` | Momentum / vol / z-score / mean-reversion | Every feature is `.shift()`-ed; negative lag raises |
| `signals.py` | Combines features into a ranked signal | Cross-sectional ranking only |
| `sizing.py` | Raw weights → vol-targeted weights | Vol-targeting scalar uses realized vol through `t-1` only |
| `risk.py` | Position / gross / net exposure caps | Static, no lookback — nothing here can leak |
| `costs.py` | Linear cost + square-root market impact | Impact uses trailing ADV only |
| `execution.py` | Weights → realized returns | Explicit `.shift(1)`; exposes `lag_bars=0` ONLY for `ablation.py` |
| `validation.py` | Walk-forward folds (purge+embargo) + leak detector | `verify_vectorized_vs_iterative` is the actual proof |
| `metrics.py` | Sharpe/Sortino/MDD/Calmar/turnover/factor exposure/PSR/DSR | DSR penalizes significance for grid-search trials |
| `backtest.py` | Walk-forward hyperparameter selection | Params chosen on train fold only; OOS read off test fold |
| `ablation.py` | Six-stage methodology comparison | One fix isolated per stage |

## The ablation study

`ablation.run_full_ablation` runs the same strategy through seven
progressively more rigorous backtest configurations:

1. **Naive features, no execution lag, no costs, in-sample fit.**
2. **Same naive features, execution-level lag turned on** — isolates whether
   a single shift, anywhere, is enough to fix the leak from stage 1.
3. **Same unlagged features, T+1 lag, realistic costs added.**
4. **Real structurally-lagged feature layer + T+1 lag + realistic costs**,
   still in-sample.
5. **Walk-forward OOS, no purge/embargo.**
6. **Walk-forward with purge/embargo** — the framework's default, and the
   number this project considers "honest."
7. **Identical stage-6 returns**, naive stats vs PSR/DSR side by side.

Read `sharpe_inflation_vs_stage6` as "how many times more (or less)
impressive did this stage look, relative to the properly-validated result."
On the bundled synthetic market, stage 1 does NOT look inflated — it
collapses to a strongly *negative* Sharpe, because the same-bar
mean-reversion feature is algebraically close to the negative of that bar's
own return, so the leak manifests as the strategy trading against itself
rather than as a suspiciously good backtest. That result is data- and
signal-dependent — see **Results on real data** below for confirmation that
it does *not* generalize as a fixed multiplier.

**Known confound, not yet fixed:** stage 3 → 4 changes two things at once
(feature layer *and* costs are both already on by stage 3, but stage 3 → 4
also swaps unlagged for structurally-lagged features), so that specific
delta can't be attributed to the feature layer alone. Add an intermediate
stage before treating it as a result.

## Results on real data

Ran on real prices via `data_loaders.py` (`yfinance` + Ken French factor
library), not the synthetic generator. Two questions, two scripts:

**1. Does the pipeline itself work on real data?**
`run_example_real_data.py` — 10 large-cap tickers (AAPL, MSFT, AMZN, GOOGL,
META, NVDA, JPM, XOM, JNJ, PG), 2015-01-01 to 2024-12-31, 2515 trading days.
Leak audit passed with `max_abs_diff = 0.0` at every one of 14 spot-checked
dates. Walk-forward backtest completed: **Sharpe -0.40, DSR ≈ 0%, max
drawdown -33.1%** — i.e. no genuine edge, correctly identified as such.

**2. Does the ablation study's "naive backtests are inflated" story
generalize?**
`universe_sensitivity_real_data.py` reruns the full 7-stage ablation across
4 ten-ticker universes × 2 ten-year windows (2005–2014, 2015–2024) = 8 runs:

| universe | window | Stage 1 (naive) Sharpe | Stage 6 (proper WF) Sharpe | Stage 6 DSR | Inflation vs. Stage 6 |
|---|---|---|---|---|---|
| big_tech | 2005–2014 | -1.68 | -0.54 | 5.3e-13 | 3.11× |
| big_tech | 2015–2024 | -1.52 | -0.40 | 4.1e-13 | 3.85× |
| industrials_value | 2005–2014 | -2.11 | -1.91 | 5.4e-76 | 1.11× |
| industrials_value | 2015–2024 | -1.38 | -1.18 | 2.4e-50 | 1.17× |
| consumer_defensive | 2005–2014 | -1.97 | -3.52 | 5.7e-184 | 0.56× |
| consumer_defensive | 2015–2024 | -1.80 | -2.17 | 4.1e-111 | 0.83× |
| financials | 2005–2014 | -1.09 | -0.84 | 3.1e-47 | 1.31× |
| financials | 2015–2024 | -1.53 | -1.38 | 1.1e-66 | 1.11× |

(full precision in `universe_sensitivity_real_data_results.csv`, regenerated
by rerunning the script)

Takeaways:

- **Stage 1 (naive, in-sample) Sharpe is negative in 8/8 runs.** This
  particular momentum/mean-reversion blend has no exploitable edge on any of
  these universes — that conclusion is robust, not an artifact of one
  ticker set.
- **The inflation factor is not a stable constant.** Median across the 8
  runs is **1.14×**, far below the 3.8× figure the single big_tech/2015–2024
  run originally produced. It even drops *below 1* for consumer defensives
  (naive Sharpe *understates* the properly-validated result there), so
  "naive backtests overstate performance by Nx" is not a claim this
  strategy/universe combination supports as a general law — the direction
  and magnitude of the bias depends on the universe.
- The methodologically robust claim is the first one (no real edge,
  confirmed by DSR ≈ 0 everywhere), not a specific inflation multiplier.
  Treat any single-universe inflation number (including the one from
  `run_ablation_real_data.py`) as an illustration, not a generalizable
  result, unless you rerun the sensitivity sweep on your own strategy.

## Known limitations

- **Survivorship bias is not handled** — needs point-in-time constituent data.
- **Significance tests assume i.i.d. daily returns** — use Newey-West/HAC
  before reporting externally.
- **`purge` must be ≥ your longest feature lookback**, set manually.
- No transaction-level fills, no bid/ask spread beyond flat bps, no borrow costs.

## Testing

```bash
pytest tests/ -v --cov=quantlab --cov-report=term-missing
```

`tests/test_leakage.py` is the one to read first. CI runs the full suite on
Python 3.10–3.12 on every push and PR.

## License

MIT — see `LICENSE`.


# 1. clone / init
git init quantlab && cd quantlab
# (place src/, tests/, examples/, .github/, pyproject.toml, requirements.txt,
#  README.md, LICENSE, .gitignore as laid out above)

# 2. environment
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows

# 3. install — pick one
pip install -e ".[dev]"            # recommended: uses pyproject.toml
# or
pip install -r requirements.txt    # if you'd rather not use editable install

# 4. confirm the install actually worked
python -c "import quantlab; print(quantlab.__file__)"

# 5. run the tests — this is your real "does everything work" check
pytest tests/ -v
# expect: 15 passed. If test_leakage.py fails, stop and fix that before
# anything else — it means the leak detector itself is broken.

# 6. run the two example scripts
python examples/run_example.py
python examples/run_ablation.py

# 7. commit and push
git add -A
git commit -m "QuantLab: leakage-audited walk-forward backtester + methodology ablation"
git branch -M main
git remote add origin https://github.com/<you>/quantlab.git
git push -u origin main