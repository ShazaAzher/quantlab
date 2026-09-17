# README.md

# QuantLab

A walk-forward backtesting framework for systematic multi-asset strategies,
built around one non-negotiable constraint: **every stage of the pipeline is
causal by construction, and that claim is checked by code, not asserted in
prose.** It also includes a six-stage ablation study that measures, on the
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

# 3. run the six-stage methodology ablation
python examples/run_ablation.py
```

Expect `run_example.py` to print a leak-audit pass, a fold-by-fold
walk-forward table, and a final performance report (Sharpe, Sortino, max
drawdown, PSR, DSR, factor exposure). Expect `run_ablation.py` to print a
six-row comparison table — see below for how to read it.

To point either script at real data instead of the synthetic generator,
replace the `make_synthetic_market(...)` call with your own `prices` /
`volume` / `factor_returns` DataFrames in the same shape.

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

`ablation.run_full_ablation` runs the same strategy through six
progressively more rigorous backtest configurations:

1. **Naive features, no execution lag, no costs, in-sample fit.**
2. **Same naive features, execution-level lag turned on** — isolates whether
   a single shift, anywhere, is enough to fix the leak from stage 1.
3. **Real structurally-lagged feature layer + realistic costs**, still in-sample.
4. **Walk-forward OOS, no purge/embargo.**
5. **Walk-forward with purge/embargo** — the framework's default, and the
   number this project considers "honest."
6. **Identical stage-5 returns**, naive stats vs PSR/DSR side by side.

Read `sharpe_inflation_vs_stage5` as "how many times more (or less)
impressive did this stage look, relative to the properly-validated result."
On the bundled synthetic market, stage 1 does NOT look inflated — it
collapses to a strongly *negative* Sharpe, because the same-bar
mean-reversion feature is algebraically close to the negative of that bar's
own return, so the leak manifests as the strategy trading against itself
rather than as a suspiciously good backtest. That result is data- and
signal-dependent — don't assume it generalizes without rerunning it.

**Known confound, not yet fixed:** stage 2 → 3 changes two things at once
(feature layer *and* costs), so that specific delta can't be attributed to
costs alone. Add an intermediate stage before treating it as a result.

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