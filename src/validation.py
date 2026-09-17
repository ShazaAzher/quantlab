# validation.py
"""
1. Walk-forward folds with PURGE (drop train bars whose lookback window
   overlaps into test) and EMBARGO (skip bars after test before next
   train starts) -- Lopez de Prado, AFML ch.7.
2. verify_vectorized_vs_iterative: the actual leak detector. Recomputes
   the pipeline at spot-check dates using ONLY truncated history and
   asserts identical output to the vectorized run.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def walk_forward_folds(index, train_size, test_size, purge=5, embargo=5, step=None) -> list[Fold]:
    step = step or test_size
    folds, start, n = [], 0, len(index)
    while True:
        train_end = start + train_size
        test_start = train_end + purge
        test_end = test_start + test_size
        if test_end + embargo > n:
            break
        train_idx = index[start: train_end - purge] if purge else index[start:train_end]
        test_idx = index[test_start:test_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            break
        folds.append(Fold(train_idx=train_idx, test_idx=test_idx))
        start += step
    if not folds:
        raise ValueError(f"No folds fit given history length n={n}.")
    return folds


def verify_vectorized_vs_iterative(pipeline_fn, prices, check_dates, atol=1e-9) -> pd.DataFrame:
    """pipeline_fn(prices_slice) -> weights/signal frame. For each check
    date d: vectorized = pipeline_fn(full).loc[d], iterative =
    pipeline_fn(prices.loc[:d]).loc[d]. A truly causal fn can't disagree."""
    full_output = pipeline_fn(prices)
    rows = []
    for d in check_dates:
        restricted = prices.loc[:d]
        if len(restricted) < 2:
            continue
        restricted_output = pipeline_fn(restricted)
        diff = (full_output.loc[d] - restricted_output.loc[d]).abs()
        max_diff = np.nanmax(diff.values) if len(diff) else np.nan
        rows.append({"date": d, "max_abs_diff": max_diff, "passed": bool(np.nan_to_num(max_diff) <= atol)})
    result = pd.DataFrame(rows).set_index("date")
    if not result["passed"].all():
        bad = result[~result["passed"]]
        raise AssertionError(
            f"LOOK-AHEAD BIAS DETECTED on {len(bad)} date(s):\n{bad}\n"
            "The pipeline function is using data beyond the check date.")
    return result