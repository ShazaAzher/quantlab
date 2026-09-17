# tests/test_validation.py
import pandas as pd
from src import validation as V


def test_folds_have_no_train_test_overlap():
    idx = pd.date_range("2015-01-01", periods=1000, freq="B")
    folds = V.walk_forward_folds(idx, train_size=300, test_size=60, purge=20, embargo=10)
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(set(fold.test_idx))


def test_purge_actually_removes_bars_adjacent_to_test():
    idx = pd.date_range("2015-01-01", periods=1000, freq="B")
    folds = V.walk_forward_folds(idx, train_size=300, test_size=60, purge=20, embargo=10)
    fold = folds[0]
    gap = (fold.test_idx[0] - fold.train_idx[-1]).days
    assert gap >= 20  # at least `purge` calendar days of separation


def test_no_folds_fit_raises_clean_error():
    idx = pd.date_range("2015-01-01", periods=50, freq="B")
    try:
        V.walk_forward_folds(idx, train_size=300, test_size=60, purge=20, embargo=10)
        assert False, "should have raised"
    except ValueError as e:
        assert "No folds fit" in str(e)