# tests/conftest.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from src.data import make_synthetic_market


@pytest.fixture(scope="module")
def market():
    m = make_synthetic_market(n_assets=6, n_days=600, seed=42)
    m["returns"] = m["prices"].pct_change()
    return m