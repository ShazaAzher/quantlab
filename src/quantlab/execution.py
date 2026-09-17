# src/quantlab/execution.py
"""
execution.py
------------
The last and most leakage-sensitive step: turning target weights into
realized P&L. The rule enforced here, non-negotiably:

    target_weight(t)  = decision made using information available AT
                         THE CLOSE of day t (already respected upstream
                         because every feature was lagged by 1).
    held_weight(t+1)  = target_weight(t)      <-- one more explicit shift
    pnl(t+1)          = held_weight(t+1) * asset_return(t+1) - costs(t+1)

So a signal computed off day-t data earns its first dollar of P&L on
day t+1's return, and the trade to get into that position is costed on
day t+1 as well (you can't have zero-cost teleportation into a position
the moment you decide on it).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from . import costs as C


def held_weights_from_target(target_weights: pd.DataFrame, lag_bars: int = 1) -> pd.DataFrame:
    """The one-line embodiment of the T+1 rule. Kept as its own function
    (not inlined in run_backtest) so it's a single, greppable, testable
    place where the execution lag lives.

    `lag_bars=0` deliberately disables the shift -- this exists ONLY so
    ablation.py can reproduce the "no execution lag" bug on purpose and
    measure how much it inflates Sharpe. Never call this with lag_bars=0
    in a real backtest; nothing here stops you, because ablation.py's
    entire point is to demonstrate what happens when nothing stops you.
    """
    if lag_bars == 0:
        return target_weights.fillna(0.0)
    return target_weights.shift(lag_bars).fillna(0.0)


def simulate(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    nav: float = 1_000_000.0,
    linear_cost_bps: float = 2.0,
    impact_coef: float = 0.1,
    lag_bars: int = 1,
) -> dict:
    held = held_weights_from_target(target_weights, lag_bars=lag_bars)
    held = held.reindex(returns.index).fillna(0.0)

    gross_pnl = (held * returns).sum(axis=1)
    cost_frac = C.total_costs(held, nav, volume, linear_cost_bps, impact_coef)
    net_returns = gross_pnl - cost_frac
    tno = C.turnover(held)

    return {
        "held_weights": held,
        "gross_returns": gross_pnl,
        "costs": cost_frac,
        "net_returns": net_returns,
        "turnover": tno,
    }