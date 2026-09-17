# execution.py
"""
The rule enforced here, non-negotiably:
    target_weight(t)  = decision made from info at close of day t
    held_weight(t+1)  = target_weight(t)      <- explicit second shift
    pnl(t+1)          = held_weight(t+1) * asset_return(t+1) - costs(t+1)
A signal built off day-t data earns its first dollar on day t+1's return,
and pays its trading cost on day t+1 too -- no zero-cost teleportation.
"""
from __future__ import annotations
import pandas as pd
from ... import costs as C


def held_weights_from_target(target_weights: pd.DataFrame) -> pd.DataFrame:
    return target_weights.shift(1).fillna(0.0)


def simulate(target_weights, returns, volume, nav=1_000_000.0,
             linear_cost_bps=2.0, impact_coef=0.1) -> dict:
    held = held_weights_from_target(target_weights).reindex(returns.index).fillna(0.0)
    gross_pnl = (held * returns).sum(axis=1)
    cost_frac = C.total_costs(held, nav, volume, linear_cost_bps, impact_coef)
    net_returns = gross_pnl - cost_frac
    return {"held_weights": held, "gross_returns": gross_pnl, "costs": cost_frac,
            "net_returns": net_returns, "turnover": C.turnover(held)}

def held_weights_from_target(target_weights: pd.DataFrame, lag_bars: int = 1) -> pd.DataFrame:
    """lag_bars=0 deliberately disables the shift -- this exists ONLY so
    ablation.py can reproduce the "no execution lag" bug on purpose and
    measure how much it costs. Never call this with lag_bars=0 in a real
    backtest; nothing here stops you, because ablation.py's entire point
    is to demonstrate what happens when nothing stops you."""
    if lag_bars == 0:
        return target_weights.fillna(0.0)
    return target_weights.shift(lag_bars).fillna(0.0)