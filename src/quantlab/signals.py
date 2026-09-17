# signals.py
from __future__ import annotations
import pandas as pd
from . import features as F


def momentum_meanrev_blend(prices, returns, mom_window=60, mr_window=5,
                            mom_weight=0.5, zscore_window=252) -> pd.DataFrame:
    """+ long-horizon momentum, - short-horizon mean-reversion, both
    z-scored (lagged) then blended and cross-sectionally ranked, so the
    book is long/short and not just 'long everything when market is up.'"""
    mom = F.momentum(prices, mom_window, lag=1)
    mom_z = F.zscore(mom, zscore_window, lag=0)  # mom already lagged

    mr = F.mean_reversion_score(prices, mr_window, lag=1)
    mr_z = F.zscore(mr, zscore_window, lag=0)

    blended = mom_weight * mom_z + (1 - mom_weight) * mr_z
    return F.cross_sectional_rank(blended)