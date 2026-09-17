# metrics.py
"""
Standard stats plus the two things a sharp interviewer checks for:
PSR (Bailey & Lopez de Prado) corrects Sharpe significance for sample
skew/kurtosis; DSR further deflates it for the number of strategy
variants you tried before picking the winner (multiple-testing bias).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

ANNUALIZATION = 252

def annualized_return(r): return r.mean() * ANNUALIZATION
def annualized_vol(r): return r.std(ddof=1) * np.sqrt(ANNUALIZATION)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / ANNUALIZATION
    sd = excess.std(ddof=1)
    # Guard near-zero variance, not just exact zero -- floating-point noise
    # on a constant series gives std ~1e-19, not exactly 0.0, which would
    # otherwise blow up into a nonsense billion-Sharpe instead of NaN.
    if pd.isna(sd) or sd < 1e-10:
        return np.nan
    return (excess.mean() / sd) * np.sqrt(ANNUALIZATION)

def sortino_ratio(returns, rf=0.0, target=0.0):
    excess = returns - rf / ANNUALIZATION
    downside = excess[excess < target]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    return np.nan if not dd else (excess.mean() / dd) * np.sqrt(ANNUALIZATION)

def max_drawdown(returns) -> dict:
    curve = (1 + returns.fillna(0)).cumprod()
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    peak = curve.loc[:trough].idxmax()
    return {"max_drawdown": drawdown.min(), "peak_date": peak, "trough_date": trough,
            "drawdown_series": drawdown}

def calmar_ratio(returns):
    mdd = max_drawdown(returns)["max_drawdown"]
    return np.nan if not mdd or np.isnan(mdd) else annualized_return(returns) / abs(mdd)

def average_turnover(turnover_series): return turnover_series.mean()

def factor_exposure(returns, factor_returns) -> dict:
    """OLS via numpy.linalg.lstsq (no statsmodels dependency)."""
    df = pd.concat([returns.rename("strat"), factor_returns], axis=1).dropna()
    y, X = df["strat"].values, df.drop(columns="strat").values
    X_design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ coefs
    ss_res, ss_tot = np.sum((y - y_hat) ** 2), np.sum((y - y.mean()) ** 2)
    return {"alpha_annualized": coefs[0] * ANNUALIZATION,
            "betas": pd.Series(coefs[1:], index=factor_returns.columns),
            "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "n_obs": len(df)}

def significance_ttest(returns) -> dict:
    """One-sample t-test vs 0. Assumes i.i.d. returns -- daily returns
    are typically weakly autocorrelated, so treat p as indicative;
    use Newey-West/HAC for a rigorous version."""
    r = returns.dropna()
    t_stat, p_value = stats.ttest_1samp(r, 0.0)
    return {"t_stat": t_stat, "p_value": p_value, "n": len(r)}

def probabilistic_sharpe_ratio(returns, benchmark_sr=0.0) -> float:
    """PSR(SR*) = Phi((SR_hat-SR*)*sqrt(n-1) / sqrt(1-skew*SR_hat+(kurt-1)/4*SR_hat^2))
    All Sharpes here are DAILY units -- don't mix with an annualized SR_hat."""
    r = returns.dropna()
    n = len(r)
    sr_hat = r.mean() / r.std(ddof=1)
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2)
    z = (sr_hat - benchmark_sr) * np.sqrt(n - 1) / denom
    return stats.norm.cdf(z)

def deflated_sharpe_ratio(returns, n_trials, trial_sr_std=None) -> dict:
    """Bailey & Lopez de Prado DSR: benchmark_sr in PSR becomes the
    expected MAX Sharpe of n_trials skill-less strategies, so significance
    is judged against how far you'd expect to get by luck alone given how
    much you searched. `trial_sr_std`: cross-sectional std (daily units)
    of Sharpes across your actual grid search -- pass the real one if you
    have it; conservative default is 1.0 daily-units (very high bar)."""
    r = returns.dropna()
    n = len(r)
    sr_hat = r.mean() / r.std(ddof=1)
    trial_sr_std = 1.0 if trial_sr_std is None else trial_sr_std
    eg = 0.5772156649
    e_max_z = ((1 - eg) * stats.norm.ppf(1 - 1.0 / n_trials) +
               eg * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))) if n_trials > 1 else 0.0
    sr_benchmark = trial_sr_std * e_max_z
    return {"deflated_sharpe_ratio": probabilistic_sharpe_ratio(r, sr_benchmark),
            "implied_benchmark_sr_daily": sr_benchmark, "n_trials_assumed": n_trials}

def full_report(net_returns, turnover_series, factor_returns=None,
                 n_trials_for_dsr=1, trial_sr_std_daily=None) -> dict:
    mdd = max_drawdown(net_returns)
    report = {
        "annualized_return": annualized_return(net_returns),
        "annualized_vol": annualized_vol(net_returns),
        "sharpe": sharpe_ratio(net_returns),
        "sortino": sortino_ratio(net_returns),
        "max_drawdown": mdd["max_drawdown"],
        "calmar": calmar_ratio(net_returns),
        "avg_daily_turnover": average_turnover(turnover_series),
        "ttest": significance_ttest(net_returns),
        "psr_vs_zero_sharpe": probabilistic_sharpe_ratio(net_returns, 0.0),
        "dsr": deflated_sharpe_ratio(net_returns, n_trials_for_dsr, trial_sr_std_daily),
    }
    if factor_returns is not None:
        report["factor_exposure"] = factor_exposure(net_returns, factor_returns)
    return report
