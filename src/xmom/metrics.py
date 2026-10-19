from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def stats(r: pd.Series) -> pd.Series:
    r = r.dropna()
    eq = (1 + r).cumprod()
    sr = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else np.nan
    return pd.Series({
        "CAGR": eq.iloc[-1] ** (252 / len(r)) - 1,
        "Vol": r.std() * np.sqrt(252),
        "Sharpe": sr,
        "MaxDD": (eq / eq.cummax()).min() - 1,
        "Calmar": (eq.iloc[-1] ** (252 / len(r)) - 1) / abs((eq / eq.cummax()).min() - 1),
    })


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    r = r.dropna()
    sr = r.mean() / r.std()
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (len(r) - 1))
    return float(norm.cdf((sr - sr_benchmark / np.sqrt(252)) / se))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_trials_var_annual: float = 0.25) -> float:
    emc = 0.5772156649
    v = sr_trials_var_annual / 252
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(r, sr0 * np.sqrt(252))


def crash_split(r: pd.Series, market: pd.Series, dd_threshold: float = -0.20) -> pd.DataFrame:
    dd = market / market.cummax() - 1
    in_crash = (dd <= dd_threshold).reindex(r.index).fillna(False)
    return pd.DataFrame({"normal": stats(r[~in_crash]), "market drawdown <= 20%": stats(r[in_crash])})
