from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import UNIVERSE


def load_yahoo(tickers=UNIVERSE, start="2007-01-01", end=None) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all").ffill()


def synthetic_prices(n_days=252 * 12, n_assets=12, momentum=0.0, seed=0, crash_every=750) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corr = 0.3 + 0.7 * np.eye(n_assets)
    L = np.linalg.cholesky(corr)
    vol = rng.uniform(0.08, 0.30, n_assets) / np.sqrt(252)
    regime = np.ones(n_days)
    for s in range(crash_every, n_days, crash_every):
        regime[s:s + 60] = 2.5
    r = np.zeros((n_days, n_assets))
    for t in range(n_days):
        drift = momentum * r[max(0, t - 60):t].sum(axis=0) / 60 if t > 60 else 0.0
        r[t] = drift + regime[t] * vol * (L @ rng.standard_normal(n_assets))
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])
