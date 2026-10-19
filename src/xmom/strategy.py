from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "USO", "DBC", "UUP", "FXE", "FXY"]


@dataclass
class Config:
    universe: list = field(default_factory=lambda: list(UNIVERSE))
    start: str = "2007-01-01"
    rebalance_freq: str = "W-FRI"
    mom_horizons: tuple = (20, 60, 120)
    mom_weights: tuple = (0.5, 0.3, 0.2)
    vol_window: int = 20
    dd_window: int = 60
    regime_z_window: int = 252
    regime_sigmoid_alpha: float = 2.0
    regime_sigmoid_center: float = 1.0
    regime_center_quantile: float = 0.90
    dd_alpha: float = 0.3
    cost_bps: float = 5.0
    long_only: bool = False


def realized_vol(returns, window):
    return returns.rolling(window).std() * np.sqrt(252)


def cs_rank(df):
    return df.rank(axis=1, pct=True) - 0.5


def zscore(x, window):
    return (x - x.rolling(window).mean()) / x.rolling(window).std()


def regime_weight(prices, cfg: Config) -> pd.Series:
    rets = prices.pct_change()
    med = realized_vol(rets, cfg.vol_window).median(axis=1)
    z = zscore(med, cfg.regime_z_window).clip(-5, 5)
    return 1 / (1 + np.exp(cfg.regime_sigmoid_alpha * (z - cfg.regime_sigmoid_center)))


def signal(prices, cfg: Config) -> pd.DataFrame:
    rets = prices.pct_change()
    mom = sum(w * prices.pct_change(h) / realized_vol(rets, h) for h, w in zip(cfg.mom_horizons, cfg.mom_weights))
    filtered = cs_rank(mom).mul(regime_weight(prices, cfg), axis=0)
    dd = prices / prices.rolling(cfg.dd_window).max() - 1
    dd_pen = cs_rank(dd.abs() / realized_vol(rets, cfg.dd_window))
    return filtered - cfg.dd_alpha * dd_pen


def target_weights(prices, cfg: Config) -> pd.DataFrame:
    vol = realized_vol(prices.pct_change(), cfg.vol_window)
    raw = signal(prices, cfg) / vol
    if cfg.long_only:
        raw = raw.clip(lower=0)
    long = raw.clip(lower=0)
    short = raw.clip(upper=0)
    w = long.div(long.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if not cfg.long_only:
        w = w + short.div(short.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    freq = {"ME": "M", "QE": "Q", "YE": "Y"}.get(cfg.rebalance_freq, cfg.rebalance_freq)
    periods = w.index.to_period(freq)
    is_last = ~periods.duplicated(keep="last")
    if w.index[-1].normalize() < periods[-1].end_time.normalize():
        is_last[-1] = False
    rebal = w[is_last]
    return rebal.reindex(prices.index, method="ffill").fillna(0)


def run_period(prices, cfg: Config, start=None, end=None):
    w_decided = target_weights(prices, cfg)
    held = w_decided.shift(1).fillna(0)
    rets = prices.pct_change().fillna(0)
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    net = gross - turnover * cfg.cost_bps / 1e4
    sl = slice(start, end)
    return net.loc[sl], held.loc[sl], turnover.loc[sl]


def fit_regime_center(train_prices, cfg: Config) -> float:
    rets = train_prices.pct_change()
    z = zscore(realized_vol(rets, cfg.vol_window).median(axis=1), cfg.regime_z_window).dropna()
    return float(z.quantile(cfg.regime_center_quantile)) if len(z) else cfg.regime_sigmoid_center


def walk_forward(prices, cfg: Config, first_test_year: int | None = None):
    years = sorted(set(prices.index.year))
    first = first_test_year or years[0] + 2
    out_r, out_w, out_t, centers = [], [], [], {}
    for y in years:
        if y < first:
            continue
        train = prices.loc[: f"{y - 1}-12-31"]
        if len(train) < cfg.regime_z_window + max(cfg.mom_horizons):
            continue
        c = Config(**{**cfg.__dict__, "regime_sigmoid_center": fit_regime_center(train, cfg)})
        centers[y] = c.regime_sigmoid_center
        hist = prices.loc[: f"{y}-12-31"]
        r, w, t = run_period(hist, c, f"{y}-01-01", f"{y}-12-31")
        out_r.append(r)
        out_w.append(w)
        out_t.append(t)
    return pd.concat(out_r), pd.concat(out_w), pd.concat(out_t), centers
