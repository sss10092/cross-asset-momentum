from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_yahoo, synthetic_prices
from .metrics import crash_split, deflated_sharpe, probabilistic_sharpe, stats
from .strategy import Config, walk_forward


def backtest(prices: pd.DataFrame, cfg: Config, fig: str | None, n_trials: int = 20):
    r, held, turnover, centers = walk_forward(prices, cfg)
    rets = prices.pct_change().loc[r.index]
    bench = pd.DataFrame({"Strategy": r, "EqualWeight": rets.mean(axis=1)})
    if {"SPY", "IEF"} <= set(prices.columns):
        bench["60/40"] = 0.6 * rets["SPY"] + 0.4 * rets["IEF"]
    pd.set_option("display.width", 160)
    print(bench.apply(stats).T.round(3).to_string())
    print(f"\nannual turnover {turnover.sum() / (len(r) / 252):.1f}x   "
          f"PSR(SR>0) {probabilistic_sharpe(r):.2f}   deflated SR ({n_trials} trials) {deflated_sharpe(r, n_trials):.2f}")
    if "SPY" in prices.columns:
        print("\n" + crash_split(r, prices["SPY"]).round(3).to_string())
    print("\nfitted regime threshold by year:", {y: round(c, 2) for y, c in centers.items()})
    if fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(fig).parent.mkdir(parents=True, exist_ok=True)
        ax = (1 + bench).cumprod().plot(figsize=(11, 4.5), lw=1.8, color=["#2a78d6", "#eb6834", "#1baf7a"][: bench.shape[1]])
        ax.set_yscale("log")
        ax.set_title("Walk-forward growth of $1 (net of 5bp costs)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig, dpi=130)
        print(f"figure: {fig}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xmom")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--start", default="2007-01-01")
    b.add_argument("--figure", default="docs/equity.png")
    b.add_argument("--long-only", action="store_true")
    sub.add_parser("demo")
    a = ap.parse_args(argv)
    if a.cmd == "backtest":
        cfg = Config(start=a.start, long_only=a.long_only)
        backtest(load_yahoo(cfg.universe, cfg.start), cfg, a.figure)
    else:
        for m in (0.0, 0.15, 0.3):
            srs = [stats(walk_forward(synthetic_prices(momentum=m, seed=s), Config())[0])["Sharpe"] for s in range(4)]
            print(f"planted momentum {m:.2f}: walk-forward Sharpe per seed {np.round(srs, 2)}  mean {np.mean(srs):.2f}")


if __name__ == "__main__":
    main()
