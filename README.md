# Cross-Asset Momentum

A regime-conditioned cross-sectional momentum strategy across 12 asset-class ETFs (US equity, Treasuries, metals, oil, broad commodities, USD, EUR, JPY), with walk-forward validation, transaction costs, and tests that make look-ahead bias impossible to reintroduce silently.

## Strategy

1. **Momentum.** Vol-adjusted returns over 20/60/120 days, weighted 0.5/0.3/0.2, ranked cross-sectionally.
2. **Regime filter.** The z-score of cross-asset median realised vol goes through a sigmoid, which scales the whole book down smoothly as stress rises.
3. **Drawdown penalty.** Assets deep in a vol-scaled 60-day drawdown are ranked down.
4. **Portfolio.** Inverse-vol weights, long book normalised to +1 and short book to -1, weekly Friday rebalance, 5bp per unit turnover.
5. **Walk-forward.** Each year, the regime threshold (the 90th percentile of historical stress) is fitted on data up to Dec 31 of the prior year, then frozen for the year.

## Validation on known answers (`xmom demo`)

| Planted momentum | Walk-forward Sharpe (4 seeds) | Mean |
|---|---|---|
| none (null) | 0.11, 0.06, -0.43, -0.64 | -0.22 |
| moderate | 1.01, 0.84, 0.37, 0.15 | 0.59 |
| strong | 2.10, 1.84, 1.45, 1.08 | 1.62 |

The strategy finds momentum when it exists and does not invent it when it doesn't.

## Real-data result

From the original notebook run (2016-2026):

| | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| Strategy | 3.0% | 12.6% | 0.30 | -27.7% |
| Equal weight | 8.9% | 9.4% | 0.95 | -19.6% |
| 60/40 | 9.8% | 10.7% | 0.93 | -21.0% |

## Bugs fixed from the notebook

1. **Same-day look-ahead.** Weights were `signal / vol`, with `vol` including the current day's return, and they were applied to that same day's return. Changing only Friday's close changed Friday's weight, which then earned Friday's return. Weights are now decided at close t-1 and earn t. `test_no_lookahead_future_prices_cannot_change_past_weights` perturbs every price after a cutoff and asserts identical positions up to it.
2. **Vol window restarted every January.** `build_portfolio` recomputed rolling vol on the test-year slice only, so the first ~20 trading days of each year had NaN weights and sat flat. Warm-up now uses all prior history.

## Statistics reported

CAGR, vol, Sharpe, max drawdown, Calmar; Probabilistic Sharpe Ratio (skew/kurtosis-adjusted); Deflated Sharpe Ratio for the number of variants tried; and performance split by whether SPY is in a 20% or deeper drawdown.

## Run

```bash
pip install -e '.[dev,data]'
xmom demo
xmom backtest
pytest -q
```
