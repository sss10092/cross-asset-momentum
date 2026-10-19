import numpy as np
import pandas as pd

from xmom.data import synthetic_prices
from xmom.metrics import deflated_sharpe, probabilistic_sharpe, stats
from xmom.strategy import Config, run_period, target_weights, walk_forward


def test_no_lookahead_future_prices_cannot_change_past_weights():
    p = synthetic_prices(n_days=1500, seed=1)
    cut = p.index[1000]
    q = p.copy()
    q.loc[q.index > cut] *= np.exp(np.random.default_rng(9).normal(0, 0.2, q.loc[q.index > cut].shape))
    _, held_p, _ = run_period(p, Config())
    _, held_q, _ = run_period(q, Config())
    upto = held_p.index[held_p.index.get_loc(cut) + 1]
    pd.testing.assert_frame_equal(held_p.loc[:upto], held_q.loc[:upto])


def test_weights_are_dollar_neutral_and_gross_two():
    w = target_weights(synthetic_prices(n_days=800, seed=2), Config())
    live = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(live.clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose(live.clip(upper=0).sum(axis=1), -1.0)


def test_weights_only_change_on_rebalance_days():
    w = target_weights(synthetic_prices(n_days=800, seed=2), Config())
    changes = w.diff().abs().sum(axis=1) > 1e-12
    assert set(w.index[changes].dayofweek) <= {4, 0}


def test_first_days_of_each_test_year_are_invested():
    r, held, _, _ = walk_forward(synthetic_prices(n_days=252 * 6, seed=3), Config())
    for y in sorted(set(held.index.year))[1:]:
        first = held[held.index.year == y].iloc[2:10]
        assert (first.abs().sum(axis=1) > 0).all()


def test_planted_momentum_is_harvested_and_null_is_not():
    cfg = Config()
    null = np.mean([stats(walk_forward(synthetic_prices(momentum=0.0, seed=s), cfg)[0])["Sharpe"] for s in range(3)])
    mom = np.mean([stats(walk_forward(synthetic_prices(momentum=0.3, seed=s), cfg)[0])["Sharpe"] for s in range(3)])
    assert mom > 1.0 and mom > null + 1.0


def test_costs_reduce_returns():
    p = synthetic_prices(n_days=1200, seed=4)
    free, _, _ = run_period(p, Config(cost_bps=0))
    paid, _, to = run_period(p, Config(cost_bps=10))
    assert np.isclose((free - paid).sum(), (to * 10 / 1e4).sum())


def test_sharpe_diagnostics_monotone():
    r = pd.Series(np.random.default_rng(0).normal(0.0006, 0.01, 2520))
    psr = probabilistic_sharpe(r)
    assert 0.9 < psr <= 1.0
    assert deflated_sharpe(r, n_trials=200) < psr


def test_holiday_friday_rebalances_on_thursday():
    p = synthetic_prices(n_days=800, seed=2)
    fridays = p.index[p.index.dayofweek == 4]
    p = p.drop(fridays[120])
    w = target_weights(p, Config())
    changes = w.index[w.diff().abs().sum(axis=1) > 1e-12]
    assert fridays[120] - pd.Timedelta(days=1) in changes
    assert set(changes.dayofweek) <= {3, 4}
