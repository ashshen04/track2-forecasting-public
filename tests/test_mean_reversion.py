"""## Executive summary (read this first)

Synthetic series test joint mean reversion without using public-card outcomes.
For example, future rows must not change a forecast made today.
"""
import numpy as np
import pandas as pd
import pytest

from baselines.mean_reversion import draw_mean_reversion
from scripts.backtest import forecast_samples
from qfbench2_track_forecasting.grid import GridSpec


def history():
    rng = np.random.default_rng(17)
    dates = pd.bdate_range('2020-01-01', periods=180)
    values = np.empty(180)
    values[0] = 4.
    for i in range(1, len(values)):
        values[i] = 2 + .75 * (values[i-1] - 2) + rng.normal(0, .1)
    frame = pd.DataFrame({'date': dates, 'asset': 'SYN-A', 'value': values})
    return {'p': pd.concat([frame, frame.assign(asset='SYN-B', value=2*values)])}, str(dates[-1].date())


def test_future_ignored_and_unsorted_horizon_keys_preserved():
    panels, asof = history()
    args = (['SYN-A', 'SYN-B'], [21, 1, 5], asof, 1000, 1)
    draws, meta = draw_mean_reversion(panels, *args)
    assert draws.shape == (1000, 2, 3)
    assert all(.5 <= v <= 1 for v in meta['phi'].values())
    future = panels['p'].iloc[-2:].assign(date=pd.Timestamp('2030-01-01'), value=1e12)
    changed, _ = draw_mean_reversion({'p': pd.concat([panels['p'], future])}, *args)
    np.testing.assert_array_equal(draws, changed)
    sorted_draws, _ = draw_mean_reversion(panels, args[0], [1, 5, 21], asof, 1000, 1)
    np.testing.assert_array_equal(draws, sorted_draws[:, :, [2, 0, 1]])
    # Perfectly dependent synthetic assets must share innovations and the centre.
    np.testing.assert_allclose(draws[:, 1], 2*draws[:, 0], atol=2e-5)


def test_constant_series_stays_constant():
    panels, asof = history()
    panels['p']['value'] = 3.
    draws, meta = draw_mean_reversion(panels, ['SYN-A'], [1, 21], asof, 200, 7)
    np.testing.assert_array_equal(draws, np.full((200, 1, 2), 3.))
    assert meta['phi']['SYN-A'] == 1.


def test_centre_variant_preserves_reference_noise_exactly():
    from qfbench2_track_forecasting.cli import _draw
    panels, asof = history()
    args = (panels, ['SYN-A', 'SYN-B'], [21, 1, 5], asof, 1000, 1)
    baseline, info = _draw(*args)
    candidate, meta = draw_mean_reversion(*args, centre_only=True)
    for ai, asset in enumerate(args[1]):
        for hi, h in enumerate(args[2]):
            shift = meta['horizon_centres'][asset][str(h)] - info['last'][asset]
            np.testing.assert_allclose(candidate[:, ai, hi]-baseline[:, ai, hi], shift, atol=1e-15)


def test_stationary_path_has_bounded_long_horizon_spread():
    panels, asof = history()
    draws, _ = draw_mean_reversion(panels, ['SYN-A'], [21, 100], asof, 10000, 1)
    # Random walk spreads would grow by sqrt(100/21); stationary AR paths should saturate.
    assert draws[:, 0, 1].std() / draws[:, 0, 0].std() < 1.2


def test_gap_is_not_used_as_a_daily_regression_pair():
    panels, asof = history()
    panels['p'].loc[panels['p'].date < '2020-05-01', 'date'] -= pd.Timedelta(days=3650)
    _, meta = draw_mean_reversion(panels, ['SYN-A'], [5], asof, 200, 1)
    assert meta['last_pair_date'] == asof
    assert meta['pair_rows'] < 120


def test_return_target_is_explicitly_refused():
    panels, asof = history()
    with pytest.raises(ValueError, match='level'):
        draw_mean_reversion(panels, ['SYN-A'], [5], asof, 200, 1, target_type='log_return')


def test_runner_uses_candidate_only_for_levels():
    panels, asof = history()
    grid = GridSpec(('SYN-A', 'SYN-B'), (5, 21))
    routed = forecast_samples('reference-mean-reversion', panels, pd.Timestamp(asof), grid, 200, 'level')
    direct, _ = draw_mean_reversion(panels, list(grid.assets), list(grid.horizons), asof, 200, 1)
    np.testing.assert_array_equal(routed, direct.reshape(200, -1))
    # Daily returns are increments; treating them as reverting price levels is inappropriate.
    panels['p']['value'] = panels['p']['value'] / 1000
    candidate = forecast_samples('reference-mean-reversion', panels, pd.Timestamp(asof), grid, 200, 'log_return')
    baseline = forecast_samples('reference', panels, pd.Timestamp(asof), grid, 200, 'log_return')
    np.testing.assert_array_equal(candidate, baseline)
