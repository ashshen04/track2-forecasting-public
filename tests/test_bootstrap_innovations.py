"""## Executive summary (read this first)

Synthetic histories check joint empirical shocks, uncertainty and cutoff safety.
Bootstrap means resampling whole historical daily change vectors with replacement.
"""
import numpy as np
import pandas as pd
import pytest
from qfbench2_track_forecasting.cli import _draw


def panel():
    dates = pd.bdate_range('2020-01-01', periods=121)
    changes = np.tile([-1., -1., -1., 3.], 30)
    values = np.r_[100., 100. + changes.cumsum()]
    frame = pd.concat([pd.DataFrame({'date': dates, 'asset': a, 'value': v})
                       for a, v in [('SYN-A', values), ('SYN-B', 300.-2*values)]])
    return {'synthetic': frame}, str(dates[-1].date()), changes


def test_empirical_shocks_preserve_support_covariance_and_skew():
    panels, asof, changes = panel()
    draws, meta = _draw(panels, ['SYN-A', 'SYN-B'], [1], asof, 40000, 7,
                        innovation='bootstrap')
    residual = draws[:, :, 0] - np.array([100., 100.])
    support = np.array([-1., 3.]) * np.sqrt(120/119)
    assert np.min(np.abs(residual[:, 0, None] - support), axis=1).max() < 1e-10
    np.testing.assert_allclose(residual[:, 1], -2*residual[:, 0], atol=1e-10)
    assert abs(residual[:, 0].mean()) < .04
    assert residual[:, 0].std() == pytest.approx(changes.std(ddof=1), rel=.02)
    assert (residual[:, 0]**3).mean() > 4
    assert meta['innovation'] == 'bootstrap'


def test_paths_reuse_earlier_shocks_and_ignore_future():
    panels, asof, _ = panel()
    args = (['SYN-A', 'SYN-B'], [5, 1], asof, 1000, 7)
    draws, _ = _draw(panels, *args, innovation='bootstrap')
    first, _ = _draw(panels, args[0], [1], asof, 1000, 7, innovation='bootstrap')
    np.testing.assert_array_equal(draws[:, :, 1], first[:, :, 0])
    panels['synthetic'] = pd.concat([panels['synthetic'], pd.DataFrame({
        'date': [pd.Timestamp('2030-01-01')], 'asset': ['SYN-A'], 'value': [1e9]})])
    future, _ = _draw(panels, *args, innovation='bootstrap')
    np.testing.assert_array_equal(draws, future)


def test_gaussian_default_is_unchanged():
    panels, asof, _ = panel()
    args = (panels, ['SYN-A', 'SYN-B'], [5], asof, 200, 7)
    default, _ = _draw(*args)
    explicit, _ = _draw(*args, innovation='gaussian')
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize('kwargs', [dict(innovation='unknown'),
    dict(innovation='bootstrap', volatility='ewma'),
    dict(innovation='bootstrap', panel_steps=np.array([[1]]))])
def test_unsupported_modes_fail(kwargs):
    panels, asof, _ = panel()
    with pytest.raises(ValueError):
        _draw(panels, ['SYN-A'], [5], asof, 200, 7, **kwargs)


def test_return_steps_keep_their_log_return_centre():
    panels, asof, changes = panel()
    frame = panels['synthetic'].copy()
    returns = np.tile([-0.01, -0.01, -0.01, 0.04], 31)[:121]
    frame.loc[frame.asset == 'SYN-A', 'value'] = returns
    frame.loc[frame.asset == 'SYN-B', 'value'] = 0.
    draws, meta = _draw({'synthetic': frame}, ['SYN-A', 'SYN-B'], [1], asof,
                        20000, 7, target_type='log_return', innovation='bootstrap')
    assert meta['daily_drift']['SYN-A'] == pytest.approx(np.log1p(returns[-120:]).mean())
    assert np.min(np.abs(draws[:, 0, 0, None] - np.log1p([-.01, .04])), axis=1).max() < .0003
    np.testing.assert_array_equal(draws[:, 1, 0], np.zeros(20000))


def test_backtest_bootstrap_route_matches_direct_sampler():
    from scripts.backtest import forecast_samples
    from qfbench2_track_forecasting.grid import GridSpec
    panels, asof, _ = panel()
    grid = GridSpec(('SYN-A', 'SYN-B'), (5, 1))
    result = forecast_samples('reference-bootstrap', panels, pd.Timestamp(asof), grid, 200, 'level')
    direct, _ = _draw(panels, list(grid.assets), list(grid.horizons), asof, 200, 1,
                      innovation='bootstrap')
    np.testing.assert_array_equal(result, direct.reshape(200, -1))
