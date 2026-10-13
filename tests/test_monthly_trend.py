"""## Executive summary (read this first)

Synthetic monthly trends must preserve the numerical forecast's joint shocks.
Future rows cannot affect fitted trends. Missing months and wrong target types refuse.
These tests establish forecasting mechanics, not accuracy on CPI outcomes.
"""
import numpy as np
import pandas as pd
import pytest
from baselines.monthly_trend import centre_shifts, validate_cpi_card
from qfbench2_track_forecasting.cli import _draw


def panel():
    dates = pd.date_range('2010-01-01', periods=60, freq='MS')
    return pd.DataFrame({'date': dates, 'asset': 'CPI_ALL',
        'value': 100 + np.cumsum(np.tile([1.5, .5], 30))})


@pytest.mark.parametrize('policy', ['recent', 'half', 'damped'])
def test_monthly_trend_preserves_shocks_and_uses_steps_not_keys(policy):
    p = panel(); steps = np.array([[8, 9]])
    base, _ = _draw({'p': p}, ['CPI_ALL'], [140, 160], '2014-12-31', 1000, 7,
                    panel_steps=steps)
    actual, meta = _draw({'p': p}, ['CPI_ALL'], [140, 160], '2014-12-31', 1000, 7,
                         panel_steps=steps, monthly_trend=policy)
    expected = np.array([8., 9.])
    if policy == 'half': expected *= .5
    if policy == 'damped': expected = np.array([sum(.9 ** np.arange(s)) for s in (8, 9)])
    np.testing.assert_allclose(actual - base, np.broadcast_to(expected, actual.shape))
    np.testing.assert_allclose(np.cov(actual[:, 0], rowvar=False), np.cov(base[:, 0], rowvar=False))
    assert meta['monthly_trend']['mean_monthly_changes']['CPI_ALL'] == pytest.approx(1.)
    assert list(meta['centre_shifts']['CPI_ALL']) == ['140', '160']


def test_future_values_cannot_change_trend_or_draws():
    p = panel(); args = (['CPI_ALL'], [21], '2014-01-15', 200, 1)
    first, _ = _draw({'p': p}, *args, panel_steps=np.array([[2]]), monthly_trend='recent')
    p.loc[p.date > pd.Timestamp('2014-01-15'), 'value'] = 1e8
    again, _ = _draw({'p': p}, *args, panel_steps=np.array([[2]]), monthly_trend='recent')
    assert np.array_equal(first, again)


def test_missing_month_in_trend_window_is_refused():
    p = panel().set_index('date').value.drop(pd.Timestamp('2014-08-01'))
    with pytest.raises(ValueError, match='consecutive'):
        centre_shifts({'CPI_ALL': p}, [[2]], 'recent')


def test_formal_opt_in_cannot_apply_cpi_trend_to_unemployment():
    card = {'targets': {'target_frequency': 'monthly', 'target_type': 'level',
                        'value_unit': 'percent_unemployed'}}
    with pytest.raises(ValueError, match='CPI index'):
        validate_cpi_card(card, 'recent')
    validate_cpi_card(card, 'none')


def test_monthly_trend_refuses_daily_steps():
    with pytest.raises(ValueError, match='explicit monthly'):
        _draw({'p': panel()}, ['CPI_ALL'], [21], '2014-12-31', 200, 1, monthly_trend='recent')
