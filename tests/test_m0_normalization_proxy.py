"""## Executive summary (read this first)

Check input-only normalization simulation against the published daily M0 procedure.
Synthetic future observations cannot alter a proxy computed at an earlier origin.
These checks do not establish exact agreement with organizer scale generation.
"""
import numpy as np
import pandas as pd
from qfbench2_track_forecasting.grid import grid_from_card
from scripts.m0_normalization_proxy import distribution, estimate


def fixture_data():
    dates = pd.bdate_range('2020-01-01', periods=70)
    panel = pd.DataFrame({'date': dates, 'asset': 'A',
                          'value': 100 + np.cumsum(np.tile([.1, -.2], 35))})
    card = {'task': {'id': 'synthetic'}, 'targets': {'asset_ids': ['A'], 'horizons': [5, 10],
             'target_type': 'level', 'target_frequency': 'daily'}}
    return {'p': panel}, card, str(dates[60].date())


def test_proxy_ignores_future_values_and_preserves_shared_horizons():
    panels, card, origin = fixture_data()
    grid = grid_from_card(card)
    mean, factor = distribution(panels, grid, origin, 'level')
    changed = {'p': panels['p'].copy()}
    changed['p'].loc[changed['p'].date > pd.Timestamp(origin), 'value'] = 1e9
    other_mean, other_factor = distribution(changed, grid, origin, 'level')
    assert np.array_equal(mean, other_mean)
    assert np.array_equal(factor, other_factor)
    cov = factor @ factor.T
    assert np.isclose(cov[0, 1], cov[0, 0] - 1.1e-9)
    scale, audit = estimate(panels, grid, origin, card, seed=4, outcomes=16)
    again, _ = estimate(changed, grid, origin, card, seed=4, outcomes=16)
    assert scale == again
    assert scale.marginal > 0 and scale.tail > 0 and scale.joint > 0
    assert audit['method'] == 'm0_daily_monte_carlo_proxy'


def test_single_cell_joint_scale_is_absent():
    panels, card, origin = fixture_data()
    card['targets']['horizons'] = [5]
    scale, _ = estimate(panels, grid_from_card(card), origin, card, seed=4, outcomes=16)
    assert scale.joint is None


def test_transfer_gap_cannot_become_one_step():
    _, card, _ = fixture_data()
    card['targets']['horizons'] = [5]
    panel = pd.DataFrame({'date': pd.to_datetime(['2019-01-02', '2019-01-03', '2019-01-04', '2020-01-02']),
                          'asset': 'A', 'value': [100., 101., 103., 1000.]})
    mean, factor = distribution({'p': panel}, grid_from_card(card), '2020-01-02', 'level')
    assert np.isclose(mean[0], 1000. + 5 * 1.5)
    assert np.isclose((factor @ factor.T)[0, 0], 5 * .5 + 1.1e-9)


def test_published_m0_return_steps_are_rows_not_differences_or_log1p():
    panels, card, origin = fixture_data()
    card['targets'].update(target_type='log_return', horizons=[5])
    panels['p']['value'] = np.tile([.01, -.02], 35)
    mean, _ = distribution(panels, grid_from_card(card), origin, 'log_return')
    # Row zero has no preceding interval. The remaining 60 rows average -0.005.
    assert np.isclose(mean[0], -.005 * 5)
