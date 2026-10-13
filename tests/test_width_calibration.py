"""## Executive summary (read this first)

Synthetic data check that width calibration changes only forecast uncertainty.
For example, width 0.8 scales every innovation by 0.8 while preserving its centre.
Historical exclusions protect evaluation intervals across cards sharing an asset.
"""

import json

import numpy as np
import pandas as pd
import pytest

from qfbench2_track_forecasting.cli import _draw
from qfbench2_track_forecasting.grid import GridSpec
from scripts.backtest import (
    WIDTH_MODELS, forecast_samples, load_excluded_intervals, main, overlaps_excluded,
    width_validation,
)


def history(target_type):
    dates = pd.bdate_range('2020-01-01', periods=161)
    rng = np.random.default_rng(4)
    shocks = rng.normal(0.001, 0.002, (161, 2))
    values = np.expm1(shocks) if target_type == 'log_return' else 100 + shocks.cumsum(axis=0)
    panel = pd.concat([pd.DataFrame({'date': dates, 'asset': asset, 'value': values[:, i]})
                       for i, asset in enumerate(['SYN-A', 'SYN-B'])])
    return {'synthetic': panel}, str(dates[-1].date())


@pytest.mark.parametrize('target_type', ['level', 'log_return'])
@pytest.mark.parametrize('scale', [0.8, 1.0, 1.2])
def test_width_preserves_theoretical_centres_and_joint_innovations(target_type, scale):
    panels, asof = history(target_type)
    args = (panels, ['SYN-A', 'SYN-B'], [21, 5], asof, 1000, 1)
    base, stats = _draw(*args, target_type=target_type)
    changed, adjusted = _draw(*args, target_type=target_type, width_scale=scale)
    centres = np.array([stats['last'][a] for a in args[1]])[:, None] + (
        np.array([stats['daily_drift'][a] for a in args[1]])[:, None] * args[2])
    np.testing.assert_allclose(changed - centres, scale * (base - centres), atol=1e-12)
    assert adjusted['daily_drift'] == stats['daily_drift']
    assert adjusted['last'] == stats['last']
    for asset in args[1]:
        assert adjusted['daily_sd'][asset] == pytest.approx(scale * stats['daily_sd'][asset])
    if scale == 1:
        np.testing.assert_array_equal(changed, base)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_width_is_refused(scale):
    panels, asof = history('level')
    with pytest.raises(ValueError, match='width_scale'):
        _draw(panels, ['SYN-A'], [5], asof, 1000, 1, width_scale=scale)


@pytest.mark.parametrize('name,scale', [('reference-width-08', 0.8), ('reference-width-12', 1.2)])
def test_backtest_routes_width_candidates(name, scale):
    panels, asof = history('level')
    grid = GridSpec(('SYN-A', 'SYN-B'), (21, 5))
    actual = forecast_samples(name, panels, pd.Timestamp(asof), grid, 1000, 'level')
    direct, _ = _draw(panels, list(grid.assets), list(grid.horizons), asof, 1000, 1,
                      width_scale=scale)
    np.testing.assert_array_equal(actual, direct.reshape(1000, -1))


def test_width_selection_is_locked_before_validation():
    folds = []
    for i in range(4):
        scores = {name: {c: value for c in ('marginal', 'joint', 'tail', 'composite')}
                  for name, value in [('reference', 2), ('reference-width-08', 1),
                                      ('reference-width-12', 3)]}
        folds.append({'asof': f'2020-0{i+1}-01', 'model_scores': scores})
    first = width_validation(folds, 2, 4)
    assert first['selected_model'] == 'reference-width-08'
    for fold in folds[2:]:
        fold['model_scores']['reference-width-08']['composite'] = 99
        fold['model_scores']['reference-width-12']['composite'] = 0.001
    second = width_validation(folds, 2, 4)
    assert second['selected_model'] == first['selected_model']
    assert second['validation']['wins'] == 0


def test_exclusion_follows_shared_assets_not_unit_names(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'units': [{'unit': 'different-card', 'assets': ['SYN-A'],
        'folds': [{'asof': '2020-01-01', 'target_dates': ['2020-01-10']},
                  {'asof': '2020-01-08', 'target_dates': ['2020-01-15']}]}]}))
    excluded = load_excluded_intervals([path])
    assert len(excluded['SYN-A']) == 1
    assert overlaps_excluded(pd.Timestamp('2020-01-09'), pd.Timestamp('2020-01-20'),
                             ['SYN-A', 'SYN-B'], excluded)
    assert not overlaps_excluded(pd.Timestamp('2020-01-09'), pd.Timestamp('2020-01-20'),
                                 ['SYN-B'], excluded)
    # (origin, end] intervals touching at the boundary have no shared target date.
    assert not overlaps_excluded(pd.Timestamp('2020-01-15'), pd.Timestamp('2020-01-20'),
                                 ['SYN-A'], excluded)


def test_invalid_exclusion_source_fails_instead_of_ignoring_it(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='historical units'):
        load_excluded_intervals([path])


def test_empty_historical_exclusions_are_not_treated_as_success(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'units': [{'status': 'skipped', 'unit': 'synthetic'}]}))
    with pytest.raises(ValueError, match='no historical holdouts'):
        load_excluded_intervals([path])


def test_width_ignores_future_rows():
    panels, asof = history('level')
    args = (['SYN-A', 'SYN-B'], [21, 5], asof, 1000, 1)
    base, _ = _draw(panels, *args, width_scale=0.8)
    future = pd.DataFrame({'date': [pd.Timestamp('2030-01-01')],
                           'asset': ['SYN-A'], 'value': [99999.]})
    noisy = {'synthetic': pd.concat([panels['synthetic'], future])}
    changed, _ = _draw(noisy, *args, width_scale=0.8)
    np.testing.assert_array_equal(base, changed)


def test_monthly_width_calibration_is_explicitly_unsupported():
    panels, asof = history('level')
    with pytest.raises(ValueError, match='daily-only'):
        _draw(panels, ['SYN-A'], [5], asof, 1000, 1, width_scale=0.8,
              panel_steps=np.array([[1]]))


def test_incomplete_width_run_does_not_claim_validation():
    assert width_validation([], 3, 6)['status'] == 'insufficient_folds'


def test_width_ties_keep_current_model():
    models = ['reference', *WIDTH_MODELS]
    folds = [{'asof': f'2020-0{i+1}-01', 'model_scores': {
        name: {component: 1 for component in ('marginal', 'joint', 'tail', 'composite')}
        for name in models}} for i in range(4)]
    assert width_validation(folds, 2, 4)['selected_model'] == 'reference'


def test_incomplete_width_candidates_fail_before_writing(tmp_path):
    output = tmp_path / 'run'
    with pytest.raises(SystemExit) as error:
        main(['--out', str(output), '--folds', '6', '--selection-folds', '3',
              '--models', 'reference', 'reference-width-08'])
    assert error.value.code == 2
    assert not output.exists()
