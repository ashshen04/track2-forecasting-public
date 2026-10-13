"""## Executive summary (read this first)

Historical CPI training must use revisions that were public at the forecast origin.
For example, a later revision cannot replace an earlier available index value.
First-release targets and a fixed later snapshot must remain different quantities.
"""
import zipfile
import numpy as np
import pandas as pd
import pytest
from scripts.validate_monthly_cpi import read_vintages, snapshot, target_row


def archive(tmp_path, rows):
    path = tmp_path / 'synthetic.zip'
    header = 'period_start_date,CPIAUCSL,realtime_start_date,realtime_end_date\n'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('synthetic.csv', header + '\n'.join(rows) + '\n')
    return path


def test_revision_and_publication_dates_control_training(tmp_path):
    frame = read_vintages(archive(tmp_path, [
        '2020-01-01,100,2020-02-10,2020-12-31',
        '2020-01-01,105,2021-01-01,',
        '2020-02-01,101,2020-03-10,',
    ]))
    before = snapshot(frame, '2020-03-01')
    assert len(before) == 1 and before.value.iloc[0] == 100
    after = snapshot(frame, '2021-03-01')
    assert after.loc[after.date == pd.Timestamp('2020-01-01'), 'value'].iloc[0] == 105
    # Artificial benchmark lag is applied in addition to genuine publication availability.
    lagged = snapshot(frame, '2020-03-10', lag_days=45)
    assert list(lagged.date) == [pd.Timestamp('2020-01-01')]
    assert target_row(frame, '2020-01', 'first_release') == (100., '2020-02-10')
    assert target_row(frame, '2020-01', 'fixed_snapshot') == (105., '2020-02-10')


def test_missing_initial_release_is_not_replaced_with_a_later_revision(tmp_path):
    frame = read_vintages(archive(tmp_path, ['2020-01-01,,2020-02-10,2020-12-31',
                                            '2020-01-01,105,2021-01-01,']))
    with pytest.raises(ValueError, match='Initial release'):
        target_row(frame, '2020-01', 'first_release')


def test_overlapping_vintage_ranges_are_refused(tmp_path):
    with pytest.raises(ValueError, match='Overlapping'):
        read_vintages(archive(tmp_path, ['2020-01-01,100,2020-02-10,2021-02-01',
                                        '2020-01-01,105,2021-01-01,']))


def test_normalization_uses_explicit_months_not_large_horizon_keys():
    from scripts.m0_normalization_proxy import distribution
    from qfbench2_track_forecasting.grid import grid_from_card
    dates = pd.date_range('2000-01-01', periods=60, freq='MS')
    p = pd.DataFrame({'date': dates, 'asset': 'CPI_ALL',
                     'value': 100 + np.cumsum(np.tile([1.5, .5], 30))})
    card = {'targets': {'asset_ids': ['CPI_ALL'], 'horizons': [140, 160], 'target_type': 'level'}}
    mean, factor = distribution({'p': p}, grid_from_card(card), '2005-01-31', 'level', panel_steps=[[8, 9]])
    # Both forecast variances and baseline expected changes use eight and nine months.
    variance = p.value.diff().var()
    assert np.isclose((factor @ factor.T)[0, 0], 8 * variance + 1.1e-9)
    assert np.isclose((factor @ factor.T)[0, 1], 8 * variance)
    assert np.isclose(mean[1] - mean[0], p.value.diff().mean())


def test_candidate_selection_uses_development_and_prefers_baseline_on_ties():
    from scripts.validate_monthly_cpi import choose_development_policy
    summary = {'comparisons': [{'policy': policy, 'mean_proxy_loss': score}
        for policy, score in [('none', 1.), ('recent', .8), ('half', .7), ('damped', .9)]]}
    assert choose_development_policy(summary)[0] == 'half'
    for r in summary['comparisons']: r['mean_proxy_loss'] = 1.
    assert choose_development_policy(summary)[0] == 'none'
