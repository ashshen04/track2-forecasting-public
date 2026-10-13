"""## Executive summary (read this first)

Synthetic events verify that analog shifts cannot use unfinished future responses.
An analog is a past event of the same type used to estimate a future change.
"""
import numpy as np
import pandas as pd
import pytest

from baselines.event_analog import analog_adjustment, classify_statement


def fixture():
    dates = pd.bdate_range('2020-01-01', periods=180)
    series = pd.DataFrame({'SYN-A': np.arange(180)*.01}, index=dates)
    events = [{'date':str(dates[i].date()),'type':'fomc_hike'} for i in (10,35,60,85,140)]
    return dates, series, events


def test_unfinished_responses_and_future_events_are_excluded():
    dates, wide, events = fixture()
    shift, audit = analog_adjustment(wide, events, 'fomc_hike', dates[100], ['SYN-A'], [21,5],
                                    np.array([[1.,1.]]), shrink=False)
    assert audit['21']['count']==3
    assert audit['5']['count']==4
    np.testing.assert_allclose(shift, [[.21,.05]])
    altered=wide.copy()
    altered.loc[dates[101]:] = 1e9
    other,_=analog_adjustment(altered,events,'fomc_hike',dates[100],['SYN-A'],[21,5],
                             np.array([[1.,1.]]),shrink=False)
    np.testing.assert_array_equal(shift,other)


def test_shrinkage_preserves_noise_and_reduces_mean_influence():
    dates,wide,events=fixture()
    raw,_=analog_adjustment(wide,events,'fomc_hike',dates[100],['SYN-A'],[5],np.ones((1,1)),shrink=False)
    shrunk,audit=analog_adjustment(wide,events,'fomc_hike',dates[100],['SYN-A'],[5],np.ones((1,1)))
    np.testing.assert_allclose(shrunk,raw*4/14)
    assert audit['5']['weight']==pytest.approx(4/14)


def test_overlapping_peers_do_not_inflate_sample_count():
    dates,wide,_=fixture()
    events=[{'date':str(dates[i].date()),'type':'fomc_hike'} for i in range(10,40)]
    shift,audit=analog_adjustment(wide,events,'fomc_hike',dates[100],['SYN-A'],[21],np.ones((1,1)))
    assert audit['21']['count']==2
    np.testing.assert_array_equal(shift,np.zeros((1,1)))


def test_duplicate_dates_and_wrong_types_are_not_extra_peers():
    dates,wide,events=fixture()
    expanded=events+events+[{'date':str(dates[15].date()),'type':'fomc_cut'}]
    a,aa=analog_adjustment(wide,events,'fomc_hike',dates[100],['SYN-A'],[5],np.ones((1,1)))
    b,bb=analog_adjustment(wide,expanded,'fomc_hike',dates[100],['SYN-A'],[5],np.ones((1,1)))
    np.testing.assert_array_equal(a,b)
    assert aa==bb


def test_missing_endpoint_is_never_filled():
    dates,wide,events=fixture()
    shift,audit=analog_adjustment(wide.drop(dates[16]),events,'fomc_hike',dates[100],
                                ['SYN-A'],[5],np.ones((1,1)))
    assert audit['5']['count']==3
    assert np.isfinite(shift).all()


def test_cap_is_relative_to_forecast_width():
    dates,wide,events=fixture()
    shift,_=analog_adjustment(wide*1000,events,'fomc_hike',dates[100],['SYN-A'],[5],np.array([[.1]]))
    assert shift[0,0]==pytest.approx(.05)


def test_runner_predictions_do_not_change_when_future_targets_change():
    from unittest.mock import patch
    from scripts.validate_event_analog import evaluate
    from qfbench2_track_forecasting.grid import GridSpec
    dates,wide,events=fixture()
    panel=wide.reset_index(names='date').melt(id_vars='date',var_name='asset',value_name='value')
    event={'date':str(dates[99].date()),'type':'fomc_hike'}
    grid=GridSpec(('SYN-A',),(5,21))
    card={'task':{'id':'synthetic-event-test'}}
    captured=[]
    def score(ctx):
        captured.append(ctx['_samples'].copy())
        return {c:1. for c in ('composite','marginal','joint','tail')}
    with patch('scripts.validate_event_analog._score',side_effect=score):
        clean=evaluate(wide,{'p':panel},events,event,card,grid,1,200)
        altered=wide.copy()
        altered.loc[dates[101]:]=1e9
        changed_panel=panel.copy()
        changed_panel.loc[changed_panel.date>dates[100],'value']=1e9
        changed=evaluate(altered,{'p':changed_panel},events,event,card,grid,1,200)
    assert clean['adjustments']==changed['adjustments']
    for before,after in zip(captured[:4],captured[4:]):
        np.testing.assert_array_equal(before,after)


def test_public_output_is_refused_before_event_data_is_read():
    from scripts.validate_event_analog import main, ROOT
    with pytest.raises(ValueError,match='outside'):
        main(['--out',str(ROOT/'out'/'event-test')])


@pytest.mark.parametrize('text,wanted',[
 ('The Committee decided today to raise its target for the federal funds rate by 25 basis points.', 'fomc_hike'),
 ('The Committee decided to raise the target range for the federal funds rate to 1 percent.', 'fomc_hike'),
 ('The Committee decided to maintain the target range for the federal funds rate.', None),
 ('The Committee expects it will soon be appropriate to raise the target range for the federal funds rate.', None),
 ('The Committee decided to raise the discount rate.', None),
])
def test_classification_requires_an_announced_funds_rate_action(text,wanted):
    assert classify_statement(text)==wanted
